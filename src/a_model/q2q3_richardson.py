"""Richardson-validated Q2/Q3 export built from completed raw runs.

The module keeps the raw solver runs immutable.  It combines adjacent
space/time levels only after checking their signed summaries and writes a
traceable formal export whose metadata records the extrapolation and the
output interpolation used to place Q2 on its required one-second grid.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil

from .q2q3_convergence import (
    MOISTURE_COLUMNS,
    RADII_CM,
    SAMPLE_COLUMNS,
    TEMPERATURE_COLUMNS,
    TOLERANCES,
    write_json,
)
from .q2q3_xlsx import CSV_HEADER, CSV_NAMES


RICHARDSON_GATE = dict(TOLERANCES)
_FIELD_GROUPS = (
    ("temperature_field_C", TEMPERATURE_COLUMNS),
    ("moisture_field", MOISTURE_COLUMNS),
)
_SCALAR_COLUMNS = tuple(
    column for column in SAMPLE_COLUMNS
    if column not in {"time_s", *TEMPERATURE_COLUMNS, *MOISTURE_COLUMNS}
)


def extrapolate_pair(coarse: float, fine: float, *, order: int) -> float:
    """Return the zero-step Richardson estimate for a refinement ratio of two."""
    if order < 1 or not all(isfinite(value) for value in (coarse, fine)):
        raise ValueError("Richardson inputs must be finite and the order must be positive.")
    factor = 2 ** order
    return (factor * fine - coarse) / (factor - 1)


def gate_result(values: dict[str, float]) -> dict:
    """Return per-metric and aggregate Richardson gate decisions."""
    checks = {
        name: isfinite(values.get(name, float("nan"))) and values[name] <= limit
        for name, limit in RICHARDSON_GATE.items()
    }
    return {"passed": all(checks.values()), "values": dict(values), "checks": checks,
            "tolerances": dict(RICHARDSON_GATE)}


def _load_rows(directory: Path) -> dict[float, dict[str, float]]:
    path = directory / "samples.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = {}
        for raw in csv.DictReader(stream):
            time_s = float(raw["time_s"])
            if time_s not in rows:
                rows[time_s] = {key: float(value) for key, value in raw.items()}
        return rows


def _summary(directory: Path) -> dict:
    """Validate immutable raw evidence, including pre-checkpoint legacy runs."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete") or summary.get("result", {}).get("status") != "event":
        raise ValueError(f"Raw convergence case is incomplete or has no event: {directory}")
    samples = directory / "samples.csv"
    if not samples.is_file() or hashlib.sha256(samples.read_bytes()).hexdigest() != summary.get("samples_sha256"):
        raise ValueError(f"Raw convergence samples changed in {directory}")
    # The first matrix predates the checkpoint integrity field introduced in
    # 5de82c8.  Its recorded source hashes and sample digest remain the
    # provenance; newer cases additionally carry the full summary hash.
    recorded = summary.get("summary_content_sha256")
    if recorded is not None:
        from .q2q3_convergence import _summary_integrity
        if recorded != _summary_integrity(summary):
            raise ValueError(f"Raw convergence summary changed in {directory}")
    return summary


def _event_time(summary: dict) -> float:
    event = summary.get("result", {}).get("event")
    if not event:
        raise ValueError(f"{summary.get('case_id', 'case')} did not reach an event.")
    return float(event["state"]["time_s"])


def _metric_max(left: dict[float, dict[str, float]], right: dict[float, dict[str, float]]) -> dict[str, float]:
    times = sorted(set(left) & set(right))
    if not times:
        raise ValueError("Richardson cases have no common samples.")
    values = {name: 0.0 for name in RICHARDSON_GATE if name != "event_time_s"}
    for time_s in times:
        for name, columns in _FIELD_GROUPS:
            values[name] = max(values[name], *(abs(left[time_s][column] - right[time_s][column])
                                               for column in columns))
        for column in _SCALAR_COLUMNS:
            values[column] = max(values[column], abs(left[time_s][column] - right[time_s][column]))
    return values


def _extrapolated_rows(coarse: dict[float, dict[str, float]], fine: dict[float, dict[str, float]],
                       *, order: int) -> dict[float, dict[str, float]]:
    times = sorted(set(coarse) & set(fine))
    if not times:
        raise ValueError("Richardson cases have no common samples.")
    columns = tuple(column for column in SAMPLE_COLUMNS if column != "time_s")
    return {time_s: {column: extrapolate_pair(coarse[time_s][column], fine[time_s][column], order=order)
                     for column in columns} for time_s in times}


def _gate_from_triplet(coarse: Path, middle: Path, fine: Path, *, order: int) -> dict:
    coarse_rows, middle_rows, fine_rows = (_load_rows(path) for path in (coarse, middle, fine))
    first = _extrapolated_rows(coarse_rows, middle_rows, order=order)
    second = _extrapolated_rows(middle_rows, fine_rows, order=order)
    values = _metric_max(first, second)
    summaries = [_summary(path) for path in (coarse, middle, fine)]
    first_event = extrapolate_pair(_event_time(summaries[0]), _event_time(summaries[1]), order=order)
    second_event = extrapolate_pair(_event_time(summaries[1]), _event_time(summaries[2]), order=order)
    values["event_time_s"] = abs(first_event - second_event)
    return {"levels": [path.name for path in (coarse, middle, fine)],
            "order": order, "event_estimates_s": [first_event, second_event],
            "common_samples": len(set(first) & set(second)),
            "gate": gate_result(values)}


def _interpolate(rows: list[tuple[float, dict[str, float]]], time_s: float) -> dict[str, float]:
    if time_s < rows[0][0] or time_s > rows[-1][0]:
        raise ValueError("Interpolation time is outside the validated source range.")
    for index in range(1, len(rows)):
        right_time, right = rows[index]
        if time_s <= right_time:
            left_time, left = rows[index - 1]
            if time_s == right_time:
                return dict(right)
            fraction = (time_s - left_time) / (right_time - left_time)
            return {column: left[column] + fraction * (right[column] - left[column])
                    for column in left}
    return dict(rows[-1][1])


def _linear_from_pair(left, right, time_s):
    left_time, left_values = left
    right_time, right_values = right
    fraction = (time_s - left_time) / (right_time - left_time)
    return {column: left_values[column] + fraction * (right_values[column] - left_values[column])
            for column in left_values}


def _write_result_csv(path: Path, rows: list[tuple[float, dict[str, float]]], columns: tuple[str, ...]) -> dict:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_HEADER)
        for time_s, values in rows:
            writer.writerow((time_s, *(values[column] for column in columns)))
    return {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_formal_export(
    *,
    space_triplet: tuple[Path, Path, Path],
    time_triplet: tuple[Path, Path, Path],
    output: Path,
) -> dict:
    """Build formal CSVs only when both Richardson axes pass every V4 gate."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Formal output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    space_gate = _gate_from_triplet(*space_triplet, order=2)
    time_gate = _gate_from_triplet(*time_triplet, order=1)
    if not space_gate["gate"]["passed"] or not time_gate["gate"]["passed"]:
        raise ValueError("Richardson space/time gate failed; formal files were not created.")

    time_coarse, time_fine = (_load_rows(time_triplet[1]), _load_rows(time_triplet[2]))
    spatial_coarse, spatial_fine = (_load_rows(space_triplet[1]), _load_rows(space_triplet[2]))
    common = sorted(set(time_coarse) & set(time_fine) & set(spatial_coarse) & set(spatial_fine))
    if not common or common[0] != 0.0:
        raise ValueError("Formal source cases must share a t=0 sample.")
    columns = tuple(column for column in SAMPLE_COLUMNS if column != "time_s")
    regular = []
    for time_s in common:
        values = {}
        for column in columns:
            temporal = extrapolate_pair(time_coarse[time_s][column], time_fine[time_s][column], order=1)
            spatial = (spatial_fine[time_s][column] - spatial_coarse[time_s][column]) / 3.0
            values[column] = temporal + spatial
        regular.append((time_s, values))

    last_left, last_right = regular[-2:]
    left_time, left_values = last_left
    right_time, right_values = last_right
    left_max, right_max = left_values["maximum_moisture"], right_values["maximum_moisture"]
    if not left_max > right_max > .15:
        raise ValueError("The final common Richardson samples do not bracket a monotone event extrapolation.")
    crossing_time = right_time + (.15 - right_max) * (right_time - left_time) / (right_max - left_max)
    # V4 distinguishes the crossing limit from the first verified strict state.
    # Add both measured event-estimate differences and the 0.01 s locator width.
    safety_s = (space_gate["gate"]["values"]["event_time_s"]
                + time_gate["gate"]["values"]["event_time_s"] + .01)
    event_time = crossing_time + safety_s
    terminal = _linear_from_pair(last_left, last_right, event_time)
    if max(terminal[column] for column in MOISTURE_COLUMNS) >= .15:
        raise ValueError("Richardson strict event endpoint is not below the moisture threshold.")
    regular.append((event_time, terminal))

    integer_rows = [(float(time_s), _interpolate(regular, float(time_s)))
                    for time_s in range(1, int(event_time) + 1)]
    q2_rows = integer_rows + [(event_time, terminal)]
    q3_rows = [(float(time_s), _interpolate(regular, float(time_s)))
               for time_s in range(60, int(event_time) + 1, 60)]
    q3_rows.append((event_time, terminal))
    files = {}
    files[CSV_NAMES[0]] = _write_result_csv(output / CSV_NAMES[0], q2_rows, TEMPERATURE_COLUMNS)
    files[CSV_NAMES[1]] = _write_result_csv(output / CSV_NAMES[1], q2_rows, MOISTURE_COLUMNS)
    files[CSV_NAMES[2]] = _write_result_csv(output / CSV_NAMES[2], q3_rows, MOISTURE_COLUMNS)

    metadata = deepcopy(summaries[1])
    metadata.update(
        result_classification="formal",
        source_files_unchanged=True,
        complete=True,
        numerical_settings={**metadata["numerical_settings"], "convergence_verified": True,
                             "formal_method": "Richardson space/order2 + time/order1; 60 s source interpolation to Q2 1 s grid"},
        result={"status": "event", "event": {"state": {"time_s": event_time},
                                                  "left_time_s": crossing_time,
                                                  "right_time_s": event_time}},
        final_snapshot={**terminal, "time_s": event_time},
        checkpoint={"state_time_s": event_time, "files": files},
        richardson={"space": space_gate, "time": time_gate,
                    "source_cases": {"space": [str(path) for path in space_triplet],
                                      "time": [str(path) for path in time_triplet]},
                    "formal_event_time_s": event_time,
                    "crossing_time_s": crossing_time,
                    "strict_endpoint_safety_margin_s": safety_s,
                    "q2_output_interpolation": "piecewise linear between validated 60 s Richardson samples"},
    )
    write_json(output / "metadata.json", metadata)
    report = {"result_classification": "formal", "convergence_verified": True,
              "space": space_gate, "time": time_gate, "crossing_time_s": crossing_time,
              "formal_event_time_s": event_time,
              "files": files}
    write_json(output / "richardson_report.json", report)
    return report


def assemble_formal_package(*, q2_direct: Path, richardson: Path, output: Path) -> dict:
    """Combine a direct 0--3 h Q2 export with the Richardson-verified Q3 export."""
    q2_direct, richardson, output = map(Path, (q2_direct, richardson, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Formal package directory is not empty: {output}")
    direct = json.loads((q2_direct / "metadata.json").read_text(encoding="utf-8"))
    numerical = direct.get("numerical_settings", {})
    result = direct.get("result", {})
    if (not direct.get("source_files_unchanged")
            or numerical.get("cells") != 320
            or numerical.get("grading_exponent") != 2.0
            or numerical.get("end_time_s") != 10800.0
            or numerical.get("time_step_schedule", {}).get("level") != "P2"
            or result.get("status") != "horizon_reached_without_event"):
        raise ValueError("Q2 direct export is not the required N320/grade2/P2/3 h run.")
    base = json.loads((richardson / "metadata.json").read_text(encoding="utf-8"))
    if (base.get("result_classification") != "formal"
            or not base.get("richardson", {}).get("space", {}).get("gate", {}).get("passed")
            or not base.get("richardson", {}).get("time", {}).get("gate", {}).get("passed")):
        raise ValueError("Richardson source is not a passed formal export.")
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        CSV_NAMES[0]: q2_direct / CSV_NAMES[0],
        CSV_NAMES[1]: q2_direct / CSV_NAMES[1],
        CSV_NAMES[2]: richardson / CSV_NAMES[2],
    }
    files = {}
    for name, source in sources.items():
        destination = output / name
        shutil.copyfile(source, destination)
        with destination.open(encoding="utf-8", newline="") as stream:
            rows = sum(1 for _ in stream) - 1
        files[name] = {"rows": rows, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    q2_end_s = float(result["state"]["time_s"])
    base["checkpoint"]["files"] = files
    base["q2_scope_end_s"] = q2_end_s
    base["q2_formal_source"] = {
        "method": "direct N320 grade2 P2 backward-Euler schedule",
        "directory": str(q2_direct),
        "metadata_sha256": hashlib.sha256((q2_direct / "metadata.json").read_bytes()).hexdigest(),
    }
    base["numerical_settings"]["q2_sampling"] = "direct 1 s accepted integration outputs through 3 h"
    base["richardson"]["q2_output_interpolation"] = None
    write_json(output / "metadata.json", base)
    report = json.loads((richardson / "richardson_report.json").read_text(encoding="utf-8"))
    report.update(files=files, q2_scope_end_s=q2_end_s,
                  q2_formal_source=base["q2_formal_source"])
    write_json(output / "richardson_report.json", report)
    return report
