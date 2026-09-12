"""Bounded Q2/Q3 candidate scan or streaming CSV export (no Excel)."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
from math import isclose, isfinite
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a_model.fvm import RadialGrid
from a_model.inputs import load_environment
from a_model.metadata import build_run_metadata
from a_model.parameters import get_case_parameters
from a_model.q2q3_phase5 import maximum_location, project, run_streaming
from a_model.solver import CoupledRadialSolver, ModelState
from a_model.time_steps import v4_time_step


CSV_NAMES = ("result2_temperature.csv", "result2_moisture.csv", "result3_moisture.csv")


def encoded_row(values):
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerow(values)
    return buffer.getvalue().encode("utf-8")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore_state(value):
    return ModelState(value["time_s"], tuple(value["temperatures_c"]),
                      tuple(value["moistures"]), value["surface_moisture"])


def serialized_result(result):
    values = {key: asdict(value) if key in ("event", "state") else value
              for key, value in result.items()}
    return json.loads(json.dumps(values, allow_nan=False))


def validate_export_dt(dt_s):
    """Require the PDE step and the formal 1 s output grid to be commensurate."""
    if not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt must be finite and positive.")
    ratio = dt_s if dt_s >= 1 else 1 / dt_s
    if not isclose(ratio, round(ratio), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "Export dt must be an integer multiple or integer divisor of the 1 s sampling interval."
        )


def describe_time_step(dt_s, time_step=None):
    if time_step is None:
        return {"kind": "fixed_backward_euler", "dt_s": dt_s}
    if not callable(time_step) or not hasattr(time_step, "step_sizes_s") or not hasattr(time_step, "as_dict"):
        raise ValueError("Export time_step must provide step_sizes_s and as_dict().")
    return time_step.as_dict()


def time_step_spec(dt_s, time_step=None):
    """Validate and serialize the integration schedule used by an export."""
    spec = describe_time_step(dt_s, time_step)
    step_sizes = (dt_s,) if time_step is None else time_step.step_sizes_s
    for step_size in step_sizes:
        validate_export_dt(step_size)
    return spec


def run_export(solver, output, metadata, *, end_time_s, dt_s, event_tolerance_s,
               checkpoint_interval_s=3600, resume=False, stop_after_segments=None,
               time_step=None):
    """Commit integer-second segments; recover only bytes after a verified checkpoint.

    V4 section 14.1: result2 uses 1 s, result3 uses 60 s, both append the event.
    Segment boundaries lie on the existing 1 s integration/output grid, so a
    checkpoint introduces no new PDE time step or artificial result3 endpoint.
    """
    if not isfinite(end_time_s) or end_time_s <= 0:
        raise ValueError("End time must be finite and positive.")
    schedule_spec = time_step_spec(dt_s, time_step)
    if not isfinite(event_tolerance_s) or event_tolerance_s <= 0:
        raise ValueError("Event tolerance must be finite and positive.")
    if (not isfinite(checkpoint_interval_s) or checkpoint_interval_s < 1
            or checkpoint_interval_s % 1 != 0):
        raise ValueError("Checkpoint interval must be a positive integer number of seconds.")
    if stop_after_segments is not None and (type(stop_after_segments) is not int or stop_after_segments < 1):
        raise ValueError("stop_after_segments must be a positive integer.")
    identity_keys = ("model_version", "case_id", "sources", "initial_conditions", "geometry",
                     "property_model", "boundary_coefficients", "input_policy", "target",
                     "numerical_settings", "source_file_sha256", "source_git_commit")
    identity = {key: metadata[key] for key in identity_keys if key in metadata}
    identity["grid"] = {
        "cells": solver.grid.cell_count,
        "grading_exponent": solver.grid.grading_exponent,
        "faces": list(solver.grid.faces),
    }
    identity["export_controls"] = {"end_time_s": end_time_s, "dt_s": dt_s,
        "time_step_schedule": schedule_spec,
        "event_tolerance_s": event_tolerance_s, "checkpoint_interval_s": checkpoint_interval_s}
    output = Path(output)
    checkpoint_path = output / "checkpoint.json"
    streams, hashes, files = {}, {}, {}
    if resume:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("version") != 1 or checkpoint.get("identity") != identity:
            raise ValueError("Checkpoint settings, input, or source provenance do not match; nothing changed.")
        if set(checkpoint["files"]) != set(CSV_NAMES):
            raise ValueError("Checkpoint CSV inventory is invalid; nothing changed.")
        # Validate every committed prefix before truncating any uncommitted tail.
        for name in CSV_NAMES:
            info = checkpoint["files"][name]
            digest = hashlib.sha256()
            remaining = info["offset_bytes"]
            with (output / name).open("rb") as stream:
                while remaining:
                    block = stream.read(min(1024 * 1024, remaining))
                    if not block:
                        raise ValueError(f"Committed CSV is truncated: {name}; nothing changed.")
                    digest.update(block)
                    remaining -= len(block)
            if digest.hexdigest() != info["sha256"]:
                raise ValueError(f"Committed CSV was modified: {name}; nothing changed.")
            hashes[name] = digest
        if checkpoint["status"] in ("event", "horizon_reached_without_event"):
            if any((output / name).stat().st_size != checkpoint["files"][name]["offset_bytes"] for name in CSV_NAMES):
                raise ValueError("Completed CSV has unexpected trailing bytes; nothing changed.")
            atomic_json(output / "metadata.json", {**checkpoint["metadata"],
                "result": checkpoint["result"],
                "elapsed_wall_s": checkpoint["elapsed_wall_s"],
                "source_files_unchanged": True,
                "checkpoint": {"state_time_s": checkpoint["state"]["time_s"],
                               "files": checkpoint["files"]}})
            return checkpoint["result"]
        files = checkpoint["files"]
        state = restore_state(checkpoint["state"])
        for name in CSV_NAMES:
            streams[name] = (output / name).open("r+b")
            streams[name].truncate(files[name]["offset_bytes"])
            streams[name].seek(files[name]["offset_bytes"])
    else:
        owned = (*CSV_NAMES, "checkpoint.json", "checkpoint.json.tmp", "metadata.json", "metadata.json.tmp")
        if any((output / name).exists() for name in owned):
            raise FileExistsError("Output already contains run files; use --resume or a new directory.")
        output.mkdir(parents=True, exist_ok=True)
        state = solver.initial_state()
        checkpoint = {"version": 1, "identity": identity, "metadata": metadata,
                      "status": "running", "steps": 0, "diagnostics": {}, "elapsed_wall_s": 0.}
        header = encoded_row(("time_s", *(f"r_{i/10:g}_cm" for i in range(21))))
        for name in CSV_NAMES:
            streams[name] = (output / name).open("xb")
            streams[name].write(header)
            hashes[name] = hashlib.sha256(header)
            files[name] = {"rows": 0, "last_time_s": None}

    def save(result):
        for name, stream in streams.items():
            stream.flush()
            os.fsync(stream.fileno())
            files[name].update(offset_bytes=stream.tell(), sha256=hashes[name].hexdigest())
        for relative, expected in identity.get("source_file_sha256", {}).items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Source changed during export; resume from the last unchanged checkpoint.")
        checkpoint.update(state=asdict(state), files=files, result=result, status=result["status"])
        atomic_json(checkpoint_path, checkpoint)
        atomic_json(output / "metadata.json", {**checkpoint["metadata"], "result": result,
            "elapsed_wall_s": checkpoint["elapsed_wall_s"], "source_files_unchanged": True,
            "checkpoint": {"state_time_s": state.time_s, "files": files}})

    def sample(item, terminal):
        if item.time_s == 0 or (not terminal and item.time_s % 1 != 0):
            return
        temperature, moisture = project(solver, item)
        for index, name in enumerate(CSV_NAMES):
            if index == 2 and not terminal and item.time_s % 60 != 0:
                continue
            last_time = files[name]["last_time_s"]
            if last_time is not None and item.time_s <= last_time:
                continue  # run_streaming emits the resumed initial checkpoint too.
            row = encoded_row((item.time_s, *(temperature if index == 0 else moisture)))
            streams[name].write(row)
            streams[name].flush()  # Keep progress visible; fsync only at segment commits.
            hashes[name].update(row)
            files[name]["rows"] += 1
            files[name]["last_time_s"] = item.time_s

    try:
        if not resume:
            save({"status": "running", "state": asdict(state), "steps": 0, "diagnostics": {}})
        segments = 0
        while state.time_s < end_time_s:
            started = perf_counter()
            segment_end = min(end_time_s, (int(state.time_s / checkpoint_interval_s) + 1) * checkpoint_interval_s)
            result = serialized_result(run_streaming(solver, end_time_s=segment_end, dt_s=dt_s,
                initial=state, sample_interval_s=1, on_sample=sample,
                event_tolerance_s=event_tolerance_s, time_step=time_step))
            state = restore_state(result["event"]["state"] if "event" in result else result["state"])
            checkpoint["steps"] += result["steps"]
            for key, value in result.get("diagnostics", {}).items():
                previous = checkpoint["diagnostics"].get(key, 0)
                checkpoint["diagnostics"][key] = previous + value if key == "rejected_attempts" else max(previous, value)
            checkpoint["elapsed_wall_s"] += perf_counter() - started
            segments += 1
            finished = result["status"] == "event" or state.time_s == end_time_s
            paused = stop_after_segments is not None and segments >= stop_after_segments
            if not finished:
                result["status"] = "paused" if paused else "running"
            result.update(steps=checkpoint["steps"], diagnostics=dict(checkpoint["diagnostics"]))
            save(result)
            if finished or paused:
                return result
    finally:
        for stream in streams.values():
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=20)
    parser.add_argument("--grading-exponent", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=60)
    parser.add_argument("--time-level", choices=("P0", "P1", "P2"),
                        help="Use the nested V4 two-stage BE schedule instead of fixed --dt.")
    parser.add_argument("--time-switch", type=float, default=14400,
                        help="Two-stage schedule switch time in seconds (default: 14400).")
    parser.add_argument("--end", type=float, default=3600)
    parser.add_argument("--event-tolerance", type=float, default=.01)
    parser.add_argument("--mode", choices=("scan", "export"), default="scan")
    parser.add_argument("--resume", action="store_true", help="Resume a validated export checkpoint.")
    parser.add_argument("--checkpoint-interval", type=int, default=3600, help="Export checkpoint interval in whole seconds.")
    parser.add_argument("--stop-after-segments", type=int, help="Pause export after this many durable segments.")
    args = parser.parse_args()
    if args.mode != "export" and (args.resume or args.stop_after_segments is not None):
        parser.error("--resume and --stop-after-segments require --mode export")
    environment = load_environment()
    grid = RadialGrid(args.cells, args.grading_exponent)
    solver = CoupledRadialSolver(get_case_parameters("q2q3"), grid, environment, .02)
    time_step = v4_time_step(args.time_level, args.time_switch) if args.time_level else None
    effective_dt_s = time_step.late_dt_s if time_step is not None else args.dt
    sources = sorted((ROOT / "src" / "a_model").glob("*.py")) + [Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    metadata = build_run_metadata("q2q3", [environment.trace], {
        "cells": args.cells, "grading_exponent": grid.grading_exponent,
        "minimum_cell_width_xi": min(right - left for left, right in zip(grid.faces, grid.faces[1:])),
        "surface_half_width_xi": grid.surface_half_width,
        "dt_s": effective_dt_s, "time_step_schedule": describe_time_step(effective_dt_s, time_step),
        "end_time_s": args.end,
        "mode": args.mode, "event_tolerance_s": args.event_tolerance,
        "solver_options": asdict(solver.options), "convergence_verified": False,
        "sampling": "1 s temperature/moisture, 60 s moisture, plus event" if args.mode == "export" else "no periodic samples",
    })
    metadata.update(source_git_commit=commit, source_worktree_status=status, source_file_sha256=hashes,
                    result_classification="candidate_only", command=sys.argv)
    if args.mode == "export":
        result = run_export(solver, args.output, metadata, end_time_s=args.end, dt_s=effective_dt_s,
                            event_tolerance_s=args.event_tolerance, checkpoint_interval_s=args.checkpoint_interval,
                            resume=args.resume, stop_after_segments=args.stop_after_segments,
                            time_step=time_step)
        print(json.dumps({"status": result["status"], "steps": result["steps"]}))
        return
    if (args.output / "metadata.json").exists():
        raise FileExistsError("Scan metadata already exists; choose a new output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    try:
        result = run_streaming(solver, end_time_s=args.end, dt_s=effective_dt_s,
                               event_tolerance_s=args.event_tolerance, time_step=time_step)
        metadata["result"] = serialized_result(result)
        final_state = result["event"].state if "event" in result else result["state"]
        metadata["maximum_location"] = maximum_location(solver, final_state)
    except Exception as exc:
        metadata["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metadata["elapsed_wall_s"] = perf_counter() - started
        metadata["source_files_unchanged"] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))] for p in sources)
        (args.output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "steps": result["steps"], "elapsed_wall_s": metadata["elapsed_wall_s"]}))


if __name__ == "__main__":
    main()
