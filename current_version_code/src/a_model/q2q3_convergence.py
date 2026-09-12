"""Measured Q2/Q3 grid/time comparisons; no automatic formal-result approval."""

from dataclasses import asdict
from contextlib import contextmanager
import csv
import hashlib
import json
from math import isfinite, log
import os
from pathlib import Path
from time import perf_counter, time_ns

from .fvm import RadialGrid, reconstruct_center
from .inputs import load_environment
from .metadata import build_run_metadata
from .q2q3_phase5 import maximum_location, project, run_streaming, surface
from .solver import CoupledRadialSolver, ModelState


RADII_CM = tuple(i / 10 for i in range(21))
TEMPERATURE_COLUMNS = tuple(f"temperature_r_{r:g}_cm_C" for r in RADII_CM)
MOISTURE_COLUMNS = tuple(f"moisture_r_{r:g}_cm" for r in RADII_CM)
SCALAR_COLUMNS = ("temperature_center_C", "temperature_surface_C", "temperature_mean_C",
                  "moisture_center", "moisture_surface", "moisture_mean",
                  "heat_flux_W_m2", "moisture_flux_m_s", "maximum_moisture")
SAMPLE_COLUMNS = ("time_s", *TEMPERATURE_COLUMNS, *MOISTURE_COLUMNS, *SCALAR_COLUMNS)
# V4 section 12.2 starter thresholds; these do not certify four-decimal accuracy.
TOLERANCES = {"temperature_field_C": .01, "moisture_field": 1e-4,
              **{name: .01 for name in SCALAR_COLUMNS[:3]},
              **{name: 1e-4 for name in SCALAR_COLUMNS[3:6]},
              "heat_flux_W_m2": .25, "moisture_flux_m_s": 8e-11,
              "maximum_moisture": 1e-4, "event_time_s": 1.}
CHECKPOINT_VERSION = 1
CHECKPOINT_INTERVAL_S = 3600.0


def snapshot(solver, state):
    temperatures, moistures = project(solver, state)
    ts, cs = surface(solver, state)
    te, ce = solver.environment.at(state.time_s)
    weights = solver.grid.weights
    scalars = (reconstruct_center(state.temperatures_c, solver.grid), ts,
               sum(w * t for w, t in zip(weights, state.temperatures_c)),
               reconstruct_center(state.moistures, solver.grid), cs,
               sum(w * c for w, c in zip(weights, state.moistures)),
               solver.case.heat_transfer_coefficient * (ts - te),
               solver.case.mass_transfer_coefficient * (cs - ce),
               maximum_location(solver, state)["maximum_moisture"])
    values = (state.time_s, *temperatures, *moistures, *scalars)
    if not all(isfinite(x) for x in values):
        raise ValueError("Non-finite convergence sample.")
    return dict(zip(SAMPLE_COLUMNS, values))


class AuditedSolver(CoupledRadialSolver):
    """Check each actual advance, including discarded event-refinement trials.

    Sums of absolute residuals bound any subset; they are not a cumulative
    physical trajectory when event bisection replays an interval.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = {"advance_calls_including_event_trials": 0, "max_heat_scaled_residual": 0.,
                      "max_moisture_scaled_residual": 0., "minimum_moisture": 2.55,
                      "sum_absolute_mass_residual_over_C0": 0.,
                      "max_absolute_heat_balance_residual_J_m3": 0.,
                      "max_picard_iterations": 0, "minimum_accepted_dt_s": None,
                      "rejected_attempts": 0}

    def advance(self, previous, dt_s):
        step = super().advance(previous, dt_s)
        state, d = step.state, step.diagnostics
        self.audit["advance_calls_including_event_trials"] += 1
        self.audit["max_heat_scaled_residual"] = max(self.audit["max_heat_scaled_residual"], d.maximum_temperature_scaled_residual)
        self.audit["max_moisture_scaled_residual"] = max(self.audit["max_moisture_scaled_residual"], d.maximum_moisture_scaled_residual)
        self.audit["minimum_moisture"] = min(self.audit["minimum_moisture"], *state.moistures,
                                              d.boundary.moisture,
                                              reconstruct_center(state.moistures, self.grid))
        mass_storage = sum(w * (new - old) for w, new, old in zip(self.grid.weights, state.moistures, previous.moistures))
        mass_out = d.accepted_dt_s * 2 / d.radius_m * self.case.mass_transfer_coefficient * (d.boundary.moisture - d.environment_moisture)
        heat_storage = sum(w * self.case.volumetric_heat_storage(c) * (new - old)
                           for w, c, new, old in zip(self.grid.weights, state.moistures, state.temperatures_c, previous.temperatures_c))
        heat_out = d.accepted_dt_s * 2 / d.radius_m * self.case.heat_transfer_coefficient * (d.boundary.temperature_c - d.environment_temperature_c)
        self.audit["sum_absolute_mass_residual_over_C0"] += abs(mass_storage + mass_out) / 2.55
        self.audit["max_absolute_heat_balance_residual_J_m3"] = max(self.audit["max_absolute_heat_balance_residual_J_m3"], abs(heat_storage + heat_out))
        self.audit["max_picard_iterations"] = max(self.audit["max_picard_iterations"], d.picard_iterations)
        self.audit["rejected_attempts"] += d.rejected_attempts
        old_dt = self.audit["minimum_accepted_dt_s"]
        self.audit["minimum_accepted_dt_s"] = min(old_dt, d.accepted_dt_s) if old_dt is not None else d.accepted_dt_s
        return step


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _state_dict(state):
    return asdict(state)


def _load_state(value, cells):
    if not isinstance(value, dict):
        raise ValueError("Checkpoint has no complete model state.")
    try:
        state = ModelState(
            float(value["time_s"]),
            tuple(float(item) for item in value["temperatures_c"]),
            tuple(float(item) for item in value["moistures"]),
            None if value.get("surface_moisture") is None else float(value["surface_moisture"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint model state is malformed.") from exc
    values = (state.time_s, *state.temperatures_c, *state.moistures)
    if state.surface_moisture is not None:
        values += (state.surface_moisture,)
    if (len(state.temperatures_c) != cells or len(state.moistures) != cells
            or not all(isfinite(item) for item in values)):
        raise ValueError("Checkpoint model state has the wrong size or non-finite values.")
    return state


def _prefix_sha256(path, length):
    digest = hashlib.sha256()
    remaining = length
    with Path(path).open("rb") as stream:
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("Checkpoint CSV is shorter than its committed byte offset.")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _json_sha256(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SUMMARY_INTEGRITY_KEYS = (
    "signature", "result", "final_snapshot", "maximum_location", "step_audit",
    "samples_sha256",
)


def _summary_integrity(metadata):
    return _json_sha256({key: metadata.get(key) for key in SUMMARY_INTEGRITY_KEYS})


def _process_is_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _case_lock(output):
    """Give one process exclusive ownership of a convergence case directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".run.lock"
    payload = json.dumps({"pid": os.getpid(), "created_time_ns": time_ns()}).encode("utf-8")
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
                pid = owner.get("pid")
            except (OSError, json.JSONDecodeError, AttributeError):
                raise RuntimeError(f"Unreadable convergence case lock: {path}") from None
            if not isinstance(pid, int) or pid <= 0:
                raise RuntimeError(f"Malformed convergence case lock: {path}")
            if _process_is_alive(pid):
                raise RuntimeError(f"Convergence case is already running under PID {pid}: {output}")
            stale = output / f".run.lock.stale.{pid}.{time_ns()}"
            try:
                path.replace(stale)
            except FileNotFoundError:
                continue
            continue
        else:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            break
    try:
        yield
    finally:
        try:
            if path.read_bytes() == payload:
                path.unlink()
        except FileNotFoundError:
            pass


def _validate_checkpoint_csv(path, offset, expected_rows, solver, state):
    """Bind the committed CSV prefix to its row count and complete model state."""
    with Path(path).open("rb") as stream:
        header = stream.readline()
        if not header or stream.tell() > offset:
            raise ValueError("Checkpoint CSV header is missing or outside the committed prefix.")
        rows = 0
        last = None
        while stream.tell() < offset:
            line = stream.readline()
            if not line or stream.tell() > offset:
                raise ValueError("Checkpoint CSV offset does not end at a complete row.")
            rows += 1
            last = line
        if stream.tell() != offset or rows != expected_rows or last is None:
            raise ValueError("Checkpoint CSV row count does not match its committed prefix.")
    try:
        fieldnames = next(csv.reader([header.decode("utf-8")]))
        values = next(csv.reader([last.decode("utf-8")]))
    except (UnicodeDecodeError, csv.Error, StopIteration) as exc:
        raise ValueError("Checkpoint CSV committed row is malformed.") from exc
    if tuple(fieldnames) != SAMPLE_COLUMNS or len(values) != len(SAMPLE_COLUMNS):
        raise ValueError("Checkpoint CSV columns are incompatible.")
    expected = snapshot(solver, state)
    try:
        actual = {key: float(value) for key, value in zip(fieldnames, values)}
    except ValueError as exc:
        raise ValueError("Checkpoint CSV committed row is non-numeric.") from exc
    if any(actual[key] != expected[key] for key in SAMPLE_COLUMNS):
        raise ValueError("Checkpoint state does not match the last committed CSV row.")


def _checkpoint_progress(audit):
    return {
        "steps": audit["advance_calls_including_event_trials"],
        "diagnostics": {
            "rejected_attempts": audit["rejected_attempts"],
            "max_heat_residual": audit["max_heat_scaled_residual"],
            "max_moisture_residual": audit["max_moisture_scaled_residual"],
        },
    }


def _validate_audit(audit, expected_keys):
    if not isinstance(audit, dict) or set(audit) != expected_keys:
        raise ValueError("Checkpoint audit state is missing or incompatible.")
    integer_keys = ("advance_calls_including_event_trials", "max_picard_iterations",
                    "rejected_attempts")
    if any(isinstance(audit[key], bool) or not isinstance(audit[key], int)
           or audit[key] < 0 for key in integer_keys):
        raise ValueError("Checkpoint audit counters are malformed.")
    minimum_dt = audit["minimum_accepted_dt_s"]
    if minimum_dt is not None and (not isinstance(minimum_dt, (int, float))
                                   or not isfinite(minimum_dt) or minimum_dt <= 0):
        raise ValueError("Checkpoint minimum accepted dt is malformed.")
    for key in expected_keys - {*integer_keys, "minimum_accepted_dt_s"}:
        value = audit[key]
        if not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError("Checkpoint audit values are malformed.")


def _merge_progress(result, progress):
    if not progress:
        return result
    result["steps"] = progress["steps"] + result.get("steps", 0)
    current = result.setdefault("diagnostics", {})
    saved = progress["diagnostics"]
    current["rejected_attempts"] = saved["rejected_attempts"] + current.get("rejected_attempts", 0)
    for key in ("max_heat_residual", "max_moisture_residual"):
        current[key] = max(saved[key], current.get(key, 0.))
    return result


def case_label(cells, dt_s, grading_exponent=1.0):
    base = f"N{cells}_dt{dt_s:g}".replace(".", "p")
    if grading_exponent == 1.0:
        return base
    return f"{base}_grade{grading_exponent:g}".replace(".", "p")


def schedule_case_label(cells, time_step, grading_exponent=1.0):
    level = getattr(time_step, "level", None)
    if not isinstance(level, str) or not level:
        raise ValueError("Time-step schedule must have a non-empty level label.")
    base = f"N{cells}_{level}"
    if grading_exponent == 1.0:
        return base
    return f"{base}_grade{grading_exponent:g}".replace(".", "p")


def _run_case_locked(
    output,
    cells,
    dt_s,
    end_time_s,
    sample_interval_s,
    provenance,
    grading_exponent=1.0,
    time_step=None,
):
    """Resume immutable complete cases or an exactly committed incomplete state."""
    step_sizes = (dt_s,) if time_step is None else tuple(time_step.step_sizes_s)
    if (not step_sizes or any(step <= 0 or sample_interval_s < step
            or abs(sample_interval_s / step - round(sample_interval_s / step)) > 1e-10
            for step in step_sizes)):
        raise ValueError("Sampling interval must be an integer multiple of every scheduled dt; do not relabel clipped scans.")
    schedule = ({"kind": "fixed_backward_euler", "dt_s": dt_s}
                if time_step is None else time_step.as_dict())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    environment = load_environment()
    grid = RadialGrid(cells, grading_exponent)
    settings = {"cells": cells, "grading_exponent": grading_exponent,
                "minimum_cell_width_xi": min(r - l for l, r in zip(grid.faces, grid.faces[1:])),
                "surface_half_width_xi": grid.surface_half_width,
                "dt_s": dt_s, "end_time_s": end_time_s,
                "time_step_schedule": schedule,
                "sample_interval_s": sample_interval_s, "event_tolerance_s": .01,
                "convergence_verified": False}
    signature = {"settings": settings, "source_file_sha256": provenance["source_file_sha256"],
                 "input_sha256": environment.trace.sha256}
    summary_path = output / "summary.json"
    if summary_path.exists():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        if saved.get("signature") != signature:
            raise ValueError(f"Existing case {output} has different inputs/settings/source; choose a new output directory.")
        if saved.get("complete"):
            if sha256(output / "samples.csv") != saved["samples_sha256"]:
                raise ValueError(f"Completed samples changed in {output}.")
            if saved.get("summary_content_sha256") != _summary_integrity(saved):
                raise ValueError(f"Completed summary changed in {output}.")
            return saved
    solver = AuditedSolver("q2q3", grid, environment, .02)
    checkpoint_path = output / "checkpoint.json"
    partial_path = output / "samples.partial.csv"
    completed_path = output / "samples.csv"
    initial = None
    committed_rows = 0
    committed_progress = None
    elapsed_before = 0.
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported convergence checkpoint in {output}.")
        if checkpoint.get("signature") != signature:
            raise ValueError(f"Checkpoint {output} has different inputs/settings/source; choose a new output directory.")
        file_record = checkpoint.get("file")
        if not isinstance(file_record, dict):
            raise ValueError("Checkpoint has no committed CSV record.")
        if file_record.get("name") != partial_path.name:
            raise ValueError("Checkpoint CSV name is incompatible.")
        offset = file_record.get("offset_bytes")
        committed_rows = file_record.get("rows")
        if (not isinstance(offset, int) or offset < 0 or not isinstance(committed_rows, int)
                or committed_rows < 1):
            raise ValueError("Checkpoint CSV offset or row count is malformed.")
        recovery_path = partial_path if partial_path.exists() else completed_path
        if not recovery_path.is_file():
            raise ValueError("Checkpoint CSV is missing.")
        if recovery_path.stat().st_size < offset:
            raise ValueError("Checkpoint CSV is shorter than its committed byte offset.")
        if _prefix_sha256(recovery_path, offset) != file_record.get("prefix_sha256"):
            raise ValueError("Checkpoint CSV committed prefix was modified.")
        initial = _load_state(checkpoint.get("state"), cells)
        if not 0 <= initial.time_s <= end_time_s:
            raise ValueError("Checkpoint state time is outside the requested horizon.")
        _validate_checkpoint_csv(recovery_path, offset, committed_rows, solver, initial)
        saved_audit = checkpoint.get("step_audit")
        _validate_audit(saved_audit, set(solver.audit))
        committed_progress = checkpoint.get("progress")
        if committed_progress != _checkpoint_progress(saved_audit):
            raise ValueError("Checkpoint progress counters are missing.")
        metadata = checkpoint.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("signature") != signature:
            raise ValueError("Checkpoint metadata is missing or incompatible.")
        elapsed_before = float(checkpoint.get("elapsed_wall_s", 0.))
        if not isfinite(elapsed_before) or elapsed_before < 0:
            raise ValueError("Checkpoint elapsed time is malformed.")
        # Validate every committed byte before truncating an uncommitted tail.
        solver.audit.update(saved_audit)
        with recovery_path.open("r+b") as stream:
            stream.truncate(offset)
        if recovery_path == completed_path:
            completed_path.replace(partial_path)
    else:
        metadata = build_run_metadata("q2q3", [environment.trace], {**settings, "solver_options": asdict(solver.options)})
        metadata.update(provenance)
        metadata.update(signature=signature, result_classification="candidate_only", complete=False)
    started = perf_counter()
    succeeded = False
    try:
        mode = "a" if checkpoint is not None else "w"
        with partial_path.open(mode, newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS)
            if checkpoint is None:
                writer.writeheader()
            last_checkpoint_time = initial.time_s if initial is not None else None
            skip_time = initial.time_s if initial is not None else None

            def save_sample(state, terminal):
                nonlocal committed_rows, last_checkpoint_time, skip_time
                if skip_time is not None and state.time_s == skip_time:
                    skip_time = None
                    return
                writer.writerow(snapshot(solver, state))
                committed_rows += 1
                due = (last_checkpoint_time is None or
                       state.time_s - last_checkpoint_time >= max(CHECKPOINT_INTERVAL_S, sample_interval_s)
                       - 1e-10)
                # Leave an event row beyond the commit so interrupted event
                # finalization is replayed from the last accepted state.
                if terminal or not due:
                    return
                stream.flush()
                os.fsync(stream.fileno())
                offset = partial_path.stat().st_size
                payload = {
                    "version": CHECKPOINT_VERSION,
                    "signature": signature,
                    "state": _state_dict(state),
                    "file": {"name": partial_path.name, "offset_bytes": offset,
                             "rows": committed_rows, "prefix_sha256": _prefix_sha256(partial_path, offset)},
                    "step_audit": solver.audit,
                    "progress": _checkpoint_progress(solver.audit),
                    "metadata": metadata,
                    "elapsed_wall_s": elapsed_before + perf_counter() - started,
                }
                write_json(checkpoint_path, payload)
                last_checkpoint_time = state.time_s

            result = run_streaming(solver, end_time_s=end_time_s, dt_s=dt_s,
                                   sample_interval_s=sample_interval_s,
                                   on_sample=save_sample, initial=initial,
                                   time_step=time_step)
            stream.flush()
            os.fsync(stream.fileno())
        result = _merge_progress(result, committed_progress)
        metadata["result"] = {k: asdict(v) if k in ("state", "event") else v for k, v in result.items()}
        state = result["event"].state if "event" in result else result["state"]
        metadata["final_snapshot"] = snapshot(solver, state)
        metadata["maximum_location"] = maximum_location(solver, state)
        partial_path.replace(completed_path)
        metadata["samples_sha256"] = sha256(completed_path)
        metadata["complete"] = True
        succeeded = True
    except Exception as exc:
        metadata["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metadata["elapsed_wall_s"] = elapsed_before + perf_counter() - started
        metadata["step_audit"] = solver.audit
        if metadata.get("complete"):
            metadata["summary_content_sha256"] = _summary_integrity(metadata)
        write_json(summary_path, metadata)
        if succeeded:
            checkpoint_path.unlink(missing_ok=True)
    return metadata


def run_case(
    output,
    cells,
    dt_s,
    end_time_s,
    sample_interval_s,
    provenance,
    grading_exponent=1.0,
    time_step=None,
):
    """Run one case with exclusive ownership and durable exact-state checkpoints."""
    with _case_lock(output):
        return _run_case_locked(
            output, cells, dt_s, end_time_s, sample_interval_s, provenance,
            grading_exponent, time_step,
        )


def periodic_samples(path, interval_s, last_time_s):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            time = float(row["time_s"])
            if time > last_time_s:
                return
            if time > 0 and time % interval_s == 0:
                yield {key: float(value) for key, value in row.items()}


def validated_summary(
    directory,
    *,
    expected_cells=None,
    expected_dt_s=None,
    expected_grading_exponent=None,
    expected_source_hashes=None,
):
    """Load one completed case only after checking its immutable raw evidence."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"Incomplete convergence case in {directory}.")
    if summary.get("summary_content_sha256") != _summary_integrity(summary):
        raise ValueError(f"Completed summary changed in {directory}.")
    samples_path = directory / "samples.csv"
    if not samples_path.is_file() or sha256(samples_path) != summary.get("samples_sha256"):
        raise ValueError(f"Completed samples changed in {directory}.")
    signature = summary.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"Missing case signature in {directory}.")
    settings = signature.get("settings")
    numerical = summary.get("numerical_settings")
    if not isinstance(settings, dict) or not isinstance(numerical, dict):
        raise ValueError(f"Missing numerical settings in {directory}.")
    for key, value in settings.items():
        if numerical.get(key) != value:
            raise ValueError(f"Signed and recorded settings differ in {directory}.")
    if expected_cells is not None and settings.get("cells") != expected_cells:
        raise ValueError(f"Unexpected cell count in {directory}.")
    if expected_dt_s is not None and settings.get("dt_s") != expected_dt_s:
        raise ValueError(f"Unexpected time step in {directory}.")
    if (
        expected_grading_exponent is not None
        and settings.get("grading_exponent") != expected_grading_exponent
    ):
        raise ValueError(f"Unexpected grading exponent in {directory}.")
    source_signature = signature.get("source_file_sha256")
    if not isinstance(source_signature, dict) or not source_signature:
        raise ValueError(f"Missing source signature in {directory}.")
    if source_signature != summary.get("source_file_sha256"):
        raise ValueError(f"Signed and recorded source hashes differ in {directory}.")
    if expected_source_hashes is not None and source_signature != expected_source_hashes:
        raise ValueError(f"Case source signature is not the current runner provenance in {directory}.")
    input_hash = signature.get("input_sha256")
    source_inputs = {item.get("sha256") for item in summary.get("sources", [])
                     if item.get("filename") == "附件1.xlsx"}
    if not input_hash or source_inputs != {input_hash}:
        raise ValueError(f"Signed and recorded input hashes differ in {directory}.")
    return summary


def compare_cases(left_directory, right_directory, *, last_time_s=None, first_time_s=0.):
    """Compare aligned physical samples, never interpolate different event rows."""
    left_directory, right_directory = Path(left_directory), Path(right_directory)
    left = validated_summary(left_directory)
    right = validated_summary(right_directory)
    if left["signature"]["source_file_sha256"] != right["signature"]["source_file_sha256"]:
        raise ValueError("Compared cases must have the same source signature.")
    if left["signature"]["input_sha256"] != right["signature"]["input_sha256"]:
        raise ValueError("Compared cases must have the same input signature.")
    li, ri = (v["numerical_settings"]["sample_interval_s"] for v in (left, right))
    if li != ri:
        raise ValueError("Compared cases must have the same sampling interval.")
    horizon = min(left["final_snapshot"]["time_s"], right["final_snapshot"]["time_s"])
    if last_time_s is not None:
        horizon = min(horizon, last_time_s)
    errors = {key: 0. for key in TOLERANCES if key != "event_time_s"}
    times = dict.fromkeys(errors, None)
    radii = dict.fromkeys(errors, None)
    count = 0
    first = last = None
    left_rows = periodic_samples(left_directory / "samples.csv", li, horizon)
    right_rows = periodic_samples(right_directory / "samples.csv", ri, horizon)
    from itertools import zip_longest
    for lrow, rrow in zip_longest(left_rows, right_rows):
        if lrow is None or rrow is None or lrow["time_s"] != rrow["time_s"]:
            raise ValueError("Compared cases have missing or misaligned common samples.")
        time = lrow["time_s"]
        if time < first_time_s:
            continue
        count += 1
        first = time if first is None else first
        last = time
        for key, columns in (("temperature_field_C", TEMPERATURE_COLUMNS), ("moisture_field", MOISTURE_COLUMNS)):
            for radius, column in zip(RADII_CM, columns):
                error = abs(lrow[column] - rrow[column])
                if times[key] is None or error > errors[key]:
                    errors[key], times[key], radii[key] = error, time, radius
        for key in SCALAR_COLUMNS:
            error = abs(lrow[key] - rrow[key])
            if times[key] is None or error > errors[key]:
                errors[key], times[key] = error, time
    if not count:
        raise ValueError("No common positive-time samples to compare.")
    event_delta = event_relative = None
    if left["result"]["status"] == right["result"]["status"] == "event":
        lt, rt = (v["result"]["event"]["state"]["time_s"] for v in (left, right))
        event_delta, event_relative = abs(lt - rt), abs(lt - rt) / rt
    errors["event_time_s"] = event_delta
    return {"left": left_directory.name, "right": right_directory.name,
            "errors": errors, "maximum_error_times_s": times, "maximum_error_radii_cm": radii,
            "event_relative_difference": event_relative, "common_sample_count": count,
            "first_common_time_s": first, "last_common_time_s": last,
            "starter_thresholds_pass": {key: value is not None and value <= TOLERANCES[key] for key, value in errors.items()}}


def order_from_differences(coarse_error, fine_error, ratio):
    if coarse_error is None or fine_error is None or coarse_error <= 0 or fine_error <= 0:
        return None
    return log(coarse_error / fine_error) / log(ratio)


def branch_jump_j60(samples_path, *, interval_s=60.0, after_time_s=14520.0):
    """Return max |Cs(t)-2Cs(t-60)+Cs(t-120)| on the post-forcing grid."""
    previous = []
    maximum_jump = 0.0
    maximum_time = None
    count_at_least_5e4 = 0
    with Path(samples_path).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            time_s = float(row["time_s"])
            if time_s % interval_s != 0:
                continue
            moisture = float(row["moisture_surface"])
            if (len(previous) == 2
                    and previous[1][0] - previous[0][0] == interval_s
                    and time_s - previous[1][0] == interval_s
                    and time_s >= after_time_s):
                jump = abs(moisture - 2.0 * previous[1][1] + previous[0][1])
                if jump > maximum_jump:
                    maximum_jump, maximum_time = jump, time_s
                if jump >= 5.0e-4:
                    count_at_least_5e4 += 1
            previous = [*previous[-1:], (time_s, moisture)]
    return {"J60": maximum_jump, "maximum_time_s": maximum_time,
            "count_at_least_5e-4": count_at_least_5e4,
            "after_time_s": after_time_s}


def matrix_report(output, cells, dts, grading_exponent=1.0, *, expected_source_hashes=None):
    output = Path(output)
    summaries = {}
    for n in cells:
        for dt in dts:
            directory = output / case_label(n, dt, grading_exponent)
            path = directory / "summary.json"
            if path.exists():
                summaries[n, dt] = validated_summary(
                    directory,
                    expected_cells=n,
                    expected_dt_s=dt,
                    expected_grading_exponent=grading_exponent,
                    expected_source_hashes=expected_source_hashes,
                )
    comparisons, orders = [], []
    for axis in ("space", "time"):
        levels = sorted(cells) if axis == "space" else sorted(dts, reverse=True)
        fixed_values = dts if axis == "space" else cells
        for fixed in fixed_values:
            keys = [(level, fixed) if axis == "space" else (fixed, level) for level in levels]
            available = [key for key in keys if key in summaries]
            if len(available) < 2:
                continue
            horizon = min(summaries[key]["final_snapshot"]["time_s"] for key in available)
            pairs = []
            for left, right in zip(available, available[1:]):
                comparison = compare_cases(
                    output / case_label(*left, grading_exponent),
                    output / case_label(*right, grading_exponent),
                    last_time_s=horizon,
                )
                comparison.update(axis=axis, fixed=fixed)
                pairs.append(comparison)
                comparisons.append(comparison)
            for i, (coarse, fine) in enumerate(zip(pairs, pairs[1:])):
                level_values = [available[j][0 if axis == "space" else 1] for j in (i, i+1, i+2)]
                ratios = [level_values[1] / level_values[0], level_values[2] / level_values[1]] if axis == "space" else [level_values[0] / level_values[1], level_values[1] / level_values[2]]
                # Equal-ratio triples only; no fabricated log2 orders for 5/2/1.
                if abs(ratios[0] - ratios[1]) < 1e-10:
                    orders.append({"axis": axis, "fixed": fixed, "levels": level_values,
                                   "common_last_time_s": horizon,
                                   "observed_orders": {key: order_from_differences(coarse["errors"][key], fine["errors"][key], ratios[0]) for key in TOLERANCES}})
    table = []
    for (n, dt), item in sorted(summaries.items()):
        event = item["result"].get("event")
        jump = branch_jump_j60(output / case_label(n, dt, grading_exponent) / "samples.csv")
        table.append({"cells": n, "grading_exponent": grading_exponent,
                      "surface_half_width_xi": item["numerical_settings"]["surface_half_width_xi"],
                      "dt_s": dt, "status": item["result"]["status"],
                      "event_time_s": event["state"]["time_s"] if event else None,
                      "event_time_h": event["state"]["time_s"] / 3600 if event else None,
                      "event_bracket_width_s": event["right_time_s"] - event["left_time_s"] if event else None,
                      "maximum_moisture": item["maximum_location"]["maximum_moisture"],
                      "maximum_xi": item["maximum_location"]["xi"],
                      "branch_J60_second_difference": jump["J60"],
                      "branch_J60_maximum_time_s": jump["maximum_time_s"],
                      "branch_J60_count_ge_5e-4": jump["count_at_least_5e-4"],
                      "elapsed_wall_s": item["elapsed_wall_s"]})
    report = {"result_classification": "candidate_only", "convergence_verified": False,
              "grading_exponent": grading_exponent,
              "completed_cases": len(table), "requested_cases": len(cells) * len(dts),
              "tolerances": TOLERANCES,
              "branch_J60_definition": "max |C_surface(t)-2*C_surface(t-60s)+C_surface(t-120s)| for regular 60s samples with t>=14520s; count uses 5e-4 diagnostic threshold",
              "tolerance_basis": "V4 section12.2 starter limits; flux limits h*0.01 and hm*1e-4. Formatting is not accuracy. Coarse long-time matrix is screening only; early1s behavior and final output schedule need separate validation.",
              "cases": table, "comparisons": comparisons, "orders": orders}
    write_json(output / "convergence_report.json", report)
    if table:
        with (output / "event_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)
    return report
