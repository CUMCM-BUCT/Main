"""Streaming Q1 convergence runs and reproducible phase-4 artifacts.

Only sampled reconstructions are retained.  Accepted ``ModelState`` objects are
consumed one at a time so the formal 0--1800 s run does not grow with the
number of time steps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite, log
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Callable, Iterable, Sequence

from .conventions import INITIAL_RADIUS_M
from .fvm import RadialGrid, reconstruct_center, sample_reconstructed
from .inputs import EnvironmentInput
from .metadata import build_run_metadata
from .parameters import get_case_parameters, parameter_snapshot
from .solver import CoupledRadialSolver, ModelState, SolverOptions, StepDiagnostics


BASE_GRID_COUNTS = (20, 40, 80)
BASE_TIME_STEPS_S = (5.0, 2.0, 1.0, 0.5)
FINE_GRID_LEVELS = (320, 640, 1280)
FINE_TIME_STEPS_S = (1.0, 0.5, 0.25)
MAXIMUM_GRID_COUNT = 1280
MINIMUM_TIME_STEP_S = 0.25
Q1_END_TIME_S = 1800.0
CONVERGENCE_INTERVAL_S = 10.0
FINE_GATE_INTERVAL_S = 1.0
FORMAL_INTERVAL_S = 1.0
OUTPUT_RADII_CM = tuple(index / 10.0 for index in range(21))
# The global balance check re-sums up to 1280 finite-volume contributions and
# subtracts nearly equal storage and boundary terms.  A 1e-6 relative budget
# covers the measured floating-point aggregation floor across N=20..1280 while
# retaining two orders of magnitude of separation from the 1e-4 mutation used
# by the conservation regression test.
MAXIMUM_RELATIVE_BALANCE_RESIDUAL = 1.0e-6
MACHINE_EPSILON = 2.220446049250313e-16
ROUND_OFF_MULTIPLIER = 256.0
FIELD_ROUND_OFF_SCALES = {
    "temperature_field_C": 100.0,
    "temperature_center_C": 100.0,
    "temperature_surface_C": 100.0,
    "temperature_mean_C": 100.0,
    "moisture_field": 3.0,
    "moisture_center": 3.0,
    "moisture_surface": 3.0,
    "moisture_mean": 3.0,
    "heat_flux_W_m2": 1.0e3,
    "moisture_flux_m_s": 3.0e-6,
}

FIELD_TOLERANCES = {
    "temperature_field_C": 1.0e-2,
    "temperature_center_C": 1.0e-2,
    "temperature_surface_C": 1.0e-2,
    "temperature_mean_C": 1.0e-2,
    "moisture_field": 1.0e-3,
    "moisture_center": 1.0e-3,
    "moisture_surface": 1.0e-3,
    "moisture_mean": 1.0e-3,
    "heat_flux_W_m2": 2.5e-1,
    "moisture_flux_m_s": 8.0e-10,
}
MINIMUM_SPACE_ORDER = 1.5
MINIMUM_TIME_ORDER = 0.8
PRIMARY_ORDER_METRICS = ("temperature_field_C", "moisture_field")


@dataclass(frozen=True)
class Q1Snapshot:
    time_s: float
    temperatures_c: tuple[float, ...]
    moistures: tuple[float, ...]
    temperature_center_c: float
    temperature_surface_c: float
    temperature_weighted_mean_c: float
    moisture_center: float
    moisture_surface: float
    moisture_weighted_mean: float
    heat_flux_w_m2: float
    moisture_flux_m_s: float


@dataclass(frozen=True)
class Q1RunSummary:
    grid_cells: int
    nominal_dt_s: float
    end_time_s: float
    sample_interval_s: float
    sample_count: int
    accepted_steps: int
    elapsed_s: float
    maximum_picard_iterations: int
    mean_picard_iterations: float
    total_rejected_attempts: int
    maximum_temperature_scaled_residual: float
    maximum_moisture_scaled_residual: float
    maximum_step_mass_balance_residual: float
    final_cumulative_mass_balance_residual: float
    maximum_relative_step_mass_balance_residual: float
    relative_cumulative_mass_balance_residual: float
    maximum_step_heat_balance_residual: float
    final_cumulative_heat_balance_residual: float
    maximum_relative_step_heat_balance_residual: float
    relative_cumulative_heat_balance_residual: float
    all_state_and_output_values_finite: bool
    minimum_moisture: float
    first_step_surface_moisture: float
    first_step_moisture_flux_m_s: float
    maximum_reconstructed_temperature_c: float


@dataclass(frozen=True)
class Q1RunResult:
    summary: Q1RunSummary
    samples: tuple[Q1Snapshot, ...]
    final_sample: Q1Snapshot
    radii_cm: tuple[float, ...]


@dataclass(frozen=True)
class Comparison:
    left_grid_cells: int
    left_dt_s: float
    right_grid_cells: int
    right_dt_s: float
    errors: dict[str, float]
    maximum_error_times_s: dict[str, float]
    maximum_error_radii_cm: dict[str, float | None]
    sample_count: int
    first_time_s: float
    last_time_s: float


class ConvergenceGateError(RuntimeError):
    """The bounded fine gate failed; candidate evidence is available on disk."""


SampleCallback = Callable[[Q1Snapshot], None]
StepCallback = Callable[
    [ModelState, StepDiagnostics, float, float, float, float], None
]


def _sample_state(
    solver: CoupledRadialSolver,
    state: ModelState,
    boundary_temperature_c: float,
    boundary_moisture: float,
    environment_temperature_c: float,
    environment_moisture: float,
    radii_cm: Sequence[float],
) -> Q1Snapshot:
    temperatures = tuple(
        sample_reconstructed(
            solver.grid,
            state.temperatures_c,
            radius_cm * 1.0e-2 / INITIAL_RADIUS_M,
            boundary_temperature_c,
        )
        for radius_cm in radii_cm
    )
    moistures = tuple(
        sample_reconstructed(
            solver.grid,
            state.moistures,
            radius_cm * 1.0e-2 / INITIAL_RADIUS_M,
            boundary_moisture,
        )
        for radius_cm in radii_cm
    )
    weights = solver.grid.weights
    snapshot = Q1Snapshot(
        state.time_s,
        temperatures,
        moistures,
        reconstruct_center(state.temperatures_c),
        boundary_temperature_c,
        sum(weight * value for weight, value in zip(weights, state.temperatures_c)),
        reconstruct_center(state.moistures),
        boundary_moisture,
        sum(weight * value for weight, value in zip(weights, state.moistures)),
        solver.case.heat_transfer_coefficient
        * (boundary_temperature_c - environment_temperature_c),
        solver.case.mass_transfer_coefficient
        * (boundary_moisture - environment_moisture),
    )
    numeric_values = (
        *snapshot.temperatures_c,
        *snapshot.moistures,
        snapshot.temperature_center_c,
        snapshot.temperature_surface_c,
        snapshot.temperature_weighted_mean_c,
        snapshot.moisture_center,
        snapshot.moisture_surface,
        snapshot.moisture_weighted_mean,
        snapshot.heat_flux_w_m2,
        snapshot.moisture_flux_m_s,
    )
    if not all(isfinite(value) for value in numeric_values):
        raise RuntimeError("A non-finite Q1 state or reconstructed output was produced.")
    if min(
        *snapshot.moistures,
        snapshot.moisture_center,
        snapshot.moisture_surface,
        snapshot.moisture_weighted_mean,
    ) < 0.0:
        raise RuntimeError("A negative Q1 moisture state or reconstructed output was produced.")
    return snapshot


def run_q1_streaming(
    environment: EnvironmentInput,
    grid_cells: int,
    nominal_dt_s: float,
    *,
    end_time_s: float = Q1_END_TIME_S,
    sample_interval_s: float = CONVERGENCE_INTERVAL_S,
    radii_cm: Sequence[float] = OUTPUT_RADII_CM,
    options: SolverOptions | None = None,
    retain_samples: bool = True,
    on_sample: SampleCallback | None = None,
    on_step: StepCallback | None = None,
) -> Q1RunResult:
    """Run Q1 while retaining only requested output samples.

    Steps are clipped to the next sampling target.  This makes the output
    schedule exact even when a future solver configuration rejects and halves
    a nominal step.
    """

    if not isfinite(end_time_s) or end_time_s <= 0.0:
        raise ValueError("end_time_s must be finite and positive.")
    if not isfinite(sample_interval_s) or sample_interval_s <= 0.0:
        raise ValueError("sample_interval_s must be finite and positive.")
    if not isfinite(nominal_dt_s) or nominal_dt_s <= 0.0:
        raise ValueError("nominal_dt_s must be finite and positive.")
    if not radii_cm or any(
        not isfinite(value) or value < 0.0 or value > 2.0 for value in radii_cm
    ):
        raise ValueError("radii_cm must be non-empty finite positions in [0, 2].")

    solver = CoupledRadialSolver(
        "q1", RadialGrid(grid_cells), environment, INITIAL_RADIUS_M, options
    )
    state = solver.initial_state()
    initial_environment = environment.at(0.0)
    # The prescribed initial field owns the t=0 output.  The water initial and
    # Robin data are intentionally incompatible at the parabolic corner; the
    # first positive time layer performs the half-cell reconstruction.  Using
    # that reconstruction at t=0 would replace the prescribed surface value by
    # a grid-dependent algebraic face value and manufacture a convergence error.
    initial_snapshot = _sample_state(
        solver,
        state,
        state.temperatures_c[-1],
        state.moistures[-1],
        *initial_environment,
        radii_cm,
    )
    retained: list[Q1Snapshot] = []
    if retain_samples:
        retained.append(initial_snapshot)
    if on_sample is not None:
        on_sample(initial_snapshot)

    start = perf_counter()
    next_sample = min(sample_interval_s, end_time_s)
    sample_count = 1
    accepted_steps = 0
    iteration_total = 0
    maximum_iterations = 0
    rejected_total = 0
    maximum_temperature_residual = 0.0
    maximum_moisture_residual = 0.0
    maximum_step_mass_residual = 0.0
    maximum_step_heat_residual = 0.0
    maximum_relative_step_mass_residual = 0.0
    maximum_relative_step_heat_residual = 0.0
    cumulative_mass_outflow = 0.0
    cumulative_heat_outflow = 0.0
    minimum_moisture = min(*state.moistures, initial_snapshot.moisture_center)
    maximum_temperature = max(
        *state.temperatures_c,
        initial_snapshot.temperature_center_c,
        initial_snapshot.temperature_surface_c,
    )
    all_values_finite = all(
        isfinite(value)
        for value in (
            state.time_s,
            *state.temperatures_c,
            *state.moistures,
            *initial_snapshot.temperatures_c,
            *initial_snapshot.moistures,
        )
    )
    first_step_surface_moisture: float | None = None
    first_step_moisture_flux: float | None = None
    final_snapshot = initial_snapshot
    initial_moisture_mean = initial_snapshot.moisture_weighted_mean
    initial_heat_content = sum(
        solver.case.volumetric_heat_storage(moisture) * weight * temperature
        for weight, moisture, temperature in zip(
            solver.grid.weights, state.moistures, state.temperatures_c
        )
    )

    while state.time_s < end_time_s:
        previous = state
        distance_to_sample = next_sample - state.time_s
        distance_to_end = end_time_s - state.time_s
        planned_dt = min(nominal_dt_s, distance_to_sample, distance_to_end)
        if planned_dt <= 0.0:
            raise RuntimeError("Streaming schedule did not make positive progress.")
        step = solver.advance(previous, planned_dt)
        state = step.state
        diagnostics = step.diagnostics
        accepted_steps += 1
        iteration_total += diagnostics.picard_iterations
        maximum_iterations = max(maximum_iterations, diagnostics.picard_iterations)
        rejected_total += diagnostics.rejected_attempts
        maximum_temperature_residual = max(
            maximum_temperature_residual,
            diagnostics.maximum_temperature_scaled_residual,
        )
        maximum_moisture_residual = max(
            maximum_moisture_residual,
            diagnostics.maximum_moisture_scaled_residual,
        )
        reconstructed_center_moisture = reconstruct_center(state.moistures)
        reconstructed_center_temperature = reconstruct_center(state.temperatures_c)
        minimum_moisture = min(
            minimum_moisture,
            min(state.moistures),
            diagnostics.boundary.moisture,
            reconstructed_center_moisture,
        )
        maximum_temperature = max(
            maximum_temperature,
            max(state.temperatures_c),
            diagnostics.boundary.temperature_c,
            reconstructed_center_temperature,
        )
        all_values_finite = all_values_finite and all(
            isfinite(value)
            for value in (
                state.time_s,
                *state.temperatures_c,
                *state.moistures,
                diagnostics.requested_dt_s,
                diagnostics.accepted_dt_s,
                diagnostics.radius_m,
                diagnostics.environment_temperature_c,
                diagnostics.environment_moisture,
                diagnostics.relaxation,
                diagnostics.maximum_temperature_update,
                diagnostics.maximum_moisture_update,
                diagnostics.maximum_temperature_relative_update,
                diagnostics.maximum_moisture_relative_update,
                diagnostics.maximum_temperature_scaled_residual,
                diagnostics.maximum_moisture_scaled_residual,
                diagnostics.boundary.temperature_c,
                diagnostics.boundary.moisture,
                diagnostics.boundary.heat_transport,
                diagnostics.boundary.moisture_transport,
                reconstructed_center_temperature,
                reconstructed_center_moisture,
            )
        )

        heat_flux = solver.case.heat_transfer_coefficient * (
            diagnostics.boundary.temperature_c
            - diagnostics.environment_temperature_c
        )
        moisture_flux = solver.case.mass_transfer_coefficient * (
            diagnostics.boundary.moisture - diagnostics.environment_moisture
        )
        dt = diagnostics.accepted_dt_s
        mass_storage_change = sum(
            weight * (new - old)
            for weight, new, old in zip(
                solver.grid.weights, state.moistures, previous.moistures
            )
        )
        mass_boundary_term = dt * 2.0 / diagnostics.radius_m * moisture_flux
        mass_residual = mass_storage_change + mass_boundary_term
        heat_storage_change = sum(
            solver.case.volumetric_heat_storage(moisture)
            * weight
            * (new - old)
            for weight, moisture, new, old in zip(
                solver.grid.weights,
                state.moistures,
                state.temperatures_c,
                previous.temperatures_c,
            )
        )
        heat_boundary_term = dt * 2.0 / diagnostics.radius_m * heat_flux
        heat_residual = heat_storage_change + heat_boundary_term
        cumulative_mass_outflow += mass_boundary_term
        cumulative_heat_outflow += heat_boundary_term
        maximum_step_mass_residual = max(maximum_step_mass_residual, abs(mass_residual))
        maximum_step_heat_residual = max(maximum_step_heat_residual, abs(heat_residual))
        relative_step_mass_residual = abs(mass_residual) / max(
            abs(mass_storage_change), abs(mass_boundary_term), 1.0e-300
        )
        relative_step_heat_residual = abs(heat_residual) / max(
            abs(heat_storage_change), abs(heat_boundary_term), 1.0e-300
        )
        maximum_relative_step_mass_residual = max(
            maximum_relative_step_mass_residual, relative_step_mass_residual
        )
        maximum_relative_step_heat_residual = max(
            maximum_relative_step_heat_residual, relative_step_heat_residual
        )
        if first_step_surface_moisture is None:
            first_step_surface_moisture = diagnostics.boundary.moisture
            first_step_moisture_flux = moisture_flux
        if on_step is not None:
            on_step(
                state,
                diagnostics,
                mass_residual,
                heat_residual,
                relative_step_mass_residual,
                relative_step_heat_residual,
            )

        tolerance = 1.0e-10 * max(1.0, abs(next_sample))
        if abs(state.time_s - next_sample) <= tolerance:
            state = ModelState(
                next_sample,
                state.temperatures_c,
                state.moistures,
                state.surface_moisture,
            )
            snapshot = _sample_state(
                solver,
                state,
                diagnostics.boundary.temperature_c,
                diagnostics.boundary.moisture,
                diagnostics.environment_temperature_c,
                diagnostics.environment_moisture,
                radii_cm,
            )
            final_snapshot = snapshot
            sample_count += 1
            if retain_samples:
                retained.append(snapshot)
            if on_sample is not None:
                on_sample(snapshot)
            next_sample = min(next_sample + sample_interval_s, end_time_s)
        elif state.time_s > next_sample:
            raise RuntimeError("Accepted step crossed a requested sample time.")

    final_heat_content = sum(
        solver.case.volumetric_heat_storage(moisture) * weight * temperature
        for weight, moisture, temperature in zip(
            solver.grid.weights, state.moistures, state.temperatures_c
        )
    )
    final_mass_storage_change = (
        state_mean(solver.grid.weights, state.moistures) - initial_moisture_mean
    )
    final_mass_residual = final_mass_storage_change + cumulative_mass_outflow
    final_heat_storage_change = final_heat_content - initial_heat_content
    final_heat_residual = final_heat_storage_change + cumulative_heat_outflow
    summary = Q1RunSummary(
        grid_cells,
        nominal_dt_s,
        end_time_s,
        sample_interval_s,
        sample_count,
        accepted_steps,
        perf_counter() - start,
        maximum_iterations,
        iteration_total / accepted_steps,
        rejected_total,
        maximum_temperature_residual,
        maximum_moisture_residual,
        maximum_step_mass_residual,
        final_mass_residual,
        maximum_relative_step_mass_residual,
        abs(final_mass_residual)
        / max(
            initial_moisture_mean,
            abs(final_mass_storage_change),
            abs(cumulative_mass_outflow),
            1.0e-300,
        ),
        maximum_step_heat_residual,
        final_heat_residual,
        maximum_relative_step_heat_residual,
        abs(final_heat_residual)
        / max(
            abs(final_heat_storage_change),
            abs(cumulative_heat_outflow),
            1.0e-300,
        ),
        all_values_finite,
        minimum_moisture,
        first_step_surface_moisture if first_step_surface_moisture is not None else float("nan"),
        first_step_moisture_flux if first_step_moisture_flux is not None else float("nan"),
        maximum_temperature,
    )
    return Q1RunResult(summary, tuple(retained), final_snapshot, tuple(radii_cm))


def state_mean(weights: Sequence[float], values: Sequence[float]) -> float:
    return sum(weight * value for weight, value in zip(weights, values))


def compare_runs(
    left: Q1RunResult,
    right: Q1RunResult,
    *,
    first_time_s: float | None = None,
) -> Comparison:
    if len(left.samples) != len(right.samples) or not left.samples:
        raise ValueError("Compared runs must retain the same non-empty sampling schedule.")
    if left.radii_cm != right.radii_cm:
        raise ValueError("Compared runs do not share identical projection positions.")
    sample_pairs = [
        (left_sample, right_sample)
        for left_sample, right_sample in zip(left.samples, right.samples)
        if first_time_s is None or left_sample.time_s >= first_time_s
    ]
    if not sample_pairs:
        raise ValueError("No common samples remain in the requested comparison interval.")
    errors = {name: -1.0 for name in FIELD_TOLERANCES}
    maximum_times = {name: sample_pairs[0][0].time_s for name in FIELD_TOLERANCES}
    maximum_radii: dict[str, float | None] = {
        "temperature_field_C": left.radii_cm[0],
        "temperature_center_C": 0.0,
        "temperature_surface_C": INITIAL_RADIUS_M * 100.0,
        "temperature_mean_C": None,
        "moisture_field": left.radii_cm[0],
        "moisture_center": 0.0,
        "moisture_surface": INITIAL_RADIUS_M * 100.0,
        "moisture_mean": None,
        "heat_flux_W_m2": INITIAL_RADIUS_M * 100.0,
        "moisture_flux_m_s": INITIAL_RADIUS_M * 100.0,
    }
    for left_sample, right_sample in sample_pairs:
        if left_sample.time_s != right_sample.time_s:
            raise ValueError("Compared runs do not share identical sampling times.")
        for metric, left_values, right_values in (
            (
                "temperature_field_C",
                left_sample.temperatures_c,
                right_sample.temperatures_c,
            ),
            ("moisture_field", left_sample.moistures, right_sample.moistures),
        ):
            for radius_cm, left_value, right_value in zip(
                left.radii_cm, left_values, right_values
            ):
                difference = abs(left_value - right_value)
                if difference > errors[metric]:
                    errors[metric] = difference
                    maximum_times[metric] = left_sample.time_s
                    maximum_radii[metric] = radius_cm
        pairs = {
            "temperature_center_C": (
                left_sample.temperature_center_c,
                right_sample.temperature_center_c,
            ),
            "temperature_surface_C": (
                left_sample.temperature_surface_c,
                right_sample.temperature_surface_c,
            ),
            "temperature_mean_C": (
                left_sample.temperature_weighted_mean_c,
                right_sample.temperature_weighted_mean_c,
            ),
            "moisture_center": (left_sample.moisture_center, right_sample.moisture_center),
            "moisture_surface": (left_sample.moisture_surface, right_sample.moisture_surface),
            "moisture_mean": (
                left_sample.moisture_weighted_mean,
                right_sample.moisture_weighted_mean,
            ),
            "heat_flux_W_m2": (left_sample.heat_flux_w_m2, right_sample.heat_flux_w_m2),
            "moisture_flux_m_s": (
                left_sample.moisture_flux_m_s,
                right_sample.moisture_flux_m_s,
            ),
        }
        for name, values in pairs.items():
            difference = abs(values[0] - values[1])
            if difference > errors[name]:
                errors[name] = difference
                maximum_times[name] = left_sample.time_s
    return Comparison(
        left.summary.grid_cells,
        left.summary.nominal_dt_s,
        right.summary.grid_cells,
        right.summary.nominal_dt_s,
        errors,
        maximum_times,
        maximum_radii,
        len(sample_pairs),
        sample_pairs[0][0].time_s,
        sample_pairs[-1][0].time_s,
    )


def observed_order(
    coarse_error: float,
    fine_error: float,
    refinement: float,
    *,
    noise_floor: float = 128.0 * MACHINE_EPSILON,
) -> float | None:
    if coarse_error <= noise_floor or fine_error <= noise_floor:
        return None
    return log(coarse_error / fine_error) / log(refinement)


def observed_order_three_levels(
    coarse_error: float,
    fine_error: float,
    coarse_step: float,
    middle_step: float,
    fine_step: float,
    *,
    noise_floor: float = 128.0 * MACHINE_EPSILON,
) -> float | None:
    """Observed order for adjacent differences on possibly unequal step ratios."""

    if coarse_error <= noise_floor or fine_error <= noise_floor:
        return None
    target = coarse_error / fine_error

    def ratio(order: float) -> float:
        if abs(order) < 1.0e-10:
            return log(coarse_step / middle_step) / log(middle_step / fine_step)
        return (coarse_step**order - middle_step**order) / (
            middle_step**order - fine_step**order
        )

    lower = -10.0
    upper = 10.0
    lower_value = ratio(lower) - target
    upper_value = ratio(upper) - target
    if lower_value * upper_value > 0.0:
        return None
    for _ in range(100):
        middle = (lower + upper) / 2.0
        value = ratio(middle) - target
        if value == 0.0:
            return middle
        if lower_value * value <= 0.0:
            upper = middle
        else:
            lower = middle
            lower_value = value
    return (lower + upper) / 2.0


def _comparison_passes_thresholds(comparison: Comparison) -> bool:
    return all(
        comparison.errors[name] <= tolerance
        for name, tolerance in FIELD_TOLERANCES.items()
    )


def _orders_and_monotonicity(
    coarse: Comparison, fine: Comparison, refinement: float
) -> tuple[dict[str, float | None], dict[str, bool | None]]:
    orders = {
        name: observed_order(
            coarse.errors[name],
            fine.errors[name],
            refinement,
            noise_floor=_metric_noise_floor(name),
        )
        for name in FIELD_TOLERANCES
    }
    monotonic = {
        name: _metric_monotonicity(
            name, coarse.errors[name], fine.errors[name]
        )
        for name in FIELD_TOLERANCES
    }
    return orders, monotonic


def _metric_noise_floor(name: str) -> float:
    return ROUND_OFF_MULTIPLIER * MACHINE_EPSILON * FIELD_ROUND_OFF_SCALES[name]


def _metric_monotonicity(
    name: str, coarse_error: float, fine_error: float
) -> bool | None:
    if (
        name not in PRIMARY_ORDER_METRICS
        and max(coarse_error, fine_error) <= _metric_noise_floor(name)
    ):
        return None
    return fine_error <= coarse_error


def _monotonicity_gate(monotonicity: dict[str, bool | None]) -> bool:
    return all(
        value is True if name in PRIMARY_ORDER_METRICS else value is not False
        for name, value in monotonicity.items()
    )


def _order_gate(orders: dict[str, float | None], minimum: float) -> bool:
    return all(orders[name] is not None and orders[name] >= minimum for name in PRIMARY_ORDER_METRICS)


def _run_key(grid_cells: int, dt_s: float) -> str:
    return f"N{grid_cells}_dt{format(dt_s, 'g')}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: float) -> str:
    return format(value, ".17g")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_head(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_status(repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def _git_tracked_status(repo_root: Path) -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(line for line in completed.stdout.splitlines() if line)


def _git_path_is_tracked(repo_root: Path, path: Path) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path.relative_to(repo_root).as_posix()],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _summary_row(summary: Q1RunSummary) -> list[object]:
    return [_float(value) if isinstance(value, float) else value for value in asdict(summary).values()]


def run_health_failures(
    result: Q1RunResult,
    options: SolverOptions,
    *,
    balance_tolerance: float = MAXIMUM_RELATIVE_BALANCE_RESIDUAL,
) -> tuple[str, ...]:
    """Return bounded, dimensionless health-gate failures for one run."""

    summary = result.summary
    failures: list[str] = []
    numeric_summary = (
        summary.nominal_dt_s,
        summary.end_time_s,
        summary.sample_interval_s,
        summary.elapsed_s,
        summary.mean_picard_iterations,
        summary.maximum_temperature_scaled_residual,
        summary.maximum_moisture_scaled_residual,
        summary.maximum_step_mass_balance_residual,
        summary.final_cumulative_mass_balance_residual,
        summary.maximum_relative_step_mass_balance_residual,
        summary.relative_cumulative_mass_balance_residual,
        summary.maximum_step_heat_balance_residual,
        summary.final_cumulative_heat_balance_residual,
        summary.maximum_relative_step_heat_balance_residual,
        summary.relative_cumulative_heat_balance_residual,
        summary.minimum_moisture,
        summary.first_step_surface_moisture,
        summary.first_step_moisture_flux_m_s,
        summary.maximum_reconstructed_temperature_c,
    )
    if not summary.all_state_and_output_values_finite or not all(
        isfinite(value) for value in numeric_summary
    ):
        failures.append("non_finite_state_or_output")
    if summary.minimum_moisture < 0.0:
        failures.append("negative_moisture")
    if summary.maximum_temperature_scaled_residual > options.residual_tolerance:
        failures.append("temperature_residual")
    if summary.maximum_moisture_scaled_residual > options.residual_tolerance:
        failures.append("moisture_residual")
    if summary.total_rejected_attempts != 0:
        failures.append("rejected_attempts")
    if summary.first_step_surface_moisture <= 2.0:
        failures.append("first_step_high_branch")
    if summary.first_step_moisture_flux_m_s <= 1.0e-6:
        failures.append("first_step_nondegenerate_flux")
    balance_values = {
        "step_mass_balance": summary.maximum_relative_step_mass_balance_residual,
        "cumulative_mass_balance": summary.relative_cumulative_mass_balance_residual,
        "step_heat_balance": summary.maximum_relative_step_heat_balance_residual,
        "cumulative_heat_balance": summary.relative_cumulative_heat_balance_residual,
    }
    failures.extend(
        name for name, value in balance_values.items() if value > balance_tolerance
    )
    for sample in result.samples:
        sample_values = (
            sample.time_s,
            *sample.temperatures_c,
            *sample.moistures,
            sample.temperature_center_c,
            sample.temperature_surface_c,
            sample.temperature_weighted_mean_c,
            sample.moisture_center,
            sample.moisture_surface,
            sample.moisture_weighted_mean,
            sample.heat_flux_w_m2,
            sample.moisture_flux_m_s,
        )
        if not all(isfinite(value) for value in sample_values):
            failures.append("non_finite_retained_sample")
            break
        if min(
            *sample.moistures,
            sample.moisture_center,
            sample.moisture_surface,
            sample.moisture_weighted_mean,
        ) < 0.0:
            failures.append("negative_retained_sample")
            break
    return tuple(dict.fromkeys(failures))


def execute_phase4(
    output_directory: Path,
    environment: EnvironmentInput,
    *,
    repo_root: Path,
    grid_counts: Sequence[int] = BASE_GRID_COUNTS,
    time_steps_s: Sequence[float] = BASE_TIME_STEPS_S,
    end_time_s: float = Q1_END_TIME_S,
    convergence_interval_s: float = CONVERGENCE_INTERVAL_S,
    fine_grid_levels: Sequence[int] = FINE_GRID_LEVELS,
    fine_time_steps_s: Sequence[float] = FINE_TIME_STEPS_S,
    fine_gate_interval_s: float = FINE_GATE_INTERVAL_S,
    formal_interval_s: float = FORMAL_INTERVAL_S,
    expected_commit: str | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    """Execute the matrix, gate convergence, and write traceable CSV/JSON files."""

    actual_commit = _git_head(repo_root)
    if expected_commit is not None and not actual_commit.startswith(expected_commit):
        raise RuntimeError(
            f"Expected commit {expected_commit}, found {actual_commit}."
        )
    source_paths = tuple(sorted((repo_root / "src" / "a_model").glob("*.py"))) + (
        repo_root / "scripts" / "run_q1_phase4.py",
        repo_root / "tests" / "test_q1_phase4.py",
    )
    untracked_sources = tuple(
        path.relative_to(repo_root).as_posix()
        for path in source_paths
        if not _git_path_is_tracked(repo_root, path)
    )
    tracked_status = _git_tracked_status(repo_root)
    if not allow_dirty and (tracked_status or untracked_sources):
        raise RuntimeError(
            "Formal phase-4 execution requires committed source files and a clean "
            "tracked worktree; commit the phase-4 code first or use allow_dirty only "
            "for tests/candidate diagnostics."
        )
    source_file_hashes = {
        path.relative_to(repo_root).as_posix(): _sha256(path) for path in source_paths
    }
    working_tree_status = _git_status(repo_root)
    if tuple(grid_counts) != BASE_GRID_COUNTS:
        raise ValueError("The required base matrix uses N=20, 40, and 80.")
    if tuple(time_steps_s) != BASE_TIME_STEPS_S:
        raise ValueError("The required base matrix uses dt=5, 2, 1, and 0.5 s.")
    if max(fine_grid_levels, default=0) > MAXIMUM_GRID_COUNT:
        raise ValueError(f"Fine-grid gate is bounded at N={MAXIMUM_GRID_COUNT}.")
    if len(fine_grid_levels) != 3 or tuple(fine_grid_levels) != tuple(
        sorted(fine_grid_levels)
    ):
        raise ValueError("fine_grid_levels must contain three increasing grid counts.")
    if any(
        right != 2 * left for left, right in zip(fine_grid_levels, fine_grid_levels[1:])
    ):
        raise ValueError("fine_grid_levels must use successive factor-two refinement.")
    if tuple(fine_grid_levels) != FINE_GRID_LEVELS:
        raise ValueError(
            "Formal spatial evidence must be N320 -> N640 -> N1280."
        )
    if min(fine_time_steps_s, default=float("inf")) < MINIMUM_TIME_STEP_S:
        raise ValueError(f"Fine time-step gate is bounded at dt={MINIMUM_TIME_STEP_S} s.")
    if len(fine_time_steps_s) != 3 or tuple(fine_time_steps_s) != tuple(
        sorted(fine_time_steps_s, reverse=True)
    ):
        raise ValueError("fine_time_steps_s must contain three decreasing time steps.")
    if any(
        abs(left / right - 2.0) > 1.0e-12
        for left, right in zip(fine_time_steps_s, fine_time_steps_s[1:])
    ):
        raise ValueError("fine_time_steps_s must use successive factor-two refinement.")
    if tuple(fine_time_steps_s) != FINE_TIME_STEPS_S:
        raise ValueError(
            "Formal temporal evidence must be dt=1 -> 0.5 -> 0.25 s."
        )
    if fine_gate_interval_s != 1.0 or formal_interval_s != 1.0:
        raise ValueError("Final convergence and formal Q1 sampling must be exactly 1 s.")
    output_directory.mkdir(parents=True, exist_ok=True)
    options = SolverOptions()
    run_results: dict[tuple[int, float], Q1RunResult] = {}
    artifact_paths: list[Path] = []
    matrix_samples_path = output_directory / "q1_matrix_samples.csv"
    matrix_qoi_path = output_directory / "q1_matrix_qoi.csv"
    matrix_summary_path = output_directory / "q1_matrix_summary.csv"
    artifact_paths.extend((matrix_samples_path, matrix_qoi_path, matrix_summary_path))

    with (
        matrix_samples_path.open("w", newline="", encoding="utf-8") as sample_stream,
        matrix_qoi_path.open("w", newline="", encoding="utf-8") as qoi_stream,
        matrix_summary_path.open("w", newline="", encoding="utf-8") as summary_stream,
    ):
        sample_writer = csv.writer(sample_stream)
        qoi_writer = csv.writer(qoi_stream)
        summary_writer = csv.writer(summary_stream)
        sample_writer.writerow(
            ("run_id", "grid_cells", "dt_s", "time_s", "radius_cm", "temperature_C", "moisture_dry_basis")
        )
        qoi_writer.writerow(
            (
                "run_id", "grid_cells", "dt_s", "time_s", "temperature_center_C",
                "temperature_surface_C", "temperature_weighted_mean_C", "moisture_center",
                "moisture_surface", "moisture_weighted_mean", "qT_W_m2", "qC_m_s",
            )
        )
        summary_fields = tuple(Q1RunSummary.__dataclass_fields__)
        summary_writer.writerow(("run_id", *summary_fields))

        def run_one(grid_cells: int, dt_s: float) -> Q1RunResult:
            key = _run_key(grid_cells, dt_s)
            result = run_q1_streaming(
                environment,
                grid_cells,
                dt_s,
                end_time_s=end_time_s,
                sample_interval_s=convergence_interval_s,
            )
            run_results[(grid_cells, dt_s)] = result
            for snapshot in result.samples:
                for radius_cm, temperature, moisture in zip(
                    OUTPUT_RADII_CM, snapshot.temperatures_c, snapshot.moistures
                ):
                    sample_writer.writerow(
                        (key, grid_cells, _float(dt_s), _float(snapshot.time_s), _float(radius_cm), _float(temperature), _float(moisture))
                    )
                qoi_writer.writerow(
                    (
                        key, grid_cells, _float(dt_s), _float(snapshot.time_s),
                        _float(snapshot.temperature_center_c), _float(snapshot.temperature_surface_c),
                        _float(snapshot.temperature_weighted_mean_c), _float(snapshot.moisture_center),
                        _float(snapshot.moisture_surface), _float(snapshot.moisture_weighted_mean),
                        _float(snapshot.heat_flux_w_m2), _float(snapshot.moisture_flux_m_s),
                    )
                )
            summary_writer.writerow((key, *_summary_row(result.summary)))
            sample_stream.flush()
            qoi_stream.flush()
            summary_stream.flush()
            return result

        for grid_cells in grid_counts:
            for dt_s in time_steps_s:
                run_one(grid_cells, dt_s)

        finest_grid = max(grid_counts)
        finest_dt = min(time_steps_s)

        def assess(grid: int, dt: float) -> dict[str, object]:
            space_coarse = compare_runs(run_results[(grid // 4, dt)], run_results[(grid // 2, dt)])
            space_fine = compare_runs(run_results[(grid // 2, dt)], run_results[(grid, dt)])
            time_coarse = compare_runs(run_results[(grid, dt * 4.0)], run_results[(grid, dt * 2.0)])
            time_fine = compare_runs(run_results[(grid, dt * 2.0)], run_results[(grid, dt)])
            space_orders, space_monotonic = _orders_and_monotonicity(space_coarse, space_fine, 2.0)
            time_orders, time_monotonic = _orders_and_monotonicity(time_coarse, time_fine, 2.0)
            return {
                "space_coarse": space_coarse,
                "space_fine": space_fine,
                "time_coarse": time_coarse,
                "time_fine": time_fine,
                "space_orders": space_orders,
                "time_orders": time_orders,
                "space_monotonic": space_monotonic,
                "time_monotonic": time_monotonic,
                "space_pass": _comparison_passes_thresholds(space_fine)
                and _order_gate(space_orders, MINIMUM_SPACE_ORDER)
                and _monotonicity_gate(space_monotonic),
                "time_pass": _comparison_passes_thresholds(time_fine)
                and _order_gate(time_orders, MINIMUM_TIME_ORDER)
                and _monotonicity_gate(time_monotonic),
            }

        assessment = assess(finest_grid, finest_dt)
    selected_grid = max(fine_grid_levels)
    selected_dt = min(fine_time_steps_s)
    fine_results: dict[tuple[int, float], Q1RunResult] = {}
    fine_samples_path = output_directory / "q1_fine_gate_samples.csv"
    fine_qoi_path = output_directory / "q1_fine_gate_qoi.csv"
    fine_summary_path = output_directory / "q1_fine_gate_summary.csv"
    candidate_diagnostics_path = output_directory / "q1_candidate_diagnostics.csv"
    artifact_paths.extend((fine_samples_path, fine_qoi_path, fine_summary_path))
    with (
        fine_samples_path.open("w", newline="", encoding="utf-8") as sample_stream,
        fine_qoi_path.open("w", newline="", encoding="utf-8") as qoi_stream,
        fine_summary_path.open("w", newline="", encoding="utf-8") as summary_stream,
        candidate_diagnostics_path.open("w", newline="", encoding="utf-8") as diagnostics_stream,
    ):
        sample_writer = csv.writer(sample_stream)
        qoi_writer = csv.writer(qoi_stream)
        summary_writer = csv.writer(summary_stream)
        diagnostics_writer = csv.writer(diagnostics_stream)
        sample_writer.writerow(
            ("run_id", "grid_cells", "dt_s", "time_s", "radius_cm", "temperature_C", "moisture_dry_basis")
        )
        qoi_writer.writerow(
            (
                "run_id", "grid_cells", "dt_s", "time_s", "temperature_center_C",
                "temperature_surface_C", "temperature_weighted_mean_C", "moisture_center",
                "moisture_surface", "moisture_weighted_mean", "qT_W_m2", "qC_m_s",
            )
        )
        summary_writer.writerow(("run_id", *Q1RunSummary.__dataclass_fields__))
        diagnostics_writer.writerow(
            (
                "time_s", "requested_dt_s", "accepted_dt_s", "picard_iterations",
                "relaxation", "rejected_attempts", "heat_linear_solves",
                "moisture_linear_solves", "max_temperature_update",
                "max_moisture_update", "max_temperature_relative_update",
                "max_moisture_relative_update", "max_temperature_scaled_residual",
                "max_moisture_scaled_residual", "step_mass_balance_residual",
                "step_heat_balance_residual", "relative_step_mass_balance_residual",
                "relative_step_heat_balance_residual", "surface_temperature_C",
                "surface_moisture", "qT_W_m2", "qC_m_s",
            )
        )
        q1_case = get_case_parameters("q1")

        def candidate_step(
            state: ModelState,
            diagnostics: StepDiagnostics,
            mass_residual: float,
            heat_residual: float,
            relative_mass_residual: float,
            relative_heat_residual: float,
        ) -> None:
            diagnostics_writer.writerow(
                (
                    _float(state.time_s), _float(diagnostics.requested_dt_s),
                    _float(diagnostics.accepted_dt_s), diagnostics.picard_iterations,
                    _float(diagnostics.relaxation), diagnostics.rejected_attempts,
                    diagnostics.heat_linear_solves, diagnostics.moisture_linear_solves,
                    _float(diagnostics.maximum_temperature_update),
                    _float(diagnostics.maximum_moisture_update),
                    _float(diagnostics.maximum_temperature_relative_update),
                    _float(diagnostics.maximum_moisture_relative_update),
                    _float(diagnostics.maximum_temperature_scaled_residual),
                    _float(diagnostics.maximum_moisture_scaled_residual),
                    _float(mass_residual), _float(heat_residual),
                    _float(relative_mass_residual), _float(relative_heat_residual),
                    _float(diagnostics.boundary.temperature_c),
                    _float(diagnostics.boundary.moisture),
                    _float(
                        q1_case.heat_transfer_coefficient
                        * (
                            diagnostics.boundary.temperature_c
                            - diagnostics.environment_temperature_c
                        )
                    ),
                    _float(
                        q1_case.mass_transfer_coefficient
                        * (
                            diagnostics.boundary.moisture
                            - diagnostics.environment_moisture
                        )
                    ),
                )
            )

        fine_keys = [(grid, selected_dt) for grid in fine_grid_levels]
        fine_keys.extend((selected_grid, dt_s) for dt_s in fine_time_steps_s)
        for grid_cells, dt_s in dict.fromkeys(fine_keys):
            key = _run_key(grid_cells, dt_s)
            result = run_q1_streaming(
                environment,
                grid_cells,
                dt_s,
                end_time_s=end_time_s,
                sample_interval_s=fine_gate_interval_s,
                on_step=(
                    candidate_step
                    if (grid_cells, dt_s) == (selected_grid, selected_dt)
                    else None
                ),
            )
            fine_results[(grid_cells, dt_s)] = result
            for snapshot in result.samples:
                for radius_cm, temperature, moisture in zip(
                    OUTPUT_RADII_CM, snapshot.temperatures_c, snapshot.moistures
                ):
                    sample_writer.writerow(
                        (
                            key, grid_cells, _float(dt_s), _float(snapshot.time_s),
                            _float(radius_cm), _float(temperature), _float(moisture),
                        )
                    )
                qoi_writer.writerow(
                    (
                        key, grid_cells, _float(dt_s), _float(snapshot.time_s),
                        _float(snapshot.temperature_center_c),
                        _float(snapshot.temperature_surface_c),
                        _float(snapshot.temperature_weighted_mean_c),
                        _float(snapshot.moisture_center), _float(snapshot.moisture_surface),
                        _float(snapshot.moisture_weighted_mean),
                        _float(snapshot.heat_flux_w_m2), _float(snapshot.moisture_flux_m_s),
                    )
                )
            summary_writer.writerow((key, *_summary_row(result.summary)))
            sample_stream.flush()
            qoi_stream.flush()
            summary_stream.flush()
            diagnostics_stream.flush()

    space_coarse = compare_runs(
        fine_results[(fine_grid_levels[0], selected_dt)],
        fine_results[(fine_grid_levels[1], selected_dt)],
        first_time_s=1.0,
    )
    space_fine = compare_runs(
        fine_results[(fine_grid_levels[1], selected_dt)],
        fine_results[(fine_grid_levels[2], selected_dt)],
        first_time_s=1.0,
    )
    time_coarse = compare_runs(
        fine_results[(selected_grid, fine_time_steps_s[0])],
        fine_results[(selected_grid, fine_time_steps_s[1])],
        first_time_s=1.0,
    )
    time_fine = compare_runs(
        fine_results[(selected_grid, fine_time_steps_s[1])],
        fine_results[(selected_grid, fine_time_steps_s[2])],
        first_time_s=1.0,
    )
    space_orders, space_monotonic = _orders_and_monotonicity(
        space_coarse, space_fine, 2.0
    )
    time_orders, time_monotonic = _orders_and_monotonicity(
        time_coarse, time_fine, 2.0
    )
    final_space_pass = (
        _comparison_passes_thresholds(space_fine)
        and _order_gate(space_orders, MINIMUM_SPACE_ORDER)
        and _monotonicity_gate(space_monotonic)
    )
    final_time_pass = (
        _comparison_passes_thresholds(time_fine)
        and _order_gate(time_orders, MINIMUM_TIME_ORDER)
        and _monotonicity_gate(time_monotonic)
    )

    comparisons: list[Comparison] = []
    ordered_grids = sorted({key[0] for key in run_results})
    ordered_dts = sorted({key[1] for key in run_results}, reverse=True)
    for dt_s in ordered_dts:
        present = [grid for grid in ordered_grids if (grid, dt_s) in run_results]
        for left, right in zip(present, present[1:]):
            comparisons.append(compare_runs(run_results[(left, dt_s)], run_results[(right, dt_s)]))
    for grid in ordered_grids:
        present = [dt for dt in ordered_dts if (grid, dt) in run_results]
        for left, right in zip(present, present[1:]):
            comparisons.append(compare_runs(run_results[(grid, left)], run_results[(grid, right)]))
    comparisons.extend((space_coarse, space_fine, time_coarse, time_fine))

    comparison_path = output_directory / "q1_convergence_comparisons.csv"
    artifact_paths.append(comparison_path)
    with comparison_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "left_run", "right_run", "sample_interval_s", "sample_count",
                "first_time_s", "last_time_s", "metric", "maximum_absolute_difference",
                "maximum_error_time_s", "maximum_error_radius_cm", "tolerance",
            )
        )
        for comparison in comparisons:
            for metric, error in comparison.errors.items():
                writer.writerow(
                    (
                        _run_key(comparison.left_grid_cells, comparison.left_dt_s),
                        _run_key(comparison.right_grid_cells, comparison.right_dt_s),
                        _float(
                            (comparison.last_time_s - comparison.first_time_s)
                            / max(comparison.sample_count - 1, 1)
                        ),
                        comparison.sample_count,
                        _float(comparison.first_time_s),
                        _float(comparison.last_time_s),
                        metric,
                        _float(error),
                        _float(comparison.maximum_error_times_s[metric]),
                        (
                            ""
                            if comparison.maximum_error_radii_cm[metric] is None
                            else _float(comparison.maximum_error_radii_cm[metric])
                        ),
                        _float(FIELD_TOLERANCES[metric]),
                    )
                )

    def serialize_comparison(comparison: Comparison) -> dict[str, object]:
        return {
            "left_run": _run_key(comparison.left_grid_cells, comparison.left_dt_s),
            "right_run": _run_key(comparison.right_grid_cells, comparison.right_dt_s),
            "errors": comparison.errors,
            "maximum_error_times_s": comparison.maximum_error_times_s,
            "maximum_error_radii_cm": comparison.maximum_error_radii_cm,
            "sampling": {
                "sample_count": comparison.sample_count,
                "first_time_s": comparison.first_time_s,
                "last_time_s": comparison.last_time_s,
                "interval_s": (
                    (comparison.last_time_s - comparison.first_time_s)
                    / max(comparison.sample_count - 1, 1)
                ),
            },
        }

    convergence_report = {
        "acceptance": {
            "minimum_space_order": MINIMUM_SPACE_ORDER,
            "minimum_time_order": MINIMUM_TIME_ORDER,
            "primary_order_metrics": PRIMARY_ORDER_METRICS,
            "absolute_tolerances": FIELD_TOLERANCES,
            "monotonicity_noise_floor": {
                name: _metric_noise_floor(name) for name in FIELD_TOLERANCES
            },
            "noise_floor_policy": (
                "Observed order is null when either adjacent error is at or below "
                "the metric-scaled floating-point floor. Non-primary monotonicity "
                "is null only when both adjacent errors are at or below that floor; "
                "primary T/C field monotonicity remains mandatory."
            ),
            "requires_finite_nonnegative_state": True,
            "maximum_scaled_residual": options.residual_tolerance,
            "required_rejected_attempts": 0,
            "maximum_relative_balance_residual": MAXIMUM_RELATIVE_BALANCE_RESIDUAL,
            "balance_roundoff_budget": (
                "The 1e-6 relative limit covers global weighted summation and "
                "storage/boundary cancellation through N=1280; short-grid probes "
                "remain at least two orders below a 1e-4 conservation mutation."
            ),
            "balance_scaling": {
                "step": "abs(balance residual) / max(abs(storage increment), abs(boundary increment))",
                "cumulative_mass": "abs(cumulative residual) / max(initial mean C, abs(total storage change), abs(total outflow))",
                "cumulative_heat": "abs(cumulative residual) / max(abs(total storage change), abs(total boundary exchange))",
            },
            "first_step_high_branch": {
                "surface_moisture_greater_than": 2.0,
                "qC_m_s_greater_than": 1.0e-6,
            },
        },
        "base_assessment": {
            "space_pass": assessment["space_pass"],
            "time_pass": assessment["time_pass"],
            "space_coarse": serialize_comparison(assessment["space_coarse"]),
            "space_fine": serialize_comparison(assessment["space_fine"]),
            "time_coarse": serialize_comparison(assessment["time_coarse"]),
            "time_fine": serialize_comparison(assessment["time_fine"]),
            "space_orders": assessment["space_orders"],
            "time_orders": assessment["time_orders"],
            "space_monotonic": assessment["space_monotonic"],
            "time_monotonic": assessment["time_monotonic"],
        },
        "bounded_fine_gate": {
            "maximum_grid_count": MAXIMUM_GRID_COUNT,
            "minimum_time_step_s": MINIMUM_TIME_STEP_S,
            "grid_levels": list(fine_grid_levels),
            "time_steps_s": list(fine_time_steps_s),
            "sample_interval_s": fine_gate_interval_s,
            "required_time_range_s": [1.0, end_time_s],
            "space_pass": final_space_pass,
            "time_pass": final_time_pass,
            "space_coarse": serialize_comparison(space_coarse),
            "space_fine": serialize_comparison(space_fine),
            "time_coarse": serialize_comparison(time_coarse),
            "time_fine": serialize_comparison(time_fine),
            "space_orders": space_orders,
            "time_orders": time_orders,
            "space_monotonic": space_monotonic,
            "time_monotonic": time_monotonic,
        },
        "candidate_formal_run": _run_key(selected_grid, selected_dt),
    }
    all_results = {**run_results, **fine_results}
    health_failures = {
        _run_key(*key): list(failures)
        for key, result in all_results.items()
        if (
            failures := run_health_failures(
                result,
                options,
                balance_tolerance=MAXIMUM_RELATIVE_BALANCE_RESIDUAL,
            )
        )
    }
    health_pass = not health_failures
    convergence_report["health_pass"] = health_pass
    convergence_report["health_failures"] = health_failures
    base_gate_pass = bool(
        assessment["space_pass"]
        and assessment["time_pass"]
        and not any(_run_key(*key) in health_failures for key in run_results)
    )
    convergence_report["base_gate_pass"] = base_gate_pass
    convergence_report["final_gate_pass"] = bool(
        final_space_pass and final_time_pass and health_pass
    )
    convergence_report["selected_formal_run"] = (
        _run_key(selected_grid, selected_dt)
        if convergence_report["final_gate_pass"]
        else None
    )

    orders_path = output_directory / "q1_convergence_orders.csv"
    artifact_paths.append(orders_path)
    with orders_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "family", "coarse_middle_fine", "metric", "observed_order",
                "coarse_adjacent_error", "fine_adjacent_error", "monotonic",
            )
        )
        order_sets: list[
            tuple[str, str, Comparison, Comparison, float, float, float]
        ] = [
            (
                "time", f"dt5_dt2_dt1_N{finest_grid}",
                compare_runs(run_results[(finest_grid, 5.0)], run_results[(finest_grid, 2.0)]),
                compare_runs(run_results[(finest_grid, 2.0)], run_results[(finest_grid, 1.0)]),
                5.0, 2.0, 1.0,
            ),
            (
                "time", f"dt2_dt1_dt0.5_N{finest_grid}",
                assessment["time_coarse"], assessment["time_fine"], 2.0, 1.0, 0.5,
            ),
            (
                "space",
                f"N{finest_grid // 4}_N{finest_grid // 2}_N{finest_grid}_dt{_float(finest_dt)}",
                assessment["space_coarse"], assessment["space_fine"], 4.0, 2.0, 1.0,
            ),
            (
                "space_fine_gate",
                f"N{fine_grid_levels[0]}_N{fine_grid_levels[1]}_N{fine_grid_levels[2]}_dt{_float(selected_dt)}",
                space_coarse, space_fine, 4.0, 2.0, 1.0,
            ),
            (
                "time_fine_gate",
                f"dt{_float(fine_time_steps_s[0])}_dt{_float(fine_time_steps_s[1])}_dt{_float(fine_time_steps_s[2])}_N{selected_grid}",
                time_coarse,
                time_fine,
                fine_time_steps_s[0],
                fine_time_steps_s[1],
                fine_time_steps_s[2],
            ),
        ]
        for family, levels, coarse, fine, h1, h2, h3 in order_sets:
            for metric in FIELD_TOLERANCES:
                order = observed_order_three_levels(
                    coarse.errors[metric],
                    fine.errors[metric],
                    h1,
                    h2,
                    h3,
                    noise_floor=_metric_noise_floor(metric),
                )
                monotonic = _metric_monotonicity(
                    metric, coarse.errors[metric], fine.errors[metric]
                )
                writer.writerow(
                    (
                        family, levels, metric,
                        "" if order is None else _float(order),
                        _float(coarse.errors[metric]), _float(fine.errors[metric]),
                        "" if monotonic is None else monotonic,
                    )
                )
    convergence_path = output_directory / "q1_convergence_report.json"
    _write_json(convergence_path, convergence_report)
    artifact_paths.append(convergence_path)

    if not convergence_report["final_gate_pass"]:
        raise ConvergenceGateError(
            "Bounded Q1 fine gate failed; candidate diagnostics were written, but "
            "formal outputs and manifest were not created or overwritten."
        )
    final_commit = _git_head(repo_root)
    final_tracked_status = _git_tracked_status(repo_root)
    final_untracked_sources = tuple(
        path.relative_to(repo_root).as_posix()
        for path in source_paths
        if not _git_path_is_tracked(repo_root, path)
    )
    final_source_hashes = {
        path.relative_to(repo_root).as_posix(): _sha256(path) for path in source_paths
    }
    if (
        tracked_status
        or untracked_sources
        or final_commit != actual_commit
        or final_tracked_status
        or final_untracked_sources
        or final_source_hashes != source_file_hashes
    ):
        raise RuntimeError(
            "Formal outputs require committed participating sources and a clean "
            "stable tracked worktree for the entire run; allow_dirty permits "
            "candidate evidence only."
        )
    working_tree_status = _git_status(repo_root)

    formal_temperature_path = output_directory / "q1_formal_temperature.csv"
    formal_moisture_path = output_directory / "q1_formal_moisture.csv"
    formal_qoi_path = output_directory / "q1_formal_qoi.csv"
    formal_diagnostics_path = output_directory / "q1_formal_diagnostics.csv"
    artifact_paths.extend(
        (formal_temperature_path, formal_moisture_path, formal_qoi_path, formal_diagnostics_path)
    )
    formal_result = fine_results[(selected_grid, selected_dt)]
    with (
        formal_temperature_path.open("w", newline="", encoding="utf-8") as temperature_stream,
        formal_moisture_path.open("w", newline="", encoding="utf-8") as moisture_stream,
        formal_qoi_path.open("w", newline="", encoding="utf-8") as qoi_stream,
    ):
        temperature_writer = csv.writer(temperature_stream)
        moisture_writer = csv.writer(moisture_stream)
        qoi_writer = csv.writer(qoi_stream)
        field_header = ("time_s", *(f"r_{format(radius, 'g')}_cm" for radius in OUTPUT_RADII_CM))
        temperature_writer.writerow(field_header)
        moisture_writer.writerow(field_header)
        qoi_writer.writerow(
            (
                "time_s", "temperature_center_C", "temperature_surface_C",
                "temperature_weighted_mean_C", "moisture_center", "moisture_surface",
                "moisture_weighted_mean", "qT_W_m2", "qC_m_s",
            )
        )
        for snapshot in formal_result.samples:
            if snapshot.time_s == 0.0:
                continue
            temperature_writer.writerow((_float(snapshot.time_s), *map(_float, snapshot.temperatures_c)))
            moisture_writer.writerow((_float(snapshot.time_s), *map(_float, snapshot.moistures)))
            qoi_writer.writerow(
                (
                    _float(snapshot.time_s), _float(snapshot.temperature_center_c),
                    _float(snapshot.temperature_surface_c), _float(snapshot.temperature_weighted_mean_c),
                    _float(snapshot.moisture_center), _float(snapshot.moisture_surface),
                    _float(snapshot.moisture_weighted_mean), _float(snapshot.heat_flux_w_m2),
                    _float(snapshot.moisture_flux_m_s),
                )
            )
    candidate_diagnostics_path.replace(formal_diagnostics_path)

    formal_summary_path = output_directory / "q1_formal_summary.json"
    _write_json(formal_summary_path, asdict(formal_result.summary))
    artifact_paths.append(formal_summary_path)

    try:
        reproduction_output = output_directory.resolve().relative_to(
            repo_root.resolve()
        ).as_posix()
    except ValueError:
        reproduction_output = "<output-directory>"

    metadata = build_run_metadata(
        "q1",
        [environment.trace],
        {
            "method": "cell_centered_radial_FVM_Backward_Euler_Picard",
            "matrix_grid_counts": list(grid_counts),
            "matrix_time_steps_s": list(time_steps_s),
            "convergence_sample_interval_s": convergence_interval_s,
            "fine_gate_grid_levels": list(fine_grid_levels),
            "fine_gate_time_steps_s": list(fine_time_steps_s),
            "fine_gate_interval_s": fine_gate_interval_s,
            "convergence_output_radii_cm": list(OUTPUT_RADII_CM),
            "formal_grid_cells": selected_grid,
            "formal_dt_s": selected_dt,
            "formal_interval_s": formal_interval_s,
            "end_time_s": end_time_s,
            "solver_options": asdict(options),
            "streaming_policy": "retain requested projected snapshots only; never retain the ModelState history",
            "reproduction_command": [
                "python",
                "scripts/run_q1_phase4.py",
                "--output",
                reproduction_output,
                "--expected-commit",
                actual_commit,
            ],
        },
        created_at=datetime.now(timezone.utc),
    )
    metadata.update(
        {
            "source_git_commit": actual_commit,
            "expected_source_git_commit": expected_commit,
            "source_worktree": {
                "dirty": bool(working_tree_status),
                "tracked_status_porcelain": list(tracked_status),
                "untracked_source_files": list(untracked_sources),
                "status_porcelain": list(working_tree_status),
                "source_file_sha256": source_file_hashes,
            },
            "parameter_snapshot": parameter_snapshot("q1"),
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
    )
    metadata_path = output_directory / "q1_run_metadata.json"
    _write_json(metadata_path, metadata)
    artifact_paths.append(metadata_path)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": actual_commit,
        "source_worktree": {
            "dirty": bool(working_tree_status),
            "tracked_status_porcelain": list(tracked_status),
            "untracked_source_files": list(untracked_sources),
            "status_porcelain": list(working_tree_status),
            "source_file_sha256": source_file_hashes,
        },
        "input_files": {
            environment.trace.filename: {
                "filename": environment.trace.filename,
                "worksheet": environment.trace.worksheet,
                "sha256": environment.trace.sha256,
                "bytes": environment.trace.byte_size,
                "records": environment.trace.records,
            }
        },
        "artifacts": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifact_paths
        },
    }
    manifest_path = output_directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "output_directory": str(output_directory.resolve()),
        "selected_formal_run": _run_key(selected_grid, selected_dt),
        "formal_summary": asdict(formal_result.summary),
        "convergence": convergence_report,
        "manifest": manifest,
    }
