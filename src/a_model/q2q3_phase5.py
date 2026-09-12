"""Streaming candidate Q2/Q3 simulations; convergence approval is separate."""

from dataclasses import dataclass, replace
from math import isfinite, ulp
from typing import Callable

from .fvm import maximum_reconstructed, reconstruct_center, sample_reconstructed
from .solver import CoupledRadialSolver, ModelState, StepFailure


@dataclass(frozen=True)
class EventResult:
    state: ModelState
    left_time_s: float
    right_time_s: float
    maximum_moisture: float
    iterations: int


def surface(solver: CoupledRadialSolver, state: ModelState) -> tuple[float, float]:
    if state.time_s == 0:
        return state.temperatures_c[-1], state.moistures[-1]
    boundary = solver.boundary_state(
        state.temperatures_c[-1], state.moistures[-1],
        *solver.environment.at(state.time_s), solver.radius.at(state.time_s),
        surface_moisture_hint=state.surface_moisture,
    )
    return boundary.temperature_c, boundary.moisture


def maximum(solver: CoupledRadialSolver, state: ModelState) -> float:
    value = maximum_reconstructed(solver.grid, state.moistures, surface(solver, state)[1])
    if not isfinite(value):
        raise StepFailure("Non-finite reconstructed event function.")
    return value


def maximum_location(solver, state):
    values = (
        reconstruct_center(state.moistures, solver.grid),
        *state.moistures,
        surface(solver, state)[1],
    )
    points = (0., *solver.grid.centers, 1.)
    index = max(range(len(values)), key=values.__getitem__)
    return {"xi": points[index], "radius_m": points[index] * solver.radius.at(state.time_s),
            "maximum_moisture": values[index]}


def advance_to(solver, state: ModelState, target: float, dt: float) -> ModelState:
    """Keep only the latest accepted checkpoint, including its Robin root hint."""
    while state.time_s < target:
        step_target = _step_target(state.time_s, dt, target)
        state = _accepted_state(state, solver.advance(state, step_target - state.time_s).state, step_target)
    return state


def _step_target(time_s, dt_s, limit):
    target = min(limit, time_s + dt_s)
    # Round only a few floating-point ulps onto an existing time constraint.
    # The solver still integrates the whole requested interval; no state is interpolated.
    return limit if limit - target <= 8 * max(ulp(limit), ulp(target)) else target


def _accepted_state(previous, state, target):
    if abs(state.time_s - target) <= 8 * max(ulp(state.time_s), ulp(target)):
        state = replace(state, time_s=target)
    if not previous.time_s < state.time_s <= target:
        raise StepFailure("Solver failed to advance within requested interval.")
    return state


def locate_event(solver, left: ModelState, right: ModelState, *, dt_s: float,
                 threshold: float = .15, tolerance_s: float = .01,
                 measure: Callable = maximum) -> EventResult:
    """Bisect reintegrations from a fixed accepted left checkpoint.

    This controls time localization, not PDE discretization error. All trials
    start from the same state and surface hint; rejected trial paths are discarded.
    """
    if not isfinite(tolerance_s) or tolerance_s <= 0 or not isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Event tolerance and dt must be finite and positive.")
    if not left.time_s < right.time_s or not measure(solver, left) >= threshold > measure(solver, right):
        raise ValueError("Event requires an above/below threshold time bracket.")
    lo, hi, accepted = left.time_s, right.time_s, right
    iterations = 0
    while hi - lo > tolerance_s:
        midpoint = (lo + hi) / 2
        if midpoint in (lo, hi):
            raise StepFailure("Event tolerance is below representable time resolution.")
        trial = advance_to(solver, left, midpoint, dt_s)
        if measure(solver, trial) < threshold:
            hi, accepted = midpoint, trial
        else:
            lo = midpoint
        iterations += 1
    return EventResult(accepted, lo, hi, measure(solver, accepted), iterations)


def run_streaming(solver, *, end_time_s: float, dt_s: float,
                  sample_interval_s: float | None = None,
                  on_sample: Callable | None = None, initial: ModelState | None = None,
                  threshold: float = .15, event_tolerance_s: float = .01,
                  measure: Callable = maximum, time_step: Callable | None = None) -> dict:
    """Stop at the first accepted-step crossing; sample by actual integration.

    Scan mode has no output time constraints. Export mode lands on the sample
    grid. A final event row is always emitted exactly once. Crossings between
    accepted endpoints require temporal refinement to exclude missed excursions.
    """
    state = initial or solver.initial_state()
    if not isfinite(end_time_s) or end_time_s < state.time_s:
        raise ValueError("End time must be finite and no earlier than initial time.")
    if not isfinite(dt_s) or dt_s <= 0 or not isfinite(event_tolerance_s) or event_tolerance_s <= 0:
        raise ValueError("dt and event tolerance must be finite and positive.")
    if not isfinite(threshold) or threshold < 0:
        raise ValueError("Threshold must be finite and non-negative.")
    if sample_interval_s is not None and (not isfinite(sample_interval_s) or sample_interval_s <= 0):
        raise ValueError("Sampling interval must be finite and positive.")
    breakpoints = tuple(getattr(time_step, "breakpoints_s", ())) if time_step is not None else ()
    if any(not isfinite(point) or point <= 0 for point in breakpoints):
        raise ValueError("Time-step breakpoints must be finite and positive.")
    if tuple(sorted(set(breakpoints))) != breakpoints:
        raise ValueError("Time-step breakpoints must be strictly increasing.")
    last_emitted = None
    def emit(item, terminal):
        nonlocal last_emitted
        if on_sample is not None and last_emitted != item.time_s:
            on_sample(item, terminal)
            last_emitted = item.time_s
    if measure(solver, state) < threshold:
        emit(state, True)
        return {"status": "event", "event": EventResult(state, state.time_s, state.time_s, measure(solver, state), 0), "steps": 0}
    emit(state, False)
    index = int(state.time_s / sample_interval_s) + 1 if sample_interval_s else 0
    steps = 0
    diagnostics = {"rejected_attempts": 0, "max_heat_residual": 0., "max_moisture_residual": 0.}
    while state.time_s < end_time_s:
        limit = end_time_s
        if state.time_s < 14400:
            limit = min(limit, (int(state.time_s / 60) + 1) * 60.)
        if sample_interval_s:
            limit = min(limit, index * sample_interval_s)
        for point in breakpoints:
            if state.time_s < point:
                limit = min(limit, point)
                break
        active_dt_s = time_step(state.time_s) if time_step is not None else dt_s
        if not isfinite(active_dt_s) or active_dt_s <= 0:
            raise ValueError("Selected dt must be finite and positive.")
        target = _step_target(state.time_s, active_dt_s, limit)
        step = solver.advance(state, target - state.time_s)
        right = _accepted_state(state, step.state, target)
        if hasattr(step, "diagnostics"):
            diagnostics["rejected_attempts"] += step.diagnostics.rejected_attempts
            diagnostics["max_heat_residual"] = max(diagnostics["max_heat_residual"], step.diagnostics.maximum_temperature_scaled_residual)
            diagnostics["max_moisture_residual"] = max(diagnostics["max_moisture_residual"], step.diagnostics.maximum_moisture_scaled_residual)
        steps += 1
        if measure(solver, right) < threshold:
            event = locate_event(solver, state, right, dt_s=active_dt_s, threshold=threshold,
                                 tolerance_s=event_tolerance_s, measure=measure)
            emit(event.state, True)
            return {"status": "event", "event": event, "steps": steps, "diagnostics": diagnostics}
        state = right
        if sample_interval_s and state.time_s == index * sample_interval_s:
            emit(state, False)
            index += 1
    emit(state, False)
    return {"status": "horizon_reached_without_event", "state": state, "steps": steps, "diagnostics": diagnostics}


def project(solver, state: ModelState) -> tuple[tuple, tuple]:
    temperature, moisture = surface(solver, state)
    return tuple(tuple(sample_reconstructed(solver.grid, cells, i / 20, face)
                       for i in range(21))
                 for cells, face in ((state.temperatures_c, temperature), (state.moistures, moisture)))
