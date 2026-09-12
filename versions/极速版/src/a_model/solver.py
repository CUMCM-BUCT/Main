"""Backward-Euler/Picard driver for the unified V4 heat-moisture model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, ulp
from numbers import Integral
from typing import Protocol, Sequence

from .conventions import INITIAL_MOISTURE_DRY_BASIS, INITIAL_TEMPERATURE_C
from .fvm import (
    FrozenOperator,
    RadialGrid,
    assemble_backward_euler,
    distance_weighted_harmonic,
    freeze_operator,
    nonlinear_row_residuals,
    reconstruct_surface,
    solve_tridiagonal,
)
from .parameters import CaseParameters, get_case_parameters


class EnvironmentProvider(Protocol):
    def at(self, time_s: float) -> tuple[float, float]: ...


class RadiusProvider(Protocol):
    def at(self, time_s: float) -> float: ...


class StateError(ValueError):
    """A state is outside the model domain and must not be silently clipped."""


class StepFailure(RuntimeError):
    """A time step remained unacceptable after the configured retries."""


@dataclass(frozen=True)
class ConstantRadius:
    radius_m: float

    def __post_init__(self) -> None:
        if not isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError("Radius must be finite and positive.")

    def at(self, time_s: float) -> float:
        if not isfinite(time_s) or time_s < 0.0:
            raise ValueError("Time must be finite and non-negative.")
        return self.radius_m


@dataclass(frozen=True)
class ModelState:
    time_s: float
    temperatures_c: tuple[float, ...]
    moistures: tuple[float, ...]
    surface_moisture: float | None = None


@dataclass(frozen=True)
class BoundaryState:
    temperature_c: float
    moisture: float
    heat_transport: float
    moisture_transport: float


@dataclass(frozen=True)
class SolverOptions:
    max_picard_iterations: int = 40
    temperature_absolute_tolerance: float = 1.0e-6
    moisture_absolute_tolerance: float = 1.0e-8
    relative_tolerance: float = 1.0e-6
    residual_tolerance: float = 1.0e-8
    relaxation: float = 1.0
    fallback_relaxation: float = 0.5
    max_step_halvings: int = 12
    minimum_dt_s: float = 1.0e-6
    boundary_iterations: int = 50
    boundary_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        counts = (
            self.max_picard_iterations,
            self.max_step_halvings,
            self.boundary_iterations,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in counts):
            raise ValueError("Iteration counts must be integers, not booleans or real values.")
        if self.max_picard_iterations < 1 or self.max_step_halvings < 0:
            raise ValueError("Iteration counts must be positive and halving count non-negative.")
        if self.boundary_iterations < 1:
            raise ValueError("Boundary iteration count must be positive.")
        tolerances = (
            self.temperature_absolute_tolerance,
            self.moisture_absolute_tolerance,
            self.relative_tolerance,
            self.residual_tolerance,
            self.minimum_dt_s,
            self.boundary_tolerance,
        )
        if any(not isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("All tolerances and the minimum time step must be finite and positive.")
        for value in (self.relaxation, self.fallback_relaxation):
            if not isfinite(value) or value <= 0.0 or value > 1.0:
                raise ValueError("Relaxation factors must lie in (0, 1].")


@dataclass(frozen=True)
class StepDiagnostics:
    requested_dt_s: float
    accepted_dt_s: float
    radius_m: float
    environment_temperature_c: float
    environment_moisture: float
    picard_iterations: int
    relaxation: float
    rejected_attempts: int
    heat_linear_solves: int
    moisture_linear_solves: int
    maximum_temperature_update: float
    maximum_moisture_update: float
    maximum_temperature_relative_update: float
    maximum_moisture_relative_update: float
    maximum_temperature_scaled_residual: float
    maximum_moisture_scaled_residual: float
    boundary: BoundaryState


@dataclass(frozen=True)
class StepResult:
    state: ModelState
    diagnostics: StepDiagnostics


@dataclass(frozen=True)
class IntegrationResult:
    states: tuple[ModelState, ...]
    steps: tuple[StepDiagnostics, ...]


@dataclass(frozen=True)
class _AttemptResult:
    temperatures_c: tuple[float, ...]
    moistures: tuple[float, ...]
    iterations: int
    heat_linear_solves: int
    moisture_linear_solves: int
    temperature_update: float
    moisture_update: float
    temperature_relative_update: float
    moisture_relative_update: float
    temperature_residual: float
    moisture_residual: float
    boundary: BoundaryState


class _PicardFailure(RuntimeError):
    pass


class _BoundaryFailure(_PicardFailure):
    """A local Robin correction failed for the current trial state and forcing."""


class CoupledRadialSolver:
    """Solve Q1, Q2/Q3, or Q4 with the same conservative radial core."""

    def __init__(
        self,
        case: str | CaseParameters,
        grid: RadialGrid,
        environment: EnvironmentProvider,
        radius: RadiusProvider | float,
        options: SolverOptions | None = None,
    ) -> None:
        self.case = get_case_parameters(case) if isinstance(case, str) else case
        self.grid = grid
        self.environment = environment
        self.radius = ConstantRadius(radius) if isinstance(radius, (int, float)) else radius
        self.options = options or SolverOptions()

    def initial_state(
        self,
        *,
        time_s: float = 0.0,
        temperature_c: float = INITIAL_TEMPERATURE_C,
        moisture: float = INITIAL_MOISTURE_DRY_BASIS,
    ) -> ModelState:
        state = ModelState(
            time_s,
            (float(temperature_c),) * self.grid.cell_count,
            (float(moisture),) * self.grid.cell_count,
            float(moisture),
        )
        self._validate_state(state)
        return state

    def advance(self, previous: ModelState, requested_dt_s: float) -> StepResult:
        """Advance one accepted step, retrying with relaxation and halved ``dt``."""

        self._validate_state(previous)
        if not isfinite(requested_dt_s) or requested_dt_s <= 0.0:
            raise ValueError("Requested time step must be finite and positive.")

        attempted_dt = float(requested_dt_s)
        rejected = 0
        failures: list[str] = []
        relaxations = (self.options.relaxation,)
        if self.options.fallback_relaxation != self.options.relaxation:
            relaxations += (self.options.fallback_relaxation,)

        for halving in range(self.options.max_step_halvings + 1):
            if attempted_dt < self.options.minimum_dt_s:
                break
            new_time = previous.time_s + attempted_dt
            if new_time <= previous.time_s:
                raise ValueError(
                    "The requested time step does not advance representable time; "
                    "use a larger dt or a smaller time origin."
                )
            environment_temperature, environment_moisture = self.environment.at(new_time)
            radius_m = self.radius.at(new_time)
            self._validate_forcing(environment_temperature, environment_moisture, radius_m)
            for relaxation in relaxations:
                try:
                    attempt = self._solve_attempt(
                        previous,
                        attempted_dt,
                        environment_temperature,
                        environment_moisture,
                        radius_m,
                        relaxation,
                    )
                except (StateError, _PicardFailure) as exc:
                    rejected += 1
                    failures.append(
                        f"dt={attempted_dt:.12g}, omega={relaxation:.6g}: {exc}"
                    )
                    continue
                state = ModelState(
                    new_time,
                    attempt.temperatures_c,
                    attempt.moistures,
                    attempt.boundary.moisture,
                )
                self._validate_state(state)
                diagnostics = StepDiagnostics(
                    requested_dt_s,
                    attempted_dt,
                    radius_m,
                    environment_temperature,
                    environment_moisture,
                    attempt.iterations,
                    relaxation,
                    rejected,
                    attempt.heat_linear_solves,
                    attempt.moisture_linear_solves,
                    attempt.temperature_update,
                    attempt.moisture_update,
                    attempt.temperature_relative_update,
                    attempt.moisture_relative_update,
                    attempt.temperature_residual,
                    attempt.moisture_residual,
                    attempt.boundary,
                )
                return StepResult(state, diagnostics)
            if halving < self.options.max_step_halvings:
                attempted_dt *= 0.5

        detail = failures[-1] if failures else "minimum time step reached"
        raise StepFailure(
            f"Step from t={previous.time_s:.12g} s failed after {rejected} rejected attempts; "
            f"last failure: {detail}."
        )

    def integrate(
        self, end_time_s: float, nominal_dt_s: float, initial: ModelState | None = None
    ) -> IntegrationResult:
        """Integrate to an exact requested end time using accepted checkpoints.

        This convenience loop guarantees only ``end_time_s``.  Formal sampling
        times and event-search trial times must be supplied by the caller as
        successive end points, because an internally halved accepted step need
        not land on an otherwise desired sampling schedule.
        """

        state = initial or self.initial_state()
        self._validate_state(state)
        if not isfinite(end_time_s) or end_time_s < state.time_s:
            raise ValueError("End time must be finite and no earlier than the initial state.")
        if not isfinite(nominal_dt_s) or nominal_dt_s <= 0.0:
            raise ValueError("Nominal time step must be finite and positive.")
        states = [state]
        steps: list[StepDiagnostics] = []
        while state.time_s < end_time_s:
            remaining = end_time_s - state.time_s
            rounding_tolerance = 2.0 * max(
                ulp(max(abs(end_time_s), 1.0)),
                ulp(max(abs(state.time_s), 1.0)),
            )
            if remaining < self.options.minimum_dt_s:
                raise StepFailure(
                    "A positive interval smaller than minimum_dt_s remains; it is not "
                    "safe to skip without an accepted physical step."
                )
            planned_to_reach_end = remaining <= nominal_dt_s + rounding_tolerance
            planned_dt = min(nominal_dt_s, remaining)
            result = self.advance(state, planned_dt)
            if result.state.time_s <= state.time_s:
                raise StepFailure(
                    "Integration made no representable time progress before the requested end."
                )
            state = result.state
            if (
                planned_to_reach_end
                and result.diagnostics.accepted_dt_s == planned_dt
                and abs(state.time_s - end_time_s) <= rounding_tolerance
            ):
                state = ModelState(
                    end_time_s,
                    state.temperatures_c,
                    state.moistures,
                    state.surface_moisture,
                )
            states.append(state)
            steps.append(result.diagnostics)
        return IntegrationResult(tuple(states), tuple(steps))

    def boundary_state(
        self,
        temperature_c: float,
        moisture: float,
        environment_temperature_c: float,
        environment_moisture: float,
        radius_m: float,
        surface_moisture_hint: float | None = None,
    ) -> BoundaryState:
        """Solve the coupled half-cell Robin reconstruction at the outer face.

        For any positive half-cell coefficient, the reconstructed face value is
        a convex combination of the last-cell and environment values.  Hence the
        moisture fixed point is bracketed by those two values.  Temperature is
        evaluated directly for each moisture trial, reducing the coupled local
        problem to a safeguarded Newton continuation correction.  The optional
        surface-moisture hint is only its initial guess; direct callers must not
        interpret it as a request for the Euclidean-nearest global root.  During
        time integration the initial hint is the previous accepted face state,
        so a fold triggers a deterministic fallback only after local correction
        fails.
        """

        if moisture < 0.0:
            raise StateError("Negative moisture reached the boundary reconstruction.")
        if surface_moisture_hint is not None and (
            not isfinite(surface_moisture_hint) or surface_moisture_hint < 0.0
        ):
            raise ValueError("Surface-moisture continuation hint must be finite and non-negative.")
        half_distance = self.grid.surface_half_width
        cell_moisture_transport = self.case.diffusivity(moisture, temperature_c)

        def evaluate(surface_moisture: float) -> tuple[float, float, float, float]:
            cell_heat = self.case.conductivity(moisture)
            surface_heat = self.case.conductivity(surface_moisture)
            boundary_heat = distance_weighted_harmonic(
                cell_heat, surface_heat, half_distance, half_distance
            )
            surface_temperature = reconstruct_surface(
                self.grid,
                temperature_c,
                environment_temperature_c,
                radius_m,
                boundary_heat,
                self.case.heat_transfer_coefficient,
            )
            surface_moisture_transport = self.case.diffusivity(
                surface_moisture, surface_temperature
            )
            boundary_moisture = distance_weighted_harmonic(
                cell_moisture_transport,
                surface_moisture_transport,
                half_distance,
                half_distance,
            )
            mapped_moisture = reconstruct_surface(
                self.grid,
                moisture,
                environment_moisture,
                radius_m,
                boundary_moisture,
                self.case.mass_transfer_coefficient,
            )
            return (
                surface_moisture - mapped_moisture,
                surface_temperature,
                boundary_heat,
                boundary_moisture,
            )

        lower = min(moisture, environment_moisture)
        upper = max(moisture, environment_moisture)
        if lower == upper:
            _, surface_temperature, boundary_heat, boundary_moisture = evaluate(lower)
            return BoundaryState(
                surface_temperature, lower, boundary_heat, boundary_moisture
            )

        if cell_moisture_transport == 0.0:
            # With zero transport on the cell side the series resistance is
            # infinite: q=0 and the film-side face takes the environment value.
            final = evaluate(environment_moisture)
            return BoundaryState(final[1], environment_moisture, final[2], 0.0)

        scale = max(abs(moisture), abs(environment_moisture), 1.0)
        target = self.options.boundary_tolerance * scale
        lower_residual, _, _, _ = evaluate(lower)
        upper_residual, _, _, _ = evaluate(upper)
        bracket_guard = 64.0 * ulp(max(abs(lower), abs(upper), 1.0))
        if lower_residual > max(bracket_guard, target) or upper_residual < -max(
            bracket_guard, target
        ):
            raise _BoundaryFailure(
                "Robin moisture root was not bracketed by cell and environment values "
                f"(F_low={lower_residual:.6e}, F_high={upper_residual:.6e})"
            )

        span = upper - lower
        hint = moisture if surface_moisture_hint is None else surface_moisture_hint
        hint = min(max(hint, lower), upper)
        # Primary path: correct the previously accepted root locally.  A trust
        # region and residual-decreasing backtracking prevent Newton from
        # crossing a neighbouring branch when the Robin relation has 3 roots.
        trial = hint
        result = evaluate(trial)
        trust = max(span / 16.0, 16.0 * ulp(scale))
        for _ in range(self.options.boundary_iterations):
            if abs(result[0]) <= target:
                final = evaluate(trial)
                return BoundaryState(final[1], trial, final[2], final[3])
            derivative_step = max(span * 1.0e-6, 64.0 * ulp(max(abs(trial), 1.0)))
            derivative_left = max(lower, trial - derivative_step)
            derivative_right = min(upper, trial + derivative_step)
            if derivative_right == derivative_left:
                break
            derivative = (
                evaluate(derivative_right)[0] - evaluate(derivative_left)[0]
            ) / (derivative_right - derivative_left)
            if not isfinite(derivative) or abs(derivative) <= 1.0e-14:
                break
            newton_step = max(-trust, min(trust, -result[0] / derivative))
            accepted = False
            for _ in range(12):
                candidate = min(max(trial + newton_step, lower), upper)
                candidate_result = evaluate(candidate)
                if abs(candidate_result[0]) < abs(result[0]):
                    trial = candidate
                    result = candidate_result
                    trust = min(span, max(trust, 2.0 * abs(newton_step)))
                    accepted = True
                    break
                newton_step *= 0.5
            if not accepted:
                break

        # Fold/global fallback: progressively refine the admissible interval and
        # use the first detected surviving bracket in lower-to-upper order.  It
        # is a deterministic branch-loss policy, not a global-nearest-root API.
        # Endpoint roots participate in that same ordering, but only after the
        # accepted hint has had a chance to continue its local branch.
        if abs(lower_residual) <= target:
            final = evaluate(lower)
            return BoundaryState(final[1], lower, final[2], final[3])
        brackets: list[tuple[float, float, float, float]] = []
        for segment_count in (16, 64, 256, 1024):
            points = [lower + span * index / segment_count for index in range(segment_count + 1)]
            samples = [(point, evaluate(point)[0]) for point in points]
            brackets = [
                (left, right, left_residual, right_residual)
                for (left, left_residual), (right, right_residual) in zip(samples, samples[1:])
                if left_residual * right_residual < 0.0
            ]
            if brackets:
                break
        if not brackets:
            if abs(upper_residual) <= target:
                final = evaluate(upper)
                return BoundaryState(final[1], upper, final[2], final[3])
            raise _BoundaryFailure(
                "Robin continuation correction failed and adaptive fallback found no root bracket"
            )
        left, right, left_residual, right_residual = brackets[0]
        trial = 0.5 * (left + right)
        result = evaluate(trial)
        for _ in range(self.options.boundary_iterations):
            if abs(result[0]) <= target:
                break
            if left_residual * result[0] <= 0.0:
                right = trial
                right_residual = result[0]
            else:
                left = trial
                left_residual = result[0]
            trial = 0.5 * (left + right)
            result = evaluate(trial)
        else:
            raise _BoundaryFailure(
                "Safeguarded Robin branch solve did not converge after "
                f"{self.options.boundary_iterations} bisections; "
                f"bracket=[{left:.12g}, {right:.12g}], F={result[0]:.6e}"
            )
        surface_moisture = trial
        final = evaluate(surface_moisture)
        if abs(final[0]) > target:
            raise _BoundaryFailure(
                f"Robin root lost consistency at return (F={final[0]:.6e}, tol={target:.6e})"
            )
        return BoundaryState(final[1], surface_moisture, final[2], final[3])

    def _solve_attempt(
        self,
        previous: ModelState,
        dt_s: float,
        environment_temperature: float,
        environment_moisture: float,
        radius_m: float,
        relaxation: float,
    ) -> _AttemptResult:
        temperatures = previous.temperatures_c
        moistures = previous.moistures
        q1_temperature = self.case.case_id == "q1"
        heat_linear_solves = 0
        moisture_linear_solves = 0
        surface_moisture_hint = (
            moistures[-1]
            if previous.surface_moisture is None
            else previous.surface_moisture
        )

        if q1_temperature:
            boundary = self.boundary_state(
                temperatures[-1],
                moistures[-1],
                environment_temperature,
                environment_moisture,
                radius_m,
                surface_moisture_hint,
            )
            surface_moisture_hint = boundary.moisture
            heat_operator = self._heat_operator(moistures, radius_m, boundary)
            temperatures = solve_tridiagonal(
                assemble_backward_euler(
                    self.grid,
                    heat_operator,
                    previous.temperatures_c,
                    dt_s,
                    environment_temperature,
                )
            )
            heat_linear_solves = 1

        for iteration in range(1, self.options.max_picard_iterations + 1):
            old_temperature_iterate = temperatures
            old_moisture_iterate = moistures

            if not q1_temperature:
                boundary = self.boundary_state(
                    temperatures[-1],
                    moistures[-1],
                    environment_temperature,
                    environment_moisture,
                    radius_m,
                    surface_moisture_hint,
                )
                surface_moisture_hint = boundary.moisture
                heat_operator = self._heat_operator(moistures, radius_m, boundary)
                raw_temperatures = solve_tridiagonal(
                    assemble_backward_euler(
                        self.grid,
                        heat_operator,
                        previous.temperatures_c,
                        dt_s,
                        environment_temperature,
                    )
                )
                temperatures = self._relax(temperatures, raw_temperatures, relaxation)
                heat_linear_solves += 1

            boundary = self.boundary_state(
                temperatures[-1],
                moistures[-1],
                environment_temperature,
                environment_moisture,
                radius_m,
                surface_moisture_hint,
            )
            surface_moisture_hint = boundary.moisture
            moisture_operator = self._moisture_operator(
                moistures, temperatures, radius_m, boundary
            )
            raw_moistures = solve_tridiagonal(
                assemble_backward_euler(
                    self.grid,
                    moisture_operator,
                    previous.moistures,
                    dt_s,
                    environment_moisture,
                )
            )
            moistures = self._relax(moistures, raw_moistures, relaxation)
            moisture_linear_solves += 1
            self._validate_moistures(moistures)

            boundary = self.boundary_state(
                temperatures[-1],
                moistures[-1],
                environment_temperature,
                environment_moisture,
                radius_m,
                surface_moisture_hint,
            )
            surface_moisture_hint = boundary.moisture
            heat_operator = self._heat_operator(moistures, radius_m, boundary)
            moisture_operator = self._moisture_operator(
                moistures, temperatures, radius_m, boundary
            )
            _, scaled_heat = nonlinear_row_residuals(
                self.grid,
                heat_operator,
                previous.temperatures_c,
                temperatures,
                dt_s,
                environment_temperature,
            )
            _, scaled_moisture = nonlinear_row_residuals(
                self.grid,
                moisture_operator,
                previous.moistures,
                moistures,
                dt_s,
                environment_moisture,
            )
            temperature_update = (
                0.0
                if q1_temperature
                else self._maximum_difference(temperatures, old_temperature_iterate)
            )
            moisture_update = self._maximum_difference(moistures, old_moisture_iterate)
            temperature_relative = (
                0.0
                if q1_temperature
                else self._relative_difference(temperatures, old_temperature_iterate)
            )
            moisture_relative = self._relative_difference(moistures, old_moisture_iterate)
            temperature_residual = max(scaled_heat)
            moisture_residual = max(scaled_moisture)

            if (
                temperature_update <= self.options.temperature_absolute_tolerance
                and moisture_update <= self.options.moisture_absolute_tolerance
                and temperature_relative <= self.options.relative_tolerance
                and moisture_relative <= self.options.relative_tolerance
                and temperature_residual <= self.options.residual_tolerance
                and moisture_residual <= self.options.residual_tolerance
            ):
                return _AttemptResult(
                    temperatures,
                    moistures,
                    iteration,
                    heat_linear_solves,
                    moisture_linear_solves,
                    temperature_update,
                    moisture_update,
                    temperature_relative,
                    moisture_relative,
                    temperature_residual,
                    moisture_residual,
                    boundary,
                )
        raise _PicardFailure(
            f"Picard iteration exceeded {self.options.max_picard_iterations} iterations "
            f"(dT={temperature_update:.6e}, dC={moisture_update:.6e}, "
            f"relT={temperature_relative:.6e}, relC={moisture_relative:.6e}, "
            f"rT={temperature_residual:.6e}, rC={moisture_residual:.6e})"
        )

    def _heat_operator(
        self, moistures: Sequence[float], radius_m: float, boundary: BoundaryState
    ) -> FrozenOperator:
        return freeze_operator(
            self.grid,
            tuple(self.case.volumetric_heat_storage(value) for value in moistures),
            tuple(self.case.conductivity(value) for value in moistures),
            radius_m,
            boundary.heat_transport,
            self.case.heat_transfer_coefficient,
        )

    def _moisture_operator(
        self,
        moistures: Sequence[float],
        temperatures: Sequence[float],
        radius_m: float,
        boundary: BoundaryState,
    ) -> FrozenOperator:
        return freeze_operator(
            self.grid,
            (1.0,) * self.grid.cell_count,
            tuple(
                self.case.diffusivity(moisture, temperature)
                for moisture, temperature in zip(moistures, temperatures)
            ),
            radius_m,
            boundary.moisture_transport,
            self.case.mass_transfer_coefficient,
        )

    @staticmethod
    def _relax(
        old_values: Sequence[float], new_values: Sequence[float], relaxation: float
    ) -> tuple[float, ...]:
        return tuple(
            old + relaxation * (new - old) for old, new in zip(old_values, new_values)
        )

    @staticmethod
    def _maximum_difference(left: Sequence[float], right: Sequence[float]) -> float:
        return max(abs(a - b) for a, b in zip(left, right))

    @classmethod
    def _relative_difference(cls, left: Sequence[float], right: Sequence[float]) -> float:
        scale = max(max(abs(value) for value in left), 1.0e-12)
        return cls._maximum_difference(left, right) / scale

    def _validate_state(self, state: ModelState) -> None:
        if not isfinite(state.time_s) or state.time_s < 0.0:
            raise StateError("State time must be finite and non-negative.")
        if len(state.temperatures_c) != self.grid.cell_count:
            raise StateError("Temperature state size does not match the grid.")
        if len(state.moistures) != self.grid.cell_count:
            raise StateError("Moisture state size does not match the grid.")
        if any(not isfinite(value) or value <= -273.15 for value in state.temperatures_c):
            raise StateError("Temperatures must be finite and above absolute zero.")
        self._validate_moistures(state.moistures)
        if state.surface_moisture is not None and (
            not isfinite(state.surface_moisture) or state.surface_moisture < 0.0
        ):
            raise StateError("Surface-moisture continuation state must be finite and non-negative.")

    @staticmethod
    def _validate_moistures(moistures: Sequence[float]) -> None:
        if any(not isfinite(value) for value in moistures):
            raise StateError("Moisture state contains a non-finite value.")
        if any(value < 0.0 for value in moistures):
            minimum = min(moistures)
            raise StateError(
                f"Negative moisture {minimum:.12g} was produced; the step is rejected without clipping"
            )

    @staticmethod
    def _validate_forcing(
        environment_temperature: float, environment_moisture: float, radius_m: float
    ) -> None:
        if not isfinite(environment_temperature) or environment_temperature <= -273.15:
            raise ValueError("Environment temperature must be finite and above absolute zero.")
        if not isfinite(environment_moisture) or environment_moisture < 0.0:
            raise ValueError("Environment moisture must be finite and non-negative.")
        if not isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("Radius must be finite and positive.")
