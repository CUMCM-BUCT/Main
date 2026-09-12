"""Verification problems for the V4 radial finite-volume core.

The analytical profiles are integrated over each annular control volume
before comparison.  The FVM unknowns are cell averages, not centre samples.
"""

from __future__ import annotations

import math
import unittest

from a_model.fvm import (
    FrozenOperator,
    RadialGrid,
    assemble_backward_euler,
    freeze_operator,
    internal_conductances,
    reconstruct_center,
    reconstruct_surface,
    solve_tridiagonal,
    surface_conductance,
)
from a_model.solver import CoupledRadialSolver


class _EqualEnvironment:
    def __init__(self, temperature_c: float, moisture: float) -> None:
        self.temperature_c = temperature_c
        self.moisture = moisture

    def at(self, time_s: float) -> tuple[float, float]:
        return self.temperature_c, self.moisture


class _RecordingLinearRadius:
    def __init__(
        self, initial_radius_m: float, final_radius_m: float, duration_s: float
    ) -> None:
        self.initial_radius_m = initial_radius_m
        self.final_radius_m = final_radius_m
        self.duration_s = duration_s
        self.queries: list[float] = []

    def at(self, time_s: float) -> float:
        self.queries.append(time_s)
        fraction = min(time_s / self.duration_s, 1.0)
        return self.initial_radius_m + fraction * (
            self.final_radius_m - self.initial_radius_m
        )


def _bessel_j(order: int, value: float) -> float:
    """Return J0 or J1 from its convergent power series."""

    if order not in (0, 1):
        raise ValueError("Only J0 and J1 are needed by the verification series.")
    half_squared = 0.25 * value * value
    term = 1.0 if order == 0 else 0.5 * value
    total = term
    for index in range(1, 200):
        term *= -half_squared / (index * (index + order))
        total += term
        if abs(term) <= 2.0e-16 * max(1.0, abs(total)):
            return total
    raise ArithmeticError("Bessel power series did not converge.")


def _bisect_root(function, left: float, right: float) -> float:
    left_value = function(left)
    right_value = function(right)
    if left_value * right_value >= 0.0:
        raise ValueError("Root interval is not bracketed.")
    for _ in range(80):
        middle = 0.5 * (left + right)
        middle_value = function(middle)
        if left_value * middle_value <= 0.0:
            right = middle
            right_value = middle_value
        else:
            left = middle
            left_value = middle_value
    return 0.5 * (left + right)


def _j0_roots(count: int) -> tuple[float, ...]:
    roots = []
    for index in range(1, count + 1):
        centre = (index - 0.25) * math.pi
        roots.append(
            _bisect_root(
                lambda value: _bessel_j(0, value),
                centre - 0.25 * math.pi,
                centre + 0.25 * math.pi,
            )
        )
    return tuple(roots)


def _dirichlet_cell_averages(
    grid: RadialGrid, fourier_number: float, roots: tuple[float, ...]
) -> tuple[float, ...]:
    values = []
    for left, right in zip(grid.faces, grid.faces[1:]):
        weight = right * right - left * left
        value = 0.0
        for root in roots:
            annular_integral = (
                right * _bessel_j(1, root * right)
                - left * _bessel_j(1, root * left)
            )
            value += (
                4.0
                * annular_integral
                / (root * root * _bessel_j(1, root) * weight)
                * math.exp(-root * root * fourier_number)
            )
        values.append(value)
    return tuple(values)


def _dirichlet_mean(fourier_number: float, roots: tuple[float, ...]) -> float:
    return sum(
        4.0 / (root * root) * math.exp(-root * root * fourier_number)
        for root in roots
    )


def _advance_constant_diffusion(
    grid: RadialGrid,
    values: tuple[float, ...],
    delta_fourier: float,
    steps: int,
    boundary_conductance: float,
) -> tuple[float, ...]:
    operator = FrozenOperator(
        (1.0,) * grid.cell_count,
        internal_conductances(grid, (1.0,) * grid.cell_count, 1.0),
        boundary_conductance,
    )
    for _ in range(steps):
        values = solve_tridiagonal(
            assemble_backward_euler(
                grid, operator, values, delta_fourier, 0.0
            )
        )
    return values


def _observed_order(
    coarse: float, medium: float, fine: float
) -> tuple[float, float]:
    return math.log(coarse / medium, 2.0), math.log(medium / fine, 2.0)


def _positive_roots(function, count: int) -> tuple[float, ...]:
    """Locate successive positive simple roots without an external library."""

    roots = []
    step = math.pi / 32.0
    left = 0.0
    left_value = function(left)
    while len(roots) < count:
        right = left + step
        right_value = function(right)
        if left_value * right_value < 0.0:
            roots.append(_bisect_root(function, left, right))
        left = right
        left_value = right_value
        if left > (count + 1.0) * math.pi:
            raise ArithmeticError("Could not locate the requested Bessel roots.")
    return tuple(roots)


def _robin_roots(biot: float, count: int) -> tuple[float, ...]:
    return _positive_roots(
        lambda value: value * _bessel_j(1, value) - biot * _bessel_j(0, value),
        count,
    )


def _robin_coefficient(root: float) -> float:
    j0 = _bessel_j(0, root)
    j1 = _bessel_j(1, root)
    return 2.0 * j1 / (root * (j0 * j0 + j1 * j1))


def _robin_cell_averages(
    grid: RadialGrid, fourier_number: float, roots: tuple[float, ...]
) -> tuple[float, ...]:
    values = []
    for left, right in zip(grid.faces, grid.faces[1:]):
        weight = right * right - left * left
        value = 0.0
        for root in roots:
            annular_integral = (
                right * _bessel_j(1, root * right)
                - left * _bessel_j(1, root * left)
            )
            value += (
                _robin_coefficient(root)
                * 2.0
                * annular_integral
                / (root * weight)
                * math.exp(-root * root * fourier_number)
            )
        values.append(value)
    return tuple(values)


def _robin_surface(fourier_number: float, roots: tuple[float, ...]) -> float:
    return sum(
        _robin_coefficient(root)
        * _bessel_j(0, root)
        * math.exp(-root * root * fourier_number)
        for root in roots
    )


def _robin_mean(fourier_number: float, roots: tuple[float, ...]) -> float:
    return sum(
        _robin_coefficient(root)
        * 2.0
        * _bessel_j(1, root)
        / root
        * math.exp(-root * root * fourier_number)
        for root in roots
    )


class CylinderVerificationTests(unittest.TestCase):
    def test_dirichlet_cylinder_converges_to_cell_averaged_j0_series(self):
        target_fourier = 0.1
        roots = _j0_roots(7)
        maximum_errors = []
        mean_errors = []
        for cell_count in (20, 40, 80):
            grid = RadialGrid(cell_count)
            steps = cell_count * cell_count
            delta_fourier = target_fourier / steps
            numerical = _advance_constant_diffusion(
                grid,
                (1.0,) * cell_count,
                delta_fourier,
                steps,
                4.0 / grid.spacing,
            )
            exact = _dirichlet_cell_averages(grid, target_fourier, roots)
            maximum_errors.append(
                max(abs(actual - expected) for actual, expected in zip(numerical, exact))
            )
            numerical_mean = sum(
                weight * value for weight, value in zip(grid.weights, numerical)
            )
            mean_errors.append(abs(numerical_mean - _dirichlet_mean(target_fourier, roots)))

        maximum_orders = _observed_order(*maximum_errors)
        mean_orders = _observed_order(*mean_errors)
        self.assertLess(maximum_errors[-1], 5.0e-4)
        self.assertLess(mean_errors[-1], 1.0e-4)
        self.assertGreaterEqual(min(maximum_orders), 1.5)
        self.assertGreaterEqual(min(mean_orders), 1.5)

    def test_robin_cylinder_converges_for_cells_surface_and_mean(self):
        biot = 1.0
        target_fourier = 0.1
        roots = _robin_roots(biot, 7)
        cell_errors = []
        surface_errors = []
        mean_errors = []
        for cell_count in (20, 40, 80):
            grid = RadialGrid(cell_count)
            steps = cell_count * cell_count
            delta_fourier = target_fourier / steps
            boundary_conductance = surface_conductance(
                grid, 1.0, 1.0, biot
            )
            numerical = _advance_constant_diffusion(
                grid,
                (1.0,) * cell_count,
                delta_fourier,
                steps,
                boundary_conductance,
            )
            exact = _robin_cell_averages(grid, target_fourier, roots)
            cell_errors.append(
                max(abs(actual - expected) for actual, expected in zip(numerical, exact))
            )
            numerical_surface = reconstruct_surface(
                grid, numerical[-1], 0.0, 1.0, 1.0, biot
            )
            surface_errors.append(
                abs(numerical_surface - _robin_surface(target_fourier, roots))
            )
            numerical_mean = sum(
                weight * value for weight, value in zip(grid.weights, numerical)
            )
            mean_errors.append(abs(numerical_mean - _robin_mean(target_fourier, roots)))

        for errors in (cell_errors, surface_errors, mean_errors):
            self.assertLess(errors[-1], 5.0e-4)
            self.assertGreaterEqual(min(_observed_order(*errors)), 1.5)
        self.assertLess(mean_errors[-1], 1.0e-4)

    def test_zero_neumann_preserves_constant_field_and_weighted_total(self):
        grid = RadialGrid(40)
        constant = (1.23456789,) * grid.cell_count
        preserved = _advance_constant_diffusion(grid, constant, 0.01, 100, 0.0)
        constant_drift = max(
            abs(actual - expected) for actual, expected in zip(preserved, constant)
        )

        nonuniform = tuple(
            0.8 + 0.3 * centre * centre for centre in grid.centers
        )
        initial_total = sum(
            weight * value for weight, value in zip(grid.weights, nonuniform)
        )
        redistributed = _advance_constant_diffusion(
            grid, nonuniform, 0.01, 100, 0.0
        )
        final_total = sum(
            weight * value for weight, value in zip(grid.weights, redistributed)
        )
        self.assertLess(constant_drift, 1.0e-12)
        self.assertLess(abs(final_total - initial_total), 1.0e-12)

    def test_small_biot_limit_matches_lumped_capacitance_with_2_bi_fo(self):
        biot = 1.0e-3
        grid = RadialGrid(40)
        delta_fourier = 0.5
        delta_tau = 2.0 * biot * delta_fourier
        boundary_conductance = surface_conductance(grid, 1.0, 1.0, biot)
        operator = FrozenOperator(
            (1.0,) * grid.cell_count,
            internal_conductances(grid, (1.0,) * grid.cell_count, 1.0),
            boundary_conductance,
        )
        values = (1.0,) * grid.cell_count
        checkpoints = {0.2, 1.0, 2.0}
        mean_errors = []
        radial_ranges = []
        step_count = round(max(checkpoints) / delta_tau)
        for step in range(1, step_count + 1):
            values = solve_tridiagonal(
                assemble_backward_euler(
                    grid, operator, values, delta_fourier, 0.0
                )
            )
            tau = step * delta_tau
            if any(abs(tau - checkpoint) < 0.5 * delta_tau for checkpoint in checkpoints):
                mean = sum(
                    weight * value for weight, value in zip(grid.weights, values)
                )
                surface = reconstruct_surface(
                    grid, values[-1], 0.0, 1.0, 1.0, biot
                )
                centre = reconstruct_center(values)
                mean_errors.append(abs(mean - math.exp(-tau)))
                radial_ranges.append(max(centre, *values, surface) - min(centre, *values, surface))

        self.assertEqual(len(mean_errors), len(checkpoints))
        self.assertLess(max(mean_errors), 5.0e-3)
        self.assertLess(max(radial_ranges), 2.0e-3)

    def test_robin_solution_approaches_equilibrium_monotonically(self):
        biot = 1.0
        grid = RadialGrid(40)
        delta_fourier = 0.01
        operator = FrozenOperator(
            (1.0,) * grid.cell_count,
            internal_conductances(grid, (1.0,) * grid.cell_count, 1.0),
            surface_conductance(grid, 1.0, 1.0, biot),
        )
        values = (1.0,) * grid.cell_count
        previous_envelope = 1.0
        previous_mean = 1.0
        for _ in range(1000):
            values = solve_tridiagonal(
                assemble_backward_euler(
                    grid, operator, values, delta_fourier, 0.0
                )
            )
            surface = reconstruct_surface(
                grid, values[-1], 0.0, 1.0, 1.0, biot
            )
            reconstructed = (reconstruct_center(values), *values, surface)
            self.assertGreaterEqual(min(reconstructed), -1.0e-14)
            envelope = max(abs(value) for value in reconstructed)
            mean = sum(
                weight * value for weight, value in zip(grid.weights, values)
            )
            self.assertLessEqual(envelope, previous_envelope + 1.0e-14)
            self.assertLessEqual(abs(mean), previous_mean + 1.0e-14)
            previous_envelope = envelope
            previous_mean = abs(mean)

        self.assertLess(abs(previous_envelope), 1.0e-6)
        self.assertLess(abs(previous_mean), 1.0e-6)

    def test_q4_zero_flux_proportional_shrinkage_does_not_concentrate_dry_basis(self):
        grid = RadialGrid(40)
        initial_radius = 0.02
        final_radius = 0.01198
        duration_s = 72.0 * 3600.0
        step_s = 1800.0
        step_count = round(duration_s / step_s)
        values = (2.55,) * grid.cell_count
        initial_total = sum(
            weight * value for weight, value in zip(grid.weights, values)
        )
        maximum_drift = 0.0
        for step in range(1, step_count + 1):
            fraction = step / step_count
            radius = initial_radius + fraction * (final_radius - initial_radius)
            operator = freeze_operator(
                grid,
                (1.0,) * grid.cell_count,
                (0.0,) * grid.cell_count,
                radius,
                0.0,
                8.0e-7,
            )
            values = solve_tridiagonal(
                assemble_backward_euler(
                    grid, operator, values, step_s, 0.0
                )
            )
            maximum_drift = max(
                maximum_drift,
                max(abs(value - 2.55) for value in values),
            )

        final_total = sum(
            weight * value for weight, value in zip(grid.weights, values)
        )
        self.assertAlmostEqual(radius, final_radius)
        self.assertLess(maximum_drift, 1.0e-10)
        self.assertLess(abs(final_total - initial_total), 1.0e-10)

    def test_q4_coupled_solver_144_step_shrinkage_probe_stays_at_equilibrium(self):
        initial_temperature = 28.0
        initial_moisture = 2.55
        initial_radius = 0.02
        final_radius = 0.01198
        duration_s = 72.0 * 3600.0
        step_s = 1800.0
        radius = _RecordingLinearRadius(
            initial_radius, final_radius, duration_s
        )
        solver = CoupledRadialSolver(
            "q4",
            RadialGrid(12),
            _EqualEnvironment(initial_temperature, initial_moisture),
            radius,
        )

        result = solver.integrate(duration_s, step_s)

        expected_query_times = [step * step_s for step in range(1, 145)]
        self.assertEqual(len(result.steps), 144)
        self.assertEqual(radius.queries, expected_query_times)
        self.assertEqual(result.states[-1].time_s, duration_s)
        self.assertAlmostEqual(result.steps[-1].radius_m, final_radius)
        temperature_drift = max(
            abs(value - initial_temperature)
            for state in result.states
            for value in state.temperatures_c
        )
        moisture_drift = max(
            abs(value - initial_moisture)
            for state in result.states
            for value in state.moistures
        )
        self.assertLess(temperature_drift, 1.0e-10)
        self.assertLess(moisture_drift, 1.0e-10)
        self.assertTrue(
            all(
                abs(step.boundary.temperature_c - initial_temperature) < 1.0e-10
                and abs(step.boundary.moisture - initial_moisture) < 1.0e-10
                for step in result.steps
            )
        )
        final_mean = sum(
            weight * value
            for weight, value in zip(
                solver.grid.weights, result.states[-1].moistures
            )
        )
        spurious_concentration = initial_moisture * (
            initial_radius / final_radius
        ) ** 2
        self.assertAlmostEqual(final_mean, initial_moisture, places=10)
        self.assertGreater(abs(final_mean - spurious_concentration), 1.0)

    def test_moving_radius_zero_flux_follows_diffusion_clock_and_conserves_total(self):
        grid = RadialGrid(40)
        diffusivity = 1.0e-10
        step_s = 1800.0
        step_count = 144
        initial_radius = 0.02
        final_radius = 0.01198
        radii = tuple(
            initial_radius
            + step / step_count * (final_radius - initial_radius)
            for step in range(1, step_count + 1)
        )
        values = tuple(
            0.8
            + 0.6
            * 0.5
            * (left * left + right * right)
            for left, right in zip(grid.faces, grid.faces[1:])
        )
        moving = values
        fixed_radius_mutation = values
        clock_reference = values
        initial_total = sum(
            weight * value for weight, value in zip(grid.weights, values)
        )
        dimensionless_operator = freeze_operator(
            grid,
            (1.0,) * grid.cell_count,
            (1.0,) * grid.cell_count,
            1.0,
            0.0,
            1.0,
        )
        fixed_operator = freeze_operator(
            grid,
            (1.0,) * grid.cell_count,
            (diffusivity,) * grid.cell_count,
            initial_radius,
            0.0,
            1.0,
        )
        maximum_clock_difference = 0.0

        for radius in radii:
            moving_operator = freeze_operator(
                grid,
                (1.0,) * grid.cell_count,
                (diffusivity,) * grid.cell_count,
                radius,
                0.0,
                1.0,
            )
            moving = solve_tridiagonal(
                assemble_backward_euler(
                    grid, moving_operator, moving, step_s, 0.0
                )
            )
            delta_fourier = diffusivity * step_s / (radius * radius)
            clock_reference = solve_tridiagonal(
                assemble_backward_euler(
                    grid,
                    dimensionless_operator,
                    clock_reference,
                    delta_fourier,
                    0.0,
                )
            )
            maximum_clock_difference = max(
                maximum_clock_difference,
                max(
                    abs(actual - expected)
                    for actual, expected in zip(moving, clock_reference)
                ),
            )

            fixed_radius_mutation = solve_tridiagonal(
                assemble_backward_euler(
                    grid,
                    fixed_operator,
                    fixed_radius_mutation,
                    step_s,
                    0.0,
                )
            )

        geometry_clock = math.fsum(step_s / (radius * radius) for radius in radii)
        fixed_geometry_clock = step_count * step_s / (initial_radius * initial_radius)
        final_total = sum(
            weight * value for weight, value in zip(grid.weights, moving)
        )
        mutation_difference = max(
            abs(actual - mutated)
            for actual, mutated in zip(moving, fixed_radius_mutation)
        )
        self.assertGreater(geometry_clock, fixed_geometry_clock)
        self.assertLess(maximum_clock_difference, 1.0e-12)
        self.assertLess(abs(final_total - initial_total), 1.0e-12)
        self.assertGreater(mutation_difference, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
