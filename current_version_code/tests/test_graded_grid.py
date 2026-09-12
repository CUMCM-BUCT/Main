"""Geometry and diffusion checks for the conservative surface-graded grid."""

import math
import unittest

from a_model.fvm import (
    FrozenOperator,
    RadialGrid,
    assemble_backward_euler,
    distance_weighted_harmonic,
    internal_conductances,
    reconstruct_center,
    reconstruct_surface,
    solve_tridiagonal,
    surface_conductance,
)
from a_model.solver import CoupledRadialSolver


class ConstantEnvironment:
    def at(self, time_s):
        return 50.0, 0.05


def bessel_j(order, value):
    half_squared = 0.25 * value * value
    term = 1.0 if order == 0 else 0.5 * value
    total = term
    for index in range(1, 200):
        term *= -half_squared / (index * (index + order))
        total += term
        if abs(term) <= 2.0e-16 * max(1.0, abs(total)):
            return total
    raise ArithmeticError("Bessel series did not converge.")


def bisect(function, left, right):
    left_value = function(left)
    for _ in range(80):
        middle = 0.5 * (left + right)
        middle_value = function(middle)
        if left_value * middle_value <= 0.0:
            right = middle
        else:
            left, left_value = middle, middle_value
    return 0.5 * (left + right)


def robin_roots(biot, count):
    roots = []
    step = math.pi / 32.0
    left = 0.0
    left_value = -biot
    while len(roots) < count:
        right = left + step
        right_value = right * bessel_j(1, right) - biot * bessel_j(0, right)
        if left_value * right_value < 0.0:
            roots.append(
                bisect(
                    lambda value: value * bessel_j(1, value)
                    - biot * bessel_j(0, value),
                    left,
                    right,
                )
            )
        left, left_value = right, right_value
    return tuple(roots)


def dirichlet_roots(count):
    roots = []
    step = math.pi / 32.0
    left = 0.0
    left_value = 1.0
    while len(roots) < count:
        right = left + step
        right_value = bessel_j(0, right)
        if left_value * right_value < 0.0:
            roots.append(bisect(lambda value: bessel_j(0, value), left, right))
        left, left_value = right, right_value
    return tuple(roots)


def robin_coefficient(root):
    j0, j1 = bessel_j(0, root), bessel_j(1, root)
    return 2.0 * j1 / (root * (j0 * j0 + j1 * j1))


def robin_cell_averages(grid, fourier_number, roots):
    result = []
    for left, right in zip(grid.faces, grid.faces[1:]):
        value = 0.0
        for root in roots:
            integral = right * bessel_j(1, root * right) - left * bessel_j(
                1, root * left
            )
            value += (
                robin_coefficient(root)
                * 2.0
                * integral
                / (root * (right * right - left * left))
                * math.exp(-root * root * fourier_number)
            )
        result.append(value)
    return tuple(result)


def robin_point(xi, fourier_number, roots):
    return math.fsum(
        robin_coefficient(root)
        * bessel_j(0, root * xi)
        * math.exp(-root * root * fourier_number)
        for root in roots
    )


def robin_mean(fourier_number, roots):
    return math.fsum(
        robin_coefficient(root)
        * 2.0
        * bessel_j(1, root)
        / root
        * math.exp(-root * root * fourier_number)
        for root in roots
    )


def dirichlet_cell_averages(grid, fourier_number, roots):
    result = []
    for left, right in zip(grid.faces, grid.faces[1:]):
        result.append(math.fsum(
            (2.0 / (root * bessel_j(1, root)))
            * 2.0
            * (right * bessel_j(1, root * right) - left * bessel_j(1, root * left))
            / (root * (right * right - left * left))
            * math.exp(-root * root * fourier_number)
            for root in roots
        ))
    return tuple(result)


def advance_diffusion(grid, delta_fourier, steps, biot=1.0):
    operator = FrozenOperator(
        (1.0,) * grid.cell_count,
        internal_conductances(grid, (1.0,) * grid.cell_count, 1.0),
        surface_conductance(grid, 1.0, 1.0, biot),
    )
    values = (1.0,) * grid.cell_count
    for _ in range(steps):
        values = solve_tridiagonal(
            assemble_backward_euler(grid, operator, values, delta_fourier, 0.0)
        )
    return values


def advance_dirichlet_diffusion(grid, delta_fourier, steps):
    operator = FrozenOperator(
        (1.0,) * grid.cell_count,
        internal_conductances(grid, (1.0,) * grid.cell_count, 1.0),
        2.0 / grid.surface_half_width,
    )
    values = (1.0,) * grid.cell_count
    for _ in range(steps):
        values = solve_tridiagonal(
            assemble_backward_euler(grid, operator, values, delta_fourier, 0.0)
        )
    return values


class SurfaceGradedGridTests(unittest.TestCase):
    def test_uniform_grid_arithmetic_and_conductances_are_unchanged(self):
        grid = RadialGrid(4)
        self.assertEqual(grid.faces, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(grid.centers, (0.125, 0.375, 0.625, 0.875))
        self.assertEqual(grid.surface_half_width, grid.spacing / 2.0)
        expected = tuple(
            2.0 * (index * grid.spacing) * 3.0
            / (4.0 * grid.spacing)
            for index in range(1, 4)
        )
        self.assertEqual(internal_conductances(grid, (3.0,) * 4, 2.0), expected)
        values = (2.0, 4.0, 6.0, 8.0)
        self.assertEqual(reconstruct_center(values, grid), reconstruct_center(values))

    def test_surface_graded_geometry_has_conservative_positive_weights(self):
        grid = RadialGrid.surface_graded(4, 2.0)
        self.assertEqual(grid.faces, (0.0, 0.4375, 0.75, 0.9375, 1.0))
        self.assertEqual(grid.centers, (0.21875, 0.59375, 0.84375, 0.96875))
        self.assertTrue(all(right > left for left, right in zip(grid.faces, grid.faces[1:])))
        self.assertTrue(all(weight > 0.0 for weight in grid.weights))
        self.assertAlmostEqual(math.fsum(grid.weights), 1.0, places=15)
        self.assertEqual(grid.surface_half_width, 0.03125)

    def test_exponent_that_collapses_floating_point_cells_is_rejected(self):
        with self.assertRaisesRegex(Exception, "collapses a cell"):
            RadialGrid.surface_graded(1000, 8.0)

    def test_internal_conductance_uses_true_two_sided_distances(self):
        grid = RadialGrid.surface_graded(4, 2.0)
        diffusivities = (1.0, 2.0, 4.0, 8.0)
        conductances = internal_conductances(grid, diffusivities, 2.0)
        for index, actual in enumerate(conductances, start=1):
            face = grid.faces[index]
            left_distance = face - grid.centers[index - 1]
            right_distance = grid.centers[index] - face
            harmonic = distance_weighted_harmonic(
                diffusivities[index - 1],
                diffusivities[index],
                left_distance,
                right_distance,
            )
            expected = 2.0 * face * harmonic / (
                4.0 * (grid.centers[index] - grid.centers[index - 1])
            )
            self.assertAlmostEqual(actual, expected, places=15)

    def test_nonuniform_center_reconstruction_recovers_even_quadratic(self):
        grid = RadialGrid.surface_graded(12, 2.0)
        constant, curvature = 0.2, 0.7
        averages = tuple(
            constant + curvature * (left * left + right * right) / 2.0
            for left, right in zip(grid.faces, grid.faces[1:])
        )
        center = reconstruct_center(averages, grid)
        self.assertAlmostEqual(center, constant, places=15)
        self.assertGreaterEqual(center, 0.0)

    def test_constant_field_and_weighted_total_are_preserved_with_zero_flux(self):
        grid = RadialGrid.surface_graded(40, 2.0)
        operator = FrozenOperator(
            (1.0,) * grid.cell_count,
            internal_conductances(grid, (1.0,) * grid.cell_count, 1.0),
            0.0,
        )
        values = (1.23456789,) * grid.cell_count
        initial_total = math.fsum(w * value for w, value in zip(grid.weights, values))
        for _ in range(20):
            values = solve_tridiagonal(
                assemble_backward_euler(grid, operator, values, 0.01, 0.0)
            )
        final_total = math.fsum(w * value for w, value in zip(grid.weights, values))
        self.assertLess(max(abs(value - 1.23456789) for value in values), 2.0e-12)
        self.assertLess(abs(final_total - initial_total), 2.0e-12)

    def test_graded_grid_converges_to_robin_cylinder_solution(self):
        target_fourier = 0.1
        roots = robin_roots(1.0, 7)
        cell_errors, center_errors, surface_errors, mean_errors = [], [], [], []
        for cell_count in (20, 40, 80):
            grid = RadialGrid.surface_graded(cell_count, 2.0)
            steps = cell_count * cell_count
            numerical = advance_diffusion(
                grid, target_fourier / steps, steps
            )
            exact = robin_cell_averages(grid, target_fourier, roots)
            cell_errors.append(
                max(abs(actual - expected) for actual, expected in zip(numerical, exact))
            )
            center_errors.append(
                abs(reconstruct_center(numerical, grid) - robin_point(0.0, target_fourier, roots))
            )
            numerical_surface = reconstruct_surface(
                grid, numerical[-1], 0.0, 1.0, 1.0, 1.0
            )
            surface_errors.append(
                abs(numerical_surface - robin_point(1.0, target_fourier, roots))
            )
            numerical_mean = math.fsum(
                weight * value for weight, value in zip(grid.weights, numerical)
            )
            mean_errors.append(abs(numerical_mean - robin_mean(target_fourier, roots)))
        for errors in (cell_errors, center_errors, surface_errors, mean_errors):
            orders = (
                math.log(errors[0] / errors[1], 2.0),
                math.log(errors[1] / errors[2], 2.0),
            )
            self.assertLess(errors[-1], 5.0e-4)
            self.assertGreaterEqual(min(orders), 1.5)

    def test_graded_grid_converges_to_dirichlet_cylinder_solution(self):
        target_fourier = 0.1
        roots = dirichlet_roots(7)
        errors = []
        for cell_count in (20, 40, 80):
            grid = RadialGrid.surface_graded(cell_count, 2.0)
            steps = cell_count * cell_count
            numerical = advance_dirichlet_diffusion(
                grid, target_fourier / steps, steps
            )
            exact = dirichlet_cell_averages(grid, target_fourier, roots)
            errors.append(max(abs(actual - expected)
                              for actual, expected in zip(numerical, exact)))
        orders = (
            math.log(errors[0] / errors[1], 2.0),
            math.log(errors[1] / errors[2], 2.0),
        )
        self.assertLess(errors[-1], 5.0e-4)
        self.assertGreaterEqual(min(orders), 1.5)

    def test_solver_boundary_state_obeys_graded_outer_half_cell(self):
        grid = RadialGrid.surface_graded(20, 2.0)
        solver = CoupledRadialSolver("q2q3", grid, ConstantEnvironment(), 0.02)
        boundary = solver.boundary_state(35.0, 1.7, 50.0, 0.05, 0.02, 1.7)
        expected_temperature = reconstruct_surface(
            grid,
            35.0,
            50.0,
            0.02,
            boundary.heat_transport,
            solver.case.heat_transfer_coefficient,
        )
        expected_moisture = reconstruct_surface(
            grid,
            1.7,
            0.05,
            0.02,
            boundary.moisture_transport,
            solver.case.mass_transfer_coefficient,
        )
        self.assertAlmostEqual(boundary.temperature_c, expected_temperature, places=14)
        self.assertAlmostEqual(boundary.moisture, expected_moisture, places=12)

    def test_outer_half_cell_controls_robin_resistance(self):
        uniform = RadialGrid(20)
        graded = RadialGrid.surface_graded(20, 2.0)
        self.assertAlmostEqual(
            graded.surface_half_width,
            uniform.surface_half_width / 20.0,
            places=15,
        )
        cell, environment = 2.0, 0.1
        radius, transport, transfer = 0.02, 4.0e-9, 8.0e-7
        surface = reconstruct_surface(
            graded, cell, environment, radius, transport, transfer
        )
        delta = radius * graded.surface_half_width
        self.assertAlmostEqual(
            transport * (cell - surface) / delta,
            transfer * (surface - environment),
            places=17,
        )


if __name__ == "__main__":
    unittest.main()
