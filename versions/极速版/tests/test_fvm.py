import math
import unittest

from a_model.fvm import (
    RadialGrid,
    assemble_backward_euler,
    distance_weighted_harmonic,
    freeze_operator,
    internal_conductances,
    nonlinear_row_residuals,
    reconstruct_center,
    reconstruct_surface,
    surface_conductance,
    solve_tridiagonal,
)
from a_model.parameters import get_case_parameters


class FiniteVolumeTests(unittest.TestCase):
    def test_grid_cell_count_requires_a_real_integer(self):
        for invalid in (True, False, 2.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RadialGrid(invalid)

    def test_radial_weights_are_exact_partition(self):
        grid = RadialGrid(8)
        self.assertAlmostEqual(sum(grid.weights), 1.0)
        self.assertAlmostEqual(grid.weights[0], grid.spacing**2)
        self.assertEqual(grid.faces[0], 0.0)
        self.assertEqual(grid.faces[-1], 1.0)

    def test_weighted_harmonic_mean_is_series_resistance(self):
        value = distance_weighted_harmonic(2.0, 8.0, 1.0, 3.0)
        self.assertAlmostEqual(value, 4.0 / (1.0 / 2.0 + 3.0 / 8.0))

    def test_zero_transport_is_the_series_resistance_limit_for_all_cases(self):
        grid = RadialGrid(2)
        for case_id in ("q1", "q2q3", "q4"):
            case = get_case_parameters(case_id)
            zero = case.diffusivity(0.0, 50.0)
            positive = case.diffusivity(0.15, 50.0)
            with self.subTest(case=case_id):
                self.assertEqual(zero, 0.0)
                self.assertEqual(distance_weighted_harmonic(zero, positive, 1.0, 1.0), 0.0)
                self.assertEqual(internal_conductances(grid, (zero, positive), 0.02), (0.0,))
                self.assertEqual(surface_conductance(grid, 0.02, zero, 8.0e-7), 0.0)
                self.assertEqual(
                    reconstruct_surface(grid, 0.4, 0.0, 0.02, zero, 8.0e-7),
                    0.0,
                )

    def test_center_reconstruction_matches_even_quadratic_cell_averages(self):
        # For u=A+B*xi^2, radial cell averages on the first two cells are
        # A+B*dx^2/2 and A+5*B*dx^2/2; equation (42) recovers A.
        grid = RadialGrid(10)
        constant = 3.25
        curvature = -0.7
        dx = grid.spacing
        first = constant + curvature * dx * dx / 2.0
        second = constant + 5.0 * curvature * dx * dx / 2.0
        self.assertAlmostEqual(reconstruct_center((first, second)), constant)

    def test_robin_reconstruction_balances_half_cell_and_film_flux(self):
        grid = RadialGrid(20)
        cell = 2.0
        environment = 0.1
        radius = 0.02
        transport = 4.0e-9
        transfer = 8.0e-7
        surface = reconstruct_surface(
            grid, cell, environment, radius, transport, transfer
        )
        delta = radius * (1.0 - grid.centers[-1])
        diffusive_flux = transport * (cell - surface) / delta
        film_flux = transfer * (surface - environment)
        self.assertAlmostEqual(diffusive_flux, film_flux, places=18)

    def test_constant_field_is_preserved_and_shared_faces_cancel(self):
        grid = RadialGrid(6)
        operator = freeze_operator(
            grid,
            (1.0,) * grid.cell_count,
            (2.0e-9,) * grid.cell_count,
            0.02,
            2.0e-9,
            8.0e-7,
        )
        previous = (1.7,) * grid.cell_count
        system = assemble_backward_euler(grid, operator, previous, 4.0, 1.7)
        solution = solve_tridiagonal(system)
        for value in solution:
            self.assertAlmostEqual(value, 1.7, places=14)
        self.assertEqual(system.upper, system.lower)
        raw, scaled = nonlinear_row_residuals(
            grid, operator, previous, solution, 4.0, 1.7
        )
        self.assertLess(max(abs(value) for value in raw), 1.0e-15)
        self.assertLess(max(scaled), 1.0e-8)

    def test_first_cell_constant_coefficient_matches_cell_centred_formula(self):
        grid = RadialGrid(10)
        coefficient = 3.0
        radius = 2.0
        operator = freeze_operator(
            grid,
            (5.0,) * grid.cell_count,
            (coefficient,) * grid.cell_count,
            radius,
            coefficient,
            7.0,
        )
        accumulation = 5.0 * grid.weights[0]
        normalized_diffusion = operator.internal_conductances[0] / accumulation
        expected = 2.0 * coefficient / (5.0 * radius * radius * grid.spacing**2)
        self.assertTrue(math.isclose(normalized_diffusion, expected, rel_tol=1.0e-14))


if __name__ == "__main__":
    unittest.main()
