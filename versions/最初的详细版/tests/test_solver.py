import unittest

from a_model.fvm import RadialGrid, distance_weighted_harmonic, reconstruct_surface
from a_model.solver import (
    ConstantRadius,
    CoupledRadialSolver,
    ModelState,
    SolverOptions,
    StateError,
)


class ConstantEnvironment:
    def __init__(self, temperature_c=50.0, moisture=0.05):
        self.temperature_c = temperature_c
        self.moisture = moisture
        self.queries = []

    def at(self, time_s):
        self.queries.append(time_s)
        return self.temperature_c, self.moisture


class RecordingRadius:
    def __init__(self):
        self.queries = []

    def at(self, time_s):
        self.queries.append(time_s)
        return 0.02 - 1.0e-8 * time_s


class JumpEnvironment:
    def __init__(self):
        self.queries = []

    def at(self, time_s):
        self.queries.append(time_s)
        if time_s >= 60.0:
            return 50.0, 0.05
        return 28.0, 2.55


class SolverTests(unittest.TestCase):
    def test_solver_option_counts_require_integers(self):
        fields = ("max_picard_iterations", "max_step_halvings", "boundary_iterations")
        for field in fields:
            for invalid in (True, 2.0, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    SolverOptions(**{field: invalid})

    def test_q1_solves_heat_once_but_iterates_nonlinear_moisture(self):
        environment = ConstantEnvironment()
        solver = CoupledRadialSolver("q1", RadialGrid(12), environment, 0.02)
        result = solver.advance(solver.initial_state(), 1.0)
        self.assertEqual(result.diagnostics.heat_linear_solves, 1)
        self.assertGreaterEqual(result.diagnostics.moisture_linear_solves, 1)
        self.assertGreater(result.state.temperatures_c[-1], 28.0)
        self.assertLess(result.state.moistures[-1], 2.55)
        self.assertLessEqual(
            result.diagnostics.maximum_moisture_scaled_residual,
            solver.options.residual_tolerance,
        )

    def test_q2q3_picard_converges_both_fields(self):
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(10), ConstantEnvironment(), ConstantRadius(0.02)
        )
        result = solver.advance(solver.initial_state(), 2.0)
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.heat_linear_solves, diagnostics.picard_iterations)
        self.assertEqual(diagnostics.moisture_linear_solves, diagnostics.picard_iterations)
        self.assertLessEqual(
            diagnostics.maximum_temperature_scaled_residual,
            solver.options.residual_tolerance,
        )
        self.assertLessEqual(
            diagnostics.maximum_moisture_scaled_residual,
            solver.options.residual_tolerance,
        )
        weights = solver.grid.weights
        moisture_balance = sum(
            weight * (new - old)
            for weight, new, old in zip(
                weights, result.state.moistures, solver.initial_state().moistures
            )
        ) + diagnostics.accepted_dt_s * 2.0 / diagnostics.radius_m * (
            solver.case.mass_transfer_coefficient
            * (diagnostics.boundary.moisture - diagnostics.environment_moisture)
        )
        heat_balance = sum(
            solver.case.volumetric_heat_storage(moisture)
            * weight
            * (new - old)
            for weight, moisture, new, old in zip(
                weights,
                result.state.moistures,
                result.state.temperatures_c,
                solver.initial_state().temperatures_c,
            )
        ) + diagnostics.accepted_dt_s * 2.0 / diagnostics.radius_m * (
            solver.case.heat_transfer_coefficient
            * (diagnostics.boundary.temperature_c - diagnostics.environment_temperature_c)
        )
        self.assertLess(abs(moisture_balance), 1.0e-13)
        self.assertLess(abs(heat_balance), 1.0e-5)

    def test_forcing_and_radius_are_evaluated_at_new_time_layer(self):
        environment = ConstantEnvironment()
        radius = RecordingRadius()
        solver = CoupledRadialSolver("q4", RadialGrid(8), environment, radius)
        initial = solver.initial_state(time_s=7.0)
        result = solver.advance(initial, 3.0)
        self.assertEqual(environment.queries, [10.0])
        self.assertEqual(radius.queries, [10.0])
        self.assertAlmostEqual(result.diagnostics.radius_m, 0.02 - 1.0e-7)

    def test_uniform_equilibrium_is_a_stationary_solution(self):
        environment = ConstantEnvironment(temperature_c=31.0, moisture=0.4)
        solver = CoupledRadialSolver("q4", RadialGrid(9), environment, 0.015)
        initial = solver.initial_state(temperature_c=31.0, moisture=0.4)
        result = solver.advance(initial, 50.0)
        for value in result.state.temperatures_c:
            self.assertAlmostEqual(value, 31.0, places=12)
        for value in result.state.moistures:
            self.assertAlmostEqual(value, 0.4, places=12)

    def test_integrate_lands_on_exact_end_time(self):
        solver = CoupledRadialSolver(
            "q1", RadialGrid(6), ConstantEnvironment(), 0.02
        )
        result = solver.integrate(2.5, 1.0)
        self.assertEqual(result.states[-1].time_s, 2.5)
        self.assertEqual([step.accepted_dt_s for step in result.steps], [1.0, 1.0, 0.5])

    def test_decimal_time_steps_do_not_leave_a_floating_point_tail(self):
        environment = ConstantEnvironment(temperature_c=28.0, moisture=2.55)
        solver = CoupledRadialSolver("q1", RadialGrid(4), environment, 0.02)
        result = solver.integrate(1.0, 0.1)
        self.assertEqual(result.states[-1].time_s, 1.0)
        self.assertEqual(len(result.steps), 10)

    def test_large_time_real_interval_advances_once_and_queries_endpoint(self):
        environment = ConstantEnvironment(temperature_c=50.0, moisture=0.05)
        solver = CoupledRadialSolver("q1", RadialGrid(4), environment, 0.02)
        initial = solver.initial_state(time_s=1.0e16)
        end_time = 1.0e16 + 2.0
        result = solver.integrate(end_time, 2.0, initial)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.states[-1].time_s, end_time)
        self.assertEqual(environment.queries, [end_time])
        self.assertGreater(result.states[-1].temperatures_c[-1], 28.0)

    def test_halved_large_time_step_is_not_relabelled_as_the_endpoint(self):
        environment = ConstantEnvironment(temperature_c=50.0, moisture=0.05)
        options = SolverOptions(max_picard_iterations=3, max_step_halvings=6)
        solver = CoupledRadialSolver(
            "q1", RadialGrid(20), environment, 0.02, options
        )
        initial_time = 2.0**55
        initial = solver.initial_state(time_s=initial_time)
        result = solver.integrate(initial_time + 32.0, 32.0, initial)
        self.assertEqual(
            [step.accepted_dt_s for step in result.steps],
            [16.0, 16.0],
        )
        self.assertEqual(
            [state.time_s - initial_time for state in result.states],
            [0.0, 16.0, 32.0],
        )

    def test_end_tolerance_does_not_hide_a_real_remaining_interval(self):
        environment = ConstantEnvironment(temperature_c=28.0, moisture=2.55)
        solver = CoupledRadialSolver("q1", RadialGrid(4), environment, 0.02)
        initial = solver.initial_state(time_s=1.0)
        result = solver.integrate(1.0 + 1.0e-5, 1.0, initial)
        self.assertEqual(result.states[-1].time_s, 1.0 + 1.0e-5)
        self.assertEqual(len(result.steps), 1)

    def test_unrepresentable_time_advance_is_rejected(self):
        environment = ConstantEnvironment(temperature_c=28.0, moisture=2.55)
        solver = CoupledRadialSolver("q1", RadialGrid(4), environment, 0.02)
        initial = solver.initial_state(time_s=1.0e16)
        with self.assertRaisesRegex(ValueError, "does not advance representable time"):
            solver.advance(initial, 1.0)
        with self.assertRaisesRegex(ValueError, "does not advance representable time"):
            solver.integrate(1.0e16 + 4.0, 1.0, initial)

    def test_loose_early_robin_acceptance_returns_coefficients_for_final_face(self):
        options = SolverOptions(boundary_iterations=1, boundary_tolerance=10.0)
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(20), ConstantEnvironment(), 0.02, options
        )
        boundary = solver.boundary_state(35.0, 1.7, 50.0, 0.05, 0.02)
        half_distance = solver.grid.spacing / 2.0
        expected_heat = distance_weighted_harmonic(
            solver.case.conductivity(1.7),
            solver.case.conductivity(boundary.moisture),
            half_distance,
            half_distance,
        )
        expected_moisture = distance_weighted_harmonic(
            solver.case.diffusivity(1.7, 35.0),
            solver.case.diffusivity(boundary.moisture, boundary.temperature_c),
            half_distance,
            half_distance,
        )
        self.assertAlmostEqual(boundary.heat_transport, expected_heat, places=15)
        self.assertAlmostEqual(boundary.moisture_transport, expected_moisture, places=20)
        mapped_temperature = reconstruct_surface(
            solver.grid,
            35.0,
            50.0,
            0.02,
            boundary.heat_transport,
            solver.case.heat_transfer_coefficient,
        )
        mapped_moisture = reconstruct_surface(
            solver.grid,
            1.7,
            0.05,
            0.02,
            boundary.moisture_transport,
            solver.case.mass_transfer_coefficient,
        )
        self.assertAlmostEqual(boundary.temperature_c, mapped_temperature, places=14)
        self.assertLessEqual(
            abs(boundary.moisture - mapped_moisture),
            options.boundary_tolerance * max(abs(boundary.moisture), 1.7, 0.05, 1.0),
        )

    def test_boundary_failure_retries_relaxation_then_halves_with_state_restored(self):
        environment = JumpEnvironment()
        options = SolverOptions(
            boundary_iterations=1,
            boundary_tolerance=1.0e-15,
            max_step_halvings=2,
        )
        solver = CoupledRadialSolver("q2q3", RadialGrid(20), environment, 0.02, options)
        initial = solver.initial_state()
        result = solver.advance(initial, 60.0)
        self.assertEqual(result.diagnostics.accepted_dt_s, 30.0)
        self.assertEqual(result.diagnostics.rejected_attempts, 2)
        self.assertEqual(environment.queries, [60.0, 30.0])
        self.assertEqual(initial, solver.initial_state())
        self.assertEqual(initial.surface_moisture, 2.55)

    def test_zero_boundary_moisture_limits_do_not_leak_discretization_errors(self):
        combinations = ((0.0, 0.0), (0.0, 0.05), (0.05, 0.0))
        for case_id in ("q1", "q2q3", "q4"):
            solver = CoupledRadialSolver(
                case_id, RadialGrid(20), ConstantEnvironment(), 0.02
            )
            for moisture, environment_moisture in combinations:
                with self.subTest(case=case_id, pair=(moisture, environment_moisture)):
                    boundary = solver.boundary_state(
                        50.0, moisture, 50.0, environment_moisture, 0.02
                    )
                    self.assertGreaterEqual(boundary.moisture_transport, 0.0)
                    if moisture == 0.0:
                        self.assertEqual(boundary.moisture, environment_moisture)
                        self.assertEqual(boundary.moisture_transport, 0.0)

    def test_robin_continuation_hint_selects_a_stable_multiple_root_branch(self):
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(20), ConstantEnvironment(), 0.02
        )
        low = solver.boundary_state(50.0, 0.1655, 50.0, 0.05, 0.02, 0.05)
        high = solver.boundary_state(50.0, 0.1655, 50.0, 0.05, 0.02, 0.1655)
        self.assertLess(low.moisture, 0.06)
        self.assertGreater(high.moisture, 0.10)

    def test_builtin_three_root_case_keeps_high_root_from_accepted_hint(self):
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(20), ConstantEnvironment(), 0.02
        )
        boundary = solver.boundary_state(
            50.0, 0.165132, 50.0, 0.05, 0.02, 0.104
        )
        self.assertGreater(boundary.moisture, 0.1037)
        self.assertLess(boundary.moisture, 0.1041)

    def test_arbitrary_hint_is_only_an_initial_guess_but_returns_a_valid_root(self):
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(20), ConstantEnvironment(), 0.02
        )
        boundary = solver.boundary_state(
            50.0, 0.1655, 50.0, 0.05, 0.02, 0.07
        )
        mapped = reconstruct_surface(
            solver.grid,
            0.1655,
            0.05,
            0.02,
            boundary.moisture_transport,
            solver.case.mass_transfer_coefficient,
        )
        self.assertLessEqual(abs(boundary.moisture - mapped), 1.0e-12)

    def test_q1_exact_endpoint_robin_root_advances_without_retry(self):
        cell_temperature = 34.331112206966445
        cell_moisture = 0.3393435818350331
        environment_temperature = 42.87484831755965
        environment_moisture = 0.020320876857062124
        surface_hint = 0.16761215596799217
        environment = ConstantEnvironment(
            temperature_c=environment_temperature,
            moisture=environment_moisture,
        )
        solver = CoupledRadialSolver("q1", RadialGrid(20), environment, 0.02)
        initial = ModelState(
            0.0,
            (cell_temperature,) * 20,
            (cell_moisture,) * 20,
            surface_hint,
        )
        result = solver.advance(initial, 1.0)
        self.assertEqual(result.diagnostics.rejected_attempts, 0)
        self.assertEqual(result.state.time_s, 1.0)
        self.assertEqual(result.state.surface_moisture, environment_moisture)
        boundary = result.diagnostics.boundary
        mapped = reconstruct_surface(
            solver.grid,
            result.state.moistures[-1],
            environment_moisture,
            result.diagnostics.radius_m,
            boundary.moisture_transport,
            solver.case.mass_transfer_coefficient,
        )
        self.assertLessEqual(
            abs(boundary.moisture - mapped),
            solver.options.boundary_tolerance,
        )

    def test_accepted_surface_hint_is_carried_across_multiple_steps(self):
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(20), ConstantEnvironment(), 0.02
        )
        state = ModelState(
            0.0,
            (50.0,) * 20,
            (0.1655,) * 20,
            0.104,
        )
        first = solver.advance(state, 0.1)
        second = solver.advance(first.state, 0.1)
        self.assertEqual(first.state.surface_moisture, first.diagnostics.boundary.moisture)
        self.assertEqual(second.state.surface_moisture, second.diagnostics.boundary.moisture)
        self.assertGreater(first.state.surface_moisture, 0.10)
        self.assertGreater(second.state.surface_moisture, 0.10)
        self.assertEqual(state.surface_moisture, 0.104)

    def test_robin_continuation_switches_safely_when_high_branch_disappears(self):
        solver = CoupledRadialSolver(
            "q2q3", RadialGrid(20), ConstantEnvironment(), 0.02
        )
        high = solver.boundary_state(50.0, 0.1655, 50.0, 0.05, 0.02, 0.1655)
        after_fold = solver.boundary_state(
            50.0, 0.1650, 50.0, 0.05, 0.02, high.moisture
        )
        self.assertGreater(high.moisture, 0.10)
        self.assertLess(after_fold.moisture, 0.06)

    def test_advance_accepts_zero_environment_moisture_for_all_cases(self):
        for case_id in ("q1", "q2q3", "q4"):
            solver = CoupledRadialSolver(
                case_id,
                RadialGrid(8),
                ConstantEnvironment(temperature_c=50.0, moisture=0.0),
                0.02,
            )
            with self.subTest(case=case_id):
                result = solver.advance(solver.initial_state(), 1.0)
                self.assertGreaterEqual(min(result.state.moistures), 0.0)

    def test_failed_picard_attempts_are_rejected_before_dt_is_halved(self):
        options = SolverOptions(max_picard_iterations=2, max_step_halvings=4)
        solver = CoupledRadialSolver(
            "q1", RadialGrid(8), ConstantEnvironment(), 0.02, options
        )
        result = solver.advance(solver.initial_state(), 1.0)
        self.assertEqual(result.diagnostics.accepted_dt_s, 0.5)
        self.assertEqual(result.diagnostics.rejected_attempts, 2)
        self.assertEqual(result.diagnostics.relaxation, 1.0)

    def test_negative_moisture_is_rejected_without_clipping(self):
        solver = CoupledRadialSolver(
            "q1", RadialGrid(4), ConstantEnvironment(), 0.02
        )
        state = ModelState(0.0, (28.0,) * 4, (2.55, 2.55, -1.0e-12, 2.55))
        with self.assertRaisesRegex(StateError, "Negative"):
            solver.advance(state, 1.0)


if __name__ == "__main__":
    unittest.main()
