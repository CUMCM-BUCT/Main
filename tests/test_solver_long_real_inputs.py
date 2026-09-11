"""Moderate-cost regressions for the real forcing/radius histories."""

import unittest

from a_model import load_environment, load_radius
from a_model.fvm import RadialGrid, reconstruct_center
from a_model.solver import CoupledRadialSolver, SolverOptions


class RealInputLongRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = load_environment()
        cls.radius = load_radius()

    def _assert_healthy_run(self, case, end_time_s, radius):
        options = SolverOptions()
        solver = CoupledRadialSolver(
            case, RadialGrid(20), self.environment, radius, options
        )
        result = solver.integrate(end_time_s, 60.0)
        self.assertEqual(result.states[-1].time_s, end_time_s)
        self.assertTrue(result.steps)
        self.assertGreaterEqual(
            min(min(state.moistures) for state in result.states), 0.0
        )
        self.assertLessEqual(
            max(step.maximum_temperature_scaled_residual for step in result.steps),
            options.residual_tolerance,
        )
        self.assertLessEqual(
            max(step.maximum_moisture_scaled_residual for step in result.steps),
            options.residual_tolerance,
        )
        self.assertTrue(
            all(
                state.surface_moisture == step.boundary.moisture
                for state, step in zip(result.states[1:], result.steps)
            )
        )
        return result

    def test_q2q3_reaches_a_coarse_first_below_target_checkpoint(self):
        result = self._assert_healthy_run("q2q3", 856_020.0, 0.02)
        final = result.states[-1]
        maximum = max(
            reconstruct_center(final.moistures),
            *final.moistures,
            final.surface_moisture,
        )
        previous = result.states[-2]
        previous_maximum = max(
            reconstruct_center(previous.moistures),
            *previous.moistures,
            previous.surface_moisture,
        )
        self.assertLess(maximum, 0.15)
        self.assertGreaterEqual(previous_maximum, 0.15)
        self.assertEqual(min(step.accepted_dt_s for step in result.steps), 60.0)
        post_fold_surface = [
            state.surface_moisture
            for state in result.states
            if 53_340.0 <= state.time_s <= 60_000.0
        ]
        jumps = [
            (index, abs(right - left))
            for index, (left, right) in enumerate(
                zip(post_fold_surface, post_fold_surface[1:])
            )
            if abs(right - left) >= 5.0e-4
        ]
        self.assertEqual(len(jumps), 1)
        self.assertLess(
            max(
                abs(right - left)
                for left, right in zip(
                    post_fold_surface[jumps[0][0] + 1 :],
                    post_fold_surface[jumps[0][0] + 2 :],
                )
            ),
            5.0e-4,
        )

    def test_q4_remains_stable_through_full_72_hour_radius_record(self):
        self._assert_healthy_run("q4", 259_200.0, self.radius)


if __name__ == "__main__":
    unittest.main()
