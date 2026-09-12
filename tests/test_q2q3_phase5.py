import unittest
from types import SimpleNamespace

from a_model.fvm import RadialGrid
from a_model.q2q3_phase5 import locate_event, maximum, run_streaming
from a_model.solver import CoupledRadialSolver, ModelState, StepFailure
from a_model.time_steps import TwoStageTimeStep, v4_time_step


class AnalyticSolver:
    grid = RadialGrid(3)

    def initial_state(self):
        return ModelState(0., (28.,) * 3, (.1, .3, .1), .12)

    def advance(self, previous, dt):
        # The hint is deliberately distinct from all cell values.
        assert previous.surface_moisture == .12
        t = previous.time_s + dt
        return SimpleNamespace(state=ModelState(t, (28.,) * 3, (.1, .3 - .01*t, .1), .12))


def measure(solver, state):
    return max(state.moistures)


class EventTests(unittest.TestCase):
    def test_v4_two_stage_time_levels_and_exact_switch(self):
        expected = {
            "P0": (0.5, 1.0),
            "P1": (0.25, 0.5),
            "P2": (0.125, 0.25),
        }
        for level, step_sizes in expected.items():
            schedule = v4_time_step(level)
            self.assertEqual(schedule.step_sizes_s, step_sizes)
            self.assertEqual(schedule.breakpoints_s, (14400.0,))
            self.assertEqual(schedule(14399.999), step_sizes[0])
            self.assertEqual(schedule(14400), step_sizes[1])

    def test_schedule_lands_on_switch_and_sampling_grid(self):
        solver = AnalyticSolver()
        advances = []
        original = solver.advance
        solver.advance = lambda state, dt: (advances.append((state.time_s, state.time_s + dt))
                                             or original(state, dt))
        rows = []
        schedule = TwoStageTimeStep("test", .5, 1., 2.)
        result = run_streaming(solver, end_time_s=4, dt_s=1, time_step=schedule,
                               sample_interval_s=1, on_sample=lambda state, final: rows.append(state.time_s),
                               threshold=0, measure=measure)
        self.assertEqual(result["status"], "horizon_reached_without_event")
        self.assertEqual([right for _, right in advances], [.5, 1., 1.5, 2., 3., 4.])
        self.assertEqual(rows, [0., 1., 2., 3., 4.])

    def test_scheduled_one_shot_and_segmented_runs_are_identical(self):
        schedule = TwoStageTimeStep("test", .5, 1., 2.)
        one = run_streaming(AnalyticSolver(), end_time_s=4, dt_s=1, time_step=schedule,
                            threshold=0, measure=measure)
        first = run_streaming(AnalyticSolver(), end_time_s=2, dt_s=1, time_step=schedule,
                              threshold=0, measure=measure)
        second = run_streaming(AnalyticSolver(), end_time_s=4, dt_s=1, time_step=schedule,
                               initial=first["state"], threshold=0, measure=measure)
        self.assertEqual(second["state"], one["state"])
        self.assertEqual(first["steps"] + second["steps"], one["steps"])

    def test_environment_minutes_and_default_switch_are_exact_steps(self):
        solver = AnalyticSolver()
        right_times = []
        original = solver.advance
        solver.advance = lambda state, dt: (right_times.append(state.time_s + dt)
                                             or original(state, dt))
        schedule = TwoStageTimeStep("test", 1000., 1000., 14400.)
        run_streaming(solver, end_time_s=14401, dt_s=1000, time_step=schedule,
                      threshold=0, measure=lambda solver, state: 1.)
        self.assertEqual(right_times[:-1], [float(value) for value in range(60, 14401, 60)])
        self.assertEqual(right_times[-1], 14401.)

    def test_production_equilibrium(self):
        environment = SimpleNamespace(at=lambda t: (28., 2.55))
        solver = CoupledRadialSolver('q2q3', RadialGrid(10), environment, .02)
        result = run_streaming(solver, end_time_s=120, dt_s=30)
        self.assertEqual(result['status'], 'horizon_reached_without_event')
        self.assertTrue(all(abs(c - 2.55) < 1e-12 for c in result['state'].moistures))

    def test_threshold_equilibrium_is_not_strictly_dry(self):
        environment = SimpleNamespace(at=lambda t: (28., .15))
        solver = CoupledRadialSolver('q2q3', RadialGrid(10), environment, .02)
        initial = ModelState(0., (28.,) * 10, (.15,) * 10, .15)
        result = run_streaming(solver, end_time_s=1, dt_s=.5, initial=initial)
        self.assertEqual(result['status'], 'horizon_reached_without_event')
        self.assertGreaterEqual(maximum(solver, result['state']), .15)

    def test_decimal_dt_lands_on_output_time(self):
        environment = SimpleNamespace(at=lambda t: (28., 2.55))
        solver = CoupledRadialSolver('q2q3', RadialGrid(10), environment, .02)
        rows = []
        result = run_streaming(solver, end_time_s=2, dt_s=.1, sample_interval_s=1,
                               on_sample=lambda state, final: rows.append(state.time_s))
        self.assertEqual(rows, [0., 1., 2.])
        self.assertEqual(result['steps'], 20)

    def test_known_interior_event_and_unique_endpoint(self):
        rows = []
        result = run_streaming(AnalyticSolver(), end_time_s=25, dt_s=8, sample_interval_s=5,
                               on_sample=lambda state, final: rows.append((state.time_s, final)), measure=measure,
                               event_tolerance_s=1e-6)
        self.assertAlmostEqual(result['event'].state.time_s, 15, places=5)
        self.assertEqual(rows[:-1], [(0, False), (5, False), (10, False), (15, False)])
        self.assertTrue(rows[-1][1])
        self.assertGreater(result['event'].state.time_s, 15)
        self.assertLess(result['event'].maximum_moisture, .15)

    def test_full_reconstruction_maximum(self):
        solver = AnalyticSolver()
        solver.environment = SimpleNamespace(at=lambda t: (50, .05))
        solver.radius = SimpleNamespace(at=lambda t: .02)
        solver.boundary_state = lambda *args, **kwargs: SimpleNamespace(temperature_c=28, moisture=.4)
        self.assertEqual(maximum(solver, ModelState(1, (28,)*3, (.1,.3,.1), .4)), .4)
        self.assertEqual(maximum(solver, solver.initial_state()), .3)

    def test_fractional_event_and_halved_steps(self):
        solver = AnalyticSolver()
        original = solver.advance
        solver.advance = lambda state, dt: original(state, min(dt, .75))
        rows = []
        result = run_streaming(solver, end_time_s=25, dt_s=8, sample_interval_s=1,
                               threshold=.147, event_tolerance_s=1e-6, measure=measure,
                               on_sample=lambda state, final: rows.append(state.time_s))
        self.assertAlmostEqual(result['event'].state.time_s, 15.3, places=5)
        self.assertEqual(rows[:-1], list(range(16)))
        self.assertEqual(len(rows), len(set(rows)))

    def test_early_and_horizon(self):
        solver = AnalyticSolver()
        initial = ModelState(0, (28,)*3, (.1,)*3, .12)
        self.assertEqual(run_streaming(solver, end_time_s=10, dt_s=2, initial=initial, measure=measure)['steps'], 0)
        self.assertEqual(run_streaming(solver, end_time_s=10, dt_s=2, measure=measure)['status'], 'horizon_reached_without_event')

    def test_bracket_and_failure(self):
        solver = AnalyticSolver()
        initial = solver.initial_state()
        with self.assertRaises(ValueError):
            locate_event(solver, initial, solver.advance(initial, 1).state, dt_s=1, measure=measure)
        solver.advance = lambda state, dt: SimpleNamespace(state=state)
        with self.assertRaises(StepFailure):
            run_streaming(solver, end_time_s=10, dt_s=2, measure=measure)


if __name__ == '__main__':
    unittest.main()
