import unittest
from types import SimpleNamespace

from a_model.fvm import RadialGrid
from a_model.q2q3_phase5 import locate_event, maximum, run_streaming
from a_model.solver import CoupledRadialSolver, ModelState, StepFailure


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
    def test_production_equilibrium(self):
        environment = SimpleNamespace(at=lambda t: (28., 2.55))
        solver = CoupledRadialSolver('q2q3', RadialGrid(10), environment, .02)
        result = run_streaming(solver, end_time_s=120, dt_s=30)
        self.assertEqual(result['status'], 'horizon_reached_without_event')
        self.assertTrue(all(abs(c - 2.55) < 1e-12 for c in result['state'].moistures))

    def test_known_interior_event_and_unique_endpoint(self):
        rows = []
        result = run_streaming(AnalyticSolver(), end_time_s=25, dt_s=8, sample_interval_s=5,
                               on_sample=lambda state, final: rows.append((state.time_s, final)), measure=measure,
                               event_tolerance_s=1e-6)
        self.assertAlmostEqual(result['event'].state.time_s, 15, places=5)
        self.assertEqual(rows, [(0, False), (5, False), (10, False), (15, True)])

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
