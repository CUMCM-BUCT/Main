import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from a_model.fvm import RadialGrid
from a_model.solver import ModelState
from a_model.time_steps import TwoStageTimeStep


SPEC = importlib.util.spec_from_file_location(
    "q2q3_runner", Path(__file__).resolve().parents[1] / "scripts" / "run_q2q3_phase5.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class AnalyticSolver:
    grid = RadialGrid(3)
    environment = SimpleNamespace(at=lambda t: (28., .05))
    radius = SimpleNamespace(at=lambda t: .02)

    def __init__(self, rate=.01, fail_at=None):
        self.rate = rate
        self.fail_at = fail_at

    def initial_state(self):
        return ModelState(0., (28.,) * 3, (.303,) * 3, .12)

    def boundary_state(self, *args, **kwargs):
        assert kwargs["surface_moisture_hint"] == .12
        return SimpleNamespace(temperature_c=28., moisture=.12)

    def advance(self, previous, dt):
        assert previous.surface_moisture == .12
        if self.fail_at is not None and previous.time_s >= self.fail_at:
            raise InterruptedError("Synthetic interruption after committed segment")
        time_s = previous.time_s + dt
        return SimpleNamespace(state=ModelState(time_s, (28.,) * 3,
                                               (.303 - self.rate * time_s,) * 3, .12))


class GradedAnalyticSolver(AnalyticSolver):
    grid = RadialGrid.surface_graded(3, 2.0)


class ResumeTests(unittest.TestCase):
    def run_export(self, output, *, resume=False, interval=7, stop=None,
                   end=25, solver=None, metadata=None, dt=1, time_step=None):
        return runner.run_export(solver or AnalyticSolver(), Path(output),
            metadata or {"source_file_sha256": {}, "source_git_commit": "test",
                         "result_classification": "candidate_only"},
            end_time_s=end, dt_s=dt, event_tolerance_s=1e-6,
            checkpoint_interval_s=interval, resume=resume, stop_after_segments=stop,
            time_step=time_step)

    def assert_same_csv(self, first, second):
        for name in runner.CSV_NAMES:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

    def times(self, output, name):
        with (output / name).open(newline="", encoding="utf-8") as stream:
            return [float(row[0]) for row in list(csv.reader(stream))[1:]]

    def test_one_shot_and_resumed_fractional_event_are_identical(self):
        with TemporaryDirectory() as temporary:
            one, resumed = Path(temporary) / "one", Path(temporary) / "resumed"
            expected = self.run_export(one, interval=100)
            paused = self.run_export(resumed, stop=1)
            self.assertEqual(paused["status"], "paused")
            self.assertEqual(paused["state"]["time_s"], 7)
            actual = self.run_export(resumed, resume=True)
            self.assertEqual(actual, expected)
            self.assert_same_csv(one, resumed)
            event = actual["event"]["state"]["time_s"]
            self.assertAlmostEqual(event, 15.3, places=5)
            self.assertLess(actual["event"]["maximum_moisture"], .15)
            self.assertEqual(self.times(resumed, runner.CSV_NAMES[0]), [*range(1, 16), event])
            self.assertEqual(self.times(resumed, runner.CSV_NAMES[2]), [event])
            before = (resumed / "checkpoint.json").read_bytes()
            (resumed / "metadata.json").unlink()
            self.assertEqual(self.run_export(resumed, resume=True), actual)
            self.assertEqual((resumed / "checkpoint.json").read_bytes(), before)
            repaired = json.loads((resumed / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(repaired["result"], actual)
            self.assertEqual(repaired["checkpoint"]["state_time_s"], event)

    def test_interrupt_recovers_uncommitted_tails(self):
        with TemporaryDirectory() as temporary:
            one, resumed = Path(temporary) / "one", Path(temporary) / "resumed"
            expected = self.run_export(one)
            with self.assertRaises(InterruptedError):
                self.run_export(resumed, solver=AnalyticSolver(fail_at=10))
            checkpoint = json.loads((resumed / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["state"]["time_s"], 7)
            self.assertEqual(checkpoint["state"]["surface_moisture"], .12)
            name = runner.CSV_NAMES[0]
            self.assertGreater((resumed / name).stat().st_size, checkpoint["files"][name]["offset_bytes"])
            self.assertEqual(self.run_export(resumed, resume=True), expected)
            self.assert_same_csv(one, resumed)

    def test_two_stage_one_shot_and_resumed_exports_are_identical(self):
        schedule = TwoStageTimeStep("test", .5, 1., 6.)
        with TemporaryDirectory() as temporary:
            one, resumed = Path(temporary) / "one", Path(temporary) / "resumed"
            expected = self.run_export(one, interval=100, time_step=schedule)
            paused = self.run_export(resumed, stop=1, time_step=schedule)
            self.assertEqual(paused["status"], "paused")
            actual = self.run_export(resumed, resume=True, time_step=schedule)
            self.assertEqual(actual, expected)
            self.assert_same_csv(one, resumed)
            checkpoint = json.loads((resumed / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["identity"]["export_controls"]["time_step_schedule"],
                             schedule.as_dict())

    def test_mismatch_refused_without_any_output_changes(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.run_export(output, stop=1)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            for controls in ({"dt": .5}, {"metadata": {"source_git_commit": "changed"}},
                             {"end": 30}, {"interval": 10},
                             {"solver": GradedAnalyticSolver()},
                             {"time_step": TwoStageTimeStep("test", .5, 1., 6.)}):
                with self.assertRaises(ValueError):
                    self.run_export(output, resume=True, **controls)
                self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
            with self.assertRaises(FileExistsError):
                self.run_export(output)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)

    def test_corrupt_committed_prefix_refused_before_tail_recovery(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(InterruptedError):
                self.run_export(output, solver=AnalyticSolver(fail_at=10))
            # Test fixture mutation, not a production recovery operation.
            path = output / runner.CSV_NAMES[2]
            data = path.read_bytes()
            path.write_bytes(b"X" + data[1:])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "modified"):
                self.run_export(output, resume=True)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)

    def test_noevent_fractional_horizon_emits_only_regular_rows(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = self.run_export(output, end=65.35, solver=AnalyticSolver(rate=.001))
            self.assertEqual(result["status"], "horizon_reached_without_event")
            self.assertEqual(result["state"]["time_s"], 65.35)
            self.assertEqual(self.times(output, runner.CSV_NAMES[0]), list(range(1, 66)))
            self.assertEqual(self.times(output, runner.CSV_NAMES[2]), [60])
            checkpoint = json.loads((output / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["files"][runner.CSV_NAMES[0]]["rows"], 65)
            self.assertEqual(checkpoint["files"][runner.CSV_NAMES[2]]["rows"], 1)

    def test_invalid_checkpoint_interval_creates_no_files(self):
        with TemporaryDirectory() as temporary:
            for interval in (0, .5, float("inf")):
                with self.assertRaises(ValueError):
                    self.run_export(temporary, interval=interval)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_export_dt_must_be_commensurate_with_one_second_grid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (dt, expected_steps) in enumerate(((2, 2), (1, 2), (.5, 4), (.25, 8))):
                output = root / f"valid_{index}"
                result = self.run_export(output, end=2, solver=AnalyticSolver(rate=.001), dt=dt)
                self.assertEqual(result["status"], "horizon_reached_without_event")
                self.assertEqual(result["steps"], expected_steps)
                self.assertEqual(self.times(output, runner.CSV_NAMES[0]), [1, 2])
            for dt in (1.875, 2.5, .3):
                output = root / f"invalid_{dt}"
                with self.assertRaisesRegex(ValueError, "1 s sampling interval"):
                    self.run_export(output, end=2, solver=AnalyticSolver(rate=.001), dt=dt)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
