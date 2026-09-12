import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from a_model.solver import ModelState, SolverOptions
from a_model.time_steps import v4_time_step
import a_model.q2q3_convergence as convergence


class FakeAuditedSolver:
    interrupt_at = None

    def __init__(self, case, grid, environment, radius):
        self.grid = grid
        self.environment = environment
        self.options = SolverOptions()
        self.audit = {
            "advance_calls_including_event_trials": 0,
            "max_heat_scaled_residual": 0.,
            "max_moisture_scaled_residual": 0.,
            "minimum_moisture": 2.55,
            "sum_absolute_mass_residual_over_C0": 0.,
            "max_absolute_heat_balance_residual_J_m3": 0.,
            "max_picard_iterations": 0,
            "minimum_accepted_dt_s": None,
            "rejected_attempts": 0,
        }

    def initial_state(self):
        return ModelState(0., (28.,) * self.grid.cell_count,
                          (2.55,) * self.grid.cell_count, 2.55)


def fake_streaming(solver, *, end_time_s, dt_s, sample_interval_s,
                   on_sample, initial=None, **unused):
    state = initial or solver.initial_state()
    on_sample(state, False)
    steps = 0
    diagnostics = {"rejected_attempts": 0, "max_heat_residual": 0.,
                   "max_moisture_residual": 0.}
    while state.time_s < end_time_s:
        time_s = min(end_time_s, state.time_s + sample_interval_s)
        state = ModelState(time_s, (28. + time_s,) * solver.grid.cell_count,
                           (2.55 - time_s / 100,) * solver.grid.cell_count,
                           2.55 - time_s / 100)
        steps += 1
        rejected = 1 if int(time_s) % 3 == 0 else 0
        diagnostics["rejected_attempts"] += rejected
        diagnostics["max_heat_residual"] = max(diagnostics["max_heat_residual"], time_s / 1000)
        diagnostics["max_moisture_residual"] = max(diagnostics["max_moisture_residual"], time_s / 2000)
        audit = solver.audit
        audit["advance_calls_including_event_trials"] += 1
        audit["rejected_attempts"] += rejected
        audit["max_heat_scaled_residual"] = max(audit["max_heat_scaled_residual"], time_s / 1000)
        audit["max_moisture_scaled_residual"] = max(audit["max_moisture_scaled_residual"], time_s / 2000)
        audit["minimum_moisture"] = min(audit["minimum_moisture"], *state.moistures)
        audit["max_picard_iterations"] = max(audit["max_picard_iterations"], 2)
        audit["minimum_accepted_dt_s"] = dt_s if audit["minimum_accepted_dt_s"] is None else min(audit["minimum_accepted_dt_s"], dt_s)
        on_sample(state, False)
        if solver.interrupt_at is not None and time_s >= solver.interrupt_at:
            raise InterruptedError("synthetic interruption after an uncommitted row")
    return {"status": "horizon_reached_without_event", "state": state,
            "steps": steps, "diagnostics": diagnostics}


def fake_snapshot(solver, state):
    return {name: state.time_s if name == "time_s" else state.time_s / 10
            for name in convergence.SAMPLE_COLUMNS}


class ConvergenceCheckpointTests(unittest.TestCase):
    provenance = {"source_file_sha256": {"runner.py": "a" * 64},
                  "source_git_commit": "test", "command": ["test"]}

    def setUp(self):
        FakeAuditedSolver.interrupt_at = None
        self.patches = (
            patch.object(convergence, "AuditedSolver", FakeAuditedSolver),
            patch.object(convergence, "load_environment",
                         return_value=SimpleNamespace(trace=SimpleNamespace(sha256="input"))),
            patch.object(convergence, "build_run_metadata",
                         side_effect=lambda case, sources, settings: {"numerical_settings": settings}),
            patch.object(convergence, "run_streaming", side_effect=fake_streaming),
            patch.object(convergence, "snapshot", side_effect=fake_snapshot),
            patch.object(convergence, "maximum_location",
                         side_effect=lambda solver, state: {"time_s": state.time_s}),
            patch.object(convergence, "CHECKPOINT_INTERVAL_S", 2.),
        )
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def run_case(self, output, *, dt=1., provenance=None):
        return convergence.run_case(output, 4, dt, 5., 1.,
                                    provenance or self.provenance, 1.)

    def test_interrupted_resume_matches_one_shot_csv_result_and_audit(self):
        with TemporaryDirectory() as temporary:
            one = Path(temporary) / "one"
            resumed = Path(temporary) / "resumed"
            expected = self.run_case(one)
            FakeAuditedSolver.interrupt_at = 3.
            with self.assertRaises(InterruptedError):
                self.run_case(resumed)
            checkpoint = json.loads((resumed / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["state"]["time_s"], 2.)
            self.assertEqual(len(checkpoint["state"]["temperatures_c"]), 4)
            self.assertGreater((resumed / "samples.partial.csv").stat().st_size,
                               checkpoint["file"]["offset_bytes"])
            FakeAuditedSolver.interrupt_at = None
            actual = self.run_case(resumed)
            self.assertEqual((one / "samples.csv").read_bytes(),
                             (resumed / "samples.csv").read_bytes())
            self.assertEqual(actual["result"], expected["result"])
            self.assertEqual(actual["final_snapshot"], expected["final_snapshot"])
            self.assertEqual(actual["step_audit"], expected["step_audit"])
            self.assertEqual(actual["samples_sha256"], expected["samples_sha256"])
            self.assertFalse((resumed / "checkpoint.json").exists())
            saved = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
            one_saved = json.loads((one / "summary.json").read_text(encoding="utf-8"))
            saved.pop("elapsed_wall_s")
            one_saved.pop("elapsed_wall_s")
            self.assertEqual(saved, one_saved)
            self.assertTrue(saved["complete"])
            self.assertNotIn("failure", saved)

    def test_modified_committed_prefix_is_refused_before_truncation(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            FakeAuditedSolver.interrupt_at = 3.
            with self.assertRaises(InterruptedError):
                self.run_case(output)
            path = output / "samples.partial.csv"
            data = path.read_bytes()
            path.write_bytes(b"X" + data[1:])
            before = {item.name: item.read_bytes() for item in output.iterdir()}
            FakeAuditedSolver.interrupt_at = None
            with self.assertRaisesRegex(ValueError, "modified"):
                self.run_case(output)
            self.assertEqual({item.name: item.read_bytes() for item in output.iterdir()}, before)

    def test_signature_mismatch_is_refused_without_changes(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            FakeAuditedSolver.interrupt_at = 3.
            with self.assertRaises(InterruptedError):
                self.run_case(output)
            before = {item.name: item.read_bytes() for item in output.iterdir()}
            changed = {**self.provenance, "source_file_sha256": {"runner.py": "b" * 64}}
            with self.assertRaisesRegex(ValueError, "different inputs/settings/source"):
                self.run_case(output, provenance=changed)
            self.assertEqual({item.name: item.read_bytes() for item in output.iterdir()}, before)

    def test_checkpoint_state_must_match_last_committed_csv_row(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            FakeAuditedSolver.interrupt_at = 3.
            with self.assertRaises(InterruptedError):
                self.run_case(output)
            path = output / "checkpoint.json"
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            checkpoint["state"]["time_s"] = 1.
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            FakeAuditedSolver.interrupt_at = None
            with self.assertRaisesRegex(ValueError, "last committed CSV row"):
                self.run_case(output)

    def test_completed_summary_integrity_is_checked(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.run_case(output)
            path = output / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            summary["final_snapshot"]["time_s"] = 4.
            path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "summary changed"):
                convergence.validated_summary(output)

    def test_live_case_lock_refuses_second_writer(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            lock = output / ".run.lock"
            lock.write_text(json.dumps({"pid": os.getpid(), "created_time_ns": 1}),
                            encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already running"):
                self.run_case(output)

    def test_v4_integer_output_steps_are_supported(self):
        with TemporaryDirectory() as temporary:
            for dt in (2., 1., .5, .25):
                result = convergence.run_case(Path(temporary) / f"dt-{dt}", 4, dt,
                                              4., 2., self.provenance, 1.)
                self.assertTrue(result["complete"])
                self.assertEqual(result["final_snapshot"]["time_s"], 4.)

    def test_two_stage_schedule_is_signed_and_resumable(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            schedule = v4_time_step("P1", 2.)
            result = convergence.run_case(
                output, 4, schedule.late_dt_s, 5., 1., self.provenance, 1., schedule
            )
            self.assertEqual(result["numerical_settings"]["time_step_schedule"],
                             schedule.as_dict())
            self.assertEqual(result["signature"]["settings"]["time_step_schedule"],
                             schedule.as_dict())


if __name__ == "__main__":
    unittest.main()
