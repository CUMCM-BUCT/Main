import csv
from dataclasses import replace
import hashlib
import json
from math import nan
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from a_model import load_environment
from a_model.q1_phase4 import (
    ConvergenceGateError,
    FIELD_TOLERANCES,
    MAXIMUM_RELATIVE_BALANCE_RESIDUAL,
    OUTPUT_RADII_CM,
    _monotonicity_gate,
    _orders_and_monotonicity,
    compare_runs,
    execute_phase4,
    run_health_failures,
    run_q1_streaming,
)
from a_model.solver import SolverOptions


class Q1Phase4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.environment = load_environment()
        cls.repo_root = Path(__file__).resolve().parents[1]

    def test_streaming_lands_on_time_and_space_sampling_contract(self):
        result = run_q1_streaming(
            self.environment,
            8,
            0.5,
            end_time_s=2.0,
            sample_interval_s=1.0,
            radii_cm=(0.0, 1.0, 2.0),
        )
        self.assertEqual([sample.time_s for sample in result.samples], [0.0, 1.0, 2.0])
        self.assertTrue(all(len(sample.temperatures_c) == 3 for sample in result.samples))
        self.assertTrue(all(len(sample.moistures) == 3 for sample in result.samples))
        self.assertEqual(result.samples[0].moisture_surface, 2.55)
        self.assertAlmostEqual(result.samples[0].moisture_flux_m_s, 2.024296e-6)
        self.assertEqual(result.samples[-1].temperature_center_c, result.samples[-1].temperatures_c[0])
        self.assertEqual(result.samples[-1].temperature_surface_c, result.samples[-1].temperatures_c[-1])

    def test_streaming_mode_does_not_retain_model_states_or_samples(self):
        emitted = []
        result = run_q1_streaming(
            self.environment,
            8,
            0.25,
            end_time_s=2.0,
            sample_interval_s=1.0,
            radii_cm=(0.0, 2.0),
            retain_samples=False,
            on_sample=lambda sample: emitted.append(sample.time_s),
        )
        self.assertEqual(result.samples, ())
        self.assertEqual(emitted, [0.0, 1.0, 2.0])
        self.assertEqual(result.summary.accepted_steps, 8)
        self.assertEqual(result.summary.sample_count, 3)

    def test_discrete_balances_and_high_initial_robin_branch(self):
        result = run_q1_streaming(
            self.environment,
            8,
            0.5,
            end_time_s=2.0,
            sample_interval_s=1.0,
        )
        summary = result.summary
        self.assertLess(summary.maximum_step_mass_balance_residual, 1.0e-12)
        self.assertLess(abs(summary.final_cumulative_mass_balance_residual), 1.0e-12)
        self.assertLess(summary.maximum_step_heat_balance_residual, 1.0e-6)
        self.assertLess(abs(summary.final_cumulative_heat_balance_residual), 1.0e-6)
        self.assertGreater(summary.first_step_surface_moisture, 2.0)
        self.assertGreater(summary.first_step_moisture_flux_m_s, 1.0e-6)
        self.assertEqual(summary.total_rejected_attempts, 0)

    def test_comparison_uses_all_common_times_and_projected_positions(self):
        coarse = run_q1_streaming(
            self.environment, 4, 0.5, end_time_s=2.0, sample_interval_s=1.0
        )
        fine = run_q1_streaming(
            self.environment, 8, 0.5, end_time_s=2.0, sample_interval_s=1.0
        )
        comparison = compare_runs(coarse, fine)
        self.assertEqual(comparison.left_grid_cells, 4)
        self.assertEqual(comparison.right_grid_cells, 8)
        self.assertEqual(
            set(comparison.errors),
            {
                "temperature_field_C", "temperature_center_C", "temperature_surface_C",
                "temperature_mean_C", "moisture_field", "moisture_center",
                "moisture_surface", "moisture_mean", "heat_flux_W_m2",
                "moisture_flux_m_s",
            },
        )
        self.assertGreater(comparison.errors["temperature_field_C"], 0.0)
        self.assertEqual(comparison.sample_count, 3)
        self.assertEqual(comparison.first_time_s, 0.0)
        self.assertEqual(comparison.last_time_s, 2.0)
        self.assertIn(1.0, [sample.time_s for sample in fine.samples])
        changed_at_one_second = replace(
            fine.samples[1],
            temperatures_c=(
                fine.samples[1].temperatures_c[0] + 1.0,
                *fine.samples[1].temperatures_c[1:],
            ),
        )
        early_changed = replace(
            fine,
            samples=(fine.samples[0], changed_at_one_second, fine.samples[2]),
        )
        gate_comparison = compare_runs(fine, early_changed, first_time_s=1.0)
        self.assertEqual(gate_comparison.sample_count, 2)
        self.assertEqual(gate_comparison.first_time_s, 1.0)
        self.assertEqual(gate_comparison.errors["temperature_field_C"], 1.0)
        self.assertEqual(
            gate_comparison.maximum_error_times_s["temperature_field_C"], 1.0
        )

        coarse_errors = {name: 1.0e-5 for name in FIELD_TOLERANCES}
        fine_errors = {name: 5.0e-6 for name in FIELD_TOLERANCES}
        coarse_errors["temperature_center_C"] = 1.0e-13
        fine_errors["temperature_center_C"] = 2.0e-13
        noise_coarse = replace(comparison, errors=coarse_errors)
        noise_fine = replace(comparison, errors=fine_errors)
        _, monotonicity = _orders_and_monotonicity(
            noise_coarse, noise_fine, 2.0
        )
        self.assertIsNone(monotonicity["temperature_center_C"])
        self.assertTrue(_monotonicity_gate(monotonicity))
        primary_coarse = replace(
            noise_coarse,
            errors={**coarse_errors, "temperature_field_C": 1.0e-13},
        )
        primary_fine = replace(
            noise_fine,
            errors={**fine_errors, "temperature_field_C": 2.0e-13},
        )
        _, primary_monotonicity = _orders_and_monotonicity(
            primary_coarse, primary_fine, 2.0
        )
        self.assertFalse(primary_monotonicity["temperature_field_C"])
        self.assertFalse(_monotonicity_gate(primary_monotonicity))

    def test_health_gate_detects_nonfinite_and_balance_failures(self):
        result = run_q1_streaming(
            self.environment, 8, 0.25, end_time_s=2.0, sample_interval_s=1.0
        )
        nonfinite = replace(
            result,
            summary=replace(
                result.summary,
                all_state_and_output_values_finite=False,
                maximum_reconstructed_temperature_c=nan,
            ),
        )
        self.assertIn(
            "non_finite_state_or_output",
            run_health_failures(nonfinite, SolverOptions()),
        )
        nonfinite_sample = replace(
            result.samples[-1],
            temperatures_c=(nan, *result.samples[-1].temperatures_c[1:]),
        )
        retained_nonfinite = replace(
            result,
            samples=(*result.samples[:-1], nonfinite_sample),
        )
        self.assertIn(
            "non_finite_retained_sample",
            run_health_failures(retained_nonfinite, SolverOptions()),
        )
        unbalanced = replace(
            result,
            summary=replace(
                result.summary,
                relative_cumulative_heat_balance_residual=1.0e-4,
            ),
        )
        self.assertIn(
            "cumulative_heat_balance",
            run_health_failures(unbalanced, SolverOptions()),
        )
        bad_solver_health = replace(
            result,
            summary=replace(
                result.summary,
                maximum_moisture_scaled_residual=2.0e-8,
                total_rejected_attempts=1,
            ),
        )
        failures = run_health_failures(bad_solver_health, SolverOptions())
        self.assertIn("moisture_residual", failures)
        self.assertIn("rejected_attempts", failures)

    def test_real_short_balance_health_through_n1280_and_mutation_margin(self):
        finest_result = None
        for grid_cells in (20, 80, 320, 1280):
            with self.subTest(grid_cells=grid_cells):
                result = run_q1_streaming(
                    self.environment,
                    grid_cells,
                    0.25,
                    end_time_s=2.0,
                    sample_interval_s=1.0,
                )
                self.assertEqual(run_health_failures(result, SolverOptions()), ())
                self.assertLessEqual(
                    result.summary.maximum_relative_step_heat_balance_residual,
                    MAXIMUM_RELATIVE_BALANCE_RESIDUAL,
                )
                self.assertLessEqual(
                    result.summary.relative_cumulative_heat_balance_residual,
                    MAXIMUM_RELATIVE_BALANCE_RESIDUAL,
                )
                finest_result = result
        assert finest_result is not None
        conservation_bug = replace(
            finest_result,
            summary=replace(
                finest_result.summary,
                maximum_relative_step_heat_balance_residual=1.0e-4,
            ),
        )
        self.assertIn(
            "step_heat_balance",
            run_health_failures(conservation_bug, SolverOptions()),
        )

    def test_dirty_and_expected_commit_guards_run_before_computation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            with patch(
                "a_model.q1_phase4._git_tracked_status",
                return_value=(" M src/a_model/q1_phase4.py",),
            ), patch("a_model.q1_phase4._git_path_is_tracked", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "clean tracked worktree"):
                    execute_phase4(
                        output,
                        self.environment,
                        repo_root=self.repo_root,
                    )
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(RuntimeError, "Expected commit"):
                execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    expected_commit="deadbeef",
                    allow_dirty=True,
                )
            self.assertFalse(output.exists())

    def test_fine_gate_configuration_has_hard_grid_and_time_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            with self.assertRaisesRegex(ValueError, "base matrix uses N=20, 40, and 80"):
                execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    grid_counts=(10, 20, 40),
                    allow_dirty=True,
                )
            with self.assertRaisesRegex(ValueError, "bounded at N=1280"):
                execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    fine_grid_levels=(640, 1280, 2560),
                    allow_dirty=True,
                )
            with self.assertRaisesRegex(ValueError, "bounded at dt=0.25"):
                execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    fine_time_steps_s=(0.5, 0.25, 0.125),
                    allow_dirty=True,
                )
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "N320 -> N640 -> N1280"):
                execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    fine_grid_levels=(160, 320, 640),
                    allow_dirty=True,
                )
            with self.assertRaisesRegex(ValueError, "dt=1 -> 0.5 -> 0.25"):
                execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    fine_time_steps_s=(2.0, 1.0, 0.5),
                    allow_dirty=True,
                )

    def test_failed_gate_preserves_existing_formal_files_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            output.mkdir()
            sentinels = {
                name: b"old-formal"
                for name in (
                    "q1_formal_temperature.csv",
                    "q1_formal_moisture.csv",
                    "q1_formal_qoi.csv",
                    "q1_formal_diagnostics.csv",
                    "q1_formal_summary.json",
                    "manifest.json",
                )
            }
            for name, content in sentinels.items():
                (output / name).write_bytes(content)
            with patch(
                "a_model.q1_phase4.FINE_GRID_LEVELS", (2, 4, 8)
            ), patch(
                "a_model.q1_phase4.BASE_GRID_COUNTS", (2, 4, 8)
            ), patch("a_model.q1_phase4._comparison_passes_thresholds", return_value=False):
                with self.assertRaises(ConvergenceGateError):
                    execute_phase4(
                        output,
                        self.environment,
                        repo_root=self.repo_root,
                        grid_counts=(2, 4, 8),
                        fine_grid_levels=(2, 4, 8),
                        fine_time_steps_s=(1.0, 0.5, 0.25),
                        end_time_s=2.0,
                        convergence_interval_s=1.0,
                        allow_dirty=True,
                    )
            for name, content in sentinels.items():
                self.assertEqual((output / name).read_bytes(), content)
            report = json.loads(
                (output / "q1_convergence_report.json").read_text(encoding="utf-8")
            )
            self.assertFalse(report["final_gate_pass"])
            self.assertIsNone(report["selected_formal_run"])
            self.assertTrue((output / "q1_candidate_diagnostics.csv").is_file())

    def test_source_drift_during_run_blocks_formal_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            output.mkdir()
            for name in ("q1_formal_summary.json", "manifest.json"):
                (output / name).write_bytes(b"old-formal")
            healthy_orders = (
                {name: 2.0 for name in FIELD_TOLERANCES},
                {name: True for name in FIELD_TOLERANCES},
            )
            with patch(
                "a_model.q1_phase4.FINE_GRID_LEVELS", (2, 4, 8)
            ), patch(
                "a_model.q1_phase4.BASE_GRID_COUNTS", (2, 4, 8)
            ), patch(
                "a_model.q1_phase4._comparison_passes_thresholds", return_value=True
            ), patch(
                "a_model.q1_phase4._orders_and_monotonicity",
                return_value=healthy_orders,
            ), patch(
                "a_model.q1_phase4.run_health_failures", return_value=()
            ), patch(
                "a_model.q1_phase4._git_head",
                side_effect=("a" * 40, "b" * 40),
            ), patch(
                "a_model.q1_phase4._git_tracked_status", return_value=()
            ), patch(
                "a_model.q1_phase4._git_path_is_tracked", return_value=True
            ), patch(
                "a_model.q1_phase4._git_status", return_value=()
            ):
                with self.assertRaisesRegex(RuntimeError, "entire run"):
                    execute_phase4(
                        output,
                        self.environment,
                        repo_root=self.repo_root,
                        grid_counts=(2, 4, 8),
                        fine_grid_levels=(2, 4, 8),
                        end_time_s=2.0,
                        convergence_interval_s=1.0,
                    )
            for name in ("q1_formal_summary.json", "manifest.json"):
                self.assertEqual((output / name).read_bytes(), b"old-formal")

    def test_cli_returns_nonzero_when_convergence_gate_fails(self):
        from scripts.run_q1_phase4 import main

        with patch(
            "scripts.run_q1_phase4.execute_phase4",
            side_effect=ConvergenceGateError("failed gate"),
        ), patch("sys.argv", ["run_q1_phase4.py"]):
            self.assertEqual(main(), 2)

    def test_short_artifact_run_has_shapes_and_verified_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "phase4"
            healthy_orders = (
                {name: 2.0 for name in FIELD_TOLERANCES},
                {name: True for name in FIELD_TOLERANCES},
            )
            with patch(
                "a_model.q1_phase4.FINE_GRID_LEVELS", (2, 4, 8)
            ), patch(
                "a_model.q1_phase4.BASE_GRID_COUNTS", (2, 4, 8)
            ), patch(
                "a_model.q1_phase4._comparison_passes_thresholds", return_value=True
            ), patch(
                "a_model.q1_phase4._orders_and_monotonicity",
                return_value=healthy_orders,
            ), patch(
                "a_model.q1_phase4.run_health_failures",
                return_value=(),
            ), patch(
                "a_model.q1_phase4._git_tracked_status",
                return_value=(),
            ), patch(
                "a_model.q1_phase4._git_path_is_tracked",
                return_value=True,
            ), patch(
                "a_model.q1_phase4._git_status",
                return_value=(),
            ):
                result = execute_phase4(
                    output,
                    self.environment,
                    repo_root=self.repo_root,
                    grid_counts=(2, 4, 8),
                    time_steps_s=(5.0, 2.0, 1.0, 0.5),
                    fine_grid_levels=(2, 4, 8),
                    fine_time_steps_s=(1.0, 0.5, 0.25),
                    end_time_s=2.0,
                    convergence_interval_s=1.0,
                )
            self.assertEqual(result["selected_formal_run"], "N8_dt0.25")
            for filename in ("q1_formal_temperature.csv", "q1_formal_moisture.csv"):
                with (output / filename).open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(len(rows), 3)
                self.assertEqual(len(rows[0]), 1 + len(OUTPUT_RADII_CM))
                self.assertTrue(all(len(row) == len(rows[0]) for row in rows[1:]))
            with (output / "q1_formal_qoi.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.reader(stream))), 3)
            with (output / "q1_formal_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.reader(stream))), 9)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["source_git_commit"]), 40)
            self.assertFalse(manifest["source_worktree"]["dirty"])
            self.assertGreaterEqual(
                set(manifest["source_worktree"]["source_file_sha256"]),
                {
                    "src/a_model/q1_phase4.py",
                    "src/a_model/solver.py",
                    "scripts/run_q1_phase4.py",
                    "tests/test_q1_phase4.py",
                },
            )
            self.assertEqual(
                manifest["input_files"]["附件1.xlsx"]["sha256"],
                self.environment.trace.sha256,
            )
            for filename, record in manifest["artifacts"].items():
                digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
                self.assertEqual(digest, record["sha256"])
                self.assertEqual((output / filename).stat().st_size, record["bytes"])
            metadata = json.loads((output / "q1_run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["numerical_settings"]["reproduction_command"][:2],
                ["python", "scripts/run_q1_phase4.py"],
            )
            self.assertNotIn(
                str(output.resolve()),
                metadata["numerical_settings"]["reproduction_command"],
            )
            fine_gate = result["convergence"]["bounded_fine_gate"]
            self.assertEqual(fine_gate["sample_interval_s"], 1.0)
            self.assertEqual(fine_gate["space_fine"]["sampling"]["sample_count"], 2)
            self.assertEqual(fine_gate["space_fine"]["sampling"]["first_time_s"], 1.0)
            self.assertEqual(fine_gate["space_fine"]["sampling"]["last_time_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
