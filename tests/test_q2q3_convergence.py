import csv
import json
from pathlib import Path
import tempfile
import unittest

from a_model.q2q3_convergence import (SAMPLE_COLUMNS, TEMPERATURE_COLUMNS, MOISTURE_COLUMNS,
                                    branch_jump_j60, case_label, compare_cases, matrix_report,
                                    order_from_differences, run_case, sha256,
                                    _summary_integrity)


class ConvergenceComparisonTests(unittest.TestCase):
    def test_graded_case_labels_are_explicit_and_uniform_labels_are_stable(self):
        self.assertEqual(case_label(40, 15), "N40_dt15")
        self.assertEqual(case_label(40, 15, 1.0), "N40_dt15")
        self.assertEqual(case_label(40, 15, 2.0), "N40_dt15_grade2")

    def write_case(self, root, name, times, offset, event=181.):
        directory = root / name
        directory.mkdir()
        settings = {"cells": 20, "dt_s": 60., "end_time_s": 240.,
                    "sample_interval_s": 60., "event_tolerance_s": .01,
                    "convergence_verified": False}
        source_hashes = {"src/a_model/solver.py": "a" * 64}
        summary = {"complete": True, "numerical_settings": settings,
                   "source_file_sha256": source_hashes,
                   "sources": [{"filename": "附件1.xlsx", "sha256": "b" * 64}],
                   "signature": {"settings": settings,
                                 "source_file_sha256": source_hashes,
                                 "input_sha256": "b" * 64},
                   "final_snapshot": {"time_s": event if event is not None else times[-1]},
                   "maximum_location": {"time_s": event if event is not None else times[-1]},
                   "step_audit": {},
                   "result": ({"status": "event", "event": {"state": {"time_s": event}}}
                              if event is not None else {"status": "horizon_reached_without_event"})}
        with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS)
            writer.writeheader()
            for time in times:
                row = dict.fromkeys(SAMPLE_COLUMNS, offset)
                row["time_s"] = time
                row[TEMPERATURE_COLUMNS[-1]] = 3 * offset
                row[MOISTURE_COLUMNS[4]] = 2 * offset
                writer.writerow(row)
        summary["samples_sha256"] = sha256(directory / "samples.csv")
        summary["summary_content_sha256"] = _summary_integrity(summary)
        (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return directory

    def test_common_times_exclude_nonmatching_event_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.write_case(root, "left", [0, 60, 120, 180, 181], 1.)
            right = self.write_case(root, "right", [0, 60, 120, 180, 184], .75, 184.)
            result = compare_cases(left, right)
            self.assertEqual(result["common_sample_count"], 3)
            self.assertEqual(result["errors"]["temperature_field_C"], .75)
            self.assertEqual(result["maximum_error_radii_cm"]["temperature_field_C"], 2.)
            self.assertEqual(result["errors"]["moisture_field"], .5)
            self.assertEqual(result["maximum_error_radii_cm"]["moisture_field"], .4)
            self.assertEqual(result["errors"]["event_time_s"], 3.)
            self.assertFalse(result["starter_thresholds_pass"]["event_time_s"])
            shortened = compare_cases(left, right, last_time_s=120.)
            self.assertEqual(shortened["common_sample_count"], 2)

    def test_missing_common_sample_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.write_case(root, "left", [0, 60, 120, 180], 1.)
            right = self.write_case(root, "right", [0, 60, 180], .75)
            with self.assertRaises(ValueError):
                compare_cases(left, right)

    def test_orders_and_unavailable_event(self):
        self.assertAlmostEqual(order_from_differences(.04, .01, 2), 2)
        self.assertAlmostEqual(order_from_differences(.02, .01, 2), 1)
        self.assertLess(order_from_differences(.01, .02, 2), 0)
        self.assertIsNone(order_from_differences(0, 0, 2))
        self.assertIsNone(order_from_differences(None, 1, 2))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.write_case(root, "left", [0, 60, 120], 1., 120.)
            right = self.write_case(root, "right", [0, 60, 120], 1., None)
            comparison = compare_cases(left, right)
            self.assertIsNone(comparison["errors"]["event_time_s"])
            self.assertFalse(comparison["starter_thresholds_pass"]["event_time_s"])

    def test_branch_j60_uses_regular_post_forcing_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = self.write_case(Path(temporary), "case", [0, 60, 120, 180, 181], 1.)
            path = directory / "samples.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            for row, value in zip(rows, (1.0, .9, .7, .69, .1)):
                row["moisture_surface"] = value
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=SAMPLE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            metric = branch_jump_j60(path, after_time_s=60.)
            self.assertAlmostEqual(metric["J60"], .19)
            self.assertEqual(metric["maximum_time_s"], 180.)
            self.assertEqual(metric["count_at_least_5e-4"], 2)

    def test_clipped_nominal_steps_are_not_mislabelled(self):
        with self.assertRaises(ValueError):
            run_case("unused", 20, 5, 120, 1, {})
        with self.assertRaises(ValueError):
            run_case("unused", 20, 2, 120, 5, {})

    def test_comparison_rejects_tampered_samples_and_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.write_case(root, "left", [0, 60, 120], 1.)
            right = self.write_case(root, "right", [0, 60, 120], .9)
            with (right / "samples.csv").open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "samples changed"):
                compare_cases(left, right)

            right = self.write_case(root, "right_source", [0, 60, 120], .9)
            summary_path = right / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["signature"]["source_file_sha256"] = {"src/a_model/solver.py": "c" * 64}
            summary["source_file_sha256"] = summary["signature"]["source_file_sha256"]
            summary["summary_content_sha256"] = _summary_integrity(summary)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same source signature"):
                compare_cases(left, right)

    def test_comparison_rejects_signed_setting_or_input_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.write_case(root, "left", [0, 60, 120], 1.)
            right = self.write_case(root, "right", [0, 60, 120], .9)
            summary_path = right / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["numerical_settings"]["dt_s"] = 30.
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings differ"):
                compare_cases(left, right)

            right = self.write_case(root, "right_input", [0, 60, 120], .9)
            summary_path = right / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["signature"]["input_sha256"] = "d" * 64
            summary["summary_content_sha256"] = _summary_integrity(summary)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input hashes differ"):
                compare_cases(left, right)

    def test_matrix_report_rejects_noncurrent_source_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.write_case(root, "N20_dt60", [0, 60, 120, 180], 1.)
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["numerical_settings"].update(
                grading_exponent=1.0, minimum_cell_width_xi=.05,
                surface_half_width_xi=.025)
            summary["signature"]["settings"] = dict(summary["numerical_settings"])
            summary["summary_content_sha256"] = _summary_integrity(summary)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "current runner provenance"):
                matrix_report(root, [20], [60], expected_source_hashes={"different": "hash"})


if __name__ == "__main__":
    unittest.main()
