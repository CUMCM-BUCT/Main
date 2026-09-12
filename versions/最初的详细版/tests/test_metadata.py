from datetime import datetime, timezone
import unittest

from a_model.inputs import SourceTrace
from a_model.metadata import build_run_metadata


class MetadataTests(unittest.TestCase):
    def test_metadata_records_assumptions_units_and_sources(self):
        source = SourceTrace("附件1.xlsx", "Sheet1", "a" * 64, 100, 241)
        metadata = build_run_metadata(
            "q2q3",
            [source],
            {"cells": 80, "time_step_s": 0.5},
            created_at=datetime(2026, 9, 11, tzinfo=timezone.utc),
        )
        self.assertEqual(metadata["model_version"], "A-V4-2026-09-11")
        self.assertEqual(metadata["sources"][0]["sha256"], "a" * 64)
        self.assertEqual(metadata["units"]["temperature_field"], "degC")
        self.assertEqual(len(metadata["assumptions"]), 6)
        self.assertEqual(metadata["boundary_coefficients"]["carried_forward_from"], "附录2")
        self.assertFalse(metadata["latent_heat_enabled"])
        self.assertEqual(metadata["initial_conditions"]["cylinder_length_m"], 0.25)
        self.assertEqual(metadata["target"]["max_moisture_dry_basis"], 0.15)
        self.assertEqual(metadata["property_model"]["coefficients"]["diffusivity_prefactor"], 2.4e-3)
        self.assertEqual(metadata["geometry"], {"kind": "fixed_radius", "radius_m": 0.02})
        self.assertNotIn("radius_in_range", metadata["input_policy"])
        self.assertEqual(metadata["runtime"]["implementation"], "CPython")
        self.assertRegex(metadata["runtime"]["python"], r"^\d+\.\d+\.\d+")
        self.assertNotEqual(metadata["runtime"]["dependencies"]["openpyxl"], "not-installed")

    def test_metadata_enforces_case_sources(self):
        environment = SourceTrace("附件1.xlsx", "Sheet1", "a" * 64, 100, 241)
        radius = SourceTrace("附件2.xlsx", "Sheet1", "b" * 64, 100, 145)
        invalid_cases = (
            ("q1", []),
            ("q1", [radius]),
            ("q2q3", [radius]),
            ("q4", [environment]),
            ("q4", [radius]),
        )
        for case_id, sources in invalid_cases:
            with self.subTest(case_id=case_id, sources=sources), self.assertRaises(ValueError):
                build_run_metadata(case_id, sources, {})

    def test_q4_records_shrinkage_geometry_only_with_both_sources(self):
        environment = SourceTrace("附件1.xlsx", "Sheet1", "a" * 64, 100, 241)
        radius = SourceTrace("附件2.xlsx", "Sheet1", "b" * 64, 100, 145)
        metadata = build_run_metadata("q4", [environment, radius], {})
        self.assertEqual(metadata["geometry"]["kind"], "proportional_radial_shrinkage")
        self.assertEqual(metadata["geometry"]["radius_after_range_m"], 0.01198)

    def test_q1_and_q2q3_record_fixed_geometry_without_shrinkage_policy(self):
        environment = SourceTrace("附件1.xlsx", "Sheet1", "a" * 64, 100, 241)
        for case_id in ("q1", "q2q3"):
            with self.subTest(case_id=case_id):
                metadata = build_run_metadata(case_id, [environment], {})
                self.assertEqual(metadata["geometry"], {"kind": "fixed_radius", "radius_m": 0.02})
                self.assertNotIn("radius_in_range", metadata["geometry"])

    def test_explicit_policy_overrides_replace_defaults(self):
        environment = SourceTrace("附件1.xlsx", "Sheet1", "a" * 64, 100, 241)
        metadata = build_run_metadata(
            "q1",
            [environment],
            {},
            input_policy={"environment": "measured_constant_test"},
            geometry_policy={"kind": "fixed_radius", "radius_m": 0.019},
        )
        self.assertEqual(metadata["input_policy"], {"environment": "measured_constant_test"})
        self.assertEqual(metadata["geometry"], {"kind": "fixed_radius", "radius_m": 0.019})


if __name__ == "__main__":
    unittest.main()
