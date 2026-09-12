import unittest

from a_model.export_contract import (
    EXPORTS,
    FIXED_RADII_Q1_Q2_Q3_CM,
    FIXED_RADII_Q4_CM,
    Q4_OUTSIDE_VALUE,
    q4_fixed_position_is_inside,
)


class ExportContractTests(unittest.TestCase):
    def test_fixed_spatial_grids(self):
        self.assertEqual(FIXED_RADII_Q1_Q2_Q3_CM, tuple(i / 10 for i in range(21)))
        self.assertEqual(FIXED_RADII_Q4_CM, tuple(i / 10 for i in range(20)))

    def test_q1_is_one_through_1800_seconds(self):
        times = EXPORTS["q1"].sample_times(1800.0)
        self.assertEqual((times[0], times[-1], len(times)), (1.0, 1800.0, 1800))

    def test_q1_rejects_any_noncontract_end(self):
        for invalid_end in (900.0, 1800.5, 1801.0):
            with self.subTest(invalid_end=invalid_end), self.assertRaises(ValueError):
                EXPORTS["q1"].sample_times(invalid_end)

    def test_event_endpoint_is_appended_once(self):
        q3 = EXPORTS["q3"]
        self.assertEqual(q3.sample_times(125.5), (60.0, 120.0, 125.5))
        self.assertEqual(q3.sample_times(180.0), (60.0, 120.0, 180.0))

    def test_q4_outside_position_is_blank_by_contract(self):
        self.assertIsNone(Q4_OUTSIDE_VALUE)
        self.assertTrue(q4_fixed_position_is_inside(1.1, 0.011))
        self.assertFalse(q4_fixed_position_is_inside(1.2, 0.011))

    def test_q4_position_check_rejects_nonfinite_and_negative_tolerance(self):
        invalid_arguments = (
            (float("nan"), 0.011, 1.0e-10),
            (1.0, float("inf"), 1.0e-10),
            (1.0, 0.011, float("nan")),
            (1.0, 0.011, -1.0e-10),
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                q4_fixed_position_is_inside(*arguments)

    def test_template_sheet_names_are_preserved(self):
        self.assertEqual(EXPORTS["q1"].sheets, ("温度", "水分浓度"))
        self.assertEqual(EXPORTS["q3"].sheets, ("Sheet1",))
        self.assertEqual(EXPORTS["q4"].sheets, ("Sheet1",))


if __name__ == "__main__":
    unittest.main()
