import math
import unittest

from a_model.conventions import to_kelvin
from a_model.parameters import PROPERTY_COEFFICIENTS, get_case_parameters, parameter_snapshot


class ParameterTests(unittest.TestCase):
    def test_q1_values(self):
        case = get_case_parameters("q1")
        self.assertEqual(case.volumetric_heat_storage(2.55), 820.0 * 2600.0)
        self.assertEqual(case.conductivity(2.55), 0.36)
        self.assertAlmostEqual(case.diffusivity(2.55, 28.0), 4.93765509e-9, places=17)

    def test_q1_constant_laws_validate_state(self):
        case = get_case_parameters("q1")
        for law in (case.density_parameter, case.heat_capacity, case.conductivity):
            for invalid_moisture in (-1.0e-12, float("nan"), float("inf")):
                with self.subTest(law=law, moisture=invalid_moisture), self.assertRaises(ValueError):
                    law(invalid_moisture)
        for invalid_temperature in (float("nan"), float("inf"), -273.15):
            with self.subTest(temperature=invalid_temperature), self.assertRaises(ValueError):
                case.diffusivity(1.0, invalid_temperature)

    def test_q2_reference_values(self):
        case = get_case_parameters("q2q3")
        self.assertAlmostEqual(case.diffusivity(2.55, 28.0), 5.64168037e-9, places=17)
        self.assertAlmostEqual(case.density_parameter(0.15), 669.2)
        self.assertAlmostEqual(case.heat_capacity(0.15), 1806.8695652173913)
        self.assertAlmostEqual(case.conductivity(0.15), 0.2595652173913043)
        self.assertAlmostEqual(case.diffusivity(0.15, 50.0), 8.00120815e-10, places=18)

    def test_q4_reference_values(self):
        case = get_case_parameters("q4")
        self.assertAlmostEqual(case.density_parameter(2.55), 989.5)
        self.assertAlmostEqual(case.heat_capacity(2.55), 3394.3661971830984)
        self.assertAlmostEqual(case.conductivity(2.55), 0.26366197183098594)
        self.assertAlmostEqual(case.diffusivity(2.55, 28.0), 1.04711230e-9, places=17)

    def test_zero_moisture_uses_safe_diffusivity_limit(self):
        for case_id in ("q1", "q2q3", "q4"):
            case = get_case_parameters(case_id)
            self.assertGreaterEqual(case.diffusivity(0.0, 50.0), 0.0)
        q2 = get_case_parameters("q2q3")
        self.assertEqual(q2.density_parameter(0.0), 650.0)
        self.assertEqual(q2.heat_capacity(0.0), 1450.0)
        self.assertEqual(q2.conductivity(0.0), 0.21)

    def test_negative_moisture_rejected(self):
        with self.assertRaises(ValueError):
            get_case_parameters("q4").diffusivity(-1.0e-12, 50.0)

    def test_kelvin_conversion_is_explicit(self):
        self.assertTrue(math.isclose(to_kelvin(50.0), 323.15))

    def test_snapshot_uses_executable_coefficient_source(self):
        snapshot = parameter_snapshot("q4")
        self.assertEqual(snapshot["coefficients"]["diffusivity_prefactor"], 4.2e-4)
        self.assertEqual(
            snapshot["coefficients"]["rho_intercept"],
            PROPERTY_COEFFICIENTS["q4"].rho_intercept,
        )


if __name__ == "__main__":
    unittest.main()
