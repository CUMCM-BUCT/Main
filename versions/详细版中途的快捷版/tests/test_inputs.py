import unittest

from a_model.inputs import EnvironmentInput, ModelInputError, RadiusInput, SourceTrace


TRACE = SourceTrace("synthetic.xlsx", "Sheet1", "0" * 64, 0, 0)


def make_environment(**overrides):
    times = tuple(index * 60.0 for index in range(241))
    values = {
        "times_s": times,
        "temperatures_c": tuple(28.0 + index / 10.0 for index in range(241)),
        "equilibrium_moistures": tuple(0.02 + index / 100_000.0 for index in range(241)),
        "trace": TRACE,
    }
    values.update(overrides)
    return EnvironmentInput(**values)


def make_radius(**overrides):
    times = tuple(index * 1800.0 for index in range(145))
    radii = tuple(0.02 - index * (0.02 - 0.01198) / 144 for index in range(145))
    values = {"times_s": times, "radii_m": radii, "trace": TRACE}
    values.update(overrides)
    return RadiusInput(**values)


class InputTests(unittest.TestCase):
    def test_environment_shape_and_main_policy(self):
        environment = make_environment()
        self.assertEqual(len(environment.times_s), 241)
        self.assertEqual(environment.at(0.0), (28.0, 0.02))
        middle = environment.at(30.0)
        expected_temperature = (environment.temperatures_c[0] + environment.temperatures_c[1]) / 2.0
        expected_moisture = (environment.equilibrium_moistures[0] + environment.equilibrium_moistures[1]) / 2.0
        self.assertAlmostEqual(middle[0], expected_temperature)
        self.assertAlmostEqual(middle[1], expected_moisture)
        self.assertEqual(environment.at(14_401.0), (50.0, 0.05))

    def test_environment_last_hour_mean_includes_both_endpoints(self):
        environment = make_environment()
        temperature, moisture = environment.last_hour_mean()
        self.assertAlmostEqual(temperature, sum(environment.temperatures_c[-61:]) / 61)
        self.assertAlmostEqual(moisture, sum(environment.equilibrium_moistures[-61:]) / 61)

    def test_invalid_nominal_platform_is_rejected(self):
        invalid_platforms = (
            {"nominal_temperature_c": float("nan")},
            {"nominal_temperature_c": -273.15},
            {"nominal_equilibrium_moisture": float("nan")},
            {"nominal_equilibrium_moisture": -1.0e-12},
        )
        for overrides in invalid_platforms:
            with self.subTest(overrides=overrides), self.assertRaises(ModelInputError):
                make_environment(**overrides)

    def test_radius_shape_units_and_freeze(self):
        radius = make_radius()
        self.assertEqual(len(radius.times_s), 145)
        self.assertAlmostEqual(radius.at(0.0), 0.02)
        self.assertAlmostEqual(radius.at(900.0), (radius.radii_m[0] + radius.radii_m[1]) / 2.0)
        self.assertAlmostEqual(radius.at(259_201.0), 0.01198)

    def test_increasing_radius_is_rejected(self):
        times = tuple(index * 1800.0 for index in range(145))
        radii = [0.02 - index * (0.02 - 0.01198) / 144 for index in range(145)]
        radii[50] = radii[49] + 0.0001
        with self.assertRaises(ModelInputError):
            RadiusInput(times, tuple(radii), TRACE)

    def test_negative_query_time_is_rejected(self):
        with self.assertRaises(ValueError):
            make_environment().at(-1.0)


if __name__ == "__main__":
    unittest.main()
