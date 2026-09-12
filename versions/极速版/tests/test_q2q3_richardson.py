import unittest

from a_model.q2q3_richardson import (
    RICHARDSON_GATE,
    extrapolate_pair,
    gate_result,
)


class Q2Q3RichardsonTests(unittest.TestCase):
    def test_first_order_time_extrapolation(self):
        # y(h) = 10 + 3h + h^2; the first-order Richardson estimate
        # cancels the linear term and leaves the expected O(h^2) remainder.
        coarse, fine = 10 + 3 * 2 + 4, 10 + 3 + 1
        self.assertAlmostEqual(extrapolate_pair(coarse, fine, order=1), 8.0)

    def test_gate_result_requires_every_metric(self):
        values = {name: limit for name, limit in RICHARDSON_GATE.items()}
        passed = gate_result(values)
        self.assertTrue(passed["passed"])
        values["event_time_s"] = RICHARDSON_GATE["event_time_s"] + 1e-9
        self.assertFalse(gate_result(values)["passed"])


if __name__ == "__main__":
    unittest.main()
