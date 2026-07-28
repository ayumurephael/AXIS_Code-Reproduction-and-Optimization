import math
import unittest

import torch

from tools.axis_repro.numerical_stability import (
    decide_gradient_step,
    top_gradient_diagnostics,
)


class GradientGuardDecisionTest(unittest.TestCase):
    def test_safe_gradient_is_applied(self):
        decision = decide_gradient_step(0.25, threshold=100.0)
        self.assertTrue(decision.should_step)
        self.assertEqual(decision.reason, "safe")

    def test_finite_spike_is_rejected(self):
        decision = decide_gradient_step(100.01, threshold=100.0)
        self.assertFalse(decision.should_step)
        self.assertEqual(decision.reason, "finite_spike")

    def test_nonfinite_gradient_is_rejected(self):
        for value in (math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                decision = decide_gradient_step(value, threshold=100.0)
                self.assertFalse(decision.should_step)
                self.assertEqual(decision.reason, "nonfinite")

    def test_invalid_threshold_is_rejected(self):
        for value in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    decide_gradient_step(1.0, threshold=value)


class GradientDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_are_bounded_and_report_nonfinite_values(self):
        first = torch.nn.Parameter(torch.zeros(4))
        first.grad = torch.tensor([3.0, 4.0, float("inf"), float("nan")])
        second = torch.nn.Parameter(torch.zeros(2))
        second.grad = torch.tensor([1.0, 2.0])

        rows = top_gradient_diagnostics(
            [("first", first), ("second", second)],
            limit=1,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "first")
        self.assertEqual(rows[0]["nonfinite_elements"], 2)
        self.assertAlmostEqual(rows[0]["finite_l2_norm"], 5.0)
        self.assertAlmostEqual(rows[0]["finite_abs_max"], 4.0)


if __name__ == "__main__":
    unittest.main()
