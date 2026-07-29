import math
import unittest

import torch

from tools.axis_repro.numerical_stability import (
    GradientGuardFailFast,
    decide_gradient_step,
    top_gradient_diagnostics,
    validate_completed_epoch_guard_audit,
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


class GradientGuardFailFastTest(unittest.TestCase):
    @staticmethod
    def tracker(**overrides):
        values = {
            "window_size": 100,
            "window_min_observations": 100,
            "max_window_skip_rate": 0.05,
            "epoch_min_observations": 500,
            "max_epoch_skip_rate": 0.01,
            "max_consecutive_skips": 512,
            "fail_on_nonfinite": True,
        }
        values.update(overrides)
        return GradientGuardFailFast(**values)

    def test_intermittent_skip_storm_triggers_window_rate(self):
        tracker = self.tracker()
        spike_steps = {5, 25, 45, 65, 85, 100}
        decision = None
        for step in range(1, 101):
            spike = step in spike_steps
            decision = tracker.observe(
                skipped=spike,
                gradient_reason="finite_spike" if spike else "safe",
            )
        self.assertTrue(decision.should_abort)
        self.assertEqual(decision.reason, "window_gradient_skip_rate")
        self.assertEqual(decision.audit["max_observed_consecutive_skips"], 1)
        self.assertAlmostEqual(decision.audit["window_skip_rate"], 0.06)

    def test_rate_equal_to_threshold_is_tolerated(self):
        tracker = self.tracker()
        spike_steps = {20, 40, 60, 80, 100}
        decision = None
        for step in range(1, 101):
            spike = step in spike_steps
            decision = tracker.observe(
                skipped=spike,
                gradient_reason="finite_spike" if spike else "safe",
            )
        self.assertFalse(decision.should_abort)
        self.assertAlmostEqual(decision.audit["window_skip_rate"], 0.05)

    def test_epoch_rate_catches_storm_when_window_policy_is_loose(self):
        tracker = self.tracker(
            max_window_skip_rate=0.50,
            epoch_min_observations=100,
        )
        spike_steps = {50, 100}
        decision = None
        for step in range(1, 101):
            spike = step in spike_steps
            decision = tracker.observe(
                skipped=spike,
                gradient_reason="finite_spike" if spike else "safe",
            )
        self.assertTrue(decision.should_abort)
        self.assertEqual(decision.reason, "epoch_gradient_skip_rate")

    def test_nonfinite_aborts_immediately(self):
        decision = self.tracker().observe(
            skipped=True,
            gradient_reason="nonfinite",
        )
        self.assertTrue(decision.should_abort)
        self.assertEqual(decision.reason, "nonfinite_gradient")

    def test_consecutive_limit_triggers_at_exact_boundary(self):
        tracker = self.tracker(max_consecutive_skips=3)
        for _ in range(2):
            decision = tracker.observe(
                skipped=True,
                gradient_reason="finite_spike",
            )
            self.assertFalse(decision.should_abort)
        decision = tracker.observe(
            skipped=True,
            gradient_reason="finite_spike",
        )
        self.assertTrue(decision.should_abort)
        self.assertEqual(decision.reason, "consecutive_gradient_skips")

    def test_reason_and_skip_flag_must_agree(self):
        tracker = self.tracker()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            tracker.observe(skipped=False, gradient_reason="finite_spike")

    def test_invalid_fail_fast_policy_is_rejected(self):
        invalid = (
            {"window_size": 0},
            {"window_min_observations": 101},
            {"max_window_skip_rate": 1.0},
            {"epoch_min_observations": 0},
            {"max_epoch_skip_rate": -0.1},
            {"max_consecutive_skips": 0},
        )
        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    self.tracker(**values)


class CompletedEpochGuardAuditTest(unittest.TestCase):
    @staticmethod
    def audit(**overrides):
        values = {
            "epoch": 5,
            "epoch_attempted_steps": 5_700,
            "epoch_skipped_steps": 11,
            "epoch_nonfinite_steps": 0,
            "epoch_skip_rate": 11 / 5_700,
            "fail_fast_triggered": False,
            "checkpoint_eligible": True,
        }
        values.update(overrides)
        return values

    def test_safe_completed_epoch_is_eligible(self):
        result = validate_completed_epoch_guard_audit(
            self.audit(),
            expected_epoch=5,
            expected_attempted_steps=5_700,
            max_epoch_skip_rate=0.01,
        )
        self.assertEqual(result["epoch_skipped_steps"], 11)

    def test_failed_epoch_is_not_checkpoint_eligible(self):
        with self.assertRaisesRegex(ValueError, "skip-rate"):
            validate_completed_epoch_guard_audit(
                self.audit(
                    epoch_skipped_steps=395,
                    epoch_skip_rate=395 / 5_700,
                ),
                expected_epoch=5,
                expected_attempted_steps=5_700,
                max_epoch_skip_rate=0.01,
            )

    def test_inconsistent_recorded_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_completed_epoch_guard_audit(
                self.audit(epoch_skip_rate=0.0),
                expected_epoch=5,
                expected_attempted_steps=5_700,
                max_epoch_skip_rate=0.01,
            )


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
