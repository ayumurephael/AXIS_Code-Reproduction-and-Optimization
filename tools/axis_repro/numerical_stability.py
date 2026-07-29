"""Numerical-stability helpers shared by formal AXIS training jobs."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Deque, Iterable, Mapping, Tuple

import torch


@dataclass(frozen=True)
class GradientGuardDecision:
    """Fail-closed decision made before an optimizer update."""

    should_step: bool
    reason: str
    gradient_norm: float
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GradientGuardFailFastDecision:
    """Decision made after one synchronized DDP guard observation."""

    should_abort: bool
    reason: str
    audit: dict

    def to_dict(self) -> dict:
        return asdict(self)


class GradientGuardFailFast:
    """Detect intermittent skip storms that a consecutive-only guard misses."""

    _VALID_REASONS = frozenset({"safe", "finite_spike", "nonfinite"})

    def __init__(
        self,
        *,
        window_size: int,
        window_min_observations: int,
        max_window_skip_rate: float,
        epoch_min_observations: int,
        max_epoch_skip_rate: float,
        max_consecutive_skips: int,
        fail_on_nonfinite: bool = True,
    ) -> None:
        if window_size <= 0:
            raise ValueError("gradient-guard window size must be positive")
        if not 1 <= window_min_observations <= window_size:
            raise ValueError(
                "gradient-guard window minimum must be in [1, window_size]"
            )
        if not 0.0 <= max_window_skip_rate < 1.0:
            raise ValueError(
                "gradient-guard maximum window skip rate must be in [0, 1)"
            )
        if epoch_min_observations <= 0:
            raise ValueError(
                "gradient-guard epoch minimum observations must be positive"
            )
        if not 0.0 <= max_epoch_skip_rate < 1.0:
            raise ValueError("gradient-guard maximum epoch skip rate must be in [0, 1)")
        if max_consecutive_skips <= 0:
            raise ValueError("gradient-guard consecutive-skip limit must be positive")
        if not isinstance(fail_on_nonfinite, bool):
            raise ValueError("gradient-guard fail_on_nonfinite must be boolean")

        self.window_size = int(window_size)
        self.window_min_observations = int(window_min_observations)
        self.max_window_skip_rate = float(max_window_skip_rate)
        self.epoch_min_observations = int(epoch_min_observations)
        self.max_epoch_skip_rate = float(max_epoch_skip_rate)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.fail_on_nonfinite = fail_on_nonfinite
        self._window: Deque[bool] = deque(maxlen=self.window_size)
        self.epoch_attempted_steps = 0
        self.epoch_skipped_steps = 0
        self.epoch_nonfinite_steps = 0
        self.consecutive_skips = 0
        self.max_observed_consecutive_skips = 0

    def snapshot(self) -> dict:
        window_observations = len(self._window)
        window_skipped_steps = sum(self._window)
        return {
            "window_size": self.window_size,
            "window_min_observations": self.window_min_observations,
            "max_window_skip_rate": self.max_window_skip_rate,
            "epoch_min_observations": self.epoch_min_observations,
            "max_epoch_skip_rate": self.max_epoch_skip_rate,
            "max_consecutive_skips": self.max_consecutive_skips,
            "fail_on_nonfinite": self.fail_on_nonfinite,
            "window_observations": window_observations,
            "window_skipped_steps": window_skipped_steps,
            "window_skip_rate": (
                0.0
                if window_observations == 0
                else window_skipped_steps / window_observations
            ),
            "epoch_attempted_steps": self.epoch_attempted_steps,
            "epoch_skipped_steps": self.epoch_skipped_steps,
            "epoch_nonfinite_steps": self.epoch_nonfinite_steps,
            "epoch_skip_rate": (
                0.0
                if self.epoch_attempted_steps == 0
                else self.epoch_skipped_steps / self.epoch_attempted_steps
            ),
            "consecutive_skips": self.consecutive_skips,
            "max_observed_consecutive_skips": (self.max_observed_consecutive_skips),
        }

    def observe(
        self,
        *,
        skipped: bool,
        gradient_reason: str,
    ) -> GradientGuardFailFastDecision:
        if gradient_reason not in self._VALID_REASONS:
            raise ValueError(f"unknown gradient-guard reason: {gradient_reason}")
        expected_skipped = gradient_reason != "safe"
        if bool(skipped) != expected_skipped:
            raise ValueError(
                "gradient-guard skipped flag disagrees with gradient reason"
            )

        skipped = bool(skipped)
        self._window.append(skipped)
        self.epoch_attempted_steps += 1
        if skipped:
            self.epoch_skipped_steps += 1
            self.consecutive_skips += 1
            self.max_observed_consecutive_skips = max(
                self.max_observed_consecutive_skips,
                self.consecutive_skips,
            )
        else:
            self.consecutive_skips = 0
        if gradient_reason == "nonfinite":
            self.epoch_nonfinite_steps += 1

        audit = self.snapshot()
        reason = "safe"
        if gradient_reason == "nonfinite" and self.fail_on_nonfinite:
            reason = "nonfinite_gradient"
        elif self.consecutive_skips >= self.max_consecutive_skips:
            reason = "consecutive_gradient_skips"
        elif (
            skipped
            and audit["window_observations"] >= self.window_min_observations
            and audit["window_skip_rate"] > self.max_window_skip_rate
        ):
            reason = "window_gradient_skip_rate"
        elif (
            skipped
            and self.epoch_attempted_steps >= self.epoch_min_observations
            and audit["epoch_skip_rate"] > self.max_epoch_skip_rate
        ):
            reason = "epoch_gradient_skip_rate"
        return GradientGuardFailFastDecision(
            should_abort=reason != "safe",
            reason=reason,
            audit=audit,
        )


def validate_completed_epoch_guard_audit(
    audit: Mapping,
    *,
    expected_epoch: int,
    expected_attempted_steps: int,
    max_epoch_skip_rate: float,
) -> dict:
    """Recompute checkpoint eligibility from a completed-epoch guard audit."""
    required = {
        "epoch",
        "epoch_attempted_steps",
        "epoch_skipped_steps",
        "epoch_nonfinite_steps",
        "epoch_skip_rate",
        "fail_fast_triggered",
        "checkpoint_eligible",
    }
    missing = sorted(required.difference(audit))
    if missing:
        raise ValueError(f"completed epoch lacks gradient-guard fields: {missing}")
    if int(audit["epoch"]) != expected_epoch:
        raise ValueError("completed-epoch gradient-guard epoch changed")
    attempted = int(audit["epoch_attempted_steps"])
    skipped = int(audit["epoch_skipped_steps"])
    nonfinite = int(audit["epoch_nonfinite_steps"])
    if attempted != expected_attempted_steps:
        raise ValueError(
            "completed-epoch gradient-guard attempt count is not the "
            "complete epoch boundary"
        )
    if not 0 <= skipped <= attempted:
        raise ValueError("completed-epoch gradient skip count is invalid")
    if not 0 <= nonfinite <= skipped:
        raise ValueError("completed-epoch nonfinite gradient count is invalid")
    recomputed_rate = 0.0 if attempted == 0 else skipped / attempted
    recorded_rate = float(audit["epoch_skip_rate"])
    if not math.isclose(recorded_rate, recomputed_rate, abs_tol=1e-12):
        raise ValueError("completed-epoch gradient skip rate is inconsistent")
    if bool(audit["fail_fast_triggered"]):
        raise ValueError("fail-fast-triggered epoch cannot be checkpointed")
    if nonfinite:
        raise ValueError("epoch with nonfinite gradients cannot be checkpointed")
    if recomputed_rate > max_epoch_skip_rate:
        raise ValueError("completed epoch exceeds gradient skip-rate limit")
    if bool(audit["checkpoint_eligible"]) is not True:
        raise ValueError("completed epoch is not checkpoint eligible")
    return dict(audit)


def decide_gradient_step(
    gradient_norm: float,
    *,
    threshold: float,
) -> GradientGuardDecision:
    """Reject non-finite or implausibly large raw global gradients."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("gradient guard threshold must be finite and positive")
    norm = float(gradient_norm)
    if not math.isfinite(norm):
        return GradientGuardDecision(False, "nonfinite", norm, float(threshold))
    if norm > threshold:
        return GradientGuardDecision(False, "finite_spike", norm, float(threshold))
    return GradientGuardDecision(True, "safe", norm, float(threshold))


def top_gradient_diagnostics(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    limit: int = 8,
) -> list[dict]:
    """Return bounded per-parameter diagnostics without mutating gradients."""
    if limit <= 0:
        raise ValueError("gradient diagnostic limit must be positive")
    rows = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach()
        nonfinite = 0
        finite_count = 0
        abs_max = 0.0
        l2_norm = 0.0
        # The vocabulary mapping weight contains more than 150M elements.
        # Chunking avoids allocating a full-size finite mask and FP32 copy on a
        # GPU that may be shared with another job.
        for chunk in detached.reshape(-1).split(1_000_000):
            finite = torch.isfinite(chunk)
            count = int(finite.sum().item())
            finite_count += count
            nonfinite += chunk.numel() - count
            if count:
                values = chunk[finite].double()
                abs_max = max(abs_max, float(values.abs().max().item()))
                l2_norm = math.hypot(l2_norm, float(values.norm(2).item()))
        if finite_count == 0:
            abs_max = None
            l2_norm = None
        rows.append(
            {
                "name": name,
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "elements": detached.numel(),
                "nonfinite_elements": nonfinite,
                "finite_l2_norm": l2_norm,
                "finite_abs_max": abs_max,
            }
        )
    rows.sort(
        key=lambda item: (
            item["nonfinite_elements"] > 0,
            item["finite_l2_norm"] or -1.0,
        ),
        reverse=True,
    )
    return rows[:limit]
