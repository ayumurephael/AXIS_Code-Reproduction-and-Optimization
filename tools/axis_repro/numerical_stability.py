"""Numerical-stability helpers shared by formal AXIS training jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Tuple

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
