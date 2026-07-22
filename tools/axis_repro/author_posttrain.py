"""Deterministic schedules for matched author-checkpoint post-training."""
from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence

import torch
from torch.utils.data import Sampler


COMPONENT_WEIGHTS = {"mc": 4, "tf": 3, "oe": 1}
COMPONENT_ORDER = tuple(COMPONENT_WEIGHTS)


class FixedIndexSampler(Sampler[int]):
    """Sampler that yields a pre-audited finite sequence of dataset positions."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(int(index) for index in indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _stable_seed(seed: int, *parts: object) -> int:
    text = ":".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def stratified_component_schedule(steps: int, *, seed: int, epoch: int = 1) -> list[str]:
    """Return an exactly 4:3:1 MC/TF/OE schedule.

    ``steps`` must be divisible by eight so that each DDP rank executes the
    same component and all three global exposure counts are exact.
    """
    total_weight = sum(COMPONENT_WEIGHTS.values())
    if steps <= 0 or steps % total_weight:
        raise ValueError(f"steps must be a positive multiple of {total_weight}")
    schedule = [
        component
        for component, weight in COMPONENT_WEIGHTS.items()
        for _ in range(steps * weight // total_weight)
    ]
    random.Random(_stable_seed(seed, "component", epoch)).shuffle(schedule)
    if Counter(schedule) != {
        component: steps * weight // total_weight
        for component, weight in COMPONENT_WEIGHTS.items()
    }:
        raise RuntimeError("stratified component schedule count mismatch")
    return schedule


def distributed_cyclic_indices(
    pool: Sequence[int],
    *,
    local_count: int,
    world_size: int,
    rank: int,
    seed: int,
    component: str,
    epoch: int = 1,
) -> list[int]:
    """Build a rank shard from globally shuffled, no-replacement cycles."""
    if not pool:
        raise ValueError(f"{component} pool is empty")
    if local_count < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed sampler dimensions")
    needed = local_count * world_size
    global_indices: list[int] = []
    cycle = 0
    while len(global_indices) < needed:
        order = list(map(int, pool))
        random.Random(_stable_seed(seed, component, epoch, cycle)).shuffle(order)
        global_indices.extend(order)
        cycle += 1
    global_indices = global_indices[:needed]
    local = global_indices[rank::world_size]
    if len(local) != local_count:
        raise RuntimeError("distributed cyclic sampler produced the wrong local count")
    return local


def cosine_warmup_multiplier(
    step: int,
    total_steps: int,
    *,
    warmup_ratio: float = 0.05,
    final_ratio: float = 0.10,
) -> float:
    if step < 0 or total_steps <= 0:
        raise ValueError("invalid LR schedule dimensions")
    if not 0.0 <= warmup_ratio < 1.0 or not 0.0 <= final_ratio <= 1.0:
        raise ValueError("invalid LR schedule ratios")
    warmup_steps = max(1, int(math.ceil(total_steps * warmup_ratio)))
    if step < warmup_steps:
        return float((step + 1) / warmup_steps)
    denominator = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, (step - warmup_steps) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    value = final_ratio + (1.0 - final_ratio) * cosine
    return float(min(1.0, max(final_ratio, value)))


def gradient_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(parameter.grad.detach().float().square().sum())
    if not squares:
        raise RuntimeError("no gradients found for norm calibration")
    return torch.stack(squares).sum().sqrt()


def calibrated_beta(
    answer_norms: Sequence[float],
    auxiliary_norms: Sequence[float],
    *,
    target_ratio: float,
    lower: float,
    upper: float,
) -> float:
    if not answer_norms or len(answer_norms) != len(auxiliary_norms):
        raise ValueError("gradient calibration samples are incomplete")
    if not 0.0 < target_ratio or not 0.0 <= lower <= upper:
        raise ValueError("invalid beta calibration bounds")
    answer = float(torch.tensor(answer_norms, dtype=torch.float64).median())
    auxiliary = float(torch.tensor(auxiliary_norms, dtype=torch.float64).median())
    if not math.isfinite(answer) or not math.isfinite(auxiliary) or auxiliary <= 0.0:
        raise ValueError("non-finite or zero calibration gradient")
    return float(min(upper, max(lower, target_ratio * answer / auxiliary)))
