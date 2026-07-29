"""Exact DDP gradient-accumulation planning and audit helpers for Phase II."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Iterator, Mapping, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class AccumulationPlan:
    world_size: int
    micro_batch_series_per_rank: int
    gradient_accumulation_steps: int
    micro_steps_per_epoch: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def full_global_batch_series(self) -> int:
        return (
            self.world_size
            * self.micro_batch_series_per_rank
            * self.gradient_accumulation_steps
        )

    @property
    def window_sizes(self) -> tuple[int, ...]:
        return accumulation_window_sizes(
            self.micro_steps_per_epoch,
            self.gradient_accumulation_steps,
        )

    @property
    def optimizer_attempts_per_epoch(self) -> int:
        return len(self.window_sizes)

    @property
    def final_window_micro_steps(self) -> int:
        return self.window_sizes[-1]

    @property
    def final_global_batch_series(self) -> int:
        return (
            self.world_size
            * self.micro_batch_series_per_rank
            * self.final_window_micro_steps
        )

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "full_global_batch_series": self.full_global_batch_series,
            "optimizer_attempts_per_epoch": self.optimizer_attempts_per_epoch,
            "full_windows_per_epoch": sum(
                size == self.gradient_accumulation_steps for size in self.window_sizes
            ),
            "partial_windows_per_epoch": sum(
                size != self.gradient_accumulation_steps for size in self.window_sizes
            ),
            "final_window_micro_steps": self.final_window_micro_steps,
            "final_global_batch_series": self.final_global_batch_series,
            "last_window_policy": "use_actual_global_valid_answer_rows",
        }


def accumulation_window_sizes(
    micro_steps: int,
    accumulation_steps: int,
) -> tuple[int, ...]:
    if not isinstance(micro_steps, int) or micro_steps <= 0:
        raise ValueError("micro_steps must be a positive integer")
    if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    full, remainder = divmod(micro_steps, accumulation_steps)
    values = [accumulation_steps] * full
    if remainder:
        values.append(remainder)
    return tuple(values)


def iter_accumulation_windows(
    iterable: Iterable[T],
    accumulation_steps: int,
) -> Iterator[list[T]]:
    if not isinstance(accumulation_steps, int) or accumulation_steps <= 0:
        raise ValueError("accumulation_steps must be a positive integer")
    window: list[T] = []
    for item in iterable:
        window.append(item)
        if len(window) == accumulation_steps:
            yield window
            window = []
    if window:
        yield window


def expected_optimizer_attempt_step(
    completed_epochs: int,
    plan: AccumulationPlan,
) -> int:
    if completed_epochs < 0:
        raise ValueError("completed_epochs cannot be negative")
    return completed_epochs * plan.optimizer_attempts_per_epoch


def expected_micro_step(
    completed_epochs: int,
    plan: AccumulationPlan,
) -> int:
    if completed_epochs < 0:
        raise ValueError("completed_epochs cannot be negative")
    return completed_epochs * plan.micro_steps_per_epoch


def ddp_window_loss_multiplier(
    *,
    world_size: int,
    global_valid_rows: float,
) -> float:
    """Scale local objective sums for DDP's rank-mean gradient reduction.

    Each microbatch in one accumulation window uses this same multiplier.  DDP
    averages the final synchronized gradient across ranks, so multiplying local
    sums by ``world_size / global_valid_rows`` yields the exact global valid-row
    mean over every rank and every microbatch in the window.
    """
    if not isinstance(world_size, int) or world_size <= 0:
        raise ValueError("world_size must be a positive integer")
    if not math.isfinite(global_valid_rows) or global_valid_rows <= 0:
        raise ValueError("global_valid_rows must be finite and positive")
    return world_size / float(global_valid_rows)


def validate_accumulated_objective_denominator(
    *,
    declared_global_valid_rows: float,
    observed_global_objective_count: float,
) -> None:
    if not math.isfinite(observed_global_objective_count):
        raise ValueError("accumulated objective denominator is nonfinite")
    if not math.isclose(
        float(declared_global_valid_rows),
        float(observed_global_objective_count),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "predeclared valid-row denominator differs from the Treatment "
            "objective_count"
        )


def local_scale_summary(values: Mapping[str, float]) -> dict:
    def rms(square_sum: str, count: str):
        denominator = float(values.get(count, 0.0))
        if denominator <= 0:
            return None
        numerator = float(values.get(square_sum, 0.0))
        if numerator < 0 or not math.isfinite(numerator):
            raise ValueError("local-input square sum is invalid")
        return math.sqrt(numerator / denominator)

    return {
        "raw_rms": rms(
            "local_input_raw_square_sum",
            "local_input_raw_element_count",
        ),
        "raw_abs_max": values.get("local_input_raw_abs_max"),
        "normalized_rms": rms(
            "local_input_normalized_square_sum",
            "local_input_normalized_element_count",
        ),
        "normalized_abs_max": values.get("local_input_normalized_abs_max"),
        "raw_element_count": values.get("local_input_raw_element_count", 0.0),
        "normalized_element_count": values.get(
            "local_input_normalized_element_count", 0.0
        ),
    }


def new_window_numerical_totals() -> dict[str, float]:
    return {
        "optimizer_attempts": 0,
        "optimizer_steps_applied": 0,
        "micro_steps": 0,
        "full_accumulation_windows": 0,
        "partial_accumulation_windows": 0,
        "gradient_norm_count": 0,
        "gradient_norm_sum": 0.0,
        "gradient_norm_max": 0.0,
        "gradient_norm_last": 0.0,
        "local_word_proj_gradient_norm_sum": 0.0,
        "local_word_proj_gradient_norm_max": 0.0,
        "local_word_proj_gradient_norm_last": 0.0,
        "clipped_windows": 0,
        "clip_coefficient_count": 0,
        "clip_coefficient_sum": 0.0,
        "clip_coefficient_min": 1.0,
        "clip_coefficient_last": 1.0,
        "skipped_windows": 0,
        "finite_spike_windows": 0,
        "nonfinite_windows": 0,
    }


def summarize_window_numerics(values: Mapping[str, float]) -> dict:
    count = int(values["gradient_norm_count"])
    clip_count = int(values["clip_coefficient_count"])
    attempts = int(values["optimizer_attempts"])
    return {
        "optimizer_attempts": attempts,
        "gradient_norm_count": count,
        "clip_coefficient_count": clip_count,
        "optimizer_steps_applied": int(values["optimizer_steps_applied"]),
        "micro_steps": int(values["micro_steps"]),
        "full_accumulation_windows": int(values["full_accumulation_windows"]),
        "partial_accumulation_windows": int(values["partial_accumulation_windows"]),
        "gradient_l2_norm_mean": (
            None if count == 0 else values["gradient_norm_sum"] / count
        ),
        "gradient_l2_norm_max": None if count == 0 else values["gradient_norm_max"],
        "gradient_l2_norm_last": None if count == 0 else values["gradient_norm_last"],
        "local_word_proj_gradient_l2_norm_mean": (
            None if count == 0 else values["local_word_proj_gradient_norm_sum"] / count
        ),
        "local_word_proj_gradient_l2_norm_max": (
            None if count == 0 else values["local_word_proj_gradient_norm_max"]
        ),
        "local_word_proj_gradient_l2_norm_last": (
            None if count == 0 else values["local_word_proj_gradient_norm_last"]
        ),
        "clipped_windows": int(values["clipped_windows"]),
        "clip_rate": (None if attempts == 0 else values["clipped_windows"] / attempts),
        "clip_coefficient_mean": (
            None if clip_count == 0 else values["clip_coefficient_sum"] / clip_count
        ),
        "clip_coefficient_min": (
            None if clip_count == 0 else values["clip_coefficient_min"]
        ),
        "clip_coefficient_last": (
            None if clip_count == 0 else values["clip_coefficient_last"]
        ),
        "skipped_windows": int(values["skipped_windows"]),
        "skip_rate": None if attempts == 0 else values["skipped_windows"] / attempts,
        "finite_spike_windows": int(values["finite_spike_windows"]),
        "nonfinite_windows": int(values["nonfinite_windows"]),
    }


def add_numeric_totals(target: dict[str, float], source: Mapping[str, float]) -> None:
    """Merge one numeric counter set without touching target-only metadata."""
    missing = sorted(set(source).difference(target))
    if missing:
        raise KeyError(f"numeric total target is missing keys: {missing}")
    for key, value in source.items():
        if key.endswith("_max"):
            target[key] = max(float(target[key]), float(value))
        elif key.endswith("_min"):
            target[key] = min(float(target[key]), float(value))
        elif key.endswith("_last"):
            target[key] = float(value)
        else:
            target[key] += float(value)
