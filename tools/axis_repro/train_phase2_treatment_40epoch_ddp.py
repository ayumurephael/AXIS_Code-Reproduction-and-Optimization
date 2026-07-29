"""Full Phase-II retraining with the loss_e2e_0723 Treatment objective.

This is a new experiment, not a continuation of the author's released
Phase-II checkpoint.  It loads only the repository-recorded Phase-I time-series
checkpoint, initializes a fresh Hint Tuner/Perceiver, caches F0 at step zero,
and trains for exactly 40 epochs. Five ranks accumulate 26 microbatches per
optimizer attempt (global batch 130); the final short window uses its actual
global valid-row denominator. The local-input normalization arm is explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.AXIS import (
    DEFAULT_LOCAL_INPUT_RMSNORM_EPS,
    LOCAL_INPUT_NORM_MODES,
    LOCAL_INPUT_NORM_NONE,
)
from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset

from .loss_e2e import ERROR_ANSWER
from .loss_e2e_40epoch_runtime import (
    LOCAL_INPUT_MAX_METRIC_KEYS,
    LOCAL_INPUT_SUM_METRIC_KEYS,
    Phase2TreatmentObjectiveModel,
    fixed_hint_checkpoint_state,
    initialize_fixed_hint_reference,
    install_fixed_hint_runtime,
    load_phase2_40epoch_checkpoint,
)
from .model_utils import (
    build_model,
    freeze_for_phase2,
    load_phase1_fresh_hint,
    sha256_file,
)
from .numerical_stability import (
    GradientGuardFailFast,
    decide_gradient_step,
    top_gradient_diagnostics,
    validate_completed_epoch_guard_audit,
)
from .phase2_accumulation import (
    AccumulationPlan,
    add_numeric_totals,
    ddp_window_loss_multiplier,
    expected_micro_step,
    expected_optimizer_attempt_step,
    iter_accumulation_windows,
    local_scale_summary,
    new_window_numerical_totals,
    summarize_window_numerics,
    validate_accumulated_objective_denominator,
)
from .prompt_boundary import (
    SIMPLIFIED_FINAL_ANSWER_V1,
    answer_continuations,
    build_simplified_prompt,
    install_simplified_prompt_runtime,
)
from .train_loss_e2e_ddp import (
    METRIC_KEYS,
    _ensure_frozen_llm_storage_dtype,
    _gradient_l2_norm,
    _parameter_snapshot,
    _relative_parameter_change,
    _summary,
    _tensor_sha256,
    build_data_audit,
    save_atomic,
)


EXPERIMENT = "loss_e2e_0723_phase2_40epoch_accumulation_v2"
FORMAL_PROTOCOL_VERSION = 2
FORMAL_EPOCHS = 40
FORMAL_WORLD_SIZE = 5
FORMAL_SEED = 72
FORMAL_TRAIN_RATIO = 0.95
FORMAL_TRAIN_SERIES = 28_500
FORMAL_VALIDATION_SERIES = 1_500
FORMAL_MICRO_BATCH_SERIES_PER_RANK = 1
FORMAL_GRADIENT_ACCUMULATION_STEPS = 26
FORMAL_MICRO_STEPS_PER_EPOCH = 5_700
FORMAL_STEPS_PER_EPOCH = 220
FORMAL_TOTAL_MICRO_STEPS = FORMAL_MICRO_STEPS_PER_EPOCH * FORMAL_EPOCHS
FORMAL_TOTAL_STEPS = FORMAL_STEPS_PER_EPOCH * FORMAL_EPOCHS
FORMAL_LR = 1e-4
FORMAL_WEIGHT_DECAY = 1e-5
FORMAL_ALPHA = 0.40
FORMAL_MAX_GRAD_NORM = 1.0
FORMAL_GRADIENT_SKIP_THRESHOLD = 100.0
FORMAL_FINITE_SPIKE_ABORT_COUNT = 2
FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS = 2
FORMAL_GRADIENT_GUARD_MAX_RECORDED_EVENTS = 16
FORMAL_GRADIENT_SKIP_WINDOW_SIZE = 100
FORMAL_GRADIENT_SKIP_WINDOW_MIN_OBSERVATIONS = 20
FORMAL_MAX_WINDOW_GRADIENT_SKIP_RATE = 0.05
FORMAL_GRADIENT_SKIP_EPOCH_MIN_OBSERVATIONS = 100
FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE = 0.01
FROZEN_LLM_DTYPE = torch.bfloat16

FORMAL_ACCUMULATION_PLAN = AccumulationPlan(
    world_size=FORMAL_WORLD_SIZE,
    micro_batch_series_per_rank=FORMAL_MICRO_BATCH_SERIES_PER_RANK,
    gradient_accumulation_steps=FORMAL_GRADIENT_ACCUMULATION_STEPS,
    micro_steps_per_epoch=FORMAL_MICRO_STEPS_PER_EPOCH,
)
if FORMAL_ACCUMULATION_PLAN.optimizer_attempts_per_epoch != FORMAL_STEPS_PER_EPOCH:
    raise RuntimeError("formal accumulation plan has the wrong optimizer-step count")


def expected_global_step_for_epoch(epoch: int) -> int:
    """Return optimizer-attempt windows consumed at an epoch boundary."""
    if not 0 <= epoch <= FORMAL_EPOCHS:
        raise ValueError(f"epoch must be in [0, {FORMAL_EPOCHS}]")
    return expected_optimizer_attempt_step(epoch, FORMAL_ACCUMULATION_PLAN)


def expected_micro_step_for_epoch(epoch: int) -> int:
    if not 0 <= epoch <= FORMAL_EPOCHS:
        raise ValueError(f"epoch must be in [0, {FORMAL_EPOCHS}]")
    return expected_micro_step(epoch, FORMAL_ACCUMULATION_PLAN)


def world_size_for_completed_epoch(epoch: int) -> int:
    if not 1 <= epoch <= FORMAL_EPOCHS:
        raise ValueError(f"epoch must be in [1, {FORMAL_EPOCHS}]")
    return FORMAL_WORLD_SIZE


def steps_per_epoch_for_completed_epoch(epoch: int) -> int:
    world_size_for_completed_epoch(epoch)
    return FORMAL_STEPS_PER_EPOCH


def micro_steps_per_epoch_for_completed_epoch(epoch: int) -> int:
    world_size_for_completed_epoch(epoch)
    return FORMAL_MICRO_STEPS_PER_EPOCH


def planned_steps_for_completed_epoch(epoch: int) -> int:
    world_size_for_completed_epoch(epoch)
    return FORMAL_TOTAL_STEPS


def learning_rate_for_epoch(epoch: int, *, initial_lr: float) -> float:
    """The confirmed author-compatible schedule is constant for all epochs."""
    if not 1 <= epoch <= FORMAL_EPOCHS:
        raise ValueError(f"epoch must be in [1, {FORMAL_EPOCHS}]")
    return float(initial_lr)


def local_input_normalization_policy(mode: str, eps: float) -> dict:
    if mode not in LOCAL_INPUT_NORM_MODES:
        raise ValueError(f"unsupported local input normalization mode: {mode!r}")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("local input RMSNorm eps must be finite and positive")
    return {
        "mode": mode,
        "affine": False,
        "eps": float(eps),
        "axis": "per_local_timestep_last_dimension",
        "compute_dtype": "torch.float32",
        "placement": "immediately_before_local_word_proj",
        "fixed_hint_affected": False,
    }


def formal_gradient_guard_policy() -> dict:
    """Return the immutable fail-fast policy recorded in formal checkpoints."""
    return {
        "unit": "optimizer_attempt_after_accumulation",
        "window_size": FORMAL_GRADIENT_SKIP_WINDOW_SIZE,
        "window_min_observations": FORMAL_GRADIENT_SKIP_WINDOW_MIN_OBSERVATIONS,
        "max_window_skip_rate": FORMAL_MAX_WINDOW_GRADIENT_SKIP_RATE,
        "epoch_min_observations": FORMAL_GRADIENT_SKIP_EPOCH_MIN_OBSERVATIONS,
        "max_epoch_skip_rate": FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE,
        "max_consecutive_skips": FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
        "finite_spike_abort_count": FORMAL_FINITE_SPIKE_ABORT_COUNT,
        "fail_on_nonfinite": True,
    }


def new_formal_gradient_guard_tracker() -> GradientGuardFailFast:
    return GradientGuardFailFast(
        window_size=FORMAL_GRADIENT_SKIP_WINDOW_SIZE,
        window_min_observations=FORMAL_GRADIENT_SKIP_WINDOW_MIN_OBSERVATIONS,
        max_window_skip_rate=FORMAL_MAX_WINDOW_GRADIENT_SKIP_RATE,
        epoch_min_observations=FORMAL_GRADIENT_SKIP_EPOCH_MIN_OBSERVATIONS,
        max_epoch_skip_rate=FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE,
        max_consecutive_skips=FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
        fail_on_nonfinite=True,
    )


def validate_formal_completed_epoch_guard(meta: Mapping, epoch: int) -> dict:
    stability = meta.get("numerical_stability", {})
    if stability.get("fail_fast_policy") != formal_gradient_guard_policy():
        raise ValueError("checkpoint gradient-guard fail-fast policy changed")
    guard = meta.get("optimization_diagnostics", {}).get("gradient_guard", {})
    audit = guard.get("last_completed_epoch", {})
    return validate_completed_epoch_guard_audit(
        audit,
        expected_epoch=epoch,
        expected_attempted_steps=FORMAL_STEPS_PER_EPOCH,
        max_epoch_skip_rate=FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE,
    )


def _completed_epoch_guard_audit(
    tracker: GradientGuardFailFast,
    *,
    epoch: int,
    fail_fast_triggered: bool,
) -> dict:
    audit = tracker.snapshot()
    audit.update(
        {
            "epoch": epoch,
            "unit": "optimizer_attempt_after_accumulation",
            "fail_fast_triggered": bool(fail_fast_triggered),
            "checkpoint_eligible": (
                not fail_fast_triggered
                and audit["epoch_nonfinite_steps"] == 0
                and audit["epoch_skipped_steps"] < FORMAL_FINITE_SPIKE_ABORT_COUNT
                and audit["epoch_skip_rate"] <= FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE
            ),
        }
    )
    return audit


def _new_optimization_state(max_grad_norm: float) -> dict:
    state = new_window_numerical_totals()
    state.update(
        {
            "max_grad_norm": max_grad_norm,
            "parameter_relative_change_by_step": {},
            "gradient_guard_threshold": FORMAL_GRADIENT_SKIP_THRESHOLD,
            "gradient_guard_skipped_steps": 0,
            "gradient_guard_finite_spike_skipped_steps": 0,
            "gradient_guard_nonfinite_skipped_steps": 0,
            "gradient_guard_consecutive_skips": 0,
            "gradient_guard_max_consecutive_skips": 0,
            "gradient_guard_consecutive_skip_limit": (
                FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS
            ),
            "gradient_guard_last_completed_epoch": None,
            "gradient_guard_events": [],
        }
    )
    return state


def _guarded_optimization_summary(state: Mapping) -> dict:
    result = summarize_window_numerics(state)
    result.update(
        {
            "max_grad_norm": state["max_grad_norm"],
            "parameter_relative_change_by_step": dict(
                state["parameter_relative_change_by_step"]
            ),
            "gradient_guard": {
                "threshold": state["gradient_guard_threshold"],
                "skipped_steps": state["gradient_guard_skipped_steps"],
                "finite_spike_skipped_steps": state[
                    "gradient_guard_finite_spike_skipped_steps"
                ],
                "nonfinite_skipped_steps": state[
                    "gradient_guard_nonfinite_skipped_steps"
                ],
                "consecutive_skips": state["gradient_guard_consecutive_skips"],
                "max_consecutive_skips": state["gradient_guard_max_consecutive_skips"],
                "consecutive_skip_limit": state[
                    "gradient_guard_consecutive_skip_limit"
                ],
                "finite_spike_abort_count": FORMAL_FINITE_SPIKE_ABORT_COUNT,
                "last_completed_epoch": state.get(
                    "gradient_guard_last_completed_epoch"
                ),
                "events": list(state["gradient_guard_events"]),
            },
        }
    )
    return result


def _restore_optimization_state(summary: Mapping, max_grad_norm: float) -> dict:
    state = _new_optimization_state(max_grad_norm)
    direct = (
        "optimizer_attempts",
        "optimizer_steps_applied",
        "micro_steps",
        "full_accumulation_windows",
        "partial_accumulation_windows",
        "clipped_windows",
        "skipped_windows",
        "finite_spike_windows",
        "nonfinite_windows",
    )
    for key in direct:
        state[key] = int(summary[key])
    count = int(summary["gradient_norm_count"])
    state["gradient_norm_count"] = count
    for prefix, summary_prefix in (
        ("gradient_norm", "gradient_l2_norm"),
        ("local_word_proj_gradient_norm", "local_word_proj_gradient_l2_norm"),
    ):
        mean = summary[f"{summary_prefix}_mean"]
        state[f"{prefix}_sum"] = 0.0 if mean is None else float(mean) * count
        state[f"{prefix}_max"] = float(summary[f"{summary_prefix}_max"] or 0.0)
        state[f"{prefix}_last"] = float(summary[f"{summary_prefix}_last"] or 0.0)
    clip_count = int(summary["clip_coefficient_count"])
    state["clip_coefficient_count"] = clip_count
    clip_mean = summary.get("clip_coefficient_mean")
    state["clip_coefficient_sum"] = (
        0.0 if clip_mean is None else float(clip_mean) * clip_count
    )
    state["clip_coefficient_min"] = float(summary.get("clip_coefficient_min") or 1.0)
    state["clip_coefficient_last"] = float(summary.get("clip_coefficient_last") or 1.0)
    state["parameter_relative_change_by_step"] = dict(
        summary.get("parameter_relative_change_by_step", {})
    )
    guard = summary.get("gradient_guard", {})
    state["gradient_guard_skipped_steps"] = int(guard.get("skipped_steps", 0))
    state["gradient_guard_finite_spike_skipped_steps"] = int(
        guard.get("finite_spike_skipped_steps", 0)
    )
    state["gradient_guard_nonfinite_skipped_steps"] = int(
        guard.get("nonfinite_skipped_steps", 0)
    )
    state["gradient_guard_consecutive_skips"] = int(guard.get("consecutive_skips", 0))
    state["gradient_guard_max_consecutive_skips"] = int(
        guard.get("max_consecutive_skips", 0)
    )
    state["gradient_guard_last_completed_epoch"] = guard.get("last_completed_epoch")
    state["gradient_guard_events"] = list(guard.get("events", []))[
        :FORMAL_GRADIENT_GUARD_MAX_RECORDED_EVENTS
    ]
    return state


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_training_split_manifest(
    manifest: Mapping,
    dataset_series: Sequence[str],
) -> dict:
    if manifest.get("seed") != FORMAL_SEED:
        raise ValueError("training manifest seed must be exactly 72")
    if manifest.get("train_ratio") != FORMAL_TRAIN_RATIO:
        raise ValueError("training manifest train_ratio must be exactly 0.95")
    train = manifest.get("train_series")
    validation = manifest.get("val_series")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("manifest must contain train_series and val_series")
    if len(train) != FORMAL_TRAIN_SERIES:
        raise ValueError(f"expected {FORMAL_TRAIN_SERIES} training series")
    if len(validation) != FORMAL_VALIDATION_SERIES:
        raise ValueError(f"expected {FORMAL_VALIDATION_SERIES} validation series")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("split manifest contains duplicate series")
    if set(train).intersection(validation):
        raise ValueError("training and validation series overlap")
    if list(dataset_series) != train:
        raise ValueError(
            "dataset training series/order differs from the frozen seed-72 manifest"
        )
    return {
        "seed": FORMAL_SEED,
        "train_ratio": FORMAL_TRAIN_RATIO,
        "train_series": len(train),
        "validation_series": len(validation),
        "ordered_train_series_sha256": hashlib.sha256(
            "\n".join(train).encode("utf-8")
        ).hexdigest(),
        "ordered_validation_series_sha256": hashlib.sha256(
            "\n".join(validation).encode("utf-8")
        ).hexdigest(),
    }


def build_prompt_boundary_audit(
    dataset: AXISAnomalyQADataset,
    tokenizer,
    *,
    num_fixed_hint_tokens: int,
    batch_size: int = 128,
) -> dict:
    """Audit every train QA without truncation before the first update."""
    lengths: list[int] = []
    prefix_lengths: list[int] = []
    answer_lengths: list[int] = []
    over_limit: list[dict] = []
    non_additive: list[dict] = []
    leading_whitespace: list[dict] = []
    terminal_suffix_violations: list[dict] = []
    prompts: list[str] = []
    continuations: list[str] = []
    identifiers: list[tuple[str, int]] = []

    def consume() -> None:
        if not prompts:
            return
        prefix_ids = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        continuation_ids = tokenizer(
            continuations,
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        joint_ids = tokenizer(
            [
                prompt + continuation
                for prompt, continuation in zip(prompts, continuations)
            ],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )["input_ids"]
        for prefix, continuation, joint, identifier in zip(
            prefix_ids,
            continuation_ids,
            joint_ids,
            identifiers,
        ):
            prefix_length = len(prefix)
            answer_length = len(continuation)
            total = prefix_length + answer_length + 1
            prefix_lengths.append(prefix_length)
            answer_lengths.append(answer_length)
            lengths.append(total)
            item = {
                "series_file": identifier[0],
                "window_index": identifier[1],
                "prefix_tokens": prefix_length,
                "answer_tokens": answer_length,
                "total_tokens_with_terminal_eos": total,
            }
            if list(prefix) + list(continuation) != list(joint):
                non_additive.append(item)
            if total > int(tokenizer.model_max_length):
                over_limit.append(item)
        prompts.clear()
        continuations.clear()
        identifiers.clear()

    rows_total = 0
    rows_included = 0
    for series_path in dataset.series_files:
        record = json.loads(series_path.read_text(encoding="utf-8"))
        time_series = torch.tensor(
            record["original_data"]["time_series"],
            dtype=torch.float32,
        )
        for window_index, item in enumerate(record["windows"]):
            rows_total += 1
            answer = str(item["answer"])
            if answer.strip() != ERROR_ANSWER:
                rows_included += 1
            if answer[:1].isspace():
                leading_whitespace.append(
                    {
                        "series_file": series_path.name,
                        "window_index": window_index,
                    }
                )
            window = item["window_range"]
            prompt = build_simplified_prompt(
                question=str(item["question"]),
                time_series=time_series,
                start_index=int(window["start"]),
                end_index=int(window["end"]),
                num_local_hint_tokens=int(window["end"]) - int(window["start"]),
                num_fixed_hint_tokens=num_fixed_hint_tokens,
            )
            if not prompt.endswith("### Final Answer\nAnswer:"):
                terminal_suffix_violations.append(
                    {
                        "series_file": series_path.name,
                        "window_index": window_index,
                    }
                )
            prompts.append(prompt)
            continuations.extend(answer_continuations([answer]))
            identifiers.append((series_path.name, window_index))
            if len(prompts) >= batch_size:
                consume()
    consume()
    if non_additive:
        raise RuntimeError("non-additive prompt/answer tokenization detected")
    if over_limit:
        raise RuntimeError("untruncated prompt/answer exceeds model context")
    if leading_whitespace:
        raise RuntimeError(
            "gold answers with leading whitespace require a new protocol"
        )
    if terminal_suffix_violations:
        raise RuntimeError("prompt does not end at the approved Answer: boundary")
    return {
        "schema_version": 1,
        "protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "rows_total": rows_total,
        "rows_included": rows_included,
        "add_special_tokens": False,
        "truncation": False,
        "intermediate_eos_count": 0,
        "terminal_eos_supervised": True,
        "answer_target_prefix": "single ASCII space",
        "train_inference_prefix_equality": "enforced by shared runtime method",
        "non_additive_rows": 0,
        "over_context_rows": 0,
        "leading_whitespace_rows": 0,
        "terminal_suffix_violations": 0,
        "tokenizer_model_max_length": int(tokenizer.model_max_length),
        "max_prefix_tokens": max(prefix_lengths),
        "max_answer_tokens": max(answer_lengths),
        "max_total_tokens_with_terminal_eos": max(lengths),
        "mean_total_tokens_with_terminal_eos": sum(lengths) / len(lengths),
    }


def _checkpoint_payload(
    *,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    micro_step: int,
    optimizer_step: int,
    meta: dict,
    cumulative: Mapping[str, float],
    successful: Mapping[str, float],
    cumulative_local_scale: Mapping[str, float],
    last_completed_epoch_metrics: Mapping,
) -> dict:
    reference = fixed_hint_checkpoint_state(model.axis)
    if reference is None:
        raise RuntimeError("Treatment checkpoint cannot be saved without F0")
    return {
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "epoch": epoch,
        # global_step is intentionally the deterministic optimizer-attempt
        # window index. Unsafe windows remain attempts but never update state.
        "global_step": global_step,
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "model_state_dict": {
            "ts_pretrain_model": model.ts_pretrain_model.state_dict(),
            "moirai_trainable": model.axis.perceiver.state_dict(),
            "fixed_hint_reference": reference,
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "reproduction_meta": meta,
        "cumulative_training_metrics": dict(cumulative),
        "successful_training_metrics": dict(successful),
        "cumulative_local_input_metrics": dict(cumulative_local_scale),
        "last_completed_epoch_training_metrics": dict(last_completed_epoch_metrics),
    }


def validate_checkpoint_step_counters(payload: Mapping) -> dict:
    """Cross-check persisted micro/attempt/update counters and guard totals."""
    summary = payload.get("reproduction_meta", {}).get("optimization_diagnostics", {})
    guard = summary.get("gradient_guard", {})
    expected = {
        "optimizer_attempts": int(payload.get("global_step", -1)),
        "optimizer_steps_applied": int(payload.get("optimizer_step", -1)),
        "micro_steps": int(payload.get("micro_step", -1)),
        "skipped_windows": int(guard.get("skipped_steps", -1)),
        "finite_spike_windows": int(guard.get("finite_spike_skipped_steps", -1)),
        "nonfinite_windows": int(guard.get("nonfinite_skipped_steps", -1)),
    }
    for key, value in expected.items():
        if int(summary.get(key, -2)) != value:
            raise ValueError(f"checkpoint counter mismatch for {key}")
    attempts = expected["optimizer_attempts"]
    applied = expected["optimizer_steps_applied"]
    skipped = expected["skipped_windows"]
    if applied + skipped != attempts:
        raise ValueError("applied plus skipped windows differs from attempts")
    if int(summary.get("clip_coefficient_count", -1)) != applied:
        raise ValueError("clip coefficient count differs from applied updates")
    if (
        int(summary.get("full_accumulation_windows", -1))
        + int(summary.get("partial_accumulation_windows", -1))
        != attempts
    ):
        raise ValueError("full plus partial accumulation windows differs from attempts")
    if int(summary.get("gradient_norm_count", -1)) != attempts:
        raise ValueError("checkpoint lacks a finite gradient norm for every attempt")
    return expected


def validate_resume_checkpoint(
    payload: Mapping,
    *,
    split_manifest_sha256: str,
    data_audit_sha256: str,
    phase1_sha256: str,
    local_input_norm_mode: str,
    local_input_rmsnorm_eps: float,
) -> dict:
    required = {
        "schema_version",
        "epoch",
        "global_step",
        "micro_step",
        "optimizer_step",
        "model_state_dict",
        "optimizer_state_dict",
        "reproduction_meta",
        "cumulative_training_metrics",
        "successful_training_metrics",
        "cumulative_local_input_metrics",
        "last_completed_epoch_training_metrics",
        "rng_state_by_rank",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {missing}")
    if int(payload["schema_version"]) != FORMAL_PROTOCOL_VERSION:
        raise ValueError("resume checkpoint predates the accumulation-v2 protocol")
    epoch = int(payload["epoch"])
    if not 1 <= epoch < FORMAL_EPOCHS:
        raise ValueError("resume epoch must be a completed epoch in [1, 39]")
    if int(payload["global_step"]) != expected_global_step_for_epoch(epoch):
        raise ValueError("resume optimizer-attempt step is not an epoch boundary")
    if int(payload["micro_step"]) != expected_micro_step_for_epoch(epoch):
        raise ValueError("resume micro-step is not an epoch boundary")

    meta = payload["reproduction_meta"]
    expected = {
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "run_purpose": "formal",
        "objective": "treatment",
        "world_size": FORMAL_WORLD_SIZE,
        "epochs": FORMAL_EPOCHS,
        "micro_steps_per_rank_epoch": FORMAL_MICRO_STEPS_PER_EPOCH,
        "optimizer_attempts_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "planned_steps": FORMAL_TOTAL_STEPS,
        "planned_micro_steps": FORMAL_TOTAL_MICRO_STEPS,
        "seed": FORMAL_SEED,
        "segment_alpha": FORMAL_ALPHA,
        "lr": FORMAL_LR,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "max_grad_norm": FORMAL_MAX_GRAD_NORM,
        "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
        "global_batch_series": (FORMAL_ACCUMULATION_PLAN.full_global_batch_series),
        "prompt_protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "split_manifest_sha256": split_manifest_sha256,
        "data_audit_sha256": data_audit_sha256,
        "phase1_checkpoint_sha256": phase1_sha256,
        "lr_schedule": {"epochs_1_40": FORMAL_LR},
        "local_input_normalization": local_input_normalization_policy(
            local_input_norm_mode,
            local_input_rmsnorm_eps,
        ),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"resume checkpoint identity mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    if meta.get("source_dirty") is not False:
        raise ValueError("resume checkpoint was produced from a dirty source tree")
    if meta.get("gradient_accumulation", {}).get("plan") != (
        FORMAL_ACCUMULATION_PLAN.to_dict()
    ):
        raise ValueError("resume checkpoint accumulation plan changed")
    stability = meta.get("numerical_stability", {})
    if stability.get("gradient_skip_threshold") != FORMAL_GRADIENT_SKIP_THRESHOLD:
        raise ValueError("resume checkpoint gradient-guard threshold changed")
    if stability.get("unsafe_updates_applied") is not False:
        raise ValueError("resume checkpoint does not prove fail-closed updates")
    validate_formal_completed_epoch_guard(meta, epoch)
    guard = meta["optimization_diagnostics"]["gradient_guard"]
    validate_checkpoint_step_counters(payload)
    expected_optimizer_steps = int(payload["global_step"]) - int(guard["skipped_steps"])
    if int(payload["optimizer_step"]) != expected_optimizer_steps:
        raise ValueError("resume optimizer-step count disagrees with skipped windows")
    rank_rng = payload["rng_state_by_rank"]
    if not isinstance(rank_rng, list) or len(rank_rng) != FORMAL_WORLD_SIZE:
        raise ValueError("resume checkpoint lacks one RNG state per DDP rank")
    if "fixed_hint_reference" not in payload["model_state_dict"]:
        raise ValueError("resume Treatment checkpoint has no cached F0")
    for key in ("cumulative_training_metrics", "successful_training_metrics"):
        if set(METRIC_KEYS).difference(payload[key]):
            raise ValueError(f"resume checkpoint lacks {key}")
    scale_keys = set(LOCAL_INPUT_SUM_METRIC_KEYS + LOCAL_INPUT_MAX_METRIC_KEYS)
    if scale_keys.difference(payload["cumulative_local_input_metrics"]):
        raise ValueError("resume checkpoint lacks local-input scale metrics")
    return dict(meta)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--phase1", required=True)
    result.add_argument("--data", required=True)
    result.add_argument("--series-split-manifest", required=True)
    result.add_argument("--output", required=True)
    result.add_argument(
        "--run-purpose",
        choices=["formal", "smoke"],
        default="formal",
    )
    result.add_argument("--epochs", type=int, default=FORMAL_EPOCHS)
    result.add_argument("--seed", type=int, default=FORMAL_SEED)
    result.add_argument("--lr", type=float, default=FORMAL_LR)
    result.add_argument("--weight-decay", type=float, default=FORMAL_WEIGHT_DECAY)
    result.add_argument("--alpha", type=float, default=FORMAL_ALPHA)
    result.add_argument("--max-grad-norm", type=float, default=FORMAL_MAX_GRAD_NORM)
    result.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=FORMAL_GRADIENT_ACCUMULATION_STEPS,
    )
    result.add_argument(
        "--local-input-norm",
        choices=sorted(LOCAL_INPUT_NORM_MODES),
        default=LOCAL_INPUT_NORM_NONE,
    )
    result.add_argument(
        "--local-input-rmsnorm-eps",
        type=float,
        default=DEFAULT_LOCAL_INPUT_RMSNORM_EPS,
    )
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--loss-chunk-size", type=int, default=64)
    result.add_argument("--expected-excluded", type=int, default=13)
    result.add_argument(
        "--gradient-skip-threshold",
        type=float,
        default=FORMAL_GRADIENT_SKIP_THRESHOLD,
    )
    result.add_argument(
        "--max-consecutive-gradient-skips",
        type=int,
        default=FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
    )
    result.add_argument("--log-every", type=int, default=10)
    result.add_argument("--resume-from")
    result.add_argument(
        "--max-steps",
        type=int,
        help="Smoke-only maximum optimizer-attempt windows, not microbatches.",
    )
    return result


def _validate_args(args, world: int) -> None:
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient accumulation steps must be positive")
    local_input_normalization_policy(
        args.local_input_norm,
        args.local_input_rmsnorm_eps,
    )
    locked = {
        "world_size": (world, FORMAL_WORLD_SIZE),
        "epochs": (args.epochs, FORMAL_EPOCHS),
        "seed": (args.seed, FORMAL_SEED),
        "lr": (args.lr, FORMAL_LR),
        "weight_decay": (args.weight_decay, FORMAL_WEIGHT_DECAY),
        "alpha": (args.alpha, FORMAL_ALPHA),
        "max_grad_norm": (args.max_grad_norm, FORMAL_MAX_GRAD_NORM),
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps,
            FORMAL_GRADIENT_ACCUMULATION_STEPS,
        ),
        "local_input_rmsnorm_eps": (
            args.local_input_rmsnorm_eps,
            DEFAULT_LOCAL_INPUT_RMSNORM_EPS,
        ),
        "gradient_skip_threshold": (
            args.gradient_skip_threshold,
            FORMAL_GRADIENT_SKIP_THRESHOLD,
        ),
        "max_consecutive_gradient_skips": (
            args.max_consecutive_gradient_skips,
            FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
        ),
    }
    if args.run_purpose == "formal":
        mismatches = [
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in locked.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "formal 40-epoch protocol mismatch: " + "; ".join(mismatches)
            )
        if args.max_steps is not None:
            raise ValueError("formal training forbids --max-steps")
    elif args.max_steps is None or args.max_steps <= 0:
        raise ValueError("smoke training requires a positive --max-steps")


def _phase2_summary(values: Mapping[str, float]) -> dict:
    """Expose the paper wording while preserving the historical key."""
    result = _summary(values)
    result["evidence_nll"] = result["explanation_nll"]
    return result


def _empty_metric_totals(keys: Sequence[str]) -> dict[str, float]:
    return {key: 0.0 for key in keys}


def _empty_local_scale_totals() -> dict[str, float]:
    return {
        **_empty_metric_totals(LOCAL_INPUT_SUM_METRIC_KEYS),
        **_empty_metric_totals(LOCAL_INPUT_MAX_METRIC_KEYS),
    }


def _add_local_scale_totals(
    target: dict[str, float],
    source: Mapping[str, float],
) -> None:
    for key in LOCAL_INPUT_SUM_METRIC_KEYS:
        target[key] += float(source[key])
    for key in LOCAL_INPUT_MAX_METRIC_KEYS:
        target[key] = max(target[key], float(source[key]))


def _single_window_numerics(
    *,
    window_size: int,
    accumulation_steps: int,
    gradient_norm: float,
    local_word_proj_gradient_norm: float,
    reason: str,
    max_grad_norm: float,
) -> dict[str, float]:
    result = new_window_numerical_totals()
    result["optimizer_attempts"] = 1
    result["micro_steps"] = window_size
    if window_size == accumulation_steps:
        result["full_accumulation_windows"] = 1
    else:
        result["partial_accumulation_windows"] = 1
    if math.isfinite(gradient_norm):
        result["gradient_norm_count"] = 1
        result["gradient_norm_sum"] = gradient_norm
        result["gradient_norm_max"] = gradient_norm
        result["gradient_norm_last"] = gradient_norm
        result["local_word_proj_gradient_norm_sum"] = local_word_proj_gradient_norm
        result["local_word_proj_gradient_norm_max"] = local_word_proj_gradient_norm
        result["local_word_proj_gradient_norm_last"] = local_word_proj_gradient_norm
    if reason == "safe":
        coefficient = min(1.0, max_grad_norm / (gradient_norm + 1e-6))
        result["optimizer_steps_applied"] = 1
        result["clip_coefficient_count"] = 1
        result["clip_coefficient_sum"] = coefficient
        result["clip_coefficient_min"] = coefficient
        result["clip_coefficient_last"] = coefficient
        result["clipped_windows"] = int(gradient_norm > max_grad_norm)
    else:
        result["skipped_windows"] = 1
        result["finite_spike_windows"] = int(reason == "finite_spike")
        result["nonfinite_windows"] = int(reason == "nonfinite")
    return result


def _rank_rng_state(local_rank: int) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(local_rank),
    }


def _restore_rank_rng_state(state: Mapping, local_rank: int) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], local_rank)


def _global_window_valid_rows(window: Sequence[Mapping], device: int) -> float:
    local_count = sum(
        answer.strip() != ERROR_ANSWER
        for batch in window
        for answer in batch["answers"]
    )
    count = torch.tensor(float(local_count), device=device, dtype=torch.float64)
    torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
    value = float(count.item())
    if value <= 0:
        raise RuntimeError("global accumulation window contains no usable QA rows")
    return value


def _all_reduce_window_outputs(
    local_metrics: Mapping[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    metric_tensor = torch.stack(
        [local_metrics[key].detach().to(torch.float64) for key in METRIC_KEYS]
    )
    torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
    metric_values = dict(zip(METRIC_KEYS, metric_tensor.tolist()))
    sums = torch.stack(
        [
            local_metrics[key].detach().to(torch.float64)
            for key in LOCAL_INPUT_SUM_METRIC_KEYS
        ]
    )
    torch.distributed.all_reduce(sums, op=torch.distributed.ReduceOp.SUM)
    maxima = torch.stack(
        [
            local_metrics[key].detach().to(torch.float64)
            for key in LOCAL_INPUT_MAX_METRIC_KEYS
        ]
    )
    torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
    scale_values = dict(zip(LOCAL_INPUT_SUM_METRIC_KEYS, sums.tolist()))
    scale_values.update(zip(LOCAL_INPUT_MAX_METRIC_KEYS, maxima.tolist()))
    return metric_values, scale_values


def _local_word_projection_gradient_norm(module) -> float:
    parameters = [
        parameter
        for name, parameter in module.named_parameters()
        if "local_word_proj" in name and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("local_word_proj has no trainable parameters")
    return float(_gradient_l2_norm(parameters).detach())


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; local CPU training is forbidden")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    _validate_args(args, world)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(hours=6))

    # Capture source identity before any output path can create untracked files.
    source_dirty = bool(_git_value("status", "--porcelain"))
    if args.run_purpose == "formal" and source_dirty:
        raise RuntimeError("formal training requires a clean source worktree")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    split_path = Path(args.series_split_manifest)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_manifest_sha256 = sha256_file(split_path)
    dataset = AXISAnomalyQADataset(
        args.data,
        split="train",
        train_ratio=FORMAL_TRAIN_RATIO,
        seed=args.seed,
    )
    split_identity = validate_training_split_manifest(
        split_manifest,
        [path.name for path in dataset.series_files],
    )
    audit = build_data_audit(dataset, args.seed)
    if audit["qa_rows_excluded"] != args.expected_excluded:
        raise RuntimeError(
            f"expected {args.expected_excluded} excluded rows, "
            f"found {audit['qa_rows_excluded']}"
        )

    base = build_model()
    install_simplified_prompt_runtime(base.axis)
    prompt_audit_box = [
        (
            build_prompt_boundary_audit(
                dataset,
                base.axis.tokenizer,
                num_fixed_hint_tokens=base.axis.num_fixed_tokens,
            )
            if rank == 0
            else None
        )
    ]
    torch.distributed.broadcast_object_list(prompt_audit_box, src=0)
    audit["prompt_boundary"] = prompt_audit_box[0]
    audit["split_identity"] = split_identity
    audit["split_manifest_sha256"] = split_manifest_sha256
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2)
    data_audit_sha256 = hashlib.sha256(audit_text.encode("utf-8")).hexdigest()
    phase1_sha256 = sha256_file(args.phase1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "data_exclusion_and_prompt_audit.json").write_text(
            audit_text,
            encoding="utf-8",
        )
    torch.distributed.barrier()

    resume_payload = None
    resume_meta = None
    if args.resume_from:
        resume_payload = torch.load(
            args.resume_from,
            map_location="cpu",
            weights_only=False,
        )
        resume_meta = validate_resume_checkpoint(
            resume_payload,
            split_manifest_sha256=split_manifest_sha256,
            data_audit_sha256=data_audit_sha256,
            phase1_sha256=phase1_sha256,
            local_input_norm_mode=args.local_input_norm,
            local_input_rmsnorm_eps=args.local_input_rmsnorm_eps,
        )
        load_phase2_40epoch_checkpoint(base, args.resume_from)
        phase1_payload = {"epoch": None, "resume": True}
    else:
        phase1_payload = load_phase1_fresh_hint(base, args.phase1)
        base.axis.perceiver.configure_local_input_normalization(
            args.local_input_norm,
            eps=args.local_input_rmsnorm_eps,
        )

    trainable_before_f0 = freeze_for_phase2(base)
    install_fixed_hint_runtime(base.axis)
    base.axis.perceiver.fix_prompt_embeddings.requires_grad_(False)
    trainable = sum(
        parameter.numel() for parameter in base.parameters() if parameter.requires_grad
    )
    initially_trainable_names = {
        name
        for name, parameter in base.axis.perceiver.named_parameters()
        if parameter.requires_grad
    }
    initial_perceiver = (
        _parameter_snapshot(base.axis.perceiver)
        if rank == 0 and resume_payload is None
        else None
    )
    base.to(local_rank)
    llm_dtype_record = _ensure_frozen_llm_storage_dtype(
        base.axis.model,
        FROZEN_LLM_DTYPE,
    )

    if resume_payload is None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = initialize_fixed_hint_reference(base.axis)
            word_embeddings = base.axis.model.get_input_embeddings().weight
            source = base.axis.perceiver.get_source_embeddings(word_embeddings)
            repeated = base.axis.perceiver.process_fixed_embeddings(
                source,
                base.axis.num_fixed_tokens,
            ).to(reference.dtype)
        difference = (reference.float() - repeated.float()).abs()
        fixed_hint_meta = {
            "policy": "cached_step0_from_fresh_phase2_initialization",
            "dtype": str(reference.dtype),
            "shape": list(reference.shape),
            "sha256": _tensor_sha256(reference),
            "repeat_max_abs_error": float(difference.max()),
            "repeat_mean_abs_error": float(difference.mean()),
            "abs_max": float(reference.float().abs().max()),
        }
    else:
        reference = fixed_hint_checkpoint_state(base.axis)
        if reference is None:
            raise RuntimeError("restored Treatment checkpoint has no F0")
        fixed_hint_meta = dict(resume_meta["fixed_hint"])
        if _tensor_sha256(reference) != fixed_hint_meta["sha256"]:
            raise RuntimeError("restored F0 hash differs from the run manifest")
        fixed_hint_meta["resume_hash_verified"] = True

    f0_hashes: list[str | None] = [None] * world
    torch.distributed.all_gather_object(f0_hashes, _tensor_sha256(reference))
    if len(set(f0_hashes)) != 1:
        raise RuntimeError("step-zero cached F0 differs across DDP ranks")
    llm = base.axis.model
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()
    llm.config.use_cache = False
    llm.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output_tensor: output_tensor.clone()
    )
    objective = Phase2TreatmentObjectiveModel(
        base,
        segment_alpha=args.alpha,
        loss_chunk_size=args.loss_chunk_size,
    )
    ddp = DDP(
        objective,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in ddp.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=FORMAL_MICRO_BATCH_SERIES_PER_RANK,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    active_plan = AccumulationPlan(
        world_size=world,
        micro_batch_series_per_rank=FORMAL_MICRO_BATCH_SERIES_PER_RANK,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        micro_steps_per_epoch=len(loader),
    )
    if args.run_purpose == "formal" and active_plan != FORMAL_ACCUMULATION_PLAN:
        raise RuntimeError(
            "formal DataLoader/accumulation plan differs from the frozen protocol"
        )

    arm = (
        "accumulation_only"
        if args.local_input_norm == LOCAL_INPUT_NORM_NONE
        else "accumulation_plus_rmsnorm"
    )
    local_norm_policy = local_input_normalization_policy(
        args.local_input_norm,
        args.local_input_rmsnorm_eps,
    )
    meta = {
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "experiment_arm": arm,
        "objective": "treatment",
        "source_branch": _git_value("branch", "--show-current"),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_dirty": source_dirty,
        "run_purpose": args.run_purpose,
        "starting_checkpoint_kind": "phase1_time_series_encoder_only",
        "phase1_checkpoint": str(Path(args.phase1).resolve()),
        "phase1_checkpoint_sha256": phase1_sha256,
        "phase1_checkpoint_epoch": phase1_payload.get("epoch"),
        "phase2_initialization": "fresh_seed72_before_ddp",
        "world_size": world,
        "micro_batch_series_per_rank": FORMAL_MICRO_BATCH_SERIES_PER_RANK,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "global_batch_series": active_plan.full_global_batch_series,
        "gradient_accumulation": {
            "plan": active_plan.to_dict(),
            "loss_reduction": "exact_global_usable_QA_row_mean",
            "ddp_scaling": ("local_objective_sum_times_world_over_global_valid_rows"),
            "no_sync": "all_nonfinal_microbatches_in_each_window",
            "operation_order": [
                "accumulate",
                "global_mean",
                "gradient_guard",
                "clip",
                "optimizer_step",
            ],
            "learning_rate_divided_by_accumulation": False,
        },
        "author_deepspeed_reference": {
            "world_size": 4,
            "gradient_accumulation_steps": 32,
            "global_batch_series": 128,
            "gradient_clipping": 1.0,
            "configuration_file": "experiments/configs/deepspeed_config.json",
        },
        "epochs": args.epochs,
        "micro_steps_per_rank_epoch": len(loader),
        "optimizer_attempts_per_epoch": active_plan.optimizer_attempts_per_epoch,
        "planned_steps": FORMAL_TOTAL_STEPS,
        "planned_micro_steps": FORMAL_TOTAL_MICRO_STEPS,
        "step_semantics": {
            "global_step": "optimizer_attempt_window",
            "micro_step": "per_rank_microbatch",
            "optimizer_step": "successfully_applied_AdamW_update",
        },
        "checkpoint_schedule": "every_complete_epoch",
        "declared_candidate_epochs": list(range(1, args.epochs + 1)),
        "lr": args.lr,
        "lr_schedule": {"epochs_1_40": args.lr},
        "weight_decay": args.weight_decay,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": list(optimizer.param_groups[0]["betas"]),
        "optimizer_eps": optimizer.param_groups[0]["eps"],
        "max_grad_norm": args.max_grad_norm,
        "numerical_stability": {
            "perceiver_wide_operations": "FP32 under BF16 outer autocast",
            "gradient_guard_scope": ("once_after_each_complete_accumulation_window"),
            "gradient_skip_threshold": args.gradient_skip_threshold,
            "max_consecutive_gradient_skips": (args.max_consecutive_gradient_skips),
            "fail_fast_policy": formal_gradient_guard_policy(),
            "unsafe_updates_applied": False,
            "bf16_grad_scaler": False,
        },
        "local_input_normalization": local_norm_policy,
        "recorded_activation_metrics": [
            "local_input_raw_rms",
            "local_input_raw_abs_max",
            "local_input_post_normalization_rms",
            "local_input_post_normalization_abs_max",
        ],
        "qk_activation_metrics_recorded": False,
        "precision": "BF16 for all 40 epochs",
        "frozen_llm_storage_dtype": "torch.bfloat16",
        "autocast_dtype": "torch.bfloat16",
        "frozen_llm_dtype_runtime": llm_dtype_record,
        "seed": args.seed,
        "train_ratio": FORMAL_TRAIN_RATIO,
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": split_manifest_sha256,
        "split_identity": split_identity,
        "segment_alpha": args.alpha,
        "tf_rule": (
            "True/False label + first complete semantic statement; "
            "explanation starts at the next statement"
        ),
        "loss_chunk_size": args.loss_chunk_size,
        "prompt_protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "prompt_boundary": audit["prompt_boundary"],
        "fixed_hint": fixed_hint_meta,
        "trainable_parameters_before_f0_freeze": trainable_before_f0,
        "trainable_parameters": trainable,
        "data_audit_sha256": data_audit_sha256,
        "data_counts": {
            key: audit[key]
            for key in (
                "train_series",
                "qa_rows_total",
                "qa_rows_included",
                "qa_rows_excluded",
                "question_type_counts",
                "included_question_type_counts",
                "segmentation_rule_counts",
                "segment_length_audit",
            )
        },
        "selection_protocol": {
            "candidate_epochs": list(range(1, args.epochs + 1)),
            "metric": (
                "effective-answer-token-weighted teacher-forced global token NLL"
            ),
            "direction": "minimize",
            "validation_seed": 72,
            "validation_series": FORMAL_VALIDATION_SERIES,
            "test_or_judge_metrics_consulted": False,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(local_rank),
        },
    }
    if resume_payload is not None:
        meta["resume"] = {
            "policy": "complete_epoch_boundary_only",
            "checkpoint": str(Path(args.resume_from).resolve()),
            "checkpoint_sha256": sha256_file(args.resume_from),
            "checkpoint_epoch": int(resume_payload["epoch"]),
            "checkpoint_global_step": int(resume_payload["global_step"]),
            "checkpoint_micro_step": int(resume_payload["micro_step"]),
            "checkpoint_optimizer_step": int(resume_payload["optimizer_step"]),
            "optimizer_state_restored": True,
            "rng_state_restored_by_rank": True,
            "cumulative_metrics_restored": True,
            "mid_epoch_microbatches_skipped": 0,
        }
    if rank == 0:
        _write_json_atomic(output / "run_manifest.json", meta)
        print(json.dumps(meta, ensure_ascii=False), flush=True)

    if resume_payload is None:
        cumulative = _empty_metric_totals(METRIC_KEYS)
        successful = _empty_metric_totals(METRIC_KEYS)
        cumulative_local_scale = _empty_local_scale_totals()
        optimization = _new_optimization_state(args.max_grad_norm)
        global_step = 0
        micro_step = 0
        optimizer_step = 0
        start_epoch = 1
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
    else:
        cumulative = {
            key: float(resume_payload["cumulative_training_metrics"][key])
            for key in METRIC_KEYS
        }
        successful = {
            key: float(resume_payload["successful_training_metrics"][key])
            for key in METRIC_KEYS
        }
        cumulative_local_scale = {
            key: float(resume_payload["cumulative_local_input_metrics"][key])
            for key in LOCAL_INPUT_SUM_METRIC_KEYS + LOCAL_INPUT_MAX_METRIC_KEYS
        }
        optimization = _restore_optimization_state(
            resume_meta["optimization_diagnostics"],
            args.max_grad_norm,
        )
        global_step = int(resume_payload["global_step"])
        micro_step = int(resume_payload["micro_step"])
        optimizer_step = int(resume_payload["optimizer_step"])
        start_epoch = int(resume_payload["epoch"]) + 1
        rank_states = resume_payload.get("rng_state_by_rank")
        if not isinstance(rank_states, list) or len(rank_states) != world:
            raise ValueError("resume checkpoint lacks one RNG state per DDP rank")
        _restore_rank_rng_state(rank_states[rank], local_rank)

    process_start_global_step = global_step
    process_start_micro_step = micro_step
    start_time = time.perf_counter()
    timed_start = None
    timed_step_origin = global_step
    stopped_early = False
    completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs + 1):
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate_for_epoch(
                epoch,
                initial_lr=args.lr,
            )
        sampler.set_epoch(epoch)
        ddp.train()
        ddp.module.base.ts_pretrain_model.eval()
        ddp.module.base.axis.model.train()
        epoch_guard = new_formal_gradient_guard_tracker()
        epoch_attempted = _empty_metric_totals(METRIC_KEYS)
        epoch_successful = _empty_metric_totals(METRIC_KEYS)
        epoch_local_scale = _empty_local_scale_totals()
        epoch_numerics = new_window_numerical_totals()
        microbatches_completed = 0

        for window_index, window in enumerate(
            iter_accumulation_windows(loader, args.gradient_accumulation_steps),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            declared_global_rows = _global_window_valid_rows(window, local_rank)
            loss_multiplier = ddp_window_loss_multiplier(
                world_size=world,
                global_valid_rows=declared_global_rows,
            )
            local_outputs = {
                key: torch.zeros((), device=local_rank, dtype=torch.float64)
                for key in (
                    METRIC_KEYS
                    + LOCAL_INPUT_SUM_METRIC_KEYS
                    + LOCAL_INPUT_MAX_METRIC_KEYS
                )
            }
            for micro_index, batch in enumerate(window):
                final_microbatch = micro_index == len(window) - 1
                sync_context = nullcontext() if final_microbatch else ddp.no_sync()
                time_series = batch["padded_sequences"].to(
                    local_rank,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                attention_masks = batch["attention_masks"].to(
                    local_rank,
                    non_blocking=True,
                )
                valid_rows = [
                    answer.strip() != ERROR_ANSWER for answer in batch["answers"]
                ]
                with sync_context:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = ddp(
                            time_series,
                            attention_masks,
                            batch["questions"],
                            batch["answers"],
                            batch["start_indices"],
                            batch["end_indices"],
                            batch["question_types"],
                            valid_rows,
                        )
                    (outputs["objective_sum"] * loss_multiplier).backward()
                for key in METRIC_KEYS + LOCAL_INPUT_SUM_METRIC_KEYS:
                    local_outputs[key] += outputs[key].detach().to(torch.float64)
                for key in LOCAL_INPUT_MAX_METRIC_KEYS:
                    local_outputs[key] = torch.maximum(
                        local_outputs[key],
                        outputs[key].detach().to(torch.float64),
                    )

            window_metrics, window_local_scale = _all_reduce_window_outputs(
                local_outputs
            )
            validate_accumulated_objective_denominator(
                declared_global_valid_rows=declared_global_rows,
                observed_global_objective_count=window_metrics["objective_count"],
            )
            gradient_norm = float(_gradient_l2_norm(ddp.parameters()).detach())
            local_word_gradient_norm = _local_word_projection_gradient_norm(ddp.module)
            local_decision = decide_gradient_step(
                gradient_norm,
                threshold=args.gradient_skip_threshold,
            )
            reason_code = {"safe": 0, "finite_spike": 1, "nonfinite": 2}[
                local_decision.reason
            ]
            synchronized_reason = torch.tensor(
                reason_code,
                device=local_rank,
                dtype=torch.int32,
            )
            torch.distributed.all_reduce(
                synchronized_reason,
                op=torch.distributed.ReduceOp.MAX,
            )
            reason = {0: "safe", 1: "finite_spike", 2: "nonfinite"}[
                int(synchronized_reason.item())
            ]
            skipped = reason != "safe"
            fail_fast = epoch_guard.observe(
                skipped=skipped,
                gradient_reason=reason,
            )
            window_numerics = _single_window_numerics(
                window_size=len(window),
                accumulation_steps=args.gradient_accumulation_steps,
                gradient_norm=gradient_norm,
                local_word_proj_gradient_norm=local_word_gradient_norm,
                reason=reason,
                max_grad_norm=args.max_grad_norm,
            )
            add_numeric_totals(epoch_numerics, window_numerics)
            add_numeric_totals(optimization, window_numerics)
            if skipped:
                optimization["gradient_guard_skipped_steps"] += 1
                if reason == "nonfinite":
                    optimization["gradient_guard_nonfinite_skipped_steps"] += 1
                else:
                    optimization["gradient_guard_finite_spike_skipped_steps"] += 1
                optimization["gradient_guard_consecutive_skips"] += 1
                optimization["gradient_guard_max_consecutive_skips"] = max(
                    optimization["gradient_guard_max_consecutive_skips"],
                    optimization["gradient_guard_consecutive_skips"],
                )
            else:
                optimization["gradient_guard_consecutive_skips"] = 0

            global_step += 1
            micro_step += len(window)
            microbatches_completed += len(window)
            for key, value in window_metrics.items():
                cumulative[key] += value
                epoch_attempted[key] += value
            _add_local_scale_totals(cumulative_local_scale, window_local_scale)
            _add_local_scale_totals(epoch_local_scale, window_local_scale)
            absolute_finite_spikes = int(optimization["finite_spike_windows"])
            explicit_abort_reason = None
            if reason == "nonfinite":
                explicit_abort_reason = "nonfinite_gradient"
            elif absolute_finite_spikes >= FORMAL_FINITE_SPIKE_ABORT_COUNT:
                explicit_abort_reason = "second_finite_gradient_spike"
            elif fail_fast.should_abort:
                explicit_abort_reason = fail_fast.reason

            if skipped:
                has_record_slot = (
                    len(optimization["gradient_guard_events"])
                    < FORMAL_GRADIENT_GUARD_MAX_RECORDED_EVENTS
                )
                diagnostics = (
                    top_gradient_diagnostics(ddp.module.named_parameters(), limit=8)
                    if rank == 0 and (has_record_slot or explicit_abort_reason)
                    else []
                )
                event = {
                    "optimizer_attempt_step": global_step,
                    "micro_step": micro_step,
                    "epoch": epoch,
                    "window_in_epoch": window_index,
                    "reason": reason,
                    "gradient_norm": (
                        gradient_norm if math.isfinite(gradient_norm) else None
                    ),
                    "local_word_proj_gradient_norm": (
                        local_word_gradient_norm
                        if math.isfinite(local_word_gradient_norm)
                        else None
                    ),
                    "accumulated_microbatches": len(window),
                    "declared_global_valid_rows": declared_global_rows,
                    "diagnostics": diagnostics,
                }
                if rank == 0 and has_record_slot:
                    optimization["gradient_guard_events"].append(event)
                optimizer.zero_grad(set_to_none=True)
                if explicit_abort_reason:
                    failure = {
                        "schema_version": FORMAL_PROTOCOL_VERSION,
                        "event": "gradient_guard_fail_fast",
                        "reason": explicit_abort_reason,
                        "optimizer_attempt_step": global_step,
                        "micro_step": micro_step,
                        "optimizer_state_applied": False,
                        "checkpoint_written": False,
                        "gradient_event": event,
                        "guard_audit": _completed_epoch_guard_audit(
                            epoch_guard,
                            epoch=epoch,
                            fail_fast_triggered=True,
                        ),
                        "window_metrics": _phase2_summary(window_metrics),
                        "epoch_attempted_metrics": _phase2_summary(epoch_attempted),
                        "epoch_successful_metrics": _phase2_summary(epoch_successful),
                    }
                    if rank == 0:
                        _write_json_atomic(
                            output / "gradient_guard_failure.json",
                            failure,
                        )
                    raise FloatingPointError(
                        f"gradient guard fail-fast ({explicit_abort_reason}) at "
                        f"optimizer attempt {global_step}; no unsafe update or "
                        "checkpoint was written"
                    )
            else:
                torch.nn.utils.clip_grad_norm_(
                    ddp.parameters(),
                    max_norm=args.max_grad_norm,
                )
                optimizer.step()
                optimizer_step += 1
                for key, value in window_metrics.items():
                    successful[key] += value
                    epoch_successful[key] += value

            if global_step == process_start_global_step + 10:
                timed_start = time.perf_counter()
                timed_step_origin = global_step
            if rank == 0 and global_step % args.log_every == 0:
                elapsed = time.perf_counter() - (timed_start or start_time)
                measured_steps = max(1, global_step - timed_step_origin)
                rate = measured_steps / elapsed
                remaining = max(0, FORMAL_TOTAL_STEPS - global_step) / rate
                print(
                    json.dumps(
                        {
                            "optimizer_attempt_step": global_step,
                            "optimizer_step": optimizer_step,
                            "micro_step": micro_step,
                            "epoch": epoch,
                            "window_in_epoch": window_index,
                            "learning_rate": args.lr,
                            "window_microbatches": len(window),
                            "window_global_valid_rows": declared_global_rows,
                            "optimizer_step_applied": not skipped,
                            "window_metrics": _phase2_summary(window_metrics),
                            "cumulative_metrics": _phase2_summary(cumulative),
                            "successful_update_metrics": _phase2_summary(successful),
                            "epoch_attempted_metrics": _phase2_summary(epoch_attempted),
                            "epoch_successful_metrics": _phase2_summary(
                                epoch_successful
                            ),
                            "local_input_scale": local_scale_summary(
                                cumulative_local_scale
                            ),
                            "epoch_local_input_scale": local_scale_summary(
                                epoch_local_scale
                            ),
                            "gradient_l2_norm_before_clip": (
                                gradient_norm if math.isfinite(gradient_norm) else None
                            ),
                            "local_word_proj_gradient_l2_norm_before_clip": (
                                local_word_gradient_norm
                                if math.isfinite(local_word_gradient_norm)
                                else None
                            ),
                            "epoch_numerics": summarize_window_numerics(epoch_numerics),
                            "optimization": _guarded_optimization_summary(optimization),
                            "optimizer_attempts_per_second": rate,
                            "estimated_remaining_hours": remaining / 3600,
                        }
                    ),
                    flush=True,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stopped_early = True
                break

        epoch_is_complete = microbatches_completed == len(loader)
        if epoch_is_complete:
            completed_guard_audit = _completed_epoch_guard_audit(
                epoch_guard,
                epoch=epoch,
                fail_fast_triggered=False,
            )
            try:
                validate_completed_epoch_guard_audit(
                    completed_guard_audit,
                    expected_epoch=epoch,
                    expected_attempted_steps=(active_plan.optimizer_attempts_per_epoch),
                    max_epoch_skip_rate=FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE,
                )
            except ValueError as error:
                if rank == 0:
                    _write_json_atomic(
                        output / "gradient_guard_checkpoint_rejected.json",
                        {
                            "schema_version": FORMAL_PROTOCOL_VERSION,
                            "epoch": epoch,
                            "optimizer_attempt_step": global_step,
                            "micro_step": micro_step,
                            "reason": str(error),
                            "checkpoint_written": False,
                            "guard_audit": completed_guard_audit,
                        },
                    )
                raise FloatingPointError(
                    "completed epoch failed gradient-guard checkpoint gate"
                ) from error
            optimization["gradient_guard_last_completed_epoch"] = completed_guard_audit
            completed_epoch = epoch
            last_completed_epoch_metrics = {
                "attempted": dict(epoch_attempted),
                "successful": dict(epoch_successful),
                "attempted_summary": _phase2_summary(epoch_attempted),
                "successful_summary": _phase2_summary(epoch_successful),
                "local_input": dict(epoch_local_scale),
                "local_input_summary": local_scale_summary(epoch_local_scale),
                "numerics": summarize_window_numerics(epoch_numerics),
            }
        else:
            completed_guard_audit = None
            last_completed_epoch_metrics = None
        rank_rng_states: list[dict | None] = [None] * world
        if epoch_is_complete:
            torch.distributed.all_gather_object(
                rank_rng_states,
                _rank_rng_state(local_rank),
            )
        torch.distributed.barrier()
        if rank == 0 and epoch_is_complete:
            if initial_perceiver is not None:
                optimization["parameter_relative_change_by_step"][str(global_step)] = {
                    "all_perceiver_parameters": _relative_parameter_change(
                        ddp.module.base.axis.perceiver,
                        initial_perceiver,
                    ),
                    "trainable_perceiver_parameters": _relative_parameter_change(
                        ddp.module.base.axis.perceiver,
                        initial_perceiver,
                        initially_trainable_names,
                    ),
                }
            meta["optimization_diagnostics"] = _guarded_optimization_summary(
                optimization
            )
            checkpoint_path = output / f"epoch_{epoch:02d}.pth"
            payload = _checkpoint_payload(
                model=ddp.module.base,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                micro_step=micro_step,
                optimizer_step=optimizer_step,
                meta=meta,
                cumulative=cumulative,
                successful=successful,
                cumulative_local_scale=cumulative_local_scale,
                last_completed_epoch_metrics=last_completed_epoch_metrics,
            )
            payload["rng_state_by_rank"] = rank_rng_states
            save_atomic(payload, checkpoint_path)
            _write_json_atomic(
                output / "latest_complete_epoch.json",
                {
                    "schema_version": FORMAL_PROTOCOL_VERSION,
                    "epoch": epoch,
                    "optimizer_attempt_step": global_step,
                    "optimizer_step": optimizer_step,
                    "micro_step": micro_step,
                    "gradient_guard": completed_guard_audit,
                    "checkpoint_file": checkpoint_path.name,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                },
            )
        elif rank == 0:
            _write_json_atomic(
                output / "incomplete_smoke_stop.json",
                {
                    "schema_version": FORMAL_PROTOCOL_VERSION,
                    "epoch_in_progress": epoch,
                    "microbatches_completed": microbatches_completed,
                    "microbatches_expected": len(loader),
                    "optimizer_attempt_step": global_step,
                    "micro_step": micro_step,
                    "checkpoint_written": False,
                    "reason": ("max_steps reached before a complete epoch boundary"),
                },
            )
        torch.distributed.barrier()
        if stopped_early:
            break

    elapsed = time.perf_counter() - start_time
    if rank == 0:
        meta["optimization_diagnostics"] = _guarded_optimization_summary(optimization)
        final = {
            "schema_version": FORMAL_PROTOCOL_VERSION,
            "experiment": EXPERIMENT,
            "experiment_arm": arm,
            "objective": "treatment",
            "world_size": world,
            "completed_epochs": completed_epoch,
            "optimizer_attempt_step": global_step,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "planned_optimizer_attempts": FORMAL_TOTAL_STEPS,
            "planned_micro_steps": FORMAL_TOTAL_MICRO_STEPS,
            "formal_complete": (
                not stopped_early
                and completed_epoch == args.epochs
                and global_step == FORMAL_TOTAL_STEPS
                and micro_step == FORMAL_TOTAL_MICRO_STEPS
            ),
            "wall_seconds_this_process": elapsed,
            "optimizer_attempts_this_process": (
                global_step - process_start_global_step
            ),
            "micro_steps_this_process": micro_step - process_start_micro_step,
            "optimizer_attempts_per_second": (
                (global_step - process_start_global_step) / elapsed
            ),
            "attempted_metrics": _phase2_summary(cumulative),
            "successful_update_metrics": _phase2_summary(successful),
            "raw_attempted_metric_sums": cumulative,
            "raw_successful_metric_sums": successful,
            "local_input_metrics": local_scale_summary(cumulative_local_scale),
            "raw_local_input_metrics": cumulative_local_scale,
            "optimization_diagnostics": _guarded_optimization_summary(optimization),
        }
        _write_json_atomic(output / "training_summary.json", final)
        print(json.dumps(final), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
