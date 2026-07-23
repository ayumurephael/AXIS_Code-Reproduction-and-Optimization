"""Two-arm continuation experiment for the 2026-07-23 loss redesign.

Both arms load the same author Phase-II checkpoint, reset AdamW identically,
use the seed-72 series order, exclude only exact generation-error labels, and
run for a fixed number of epochs.  The control retains dynamic fixed hints and
global continuation-token NLL.  The treatment uses cached F0 and the
row-balanced conclusion/explanation objective.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset

from .loss_e2e import (
    ERROR_ANSWER,
    content_word_count,
    normalize_question_type,
    segment_answer,
)
from .loss_e2e_runtime import (
    ContinuationObjectiveModel,
    fixed_hint_checkpoint_state,
    initialize_fixed_hint_reference,
    install_fixed_hint_runtime,
    load_loss_e2e_checkpoint,
    set_fixed_hint_reference,
)
from .model_utils import build_model, freeze_for_phase2, load_axis_payload, sha256_file


METRIC_KEYS = (
    "objective_sum",
    "objective_count",
    "token_nll_sum",
    "token_count",
    "valid_row_count",
    "conclusion_mean_sum",
    "conclusion_row_count",
    "explanation_mean_sum",
    "explanation_row_count",
    "both_segment_row_count",
    "conclusion_only_row_count",
    "explanation_only_row_count",
    "empty_segment_row_count",
    "multiple_choice_objective_sum",
    "multiple_choice_row_count",
    "true_false_objective_sum",
    "true_false_row_count",
    "open_ended_objective_sum",
    "open_ended_row_count",
)
FROZEN_LLM_STORAGE_DTYPE_SCHEDULE = {
    1: torch.float16,
    2: torch.bfloat16,
}
LEGACY_EPOCH1_FP16_SOURCE_COMMITS = frozenset(
    {"bcbf73dee7343837be86cff1ae392d8e26fd67cb"}
)


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ordered_series_sha256(dataset: AXISAnomalyQADataset) -> str:
    payload = "\n".join(path.name for path in dataset.series_files).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _length_summary(values: Sequence[int]) -> dict:
    if not values:
        return {
            "count": 0,
            "zero_count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "zero_count": sum(value == 0 for value in values),
        "min": min(values),
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _segment_lengths(answer: str, segmentation) -> dict:
    conclusion = (
        ""
        if segmentation.conclusion is None
        else answer[segmentation.conclusion.start : segmentation.conclusion.end]
    )
    explanation = (
        ""
        if segmentation.explanation is None
        else answer[segmentation.explanation.start : segmentation.explanation.end]
    )
    return {
        "conclusion_chars": len(conclusion),
        "explanation_chars": len(explanation),
        "conclusion_content_words": content_word_count(conclusion),
        "explanation_content_words": content_word_count(explanation),
    }


def _summarize_length_groups(groups: Mapping[str, Mapping[str, list[int]]]) -> dict:
    return {
        group: {
            metric: _length_summary(values)
            for metric, values in sorted(metrics.items())
        }
        for group, metrics in sorted(groups.items())
    }


def _excerpt(text: str, limit: int = 360) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


def build_data_audit(dataset: AXISAnomalyQADataset, seed: int) -> dict:
    type_counts: collections.Counter[str] = collections.Counter()
    included_type_counts: collections.Counter[str] = collections.Counter()
    rule_counts: collections.Counter[str] = collections.Counter()
    lengths_by_type: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    lengths_by_rule: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    samples_by_rule: dict[str, list[dict]] = collections.defaultdict(list)
    exclusions = []
    total = 0
    for series_path in dataset.series_files:
        with open(series_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        for window_index, item in enumerate(record["windows"]):
            answer = str(item["answer"])
            question_type = normalize_question_type(
                str(item.get("question_type", "")),
                answer,
            )
            total += 1
            type_counts[question_type] += 1
            if answer.strip() == ERROR_ANSWER:
                window_range = item.get("window_range", {})
                exclusions.append(
                    {
                        "series_file": series_path.name,
                        "window_index": window_index,
                        "question_type": question_type,
                        "window_start": window_range.get("start"),
                        "window_end": window_range.get("end"),
                        "reason": "exact_generation_error_label",
                    }
                )
                continue
            included_type_counts[question_type] += 1
            segmentation = segment_answer(
                answer,
                str(item.get("question_type", "")),
            )
            rule_key = f"{question_type}/{segmentation.rule}"
            rule_counts[rule_key] += 1
            if segmentation.fallback:
                rule_counts[f"{question_type}/fallback"] += 1
            lengths = _segment_lengths(answer, segmentation)
            for metric, value in lengths.items():
                lengths_by_type[question_type][metric].append(value)
                lengths_by_rule[rule_key][metric].append(value)
            if len(samples_by_rule[rule_key]) < 3:
                conclusion = (
                    ""
                    if segmentation.conclusion is None
                    else answer[
                        segmentation.conclusion.start : segmentation.conclusion.end
                    ]
                )
                explanation = (
                    ""
                    if segmentation.explanation is None
                    else answer[
                        segmentation.explanation.start : segmentation.explanation.end
                    ]
                )
                conclusion_excerpt, conclusion_shortened = _excerpt(conclusion)
                explanation_excerpt, explanation_shortened = _excerpt(explanation)
                samples_by_rule[rule_key].append(
                    {
                        "series_file": series_path.name,
                        "window_index": window_index,
                        "fallback": segmentation.fallback,
                        **lengths,
                        "conclusion_excerpt": conclusion_excerpt,
                        "explanation_excerpt": explanation_excerpt,
                        "excerpt_shortened": (
                            conclusion_shortened or explanation_shortened
                        ),
                    }
                )
    return {
        "schema_version": 2,
        "seed": seed,
        "train_ratio": 0.95,
        "train_series": len(dataset),
        "ordered_train_series_sha256": _ordered_series_sha256(dataset),
        "qa_rows_total": total,
        "qa_rows_included": total - len(exclusions),
        "qa_rows_excluded": len(exclusions),
        "question_type_counts": dict(sorted(type_counts.items())),
        "included_question_type_counts": dict(sorted(included_type_counts.items())),
        "segmentation_rule_counts": dict(sorted(rule_counts.items())),
        "segment_length_audit": {
            "units": {
                "chars": "Python Unicode code points after span trimming",
                "content_words": ("ASCII word/number units used by the OE short rule"),
                "percentiles": "nearest-rank",
            },
            "by_question_type": _summarize_length_groups(lengths_by_type),
            "by_rule": _summarize_length_groups(lengths_by_rule),
            "deterministic_first_three_samples_by_rule": {
                key: value for key, value in sorted(samples_by_rule.items())
            },
        },
        "exclusions": exclusions,
    }


def build_answer_tokenization_audit(
    dataset: AXISAnomalyQADataset,
    tokenizer,
    *,
    batch_size: int = 512,
) -> dict:
    """Audit the exact standalone answer encoding without truncation."""
    model_max_length = int(tokenizer.model_max_length)
    lengths: list[int] = []
    longest: list[dict] = []
    over_limit: list[dict] = []
    pending_prompts: list[str] = []
    pending_ids: list[tuple[str, int]] = []

    def consume() -> None:
        if not pending_prompts:
            return
        encoded = tokenizer(
            pending_prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        for token_ids, (series_file, window_index) in zip(encoded, pending_ids):
            length = len(token_ids)
            lengths.append(length)
            item = {
                "series_file": series_file,
                "window_index": window_index,
                "tokens_with_answer_prefix_and_specials": length,
            }
            longest.append(item)
            if length > model_max_length:
                over_limit.append(item)
        pending_prompts.clear()
        pending_ids.clear()

    for series_path in dataset.series_files:
        with open(series_path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        for window_index, item in enumerate(record["windows"]):
            answer = str(item["answer"])
            if answer.strip() == ERROR_ANSWER:
                continue
            pending_prompts.append(f"Answer: {answer}")
            pending_ids.append((series_path.name, window_index))
            if len(pending_prompts) >= batch_size:
                consume()
    consume()
    longest.sort(
        key=lambda item: (
            -item["tokens_with_answer_prefix_and_specials"],
            item["series_file"],
            item["window_index"],
        )
    )
    return {
        "policy": (
            "standalone answer encoding uses truncation=False; any mismatch "
            "with the baseline full input fails closed"
        ),
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "tokenizer_model_max_length": model_max_length,
        "answer_rows_audited": len(lengths),
        "truncated_answer_count": len(over_limit),
        "untruncated_token_lengths": _length_summary(lengths),
        "ten_longest_answer_identifiers": longest[:10],
        "over_limit_identifiers": over_limit,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _checkpoint_payload(
    *,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    meta: dict,
    cumulative: Mapping[str, float],
) -> dict:
    state = {
        "ts_pretrain_model": model.ts_pretrain_model.state_dict(),
        "moirai_trainable": model.axis.perceiver.state_dict(),
    }
    reference = fixed_hint_checkpoint_state(model.axis)
    if reference is not None:
        state["fixed_hint_reference"] = reference
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "reproduction_meta": meta,
        "cumulative_training_metrics": dict(cumulative),
    }


def _ratio(values: Mapping[str, float], numerator: str, denominator: str):
    count = values.get(denominator, 0.0)
    return None if count == 0 else values.get(numerator, 0.0) / count


def _summary(values: Mapping[str, float]) -> dict:
    return {
        "objective": _ratio(values, "objective_sum", "objective_count"),
        "global_token_nll": _ratio(values, "token_nll_sum", "token_count"),
        "conclusion_nll": _ratio(values, "conclusion_mean_sum", "conclusion_row_count"),
        "explanation_nll": _ratio(
            values, "explanation_mean_sum", "explanation_row_count"
        ),
        "mc_objective": _ratio(
            values,
            "multiple_choice_objective_sum",
            "multiple_choice_row_count",
        ),
        "tf_objective": _ratio(
            values,
            "true_false_objective_sum",
            "true_false_row_count",
        ),
        "oe_objective": _ratio(
            values,
            "open_ended_objective_sum",
            "open_ended_row_count",
        ),
        "objective_denominator": values.get("objective_count", 0.0),
        "valid_rows": values.get("valid_row_count", 0.0),
        "token_count": values.get("token_count", 0.0),
        "both_segment_rows": values.get("both_segment_row_count", 0.0),
        "conclusion_only_rows": values.get("conclusion_only_row_count", 0.0),
        "explanation_only_rows": values.get("explanation_only_row_count", 0.0),
        "empty_segment_rows": values.get("empty_segment_row_count", 0.0),
    }


def _parameter_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def _relative_parameter_change(
    module: torch.nn.Module,
    initial: Mapping[str, torch.Tensor],
    names: Optional[set[str]] = None,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, parameter in module.named_parameters():
        if names is not None and name not in names:
            continue
        reference = initial[name]
        current = parameter.detach().float().cpu()
        numerator += float((current - reference).square().sum())
        denominator += float(reference.square().sum())
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return math.sqrt(numerator / denominator)


def _gradient_l2_norm(parameters) -> torch.Tensor:
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        raise RuntimeError("no gradients were produced for trainable parameters")
    return torch.stack(norms).norm(2)


def _optimization_summary(state: Mapping) -> dict:
    count = int(state["gradient_norm_count"])
    return {
        "gradient_norm_count": count,
        "gradient_l2_norm_mean": (
            None if count == 0 else state["gradient_norm_sum"] / count
        ),
        "gradient_l2_norm_max": (None if count == 0 else state["gradient_norm_max"]),
        "gradient_l2_norm_last": (None if count == 0 else state["gradient_norm_last"]),
        "nonfinite_gradient_steps": state["nonfinite_gradient_steps"],
        "clipped_steps": state["clipped_steps"],
        "max_grad_norm": state["max_grad_norm"],
        "parameter_relative_change_by_step": dict(
            state["parameter_relative_change_by_step"]
        ),
    }


def _optimization_state_from_summary(
    summary: Mapping,
    *,
    expected_max_grad_norm: Optional[float],
) -> dict:
    """Restore auditable optimizer diagnostics from an epoch checkpoint."""
    count = int(summary["gradient_norm_count"])
    mean = summary["gradient_l2_norm_mean"]
    if count <= 0 or mean is None:
        raise ValueError("resume checkpoint has no gradient diagnostics")
    if summary.get("max_grad_norm") != expected_max_grad_norm:
        raise ValueError("resume checkpoint gradient-clipping policy changed")
    return {
        "gradient_norm_count": count,
        "gradient_norm_sum": float(mean) * count,
        "gradient_norm_max": float(summary["gradient_l2_norm_max"]),
        "gradient_norm_last": float(summary["gradient_l2_norm_last"]),
        "nonfinite_gradient_steps": int(summary["nonfinite_gradient_steps"]),
        "clipped_steps": int(summary["clipped_steps"]),
        "max_grad_norm": expected_max_grad_norm,
        "parameter_relative_change_by_step": dict(
            summary["parameter_relative_change_by_step"]
        ),
    }


def _epoch_learning_rate(
    epoch: int,
    *,
    initial_lr: float,
    epoch2_lr: Optional[float],
) -> float:
    if epoch < 1:
        raise ValueError("epoch numbering starts at one")
    return float(epoch2_lr if epoch >= 2 and epoch2_lr is not None else initial_lr)


def _validate_formal_optimizer_schedule(
    *,
    initial_lr: float,
    epoch2_lr: Optional[float],
    weight_decay: float,
    max_grad_norm: Optional[float],
) -> None:
    expected = {
        "initial_lr": (initial_lr, 1e-4),
        "epoch2_lr": (epoch2_lr, 3e-5),
        "weight_decay": (weight_decay, 1e-5),
        "max_grad_norm": (max_grad_norm, None),
    }
    mismatches = [
        f"{name}={actual!r} (expected {locked!r})"
        for name, (actual, locked) in expected.items()
        if actual != locked
    ]
    if mismatches:
        raise ValueError("formal optimizer schedule mismatch: " + "; ".join(mismatches))


def _epoch_frozen_llm_storage_dtype(epoch: int) -> torch.dtype:
    try:
        return FROZEN_LLM_STORAGE_DTYPE_SCHEDULE[epoch]
    except KeyError as error:
        raise ValueError(
            "the locked frozen-LLM dtype schedule contains only epochs 1 and 2"
        ) from error


def _ensure_frozen_llm_storage_dtype(
    llm: torch.nn.Module,
    target_dtype: torch.dtype,
) -> dict:
    """Apply and verify the locked storage dtype without touching trainable state."""
    if target_dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("frozen LLM storage dtype must be FP16 or BF16")
    trainable = [
        name for name, parameter in llm.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise RuntimeError(
            "frozen LLM dtype transition found trainable parameters: "
            + ", ".join(trainable[:5])
        )
    floating = [
        (name, parameter)
        for name, parameter in llm.named_parameters()
        if parameter.is_floating_point()
    ]
    if not floating:
        raise RuntimeError("frozen LLM has no floating-point parameters")
    before = collections.Counter(str(parameter.dtype) for _, parameter in floating)
    transition_applied = set(before) != {str(target_dtype)}
    if transition_applied:
        llm.to(dtype=target_dtype)
    after = collections.Counter(
        str(parameter.dtype)
        for _, parameter in llm.named_parameters()
        if parameter.is_floating_point()
    )
    if set(after) != {str(target_dtype)}:
        raise RuntimeError(f"frozen LLM dtype transition failed: {dict(after)}")
    input_embeddings = llm.get_input_embeddings()
    output_embeddings = llm.get_output_embeddings()
    input_dtype = str(input_embeddings.weight.dtype)
    output_dtype = (
        None if output_embeddings is None else str(output_embeddings.weight.dtype)
    )
    if input_dtype != str(target_dtype) or output_dtype != str(target_dtype):
        raise RuntimeError(
            "input/output embedding dtype differs from the locked epoch dtype"
        )
    return {
        "before_parameter_dtypes": dict(before),
        "after_parameter_dtypes": dict(after),
        "after_dtype": str(target_dtype),
        "input_embedding_dtype": input_dtype,
        "output_embedding_dtype": output_dtype,
        "transition_applied": transition_applied,
    }


def _validated_epoch1_frozen_llm_dtype(meta: Mapping) -> dict:
    """Prove that a resume checkpoint's completed first epoch used FP16."""
    observed = meta.get("frozen_llm_storage_dtype_observed_by_epoch", {}).get("1")
    if observed is not None:
        if observed != "torch.float16":
            raise ValueError(
                "resume checkpoint epoch-1 frozen LLM dtype is not torch.float16"
            )
        return {
            "dtype": observed,
            "provenance": "checkpoint_manifest_runtime_validation",
        }
    source_commit = meta.get("source_commit")
    if source_commit in LEGACY_EPOCH1_FP16_SOURCE_COMMITS:
        return {
            "dtype": "torch.float16",
            "provenance": "validated_legacy_source_commit",
            "source_commit": source_commit,
        }
    raise ValueError(
        "resume checkpoint lacks auditable epoch-1 frozen LLM dtype evidence"
    )


def _validate_epoch_boundary_resume(
    payload: Mapping,
    *,
    arm: str,
    steps_per_epoch: int,
    seed: int,
    alpha: float,
    initial_lr: float,
    weight_decay: float,
    data_audit_sha256: str,
    author_checkpoint_sha256: str,
) -> Mapping:
    """Fail closed unless a checkpoint is the exact completed epoch-1 state."""
    required = {
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "reproduction_meta",
        "cumulative_training_metrics",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"resume checkpoint is missing fields: {missing}")
    if int(payload["epoch"]) != 1 or int(payload["global_step"]) != steps_per_epoch:
        raise ValueError(
            "formal resume is allowed only at the completed epoch-1 boundary"
        )
    meta = payload["reproduction_meta"]
    expected = {
        "experiment": "loss_e2e_0723",
        "arm": arm,
        "run_purpose": "formal",
        "world_size": 3,
        "epochs": 2,
        "planned_steps": 2 * steps_per_epoch,
        "seed": seed,
        "segment_alpha": alpha,
        "lr": initial_lr,
        "weight_decay": weight_decay,
        "data_audit_sha256": data_audit_sha256,
        "author_checkpoint_sha256": author_checkpoint_sha256,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"resume checkpoint identity mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    if meta.get("source_dirty") is not False:
        raise ValueError("resume checkpoint was produced from a dirty source tree")
    _validated_epoch1_frozen_llm_dtype(meta)
    cumulative = payload["cumulative_training_metrics"]
    missing_metrics = sorted(set(METRIC_KEYS).difference(cumulative))
    if missing_metrics:
        raise ValueError(
            f"resume checkpoint lacks cumulative metrics: {missing_metrics}"
        )
    expected_valid_rows = int(meta["data_counts"]["qa_rows_included"])
    if int(cumulative["valid_row_count"]) != expected_valid_rows:
        raise ValueError("resume checkpoint does not contain one full data epoch")
    if int(meta["optimization_diagnostics"]["gradient_norm_count"]) != steps_per_epoch:
        raise ValueError("resume checkpoint gradient count is not one full epoch")
    if int(meta["optimization_diagnostics"]["nonfinite_gradient_steps"]) != 0:
        raise ValueError("resume checkpoint already contains non-finite gradients")
    state = payload["model_state_dict"]
    if arm == "treatment" and "fixed_hint_reference" not in state:
        raise ValueError("treatment resume checkpoint has no cached F0")
    return meta


def _calibration_subset_manifest(
    dataset: AXISAnomalyQADataset,
    *,
    world_size: int,
    seed: int,
    steps_per_rank: int,
) -> dict:
    rank_series: dict[str, list[str]] = {}
    all_names: list[str] = []
    for rank in range(world_size):
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        sampler.set_epoch(1)
        indices = list(iter(sampler))[:steps_per_rank]
        names = [dataset.series_files[index].name for index in indices]
        rank_series[str(rank)] = names
        all_names.extend(names)
    if len(set(all_names)) != len(all_names):
        raise RuntimeError("calibration subset unexpectedly contains duplicates")
    ordered_payload = "\n".join(
        f"{rank}:{position}:{name}"
        for rank, names in sorted(rank_series.items())
        for position, name in enumerate(names)
    )
    return {
        "schema_version": 1,
        "selection_only": True,
        "source_split": "seed-72 95% training split",
        "sampler_epoch": 1,
        "world_size": world_size,
        "steps_per_rank": steps_per_rank,
        "unique_series": len(set(all_names)),
        "ordered_rank_series_sha256": hashlib.sha256(
            ordered_payload.encode("utf-8")
        ).hexdigest(),
        "rank_series": rank_series,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--arm", choices=["control", "treatment"], required=True)
    result.add_argument("--data", default="data/anomaly_llava_training_dataset")
    result.add_argument("--output", required=True)
    result.add_argument(
        "--run-purpose",
        choices=["formal", "calibration", "smoke"],
        default="formal",
    )
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument(
        "--epoch2-lr",
        type=float,
        help="Optional epoch-2 LR; used identically by both formal arms.",
    )
    result.add_argument(
        "--resume-from",
        help=(
            "Resume only from a fail-closed completed epoch-1 checkpoint; "
            "mid-epoch recovery is forbidden."
        ),
    )
    result.add_argument("--seed", type=int, default=72)
    result.add_argument("--alpha", type=float, default=0.40)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--loss-chunk-size", type=int, default=64)
    result.add_argument("--expected-excluded", type=int, default=13)
    result.add_argument("--log-every", type=int, default=20)
    result.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Optional global gradient clipping threshold; omitted in locked runs.",
    )
    result.add_argument(
        "--max-steps",
        type=int,
        help="Smoke-test escape hatch; omit for the formal fixed-epoch run.",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; local CPU execution is forbidden")
    if args.run_purpose == "formal":
        if args.epochs != 2 or args.max_steps is not None:
            raise ValueError(
                "formal loss_e2e_0723 runs require 2 epochs and no max-steps"
            )
        _validate_formal_optimizer_schedule(
            initial_lr=args.lr,
            epoch2_lr=args.epoch2_lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
        )
    elif args.run_purpose == "calibration":
        if args.epochs != 2 or args.max_steps != 200:
            raise ValueError("calibration is fixed to 200 steps from the first epoch")
    elif args.max_steps is None:
        raise ValueError("smoke runs require --max-steps")
    if args.seed != 72:
        raise ValueError("loss_e2e_0723 training split is fixed to seed=72")
    if args.arm == "treatment" and args.alpha != 0.40:
        raise ValueError("the confirmed treatment alpha is exactly 0.40")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive when provided")
    if args.epoch2_lr is not None and args.epoch2_lr <= 0:
        raise ValueError("--epoch2-lr must be positive when provided")
    if args.resume_from is not None:
        if args.run_purpose != "formal":
            raise ValueError("resume is supported only for formal runs")
        if args.epoch2_lr is None:
            raise ValueError("formal resume requires an explicit --epoch2-lr")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")
    if world != 3:
        raise ValueError("formal experiment requires exactly three DDP ranks")

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = AXISAnomalyQADataset(
        args.data,
        split="train",
        train_ratio=0.95,
        seed=args.seed,
    )
    audit = build_data_audit(dataset, args.seed)
    if audit["qa_rows_excluded"] != args.expected_excluded:
        raise RuntimeError(
            f"expected {args.expected_excluded} corrupt labels, "
            f"found {audit['qa_rows_excluded']}"
        )
    base = build_model()
    tokenization_box = [
        (
            build_answer_tokenization_audit(dataset, base.axis.tokenizer)
            if rank == 0
            else None
        )
    ]
    torch.distributed.broadcast_object_list(tokenization_box, src=0)
    audit["answer_tokenization"] = tokenization_box[0]
    if audit["answer_tokenization"]["truncated_answer_count"] != 0:
        raise RuntimeError(
            "formal training forbids truncated answers; inspect the data audit"
        )
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2)
    audit_sha256 = hashlib.sha256(audit_text.encode("utf-8")).hexdigest()
    author_checkpoint_sha256 = sha256_file(args.checkpoint)
    steps_per_epoch = math.ceil(len(dataset) / world)
    if rank == 0:
        (output / "data_exclusion_manifest.json").write_text(
            audit_text,
            encoding="utf-8",
        )
    torch.distributed.barrier()

    author_payload = load_loss_e2e_checkpoint(base, args.checkpoint)
    author_has_optimizer = "optimizer_state_dict" in author_payload
    trainable_before_f0 = freeze_for_phase2(base)
    if args.arm == "treatment":
        install_fixed_hint_runtime(base.axis)
        # F0 is a reference, not an optimizable parameter in the treatment arm.
        base.axis.perceiver.fix_prompt_embeddings.requires_grad_(False)
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    initial_perceiver = _parameter_snapshot(base.axis.perceiver) if rank == 0 else None
    resume_payload = None
    resume_checkpoint_meta = None
    if args.resume_from is not None:
        resume_payload = torch.load(
            args.resume_from,
            map_location="cpu",
            weights_only=False,
        )
        resume_checkpoint_meta = _validate_epoch_boundary_resume(
            resume_payload,
            arm=args.arm,
            steps_per_epoch=steps_per_epoch,
            seed=args.seed,
            alpha=args.alpha,
            initial_lr=args.lr,
            weight_decay=args.weight_decay,
            data_audit_sha256=audit_sha256,
            author_checkpoint_sha256=author_checkpoint_sha256,
        )
        load_axis_payload(base, resume_payload, strict=True)
        if args.arm == "treatment":
            set_fixed_hint_reference(
                base.axis,
                resume_payload["model_state_dict"]["fixed_hint_reference"],
            )
    initially_trainable_names = {
        name
        for name, parameter in base.axis.perceiver.named_parameters()
        if parameter.requires_grad
    }
    base.to(local_rank)

    fixed_hint_meta = {
        "policy": "dynamic",
        "dtype": None,
        "shape": None,
        "sha256": None,
        "repeat_max_abs_error": None,
        "repeat_mean_abs_error": None,
    }
    if args.arm == "treatment":
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
                "policy": "cached_step0",
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
                raise RuntimeError("restored treatment checkpoint has no F0")
            fixed_hint_meta = dict(resume_checkpoint_meta["fixed_hint"])
            if _tensor_sha256(reference) != fixed_hint_meta["sha256"]:
                raise RuntimeError("restored F0 hash differs from epoch-1 manifest")
            fixed_hint_meta["resume_hash_verified"] = True
            fixed_hint_meta["abs_max"] = float(reference.float().abs().max())
    llm = base.axis.model
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()
    llm.config.use_cache = False
    llm.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output_tensor: output_tensor.clone()
    )

    continuation = ContinuationObjectiveModel(
        base,
        objective_mode=args.arm,
        segment_alpha=args.alpha,
        loss_chunk_size=args.loss_chunk_size,
    )
    ddp = DDP(
        continuation,
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
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    planned_steps = args.epochs * len(loader)
    if len(loader) != steps_per_epoch:
        raise RuntimeError("DDP loader length changed after resume validation")
    milestones = {
        max(1, round(planned_steps * fraction)) for fraction in (0.25, 0.50, 0.75, 1.00)
    }
    calibration_manifest_sha256 = None
    if args.run_purpose == "calibration":
        calibration_manifest = _calibration_subset_manifest(
            dataset,
            world_size=world,
            seed=args.seed,
            steps_per_rank=args.max_steps,
        )
        calibration_text = json.dumps(
            calibration_manifest, ensure_ascii=False, indent=2
        )
        calibration_manifest_sha256 = hashlib.sha256(
            calibration_text.encode("utf-8")
        ).hexdigest()
        if rank == 0:
            (output / "calibration_subset_manifest.json").write_text(
                calibration_text,
                encoding="utf-8",
            )

    frozen_llm_observed: dict[str, str] = {}
    frozen_llm_runtime: dict[str, dict] = {}
    frozen_llm_provenance: dict[str, str] = {}
    if resume_payload is not None:
        epoch1_dtype = _validated_epoch1_frozen_llm_dtype(resume_checkpoint_meta)
        frozen_llm_observed["1"] = epoch1_dtype["dtype"]
        frozen_llm_provenance["1"] = epoch1_dtype["provenance"]
    frozen_llm_schedule = {
        str(epoch): str(dtype)
        for epoch, dtype in FROZEN_LLM_STORAGE_DTYPE_SCHEDULE.items()
    }

    meta = {
        "experiment": "loss_e2e_0723",
        "arm": args.arm,
        "source_branch": _git_value("branch", "--show-current"),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_dirty": bool(_git_value("status", "--porcelain")),
        "run_purpose": args.run_purpose,
        "author_checkpoint": str(Path(args.checkpoint).resolve()),
        "author_checkpoint_sha256": author_checkpoint_sha256,
        "author_checkpoint_epoch": author_payload.get("epoch"),
        "author_checkpoint_has_optimizer_state": author_has_optimizer,
        "optimizer_reset_reason": (
            "completed epoch-1 optimizer state restored"
            if resume_payload is not None
            else (
                "author checkpoint contains no optimizer_state_dict"
                if not author_has_optimizer
                else "fresh two-arm optimizer initialized from the author checkpoint"
            )
        ),
        "world_size": world,
        "batch_size_series_per_rank": 1,
        "qa_rows_per_series": 2,
        "epochs": args.epochs,
        "steps_per_rank_epoch": len(loader),
        "planned_steps": planned_steps,
        "milestone_steps": sorted(milestones),
        "lr": args.lr,
        "epoch_learning_rates": {
            "1": _epoch_learning_rate(1, initial_lr=args.lr, epoch2_lr=args.epoch2_lr),
            "2": _epoch_learning_rate(2, initial_lr=args.lr, epoch2_lr=args.epoch2_lr),
        },
        "weight_decay": args.weight_decay,
        "optimizer": "torch.optim.AdamW",
        "frozen_llm_storage_dtype_policy": ("epoch1_fp16_epoch2_bfloat16"),
        "frozen_llm_storage_dtype_schedule": frozen_llm_schedule,
        "frozen_llm_storage_dtype_observed_by_epoch": frozen_llm_observed,
        "frozen_llm_storage_dtype_observation_provenance": (frozen_llm_provenance),
        "frozen_llm_storage_dtype_runtime": frozen_llm_runtime,
        "optimizer_betas": list(optimizer.param_groups[0]["betas"]),
        "optimizer_eps": optimizer.param_groups[0]["eps"],
        "scheduler": None,
        "warmup_steps": 0,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "calibration_subset_manifest_sha256": (calibration_manifest_sha256),
        "segment_alpha": args.alpha,
        "tf_rule": (
            "conclusion = True/False label + first complete semantic statement; "
            "explanation starts at the next statement; no overlap"
        ),
        "loss_chunk_size": args.loss_chunk_size,
        "amp_dtype": "torch.bfloat16",
        "trainable_parameters_before_f0_freeze": trainable_before_f0,
        "trainable_parameters": trainable,
        "data_audit_sha256": audit_sha256,
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
                "answer_tokenization",
            )
        },
        "fixed_hint": fixed_hint_meta,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(local_rank),
        },
    }
    if resume_payload is not None:
        meta["resume"] = {
            "policy": "completed_epoch_boundary_only",
            "checkpoint": str(Path(args.resume_from).resolve()),
            "checkpoint_sha256": sha256_file(args.resume_from),
            "checkpoint_epoch": int(resume_payload["epoch"]),
            "checkpoint_global_step": int(resume_payload["global_step"]),
            "checkpoint_source_commit": resume_checkpoint_meta["source_commit"],
            "optimizer_state_restored": True,
            "cumulative_metrics_restored": True,
            "sampler_resume_epoch": int(resume_payload["epoch"]) + 1,
            "mid_epoch_batches_skipped": 0,
        }
    if rank == 0:
        (output / "run_manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(meta, ensure_ascii=False), flush=True)

    if resume_payload is None:
        cumulative = {key: 0.0 for key in METRIC_KEYS}
        optimization = {
            "gradient_norm_count": 0,
            "gradient_norm_sum": 0.0,
            "gradient_norm_max": 0.0,
            "gradient_norm_last": None,
            "nonfinite_gradient_steps": 0,
            "clipped_steps": 0,
            "max_grad_norm": args.max_grad_norm,
            "parameter_relative_change_by_step": {},
        }
        global_step = 0
        start_epoch = 1
    else:
        cumulative = {
            key: float(resume_payload["cumulative_training_metrics"][key])
            for key in METRIC_KEYS
        }
        optimization = _optimization_state_from_summary(
            resume_checkpoint_meta["optimization_diagnostics"],
            expected_max_grad_norm=args.max_grad_norm,
        )
        global_step = int(resume_payload["global_step"])
        start_epoch = int(resume_payload["epoch"]) + 1
    process_start_step = global_step
    process_start_valid_rows = cumulative["valid_row_count"]
    start_time = time.perf_counter()
    timed_start = None
    timed_step_origin = process_start_step
    stopped_early = False
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_lr = _epoch_learning_rate(
            epoch,
            initial_lr=args.lr,
            epoch2_lr=args.epoch2_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        dtype_record = _ensure_frozen_llm_storage_dtype(
            ddp.module.base.axis.model,
            _epoch_frozen_llm_storage_dtype(epoch),
        )
        epoch_key = str(epoch)
        meta["frozen_llm_storage_dtype_observed_by_epoch"][epoch_key] = dtype_record[
            "after_dtype"
        ]
        meta["frozen_llm_storage_dtype_observation_provenance"][
            epoch_key
        ] = "runtime_parameter_validation"
        meta["frozen_llm_storage_dtype_runtime"][epoch_key] = dtype_record
        if rank == 0:
            (output / "run_manifest.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        torch.distributed.barrier()
        sampler.set_epoch(epoch)
        ddp.train()
        ddp.module.base.ts_pretrain_model.eval()
        # Gradient checkpointing in Transformers is gated on training mode.
        # Qwen2 attention dropout is zero in the pinned author configuration.
        ddp.module.base.axis.model.train()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            time_series = batch["padded_sequences"].to(
                local_rank,
                dtype=torch.float32,
                non_blocking=True,
            )
            attention_masks = batch["attention_masks"].to(
                local_rank,
                non_blocking=True,
            )
            valid_rows = [answer.strip() != ERROR_ANSWER for answer in batch["answers"]]
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

            packed = torch.stack(
                [outputs[key].detach().to(torch.float64) for key in METRIC_KEYS]
            )
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            global_denominator = packed[METRIC_KEYS.index("objective_count")]
            if global_denominator.item() <= 0:
                raise RuntimeError("a global DDP batch contains no usable QA rows")
            # DDP averages gradients across ranks. Multiplication by world makes
            # that average equal the desired global numerator/global denominator.
            loss = outputs["objective_sum"] * world / global_denominator
            loss.backward()
            if args.max_grad_norm is None:
                gradient_norm_tensor = _gradient_l2_norm(ddp.parameters())
            else:
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    ddp.parameters(),
                    max_norm=args.max_grad_norm,
                )
            gradient_norm = float(gradient_norm_tensor.detach())
            if not math.isfinite(gradient_norm):
                optimization["nonfinite_gradient_steps"] += 1
                raise FloatingPointError(
                    f"non-finite gradient norm at step {global_step + 1}"
                )
            optimization["gradient_norm_count"] += 1
            optimization["gradient_norm_sum"] += gradient_norm
            optimization["gradient_norm_max"] = max(
                optimization["gradient_norm_max"], gradient_norm
            )
            optimization["gradient_norm_last"] = gradient_norm
            if args.max_grad_norm is not None and gradient_norm > args.max_grad_norm:
                optimization["clipped_steps"] += 1
            optimizer.step()
            global_step += 1
            for key, value in zip(METRIC_KEYS, packed.tolist()):
                cumulative[key] += value

            if global_step == process_start_step + 10:
                timed_start = time.perf_counter()
                timed_step_origin = global_step
            if rank == 0 and global_step % args.log_every == 0:
                elapsed = time.perf_counter() - (timed_start or start_time)
                measured_steps = max(1, global_step - timed_step_origin)
                rate = measured_steps / elapsed
                remaining = max(0, planned_steps - global_step) / rate
                step_values = dict(zip(METRIC_KEYS, packed.tolist()))
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "step_metrics": _summary(step_values),
                            "cumulative_metrics": _summary(cumulative),
                            "gradient_l2_norm": gradient_norm,
                            "optimization": _optimization_summary(optimization),
                            "steps_per_second_per_rank": rate,
                            "estimated_remaining_hours": remaining / 3600,
                        }
                    ),
                    flush=True,
                )

            if global_step in milestones and rank == 0:
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
                meta["optimization_diagnostics"] = _optimization_summary(optimization)
                save_atomic(
                    _checkpoint_payload(
                        model=ddp.module.base,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        meta=meta,
                        cumulative=cumulative,
                    ),
                    output / f"step_{global_step}.pth",
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stopped_early = True
                break
        torch.distributed.barrier()
        if rank == 0:
            save_atomic(
                _checkpoint_payload(
                    model=ddp.module.base,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    meta=meta,
                    cumulative=cumulative,
                ),
                output / f"epoch_{epoch}.pth",
            )
        torch.distributed.barrier()
        if stopped_early:
            break

    elapsed = time.perf_counter() - start_time
    if rank == 0:
        if str(global_step) not in optimization["parameter_relative_change_by_step"]:
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
        meta["optimization_diagnostics"] = _optimization_summary(optimization)
        final = {
            "arm": args.arm,
            "world_size": world,
            "completed_epochs": epoch,
            "steps": global_step,
            "planned_steps": planned_steps,
            "formal_complete": not stopped_early and global_step == planned_steps,
            "epoch_learning_rates": meta["epoch_learning_rates"],
            "frozen_llm_storage_dtype_schedule": meta[
                "frozen_llm_storage_dtype_schedule"
            ],
            "frozen_llm_storage_dtype_observed_by_epoch": meta[
                "frozen_llm_storage_dtype_observed_by_epoch"
            ],
            "frozen_llm_storage_dtype_runtime": meta[
                "frozen_llm_storage_dtype_runtime"
            ],
            "wall_seconds_this_process": elapsed,
            "steps_executed_this_process": global_step - process_start_step,
            "steps_per_second_per_rank": ((global_step - process_start_step) / elapsed),
            "qa_rows_per_second_global": (
                (cumulative["valid_row_count"] - process_start_valid_rows) / elapsed
            ),
            "metrics": _summary(cumulative),
            "raw_metric_sums": cumulative,
            "optimization_diagnostics": _optimization_summary(optimization),
        }
        (output / "training_summary.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(final), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
