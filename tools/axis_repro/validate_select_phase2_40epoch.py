"""Validate all 40 formal checkpoints and select the unique minimum NLL."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset

from .loss_e2e import ERROR_ANSWER, normalize_question_type
from .loss_e2e_40epoch_runtime import (
    Phase2TreatmentObjectiveModel,
    load_phase2_40epoch_checkpoint,
)
from .model_utils import build_model, freeze_for_phase2, sha256_file
from .prompt_boundary import SIMPLIFIED_FINAL_ANSWER_V1
from .train_phase2_treatment_40epoch_ddp import (
    EXPERIMENT,
    FORMAL_ACCUMULATION_PLAN,
    FORMAL_ALPHA,
    FORMAL_EPOCHS,
    FORMAL_GRADIENT_ACCUMULATION_STEPS,
    FORMAL_GRADIENT_SKIP_THRESHOLD,
    FORMAL_LR,
    FORMAL_MAX_GRAD_NORM,
    FORMAL_MICRO_STEPS_PER_EPOCH,
    FORMAL_PROTOCOL_VERSION,
    FORMAL_SEED,
    FORMAL_STEPS_PER_EPOCH,
    FORMAL_TOTAL_MICRO_STEPS,
    FORMAL_TOTAL_STEPS,
    FORMAL_TRAIN_RATIO,
    FORMAL_TRAIN_SERIES,
    FORMAL_VALIDATION_SERIES,
    FORMAL_WEIGHT_DECAY,
    FORMAL_WORLD_SIZE,
    expected_global_step_for_epoch,
    expected_micro_step_for_epoch,
    validate_formal_completed_epoch_guard,
    validate_checkpoint_step_counters,
)


EXPECTED_VALIDATION_ROWS = 3_000
SELECTION_METRIC = "effective-answer-token-weighted teacher-forced global token NLL"
EFFECTIVE_TOKEN_DEFINITION = (
    "shifted labels != -100 on non-error rows under the boundary-correct "
    "global continuation objective, including terminal EOS"
)


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


def validate_split_manifest(
    manifest: Mapping,
    dataset_series: Sequence[str],
) -> dict:
    if manifest.get("seed") != FORMAL_SEED:
        raise ValueError("validation manifest seed must be exactly 72")
    if manifest.get("train_ratio") != FORMAL_TRAIN_RATIO:
        raise ValueError("validation manifest train_ratio must be exactly 0.95")
    train = manifest.get("train_series")
    validation = manifest.get("val_series")
    if not isinstance(train, list) or not isinstance(validation, list):
        raise ValueError("manifest must contain train_series and val_series")
    if len(train) != FORMAL_TRAIN_SERIES:
        raise ValueError("training split size changed")
    if len(validation) != FORMAL_VALIDATION_SERIES:
        raise ValueError("validation split size changed")
    if len(set(train)) != len(train) or len(set(validation)) != len(validation):
        raise ValueError("split manifest contains duplicate series")
    if set(train).intersection(validation):
        raise ValueError("training and validation series overlap")
    if list(dataset_series) != validation:
        raise ValueError(
            "dataset validation series/order differs from the frozen manifest"
        )
    return {
        "seed": FORMAL_SEED,
        "train_ratio": FORMAL_TRAIN_RATIO,
        "train_series": len(train),
        "validation_series": len(validation),
        "ordered_validation_series_sha256": hashlib.sha256(
            "\n".join(validation).encode("utf-8")
        ).hexdigest(),
    }


def build_validation_data_manifest(
    dataset: AXISAnomalyQADataset,
    split_identity: Mapping,
    split_manifest_sha256: str,
) -> dict:
    total = 0
    exclusions = []
    type_counts: collections.Counter[str] = collections.Counter()
    included_counts: collections.Counter[str] = collections.Counter()
    for series_path in dataset.series_files:
        record = json.loads(series_path.read_text(encoding="utf-8"))
        windows = record.get("windows", [])
        if len(windows) != 2:
            raise ValueError(f"{series_path.name} does not contain exactly 2 QA rows")
        for window_index, item in enumerate(windows):
            total += 1
            answer = str(item.get("answer", ""))
            question_type = normalize_question_type(
                str(item.get("question_type", "")),
                answer,
            )
            type_counts[question_type] += 1
            if answer.strip() == ERROR_ANSWER:
                exclusions.append(
                    {
                        "series_file": series_path.name,
                        "window_index": window_index,
                        "question_type": question_type,
                        "reason": "exact_generation_error_label",
                    }
                )
            else:
                included_counts[question_type] += 1
    if total != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_ROWS} validation rows, got {total}"
        )
    return {
        "schema_version": 1,
        "purpose": "40_epoch_checkpoint_selection",
        "split": dict(split_identity),
        "split_manifest_sha256": split_manifest_sha256,
        "qa_rows_total": total,
        "qa_rows_included": total - len(exclusions),
        "qa_rows_excluded": len(exclusions),
        "question_type_counts": dict(sorted(type_counts.items())),
        "included_question_type_counts": dict(sorted(included_counts.items())),
        "exclusions": exclusions,
    }


def select_checkpoint_paths(values: Sequence[str]) -> list[Path]:
    if len(values) != FORMAL_EPOCHS:
        raise ValueError("exactly 40 checkpoint paths are required")
    by_epoch: dict[int, Path] = {}
    for value in values:
        path = Path(value)
        suffix = path.stem.removeprefix("epoch_")
        if not suffix.isdigit():
            raise ValueError(f"checkpoint must be named epoch_<N>.pth: {path.name}")
        epoch = int(suffix)
        if epoch in by_epoch:
            raise ValueError(f"duplicate checkpoint epoch: {epoch}")
        by_epoch[epoch] = path
    if set(by_epoch) != set(range(1, FORMAL_EPOCHS + 1)):
        raise ValueError("candidate epochs must be exactly 1 through 40")
    return [by_epoch[epoch] for epoch in range(1, FORMAL_EPOCHS + 1)]


def validate_checkpoint_identity(
    payload: Mapping,
    *,
    expected_epoch: int,
    split_manifest_sha256: str,
) -> dict:
    if int(payload.get("schema_version", -1)) != FORMAL_PROTOCOL_VERSION:
        raise ValueError("candidate predates the accumulation-v2 protocol")
    epoch = int(payload.get("epoch", -1))
    step = int(payload.get("global_step", -1))
    micro_step = int(payload.get("micro_step", -1))
    optimizer_step = int(payload.get("optimizer_step", -1))
    if epoch != expected_epoch:
        raise ValueError(
            f"checkpoint epoch {epoch} does not match candidate {expected_epoch}"
        )
    if step != expected_global_step_for_epoch(epoch):
        raise ValueError("optimizer-attempt step is not the complete epoch boundary")
    if micro_step != expected_micro_step_for_epoch(epoch):
        raise ValueError("micro-step is not the complete epoch boundary")

    meta = payload.get("reproduction_meta", {})
    expected = {
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "objective": "treatment",
        "run_purpose": "formal",
        "world_size": FORMAL_WORLD_SIZE,
        "epochs": FORMAL_EPOCHS,
        "micro_steps_per_rank_epoch": FORMAL_MICRO_STEPS_PER_EPOCH,
        "optimizer_attempts_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "planned_steps": FORMAL_TOTAL_STEPS,
        "planned_micro_steps": FORMAL_TOTAL_MICRO_STEPS,
        "seed": FORMAL_SEED,
        "segment_alpha": FORMAL_ALPHA,
        "lr": FORMAL_LR,
        "lr_schedule": {"epochs_1_40": FORMAL_LR},
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "max_grad_norm": FORMAL_MAX_GRAD_NORM,
        "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
        "global_batch_series": FORMAL_ACCUMULATION_PLAN.full_global_batch_series,
        "prompt_protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "split_manifest_sha256": split_manifest_sha256,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"checkpoint identity mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    if meta.get("source_dirty") is not False:
        raise ValueError("candidate checkpoint was produced from a dirty source tree")
    if meta.get("gradient_accumulation", {}).get("plan") != (
        FORMAL_ACCUMULATION_PLAN.to_dict()
    ):
        raise ValueError("candidate accumulation plan changed")

    norm = meta.get("local_input_normalization", {})
    if norm.get("mode") not in {"none", "rmsnorm"}:
        raise ValueError("candidate has an unknown local-input normalization arm")
    if norm.get("affine") is not False or float(norm.get("eps", -1)) != 1e-6:
        raise ValueError("candidate RMSNorm policy differs from the confirmed protocol")
    expected_arm = (
        "accumulation_only" if norm["mode"] == "none" else "accumulation_plus_rmsnorm"
    )
    if meta.get("experiment_arm") != expected_arm:
        raise ValueError("candidate arm label disagrees with normalization mode")

    stability = meta.get("numerical_stability", {})
    if stability.get("gradient_skip_threshold") != FORMAL_GRADIENT_SKIP_THRESHOLD:
        raise ValueError("candidate gradient-guard threshold changed")
    if stability.get("unsafe_updates_applied") is not False:
        raise ValueError("candidate does not prove fail-closed optimizer updates")
    validate_formal_completed_epoch_guard(meta, epoch)
    guard = meta["optimization_diagnostics"]["gradient_guard"]
    validate_checkpoint_step_counters(payload)
    if optimizer_step != step - int(guard["skipped_steps"]):
        raise ValueError("optimizer-step count disagrees with guarded windows")

    state = payload.get("model_state_dict", {})
    if "fixed_hint_reference" not in state:
        raise ValueError("Treatment candidate has no cached F0")
    rank_rng = payload.get("rng_state_by_rank")
    if not isinstance(rank_rng, list) or len(rank_rng) != FORMAL_WORLD_SIZE:
        raise ValueError("candidate lacks one RNG state per DDP rank")
    required_metrics = {
        "cumulative_training_metrics",
        "successful_training_metrics",
        "cumulative_local_input_metrics",
        "last_completed_epoch_training_metrics",
    }
    if required_metrics.difference(payload):
        raise ValueError("candidate lacks accumulation-v2 metric records")
    return {
        "epoch": epoch,
        "global_step": step,
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "experiment_arm": expected_arm,
        "local_input_normalization": dict(norm),
        "checkpoint_source_commit": meta.get("source_commit"),
        "phase1_checkpoint_sha256": meta.get("phase1_checkpoint_sha256"),
        "training_data_audit_sha256": meta.get("data_audit_sha256"),
        "fixed_hint_sha256": meta.get("fixed_hint", {}).get("sha256"),
        "prompt_protocol": meta.get("prompt_protocol"),
        "gradient_guard": guard,
        "numerical_stability": stability,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoints", nargs="+", required=True)
    result.add_argument("--data", required=True)
    result.add_argument("--series-split-manifest", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--loss-chunk-size", type=int, default=64)
    result.add_argument("--log-every", type=int, default=100)
    return result


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; local CPU validation is forbidden")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != FORMAL_WORLD_SIZE:
        raise ValueError("formal selection requires exactly five DDP ranks")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        "nccl",
        timeout=timedelta(hours=6),
    )
    random_seed = FORMAL_SEED + rank
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    checkpoints = select_checkpoint_paths(args.checkpoints)
    for path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(path)
    split_path = Path(args.series_split_manifest)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_manifest_sha256 = sha256_file(split_path)
    dataset = AXISAnomalyQADataset(
        args.data,
        split="val",
        train_ratio=FORMAL_TRAIN_RATIO,
        seed=FORMAL_SEED,
    )
    split_identity = validate_split_manifest(
        split_manifest,
        [path.name for path in dataset.series_files],
    )
    data_manifest_box = [None]
    if rank == 0:
        data_manifest_box[0] = build_validation_data_manifest(
            dataset,
            split_identity,
            split_manifest_sha256,
        )
    torch.distributed.broadcast_object_list(data_manifest_box, src=0)
    data_manifest = data_manifest_box[0]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data_manifest_path = output / "validation_data_manifest.json"
    if rank == 0:
        _write_json_atomic(data_manifest_path, data_manifest)
    torch.distributed.barrier()
    data_manifest_hash_box = [sha256_file(data_manifest_path) if rank == 0 else None]
    torch.distributed.broadcast_object_list(data_manifest_hash_box, src=0)
    validation_data_manifest_sha256 = data_manifest_hash_box[0]

    rank_indices = list(range(rank, len(dataset), world))
    loader = DataLoader(
        Subset(dataset, rank_indices),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    if len(loader) != FORMAL_VALIDATION_SERIES // world:
        raise RuntimeError("validation rank partition size changed")

    base = build_model()
    freeze_for_phase2(base)
    base.to(local_rank)
    base.axis.model.to(dtype=torch.bfloat16)
    base.eval()
    evaluator = Phase2TreatmentObjectiveModel(
        base,
        segment_alpha=FORMAL_ALPHA,
        loss_chunk_size=args.loss_chunk_size,
    )
    evaluator.eval()

    summaries = []
    shared_identity = None
    for expected_epoch, checkpoint_path in enumerate(checkpoints, start=1):
        started = time.perf_counter()
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        identity = validate_checkpoint_identity(
            payload,
            expected_epoch=expected_epoch,
            split_manifest_sha256=split_manifest_sha256,
        )
        load_phase2_40epoch_checkpoint(base, checkpoint_path)
        base.axis.fixed_hint_reference = base.axis.fixed_hint_reference.to(local_rank)
        base.eval()

        identity_fields = {
            key: identity[key]
            for key in (
                "experiment_arm",
                "local_input_normalization",
                "phase1_checkpoint_sha256",
                "training_data_audit_sha256",
                "fixed_hint_sha256",
                "prompt_protocol",
            )
        }
        if shared_identity is None:
            shared_identity = identity_fields
        elif identity_fields != shared_identity:
            raise ValueError("candidate checkpoint identities differ across epochs")

        checkpoint_hash_box = [sha256_file(checkpoint_path) if rank == 0 else None]
        torch.distributed.broadcast_object_list(checkpoint_hash_box, src=0)
        local_totals = torch.zeros(11, dtype=torch.float64, device=local_rank)
        for batch_index, batch in enumerate(loader, start=1):
            valid_rows = [answer.strip() != ERROR_ANSWER for answer in batch["answers"]]
            time_series = batch["padded_sequences"].to(
                local_rank,
                dtype=torch.float32,
                non_blocking=True,
            )
            attention_masks = batch["attention_masks"].to(
                local_rank,
                non_blocking=True,
            )
            with torch.inference_mode(), torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                metrics = evaluator(
                    time_series,
                    attention_masks,
                    batch["questions"],
                    batch["answers"],
                    batch["start_indices"],
                    batch["end_indices"],
                    batch["question_types"],
                    valid_rows,
                )
            local_totals[0] += metrics["token_nll_sum"].double()
            local_totals[1] += metrics["token_count"].double()
            local_totals[2] += metrics["valid_row_count"].double()
            local_totals[3] += len(valid_rows)
            local_totals[4] += len(valid_rows) - sum(valid_rows)
            local_totals[5] += metrics["conclusion_mean_sum"].double()
            local_totals[6] += metrics["conclusion_row_count"].double()
            local_totals[7] += metrics["explanation_mean_sum"].double()
            local_totals[8] += metrics["explanation_row_count"].double()
            local_totals[9] += metrics["objective_sum"].double()
            local_totals[10] += metrics["objective_count"].double()
            if rank == 0 and batch_index % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "candidate_epoch": expected_epoch,
                            "rank0_series_complete": batch_index,
                            "rank0_series_total": len(loader),
                        }
                    ),
                    flush=True,
                )
        torch.distributed.all_reduce(
            local_totals,
            op=torch.distributed.ReduceOp.SUM,
        )
        (
            token_sum,
            token_count,
            valid_rows,
            total_rows,
            excluded_rows,
            conclusion_sum,
            conclusion_count,
            evidence_sum,
            evidence_count,
            treatment_objective_sum,
            treatment_objective_count,
        ) = local_totals.tolist()
        if not math.isfinite(token_sum) or token_sum <= 0 or token_count <= 0:
            raise FloatingPointError("validation NLL is not finite and positive")
        segment_values = (
            conclusion_sum,
            evidence_sum,
            treatment_objective_sum,
        )
        segment_counts = (
            conclusion_count,
            evidence_count,
            treatment_objective_count,
        )
        if any(not math.isfinite(value) for value in segment_values) or any(
            count <= 0 for count in segment_counts
        ):
            raise FloatingPointError("validation segment NLL is invalid")
        observed = {
            "total_rows": int(total_rows),
            "valid_rows": int(valid_rows),
            "excluded_rows": int(excluded_rows),
        }
        expected = {
            "total_rows": data_manifest["qa_rows_total"],
            "valid_rows": data_manifest["qa_rows_included"],
            "excluded_rows": data_manifest["qa_rows_excluded"],
        }
        if observed != expected:
            raise RuntimeError(
                f"validation row audit mismatch: {observed} != {expected}"
            )
        summary = {
            "schema_version": FORMAL_PROTOCOL_VERSION,
            "experiment": EXPERIMENT,
            "experiment_arm": identity["experiment_arm"],
            "local_input_normalization": identity["local_input_normalization"],
            "objective": "treatment",
            "candidate_epoch": expected_epoch,
            "candidate_global_step": identity["global_step"],
            "candidate_micro_step": identity["micro_step"],
            "candidate_optimizer_step": identity["optimizer_step"],
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_hash_box[0],
            **identity_fields,
            "selection_metric": SELECTION_METRIC,
            "effective_token_definition": EFFECTIVE_TOKEN_DEFINITION,
            "metric_direction": "minimize",
            "token_nll_sum": token_sum,
            "effective_answer_token_count": int(token_count),
            "global_token_nll": token_sum / token_count,
            "conclusion_nll": conclusion_sum / conclusion_count,
            "evidence_nll": evidence_sum / evidence_count,
            "treatment_row_objective": (
                treatment_objective_sum / treatment_objective_count
            ),
            "conclusion_row_count": int(conclusion_count),
            "evidence_row_count": int(evidence_count),
            "treatment_objective_row_count": int(treatment_objective_count),
            "validation_series": len(dataset),
            "training_epoch_metrics": payload["last_completed_epoch_training_metrics"],
            **observed,
            "validation_seed": FORMAL_SEED,
            "train_ratio": FORMAL_TRAIN_RATIO,
            "split_manifest_sha256": split_manifest_sha256,
            "validation_data_manifest_sha256": (validation_data_manifest_sha256),
            "rank_partition": "series_index_mod_world_size",
            "world_size": world,
            "batch_size_per_rank_series": 1,
            "frozen_llm_storage_dtype": "torch.bfloat16",
            "autocast_dtype": "torch.bfloat16",
            "loss_accumulation_dtype": "torch.float64",
            "loss_chunk_size": args.loss_chunk_size,
            "wall_seconds": time.perf_counter() - started,
            "evaluator_source_commit": _git_value("rev-parse", "HEAD"),
            "evaluator_source_dirty": bool(_git_value("status", "--porcelain")),
        }
        if rank == 0:
            _write_json_atomic(
                output / f"epoch_{expected_epoch:02d}.json",
                summary,
            )
            print(json.dumps(summary), flush=True)
        summaries.append(summary)
        del payload
        torch.distributed.barrier()

    token_counts = {summary["effective_answer_token_count"] for summary in summaries}
    if len(token_counts) != 1:
        raise RuntimeError("effective validation token denominator drifted")
    selected = min(
        summaries,
        key=lambda row: (row["global_token_nll"], row["candidate_epoch"]),
    )
    if rank == 0:
        selection = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "objective": "treatment",
            "selection_metric": SELECTION_METRIC,
            "effective_token_definition": EFFECTIVE_TOKEN_DEFINITION,
            "metric_direction": "minimize",
            "tie_break": "lower epoch",
            "declared_candidate_epochs": list(range(1, FORMAL_EPOCHS + 1)),
            "validation_seed": FORMAL_SEED,
            "validation_series": FORMAL_VALIDATION_SERIES,
            "train_ratio": FORMAL_TRAIN_RATIO,
            "test_or_judge_metrics_consulted": False,
            "candidates": summaries,
            "selected": {
                key: selected[key]
                for key in (
                    "candidate_epoch",
                    "candidate_global_step",
                    "checkpoint_file",
                    "checkpoint_sha256",
                    "global_token_nll",
                    "token_nll_sum",
                    "effective_answer_token_count",
                )
            },
        }
        _write_json_atomic(output / "best_checkpoint.json", selection)
        _write_json_atomic(
            output / "validation_run_summary.json",
            {
                "schema_version": 1,
                "experiment": EXPERIMENT,
                "candidate_epochs": list(range(1, FORMAL_EPOCHS + 1)),
                "candidates": summaries,
                "selected_epoch": selected["candidate_epoch"],
                "test_or_judge_metrics_consulted": False,
            },
        )
        print(json.dumps(selection["selected"]), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
