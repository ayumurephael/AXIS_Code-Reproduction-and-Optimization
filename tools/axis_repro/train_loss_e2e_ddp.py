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
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset

from .loss_e2e import ERROR_ANSWER, count_segmentation_rules, normalize_question_type
from .loss_e2e_runtime import (
    ContinuationObjectiveModel,
    fixed_hint_checkpoint_state,
    initialize_fixed_hint_reference,
    install_fixed_hint_runtime,
    load_loss_e2e_checkpoint,
)
from .model_utils import build_model, freeze_for_phase2, sha256_file


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


def build_data_audit(dataset: AXISAnomalyQADataset, seed: int) -> dict:
    type_counts: collections.Counter[str] = collections.Counter()
    included_type_counts: collections.Counter[str] = collections.Counter()
    rule_counts: collections.Counter[str] = collections.Counter()
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
            rule_counts.update(
                count_segmentation_rules(
                    [answer],
                    [str(item.get("question_type", ""))],
                )
            )
    return {
        "schema_version": 1,
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
        "exclusions": exclusions,
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
        "conclusion_nll": _ratio(
            values, "conclusion_mean_sum", "conclusion_row_count"
        ),
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--arm", choices=["control", "treatment"], required=True)
    result.add_argument("--data", default="data/anomaly_llava_training_dataset")
    result.add_argument("--output", required=True)
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=1e-5)
    result.add_argument("--seed", type=int, default=72)
    result.add_argument("--alpha", type=float, default=0.40)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--loss-chunk-size", type=int, default=64)
    result.add_argument("--expected-excluded", type=int, default=13)
    result.add_argument("--log-every", type=int, default=20)
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
    if args.epochs != 2 and args.max_steps is None:
        raise ValueError("formal loss_e2e_0723 runs must use exactly 2 epochs")
    if args.seed != 72:
        raise ValueError("loss_e2e_0723 training split is fixed to seed=72")
    if args.arm == "treatment" and args.alpha != 0.40:
        raise ValueError("the confirmed treatment alpha is exactly 0.40")

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
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2)
    audit_sha256 = hashlib.sha256(audit_text.encode("utf-8")).hexdigest()
    if rank == 0:
        (output / "data_exclusion_manifest.json").write_text(
            audit_text,
            encoding="utf-8",
        )
    torch.distributed.barrier()

    base = build_model()
    author_payload = load_loss_e2e_checkpoint(base, args.checkpoint)
    author_has_optimizer = "optimizer_state_dict" in author_payload
    trainable_before_f0 = freeze_for_phase2(base)
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
        install_fixed_hint_runtime(base.axis)
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
        }
        # F0 is a reference, not an optimizable parameter in the treatment arm.
        base.axis.perceiver.fix_prompt_embeddings.requires_grad_(False)

    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
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
    milestones = {
        max(1, round(planned_steps * fraction))
        for fraction in (0.25, 0.50, 0.75, 1.00)
    }
    meta = {
        "experiment": "loss_e2e_0723",
        "arm": args.arm,
        "source_branch": _git_value("branch", "--show-current"),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_dirty": bool(_git_value("status", "--porcelain")),
        "author_checkpoint": str(Path(args.checkpoint).resolve()),
        "author_checkpoint_sha256": sha256_file(args.checkpoint),
        "author_checkpoint_epoch": author_payload.get("epoch"),
        "author_checkpoint_has_optimizer_state": author_has_optimizer,
        "optimizer_reset_reason": (
            "author checkpoint contains no optimizer_state_dict"
            if not author_has_optimizer
            else "two-arm protocol requires identical fresh optimizers"
        ),
        "world_size": world,
        "batch_size_series_per_rank": 1,
        "qa_rows_per_series": 2,
        "epochs": args.epochs,
        "steps_per_rank_epoch": len(loader),
        "planned_steps": planned_steps,
        "milestone_steps": sorted(milestones),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": list(optimizer.param_groups[0]["betas"]),
        "optimizer_eps": optimizer.param_groups[0]["eps"],
        "seed": args.seed,
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
    if rank == 0:
        (output / "run_manifest.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(meta, ensure_ascii=False), flush=True)

    cumulative = {key: 0.0 for key in METRIC_KEYS}
    global_step = 0
    start_time = time.perf_counter()
    timed_start = None
    stopped_early = False
    for epoch in range(1, args.epochs + 1):
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
            valid_rows = [
                answer.strip() != ERROR_ANSWER
                for answer in batch["answers"]
            ]
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
            optimizer.step()
            global_step += 1
            for key, value in zip(METRIC_KEYS, packed.tolist()):
                cumulative[key] += value

            if global_step == 10:
                timed_start = time.perf_counter()
            if rank == 0 and global_step % args.log_every == 0:
                elapsed = time.perf_counter() - (timed_start or start_time)
                measured_steps = max(1, global_step - (10 if timed_start else 0))
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
                            "steps_per_second_per_rank": rate,
                            "estimated_remaining_hours": remaining / 3600,
                        }
                    ),
                    flush=True,
                )

            if global_step in milestones and rank == 0:
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
        final = {
            "arm": args.arm,
            "world_size": world,
            "completed_epochs": epoch,
            "steps": global_step,
            "planned_steps": planned_steps,
            "formal_complete": not stopped_early and global_step == planned_steps,
            "wall_seconds": elapsed,
            "steps_per_second_per_rank": global_step / elapsed,
            "qa_rows_per_second_global": (
                cumulative["valid_row_count"] / elapsed
            ),
            "metrics": _summary(cumulative),
            "raw_metric_sums": cumulative,
        }
        (output / "training_summary.json").write_text(
            json.dumps(final, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(final), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
