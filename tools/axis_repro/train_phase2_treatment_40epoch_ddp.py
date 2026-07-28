"""Full Phase-II retraining with the loss_e2e_0723 Treatment objective.

This is a new experiment, not a continuation of the author's released
Phase-II checkpoint.  It loads only the repository-recorded Phase-I time-series
checkpoint, initializes a fresh Hint Tuner/Perceiver, caches F0 at step zero,
and trains for exactly 40 epochs. Epoch 1 was completed on five ranks; at the
user-requested complete-epoch boundary, epochs 2--40 migrate to four ranks.
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
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import transformers
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset

from .loss_e2e import ERROR_ANSWER
from .loss_e2e_40epoch_runtime import (
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
    _optimization_state_from_summary,
    _optimization_summary,
    _parameter_snapshot,
    _relative_parameter_change,
    _summary,
    _tensor_sha256,
    build_data_audit,
    save_atomic,
)


EXPERIMENT = "loss_e2e_0723_phase2_40epoch"
FORMAL_EPOCHS = 40
LEGACY_WORLD_SIZE = 5
LEGACY_STEPS_PER_EPOCH = 5_700
MIGRATION_COMPLETED_EPOCH = 1
FORMAL_WORLD_SIZE = 4
FORMAL_SEED = 72
FORMAL_TRAIN_RATIO = 0.95
FORMAL_TRAIN_SERIES = 28_500
FORMAL_VALIDATION_SERIES = 1_500
FORMAL_STEPS_PER_EPOCH = 7_125
FORMAL_TOTAL_STEPS = (
    MIGRATION_COMPLETED_EPOCH * LEGACY_STEPS_PER_EPOCH
    + (FORMAL_EPOCHS - MIGRATION_COMPLETED_EPOCH) * FORMAL_STEPS_PER_EPOCH
)
FORMAL_LR = 1e-4
FORMAL_WEIGHT_DECAY = 1e-5
FORMAL_ALPHA = 0.40
FORMAL_MAX_GRAD_NORM = 1.0
FROZEN_LLM_DTYPE = torch.bfloat16


def expected_global_step_for_epoch(epoch: int) -> int:
    """Return the complete-boundary step under the 5-to-4 GPU schedule."""
    if not 0 <= epoch <= FORMAL_EPOCHS:
        raise ValueError(f"epoch must be in [0, {FORMAL_EPOCHS}]")
    legacy_epochs = min(epoch, MIGRATION_COMPLETED_EPOCH)
    four_gpu_epochs = max(0, epoch - MIGRATION_COMPLETED_EPOCH)
    return (
        legacy_epochs * LEGACY_STEPS_PER_EPOCH
        + four_gpu_epochs * FORMAL_STEPS_PER_EPOCH
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
        raise RuntimeError("gold answers with leading whitespace require a new protocol")
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
    meta: dict,
    cumulative: Mapping[str, float],
) -> dict:
    reference = fixed_hint_checkpoint_state(model.axis)
    if reference is None:
        raise RuntimeError("Treatment checkpoint cannot be saved without F0")
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {
            "ts_pretrain_model": model.ts_pretrain_model.state_dict(),
            "moirai_trainable": model.axis.perceiver.state_dict(),
            "fixed_hint_reference": reference,
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "reproduction_meta": meta,
        "cumulative_training_metrics": dict(cumulative),
    }


def validate_resume_checkpoint(
    payload: Mapping,
    *,
    split_manifest_sha256: str,
    data_audit_sha256: str,
    phase1_sha256: str,
) -> dict:
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
    epoch = int(payload["epoch"])
    step = int(payload["global_step"])
    if not 1 <= epoch < FORMAL_EPOCHS:
        raise ValueError("resume epoch must be a completed epoch in [1, 39]")
    if step != expected_global_step_for_epoch(epoch):
        raise ValueError("resume checkpoint is not at a complete epoch boundary")
    meta = payload["reproduction_meta"]
    is_legacy_epoch = epoch <= MIGRATION_COMPLETED_EPOCH
    expected_world = LEGACY_WORLD_SIZE if is_legacy_epoch else FORMAL_WORLD_SIZE
    expected_steps_per_epoch = (
        LEGACY_STEPS_PER_EPOCH if is_legacy_epoch else FORMAL_STEPS_PER_EPOCH
    )
    expected_planned_steps = (
        FORMAL_EPOCHS * LEGACY_STEPS_PER_EPOCH
        if is_legacy_epoch
        else FORMAL_TOTAL_STEPS
    )
    expected = {
        "experiment": EXPERIMENT,
        "run_purpose": "formal",
        "objective": "treatment",
        "world_size": expected_world,
        "epochs": FORMAL_EPOCHS,
        "steps_per_rank_epoch": expected_steps_per_epoch,
        "planned_steps": expected_planned_steps,
        "seed": FORMAL_SEED,
        "segment_alpha": FORMAL_ALPHA,
        "lr": FORMAL_LR,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "max_grad_norm": FORMAL_MAX_GRAD_NORM,
        "prompt_protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "split_manifest_sha256": split_manifest_sha256,
        "data_audit_sha256": data_audit_sha256,
        "phase1_checkpoint_sha256": phase1_sha256,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"resume checkpoint identity mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    if meta.get("source_dirty") is not False:
        raise ValueError("resume checkpoint was produced from a dirty source tree")
    if "fixed_hint_reference" not in payload["model_state_dict"]:
        raise ValueError("resume Treatment checkpoint has no cached F0")
    if set(METRIC_KEYS).difference(payload["cumulative_training_metrics"]):
        raise ValueError("resume checkpoint lacks cumulative metrics")
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
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--loss-chunk-size", type=int, default=64)
    result.add_argument("--expected-excluded", type=int, default=13)
    result.add_argument("--log-every", type=int, default=50)
    result.add_argument("--resume-from")
    result.add_argument("--max-steps", type=int)
    return result


def _validate_args(args, world: int) -> None:
    locked = {
        "world_size": (world, FORMAL_WORLD_SIZE),
        "epochs": (args.epochs, FORMAL_EPOCHS),
        "seed": (args.seed, FORMAL_SEED),
        "lr": (args.lr, FORMAL_LR),
        "weight_decay": (args.weight_decay, FORMAL_WEIGHT_DECAY),
        "alpha": (args.alpha, FORMAL_ALPHA),
        "max_grad_norm": (args.max_grad_norm, FORMAL_MAX_GRAD_NORM),
    }
    if args.run_purpose == "formal":
        mismatches = [
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in locked.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError("formal 40-epoch protocol mismatch: " + "; ".join(mismatches))
        if args.max_steps is not None:
            raise ValueError("formal training forbids --max-steps")
        if args.resume_from is None:
            raise ValueError(
                "formal 4-GPU migration requires a complete-epoch --resume-from"
            )
    elif args.max_steps is None or args.max_steps <= 0:
        raise ValueError("smoke training requires a positive --max-steps")


def main() -> None:
    args = parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; local CPU training is forbidden")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    _validate_args(args, world)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl")

    # Model/F0 initialization must be bit-identical on every rank.  Per-rank
    # training RNG streams are installed only after DDP construction.
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
        )
        load_phase2_40epoch_checkpoint(base, args.resume_from)
        phase1_payload = {"epoch": None, "resume": True}
    else:
        phase1_payload = load_phase1_fresh_hint(base, args.phase1)

    trainable_before_f0 = freeze_for_phase2(base)
    install_fixed_hint_runtime(base.axis)
    base.axis.perceiver.fix_prompt_embeddings.requires_grad_(False)
    trainable = sum(
        parameter.numel()
        for parameter in base.parameters()
        if parameter.requires_grad
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
    torch.distributed.all_gather_object(
        f0_hashes,
        _tensor_sha256(reference),
    )
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
        (
            parameter
            for parameter in ddp.parameters()
            if parameter.requires_grad
        ),
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
    if len(loader) != FORMAL_STEPS_PER_EPOCH:
        raise RuntimeError(
            f"expected {FORMAL_STEPS_PER_EPOCH} optimizer steps per epoch, "
            f"got {len(loader)}"
        )
    planned_steps = FORMAL_TOTAL_STEPS
    source_dirty = bool(_git_value("status", "--porcelain"))
    if args.run_purpose == "formal" and source_dirty:
        raise RuntimeError("formal training requires a clean source worktree")

    meta = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
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
        "micro_batch_series_per_rank": 1,
        "gradient_accumulation_steps": 1,
        "global_batch_series": world,
        "world_size_schedule": {
            "epoch_1": LEGACY_WORLD_SIZE,
            "epochs_2_40": FORMAL_WORLD_SIZE,
        },
        "qa_rows_per_series": 2,
        "epochs": args.epochs,
        "steps_per_rank_epoch": len(loader),
        "planned_steps": planned_steps,
        "optimizer_step_schedule": {
            "epoch_1": LEGACY_STEPS_PER_EPOCH,
            "epochs_2_40": FORMAL_STEPS_PER_EPOCH,
            "total": FORMAL_TOTAL_STEPS,
        },
        "migration": {
            "policy": "complete_epoch_boundary_only",
            "completed_5gpu_epochs": MIGRATION_COMPLETED_EPOCH,
            "active_4gpu_epochs": FORMAL_EPOCHS - MIGRATION_COMPLETED_EPOCH,
            "partial_epoch_2_steps_discarded": 1_050,
            "reason": "user_requested_one_free_H100",
        },
        "checkpoint_schedule": "every_complete_epoch",
        "declared_candidate_epochs": list(range(1, args.epochs + 1)),
        "lr": args.lr,
        "lr_schedule": "constant",
        "weight_decay": args.weight_decay,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": list(optimizer.param_groups[0]["betas"]),
        "optimizer_eps": optimizer.param_groups[0]["eps"],
        "max_grad_norm": args.max_grad_norm,
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
            "optimizer_state_restored": True,
            "source_world_size": resume_meta["world_size"],
            "active_world_size": world,
            "cumulative_metrics_restored": True,
            "mid_epoch_batches_skipped": 0,
        }
    if rank == 0:
        _write_json_atomic(output / "run_manifest.json", meta)
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
            resume_meta["optimization_diagnostics"],
            expected_max_grad_norm=args.max_grad_norm,
        )
        global_step = int(resume_payload["global_step"])
        start_epoch = int(resume_payload["epoch"]) + 1

    # Independent rank RNG streams are appropriate after identical model/F0
    # initialization has been verified.
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    process_start_step = global_step
    start_time = time.perf_counter()
    timed_start = None
    timed_step_origin = global_step
    stopped_early = False
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        sampler.set_epoch(epoch)
        ddp.train()
        ddp.module.base.ts_pretrain_model.eval()
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
                [
                    outputs[key].detach().to(torch.float64)
                    for key in METRIC_KEYS
                ]
            )
            torch.distributed.all_reduce(
                packed,
                op=torch.distributed.ReduceOp.SUM,
            )
            denominator = packed[METRIC_KEYS.index("objective_count")]
            if denominator.item() <= 0:
                raise RuntimeError("global DDP batch contains no usable QA rows")
            loss = outputs["objective_sum"] * world / denominator
            loss.backward()
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
                optimization["gradient_norm_max"],
                gradient_norm,
            )
            optimization["gradient_norm_last"] = gradient_norm
            if gradient_norm > args.max_grad_norm:
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
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "step_in_epoch": global_step
                            - expected_global_step_for_epoch(epoch - 1),
                            "step_metrics": _summary(
                                dict(zip(METRIC_KEYS, packed.tolist()))
                            ),
                            "cumulative_metrics": _summary(cumulative),
                            "gradient_l2_norm_before_clip": gradient_norm,
                            "optimization": _optimization_summary(optimization),
                            "steps_per_second_per_rank": rate,
                            "estimated_remaining_hours": remaining / 3600,
                        }
                    ),
                    flush=True,
                )
            if args.max_steps is not None and global_step >= args.max_steps:
                stopped_early = True
                break

        completed_epoch = epoch
        torch.distributed.barrier()
        if rank == 0:
            if initial_perceiver is not None:
                optimization["parameter_relative_change_by_step"][
                    str(global_step)
                ] = {
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
            checkpoint_path = output / f"epoch_{epoch:02d}.pth"
            save_atomic(
                _checkpoint_payload(
                    model=ddp.module.base,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    meta=meta,
                    cumulative=cumulative,
                ),
                checkpoint_path,
            )
            _write_json_atomic(
                output / "latest_complete_epoch.json",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "checkpoint_file": checkpoint_path.name,
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                },
            )
        torch.distributed.barrier()
        if stopped_early:
            break

    elapsed = time.perf_counter() - start_time
    if rank == 0:
        meta["optimization_diagnostics"] = _optimization_summary(optimization)
        final = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "objective": "treatment",
            "world_size": world,
            "completed_epochs": completed_epoch,
            "steps": global_step,
            "planned_steps": planned_steps,
            "formal_complete": (
                not stopped_early
                and completed_epoch == args.epochs
                and global_step == FORMAL_TOTAL_STEPS
            ),
            "wall_seconds_this_process": elapsed,
            "steps_executed_this_process": global_step - process_start_step,
            "steps_per_second_per_rank": (
                (global_step - process_start_step) / elapsed
            ),
            "metrics": _summary(cumulative),
            "raw_metric_sums": cumulative,
            "optimization_diagnostics": _optimization_summary(optimization),
        }
        _write_json_atomic(output / "training_summary.json", final)
        print(json.dumps(final), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
