"""Exact, leakage-free validation NLL for loss_e2e_0723 checkpoints."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


from .loss_e2e import ERROR_ANSWER, normalize_question_type
from .loss_e2e_runtime import (
    ContinuationObjectiveModel,
    install_fixed_hint_runtime,
    set_fixed_hint_reference,
)
from .model_utils import (
    build_model,
    freeze_for_phase2,
    load_axis_payload,
    sha256_file,
)

if TYPE_CHECKING:
    from src.models.AXIS.dataset import AXISAnomalyQADataset


CANDIDATE_STEP_TO_EPOCH = {
    4750: 1,
    9500: 1,
    14250: 2,
    19000: 2,
}
VALIDATION_SEED = 72
TRAIN_RATIO = 0.95
EXPECTED_TRAIN_SERIES = 28500
EXPECTED_VALIDATION_SERIES = 1500
EXPECTED_VALIDATION_ROWS = 3000
EFFECTIVE_TOKEN_DEFINITION = (
    "shifted AXIS labels != -100 on non-error rows under the common global "
    "continuation objective"
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
    """Fail closed unless the manifest is the fixed seed-72 95/5 split."""
    if manifest.get("seed") != VALIDATION_SEED:
        raise ValueError("validation manifest seed must be exactly 72")
    if manifest.get("train_ratio") != TRAIN_RATIO:
        raise ValueError("validation manifest train_ratio must be exactly 0.95")
    train_series = manifest.get("train_series")
    val_series = manifest.get("val_series")
    if not isinstance(train_series, list) or not isinstance(val_series, list):
        raise ValueError("validation manifest must contain train_series and val_series")
    if len(train_series) != EXPECTED_TRAIN_SERIES:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_SERIES} training series, got {len(train_series)}"
        )
    if len(val_series) != EXPECTED_VALIDATION_SERIES:
        raise ValueError(
            "expected "
            f"{EXPECTED_VALIDATION_SERIES} validation series, got {len(val_series)}"
        )
    if len(set(train_series)) != len(train_series):
        raise ValueError("training series manifest contains duplicates")
    if len(set(val_series)) != len(val_series):
        raise ValueError("validation series manifest contains duplicates")
    if set(train_series).intersection(val_series):
        raise ValueError("training and validation series overlap")
    if list(dataset_series) != val_series:
        raise ValueError(
            "dataset validation series/order differs from the frozen seed-72 manifest"
        )
    ordered_sha256 = hashlib.sha256("\n".join(val_series).encode("utf-8")).hexdigest()
    return {
        "seed": VALIDATION_SEED,
        "train_ratio": TRAIN_RATIO,
        "train_series": len(train_series),
        "validation_series": len(val_series),
        "ordered_validation_series_sha256": ordered_sha256,
    }


def build_validation_data_manifest(
    dataset: AXISAnomalyQADataset,
    split_identity: Mapping,
    split_manifest_sha256: str,
) -> dict:
    exclusions = []
    question_type_counts: collections.Counter[str] = collections.Counter()
    included_question_type_counts: collections.Counter[str] = collections.Counter()
    total_rows = 0
    for series_path in dataset.series_files:
        record = json.loads(series_path.read_text(encoding="utf-8"))
        windows = record.get("windows", [])
        if len(windows) != 2:
            raise ValueError(
                f"{series_path.name} contains {len(windows)} QA rows; expected 2"
            )
        for window_index, item in enumerate(windows):
            answer = str(item.get("answer", ""))
            question_type = normalize_question_type(
                str(item.get("question_type", "")),
                answer,
            )
            total_rows += 1
            question_type_counts[question_type] += 1
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
                included_question_type_counts[question_type] += 1
    if total_rows != EXPECTED_VALIDATION_ROWS:
        raise ValueError(
            f"expected {EXPECTED_VALIDATION_ROWS} validation rows, got {total_rows}"
        )
    return {
        "schema_version": 1,
        "purpose": "loss_e2e_0723_checkpoint_selection",
        "split": dict(split_identity),
        "split_manifest_sha256": split_manifest_sha256,
        "qa_rows_total": total_rows,
        "qa_rows_included": total_rows - len(exclusions),
        "qa_rows_excluded": len(exclusions),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "included_question_type_counts": dict(
            sorted(included_question_type_counts.items())
        ),
        "exclusions": exclusions,
    }


def validate_checkpoint_identity(payload: Mapping, arm: str) -> dict:
    required = {
        "epoch",
        "global_step",
        "model_state_dict",
        "reproduction_meta",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"candidate checkpoint is missing fields: {missing}")
    step = int(payload["global_step"])
    if step not in CANDIDATE_STEP_TO_EPOCH:
        raise ValueError(f"checkpoint step {step} is not a declared milestone")
    epoch = int(payload["epoch"])
    if epoch != CANDIDATE_STEP_TO_EPOCH[step]:
        raise ValueError(
            f"checkpoint step {step} has epoch {epoch}, expected "
            f"{CANDIDATE_STEP_TO_EPOCH[step]}"
        )
    meta = payload["reproduction_meta"]
    expected = {
        "experiment": "loss_e2e_0723",
        "arm": arm,
        "run_purpose": "formal",
        "world_size": 3,
        "epochs": 2,
        "planned_steps": 19000,
        "seed": VALIDATION_SEED,
        "segment_alpha": 0.40,
        "lr": 1e-4,
        "weight_decay": 1e-5,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(
                f"checkpoint identity mismatch for {key}: "
                f"{meta.get(key)!r} != {value!r}"
            )
    if meta.get("source_dirty") is not False:
        raise ValueError("candidate checkpoint was produced from a dirty source tree")
    data_counts = meta.get("data_counts", {})
    expected_counts = {
        "train_series": EXPECTED_TRAIN_SERIES,
        "qa_rows_total": 57000,
        "qa_rows_included": 56987,
        "qa_rows_excluded": 13,
    }
    for key, value in expected_counts.items():
        if data_counts.get(key) != value:
            raise ValueError(
                f"checkpoint data count mismatch for {key}: "
                f"{data_counts.get(key)!r} != {value!r}"
            )
    state = payload["model_state_dict"]
    if arm == "treatment" and "fixed_hint_reference" not in state:
        raise ValueError("treatment candidate has no cached fixed-hint reference")
    if arm == "control" and "fixed_hint_reference" in state:
        raise ValueError(
            "control candidate unexpectedly contains a fixed-hint reference"
        )
    return {
        "step": step,
        "epoch": epoch,
        "checkpoint_source_commit": meta.get("source_commit"),
        "training_data_audit_sha256": meta.get("data_audit_sha256"),
        "author_checkpoint_sha256": meta.get("author_checkpoint_sha256"),
    }


def select_declared_checkpoint_paths(paths: Sequence[Path]) -> list[Path]:
    if len(paths) != len(CANDIDATE_STEP_TO_EPOCH):
        raise ValueError("exactly four checkpoint paths are required")
    by_step = {}
    for path in paths:
        match = path.stem.removeprefix("step_")
        if not match.isdigit():
            raise ValueError(f"candidate filename is not step_<N>.pth: {path.name}")
        step = int(match)
        if step in by_step:
            raise ValueError(f"duplicate checkpoint step: {step}")
        by_step[step] = path
    if set(by_step) != set(CANDIDATE_STEP_TO_EPOCH):
        raise ValueError("candidate steps must be exactly 4750, 9500, 14250, and 19000")
    return [by_step[step] for step in sorted(by_step)]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--checkpoints", nargs="+", required=True)
    result.add_argument("--arm", choices=["control", "treatment"], required=True)
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
    from src.models.AXIS.AXIS_test import collate_fn
    from src.models.AXIS.dataset import AXISAnomalyQADataset

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 3:
        raise ValueError("formal checkpoint selection requires exactly three ranks")
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(hours=2))
    random.seed(VALIDATION_SEED + rank)
    np.random.seed(VALIDATION_SEED + rank)
    torch.manual_seed(VALIDATION_SEED + rank)
    torch.cuda.manual_seed_all(VALIDATION_SEED + rank)

    checkpoint_paths = select_declared_checkpoint_paths(
        [Path(value) for value in args.checkpoints]
    )
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    split_path = Path(args.series_split_manifest)
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    split_manifest_sha256 = sha256_file(split_path)
    dataset = AXISAnomalyQADataset(
        args.data,
        split="val",
        train_ratio=TRAIN_RATIO,
        seed=VALIDATION_SEED,
    )
    dataset_series = [path.name for path in dataset.series_files]
    split_identity = validate_split_manifest(split_manifest, dataset_series)

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
    data_manifest_path = output / "validation_data_manifest.json"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(data_manifest_path, data_manifest)
    torch.distributed.barrier()
    data_manifest_sha256_box = [sha256_file(data_manifest_path) if rank == 0 else None]
    torch.distributed.broadcast_object_list(
        data_manifest_sha256_box,
        src=0,
    )
    data_manifest_sha256 = data_manifest_sha256_box[0]
    torch.distributed.barrier()

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
    if sum(len(range(worker, len(dataset), world)) for worker in range(world)) != len(
        dataset
    ):
        raise RuntimeError("validation rank partition is not exhaustive")

    base = build_model()
    freeze_for_phase2(base)
    if args.arm == "treatment":
        install_fixed_hint_runtime(base.axis)
    base.to(local_rank)
    base.eval()
    llm_dtypes = {
        str(parameter.dtype)
        for parameter in base.axis.model.parameters()
        if parameter.is_floating_point()
    }
    if llm_dtypes != {"torch.float16"}:
        raise RuntimeError(
            f"formal validation requires FP16 frozen LLM storage, got {llm_dtypes}"
        )
    evaluator = ContinuationObjectiveModel(
        base,
        objective_mode="control",
        segment_alpha=0.40,
        loss_chunk_size=args.loss_chunk_size,
    )
    evaluator.eval()

    summaries = []
    for checkpoint_path in checkpoint_paths:
        started = time.perf_counter()
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        identity = validate_checkpoint_identity(payload, args.arm)
        load_axis_payload(base, payload, strict=True)
        if args.arm == "treatment":
            set_fixed_hint_reference(
                base.axis,
                payload["model_state_dict"]["fixed_hint_reference"].to(local_rank),
            )
        base.eval()

        checkpoint_sha_box = [sha256_file(checkpoint_path) if rank == 0 else None]
        torch.distributed.broadcast_object_list(checkpoint_sha_box, src=0)
        local_totals = torch.zeros(5, dtype=torch.float64, device=local_rank)
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
                dtype=torch.float16,
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
            if rank == 0 and batch_index % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "step": identity["step"],
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
        token_nll_sum, token_count, valid_rows, total_rows, excluded_rows = (
            local_totals.tolist()
        )
        if not math.isfinite(token_nll_sum) or token_nll_sum <= 0:
            raise FloatingPointError(
                "validation token NLL sum is not finite and positive"
            )
        if token_count <= 0:
            raise RuntimeError("validation contains no effective answer tokens")
        expected = {
            "total_rows": data_manifest["qa_rows_total"],
            "valid_rows": data_manifest["qa_rows_included"],
            "excluded_rows": data_manifest["qa_rows_excluded"],
        }
        observed = {
            "total_rows": int(total_rows),
            "valid_rows": int(valid_rows),
            "excluded_rows": int(excluded_rows),
        }
        if observed != expected:
            raise RuntimeError(
                f"validation row audit mismatch: observed={observed}, expected={expected}"
            )
        summary = {
            "schema_version": 1,
            "experiment": "loss_e2e_0723",
            "arm": args.arm,
            "candidate_step": identity["step"],
            "candidate_epoch": identity["epoch"],
            "checkpoint_file": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha_box[0],
            "checkpoint_source_commit": identity["checkpoint_source_commit"],
            "author_checkpoint_sha256": identity["author_checkpoint_sha256"],
            "training_data_audit_sha256": identity["training_data_audit_sha256"],
            "selection_metric": (
                "effective-answer-token-weighted teacher-forced global token NLL"
            ),
            "effective_token_definition": EFFECTIVE_TOKEN_DEFINITION,
            "metric_direction": "minimize",
            "token_nll_sum": token_nll_sum,
            "effective_answer_token_count": int(token_count),
            "global_token_nll": token_nll_sum / token_count,
            "validation_series": len(dataset),
            **observed,
            "validation_seed": VALIDATION_SEED,
            "train_ratio": TRAIN_RATIO,
            "split_manifest_sha256": split_manifest_sha256,
            "validation_data_manifest_sha256": data_manifest_sha256,
            "rank_partition": "series_index_mod_world_size",
            "world_size": world,
            "batch_size_per_rank_series": 1,
            "frozen_llm_storage_dtype": "torch.float16",
            "autocast_dtype": "torch.float16",
            "loss_accumulation_dtype": "torch.float64",
            "loss_chunk_size": args.loss_chunk_size,
            "wall_seconds": time.perf_counter() - started,
            "evaluator_source_commit": _git_value("rev-parse", "HEAD"),
            "evaluator_source_dirty": bool(_git_value("status", "--porcelain")),
        }
        if rank == 0:
            _write_json_atomic(
                output / f"step_{identity['step']}.json",
                summary,
            )
            print(json.dumps(summary), flush=True)
        summaries.append(summary)
        del payload
        torch.distributed.barrier()

    if rank == 0:
        _write_json_atomic(
            output / "validation_run_summary.json",
            {
                "schema_version": 1,
                "experiment": "loss_e2e_0723",
                "arm": args.arm,
                "candidate_steps": [summary["candidate_step"] for summary in summaries],
                "candidates": summaries,
                "test_or_judge_metrics_consulted": False,
            },
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
