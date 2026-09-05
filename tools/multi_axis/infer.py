from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from src.models.MultiAXIS.config import MultiAxisConfig
from src.models.MultiAXIS.ablation import (
    ABLATION_VARIANTS,
    FULL_VARIANT,
    ablation_spec,
)
from src.models.MultiAXIS.data import ManifestDataset, collate_multiaxis
from src.models.MultiAXIS.model import MultiAxisForConditionalGeneration
from src.models.MultiAXIS.response_contracts import response_contract_error
from src.models.MultiAXIS.timercd import sha256_file
from tools.multi_axis.datasets import (
    OFFICIAL_BIAS_NEUTRALIZED_DATASETS,
    SUPPORTED_EVALUATION_DATASETS,
    validate_datasets,
)
from tools.multi_axis.distributed import collect_distributed_topology


DATASETS = OFFICIAL_BIAS_NEUTRALIZED_DATASETS


def source_commit() -> str | None:
    explicit = os.environ.get("MULTI_AXIS_SOURCE_COMMIT")
    if explicit:
        value = explicit.strip().lower()
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("MULTI_AXIS_SOURCE_COMMIT must be a full Git commit hash")
        return value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip().lower()
    return value if len(value) == 40 else None


def setup():
    if "RANK" not in os.environ:
        raise RuntimeError("Formal inference must be launched with torchrun")
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device("cuda", local_rank)


def atomic_write_jsonl(path: Path, records: List[Dict]):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_existing(path: Path):
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[int(record["index"])] = record
    return records


def model_inputs(batch, device):
    return {
        "normalized_series": batch["normalized_series"].to(device),
        "time_mask": batch["time_mask"].to(device),
        "channel_mask": batch["channel_mask"].to(device),
        "questions": batch["questions"],
        "intervals": batch["intervals"],
        "channel_counts": batch["channel_counts"],
        "window_values": batch["window_values"],
        "channel_means": batch["channel_means"],
        "channel_stds": batch["channel_stds"],
        "channel_ids": batch["channel_ids"],
        "question_groups": batch["question_groups"],
        "image_paths": batch["image_paths"],
    }


def dataset_indices(
    dataset: ManifestDataset,
    rank: int,
    world: int,
    shard_count: int = 1,
    shard_indices: tuple[int, ...] = (0,),
):
    # Whole base-sample groups stay together so TimeRCD reuse and shard boundaries
    # are exact. Selected groups are then balanced across ranks in this run.
    selected = [
        group
        for group_number, group in enumerate(dataset.groups)
        if group_number % shard_count in set(shard_indices)
    ]
    return [
        index
        for selected_number, group in enumerate(selected)
        if selected_number % world == rank
        for index in group
    ]


def run_dataset(
    model,
    dataset_name: str,
    manifest_dir: Path,
    data_root: Path,
    image_root: Path | None,
    output_dir: Path,
    rank: int,
    world: int,
    device: torch.device,
    shard_count: int,
    shard_indices: tuple[int, ...],
    ablation_variant: str,
):
    spec = ablation_spec(ablation_variant)
    dataset = ManifestDataset(
        manifest_dir / f"eval_{dataset_name}.jsonl",
        data_root,
        window_epsilon=model.config.hints.window_epsilon,
        window_scale=model.config.hints.window_scale,
        image_root=image_root,
        require_image=model.config.vision.enabled,
        renderer_version=model.config.vision.renderer_version,
    )
    rank_path = output_dir / f"{dataset_name}.rank{rank}.jsonl"
    completed = read_existing(rank_path)
    indices = dataset_indices(
        dataset,
        rank,
        world,
        shard_count=shard_count,
        shard_indices=shard_indices,
    )
    cache: Dict = {}
    records = list(completed.values())
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prototype = model.build_prototype_bank()
        for position, index in enumerate(indices, 1):
            if index in completed:
                continue
            sample = dataset[index]
            batch = collate_multiaxis([sample])
            inputs = model_inputs(batch, device)
            if cache.get("base_sample_id") != sample["base_sample_id"]:
                encoded = model.timercd(inputs["normalized_series"], inputs["time_mask"], inputs["channel_mask"])
                cache = {"base_sample_id": sample["base_sample_id"], "encoded": encoded}
            started = time.monotonic()
            response = model.generate_answers(
                **inputs,
                prototype_override=prototype,
                timercd_override=cache["encoded"],
                ablation_variant=ablation_variant,
            )[0]
            record = {
                "index": index,
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "question_group": sample["question_group"],
                "raw_response": response,
                "contract_error": response_contract_error(
                    response, sample["question_group"]
                ),
                "generation_seconds": time.monotonic() - started,
                "model": model.config.llm.model_name,
                "ablation_variant": ablation_variant,
                "image_id": sample["image_id"] if spec.visual else None,
                "native_generation_metadata_audit": getattr(
                    model, "last_generation_metadata_audit", None
                ),
            }
            records.append(record)
            completed[index] = record
            atomic_write_jsonl(rank_path, sorted(records, key=lambda item: item["index"]))
    dist.barrier()
    if rank == 0:
        expected_indices = sorted(
            index
            for source_rank in range(world)
            for index in dataset_indices(
                dataset,
                source_rank,
                world,
                shard_count=shard_count,
                shard_indices=shard_indices,
            )
        )
        merged = []
        for source_rank in range(world):
            merged.extend(read_existing(output_dir / f"{dataset_name}.rank{source_rank}.jsonl").values())
        by_index = {int(record["index"]): record for record in merged}
        if set(by_index) != set(expected_indices):
            missing = sorted(set(expected_indices) - set(by_index))
            unexpected = sorted(set(by_index) - set(expected_indices))
            raise RuntimeError(
                f"{dataset_name}: shard inference identity mismatch; "
                f"missing={missing[:20]}, unexpected={unexpected[:20]}"
            )
        for index in expected_indices:
            if by_index[index]["sample_id"] != dataset[index]["sample_id"]:
                raise RuntimeError(f"{dataset_name}: sample identity mismatch at {index}")
        atomic_write_jsonl(
            output_dir / f"{dataset_name}.predictions.jsonl",
            [by_index[index] for index in expected_indices],
        )
        (output_dir / f"{dataset_name}.complete.json").write_text(
            json.dumps(
                {
                    "dataset": dataset_name,
                    "examples": len(expected_indices),
                    "dataset_examples": len(dataset),
                    "ranks": world,
                    "model": model.config.llm.model_name,
                    "ablation_variant": ablation_variant,
                    "ablation_spec": spec.to_dict(),
                    "inference_shard_count": shard_count,
                    "inference_shard_indices": list(shard_indices),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--timercd-checkpoint", required=True)
    parser.add_argument("--hint-checkpoint", required=True)
    parser.add_argument("--question-semantic-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_EVALUATION_DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--expected-inference-world-size",
        type=int,
        help="Explicit audited inference world size; defaults to the training profile.",
    )
    parser.add_argument(
        "--expected-inference-nodes",
        type=int,
        default=1,
        help="Audited node count for this torchrun invocation (default: one host).",
    )
    parser.add_argument(
        "--expected-gpu-substring",
        choices=("A100", "A800", "H100", "H800"),
        help="Required substring in every GPU model name for this inference worker.",
    )
    parser.add_argument(
        "--ablation-variant",
        choices=ABLATION_VARIANTS,
        default=FULL_VARIANT,
        help="Inference-time evidence-interface intervention.",
    )
    parser.add_argument("--inference-shard-count", type=int, default=1)
    parser.add_argument(
        "--inference-shard-indices",
        type=int,
        nargs="+",
        default=[0],
        help="Group-shard residues assigned to this inference run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.datasets = list(validate_datasets(args.datasets))
    config = MultiAxisConfig.load_json(args.config)
    if config.vision.enabled and not args.image_root:
        raise ValueError("--image-root is required for image-enabled inference")
    rank, world, local_rank, device = setup()
    expected_inference_world_size = (
        args.expected_inference_world_size or config.training.expected_world_size
    )
    if expected_inference_world_size <= 0:
        raise ValueError("--expected-inference-world-size must be positive")
    if args.expected_inference_nodes <= 0:
        raise ValueError("--expected-inference-nodes must be positive")
    shard_indices = tuple(args.inference_shard_indices)
    if args.inference_shard_count <= 0:
        raise ValueError("--inference-shard-count must be positive")
    if len(shard_indices) != len(set(shard_indices)):
        raise ValueError("--inference-shard-indices must be unique")
    if any(not 0 <= index < args.inference_shard_count for index in shard_indices):
        raise ValueError("Every inference shard index must be in [0, shard_count)")
    if world != expected_inference_world_size:
        raise RuntimeError(
            f"Expected {expected_inference_world_size} inference GPUs, got {world}"
        )
    topology = collect_distributed_topology(
        rank,
        world,
        local_rank,
        expected_nodes=args.expected_inference_nodes,
        required_gpu_substring=args.expected_gpu_substring,
    )
    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model = MultiAxisForConditionalGeneration.from_pretrained(config, token=os.environ.get("HF_TOKEN") or None, device=device)
    model.configure_question_semantic_cache(args.question_semantic_cache_dir)
    model.load_timercd_checkpoint(args.timercd_checkpoint)
    model.timercd.to(device)
    model.hint_tuner.to(device)
    checkpoint = model.load_hint_checkpoint(args.hint_checkpoint, strict=True)
    model.eval()
    attention_audit = model.attention_backend_audit()
    if rank == 0:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": config.llm.model_name,
            "pretrained_source": getattr(model, "pretrained_source", config.llm.model_name),
            "hint_checkpoint": str(Path(args.hint_checkpoint).resolve()),
            "hint_checkpoint_sha256": sha256_file(args.hint_checkpoint),
            "hint_checkpoint_epoch": checkpoint.get("epoch"),
            "timercd_sha256": model.timercd.checkpoint_sha256,
            "world_size": world,
            "training_world_size": config.training.expected_world_size,
            "expected_inference_world_size": expected_inference_world_size,
            "distributed_topology": topology,
            "datasets": args.datasets,
            "generation": config.generation.__dict__,
            "llm": config.llm.__dict__,
            "vision": config.vision.__dict__,
            "attention_backend_audit": attention_audit,
            "inference_source_commit": source_commit(),
            "ablation_variant": args.ablation_variant,
            "ablation_spec": ablation_spec(args.ablation_variant).to_dict(),
            "inference_shard_count": args.inference_shard_count,
            "inference_shard_indices": list(shard_indices),
        }
        (output_dir / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    for dataset_name in args.datasets:
        run_dataset(
            model,
            dataset_name,
            Path(args.manifest_dir),
            Path(args.data_root),
            Path(args.image_root) if args.image_root else None,
            output_dir,
            rank,
            world,
            device,
            args.inference_shard_count,
            shard_indices,
            args.ablation_variant,
        )
    if rank == 0:
        (output_dir / "INFERENCE_COMPLETE").write_text(
            json.dumps(
                {
                    "datasets": args.datasets,
                    "inference_shard_count": args.inference_shard_count,
                    "inference_shard_indices": list(shard_indices),
                    "ablation_variant": args.ablation_variant,
                }
            ),
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
