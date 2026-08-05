from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist

from src.models.MultiAXIS.config import MultiAxisConfig
from src.models.MultiAXIS.data import ManifestDataset, collate_multiaxis
from src.models.MultiAXIS.model import MultiAxisForConditionalGeneration


DATASETS = ("478new", "SMD", "SWaT", "LEMMA-RCA", "VTA")


def setup():
    if "RANK" not in os.environ:
        raise RuntimeError("Formal inference must be launched with torchrun")
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, torch.device("cuda", local_rank)


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
    }


def dataset_indices(dataset: ManifestDataset, rank: int, world: int):
    # Whole groups stay together so the frozen TimeRCD result is reused exactly.
    return [index for group_number, group in enumerate(dataset.groups) if group_number % world == rank for index in group]


def run_dataset(model, dataset_name: str, manifest_dir: Path, data_root: Path, output_dir: Path, rank: int, world: int, device: torch.device):
    dataset = ManifestDataset(
        manifest_dir / f"eval_{dataset_name}.jsonl",
        data_root,
        window_epsilon=model.config.hints.window_epsilon,
        window_scale=model.config.hints.window_scale,
    )
    rank_path = output_dir / f"{dataset_name}.rank{rank}.jsonl"
    completed = read_existing(rank_path)
    indices = dataset_indices(dataset, rank, world)
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
            )[0]
            record = {
                "index": index,
                "sample_id": sample["sample_id"],
                "dataset": dataset_name,
                "question_group": sample["question_group"],
                "raw_response": response,
                "generation_seconds": time.monotonic() - started,
                "model": model.config.llm.model_name,
            }
            records.append(record)
            completed[index] = record
            atomic_write_jsonl(rank_path, sorted(records, key=lambda item: item["index"]))
    dist.barrier()
    if rank == 0:
        merged = []
        for source_rank in range(world):
            merged.extend(read_existing(output_dir / f"{dataset_name}.rank{source_rank}.jsonl").values())
        by_index = {int(record["index"]): record for record in merged}
        if len(by_index) != len(dataset):
            missing = sorted(set(range(len(dataset))) - set(by_index))
            raise RuntimeError(f"{dataset_name}: incomplete inference, missing {missing[:20]}")
        for index in range(len(dataset)):
            if by_index[index]["sample_id"] != dataset[index]["sample_id"]:
                raise RuntimeError(f"{dataset_name}: sample identity mismatch at {index}")
        atomic_write_jsonl(output_dir / f"{dataset_name}.predictions.jsonl", [by_index[index] for index in range(len(dataset))])
        (output_dir / f"{dataset_name}.complete.json").write_text(json.dumps({"dataset": dataset_name, "examples": len(dataset), "ranks": world, "model": model.config.llm.model_name}, indent=2), encoding="utf-8")
    dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--timercd-checkpoint", required=True)
    parser.add_argument("--hint-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    return parser.parse_args()


def main():
    args = parse_args()
    config = MultiAxisConfig.load_json(args.config)
    rank, world, device = setup()
    if world != config.training.expected_world_size:
        raise RuntimeError(f"Expected {config.training.expected_world_size} inference GPUs, got {world}")
    output_dir = Path(args.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    model = MultiAxisForConditionalGeneration.from_pretrained(config, token=os.environ.get("HF_TOKEN") or None, device=device)
    model.load_timercd_checkpoint(args.timercd_checkpoint)
    model.timercd.to(device)
    model.hint_tuner.to(device)
    checkpoint = model.load_hint_checkpoint(args.hint_checkpoint, strict=True)
    model.eval()
    if rank == 0:
        manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": config.llm.model_name,
            "pretrained_source": getattr(model, "pretrained_source", config.llm.model_name),
            "hint_checkpoint": str(Path(args.hint_checkpoint).resolve()),
            "hint_checkpoint_epoch": checkpoint.get("epoch"),
            "timercd_sha256": model.timercd.checkpoint_sha256,
            "world_size": world,
            "datasets": args.datasets,
            "generation": config.generation.__dict__,
        }
        (output_dir / "inference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    for dataset_name in args.datasets:
        run_dataset(model, dataset_name, Path(args.manifest_dir), Path(args.data_root), output_dir, rank, world, device)
    if rank == 0:
        (output_dir / "INFERENCE_COMPLETE").write_text(json.dumps({"datasets": args.datasets}), encoding="utf-8")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
