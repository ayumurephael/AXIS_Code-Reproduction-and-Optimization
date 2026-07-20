"""Author-compatible series-batched inference, including legacy checkpoints."""
from __future__ import annotations

import argparse
import collections
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from . import common
try:
    from .io_utils import append_jsonl
    common.append_jsonl = append_jsonl
except ImportError:
    append_jsonl = common.append_jsonl
from .common import load_axis_records, read_jsonl
from .model_utils import build_model, sha256_file


def load_compatible_checkpoint(model, path: str | Path, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    model.ts_pretrain_model.load_state_dict(state["ts_pretrain_model"], strict=strict)
    perceiver = state["moirai_trainable"]
    prefix = "perceiver."
    if perceiver and all(key.startswith(prefix) for key in perceiver):
        perceiver = {
            key[len(prefix):]: value for key, value in perceiver.items()
        }
    model.axis.perceiver.load_state_dict(perceiver, strict=strict)
    return payload


def dist_info() -> tuple[int, int, int]:
    rank = int(os.getenv("RANK", "0"))
    world = int(os.getenv("WORLD_SIZE", "1"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        torch.distributed.init_process_group("nccl")
    return rank, world, local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/AXIS_qa_test")
    parser.add_argument("--model-path")
    parser.add_argument("--subset", choices=["paper140", "full"], default="paper140")
    parser.add_argument("--modes", nargs="+", default=["base"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--batching", choices=["per_record", "series"], default="series"
    )
    parser.add_argument("--compute-loss", action="store_true")
    args = parser.parse_args()

    rank, world, local = dist_info()
    records = load_axis_records(
        args.data, args.subset, max_records=args.max_records
    )
    if args.batching == "series":
        grouped = collections.OrderedDict()
        for record in records:
            grouped.setdefault(record.series_file, []).append(record)
        groups = list(grouped.values())[rank::world]
    else:
        groups = [[record] for record in records[rank::world]]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    shard = output / f"rank{rank}.jsonl"
    done = {
        (row["record_id"], row["mode"]) for row in read_jsonl(shard)
    } if shard.exists() else set()

    if args.model_path:
        from experiments.configs.axis_config import default_config
        default_config.llm_config.model_name = args.model_path
    model = build_model()
    checkpoint = load_compatible_checkpoint(model, args.checkpoint)
    model.to(torch.device("cuda", local)).eval()
    for group in groups:
        first = group[0]
        series = torch.tensor(
            first.time_series, dtype=torch.float32, device=local
        ).unsqueeze(0).repeat(len(group), 1)
        mask = torch.ones_like(series, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            embeddings = model.ts_pretrain_model(series, mask=mask)
            questions = [row.question for row in group]
            answers = [row.answer for row in group]
            starts = [row.start_index for row in group]
            ends = [row.end_index for row in group]
            for mode in args.modes:
                if all((row.record_id, mode) in done for row in group):
                    continue
                if any((row.record_id, mode) in done for row in group):
                    raise RuntimeError("partially completed series batch")
                ablation = None if mode == "base" else mode
                loss = None
                if args.compute_loss:
                    loss = model.axis(
                        embeddings, series, questions, answers, starts, ends,
                        ablation_mode=ablation,
                    )
                responses = model.axis.generate(
                    embeddings, series, questions, answers, starts, ends,
                    ablation_mode=ablation,
                )
                if len(responses) != len(group):
                    raise RuntimeError("response count differs from series batch")
                for row, response in zip(group, responses):
                    payload = asdict(row)
                    payload.update({
                        "mode": mode,
                        "response": response,
                        "loss": None if loss is None else float(loss),
                    })
                    append_jsonl(shard, payload)
                    done.add((row.record_id, mode))

    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        rows = []
        for index in range(world):
            rows.extend(read_jsonl(output / f"rank{index}.jsonl"))
        rows.sort(key=lambda row: (row["record_id"], row["mode"]))
        merged = output / "predictions.jsonl"
        merged.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_avg_loss": checkpoint.get("avg_loss"),
            "subset": args.subset,
            "world_size": world,
            "modes": args.modes,
            "rows": len(rows),
            "batching": args.batching,
            "compute_loss": args.compute_loss,
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"wrote {len(rows)} rows to {merged}")
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
