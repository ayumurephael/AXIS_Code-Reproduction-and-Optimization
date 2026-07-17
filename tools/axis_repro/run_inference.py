from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import torch

from .common import load_axis_records, read_jsonl
from .io_utils import append_jsonl
from .architecture_redesign import add_architecture_arguments
from .model_utils import build_model, load_axis_checkpoint, sha256_file


def dist_info():
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        # Validation records have highly variable generation lengths. Bind the
        # rank before NCCL initialization and allow fast ranks to wait for a
        # straggler without hitting the default ten-minute store timeout.
        torch.distributed.init_process_group("nccl", timeout=timedelta(hours=2))
    return rank, world, local


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/AXIS_qa_test")
    parser.add_argument("--subset", choices=["paper140", "full"], default="paper140")
    parser.add_argument("--series-split-manifest")
    parser.add_argument("--series-split-key", default="val_series")
    parser.add_argument("--modes", nargs="+", default=["base"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int)
    add_architecture_arguments(parser, default_variant="loss_only")
    return parser


def main() -> None:
    a = build_parser().parse_args()
    rank, world, local = dist_info()
    records = load_axis_records(a.data, a.subset, max_records=a.max_records)
    if a.series_split_manifest:
        allowed = set(json.loads(Path(a.series_split_manifest).read_text(encoding="utf-8"))[a.series_split_key])
        records = [r for r in records if r.series_file in allowed]
    records = records[rank::world]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    shard = out / f"rank{rank}.jsonl"
    done = {(x["record_id"], x["mode"]) for x in read_jsonl(shard)} if shard.exists() else set()

    model = build_model(
        architecture_variant=a.architecture_variant,
        qk_norm_seq_len=a.qk_norm_seq_len,
        gate_bias=a.gate_bias,
    )
    load_axis_checkpoint(model, a.checkpoint)
    model.to(torch.device("cuda", local)).eval()
    for r in records:
        ts = torch.tensor(r.time_series, dtype=torch.float32, device=local).unsqueeze(0)
        mask = torch.ones_like(ts, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            local_embeddings = model.ts_pretrain_model(ts, mask=mask)
            for mode in a.modes:
                if (r.record_id, mode) in done:
                    continue
                ablation = None if mode == "base" else mode
                loss = model.axis(local_embeddings, ts, [r.question], [r.answer],
                                  [r.start_index], [r.end_index], ablation_mode=ablation)
                response = model.axis.generate(local_embeddings, ts, [r.question], [r.answer],
                                               [r.start_index], [r.end_index], ablation_mode=ablation)[0]
                row = asdict(r); row.update({"mode": mode, "response": response, "loss": float(loss)})
                append_jsonl(shard, row); done.add((r.record_id, mode))
    if world > 1:
        torch.distributed.barrier(device_ids=[local])
        torch.distributed.destroy_process_group()
    if rank == 0:
        rows = []
        for i in range(world): rows.extend(read_jsonl(out / f"rank{i}.jsonl"))
        rows.sort(key=lambda x: (x["record_id"], x["mode"]))
        merged = out / "predictions.jsonl"
        merged.write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in rows), encoding="utf-8")
        (out / "run_manifest.json").write_text(json.dumps({
            "checkpoint": str(Path(a.checkpoint).resolve()), "checkpoint_sha256": sha256_file(a.checkpoint),
            "subset": a.subset, "world_size": world, "modes": a.modes, "rows": len(rows),
            "architecture_variant": a.architecture_variant,
            "qk_norm_seq_len": a.qk_norm_seq_len,
            "gate_bias": a.gate_bias,
        }, indent=2), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {merged}")


if __name__ == "__main__":
    main()

