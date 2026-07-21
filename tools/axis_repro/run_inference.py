from __future__ import annotations

import argparse
import collections
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
    parser.add_argument(
        "--batching", choices=["per_record", "series"], default="per_record",
        help="series reproduces AXIS_test.py: all QA items for one time series "
             "are generated in the same batch.",
    )
    parser.add_argument(
        "--skip-loss", action="store_true",
        help="Skip teacher-forced loss during test generation, as in the "
             "author's AXIS_test.py path.",
    )
    add_architecture_arguments(parser, default_variant="loss_only")
    return parser


def main() -> None:
    a = build_parser().parse_args()
    rank, world, local = dist_info()
    records = load_axis_records(a.data, a.subset, max_records=a.max_records)
    if a.series_split_manifest:
        allowed = set(json.loads(Path(a.series_split_manifest).read_text(encoding="utf-8"))[a.series_split_key])
        records = [r for r in records if r.series_file in allowed]
    if a.batching == "series":
        grouped = collections.OrderedDict()
        for record in records:
            grouped.setdefault(record.series_file, []).append(record)
        work_items = list(grouped.values())[rank::world]
    else:
        work_items = [[record] for record in records[rank::world]]
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
    for group in work_items:
        first = group[0]
        # Repeat the series before the TS encoder, as the author's collate_fn
        # does. Repeating only its embedding is not numerically identical in
        # mixed precision.
        ts = torch.tensor(first.time_series, dtype=torch.float32, device=local)
        ts = ts.unsqueeze(0).repeat(len(group), 1)
        mask = torch.ones_like(ts, dtype=torch.bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            local_embeddings = model.ts_pretrain_model(ts, mask=mask)
            for mode in a.modes:
                pending = [r for r in group if (r.record_id, mode) not in done]
                if not pending:
                    continue
                if len(pending) != len(group):
                    raise RuntimeError(
                        "A partially completed series batch cannot be resumed "
                        "without changing batch semantics; remove only that "
                        "series from the shard and retry."
                    )
                ablation = None if mode == "base" else mode
                questions = [r.question for r in group]
                answers = [r.answer for r in group]
                starts = [r.start_index for r in group]
                ends = [r.end_index for r in group]
                loss = None
                if not a.skip_loss:
                    loss = model.axis(
                        local_embeddings, ts, questions, answers, starts, ends,
                        ablation_mode=ablation,
                    )
                responses = model.axis.generate(
                    local_embeddings, ts, questions, answers, starts, ends,
                    ablation_mode=ablation,
                )
                if len(responses) != len(group):
                    raise RuntimeError(
                        f"generate returned {len(responses)} responses for "
                        f"a series batch of {len(group)}"
                    )
                for r, response in zip(group, responses):
                    row = asdict(r)
                    row.update({
                        "mode": mode,
                        "response": response,
                        "loss": None if loss is None else float(loss),
                    })
                    append_jsonl(shard, row)
                    done.add((r.record_id, mode))
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
            "batching": a.batching, "skip_loss": a.skip_loss,
            "architecture_variant": a.architecture_variant,
            "qk_norm_seq_len": a.qk_norm_seq_len,
            "gate_bias": a.gate_bias,
        }, indent=2), encoding="utf-8")
        print(f"wrote {len(rows)} rows to {merged}")


if __name__ == "__main__":
    main()

