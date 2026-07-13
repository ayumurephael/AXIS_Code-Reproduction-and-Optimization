from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.AXIS_test import collate_fn
from src.models.AXIS.dataset import AXISAnomalyQADataset
from .model_utils import build_model, checkpoint_payload, freeze_for_phase2, load_phase1_fresh_hint, sha256_file


def save_atomic(payload, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp); os.replace(tmp, path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--phase1", required=True)
    p.add_argument("--data", default="data/anomaly_llava_training_dataset")
    p.add_argument("--output", default="experiments/reproduction/phase2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=72)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--max-steps", type=int, help="For measured benchmark/smoke runs only")
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local); torch.distributed.init_process_group("nccl")
    random.seed(a.seed + rank); np.random.seed(a.seed + rank); torch.manual_seed(a.seed + rank)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)

    model = build_model(); load_phase1_fresh_hint(model, a.phase1)
    trainable = freeze_for_phase2(model); model.to(local)
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(ddp.module.axis.perceiver.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    dataset = AXISAnomalyQADataset(a.data, split="train", train_ratio=0.95, seed=a.seed)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=a.seed)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=a.num_workers,
                        pin_memory=True, persistent_workers=a.num_workers > 0, collate_fn=collate_fn)
    planned = a.epochs * len(loader); global_step = 0; run_start = time.perf_counter(); timed_start = None
    meta = {"phase1": str(Path(a.phase1).resolve()), "phase1_sha256": sha256_file(a.phase1),
            "world_size": world, "lr": a.lr, "weight_decay": a.weight_decay,
            "epochs": a.epochs, "seed": a.seed, "trainable_parameters": trainable,
            "train_series": len(dataset), "steps_per_rank_epoch": len(loader)}
    if rank == 0: print(json.dumps(meta), flush=True)
    stop = False
    for epoch in range(1, a.epochs + 1):
        sampler.set_epoch(epoch); ddp.train(); ddp.module.ts_pretrain_model.eval(); ddp.module.axis.model.eval()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            ts = batch["padded_sequences"].to(local, dtype=torch.float32, non_blocking=True)
            mask = batch["attention_masks"].to(local, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(ts, mask, batch["questions"], batch["answers"],
                           batch["start_indices"], batch["end_indices"])
            loss.backward(); optimizer.step(); global_step += 1
            if global_step == 10: timed_start = time.perf_counter()
            if rank == 0 and global_step % 20 == 0:
                elapsed = time.perf_counter() - (timed_start or run_start)
                measured = max(1, global_step - (10 if timed_start else 0)) / elapsed
                eta = max(0, planned-global_step) / measured
                print(json.dumps({"step": global_step, "loss": float(loss),
                                  "steps_per_second_per_rank": measured,
                                  "estimated_remaining_hours": eta/3600}), flush=True)
            if a.save_every and global_step % a.save_every == 0 and rank == 0:
                save_atomic(checkpoint_payload(ddp.module, optimizer, epoch, global_step, meta), out/f"step_{global_step}.pth")
            if a.max_steps and global_step >= a.max_steps: stop = True; break
        torch.distributed.barrier()
        if rank == 0:
            save_atomic(checkpoint_payload(ddp.module, optimizer, epoch, global_step, meta), out/f"epoch_{epoch}.pth")
        torch.distributed.barrier()
        if stop: break
    if rank == 0:
        elapsed = time.perf_counter()-run_start
        (out/"runtime.json").write_text(json.dumps({"world_size": world, "steps": global_step,
            "wall_seconds": elapsed, "series_per_second_global": global_step*world/elapsed,
            "projected_3epoch_hours": (3*len(loader))/(global_step/elapsed)/3600}, indent=2), encoding="utf-8")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
