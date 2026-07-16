"""Phase-II training with source-likelihood-ratio counterfactual supervision."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from . import model_utils
from .loss_redesign import (
    SLR_MODES,
    CounterfactualAXISDataset,
    compute_source_likelihood_ratio_loss,
    counterfactual_collate_fn,
    evidence_mode_spec,
    memory_safe_token_nll_sums,
)
from .model_utils import checkpoint_payload, freeze_for_phase2, load_phase1_fresh_hint, sha256_file


_PLAIN_BUILD = model_utils.build_model


def _context_nll(
    axis,
    local_embeddings,
    time_series,
    questions,
    answers,
    start_indices,
    end_indices,
    question_types,
    *,
    evidence_mode=None,
    local_window_embeddings_override=None,
    window_values_override=None,
):
    ids, mask, labels, _, head_mask = axis.generate_input_ids_and_labels(
        questions,
        answers,
        time_series,
        start_indices,
        end_indices,
        question_types=question_types,
        window_values_override=window_values_override,
        evidence_mode=evidence_mode,
        return_answer_head_mask=True,
    )
    device = axis.get_device()
    ids = ids.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    head_mask = head_mask.to(device)
    embeds = axis.get_hint_embeddings(
        ids,
        local_embeddings,
        start_indices,
        end_indices,
        local_window_embeddings_override=local_window_embeddings_override,
    )
    hidden = axis.model.model(
        inputs_embeds=embeds,
        attention_mask=mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[:, :-1]
    return memory_safe_token_nll_sums(
        axis.model.lm_head,
        hidden,
        labels[:, 1:],
        head_mask[:, 1:],
        chunk_size=64,
        checkpoint_chunks=True,
    )


def _axis_loss_redesign_forward(
    self,
    local_embeddings,
    time_series,
    questions,
    answers,
    start_indices,
    end_indices,
    return_logits=False,
    ablation_mode=None,
    question_types=None,
    negative_local_window_embeddings=None,
    negative_window_values=None,
    slr_mode=None,
    source_valid=None,
    beta=1.0,
    margin=math.log(2.0),
):
    if return_logits:
        raise ValueError("loss-redesign training does not materialize full logits")
    if ablation_mode is not None:
        raise ValueError("training-time SLR is incompatible with inference ablation_mode")
    if question_types is None or slr_mode is None or source_valid is None:
        raise ValueError("question_types, slr_mode, and source_valid are required")

    full_sums, full_counts, full_head_sums, full_head_counts = _context_nll(
        self,
        local_embeddings,
        time_series,
        questions,
        answers,
        start_indices,
        end_indices,
        question_types,
    )
    answer_loss = full_sums.sum() / full_counts.sum().clamp_min(1)

    include_local, include_window, changed_source = evidence_mode_spec(slr_mode)
    if include_local and include_window:
        positive_head_sums = full_head_sums
        positive_head_counts = full_head_counts
    else:
        _, _, positive_head_sums, positive_head_counts = _context_nll(
            self,
            local_embeddings,
            time_series,
            questions,
            answers,
            start_indices,
            end_indices,
            question_types,
            evidence_mode=slr_mode,
        )

    local_override = negative_local_window_embeddings if changed_source == "local" else None
    window_override = negative_window_values if changed_source == "window" else None
    if changed_source == "local" and local_override is None:
        raise ValueError("Local SLR mode requires negative Local embeddings")
    if changed_source == "window" and window_override is None:
        raise ValueError("Window SLR mode requires negative Window values")
    _, _, negative_head_sums, negative_head_counts = _context_nll(
        self,
        local_embeddings,
        time_series,
        questions,
        answers,
        start_indices,
        end_indices,
        question_types,
        evidence_mode=slr_mode,
        local_window_embeddings_override=local_override,
        window_values_override=window_override,
    )
    valid = source_valid.to(answer_loss.device).bool()
    valid = valid & positive_head_counts.gt(0) & negative_head_counts.eq(positive_head_counts)
    loss, stats = compute_source_likelihood_ratio_loss(
        answer_loss,
        positive_head_sums,
        negative_head_sums,
        positive_head_counts,
        valid,
        beta=beta,
        margin=margin,
    )
    stats.update({
        "mode": slr_mode,
        "source": changed_source,
        "mean_positive_head_nll": float(positive_head_sums[valid].mean().detach()) if bool(valid.any()) else None,
        "mean_negative_head_nll": float(negative_head_sums[valid].mean().detach()) if bool(valid.any()) else None,
    })
    self._last_loss_stats = stats
    return loss


def build_model():
    model = _PLAIN_BUILD()
    llm = model.axis.model
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()
    llm.config.use_cache = False
    llm.get_input_embeddings().register_forward_hook(lambda _module, _inputs, output: output.clone())
    model.axis.forward = types.MethodType(_axis_loss_redesign_forward, model.axis)
    llm.eval = types.MethodType(lambda self: self, llm)
    return model


def save_atomic(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--counterfactual-index", required=True)
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--output", default="experiments/reproduction/phase2_loss_redesign")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=math.log(2.0))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    mode_rng = random.Random(args.seed + 100003 * rank)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = build_model()
    load_phase1_fresh_hint(model, args.phase1)
    trainable = freeze_for_phase2(model)
    model.to(local)
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    optimizer = torch.optim.AdamW(
        ddp.module.axis.perceiver.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    dataset = CounterfactualAXISDataset(
        args.data,
        args.counterfactual_index,
        split="train",
        train_ratio=0.95,
        seed=args.seed,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=counterfactual_collate_fn,
    )
    planned = args.epochs * len(loader)
    global_step = 0
    run_start = time.perf_counter()
    timed_start = None
    metadata = {
        "objective": "answer_nll_plus_source_likelihood_ratio",
        "phase1": str(Path(args.phase1).resolve()),
        "phase1_sha256": sha256_file(args.phase1),
        "counterfactual_index": str(Path(args.counterfactual_index).resolve()),
        "counterfactual_index_sha256": sha256_file(args.counterfactual_index),
        "world_size": world,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "beta": args.beta,
        "margin": args.margin,
        "modes": list(SLR_MODES),
        "trainable_parameters": trainable,
        "train_series": len(dataset),
        "steps_per_rank_epoch": len(loader),
    }
    if rank == 0:
        print(json.dumps(metadata), flush=True)

    stop = False
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        ddp.train()
        ddp.module.ts_pretrain_model.eval()
        ddp.module.axis.model.eval()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            mode = mode_rng.choice(SLR_MODES)
            _, _, source = evidence_mode_spec(mode)
            valid_key = "local_source_valid" if source == "local" else "window_source_valid"
            positive = batch["padded_sequences"].to(local, dtype=torch.float32, non_blocking=True)
            positive_mask = batch["attention_masks"].to(local, non_blocking=True)
            negative_local = batch["negative_local_sequences"].to(local, dtype=torch.float32, non_blocking=True)
            negative_local_mask = batch["negative_local_masks"].to(local, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(
                    positive,
                    positive_mask,
                    batch["questions"],
                    batch["answers"],
                    batch["start_indices"],
                    batch["end_indices"],
                    question_types=batch["question_types"],
                    negative_local_sequences=negative_local,
                    negative_local_masks=negative_local_mask,
                    negative_local_start_indices=batch["negative_local_start_indices"],
                    negative_local_end_indices=batch["negative_local_end_indices"],
                    negative_window_values=batch["negative_window_values"],
                    slr_mode=mode,
                    source_valid=batch[valid_key].to(local, non_blocking=True),
                    beta=args.beta,
                    margin=args.margin,
                )
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step == 10:
                timed_start = time.perf_counter()
            if rank == 0 and global_step % 20 == 0:
                elapsed = time.perf_counter() - (timed_start or run_start)
                measured = max(1, global_step - (10 if timed_start else 0)) / elapsed
                log = dict(ddp.module.axis._last_loss_stats)
                log.update({
                    "step": global_step,
                    "loss": float(loss.detach()),
                    "steps_per_second_per_rank": measured,
                    "estimated_remaining_hours": max(0, planned - global_step) / measured / 3600,
                })
                print(json.dumps(log), flush=True)
            if args.save_every and global_step % args.save_every == 0 and rank == 0:
                save_atomic(
                    checkpoint_payload(ddp.module, optimizer, epoch, global_step, metadata),
                    output / f"step_{global_step}.pth",
                )
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        torch.distributed.barrier()
        if rank == 0:
            save_atomic(
                checkpoint_payload(ddp.module, optimizer, epoch, global_step, metadata),
                output / f"epoch_{epoch}.pth",
            )
        torch.distributed.barrier()
        if stop:
            break

    if rank == 0:
        elapsed = time.perf_counter() - run_start
        (output / "runtime.json").write_text(
            json.dumps({
                "world_size": world,
                "steps": global_step,
                "wall_seconds": elapsed,
                "series_per_second_global": global_step * world / elapsed,
                "projected_3epoch_hours": (3 * len(loader)) / (global_step / elapsed) / 3600,
            }, indent=2),
            encoding="utf-8",
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
