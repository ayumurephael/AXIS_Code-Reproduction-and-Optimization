"""Matched post-training from the released author Phase-II checkpoint.

The control and treatment arms reset identical optimizers and consume the same
factual answer stream. The treatment adds one QA-row auxiliary component per
step from an exact 4:3:1 MC/TF/OE schedule. This entry point is intentionally
separate from the Phase-I recovery trainer so released-checkpoint semantics
cannot be enabled accidentally in the canonical reproduction path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.models.AXIS.dataset import AXISAnomalyQADataset
from .architecture_redesign import ARCHITECTURE_VARIANTS, add_architecture_arguments
from .author_posttrain import (
    COMPONENT_ORDER,
    FixedIndexSampler,
    calibrated_beta,
    cosine_warmup_multiplier,
    distributed_cyclic_indices,
    gradient_l2_norm,
    stratified_component_schedule,
)
from .loss_redesign import (
    COUNTERFACTUAL_INDEX_VERSION,
    CounterfactualAXISDataset,
    CounterfactualQAView,
    answer_collate_fn,
    counterfactual_collate_fn,
    scheduled_beta,
    select_state_verbalizers,
)
from .model_utils import checkpoint_payload, freeze_for_phase2, load_axis_checkpoint, sha256_file
from .question_type_objectives import SUPERVISION_CACHE_VERSION, select_tf_verbalizers
from .train_phase2_loss_final import _empty_component_stats, _reduce_step_stats, build_model
from .train_phase2_loss_redesign import phase2_parameter_groups, save_atomic


TARGET_GRADIENT_RATIOS = {"mc": 0.15, "tf": 0.12, "oe": 0.08}
BETA_BOUNDS = {"mc": (0.02, 0.50), "tf": (0.02, 0.50), "oe": (0.01, 0.20)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--init-phase2", required=True)
    parser.add_argument("--expected-init-sha256", required=True)
    parser.add_argument("--counterfactual-index")
    parser.add_argument("--supervision-cache")
    parser.add_argument("--expected-donor-top-m", type=int, default=1)
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1600)
    parser.add_argument("--calibration-steps", type=int, default=200)
    parser.add_argument("--local-lr", type=float, default=1e-5)
    parser.add_argument("--attention-lr", type=float, default=2e-5)
    parser.add_argument("--prompt-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--lr-warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-final-ratio", type=float, default=0.10)
    parser.add_argument("--beta-warmup-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=400)
    add_architecture_arguments(parser, default_variant="loss_only")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.architecture_variant != "loss_only" or args.qk_norm_seq_len is not None:
        raise ValueError("author post-training requires the exact loss_only architecture")
    if args.epochs != 1:
        raise ValueError("the predeclared primary experiment contains exactly one auxiliary epoch")
    if args.steps_per_epoch <= 0 or args.steps_per_epoch % 8:
        raise ValueError("--steps-per-epoch must be a positive multiple of eight")
    if args.calibration_steps <= 0 or args.calibration_steps % 8:
        raise ValueError("--calibration-steps must be a positive multiple of eight")
    if args.arm == "treatment" and not (args.counterfactual_index and args.supervision_cache):
        raise ValueError("treatment requires the counterfactual index and supervision cache")
    if args.arm == "control" and (args.counterfactual_index or args.supervision_cache):
        raise ValueError("control must not materialize counterfactual data")
    actual_sha = sha256_file(args.init_phase2)
    if actual_sha.lower() != args.expected_init_sha256.lower():
        raise ValueError(f"released author initializer SHA mismatch: {actual_sha}")


def _set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def _answer_loader(dataset, *, world: int, rank: int, seed: int, epoch: int, workers: int):
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    sampler.set_epoch(epoch)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=answer_collate_fn,
    )
    return loader


def _auxiliary_loaders(
    view: CounterfactualQAView,
    schedule: list[str],
    *,
    world: int,
    rank: int,
    seed: int,
    epoch: int,
    workers: int,
) -> dict[str, DataLoader]:
    counts = Counter(schedule)
    loaders = {}
    for component in COMPONENT_ORDER:
        indices = distributed_cyclic_indices(
            view.positions_by_component[component],
            local_count=counts[component],
            world_size=world,
            rank=rank,
            seed=seed,
            component=component,
            epoch=epoch,
        )
        loaders[component] = DataLoader(
            view,
            batch_size=1,
            sampler=FixedIndexSampler(indices),
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            collate_fn=counterfactual_collate_fn,
        )
    return loaders


def _clone_stats(axis) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in axis._last_component_stats.items()}


def _answer_forward(ddp: DDP, batch: dict, device: int) -> tuple[torch.Tensor, dict]:
    positive = batch["padded_sequences"].to(device, dtype=torch.float32, non_blocking=True)
    mask = batch["attention_masks"].to(device, non_blocking=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        local_embeddings = ddp.module.ts_pretrain_model(positive, mask=mask)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = ddp(
            positive,
            mask,
            batch["questions"],
            batch["answers"],
            batch["start_indices"],
            batch["end_indices"],
            precomputed_local_embeddings=local_embeddings,
            question_types=batch["question_types"],
            loss_component="answer",
        )
    return loss, _clone_stats(ddp.module.axis)


def _auxiliary_forward(
    ddp: DDP,
    batch: dict,
    component: str,
    beta: float,
    device: int,
    state_verbalizers,
    tf_verbalizers,
) -> tuple[torch.Tensor, dict]:
    positive = batch["padded_sequences"].to(device, dtype=torch.float32, non_blocking=True)
    positive_mask = batch["attention_masks"].to(device, non_blocking=True)
    counterfactual = batch["counterfactual_sequences"].to(
        device, dtype=torch.float32, non_blocking=True
    )
    counterfactual_mask = batch["counterfactual_masks"].to(device, non_blocking=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        local_embeddings = ddp.module.ts_pretrain_model(positive, mask=positive_mask)
        counterfactual_embeddings = ddp.module.ts_pretrain_model(
            counterfactual, mask=counterfactual_mask
        )
    betas = {name: 0.0 for name in COMPONENT_ORDER}
    betas[component] = float(beta)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = ddp(
            positive,
            positive_mask,
            batch["questions"],
            batch["answers"],
            batch["start_indices"],
            batch["end_indices"],
            precomputed_local_embeddings=local_embeddings,
            precomputed_counterfactual_embeddings=counterfactual_embeddings,
            counterfactual_sequences=counterfactual,
            question_types=batch["question_types"],
            positive_window_values=batch["positive_window_values"],
            counterfactual_window_values=batch["counterfactual_window_values"],
            state_targets=batch["state_targets"].to(device, non_blocking=True),
            mc_pair_valid=batch["mc_pair_valid"].to(device, non_blocking=True),
            tf_pair_valid=batch["tf_pair_valid"].to(device, non_blocking=True),
            tf_targets=batch["tf_targets"].to(device, non_blocking=True),
            oe_pair_valid=batch["oe_pair_valid"].to(device, non_blocking=True),
            state_question=state_verbalizers.question,
            normal_id=state_verbalizers.normal_id,
            anomalous_id=state_verbalizers.anomalous_id,
            tf_false_id=tf_verbalizers.false_id,
            tf_true_id=tf_verbalizers.true_id,
            beta_mc=betas["mc"],
            beta_tf=betas["tf"],
            beta_oe=betas["oe"],
            loss_component=component,
        )
    return loss, _clone_stats(ddp.module.axis)


def _calibrate_betas(
    ddp: DDP,
    optimizer,
    answer_dataset,
    qa_view,
    *,
    steps: int,
    world: int,
    rank: int,
    seed: int,
    workers: int,
    device: int,
    state_verbalizers,
    tf_verbalizers,
) -> tuple[dict[str, float], dict]:
    schedule = stratified_component_schedule(steps, seed=seed, epoch=0)
    answer_loader = _answer_loader(
        answer_dataset, world=world, rank=rank, seed=seed + 100003, epoch=0, workers=workers
    )
    auxiliary_loaders = _auxiliary_loaders(
        qa_view,
        schedule,
        world=world,
        rank=rank,
        seed=seed + 200003,
        epoch=0,
        workers=workers,
    )
    answer_iterator = iter(answer_loader)
    auxiliary_iterators = {name: iter(loader) for name, loader in auxiliary_loaders.items()}
    evidence_parameters = [
        parameter
        for name, parameter in ddp.module.axis.perceiver.named_parameters()
        if parameter.requires_grad and name != "fix_prompt_embeddings"
    ]
    norms = {
        component: {"answer": [], "auxiliary": []}
        for component in COMPONENT_ORDER
    }
    ddp.train()
    ddp.module.ts_pretrain_model.eval()
    ddp.module.axis.model.eval()
    for step, component in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        answer_loss, _stats = _answer_forward(ddp, next(answer_iterator), device)
        answer_loss.backward()
        answer_norm = float(gradient_l2_norm(evidence_parameters).item())

        optimizer.zero_grad(set_to_none=True)
        auxiliary_loss, _stats = _auxiliary_forward(
            ddp,
            next(auxiliary_iterators[component]),
            component,
            1.0,
            device,
            state_verbalizers,
            tf_verbalizers,
        )
        auxiliary_loss.backward()
        auxiliary_norm = float(gradient_l2_norm(evidence_parameters).item())
        norms[component]["answer"].append(answer_norm)
        norms[component]["auxiliary"].append(auxiliary_norm)
        optimizer.zero_grad(set_to_none=True)
        if rank == 0 and step % 20 == 0:
            print(json.dumps({
                "event": "beta_calibration_progress",
                "step": step,
                "steps": steps,
                "component": component,
                "answer_grad_norm": answer_norm,
                "auxiliary_grad_norm": auxiliary_norm,
            }), flush=True)

    betas = {}
    summary = {}
    for component in COMPONENT_ORDER:
        lower, upper = BETA_BOUNDS[component]
        betas[component] = calibrated_beta(
            norms[component]["answer"],
            norms[component]["auxiliary"],
            target_ratio=TARGET_GRADIENT_RATIOS[component],
            lower=lower,
            upper=upper,
        )
        summary[component] = {
            "samples": len(norms[component]["answer"]),
            "answer_grad_norm_median": float(torch.tensor(norms[component]["answer"]).median()),
            "auxiliary_grad_norm_median": float(torch.tensor(norms[component]["auxiliary"]).median()),
            "target_gradient_ratio": TARGET_GRADIENT_RATIOS[component],
            "beta_bounds": [lower, upper],
            "calibrated_beta": betas[component],
        }
    beta_tensor = torch.tensor([betas[name] for name in COMPONENT_ORDER], device=device)
    torch.distributed.broadcast(beta_tensor, src=0)
    betas = {name: float(beta_tensor[index].item()) for index, name in enumerate(COMPONENT_ORDER)}
    torch.distributed.barrier()
    return betas, summary


def _epoch_summary(values: torch.Tensor, steps: int, world: int) -> dict:
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    data = values.tolist()

    def mean(sum_index: int, count_index: int) -> float:
        return data[sum_index] / data[count_index] if data[count_index] else 0.0

    return {
        "answer_loss": mean(0, 1),
        "mc_state_loss": mean(2, 3),
        "mc_pairs": int(data[4]),
        "mc_accuracy": data[5] / data[3] if data[3] else 0.0,
        "tf_q_loss": mean(8, 9),
        "tf_pairs": int(data[10]),
        "tf_accuracy": data[11] / data[9] if data[9] else 0.0,
        "oe_qcar_loss": mean(14, 15),
        "oe_pairs": int(data[16]),
        "oe_mean_delta": data[17] / data[16] if data[16] else 0.0,
        "oe_positive_delta_rate": data[18] / data[16] if data[16] else 0.0,
        "mean_grad_norm": data[19] / (steps * world),
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    if world != 3:
        raise ValueError("the predeclared matched experiment requires exactly three GPUs")
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl")
    _set_seed(args.seed, rank)

    phase1_sha = sha256_file(args.phase1)
    initializer_sha = sha256_file(args.init_phase2)
    answer_dataset = AXISAnomalyQADataset(
        args.data, split="train", train_ratio=0.95, seed=args.seed
    )
    counterfactual_dataset = None
    qa_view = None
    if args.arm == "treatment":
        counterfactual_dataset = CounterfactualAXISDataset(
            args.data,
            args.counterfactual_index,
            split="train",
            train_ratio=0.95,
            seed=args.seed,
            supervision_cache=args.supervision_cache,
        )
        if counterfactual_dataset.index_metadata.get("phase1_sha256") != phase1_sha:
            raise ValueError("counterfactual index Phase-I SHA mismatch")
        actual_top_m = int(
            counterfactual_dataset.index_metadata.get("policy", {}).get("donor_top_m", -1)
        )
        if actual_top_m != args.expected_donor_top_m:
            raise ValueError(f"formal donor Top-M {actual_top_m} != {args.expected_donor_top_m}")
        qa_view = CounterfactualQAView(counterfactual_dataset)

    model = build_model("loss_only", None, args.gate_bias)
    initialization_payload = load_axis_checkpoint(model, args.init_phase2, strict=True)
    if initialization_payload.get("reproduction_meta"):
        raise ValueError("post-training initializer must be the released author checkpoint")
    trainable = freeze_for_phase2(model)
    state_verbalizers = select_state_verbalizers(model.axis.tokenizer)
    tf_verbalizers = select_tf_verbalizers(model.axis.tokenizer)
    model.to(local)
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    optimizer_groups, optimizer_metadata = phase2_parameter_groups(
        ddp.module.axis.perceiver,
        local_lr=args.local_lr,
        attention_lr=args.attention_lr,
        prompt_lr=args.prompt_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups, betas=(0.9, 0.95), eps=1e-10
    )
    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    if args.arm == "treatment":
        calibrated_betas, calibration = _calibrate_betas(
            ddp,
            optimizer,
            answer_dataset,
            qa_view,
            steps=args.calibration_steps,
            world=world,
            rank=rank,
            seed=args.seed,
            workers=args.num_workers,
            device=local,
            state_verbalizers=state_verbalizers,
            tf_verbalizers=tf_verbalizers,
        )
    else:
        calibrated_betas = {component: 0.0 for component in COMPONENT_ORDER}
        calibration = None
    _set_seed(args.seed, rank)
    optimizer.zero_grad(set_to_none=True)

    architecture_flags = ARCHITECTURE_VARIANTS["loss_only"]
    pool_counts = (
        {name: len(qa_view.positions_by_component[name]) for name in COMPONENT_ORDER}
        if qa_view is not None else None
    )
    local_schedule_counts = Counter(
        stratified_component_schedule(args.steps_per_epoch, seed=args.seed)
    ) if args.arm == "treatment" else None
    global_exposure = (
        {name: int(local_schedule_counts[name] * world) for name in COMPONENT_ORDER}
        if local_schedule_counts is not None else None
    )
    metadata = {
        "objective": (
            "author_posttrain_answer_control_v1"
            if args.arm == "control" else "author_posttrain_question_type_joint_v1"
        ),
        "objective_version": 6,
        "stage": "author_checkpoint_posttrain",
        "arm": args.arm,
        "phase1": str(Path(args.phase1).resolve()),
        "phase1_sha256": phase1_sha,
        "init_phase2": str(Path(args.init_phase2).resolve()),
        "init_phase2_sha256": initializer_sha,
        "initializer_kind": "released_author_phase2_best",
        "optimizer_reset": True,
        "resume_optimizer": False,
        "world_size": world,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "seed": args.seed,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1e-10},
        "optimizer_groups": optimizer_metadata,
        "lr_schedule": {
            "name": "cosine_with_warmup",
            "warmup_ratio": args.lr_warmup_ratio,
            "final_ratio": args.lr_final_ratio,
        },
        "gradient_clip": args.gradient_clip,
        "beta_calibration_steps": args.calibration_steps if args.arm == "treatment" else 0,
        "beta_calibration": calibration,
        "calibrated_betas": calibrated_betas,
        "beta_warmup_ratio": args.beta_warmup_ratio if args.arm == "treatment" else 0.0,
        "auxiliary_sampling": {
            "unit": "eligible_qa_row",
            "ratio_mc_tf_oe": [4, 3, 1],
            "same_component_on_all_ranks": True,
            "global_exposure_per_epoch": global_exposure,
            "valid_pool_counts": pool_counts,
            "cycle_policy": "global_without_replacement_then_seeded_reshuffle",
            "answer_stream": "independent_uniform_series_stream",
        } if args.arm == "treatment" else None,
        "counterfactual_index": str(Path(args.counterfactual_index).resolve()) if args.counterfactual_index else None,
        "counterfactual_index_sha256": sha256_file(args.counterfactual_index) if args.counterfactual_index else None,
        "counterfactual_index_version": COUNTERFACTUAL_INDEX_VERSION if args.counterfactual_index else None,
        "supervision_cache": str(Path(args.supervision_cache).resolve()) if args.supervision_cache else None,
        "supervision_cache_sha256": sha256_file(args.supervision_cache) if args.supervision_cache else None,
        "supervision_cache_version": SUPERVISION_CACHE_VERSION if args.supervision_cache else None,
        "fixed_hint_answer_gradient": True,
        "fixed_hint_auxiliary_gradient": False,
        "fixed_hint_auxiliary_isolation": "processed_fixed_hint_output_stop_gradient",
        "component_backward": "answer_then_one_stratified_auxiliary_same_optimizer_step",
        "trainable_parameters": trainable,
        "train_series": len(answer_dataset),
        "architecture": {
            "variant": "loss_only",
            **architecture_flags,
            "qk_norm_seq_len": None,
            "fixed_query_tokens": 30,
            "task_prompt_tokens": None,
            "fixed_hint_path": "learned_queries_to_shared_prototype_cross_attention",
        },
        "state_verbalizers": state_verbalizers.to_metadata(),
        "tf_verbalizers": tf_verbalizers.to_metadata(),
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "run_config.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps({"event": "run_config", **metadata}, ensure_ascii=False), flush=True)

    planned_steps = args.epochs * args.steps_per_epoch
    global_step = 0
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        answer_loader = _answer_loader(
            answer_dataset,
            world=world,
            rank=rank,
            seed=args.seed,
            epoch=epoch,
            workers=args.num_workers,
        )
        if len(answer_loader) < args.steps_per_epoch:
            raise ValueError("answer stream is shorter than the predeclared epoch")
        answer_iterator = iter(answer_loader)
        schedule = (
            stratified_component_schedule(args.steps_per_epoch, seed=args.seed, epoch=epoch)
            if args.arm == "treatment" else [None] * args.steps_per_epoch
        )
        if args.arm == "treatment":
            loaders = _auxiliary_loaders(
                qa_view,
                schedule,
                world=world,
                rank=rank,
                seed=args.seed,
                epoch=epoch,
                workers=args.num_workers,
            )
            auxiliary_iterators = {name: iter(loader) for name, loader in loaders.items()}
        else:
            auxiliary_iterators = {}
        epoch_values = torch.zeros(20, dtype=torch.float64, device=local)
        epoch_start = time.perf_counter()
        ddp.train()
        ddp.module.ts_pretrain_model.eval()
        ddp.module.axis.model.eval()
        for epoch_step, component in enumerate(schedule, start=1):
            lr_multiplier = cosine_warmup_multiplier(
                global_step,
                planned_steps,
                warmup_ratio=args.lr_warmup_ratio,
                final_ratio=args.lr_final_ratio,
            )
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * lr_multiplier
            betas = {
                name: scheduled_beta(
                    global_step,
                    planned_steps,
                    target_beta=calibrated_betas[name],
                    warmup_ratio=args.beta_warmup_ratio,
                ) for name in COMPONENT_ORDER
            }
            optimizer.zero_grad(set_to_none=True)
            answer_loss, answer_stats = _answer_forward(ddp, next(answer_iterator), local)
            if not bool(torch.isfinite(answer_loss.detach())):
                raise FloatingPointError(f"non-finite answer loss at step {global_step}")
            answer_loss.backward()
            component_stats = {"answer": answer_stats}
            if component is not None:
                auxiliary_loss, auxiliary_stats = _auxiliary_forward(
                    ddp,
                    next(auxiliary_iterators[component]),
                    component,
                    betas[component],
                    local,
                    state_verbalizers,
                    tf_verbalizers,
                )
                if not bool(torch.isfinite(auxiliary_loss.detach())):
                    raise FloatingPointError(f"non-finite {component} loss at step {global_step}")
                auxiliary_loss.backward()
                component_stats[component] = auxiliary_stats
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp.module.axis.perceiver.parameters(), args.gradient_clip
            )
            if not bool(torch.isfinite(grad_norm.detach())):
                raise FloatingPointError(f"non-finite gradient norm at step {global_step}")
            optimizer.step()
            global_step += 1

            mc = component_stats.get("mc", _empty_component_stats(grad_norm))
            tf = component_stats.get("tf", _empty_component_stats(grad_norm))
            oe = component_stats.get("oe", _empty_component_stats(grad_norm))
            epoch_values += torch.stack([
                answer_stats["loss_sum"], answer_stats["count"],
                mc["loss_sum"], mc["count"], mc["pairs"], mc["correct"],
                mc["factual_correct"], mc["counterfactual_correct"],
                tf["loss_sum"], tf["count"], tf["pairs"], tf["correct"],
                tf["factual_correct"], tf["counterfactual_correct"],
                oe["loss_sum"], oe["count"], oe["pairs"],
                oe["delta_sum"], oe["positive_delta"], grad_norm.detach().float(),
            ]).to(device=local, dtype=torch.float64)

            if global_step % 20 == 0:
                log = _reduce_step_stats(component_stats, grad_norm, betas)
                if rank == 0:
                    print(json.dumps({
                        "event": "train_step",
                        "arm": args.arm,
                        "epoch": epoch,
                        "epoch_step": epoch_step,
                        "global_step": global_step,
                        "active_auxiliary": component,
                        "lr_multiplier": lr_multiplier,
                        "local_lr": optimizer.param_groups[0]["lr"],
                        **log,
                    }), flush=True)
            if rank == 0 and global_step % args.save_every == 0:
                save_atomic(
                    checkpoint_payload(ddp.module, optimizer, epoch, global_step, metadata),
                    output / f"step_{global_step}.pth",
                )

        summary = _epoch_summary(epoch_values, args.steps_per_epoch, world)
        elapsed = time.perf_counter() - epoch_start
        if rank == 0:
            print(json.dumps({
                "event": "epoch_summary",
                "arm": args.arm,
                "epoch": epoch,
                "steps": args.steps_per_epoch,
                "elapsed_hours": elapsed / 3600.0,
                "steps_per_second": args.steps_per_epoch / elapsed,
                "calibrated_betas": calibrated_betas,
                **summary,
            }), flush=True)
            payload = checkpoint_payload(ddp.module, optimizer, epoch, global_step, metadata)
            payload["epoch_summary"] = summary
            save_atomic(payload, output / f"epoch_{epoch}.pth")
            save_atomic(payload, output / "final.pth")

    torch.distributed.barrier()
    if rank == 0:
        print(json.dumps({
            "event": "training_complete",
            "arm": args.arm,
            "global_step": global_step,
            "elapsed_hours": (time.perf_counter() - run_start) / 3600.0,
            "checkpoint": str(output / "final.pth"),
        }), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
