"""Phase-II training with coherent counterfactual state supervision."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from . import model_utils
from .architecture_redesign import (
    ARCHITECTURE_VARIANTS,
    add_architecture_arguments,
    local_length_percentile,
    qk_scale_initial_value,
)
from .loss_redesign import (
    COUNTERFACTUAL_INDEX_VERSION,
    CounterfactualAXISDataset,
    counterfactual_collate_fn,
    memory_safe_token_nll_sums,
    scheduled_beta,
    select_state_verbalizers,
)
from .model_utils import checkpoint_payload, freeze_for_phase2, load_phase1_fresh_hint, sha256_file


_PLAIN_BUILD = model_utils.build_model


def _answer_context_nll(
    axis,
    local_embeddings,
    time_series,
    questions,
    answers,
    start_indices,
    end_indices,
):
    ids, mask, labels, _ = axis.generate_input_ids_and_labels(
        questions,
        answers,
        time_series,
        start_indices,
        end_indices,
    )
    device = axis.get_device()
    ids = ids.to(device)
    mask = mask.to(device)
    labels = labels.to(device)
    embeds = axis.get_hint_embeddings(ids, local_embeddings, start_indices, end_indices)
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
        chunk_size=64,
        checkpoint_chunks=True,
    )


def _axis_consistent_state_forward(
    self,
    local_embeddings,
    time_series,
    questions,
    answers,
    start_indices,
    end_indices,
    return_logits=False,
    ablation_mode=None,
    counterfactual_local_embeddings=None,
    counterfactual_time_series=None,
    positive_window_values=None,
    counterfactual_window_values=None,
    state_targets=None,
    counterfactual_valid=None,
    state_question=None,
    normal_id=None,
    anomalous_id=None,
    beta=0.1,
):
    if return_logits:
        raise ValueError("state-supervised training does not materialize full logits")
    if ablation_mode is not None:
        raise ValueError("training-time state supervision is incompatible with inference ablation")
    required = (
        counterfactual_local_embeddings,
        counterfactual_time_series,
        positive_window_values,
        counterfactual_window_values,
        state_targets,
        counterfactual_valid,
        state_question,
        normal_id,
        anomalous_id,
    )
    if any(value is None for value in required):
        raise ValueError("coherent counterfactual state inputs are required")

    answer_sums, answer_counts = _answer_context_nll(
        self,
        local_embeddings,
        time_series,
        questions,
        answers,
        start_indices,
        end_indices,
    )
    answer_nll_sum = answer_sums.sum()
    answer_token_count = answer_counts.sum().clamp_min(1)
    answer_loss = answer_nll_sum / answer_token_count

    valid = counterfactual_valid.to(answer_loss.device).bool()
    valid_indices = valid.nonzero(as_tuple=True)[0]
    valid_pairs = int(valid_indices.numel())
    if valid_pairs:
        selected = valid_indices.tolist()
        state_local = torch.cat(
            [local_embeddings[valid], counterfactual_local_embeddings[valid]],
            dim=0,
        )
        state_time = torch.cat(
            [time_series[valid], counterfactual_time_series[valid]],
            dim=0,
        )
        state_starts = [start_indices[index] for index in selected] * 2
        state_ends = [end_indices[index] for index in selected] * 2
        state_windows = (
            [positive_window_values[index] for index in selected]
            + [counterfactual_window_values[index] for index in selected]
        )
        original_targets = state_targets.to(answer_loss.device).long()[valid]
        paired_targets = torch.cat([original_targets, 1 - original_targets], dim=0)
        logits = self.state_logits(
            state_local,
            state_time,
            state_starts,
            state_ends,
            state_windows,
            state_question,
            int(normal_id),
            int(anomalous_id),
            fixed_hint_frozen=True,
        )
        state_loss_sum = F.cross_entropy(logits.float(), paired_targets, reduction="sum")
        state_rows = paired_targets.numel()
        state_loss = state_loss_sum / state_rows
        predictions = logits.argmax(dim=-1)
        original_correct = predictions[:valid_pairs].eq(original_targets).sum()
        counterfactual_correct = predictions[valid_pairs:].eq(1 - original_targets).sum()
    else:
        state_loss = answer_loss.new_zeros(())
        state_loss_sum = answer_loss.new_zeros(())
        state_rows = 0
        original_correct = torch.zeros((), device=answer_loss.device, dtype=torch.long)
        counterfactual_correct = torch.zeros((), device=answer_loss.device, dtype=torch.long)

    total = answer_loss + float(beta) * state_loss
    self._last_loss_stats = {
        "answer_nll_sum": answer_nll_sum.detach(),
        "answer_tokens": answer_token_count.detach(),
        "state_loss_sum": state_loss_sum.detach(),
        "state_rows": torch.tensor(state_rows, device=answer_loss.device),
        "valid_pairs": torch.tensor(valid_pairs, device=answer_loss.device),
        "original_correct": original_correct.detach(),
        "counterfactual_correct": counterfactual_correct.detach(),
        "beta": torch.tensor(float(beta), device=answer_loss.device),
        "total_loss": total.detach(),
    }
    return total


def build_model(
    architecture_variant: str = "loss_only",
    qk_norm_seq_len: int | None = None,
    gate_bias: float = -2.0,
):
    model = _PLAIN_BUILD(
        architecture_variant=architecture_variant,
        qk_norm_seq_len=qk_norm_seq_len,
        gate_bias=gate_bias,
    )
    llm = model.axis.model
    llm.gradient_checkpointing_enable()
    llm.enable_input_require_grads()
    llm.config.use_cache = False
    llm.get_input_embeddings().register_forward_hook(lambda _module, _inputs, output: output.clone())
    model.axis.forward = types.MethodType(_axis_consistent_state_forward, model.axis)
    llm.eval = types.MethodType(lambda self: self, llm)
    return model


def phase2_parameter_groups(
    perceiver,
    *,
    local_lr: float = 5e-5,
    attention_lr: float = 1e-4,
    prompt_lr: float = 5e-5,
    weight_decay: float = 1e-5,
) -> tuple[list[dict], dict[str, dict[str, float | int]]]:
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "local_continuous": [],
        "prototype_attention": [],
        "task_prompt": [],
    }
    for name, parameter in perceiver.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("mapping_layer.", "local_attention.")):
            grouped["prototype_attention"].append(parameter)
        elif "prompt_embeddings" in name:
            grouped["task_prompt"].append(parameter)
        else:
            grouped["local_continuous"].append(parameter)
    learning_rates = {
        "local_continuous": local_lr,
        "prototype_attention": attention_lr,
        "task_prompt": prompt_lr,
    }
    optimizer_groups = []
    metadata = {}
    for name, parameters in grouped.items():
        if not parameters:
            continue
        optimizer_groups.append({
            "params": parameters,
            "lr": learning_rates[name],
            "weight_decay": weight_decay,
            "group_name": name,
        })
        metadata[name] = {
            "lr": float(learning_rates[name]),
            "weight_decay": float(weight_decay),
            "parameters": int(sum(parameter.numel() for parameter in parameters)),
        }
    expected = sum(parameter.numel() for parameter in perceiver.parameters() if parameter.requires_grad)
    actual = sum(item["parameters"] for item in metadata.values())
    if expected != actual:
        raise RuntimeError(f"optimizer parameter grouping mismatch: {actual} != {expected}")
    return optimizer_groups, metadata


def _aggregate_log_stats(stats: dict[str, torch.Tensor], grad_norm: torch.Tensor) -> dict[str, float | int]:
    keys = (
        "answer_nll_sum",
        "answer_tokens",
        "state_loss_sum",
        "state_rows",
        "valid_pairs",
        "original_correct",
        "counterfactual_correct",
        "beta",
        "total_loss",
    )
    values = torch.stack([stats[key].float() for key in keys] + [grad_norm.detach().float()])
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    world = torch.distributed.get_world_size()
    reduced = dict(zip(keys + ("grad_norm",), values.tolist()))
    answer_tokens = max(1.0, reduced["answer_tokens"])
    state_rows = max(1.0, reduced["state_rows"])
    valid_pairs = max(1.0, reduced["valid_pairs"])
    return {
        "answer_loss": reduced["answer_nll_sum"] / answer_tokens,
        "state_loss": reduced["state_loss_sum"] / state_rows if reduced["state_rows"] else 0.0,
        "valid_pairs": int(reduced["valid_pairs"]),
        "state_rows": int(reduced["state_rows"]),
        "state_accuracy": (
            reduced["original_correct"] + reduced["counterfactual_correct"]
        ) / state_rows if reduced["state_rows"] else 0.0,
        "original_state_accuracy": reduced["original_correct"] / valid_pairs if reduced["valid_pairs"] else 0.0,
        "counterfactual_state_accuracy": reduced["counterfactual_correct"] / valid_pairs if reduced["valid_pairs"] else 0.0,
        "beta": reduced["beta"] / world,
        "loss": reduced["total_loss"] / world,
        "grad_norm": reduced["grad_norm"] / world,
    }


def save_atomic(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_parser(default_architecture_variant: str = "loss_only") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--counterfactual-index", required=True)
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--output", default="experiments/reproduction/phase2_consistent_state")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--local-lr", type=float, default=5e-5)
    parser.add_argument("--attention-lr", type=float, default=1e-4)
    parser.add_argument("--prompt-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--beta-warmup-ratio", type=float, default=0.1)
    add_architecture_arguments(parser, default_variant=default_architecture_variant)
    return parser


def main(default_architecture_variant: str = "loss_only") -> None:
    args = build_parser(default_architecture_variant).parse_args()
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

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = CounterfactualAXISDataset(
        args.data,
        args.counterfactual_index,
        split="train",
        train_ratio=0.95,
        seed=args.seed,
    )
    phase1_sha = sha256_file(args.phase1)
    if dataset.index_metadata.get("phase1_sha256") != phase1_sha:
        raise ValueError("counterfactual index was built with a different Phase-I checkpoint")

    architecture_flags = ARCHITECTURE_VARIANTS[args.architecture_variant]
    qk_norm_seq_len = args.qk_norm_seq_len
    qk_length_source = None
    if architecture_flags["qk_norm"]:
        if qk_norm_seq_len is None and rank == 0:
            qk_norm_seq_len = local_length_percentile(dataset.series_files, 97.5)
        qk_length_tensor = torch.tensor([qk_norm_seq_len or 0], dtype=torch.int64, device=local)
        torch.distributed.broadcast(qk_length_tensor, src=0)
        qk_norm_seq_len = int(qk_length_tensor.item())
        qk_length_source = "cli_override" if args.qk_norm_seq_len is not None else "seeded_train_local_p97.5"
    else:
        qk_norm_seq_len = None
    architecture_meta = {
        "variant": args.architecture_variant,
        **architecture_flags,
        "qk_norm_seq_len": qk_norm_seq_len,
        "qk_scale_initial": qk_scale_initial_value(qk_norm_seq_len) if qk_norm_seq_len is not None else None,
        "qk_length_source": qk_length_source,
        "qk_length_percentile": 97.5 if qk_length_source == "seeded_train_local_p97.5" else None,
        "gate_bias": args.gate_bias,
        "task_prompt_tokens": 30,
    }
    fixed_hint_state_isolation = (
        "direct_task_prompt_output_stop_gradient"
        if architecture_flags["direct_task_prompt"]
        else "processed_fixed_output_stop_gradient_shared_processor"
    )
    model = build_model(
        architecture_variant=args.architecture_variant,
        qk_norm_seq_len=qk_norm_seq_len,
        gate_bias=args.gate_bias,
    )
    load_phase1_fresh_hint(model, args.phase1)
    trainable = freeze_for_phase2(model)
    verbalizers = select_state_verbalizers(model.axis.tokenizer)
    model.to(local)
    ddp = DDP(model, device_ids=[local], output_device=local, broadcast_buffers=False)
    optimizer_groups, optimizer_metadata = phase2_parameter_groups(
        ddp.module.axis.perceiver,
        local_lr=args.local_lr,
        attention_lr=args.attention_lr,
        prompt_lr=args.prompt_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(optimizer_groups)
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
        "objective": "answer_nll_plus_coherent_binary_state_ce_v3",
        "objective_version": 3,
        "phase1": str(Path(args.phase1).resolve()),
        "phase1_sha256": phase1_sha,
        "counterfactual_index": str(Path(args.counterfactual_index).resolve()),
        "counterfactual_index_sha256": sha256_file(args.counterfactual_index),
        "counterfactual_index_version": COUNTERFACTUAL_INDEX_VERSION,
        "counterfactual_policy": dataset.index_metadata.get("policy"),
        "counterfactual_stats": dataset.index_metadata.get("stats"),
        "world_size": world,
        "epochs": args.epochs,
        "seed": args.seed,
        "beta_target": args.beta,
        "beta_warmup_ratio": args.beta_warmup_ratio,
        "gradient_clip": args.gradient_clip,
        "optimizer_groups": optimizer_metadata,
        "state_verbalizers": verbalizers.to_metadata(),
        "fixed_hint_state_gradient": False,
        "fixed_hint_answer_gradient": True,
        "fixed_hint_freeze_scope": "state_loss_only",
        "fixed_hint_state_isolation": fixed_hint_state_isolation,
        "state_prompt_independent": True,
        "state_pairing": "factual_counterfactual_same_step",
        "state_logits_source": "frozen_lm_head_next_token_two_class",
        "state_label_policy": "strict_single_token_numeric_0_1",
        "state_verbalizer_selection": "prefer_space_prefixed_then_bare_numeric",
        "soft_embedding_injection": "non_inplace_index_copy",
        "trainable_parameters": trainable,
        "train_series": len(dataset),
        "steps_per_rank_epoch": len(loader),
        "architecture": architecture_meta,
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
            effective_beta = scheduled_beta(
                global_step,
                planned,
                target_beta=args.beta,
                warmup_ratio=args.beta_warmup_ratio,
            )
            positive = batch["padded_sequences"].to(local, dtype=torch.float32, non_blocking=True)
            positive_mask = batch["attention_masks"].to(local, non_blocking=True)
            counterfactual = batch["counterfactual_sequences"].to(local, dtype=torch.float32, non_blocking=True)
            counterfactual_mask = batch["counterfactual_masks"].to(local, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = ddp(
                    positive,
                    positive_mask,
                    batch["questions"],
                    batch["answers"],
                    batch["start_indices"],
                    batch["end_indices"],
                    counterfactual_sequences=counterfactual,
                    counterfactual_masks=counterfactual_mask,
                    positive_window_values=batch["positive_window_values"],
                    counterfactual_window_values=batch["counterfactual_window_values"],
                    state_targets=batch["state_targets"].to(local, non_blocking=True),
                    counterfactual_valid=batch["counterfactual_valid"].to(local, non_blocking=True),
                    state_question=verbalizers.question,
                    normal_id=verbalizers.normal_id,
                    anomalous_id=verbalizers.anomalous_id,
                    beta=effective_beta,
                )
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"non-finite training loss at step {global_step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp.module.axis.perceiver.parameters(),
                max_norm=args.gradient_clip,
            )
            if not bool(torch.isfinite(grad_norm.detach())):
                raise FloatingPointError(f"non-finite gradient norm at step {global_step}")
            optimizer.step()
            global_step += 1
            if global_step == 10:
                timed_start = time.perf_counter()
            if global_step % 20 == 0:
                log = _aggregate_log_stats(ddp.module.axis._last_loss_stats, grad_norm)
                if rank == 0:
                    elapsed = time.perf_counter() - (timed_start or run_start)
                    measured = max(1, global_step - (10 if timed_start else 0)) / elapsed
                    log.update({
                        "step": global_step,
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
