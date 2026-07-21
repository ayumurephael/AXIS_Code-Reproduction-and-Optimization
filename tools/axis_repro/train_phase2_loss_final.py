"""Two-stage Phase-II trainer for the final question-type-routed objective.

Stage ``answer_only`` performs only factual full-answer training. Stage ``joint``
keeps that loss and adds MC state CE, TF question-conditioned CE, and OE QCAR.
The four components are backpropagated sequentially inside one optimizer step to
avoid retaining several 7B-LLM graphs at once.
"""
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

from src.models.AXIS.dataset import AXISAnomalyQADataset
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
    answer_collate_fn,
    counterfactual_collate_fn,
    memory_safe_token_nll_sums,
    scheduled_beta,
    select_state_verbalizers,
)
from .model_utils import (
    checkpoint_payload,
    freeze_for_phase2,
    load_axis_checkpoint,
    load_phase1_fresh_hint,
    sha256_file,
)
from .question_type_objectives import (
    SUPERVISION_CACHE_VERSION,
    ddp_global_mean,
    select_tf_verbalizers,
)
from .train_phase2_loss_redesign import phase2_parameter_groups, save_atomic


_PLAIN_BUILD = model_utils.build_model
COMPONENTS = ("answer", "mc", "tf", "oe")


def _answer_context_nll(
    axis,
    local_embeddings,
    time_series,
    questions,
    answers,
    start_indices,
    end_indices,
):
    input_ids, attention_mask, labels, _ = axis.generate_input_ids_and_labels(
        questions,
        answers,
        time_series,
        start_indices,
        end_indices,
    )
    device = axis.get_device()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    embeddings = axis.get_hint_embeddings(
        input_ids,
        local_embeddings,
        start_indices,
        end_indices,
        detach_fixed_hint=False,
    )
    hidden = axis.model.model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
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


def _trainable_zero(axis) -> torch.Tensor:
    terms = [parameter.reshape(-1)[0] for parameter in axis.perceiver.parameters() if parameter.requires_grad]
    if not terms:
        raise RuntimeError("Phase-II has no trainable Perceiver parameters")
    return torch.stack(terms).sum() * 0.0


def _empty_component_stats(reference: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = reference.detach().new_zeros((), dtype=torch.float32)
    return {
        "loss_sum": zero,
        "count": zero,
        "pairs": zero,
        "correct": zero,
        "factual_correct": zero,
        "counterfactual_correct": zero,
        "delta_sum": zero,
        "positive_delta": zero,
    }


def _paired_rows(
    local_embeddings,
    counterfactual_local_embeddings,
    time_series,
    counterfactual_time_series,
    start_indices,
    end_indices,
    positive_window_values,
    counterfactual_window_values,
    valid,
):
    indices = valid.nonzero(as_tuple=True)[0]
    selected = indices.tolist()
    paired_local = torch.cat(
        [local_embeddings[valid], counterfactual_local_embeddings[valid]], dim=0
    )
    paired_time = torch.cat(
        [time_series[valid], counterfactual_time_series[valid]], dim=0
    )
    paired_starts = [start_indices[index] for index in selected] * 2
    paired_ends = [end_indices[index] for index in selected] * 2
    paired_windows = (
        [positive_window_values[index] for index in selected]
        + [counterfactual_window_values[index] for index in selected]
    )
    return selected, paired_local, paired_time, paired_starts, paired_ends, paired_windows


def _axis_component_forward(
    self,
    local_embeddings,
    time_series,
    questions,
    answers,
    start_indices,
    end_indices,
    *,
    loss_component,
    question_types=None,
    counterfactual_local_embeddings=None,
    counterfactual_time_series=None,
    positive_window_values=None,
    counterfactual_window_values=None,
    state_targets=None,
    mc_pair_valid=None,
    tf_pair_valid=None,
    tf_targets=None,
    oe_pair_valid=None,
    state_question=None,
    normal_id=None,
    anomalous_id=None,
    tf_false_id=None,
    tf_true_id=None,
    beta_mc=0.0,
    beta_tf=0.0,
    beta_oe=0.0,
):
    if loss_component not in COMPONENTS:
        raise ValueError(f"unknown loss component: {loss_component}")
    graph_zero = _trainable_zero(self)
    if loss_component == "answer":
        row_sums, row_counts = _answer_context_nll(
            self,
            local_embeddings,
            time_series,
            questions,
            answers,
            start_indices,
            end_indices,
        )
        local_sum = row_sums.sum()
        local_count = row_counts.sum()
        if int(local_count.item()) <= 0:
            raise RuntimeError("answer component has no target tokens")
        loss = ddp_global_mean(local_sum, local_count) + graph_zero
        stats = _empty_component_stats(local_sum)
        stats.update({"loss_sum": local_sum.detach(), "count": local_count.detach()})
        self._last_component_stats = stats
        return loss

    required = (
        question_types,
        counterfactual_local_embeddings,
        counterfactual_time_series,
        positive_window_values,
        counterfactual_window_values,
        state_targets,
        mc_pair_valid,
        tf_pair_valid,
        tf_targets,
        oe_pair_valid,
    )
    if any(value is None for value in required):
        raise ValueError("joint objective inputs are incomplete")
    device = local_embeddings.device
    if len(question_types) != local_embeddings.shape[0]:
        raise ValueError("question type count does not match flattened QA rows")

    if loss_component == "mc":
        valid = mc_pair_valid.to(device).bool()
        if any(valid[index] and question_types[index] != "multiple_choice" for index in range(len(question_types))):
            raise ValueError("MC valid set contains another question type")
        pair_count = int(valid.sum().item())
        stats = _empty_component_stats(graph_zero)
        if pair_count:
            selected, paired_local, paired_time, paired_starts, paired_ends, paired_windows = _paired_rows(
                local_embeddings,
                counterfactual_local_embeddings,
                time_series,
                counterfactual_time_series,
                start_indices,
                end_indices,
                positive_window_values,
                counterfactual_window_values,
                valid,
            )
            factual_targets = state_targets.to(device).long()[valid]
            paired_targets = torch.cat([factual_targets, 1 - factual_targets])
            logits = self.state_logits(
                paired_local,
                paired_time,
                paired_starts,
                paired_ends,
                paired_windows,
                state_question,
                int(normal_id),
                int(anomalous_id),
                fixed_hint_frozen=True,
            )
            local_sum = F.cross_entropy(logits.float(), paired_targets, reduction="sum")
            local_count = paired_targets.numel()
            predictions = logits.argmax(dim=-1)
            stats.update({
                "loss_sum": local_sum.detach(),
                "count": torch.tensor(float(local_count), device=device),
                "pairs": torch.tensor(float(pair_count), device=device),
                "correct": predictions.eq(paired_targets).sum().detach().float(),
                "factual_correct": predictions[:pair_count].eq(factual_targets).sum().detach().float(),
                "counterfactual_correct": predictions[pair_count:].eq(1 - factual_targets).sum().detach().float(),
            })
        else:
            local_sum = graph_zero
            local_count = 0
        raw_loss = ddp_global_mean(local_sum, local_count)
        loss = float(beta_mc) * raw_loss + graph_zero

    elif loss_component == "tf":
        valid = tf_pair_valid.to(device).bool()
        if any(valid[index] and question_types[index] != "true_false" for index in range(len(question_types))):
            raise ValueError("TF valid set contains another question type")
        pair_count = int(valid.sum().item())
        stats = _empty_component_stats(graph_zero)
        if pair_count:
            selected, paired_local, paired_time, paired_starts, paired_ends, paired_windows = _paired_rows(
                local_embeddings,
                counterfactual_local_embeddings,
                time_series,
                counterfactual_time_series,
                start_indices,
                end_indices,
                positive_window_values,
                counterfactual_window_values,
                valid,
            )
            factual_targets = tf_targets.to(device).long()[valid]
            if bool(((factual_targets < 0) | (factual_targets > 1)).any()):
                raise ValueError("TF valid row has an invalid target")
            paired_targets = torch.cat([factual_targets, 1 - factual_targets])
            selected_questions = [questions[index] for index in selected]
            logits = self.tf_logits(
                paired_local,
                paired_time,
                selected_questions + selected_questions,
                paired_starts,
                paired_ends,
                paired_windows,
                int(tf_false_id),
                int(tf_true_id),
                fixed_hint_frozen=True,
            )
            local_sum = F.cross_entropy(logits.float(), paired_targets, reduction="sum")
            local_count = paired_targets.numel()
            predictions = logits.argmax(dim=-1)
            stats.update({
                "loss_sum": local_sum.detach(),
                "count": torch.tensor(float(local_count), device=device),
                "pairs": torch.tensor(float(pair_count), device=device),
                "correct": predictions.eq(paired_targets).sum().detach().float(),
                "factual_correct": predictions[:pair_count].eq(factual_targets).sum().detach().float(),
                "counterfactual_correct": predictions[pair_count:].eq(1 - factual_targets).sum().detach().float(),
            })
        else:
            local_sum = graph_zero
            local_count = 0
        raw_loss = ddp_global_mean(local_sum, local_count)
        loss = float(beta_tf) * raw_loss + graph_zero

    else:
        valid = oe_pair_valid.to(device).bool()
        if any(valid[index] and question_types[index] != "open_ended" for index in range(len(question_types))):
            raise ValueError("OE valid set contains another question type")
        pair_count = int(valid.sum().item())
        stats = _empty_component_stats(graph_zero)
        if pair_count:
            selected, paired_local, paired_time, paired_starts, paired_ends, paired_windows = _paired_rows(
                local_embeddings,
                counterfactual_local_embeddings,
                time_series,
                counterfactual_time_series,
                start_indices,
                end_indices,
                positive_window_values,
                counterfactual_window_values,
                valid,
            )
            selected_questions = [questions[index] for index in selected]
            selected_answers = [answers[index] for index in selected]
            scores, token_counts = self.oe_answer_scores(
                paired_local,
                paired_time,
                selected_questions + selected_questions,
                selected_answers + selected_answers,
                paired_starts,
                paired_ends,
                paired_windows,
            )
            if not torch.equal(token_counts[:pair_count], token_counts[pair_count:]):
                raise RuntimeError("OE factual/counterfactual answer token counts differ")
            deltas = scores[:pair_count] - scores[pair_count:]
            rows = F.softplus(-deltas)
            local_sum = rows.sum()
            local_count = pair_count
            stats.update({
                "loss_sum": local_sum.detach(),
                "count": torch.tensor(float(local_count), device=device),
                "pairs": torch.tensor(float(pair_count), device=device),
                "delta_sum": deltas.sum().detach().float(),
                "positive_delta": deltas.gt(0).sum().detach().float(),
            })
        else:
            local_sum = graph_zero
            local_count = 0
        raw_loss = ddp_global_mean(local_sum, local_count)
        loss = float(beta_oe) * raw_loss + graph_zero

    self._last_component_stats = stats
    return loss


def _combined_component_forward(
    self,
    padded_sequences,
    attention_masks,
    questions,
    answers,
    start_indices,
    end_indices,
    *,
    precomputed_local_embeddings=None,
    precomputed_counterfactual_embeddings=None,
    counterfactual_sequences=None,
    **axis_kwargs,
):
    if precomputed_local_embeddings is None:
        with torch.no_grad():
            precomputed_local_embeddings = self.ts_pretrain_model(
                padded_sequences, mask=attention_masks
            )
    return self.axis(
        local_embeddings=precomputed_local_embeddings,
        time_series=padded_sequences,
        questions=questions,
        answers=answers,
        start_indices=start_indices,
        end_indices=end_indices,
        counterfactual_local_embeddings=precomputed_counterfactual_embeddings,
        counterfactual_time_series=counterfactual_sequences,
        **axis_kwargs,
    )


def build_model(
    architecture_variant: str = "full",
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
    llm.get_input_embeddings().register_forward_hook(
        lambda _module, _inputs, output: output.clone()
    )
    model.axis.forward = types.MethodType(_axis_component_forward, model.axis)
    model.forward = types.MethodType(_combined_component_forward, model)
    llm.eval = types.MethodType(lambda self: self, llm)
    return model


def _reduce_step_stats(component_stats, grad_norm, betas):
    device = grad_norm.device
    names = (
        "answer_loss_sum", "answer_count",
        "mc_loss_sum", "mc_count", "mc_pairs", "mc_correct",
        "mc_factual_correct", "mc_counterfactual_correct",
        "tf_loss_sum", "tf_count", "tf_pairs", "tf_correct",
        "tf_factual_correct", "tf_counterfactual_correct",
        "oe_loss_sum", "oe_count", "oe_pairs", "oe_delta_sum", "oe_positive_delta",
        "grad_norm",
    )
    answer = component_stats["answer"]
    mc = component_stats.get("mc", _empty_component_stats(grad_norm))
    tf = component_stats.get("tf", _empty_component_stats(grad_norm))
    oe = component_stats.get("oe", _empty_component_stats(grad_norm))
    values = torch.stack([
        answer["loss_sum"], answer["count"],
        mc["loss_sum"], mc["count"], mc["pairs"], mc["correct"],
        mc["factual_correct"], mc["counterfactual_correct"],
        tf["loss_sum"], tf["count"], tf["pairs"], tf["correct"],
        tf["factual_correct"], tf["counterfactual_correct"],
        oe["loss_sum"], oe["count"], oe["pairs"], oe["delta_sum"], oe["positive_delta"],
        grad_norm.detach().float(),
    ]).to(device=device, dtype=torch.float32)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    reduced = dict(zip(names, values.tolist()))
    world = torch.distributed.get_world_size()

    def mean(sum_name, count_name):
        return reduced[sum_name] / reduced[count_name] if reduced[count_name] else 0.0

    answer_loss = mean("answer_loss_sum", "answer_count")
    mc_loss = mean("mc_loss_sum", "mc_count")
    tf_loss = mean("tf_loss_sum", "tf_count")
    oe_loss = mean("oe_loss_sum", "oe_count")
    return {
        "answer_loss": answer_loss,
        "mc_state_loss": mc_loss,
        "tf_q_loss": tf_loss,
        "oe_qcar_loss": oe_loss,
        "loss": answer_loss + betas["mc"] * mc_loss + betas["tf"] * tf_loss + betas["oe"] * oe_loss,
        "mc_pairs": int(reduced["mc_pairs"]),
        "tf_pairs": int(reduced["tf_pairs"]),
        "oe_pairs": int(reduced["oe_pairs"]),
        "mc_accuracy": reduced["mc_correct"] / reduced["mc_count"] if reduced["mc_count"] else 0.0,
        "tf_accuracy": reduced["tf_correct"] / reduced["tf_count"] if reduced["tf_count"] else 0.0,
        "oe_mean_delta": reduced["oe_delta_sum"] / reduced["oe_pairs"] if reduced["oe_pairs"] else 0.0,
        "oe_positive_delta_rate": reduced["oe_positive_delta"] / reduced["oe_pairs"] if reduced["oe_pairs"] else 0.0,
        "beta_mc": betas["mc"],
        "beta_tf": betas["tf"],
        "beta_oe": betas["oe"],
        "grad_norm": reduced["grad_norm"] / world,
    }


def build_parser(default_architecture_variant: str = "full") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("answer_only", "joint"), required=True)
    parser.add_argument("--phase1", required=True)
    parser.add_argument("--init-phase2")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--counterfactual-index")
    parser.add_argument("--supervision-cache")
    parser.add_argument("--expected-donor-top-m", type=int, default=1)
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--output", required=True)
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
    parser.add_argument("--beta-mc", type=float, default=0.10)
    parser.add_argument("--beta-tf", type=float, default=0.05)
    parser.add_argument("--beta-oe", type=float, default=0.05)
    parser.add_argument("--beta-warmup-ratio", type=float, default=0.10)
    add_architecture_arguments(parser, default_variant=default_architecture_variant)
    return parser


def main(default_architecture_variant: str = "full") -> None:
    args = build_parser(default_architecture_variant).parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.resume_optimizer and not args.init_phase2:
        raise ValueError("--resume-optimizer requires --init-phase2")
    if args.stage == "joint" and not (args.counterfactual_index and args.supervision_cache):
        raise ValueError("joint stage requires counterfactual index and supervision cache")
    if args.stage == "answer_only" and (args.counterfactual_index or args.supervision_cache):
        raise ValueError("answer-only stage must not materialize counterfactual inputs")
    if args.stage == "answer_only" and args.init_phase2:
        raise ValueError("formal answer-only recovery must start from Phase-I")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    torch.distributed.init_process_group("nccl")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "answer_only":
        dataset = AXISAnomalyQADataset(
            args.data, split="train", train_ratio=0.95, seed=args.seed
        )
        collate = answer_collate_fn
    else:
        dataset = CounterfactualAXISDataset(
            args.data,
            args.counterfactual_index,
            split="train",
            train_ratio=0.95,
            seed=args.seed,
            supervision_cache=args.supervision_cache,
        )
        actual_top_m = int(dataset.index_metadata.get("policy", {}).get("donor_top_m", -1))
        if actual_top_m != args.expected_donor_top_m:
            raise ValueError(f"formal donor Top-M {actual_top_m} != {args.expected_donor_top_m}")
        collate = counterfactual_collate_fn

    phase1_sha = sha256_file(args.phase1)
    if args.stage == "joint" and dataset.index_metadata.get("phase1_sha256") != phase1_sha:
        raise ValueError("counterfactual index Phase-I SHA mismatch")
    architecture_flags = ARCHITECTURE_VARIANTS[args.architecture_variant]
    qk_norm_seq_len = args.qk_norm_seq_len
    qk_length_source = None
    if architecture_flags["qk_norm"]:
        if qk_norm_seq_len is None and rank == 0:
            qk_norm_seq_len = local_length_percentile(dataset.series_files, 97.5)
        length_tensor = torch.tensor([qk_norm_seq_len or 0], dtype=torch.int64, device=local)
        torch.distributed.broadcast(length_tensor, src=0)
        qk_norm_seq_len = int(length_tensor.item())
        qk_length_source = "cli_override" if args.qk_norm_seq_len is not None else "seeded_train_local_p97.5"

    model = build_model(args.architecture_variant, qk_norm_seq_len, args.gate_bias)
    if args.init_phase2:
        initialization_payload = load_axis_checkpoint(model, args.init_phase2, strict=True)
        parent_sha = sha256_file(args.init_phase2)
        parent_meta = initialization_payload.get("reproduction_meta", {})
        if parent_meta.get("stage") != "answer_only":
            raise ValueError("joint stage must initialize from an answer-only checkpoint")
        if parent_meta.get("phase1_sha256") != phase1_sha:
            raise ValueError("answer-only checkpoint Phase-I SHA mismatch")
    else:
        initialization_payload = load_phase1_fresh_hint(model, args.phase1)
        parent_sha = None
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
    optimizer = torch.optim.AdamW(optimizer_groups)
    if args.resume_optimizer:
        if "optimizer_state_dict" not in initialization_payload:
            raise ValueError("initial Phase-II checkpoint has no optimizer state")
        optimizer.load_state_dict(initialization_payload["optimizer_state_dict"])

    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate,
    )
    planned = args.epochs * len(loader)
    global_step = 0
    run_start = time.perf_counter()
    timed_start = None
    architecture_meta = {
        "variant": args.architecture_variant,
        **architecture_flags,
        "qk_norm_seq_len": qk_norm_seq_len,
        "qk_scale_initial": qk_scale_initial_value(qk_norm_seq_len) if qk_norm_seq_len else None,
        "qk_length_source": qk_length_source,
        "qk_length_percentile": 97.5 if qk_length_source == "seeded_train_local_p97.5" else None,
        "gate_bias": args.gate_bias,
        "task_prompt_tokens": 30,
    }
    metadata = {
        "objective": "answer_only_recovery_v1" if args.stage == "answer_only" else "question_type_routed_joint_v1",
        "objective_version": 4,
        "stage": args.stage,
        "phase1": str(Path(args.phase1).resolve()),
        "phase1_sha256": phase1_sha,
        "init_phase2": str(Path(args.init_phase2).resolve()) if args.init_phase2 else None,
        "init_phase2_sha256": parent_sha,
        "resume_optimizer": bool(args.resume_optimizer),
        "stage_transition_seed_reset": bool(args.init_phase2),
        "auxiliary_warmup_restarted": args.stage == "joint",
        "counterfactual_index": str(Path(args.counterfactual_index).resolve()) if args.counterfactual_index else None,
        "counterfactual_index_sha256": sha256_file(args.counterfactual_index) if args.counterfactual_index else None,
        "counterfactual_index_version": COUNTERFACTUAL_INDEX_VERSION if args.counterfactual_index else None,
        "counterfactual_policy": dataset.index_metadata.get("policy") if args.stage == "joint" else None,
        "supervision_cache": str(Path(args.supervision_cache).resolve()) if args.supervision_cache else None,
        "supervision_cache_sha256": sha256_file(args.supervision_cache) if args.supervision_cache else None,
        "supervision_cache_version": SUPERVISION_CACHE_VERSION if args.supervision_cache else None,
        "supervision_policy": dataset.supervision_metadata.get("policy") if args.stage == "joint" else None,
        "supervision_stats": dataset.supervision_metadata.get("stats") if args.stage == "joint" else None,
        "world_size": world,
        "epochs": args.epochs,
        "seed": args.seed,
        "beta_mc_target": args.beta_mc if args.stage == "joint" else 0.0,
        "beta_tf_target": args.beta_tf if args.stage == "joint" else 0.0,
        "beta_oe_target": args.beta_oe if args.stage == "joint" else 0.0,
        "beta_warmup_ratio": args.beta_warmup_ratio if args.stage == "joint" else 0.0,
        "gradient_clip": args.gradient_clip,
        "optimizer_groups": optimizer_metadata,
        "state_verbalizers": state_verbalizers.to_metadata(),
        "tf_verbalizers": tf_verbalizers.to_metadata(),
        "question_type_routing": {
            "multiple_choice": "answer+state_ce",
            "true_false": "answer+tf_q_ce",
            "open_ended": "answer+oe_qcar",
        },
        "fixed_hint_auxiliary_gradient": False,
        "fixed_hint_answer_gradient": True,
        "fixed_hint_auxiliary_isolation": "direct_task_prompt_output_stop_gradient",
        "ddp_auxiliary_denominator": "global_valid_rows_per_component",
        "oe_qcar_target_tokens": "answer_content_only_no_prefix_eos_padding",
        "oe_qcar_counterfactual_policy": "abnormal_self_normal_patch_only",
        "component_backward": "sequential_same_optimizer_step",
        "trainable_parameters": trainable,
        "train_series": len(dataset),
        "steps_per_rank_epoch": len(loader),
        "architecture": architecture_meta,
    }
    if rank == 0:
        print(json.dumps(metadata), flush=True)

    stop = False
    for epoch in range(1, args.epochs + 1):
        epoch_totals = torch.zeros(20, dtype=torch.float64, device=local)
        epoch_steps = 0
        epoch_start = time.perf_counter()
        sampler.set_epoch(epoch)
        ddp.train()
        ddp.module.ts_pretrain_model.eval()
        ddp.module.axis.model.eval()
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            positive = batch["padded_sequences"].to(local, dtype=torch.float32, non_blocking=True)
            positive_mask = batch["attention_masks"].to(local, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                local_embeddings = ddp.module.ts_pretrain_model(positive, mask=positive_mask)
            counterfactual = None
            counterfactual_embeddings = None
            if args.stage == "joint":
                counterfactual = batch["counterfactual_sequences"].to(
                    local, dtype=torch.float32, non_blocking=True
                )
                counterfactual_mask = batch["counterfactual_masks"].to(local, non_blocking=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    counterfactual_embeddings = ddp.module.ts_pretrain_model(
                        counterfactual, mask=counterfactual_mask
                    )
                betas = {
                    "mc": scheduled_beta(global_step, planned, target_beta=args.beta_mc, warmup_ratio=args.beta_warmup_ratio),
                    "tf": scheduled_beta(global_step, planned, target_beta=args.beta_tf, warmup_ratio=args.beta_warmup_ratio),
                    "oe": scheduled_beta(global_step, planned, target_beta=args.beta_oe, warmup_ratio=args.beta_warmup_ratio),
                }
            else:
                betas = {"mc": 0.0, "tf": 0.0, "oe": 0.0}

            common_kwargs = {
                "precomputed_local_embeddings": local_embeddings,
                "precomputed_counterfactual_embeddings": counterfactual_embeddings,
                "counterfactual_sequences": counterfactual,
                "question_types": batch["question_types"],
                "beta_mc": betas["mc"],
                "beta_tf": betas["tf"],
                "beta_oe": betas["oe"],
            }
            if args.stage == "joint":
                common_kwargs.update({
                    "positive_window_values": batch["positive_window_values"],
                    "counterfactual_window_values": batch["counterfactual_window_values"],
                    "state_targets": batch["state_targets"].to(local, non_blocking=True),
                    "mc_pair_valid": batch["mc_pair_valid"].to(local, non_blocking=True),
                    "tf_pair_valid": batch["tf_pair_valid"].to(local, non_blocking=True),
                    "tf_targets": batch["tf_targets"].to(local, non_blocking=True),
                    "oe_pair_valid": batch["oe_pair_valid"].to(local, non_blocking=True),
                    "state_question": state_verbalizers.question,
                    "normal_id": state_verbalizers.normal_id,
                    "anomalous_id": state_verbalizers.anomalous_id,
                    "tf_false_id": tf_verbalizers.false_id,
                    "tf_true_id": tf_verbalizers.true_id,
                })

            component_stats = {}
            active_components = COMPONENTS if args.stage == "joint" else ("answer",)
            for component in active_components:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    component_loss = ddp(
                        positive,
                        positive_mask,
                        batch["questions"],
                        batch["answers"],
                        batch["start_indices"],
                        batch["end_indices"],
                        loss_component=component,
                        **common_kwargs,
                    )
                if not bool(torch.isfinite(component_loss.detach())):
                    raise FloatingPointError(f"non-finite {component} loss at step {global_step}")
                component_loss.backward()
                component_stats[component] = {
                    key: value.detach().clone()
                    for key, value in ddp.module.axis._last_component_stats.items()
                }
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp.module.axis.perceiver.parameters(), args.gradient_clip
            )
            if not bool(torch.isfinite(grad_norm.detach())):
                raise FloatingPointError(f"non-finite gradient norm at step {global_step}")
            answer_stats = component_stats["answer"]
            mc_stats = component_stats.get("mc", _empty_component_stats(grad_norm))
            tf_stats = component_stats.get("tf", _empty_component_stats(grad_norm))
            oe_stats = component_stats.get("oe", _empty_component_stats(grad_norm))
            epoch_totals += torch.stack([
                answer_stats["loss_sum"], answer_stats["count"],
                mc_stats["loss_sum"], mc_stats["count"], mc_stats["pairs"], mc_stats["correct"],
                mc_stats["factual_correct"], mc_stats["counterfactual_correct"],
                tf_stats["loss_sum"], tf_stats["count"], tf_stats["pairs"], tf_stats["correct"],
                tf_stats["factual_correct"], tf_stats["counterfactual_correct"],
                oe_stats["loss_sum"], oe_stats["count"], oe_stats["pairs"],
                oe_stats["delta_sum"], oe_stats["positive_delta"], grad_norm.detach().float(),
            ]).to(device=local, dtype=torch.float64)
            epoch_steps += 1
            optimizer.step()
            global_step += 1
            if global_step == 10:
                timed_start = time.perf_counter()
            if global_step % 20 == 0:
                log = _reduce_step_stats(component_stats, grad_norm, betas)
                if rank == 0:
                    elapsed = time.perf_counter() - (timed_start or run_start)
                    measured = max(1, global_step - (10 if timed_start else 0)) / elapsed
                    log.update({
                        "stage": args.stage,
                        "epoch": epoch,
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
        torch.distributed.all_reduce(epoch_totals, op=torch.distributed.ReduceOp.SUM)
        if rank == 0:
            values = epoch_totals.tolist()
            def epoch_mean(sum_index, count_index):
                return values[sum_index] / values[count_index] if values[count_index] else 0.0
            answer_epoch_loss = epoch_mean(0, 1)
            mc_epoch_loss = epoch_mean(2, 3)
            tf_epoch_loss = epoch_mean(8, 9)
            oe_epoch_loss = epoch_mean(14, 15)
            epoch_summary = {
                "event": "epoch_summary",
                "stage": args.stage,
                "epoch": epoch,
                "global_step": global_step,
                "answer_loss": answer_epoch_loss,
                "mc_state_loss": mc_epoch_loss,
                "tf_q_loss": tf_epoch_loss,
                "oe_qcar_loss": oe_epoch_loss,
                "loss_at_epoch_end_beta": (
                    answer_epoch_loss
                    + betas["mc"] * mc_epoch_loss
                    + betas["tf"] * tf_epoch_loss
                    + betas["oe"] * oe_epoch_loss
                ),
                "mc_pairs": int(values[4]),
                "tf_pairs": int(values[10]),
                "oe_pairs": int(values[16]),
                "mc_accuracy": values[5] / values[3] if values[3] else 0.0,
                "tf_accuracy": values[11] / values[9] if values[9] else 0.0,
                "oe_mean_delta": values[17] / values[16] if values[16] else 0.0,
                "oe_positive_delta_rate": values[18] / values[16] if values[16] else 0.0,
                "beta_mc_end": betas["mc"],
                "beta_tf_end": betas["tf"],
                "beta_oe_end": betas["oe"],
                "grad_norm_mean": values[19] / (max(1, epoch_steps) * world),
                "epoch_wall_seconds": time.perf_counter() - epoch_start,
            }
            print(json.dumps(epoch_summary), flush=True)
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
                "stage": args.stage,
                "world_size": world,
                "epochs": args.epochs,
                "steps": global_step,
                "wall_seconds": elapsed,
                "series_per_second_global": global_step * world / elapsed,
                "projected_stage_hours": planned / (global_step / elapsed) / 3600,
            }, indent=2),
            encoding="utf-8",
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
