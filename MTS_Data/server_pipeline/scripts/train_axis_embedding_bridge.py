from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.answer_schema import (
    EXACT_LABEL_ANSWER_FORMAT,
    normalize_answer_label,
    schema_label_exact_teacher,
    target_answer_label,
)
from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.generator import ANOMALY_TYPES
from src.mvaxis.llm_client import LocalHFChatClient
from src.mvaxis.prompts import AXIS_HINT_TOKEN_PLACEHOLDER, build_prompt
from src.mvaxis.proposal import propose_interval, sample_with_interval
from src.mvaxis.question_provider import ensure_questions
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


SYSTEM_PROMPT = (
    "Return exactly one valid json object following the requested schema. "
    "The first generated character must be { and the last generated character must be }. "
    "Do not include markdown fences, bullet lists, prose before json, or prose after json. "
    "The target interval was proposed by an anomaly head; "
    "verify it against evidence and do not assume it is anomalous."
)


COMPACT_SYSTEM_PROMPT = "You are a concise JSON API for multivariate time-series QA. Return one valid compact JSON object only."


def _messages(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _axis_hint_block(client: LocalHFChatClient, global_token_count: int, channel_token_count: int) -> str:
    return f"{client.GLOBAL_HINT_TOKEN * global_token_count}{client.CHANNEL_HINT_TOKEN * channel_token_count}"


def _inject_axis_hint_tokens(
    prompt: str,
    client: LocalHFChatClient,
    global_token_count: int,
    channel_token_count: int,
) -> str:
    token_block = _axis_hint_block(client, global_token_count, channel_token_count)
    if AXIS_HINT_TOKEN_PLACEHOLDER in prompt:
        return prompt.replace(AXIS_HINT_TOKEN_PLACEHOLDER, token_block, 1)
    return prompt + "\n\n" + token_block + "\n"


def _apply_prompt_variant(prompt: str, variant: str) -> str:
    if variant == "current":
        return prompt
    if variant == "compact_json":
        head = prompt.split("### Answer Format", 1)[0].rstrip()
        return (
            head
            + "\n\n### Answer Format\n"
            "Return exactly one compact JSON object and no markdown. Keep every string short.\n"
            "Required top-level keys: fact_check, evidence_chain, reasoning_process, reasoning_summary, question_answer, final_answer, abnormality_score, answer_confidence.\n"
            "fact_check keys: is_anomalous, root_cause_channel, root_cause_channels, affected_channels, anomaly_type, anomaly_scope, abnormal_edges, causal_path.\n"
            "Use null or [] when unsupported. evidence_chain must contain 1 or 2 short complete sentences. "
            "reasoning_process must contain 2 or 3 short observable reasoning steps. reasoning_summary must be one short sentence. question_answer must answer the exact question or choice. "
            "abnormality_score and answer_confidence must be numbers from 0 to 1."
        )
    if variant == "compact_label_json":
        head = prompt.split("### Answer Format", 1)[0].rstrip()
        return head + "\n\n### Answer Format\n" + EXACT_LABEL_ANSWER_FORMAT
    raise ValueError(f"Unsupported prompt variant for bridge training: {variant}")


def _system_prompt_for_variant(variant: str) -> str:
    return COMPACT_SYSTEM_PROMPT if variant in {"compact_json", "compact_label_json"} else SYSTEM_PROMPT


def _downsample_tensor(client: LocalHFChatClient, embeddings: Any, max_tokens: int) -> torch.Tensor:
    if isinstance(embeddings, torch.Tensor):
        tensor = embeddings.to(device=client.device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(embeddings, dtype=torch.float32, device=client.device)
    if tensor.ndim != 2:
        raise ValueError("AXIS hint embeddings must have shape [num_tokens, d_proj]")
    if tensor.shape[0] > int(max_tokens):
        idx = torch.linspace(0, tensor.shape[0] - 1, int(max_tokens), device=client.device).long()
        tensor = tensor.index_select(0, idx)
    return tensor


def _downsample_hint_bundle(client: LocalHFChatClient, hint_bundle: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    axis_cfg = client.axis_hints_config
    max_global = int(axis_cfg.get("max_global_tokens", axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens", 4))))
    max_channel = int(axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens", 64)))
    embeddings = hint_bundle["embeddings"]
    return {
        "global": _downsample_tensor(client, embeddings["global"], max_global),
        "channel": _downsample_tensor(client, embeddings["channel"], max_channel),
    }


def _answer_label_from_output(target_output: Dict[str, Any]) -> Optional[str]:
    return target_answer_label(target_output)


def _teacher_answer(target_output: Dict[str, Any], mode: str = "full") -> Dict[str, Any]:
    fact = dict(target_output.get("fact_check") or {})
    if mode == "schema_label_exact":
        return schema_label_exact_teacher(target_output)
    if mode == "label_only":
        is_anomalous = bool(fact.get("is_anomalous", False))
        compact_fact = {
            "is_anomalous": is_anomalous,
            "anomaly_type": fact.get("anomaly_type") if is_anomalous else None,
            "anomaly_scope": fact.get("anomaly_scope") if is_anomalous else None,
            "answer_label": _answer_label_from_output(target_output),
            "choice_answer": normalize_answer_label(fact.get("choice_answer")) if fact.get("question_answer_type") == "choice" else None,
            "question_answer_type": fact.get("question_answer_type"),
            "question_id": fact.get("question_id"),
        }
        return {
            "fact_check": compact_fact,
            "evidence_chain": target_output.get("evidence_chain") or [],
            "reasoning_process": target_output.get("reasoning_process") or [],
            "reasoning_summary": _short_text(str(target_output.get("reasoning_summary", "")), 300),
            "question_answer": _short_text(str(target_output.get("question_answer", target_output.get("final_answer", ""))), 240),
            "final_answer": _short_text(str(target_output.get("final_answer", "")), 240),
            "abnormality_score": float(target_output.get("abnormality_score", 0.0)),
            "answer_confidence": float(target_output.get("answer_confidence", 0.0)),
        }
    compact_fact = {
        "is_anomalous": bool(fact.get("is_anomalous", False)),
        "root_cause_channel": fact.get("root_cause_channel"),
        "root_cause_channels": list(fact.get("root_cause_channels") or ([] if fact.get("root_cause_channel") is None else [fact.get("root_cause_channel")])),
        "affected_channels": list(fact.get("affected_channels") or []),
        "anomaly_type": fact.get("anomaly_type"),
        "anomaly_scope": fact.get("anomaly_scope"),
        "abnormal_edges": list(fact.get("abnormal_edges") or []),
        "causal_path": list(fact.get("causal_path") or []),
        "has_relation_break": fact.get("has_relation_break"),
        "left_right_gap_level": fact.get("left_right_gap_level", "unknown"),
    }
    return {
        "fact_check": compact_fact,
        "evidence_chain": target_output.get("evidence_chain") or [],
        "reasoning_process": target_output.get("reasoning_process") or [],
        "reasoning_summary": _short_text(str(target_output.get("reasoning_summary", "")), 360),
        "question_answer": _short_text(str(target_output.get("question_answer", target_output.get("final_answer", ""))), 300),
        "final_answer": _short_text(str(target_output.get("final_answer", "")), 300),
        "abnormality_score": float(target_output.get("abnormality_score", 0.0)),
        "answer_confidence": float(target_output.get("answer_confidence", 0.0)),
    }


def _short_text(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _sample_interval(
    sample: Dict[str, Any],
    encoder: AXISMultivariateIntervalProposer,
    config: Dict[str, Any],
    interval_source: str,
    window_size: int,
    stride: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if interval_source == "truth":
        start = int(sample["target_interval"]["start"])
        end = int(sample["target_interval"]["end"])
        proposal = {
            "start": start,
            "end": end,
            "proposal_score": None,
            "source": "synthetic_truth_interval_for_bridge_training",
            "predicted_root_cause_channel": None,
        }
        return sample_with_interval(sample, start, end), proposal
    if interval_source != "proposal":
        raise ValueError(f"Unsupported interval_source: {interval_source}")
    proposal = propose_interval(
        sample,
        encoder,
        hint_tuner=None,
        window_size=window_size,
        stride=stride,
        top_k_channels=int(config["model"]["top_k_channels"]),
    )
    if proposal["start"] is None:
        return sample, proposal
    return sample_with_interval(sample, int(proposal["start"]), int(proposal["end"])), proposal


def _build_training_item(
    sample: Dict[str, Any],
    encoder: AXISMultivariateIntervalProposer,
    client: LocalHFChatClient,
    config: Dict[str, Any],
    interval_source: str,
    window_size: int,
    stride: int,
    prompt_variant: str,
    teacher_answer_mode: str,
) -> Tuple[str, str, Dict[str, torch.Tensor], Dict[str, Any]]:
    prompt_sample, proposal = _sample_interval(sample, encoder, config, interval_source, window_size, stride)
    encoder.global_hint_head.train()
    mi = convert_to_model_input(prompt_sample)
    interval_context = {
        "interval_source": "synthetic_truth_interval_for_bridge_training"
        if interval_source == "truth"
        else "encoder_or_anomaly_head_sliding_window_proposal",
        "instruction": "Analyze this interval, then decide whether it is truly anomalous.",
        "hint_delivery": "AXIS embedding hints are injected at the input-embedding level.",
    }
    start, end = mi.interval
    axis_cfg = client.axis_hints_config
    if hasattr(encoder, "make_embedding_hint_tensors"):
        hint_bundle = encoder.make_embedding_hint_tensors(
            prompt_sample,
            start,
            end,
            top_k_channels=axis_cfg.get("top_k_channels"),
            max_channel_tokens=axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens")),
            max_global_tokens=axis_cfg.get("max_global_tokens", axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens"))),
        )
    else:
        hint_bundle = encoder.make_embedding_hints(
            prompt_sample,
            start,
            end,
            top_k_channels=axis_cfg.get("top_k_channels"),
            max_channel_tokens=axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens")),
            max_global_tokens=axis_cfg.get("max_global_tokens", axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens"))),
        )
    prompt = build_prompt(
        mi,
        hint_summary={},
        interval_context=interval_context,
        include_soft_hints=False,
        include_normal_counterpart=False,
        embedding_hint_trace=hint_bundle["trace"],
    )
    prompt = _apply_prompt_variant(prompt, prompt_variant)
    hint_tensors = _downsample_hint_bundle(client, hint_bundle)
    prompt_with_hints = _inject_axis_hint_tokens(
        prompt,
        client,
        int(hint_tensors["global"].shape[0]),
        int(hint_tensors["channel"].shape[0]),
    )
    answer = json.dumps(_teacher_answer(sample["target_output"], teacher_answer_mode), ensure_ascii=False, separators=(",", ":")) + (
        client.tokenizer.eos_token or ""
    )
    return prompt_with_hints, answer, hint_tensors, proposal


def _channel_index(channel_id: Any, num_channels: int) -> int:
    if not isinstance(channel_id, str) or not channel_id.startswith("ch_"):
        return num_channels
    try:
        idx = int(channel_id.split("_", 1)[1])
    except ValueError:
        return num_channels
    return idx if 0 <= idx < num_channels else num_channels


def _fact_labels(sample: Dict[str, Any], num_channels: int, device: torch.device) -> Dict[str, torch.Tensor]:
    fact = sample.get("target_output", {}).get("fact_check", {})
    root_label = _channel_index(fact.get("root_cause_channel"), num_channels)
    type_to_idx = {name: i for i, name in enumerate(ANOMALY_TYPES)}
    type_label = type_to_idx.get(fact.get("anomaly_type"), len(ANOMALY_TYPES))
    affected = torch.zeros(num_channels, dtype=torch.float32, device=device)
    for channel_id in fact.get("affected_channels") or []:
        idx = _channel_index(channel_id, num_channels)
        if idx < num_channels:
            affected[idx] = 1.0
    return {
        "root": torch.tensor([root_label], dtype=torch.long, device=device),
        "type": torch.tensor([type_label], dtype=torch.long, device=device),
        "affected": affected.unsqueeze(0),
    }


def _label_string(sample: Dict[str, Any], key: str) -> Optional[str]:
    target = sample.get("target_output") or {}
    fact = target.get("fact_check") or {}
    if key == "anomaly_type":
        if not bool(fact.get("is_anomalous", False)):
            return "normal"
        return str(fact.get("anomaly_type") or "unknown_anomaly_type")
    if key == "answer":
        return _answer_label_from_output(target)
    raise ValueError(f"Unsupported label key: {key}")


def _build_label_vocab(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    labels = sorted({label for row in rows if (label := _label_string(row, key)) is not None})
    if key == "anomaly_type" and "normal" not in labels:
        labels.insert(0, "normal")
    return {label: idx for idx, label in enumerate(labels)}


def _global_label_auxiliary_loss(
    global_hint: torch.Tensor,
    sample: Dict[str, Any],
    aux_heads: Dict[str, nn.Module],
    anomaly_type_vocab: Dict[str, int],
    answer_label_vocab: Dict[str, int],
    *,
    type_weight: float,
    answer_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: List[torch.Tensor] = []
    stats: Dict[str, float] = {}
    pooled = global_hint.unsqueeze(0)
    type_label = _label_string(sample, "anomaly_type") or "unknown_anomaly_type"
    type_idx = anomaly_type_vocab.get(type_label, anomaly_type_vocab.get("unknown_anomaly_type", 0))
    type_target = torch.tensor([type_idx], dtype=torch.long, device=global_hint.device)
    type_logits = aux_heads["anomaly_type"](pooled)
    type_loss = F.cross_entropy(type_logits, type_target)
    losses.append(float(type_weight) * type_loss)
    stats["aux_type_loss"] = float(type_loss.detach().cpu())
    stats["aux_type_acc"] = float((type_logits.argmax(dim=-1) == type_target).float().mean().detach().cpu())
    stats["aux_type_label"] = type_label

    answer_label = _label_string(sample, "answer")
    if answer_label is not None and answer_label in answer_label_vocab and answer_label_vocab:
        answer_target = torch.tensor([answer_label_vocab[answer_label]], dtype=torch.long, device=global_hint.device)
        answer_logits = aux_heads["answer"](pooled)
        answer_loss = F.cross_entropy(answer_logits, answer_target)
        losses.append(float(answer_weight) * answer_loss)
        stats["aux_answer_loss"] = float(answer_loss.detach().cpu())
        stats["aux_answer_acc"] = float((answer_logits.argmax(dim=-1) == answer_target).float().mean().detach().cpu())
        stats["aux_answer_label"] = answer_label
    else:
        stats["aux_answer_loss"] = 0.0
        stats["aux_answer_acc"] = 0.0
        stats["aux_answer_label"] = answer_label or "skipped_open_answer"

    if not losses:
        return global_hint.new_tensor(0.0), stats
    total = torch.stack(losses).sum()
    stats["weighted_aux_loss"] = float(total.detach().cpu())
    return total, stats


def _auxiliary_loss(
    pooled_hint: torch.Tensor,
    sample: Dict[str, Any],
    aux_heads: Dict[str, nn.Module],
    num_channels: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    labels = _fact_labels(sample, num_channels, pooled_hint.device)
    root_logits = aux_heads["root"](pooled_hint.unsqueeze(0))
    type_logits = aux_heads["anomaly_type"](pooled_hint.unsqueeze(0))
    affected_logits = aux_heads["affected"](pooled_hint.unsqueeze(0))
    root_loss = F.cross_entropy(root_logits, labels["root"])
    type_loss = F.cross_entropy(type_logits, labels["type"])
    affected_loss = F.binary_cross_entropy_with_logits(affected_logits, labels["affected"])
    loss = root_loss + type_loss + affected_loss
    with torch.no_grad():
        affected_pred = torch.sigmoid(affected_logits).ge(0.5).float()
        affected_truth = labels["affected"]
        tp = (affected_pred * affected_truth).sum()
        precision = tp / torch.clamp(affected_pred.sum(), min=1.0)
        recall = tp / torch.clamp(affected_truth.sum(), min=1.0)
        f1 = (2 * precision * recall / torch.clamp(precision + recall, min=1e-8)).item()
    return loss, {
        "aux_root_loss": float(root_loss.detach().cpu()),
        "aux_type_loss": float(type_loss.detach().cpu()),
        "aux_affected_loss": float(affected_loss.detach().cpu()),
        "aux_root_acc": float((root_logits.argmax(dim=-1) == labels["root"]).float().mean().detach().cpu()),
        "aux_type_acc": float((type_logits.argmax(dim=-1) == labels["type"]).float().mean().detach().cpu()),
        "aux_affected_f1": float(f1),
    }


def _loss_for_item(
    client: LocalHFChatClient,
    prompt: str,
    answer: str,
    hint_tensors: Dict[str, torch.Tensor],
    system_prompt: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if client.perceiver is None:
        raise RuntimeError("axis_hints.enabled must be true for bridge training")
    messages = _messages(prompt, system_prompt=system_prompt)
    prompt_text = client._render_messages(messages)
    full_text = prompt_text + answer
    prompt_ids = client.tokenizer(prompt_text, return_tensors="pt").to(client.device)
    full = client.tokenizer(full_text, return_tensors="pt").to(client.device)
    input_ids = full["input_ids"]
    attention_mask = full["attention_mask"]
    labels = input_ids.clone()
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    labels[:, :prompt_len] = -100

    input_embeddings = client.model.get_input_embeddings()(input_ids).detach()
    embedding_device = input_embeddings.device
    embedding_dtype = input_embeddings.dtype
    # Keep the trainable bridge in fp32 even when the LLM runs in fp16.
    # AdamW on a tiny fp16 Perceiver is numerically fragile and quickly
    # produces NaNs; only the final injected embeddings are cast to the LLM dtype.
    if next(client.perceiver.parameters()).device != embedding_device or next(client.perceiver.parameters()).dtype != torch.float32:
        client.perceiver.to(device=embedding_device, dtype=torch.float32)
    word_embeddings = client.model.get_input_embeddings().weight.detach().to(device=embedding_device, dtype=torch.float32)
    source_embeddings = client.perceiver.get_source_embeddings(word_embeddings)
    global_tensor = hint_tensors["global"].to(device=embedding_device, dtype=torch.float32)
    channel_tensor = hint_tensors["channel"].to(device=embedding_device, dtype=torch.float32)
    projected_global = client.perceiver.process_local_embeddings(global_tensor, source_embeddings)
    projected_channel = client.perceiver.process_local_embeddings(channel_tensor, source_embeddings)

    global_positions = (input_ids[0] == client.global_hint_token_id).nonzero(as_tuple=True)[0]
    channel_positions = (input_ids[0] == client.channel_hint_token_id).nonzero(as_tuple=True)[0]
    neutral_ids = client.tokenizer("\n", add_special_tokens=False)["input_ids"]
    neutral_id = int(neutral_ids[0]) if neutral_ids else int(client.tokenizer.eos_token_id)
    neutral_embedding = client.model.get_input_embeddings().weight[neutral_id].detach().to(
        device=embedding_device,
        dtype=input_embeddings.dtype,
    )
    if len(global_positions) != projected_global.shape[0]:
        raise RuntimeError(
            f"global hint token count {len(global_positions)} does not match global embeddings {projected_global.shape[0]}"
        )
    if len(channel_positions) != projected_channel.shape[0]:
        raise RuntimeError(
            f"channel hint token count {len(channel_positions)} does not match channel embeddings {projected_channel.shape[0]}"
        )
    global_scale = float(client.axis_hints_config.get("global_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
    channel_scale = float(client.axis_hints_config.get("channel_injection_scale", client.axis_hints_config.get("injection_scale", 1.0)))
    global_scale = max(0.0, min(1.0, global_scale))
    channel_scale = max(0.0, min(1.0, channel_scale))
    global_injected = (
        (1.0 - global_scale) * neutral_embedding.unsqueeze(0).expand(len(global_positions), -1)
        + global_scale * projected_global.to(input_embeddings.dtype)
    )
    input_embeddings = input_embeddings.index_copy(
        1,
        global_positions,
        global_injected.unsqueeze(0),
    )
    channel_injected = (
        (1.0 - channel_scale) * neutral_embedding.unsqueeze(0).expand(len(channel_positions), -1)
        + channel_scale * projected_channel.to(input_embeddings.dtype)
    )
    input_embeddings = input_embeddings.index_copy(
        1,
        channel_positions,
        channel_injected.unsqueeze(0),
    )

    outputs = client.model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    pooled_global = projected_global.mean(dim=0)
    pooled_hint = torch.cat([projected_global, projected_channel], dim=0).mean(dim=0)
    return outputs.loss, pooled_global, pooled_hint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_smoke.json")
    parser.add_argument("--llm-config", default="configs/local_qwen25_3b_cuda.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--interval-source", choices=["truth", "proposal"], default="truth")
    parser.add_argument("--prompt-variant", choices=["current", "compact_json", "compact_label_json"], default="compact_json")
    parser.add_argument("--teacher-answer-mode", choices=["full", "label_only", "schema_label_exact"], default="full")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--aux-mode", choices=["legacy", "label_global"], default="legacy")
    parser.add_argument("--aux-root-weight", type=float, default=0.2)
    parser.add_argument("--aux-type-weight", type=float, default=0.2)
    parser.add_argument("--aux-affected-weight", type=float, default=0.1)
    parser.add_argument("--aux-answer-weight", type=float, default=0.2)
    parser.add_argument("--output-checkpoint", default="outputs/checkpoints/qwen25_3b_axis_perceiver_bridge.pt")
    parser.add_argument("--output-encoder-checkpoint", default=None)
    parser.add_argument("--report", default="outputs/qwen25_3b_axis_bridge_train_report.json")
    parser.add_argument("--question-provider", default="bank", choices=["bank", "none"])
    parser.add_argument("--question-seed", type=int, default=520)
    parser.add_argument("--overwrite-questions", action="store_true")
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    llm_config = load_json(ROOT / args.llm_config)
    axis_cfg = llm_config.setdefault("axis_hints", {})
    axis_cfg["enabled"] = True
    checkpoint_path = axis_cfg.get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = Path(str(checkpoint_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.exists():
            axis_cfg["checkpoint_path"] = None
    llm_config["use_cache"] = False

    rows = read_jsonl(ROOT / config["data"]["output_dir"] / f"{args.split}.jsonl")
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = ensure_questions(
        rows,
        seed=int(args.question_seed),
        provider=args.question_provider,
        overwrite=bool(args.overwrite_questions),
        overwrite_fixed_placeholder=True,
        copy_samples=False,
    )
    channel_counts = sorted(
        {
            int((row.get("series") or {}).get("shape", [0, len(row.get("channels") or [])])[1])
            for row in rows
        }
    )
    if len(channel_counts) > 1:
        raise RuntimeError(
            "train_axis_embedding_bridge.py still assumes a fixed channel count for its legacy auxiliary heads. "
            f"Found mixed widths in the dataset: {channel_counts}. "
            "Please group samples by channel count or use mvaxis_train_nexttoken_hints.py for any-variate global-hint training."
        )
    system_prompt = _system_prompt_for_variant(args.prompt_variant)

    proposal_ckpt = relative_to_root(
        ROOT,
        args.interval_proposer_checkpoint or config["train_interval_proposal"]["checkpoint_path"],
    )
    try:
        encoder = AXISMultivariateIntervalProposer.load(proposal_ckpt, config)
    except Exception:
        encoder, _ = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
            proposal_ckpt,
            config,
            threshold=float((config.get("interval_proposal") or {}).get("threshold", 0.5)),
        )
    encoder.eval()
    for param in encoder.ts_encoder.parameters():
        param.requires_grad = False
    for param in encoder.anomaly_head.parameters():
        param.requires_grad = False
    encoder.global_hint_head.train()
    for param in encoder.global_hint_head.parameters():
        param.requires_grad = True

    client = LocalHFChatClient(llm_config)
    client.model.eval()
    client.model.config.use_cache = False
    if hasattr(client.model, "gradient_checkpointing_enable"):
        try:
            client.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            client.model.gradient_checkpointing_enable()
    for param in client.model.parameters():
        param.requires_grad = False
    if client.perceiver is None:
        raise RuntimeError("Local LLM config must enable axis_hints")
    client.perceiver.train()
    for param in client.perceiver.parameters():
        param.requires_grad = True
    hidden_size = int(client.model.config.hidden_size)
    num_channels = int(config["data"]["num_channels"])
    anomaly_type_vocab = _build_label_vocab(rows, "anomaly_type")
    answer_label_vocab = _build_label_vocab(rows, "answer")
    if args.aux_mode == "label_global":
        aux_heads = nn.ModuleDict(
            {
                "anomaly_type": nn.Linear(hidden_size, max(1, len(anomaly_type_vocab))),
                "answer": nn.Linear(hidden_size, max(1, len(answer_label_vocab))),
            }
        ).to(device=client.device, dtype=torch.float32)
    else:
        aux_heads = nn.ModuleDict(
            {
                "root": nn.Linear(hidden_size, num_channels + 1),
                "anomaly_type": nn.Linear(hidden_size, len(ANOMALY_TYPES) + 1),
                "affected": nn.Linear(hidden_size, num_channels),
            }
        ).to(device=client.device, dtype=torch.float32)
    aux_heads.train()
    for param in aux_heads.parameters():
        param.requires_grad = True

    trainable_params = list(client.perceiver.parameters()) + list(encoder.global_hint_head.parameters())
    if float(args.aux_loss_weight) > 0:
        trainable_params += list(aux_heads.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    started = time.perf_counter()
    history: List[Dict[str, Any]] = []
    global_step = 0
    accum = max(1, int(args.gradient_accumulation_steps))
    for epoch in range(int(args.epochs)):
        epoch_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for idx, sample in enumerate(rows):
            prompt, answer, hint_tensors, proposal = _build_training_item(
                sample,
                encoder,
                client,
                config,
                args.interval_source,
                int(args.window_size),
                int(args.stride),
                str(args.prompt_variant),
                str(args.teacher_answer_mode),
            )
            lm_loss, pooled_global, pooled_hint = _loss_for_item(client, prompt, answer, hint_tensors, system_prompt)
            loss = lm_loss
            aux_stats: Dict[str, float] = {}
            if float(args.aux_loss_weight) > 0:
                if args.aux_mode == "label_global":
                    weighted_aux_loss, aux_stats = _global_label_auxiliary_loss(
                        pooled_global,
                        sample,
                        aux_heads,
                        anomaly_type_vocab,
                        answer_label_vocab,
                        type_weight=float(args.aux_type_weight),
                        answer_weight=float(args.aux_answer_weight),
                    )
                    loss = loss + float(args.aux_loss_weight) * weighted_aux_loss
                else:
                    aux_loss, aux_stats = _auxiliary_loss(pooled_hint, sample, aux_heads, num_channels)
                    weighted_aux = (
                        float(args.aux_root_weight) * aux_stats["aux_root_loss"]
                        + float(args.aux_type_weight) * aux_stats["aux_type_loss"]
                        + float(args.aux_affected_weight) * aux_stats["aux_affected_loss"]
                    )
                    # Recompute weighted tensor loss so gradients flow through the heads and Perceiver.
                    labels = _fact_labels(sample, num_channels, pooled_hint.device)
                    root_logits = aux_heads["root"](pooled_hint.unsqueeze(0))
                    type_logits = aux_heads["anomaly_type"](pooled_hint.unsqueeze(0))
                    affected_logits = aux_heads["affected"](pooled_hint.unsqueeze(0))
                    weighted_aux_loss = (
                        float(args.aux_root_weight) * F.cross_entropy(root_logits, labels["root"])
                        + float(args.aux_type_weight) * F.cross_entropy(type_logits, labels["type"])
                        + float(args.aux_affected_weight) * F.binary_cross_entropy_with_logits(affected_logits, labels["affected"])
                    )
                    loss = loss + float(args.aux_loss_weight) * weighted_aux_loss
                    aux_stats["weighted_aux_loss"] = float(weighted_aux)
            (loss / accum).backward()
            epoch_loss += float(loss.detach().cpu())
            seen += 1
            if seen % accum == 0 or idx == len(rows) - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available() and global_step % 8 == 0:
                    torch.cuda.empty_cache()
                global_step += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "sample_index": idx,
                    "sample_id": sample["sample_id"],
                    "loss": float(loss.detach().cpu()),
                    "lm_loss": float(lm_loss.detach().cpu()),
                    "interval_source": args.interval_source,
                    "proposal_start": proposal.get("start"),
                    "proposal_end": proposal.get("end"),
                    "global_hint_tokens": int(hint_tensors["global"].shape[0]),
                    "channel_hint_tokens": int(hint_tensors["channel"].shape[0]),
                    **aux_stats,
                }
            )
        print(json.dumps({"epoch": epoch + 1, "mean_loss": epoch_loss / max(1, seen), "samples": seen}, ensure_ascii=False))

    output_path = ROOT / args.output_checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_encoder_checkpoint = (
        ROOT / args.output_encoder_checkpoint
        if args.output_encoder_checkpoint
        else output_path.with_name(output_path.stem + "_encoder.pt")
    )
    output_encoder_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    saved_axis_hints = dict(llm_config.get("axis_hints") or {})
    saved_axis_hints["checkpoint_path"] = str(Path(args.output_checkpoint).as_posix())
    torch.save(
        {
            "perceiver": client.perceiver.state_dict(),
            "global_hint_head": encoder.global_hint_head.state_dict(),
            "aux_heads": aux_heads.state_dict(),
            "llm_model": llm_config.get("model"),
            "axis_hints": saved_axis_hints,
            "aux_mode": args.aux_mode,
            "anomaly_type_vocab": anomaly_type_vocab,
            "answer_label_vocab": answer_label_vocab,
            "train_args": vars(args),
        },
        output_path,
    )
    encoder.save(str(output_encoder_checkpoint))
    report = {
        "checkpoint_path": str(output_path),
        "encoder_checkpoint_path": str(output_encoder_checkpoint),
        "num_samples": len(rows),
        "epochs": int(args.epochs),
        "global_steps": global_step,
        "mean_loss": sum(x["loss"] for x in history) / max(1, len(history)),
        "question_provider": args.question_provider,
        "question_seed": int(args.question_seed),
        "overwrite_questions": bool(args.overwrite_questions),
        "teacher_answer_mode": args.teacher_answer_mode,
        "aux_mode": args.aux_mode,
        "anomaly_type_vocab": anomaly_type_vocab,
        "answer_label_vocab": answer_label_vocab,
        "elapsed_seconds": time.perf_counter() - started,
        "history_tail": history[-10:],
    }
    save_json(report, ROOT / args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
