from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.llm_client import LocalHFChatClient, LocalHFVLChatClient, create_llm_client
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _question_text(row: Dict[str, Any]) -> str:
    question = row.get("question")
    if question is None:
        windows = row.get("windows") or []
        if windows and isinstance(windows[0], dict):
            question = windows[0].get("question")
    return "" if question is None else str(question).strip()


def _question_frame(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("question_frame_override")
    if value is not None:
        text = str(value).strip()
        if text:
            return text
    artifacts = row.get("question_generation_artifacts") or {}
    for key in ("frame", "frame_type", "polarity"):
        value = artifacts.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def _focus_key(row: Dict[str, Any]) -> Optional[str]:
    artifacts = row.get("question_generation_artifacts") or {}
    for key in ("focus_type", "focus", "semantic_focus"):
        value = artifacts.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    for key in ("question_family", "source_question_type", "question_answer_type"):
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                return text
    return None


def _build_pair_lookup(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    grouped: Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        pair_id = row.get("question_pair_id")
        focus_key = _focus_key(row)
        if not pair_id or not focus_key:
            continue
        grouped[(str(pair_id), str(focus_key))].append((idx, row))
    lookup: Dict[int, int] = {}
    for _, items in grouped.items():
        if len(items) < 2:
            continue
        anomaly_items = [item for item in items if _question_frame(item[1]) == "anomaly_frame"]
        normal_items = [item for item in items if _question_frame(item[1]) == "normal_frame"]
        if not anomaly_items or not normal_items:
            continue
        for anomaly_idx, _ in anomaly_items:
            for normal_idx, _ in normal_items:
                lookup[anomaly_idx] = normal_idx
                lookup[normal_idx] = anomaly_idx
    return lookup


def _permute_sample(sample: Dict[str, Any], permutation: List[int]) -> Dict[str, Any]:
    permuted = copy.deepcopy(sample)
    series = permuted.get("series") or {}
    series["values"] = [[row[idx] for idx in permutation] for row in series.get("values") or []]
    series["labels"] = [[row[idx] for idx in permutation] for row in series.get("labels") or []]
    permuted["series"] = series
    channels = permuted.get("channels") or []
    permuted["channels"] = [channels[idx] for idx in permutation]
    normal_values = permuted.get("normal_series")
    if normal_values is not None:
        permuted["normal_series"] = [[row[idx] for idx in permutation] for row in normal_values]
    original_data = permuted.get("original_data") or {}
    if original_data.get("normal_series") is not None:
        original_data["normal_series"] = [[row[idx] for idx in permutation] for row in original_data["normal_series"]]
    if original_data.get("time_series") is not None:
        original_data["time_series"] = [[row[idx] for idx in permutation] for row in original_data["time_series"]]
    permuted["original_data"] = original_data
    return permuted


def _masked_time_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_logits = logits[mask]
    valid_labels = labels[mask]
    if valid_logits.numel() == 0:
        return logits.new_zeros(())
    return F.cross_entropy(valid_logits, valid_labels)


def _question_attention_map(
    diagnostics: Dict[str, Any],
    num_question_queries: int,
    time_mask: torch.Tensor,
) -> Optional[torch.Tensor]:
    channel_attention = diagnostics.get("channel_attention")
    if channel_attention is None:
        return None
    if channel_attention.ndim != 4:
        return None
    question_attention = channel_attention[:, :, -int(num_question_queries) :, :]
    valid = time_mask.bool().unsqueeze(-1).unsqueeze(-1)
    denom = valid.to(question_attention.dtype).sum().clamp_min(1.0)
    return (question_attention * valid).sum(dim=1) / denom


def _load_rows(path: str | Path, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = read_jsonl(_resolve(path))
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def _freeze_local_client(query_mode: str, llm_config_path: Optional[str]) -> Optional[LocalHFChatClient | LocalHFVLChatClient]:
    mode = str(query_mode or "data_only").strip().lower()
    if mode == "data_only":
        return None
    if not llm_config_path:
        raise RuntimeError("Warmup query_mode requires --llm-config for frozen question embeddings.")
    llm_config = load_json(_resolve(llm_config_path))
    client = create_llm_client(llm_config)
    if not isinstance(client, (LocalHFChatClient, LocalHFVLChatClient)):
        raise RuntimeError("Warmup frozen_llm_embedding/hybrid currently supports only local_hf/local_hf_vl clients.")
    client.model.eval()
    for param in client.model.parameters():
        param.requires_grad = False
    if getattr(client, "perceiver", None) is not None:
        client.perceiver.eval()
        for param in client.perceiver.parameters():
            param.requires_grad = False
    return client


def _query_mode_and_mix(
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[str, float, int]:
    global_cfg = (config.get("model", {}).get("global_hint_head", {}) or {})
    mode = str(args.query_mode or global_cfg.get("query_mode", "data_only")).strip().lower()
    mix_alpha = float(args.query_mix_alpha if args.query_mix_alpha is not None else global_cfg.get("query_mix_alpha", 0.5))
    max_question_tokens = int(args.max_question_tokens if args.max_question_tokens is not None else global_cfg.get("max_question_tokens", 128))
    return mode, mix_alpha, max_question_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/question/smoke/question_regular_channelname_smoke_0603gh_combined/questions_100_student.jsonl")
    parser.add_argument("--config", default="configs/torch_fixed60_qa600_timercd.json")
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--global-anom-loss-weight", type=float, default=1.0)
    parser.add_argument("--perm-loss-weight", type=float, default=0.05)
    parser.add_argument("--pair-loss-weight", type=float, default=0.05)
    parser.add_argument("--disable-perm-loss", action="store_true")
    parser.add_argument("--disable-pair-loss", action="store_true")
    parser.add_argument("--disable-global-anomaly-head", action="store_true")
    parser.add_argument("--query-mode", choices=["data_only", "frozen_llm_embedding", "hybrid"], default=None)
    parser.add_argument("--query-mix-alpha", type=float, default=None)
    parser.add_argument("--max-question-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--progress-percent-step", type=int, default=10)
    parser.add_argument("--output-checkpoint", default="checkpoints/global_hint_warmup.pt")
    parser.add_argument("--report", default="outputs/runs/global_hint_warmup/report.json")
    parser.add_argument("--run-note", default="")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    config = load_json(_resolve(args.config))
    rows = _load_rows(args.data, args.limit)
    if not rows:
        raise RuntimeError("No rows loaded for warmup training.")
    pair_lookup = _build_pair_lookup(rows)
    query_mode, query_mix_alpha, max_question_tokens = _query_mode_and_mix(config, args)
    client = _freeze_local_client(query_mode, args.llm_config)

    checkpoint_path = args.interval_proposer_checkpoint or (config.get("train_interval_proposal") or {}).get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("Missing interval proposer checkpoint. Set --interval-proposer-checkpoint or config.train_interval_proposal.checkpoint_path.")
    ckpt = relative_to_root(ROOT, checkpoint_path)
    try:
        encoder = AXISMultivariateIntervalProposer.load(ckpt, config)
    except Exception:
        encoder, _ = AXISMultivariateIntervalProposer.load_timercd_checkpoint(ckpt, config)
    encoder.eval()
    for param in encoder.ts_encoder.parameters():
        param.requires_grad = False
    for param in encoder.anomaly_head.parameters():
        param.requires_grad = False
    encoder.global_hint_head.train()
    for param in encoder.global_hint_head.parameters():
        param.requires_grad = True

    d_proj = int(getattr(encoder.global_hint_head, "d_proj", config.get("model", {}).get("ts_encoder", {}).get("d_proj", 256)))
    global_anomaly_head = torch.nn.Linear(d_proj, 2).to(encoder.device)
    if args.disable_global_anomaly_head:
        global_anomaly_head.eval()
        for param in global_anomaly_head.parameters():
            param.requires_grad = False
    else:
        global_anomaly_head.train()

    if args.init_checkpoint:
        warm_state = torch.load(_resolve(args.init_checkpoint), map_location="cpu")
        global_state = warm_state.get("global_hint_head")
        if global_state:
            encoder.global_hint_head.load_state_dict(global_state, strict=False)
        anomaly_state = warm_state.get("global_anomaly_head")
        if anomaly_state:
            global_anomaly_head.load_state_dict(anomaly_state, strict=False)

    trainable: List[torch.nn.Parameter] = list(encoder.global_hint_head.parameters())
    if not args.disable_global_anomaly_head:
        trainable += list(global_anomaly_head.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    accum = max(1, int(args.gradient_accumulation_steps))
    history: List[Dict[str, Any]] = []
    started = time.perf_counter()

    index_order = list(range(len(rows)))
    for epoch in range(int(args.epochs)):
        random.shuffle(index_order)
        optimizer.zero_grad(set_to_none=True)
        milestones = []
        if len(index_order) > 0 and int(args.progress_percent_step) > 0:
            seen = set()
            for pct in range(int(args.progress_percent_step), 101, int(args.progress_percent_step)):
                mark = max(1, int(np.ceil(len(index_order) * pct / 100.0)))
                if mark not in seen:
                    seen.add(mark)
                    milestones.append((mark, pct))
        next_milestone = 0

        for step_idx, row_idx in enumerate(index_order):
            row = rows[row_idx]
            question_text = _question_text(row)
            forward = encoder.global_hint_forward_tensors(
                row,
                question_text=question_text,
                llm_question_embedder=client,
                query_mode=query_mode,
                query_mix_alpha=query_mix_alpha,
                max_question_tokens=max_question_tokens,
            )
            global_states = forward["global_states"]
            time_mask = forward["time_mask"].bool()
            time_labels = forward["time_labels"].long()

            logits_global = global_anomaly_head(global_states)
            loss_global = _masked_time_ce(
                logits_global[0],
                time_labels[0],
                time_mask[0],
            )

            loss_perm = global_states.new_zeros(())
            if not bool(args.disable_perm_loss):
                channel_count = int(global_states.shape[0]) if global_states.ndim == 2 else int(row["series"]["shape"][1])
                permutation = torch.randperm(channel_count).tolist()
                permuted_row = _permute_sample(row, permutation)
                forward_perm = encoder.global_hint_forward_tensors(
                    permuted_row,
                    question_text=question_text,
                    llm_question_embedder=client,
                    query_mode=query_mode,
                    query_mix_alpha=query_mix_alpha,
                    max_question_tokens=max_question_tokens,
                )
                common_mask = time_mask & forward_perm["time_mask"].bool()
                if common_mask.any():
                    loss_perm = F.mse_loss(
                        global_states[common_mask],
                        forward_perm["global_states"][common_mask],
                    )

            loss_pair = global_states.new_zeros(())
            pair_idx = pair_lookup.get(row_idx)
            if (
                not bool(args.disable_pair_loss)
                and pair_idx is not None
                and forward["query_mode_used"] != "data_only"
            ):
                pair_row = rows[pair_idx]
                pair_forward = encoder.global_hint_forward_tensors(
                    pair_row,
                    question_text=_question_text(pair_row),
                    llm_question_embedder=client,
                    query_mode=query_mode,
                    query_mix_alpha=query_mix_alpha,
                    max_question_tokens=max_question_tokens,
                )
                current_map = _question_attention_map(
                    forward["diagnostics"],
                    int(encoder.global_hint_head.question_adapter.num_question_queries),
                    forward["time_mask"][0],
                )
                pair_map = _question_attention_map(
                    pair_forward["diagnostics"],
                    int(encoder.global_hint_head.question_adapter.num_question_queries),
                    pair_forward["time_mask"][0],
                )
                if current_map is not None and pair_map is not None and current_map.shape == pair_map.shape:
                    loss_pair = F.mse_loss(current_map, pair_map)

            total_loss = (
                float(args.global_anom_loss_weight) * loss_global
                + float(0.0 if args.disable_perm_loss else args.perm_loss_weight) * loss_perm
                + float(0.0 if args.disable_pair_loss else args.pair_loss_weight) * loss_pair
            )
            (total_loss / accum).backward()

            if (step_idx + 1) % accum == 0 or step_idx + 1 == len(index_order):
                torch.nn.utils.clip_grad_norm_(trainable, float(args.max_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            record = {
                "epoch": epoch + 1,
                "row_index": int(row_idx),
                "sample_id": row.get("sample_id"),
                "question_pair_id": row.get("question_pair_id"),
                "query_mode_requested": forward["query_mode_requested"],
                "query_mode_used": forward["query_mode_used"],
                "loss_global_anom": float(loss_global.detach().cpu()),
                "loss_perm": float(loss_perm.detach().cpu()),
                "loss_pair": float(loss_pair.detach().cpu()),
                "loss_total": float(total_loss.detach().cpu()),
                "time_anomaly_rate": float(time_labels[0][time_mask[0]].float().mean().detach().cpu()) if time_mask[0].any() else 0.0,
            }
            history.append(record)

            while next_milestone < len(milestones) and (step_idx + 1) >= milestones[next_milestone][0]:
                _, pct = milestones[next_milestone]
                print(
                    json.dumps(
                        {
                            "stage": "warmup_progress",
                            "epoch": epoch + 1,
                            "progress_percent": pct,
                            "progress": f"{step_idx + 1}/{len(index_order)}",
                            **record,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_milestone += 1

    output_path = _resolve(args.output_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_hint_head": encoder.global_hint_head.state_dict(),
            "global_anomaly_head": global_anomaly_head.state_dict(),
            "train_args": vars(args),
            "query_mode": query_mode,
            "query_mix_alpha": float(query_mix_alpha),
            "max_question_tokens": int(max_question_tokens),
            "run_note": str(args.run_note),
        },
        output_path,
    )

    report = {
        "checkpoint_path": str(output_path),
        "num_samples": len(rows),
        "num_paired_samples": int(len(pair_lookup)),
        "epochs": int(args.epochs),
        "query_mode": query_mode,
        "query_mix_alpha": float(query_mix_alpha),
        "max_question_tokens": int(max_question_tokens),
        "global_anom_loss_weight": float(args.global_anom_loss_weight),
        "perm_loss_weight": float(0.0 if args.disable_perm_loss else args.perm_loss_weight),
        "pair_loss_weight": float(0.0 if args.disable_pair_loss else args.pair_loss_weight),
        "mean_total_loss": float(sum(x["loss_total"] for x in history) / max(1, len(history))),
        "mean_global_anom_loss": float(sum(x["loss_global_anom"] for x in history) / max(1, len(history))),
        "mean_perm_loss": float(sum(x["loss_perm"] for x in history) / max(1, len(history))),
        "mean_pair_loss": float(sum(x["loss_pair"] for x in history) / max(1, len(history))),
        "elapsed_seconds": time.perf_counter() - started,
        "history_tail": history[-10:],
        "run_note": str(args.run_note),
    }
    report_path = _resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
