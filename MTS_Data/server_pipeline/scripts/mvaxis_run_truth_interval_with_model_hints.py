from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from run_llm_on_proposals import _make_axis_embedding_hint_bundle, _uses_axis_embedding_hints
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.llm_client import MockLLMClient, create_llm_client
from src.mvaxis.prompts import build_prompt, parse_json_output
from src.mvaxis.proposal import (
    affected_f1,
    interval_coverage,
    interval_iou,
    sample_with_interval,
    score_explanation_against_truth,
    true_affected_channels,
    true_anomaly_interval,
)
from src.mvaxis.question_provider import ensure_questions
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


SYSTEM_PROMPT = (
    "Return exactly one valid json object following the requested schema. "
    "The first generated character must be { and the last generated character must be }. "
    "Do not include markdown fences, bullet lists, prose before json, or prose after json. "
    "The target interval is the evaluation interval; verify it against evidence and model hints."
)


def _messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _parse_or_empty(text: str) -> Dict[str, Any]:
    try:
        return parse_json_output(text)
    except Exception:
        return {}


def _active_indices(sample: Dict[str, Any]) -> List[int]:
    channels = sample["channels"]
    active = [idx for idx, ch in enumerate(channels) if ch.get("active", True)]
    return active or list(range(len(channels)))


def _local_hints_from_probs(
    sample: Dict[str, Any],
    probs: np.ndarray,
    start: int,
    end: int,
    *,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    active = _active_indices(sample)
    active_probs = probs[start:end][:, active]
    if active_probs.size == 0:
        return []
    time_score = active_probs.max(axis=1)
    top = np.argsort(-time_score)[: min(limit, len(time_score))]
    hints = []
    for rel_idx in top:
        abs_idx = start + int(rel_idx)
        ch_idx = int(active[int(np.argmax(probs[abs_idx, active]))])
        hints.append(
            {
                "time_index": abs_idx,
                "relative_index": int(rel_idx),
                "score": float(probs[abs_idx, ch_idx]),
                "top_channel_id": sample["channels"][ch_idx]["channel_id"],
            }
        )
    return hints


def _truth_interval_model_hints(
    sample: Dict[str, Any],
    encoder: AXISMultivariateIntervalProposer,
    top_k_channels: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    truth = true_anomaly_interval(sample)
    if truth is None:
        start = int(sample["target_interval"]["start"])
        end = int(sample["target_interval"]["end"])
        interval_source = "target_interval_no_anomaly"
    else:
        start, end = int(truth[0]), int(truth[1])
        interval_source = "oracle_truth_interval_model_hints"

    probs = encoder.anomaly_probabilities(sample)
    active = _active_indices(sample)
    window_probs = probs[start:end][:, active]
    if window_probs.size == 0:
        score = 0.0
        max_point = 0.0
        channel_order: List[int] = active[:]
    else:
        time_score = window_probs.max(axis=1)
        score = float(time_score.mean())
        max_point = float(time_score.max())
        channel_order = [
            int(active[i])
            for i in np.argsort(-window_probs.max(axis=0))[: max(1, min(top_k_channels, len(active)))]
        ]
    threshold = float(getattr(encoder, "threshold", 0.5))
    is_anomalous_proposal = bool(score >= threshold)
    candidate_root = sample["channels"][channel_order[0]]["channel_id"] if channel_order else None
    predicted_root = candidate_root if is_anomalous_proposal else None
    channel_hints = [
        {
            "channel_id": sample["channels"][idx]["channel_id"],
            "score": float(probs[start:end, idx].mean()) if end > start else 0.0,
            "encoder_prob": float(probs[start:end, idx].max()) if end > start else 0.0,
        }
        for idx in channel_order
    ]
    local_hints = _local_hints_from_probs(sample, probs, start, end)
    proposal = {
        "start": start,
        "end": end,
        "proposal_score": score,
        "hint_anomaly_probability": score,
        "encoder_max_channel_probability": max_point,
        "max_point_score": max_point,
        "predicted_root_cause_channel": candidate_root,
        "channel_hints": channel_hints,
        "is_anomalous_proposal": is_anomalous_proposal,
        "threshold": threshold,
        "source": interval_source,
        "global_hint": {
            "anomaly_probability": score,
            "predicted_interval": [start, end],
            "predicted_root_cause_channel": predicted_root,
            "candidate_root_cause_channel": candidate_root,
            "is_anomalous_proposal": is_anomalous_proposal,
            "hint_source": "AXIS anomaly_head evaluated on truth interval",
        },
        "local_hints": local_hints,
        "soft_hints": {
            "interval_proposal": {
                "source": interval_source,
                "start": start,
                "end": end,
                "score": score,
                "threshold": threshold,
                "is_anomalous_proposal": is_anomalous_proposal,
            },
            "global_hint": {
                "anomaly_probability": score,
                "predicted_interval": [start, end],
                "predicted_root_cause_channel": predicted_root,
                "candidate_root_cause_channel": candidate_root,
                "is_anomalous_proposal": is_anomalous_proposal,
                "hint_source": "AXIS anomaly_head evaluated on truth interval",
            },
            "channel_hints": channel_hints,
            "local_hints": local_hints,
            "usage_policy": "Hints are model outputs computed on the truth interval, not ground truth labels.",
        },
        "ranked_candidates": [],
    }
    prompt_sample = sample_with_interval(sample, start, end)
    return prompt_sample, proposal


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    comparisons = [record["comparison"] for record in records]

    def mean(key: str, rows: Optional[List[Dict[str, Any]]] = None) -> float:
        rows = comparisons if rows is None else rows
        return sum(float(row[key]) for row in rows) / max(1, len(rows))

    abnormal = [row for row in comparisons if row.get("truth_interval") is not None]
    return {
        "num_samples": len(records),
        "json_parse_rate": sum(
            float(isinstance((record["parsed_response"] or {}).get("fact_check"), dict))
            for record in records
        )
        / max(1, len(records)),
        "proposal_mean_temporal_iou": mean("proposal_temporal_iou"),
        "proposal_mean_truth_coverage": mean("proposal_truth_coverage"),
        "proposal_abnormal_mean_temporal_iou": mean("proposal_temporal_iou", abnormal),
        "proposal_abnormal_mean_truth_coverage": mean("proposal_truth_coverage", abnormal),
        "proposal_close_rate": mean("proposal_is_close"),
        "llm_anomaly_accuracy": mean("llm_anomaly_match"),
        "llm_root_accuracy": mean("llm_root_match"),
        "llm_mean_affected_f1": mean("llm_affected_f1"),
        "llm_type_accuracy": mean("llm_type_match"),
    }


def _raw_examples(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    priorities = [
        lambda r: bool(r.get("comparison", {}).get("llm_anomaly_match") is False),
        lambda r: bool((r.get("target_output", {}).get("fact_check") or {}).get("is_anomalous")),
        lambda r: True,
    ]
    for pred in priorities:
        for idx, record in enumerate(records):
            if idx in seen or not pred(record):
                continue
            selected.append(record)
            seen.add(idx)
            if len(selected) >= limit:
                return [_raw_example(record) for record in selected]
    return [_raw_example(record) for record in selected]


def _raw_example(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "proposal": record.get("proposal"),
        "comparison": record.get("comparison"),
        "raw_response": record.get("raw_response"),
        "parsed_fact_check": (record.get("parsed_response") or {}).get("fact_check") or {},
        "target_fact_check": (record.get("target_output") or {}).get("fact_check") or {},
        "axis_embedding_hints_used": record.get("axis_embedding_hints_used"),
    }


def _prompt_examples(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    return [
        {
            "sample_id": record.get("sample_id"),
            "prompt": record.get("prompt"),
            "proposal": record.get("proposal"),
            "target_fact_check": (record.get("target_output") or {}).get("fact_check") or {},
            "axis_embedding_hints_used": record.get("axis_embedding_hints_used"),
        }
        for record in records[: max(0, limit)]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale_nocf_schema.json")
    parser.add_argument("--llm-config", default="configs/local_qwen25_7b_cuda_remote_semantic_truth_bridge.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--hint-delivery", choices=["text", "embedding"], default="text")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default="outputs/runs/truth_interval_model_hints/truth_interval_model_hints.jsonl")
    parser.add_argument("--report", default="outputs/runs/truth_interval_model_hints/report.json")
    parser.add_argument("--raw-log-examples", type=int, default=4)
    parser.add_argument("--raw-log-path", default=None)
    parser.add_argument("--dump-prompt-examples", type=int, default=2)
    parser.add_argument("--prompt-log-path", default=None)
    parser.add_argument("--question-provider", default="bank", choices=["bank", "none"])
    parser.add_argument("--question-seed", type=int, default=520)
    parser.add_argument("--overwrite-questions", action="store_true")
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    rows = read_jsonl(ROOT / config["data"]["output_dir"] / f"{args.split}.jsonl")
    rows = rows[: args.limit] if args.limit is not None else rows
    rows = ensure_questions(
        rows,
        seed=int(args.question_seed),
        provider=args.question_provider,
        overwrite=bool(args.overwrite_questions),
        overwrite_fixed_placeholder=True,
        copy_samples=False,
    )
    checkpoint = args.interval_proposer_checkpoint or config["train_interval_proposal"]["checkpoint_path"]
    checkpoint_path = relative_to_root(ROOT, checkpoint)
    try:
        encoder = AXISMultivariateIntervalProposer.load(checkpoint_path, config)
    except KeyError:
        encoder, _ = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
            checkpoint_path,
            config,
            threshold=float((config.get("interval_proposal") or {}).get("threshold", 0.5)),
        )
    llm_config = load_json(ROOT / args.llm_config)
    real_client = None if args.mock else create_llm_client(llm_config)
    mock_client = MockLLMClient("target")
    top_k = int(config["model"].get("top_k_channels", 3))

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(rows):
        prompt_sample, proposal = _truth_interval_model_hints(sample, encoder, top_k)
        model_input = convert_to_model_input(prompt_sample)
        axis_embedding_mode = (
            args.hint_delivery == "embedding"
            and not args.mock
            and _uses_axis_embedding_hints(real_client, encoder, proposal, llm_config)
        )
        hint_bundle = (
            _make_axis_embedding_hint_bundle(encoder, sample, proposal, llm_config)
            if axis_embedding_mode
            else None
        )
        if axis_embedding_mode:
            interval_context = {
                "interval_source": "target_interval",
                "instruction": "Analyze this target interval using observed window evidence and AXIS embedding hints.",
                "hint_delivery": "AXIS embedding hints are injected at the input-embedding level.",
            }
        else:
            interval_context = {
                "interval_source": proposal["source"],
                "proposal_score": proposal.get("proposal_score"),
                "hint_anomaly_probability": proposal.get("hint_anomaly_probability"),
                "encoder_max_channel_probability": proposal.get("encoder_max_channel_probability"),
                "predicted_root_cause_channel": proposal.get("predicted_root_cause_channel"),
                "instruction": "Analyze this truth interval with model-derived hints and evidence card.",
                "hint_delivery": "soft textual hints",
            }
        prompt = build_prompt(
            model_input,
            proposal["soft_hints"],
            interval_context=interval_context,
            include_soft_hints=not axis_embedding_mode,
            include_evidence_card=True,
            include_normal_counterpart=False,
            embedding_hint_trace=hint_bundle["trace"] if hint_bundle else None,
        )
        if args.mock:
            response = mock_client.complete_with_target(sample["target_output"])
        elif axis_embedding_mode:
            response = real_client.complete_with_axis_hints(_messages(prompt), hint_bundle["embeddings"])
        else:
            response = real_client.complete(_messages(prompt))
        parsed = _parse_or_empty(response.content)
        comparison = score_explanation_against_truth(sample, proposal, parsed)
        records.append(
            {
                "index": idx,
                "sample_id": sample["sample_id"],
                "proposal": proposal,
                "comparison": comparison,
                "prompt": prompt,
                "raw_response": response.content,
                "parsed_response": parsed,
                "target_output": sample["target_output"],
                "latency_seconds": response.latency_seconds,
                "axis_embedding_hints_used": bool(axis_embedding_mode),
                "embedding_hint_trace": hint_bundle["trace"] if hint_bundle else None,
                "llm_raw_metadata": response.raw if args.mock else {"model": llm_config.get("model"), **response.raw},
            }
        )

    output_path = ROOT / args.output
    write_jsonl(records, output_path)
    report = _aggregate(records)
    report.update(
        {
            "output_path": str(output_path),
            "config": args.config,
            "split": args.split,
            "limit": args.limit,
            "interval_mode": "truth_interval_with_model_derived_hints",
            "hint_delivery": args.hint_delivery,
            "checkpoint": str(checkpoint_path),
            "mock": bool(args.mock),
            "question_provider": args.question_provider,
            "question_seed": int(args.question_seed),
            "overwrite_questions": bool(args.overwrite_questions),
            "llm_config": {
                "provider": llm_config.get("provider"),
                "base_url": llm_config.get("base_url"),
                "model": llm_config.get("model"),
            },
        }
    )
    save_json(report, ROOT / args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    raw_examples = _raw_examples(records, int(args.raw_log_examples))
    if raw_examples:
        raw_path = Path(args.raw_log_path) if args.raw_log_path else output_path.with_name(output_path.stem + "_raw_examples.json")
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        save_json({"examples": raw_examples}, raw_path)
        print(json.dumps({"raw_log_path": str(raw_path)}, ensure_ascii=False, indent=2))

    prompt_examples = _prompt_examples(records, int(args.dump_prompt_examples))
    if prompt_examples:
        prompt_path = Path(args.prompt_log_path) if args.prompt_log_path else output_path.with_name(output_path.stem + "_prompt_examples.json")
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        save_json({"examples": prompt_examples}, prompt_path)
        first_txt = prompt_path.with_suffix(".txt")
        first_txt.write_text(prompt_examples[0]["prompt"], encoding="utf-8")
        print(json.dumps({"prompt_log_path": str(prompt_path), "first_prompt_txt": str(first_txt)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
