from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.answer_schema import EXACT_LABEL_ANSWER_FORMAT
from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.llm_client import MockLLMClient, create_llm_client
from src.mvaxis.prompts import build_prompt, parse_json_output
from src.mvaxis.proposal import propose_interval, sample_with_interval, score_explanation_against_truth, true_anomaly_interval
from src.mvaxis.question_provider import ensure_questions
from src.mvaxis.training import load_encoder, load_hint_tuner
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


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


def _apply_prompt_variant(prompt: str, variant: str) -> str:
    if variant == "current":
        return prompt
    head = prompt.split("### Answer Format", 1)[0].rstrip()
    if variant == "compact_label_json":
        return head + "\n\n### Answer Format\n" + EXACT_LABEL_ANSWER_FORMAT
    raise ValueError(f"Unsupported prompt variant: {variant}")


def _system_prompt_for_variant(variant: str) -> str:
    return COMPACT_SYSTEM_PROMPT if variant == "compact_label_json" else SYSTEM_PROMPT


def _parse_or_empty(text: str) -> Dict[str, Any]:
    try:
        return parse_json_output(text)
    except Exception:
        return {}


def _filter_hints(hints: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if not hints:
        return {}
    filtered = dict(hints)
    if args.hide_global_hints:
        filtered.pop("global_hint", None)
    if args.hide_channel_hints:
        filtered.pop("channel_hints", None)
    if args.hide_local_hints:
        filtered.pop("local_hints", None)
    if args.hide_interval_proposal_hint:
        filtered.pop("interval_proposal", None)
    return filtered


def _proposal_soft_hints(proposal: Dict[str, Any]) -> Dict[str, Any]:
    if proposal.get("soft_hints"):
        return proposal["soft_hints"]
    return {
        "interval_proposal": {
            "source": proposal.get("source"),
            "start": proposal.get("start"),
            "end": proposal.get("end"),
            "score": proposal.get("proposal_score"),
            "threshold": proposal.get("threshold"),
            "is_anomalous_proposal": proposal.get("is_anomalous_proposal"),
        },
        "global_hint": proposal.get("global_hint")
        or {
            "anomaly_probability": proposal.get("proposal_score"),
            "predicted_interval": [proposal.get("start"), proposal.get("end")]
            if proposal.get("start") is not None
            else None,
            "predicted_root_cause_channel": proposal.get("predicted_root_cause_channel"),
            "is_anomalous_proposal": proposal.get("is_anomalous_proposal"),
            "hint_source": proposal.get("source"),
        },
        "channel_hints": proposal.get("channel_hints") or [],
        "local_hints": proposal.get("local_hints") or [],
        "usage_policy": "Hints are model outputs, not ground truth. The LLM must verify them against evidence_card.",
    }


def _complete_with_optional_axis_hints(
    client: Any,
    messages: List[Dict[str, str]],
    encoder: Any,
    sample: Dict[str, Any],
    proposal: Dict[str, Any],
    llm_config: Dict[str, Any],
    use_axis_hints: bool = True,
    hint_bundle: Dict[str, Any] | None = None,
):
    axis_hints_enabled = use_axis_hints and bool((llm_config.get("axis_hints") or {}).get("enabled", False))
    if (
        axis_hints_enabled
        and proposal.get("start") is not None
        and hasattr(client, "complete_with_axis_hints")
        and hasattr(encoder, "make_embedding_hints")
    ):
        if hint_bundle is None:
            hint_bundle = _make_axis_embedding_hint_bundle(encoder, sample, proposal, llm_config)
        return client.complete_with_axis_hints(messages, hint_bundle["embeddings"])
    return client.complete(messages)


def _make_axis_embedding_hint_bundle(
    encoder: Any,
    sample: Dict[str, Any],
    proposal: Dict[str, Any],
    llm_config: Dict[str, Any],
    *,
    include_global: bool = True,
    include_channel: bool = True,
) -> Dict[str, Any]:
    if proposal.get("start") is None or not hasattr(encoder, "make_embedding_hints"):
        raise ValueError("AXIS embedding hints require a proposal interval and encoder.make_embedding_hints().")
    axis_cfg = llm_config.get("axis_hints") or {}
    max_global = axis_cfg.get("max_global_tokens", axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens")))
    max_channel = axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens"))
    bundle = encoder.make_embedding_hints(
        sample,
        int(proposal["start"]),
        int(proposal["end"]),
        top_k_channels=axis_cfg.get("top_k_channels"),
        max_channel_tokens=max_channel,
        max_global_tokens=max_global,
    )
    embeddings = dict(bundle.get("embeddings") or {})
    trace = dict(bundle.get("trace") or {})
    if not include_global:
        embeddings.pop("global", None)
        trace["global"] = {
            "source_interval": [int(proposal["start"]), int(proposal["end"])],
            "sampled_time_indices": [],
            "num_global_tokens": 0,
        }
    if not include_channel:
        embeddings.pop("channel", None)
        trace["channel"] = {
            "source_interval": [int(proposal["start"]), int(proposal["end"])],
            "num_channel_tokens": 0,
            "sampled_positions": [],
        }
    if not embeddings:
        raise ValueError("At least one embedding hint family must be enabled.")
    bundle["embeddings"] = embeddings
    bundle["trace"] = trace
    return bundle


def _uses_axis_embedding_hints(client: Any, encoder: Any, proposal: Dict[str, Any], llm_config: Dict[str, Any]) -> bool:
    return (
        bool((llm_config.get("axis_hints") or {}).get("enabled", False))
        and proposal.get("start") is not None
        and hasattr(client, "complete_with_axis_hints")
        and hasattr(encoder, "make_embedding_hints")
    )


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [r["comparison"] for r in records]
    def mean(key: str) -> float:
        return sum(float(x[key]) for x in scores) / max(1, len(scores))
    def optional_mean(key: str) -> float:
        values = [x.get(key) for x in scores if x.get(key) is not None]
        return sum(float(x) for x in values) / max(1, len(values))
    def optional_count(key: str) -> int:
        return sum(1 for x in scores if x.get(key) is not None)
    return {
        "num_samples": len(records),
        "json_parse_rate": sum(float(isinstance((r["parsed_response"] or {}).get("fact_check"), dict)) for r in records) / max(1, len(records)),
        "proposal_mean_temporal_iou": mean("proposal_temporal_iou"),
        "proposal_mean_truth_coverage": mean("proposal_truth_coverage"),
        "proposal_close_rate": mean("proposal_is_close"),
        "llm_anomaly_accuracy": mean("llm_anomaly_match"),
        "llm_root_accuracy": mean("llm_root_match"),
        "llm_mean_affected_f1": mean("llm_affected_f1"),
        "llm_type_accuracy": mean("llm_type_match"),
        "llm_answer_label_accuracy": optional_mean("llm_answer_label_match"),
        "llm_answer_label_count": optional_count("llm_answer_label_match"),
    }


def _raw_examples(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    selected: List[Dict[str, Any]] = []
    seen = set()
    priority = [
        lambda r: bool(r.get("comparison", {}).get("llm_anomaly_match") is False),
        lambda r: bool((r.get("target_output", {}).get("fact_check") or {}).get("is_anomalous")),
        lambda r: True,
    ]
    for pred in priority:
        for idx, record in enumerate(records):
            if idx in seen or not pred(record):
                continue
            selected.append(record)
            seen.add(idx)
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    examples = []
    for record in selected:
        target_fact = (record.get("target_output") or {}).get("fact_check") or {}
        parsed_fact = (record.get("parsed_response") or {}).get("fact_check") or {}
        examples.append(
            {
                "sample_id": record.get("sample_id"),
                "proposal": record.get("proposal"),
                "comparison": record.get("comparison"),
                "raw_response": record.get("raw_response"),
                "parsed_fact_check": parsed_fact,
                "target_fact_check": target_fact,
            }
        )
    return examples


def _prompt_examples(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    examples = []
    for record in records[: max(0, limit)]:
        examples.append(
            {
                "sample_id": record.get("sample_id"),
                "prompt": record.get("prompt"),
                "target_fact_check": (record.get("target_output") or {}).get("fact_check") or {},
                "proposal": record.get("proposal"),
            }
        )
    return examples


def _truth_interval_proposal(sample: Dict[str, Any]) -> Dict[str, Any]:
    truth = true_anomaly_interval(sample)
    if truth is None:
        start = int(sample["target_interval"]["start"])
        end = int(sample["target_interval"]["end"])
        return {
            "start": start,
            "end": end,
            "proposal_score": None,
            "max_point_score": None,
            "predicted_root_cause_channel": None,
            "channel_hints": [],
            "is_anomalous_proposal": False,
            "source": "oracle_target_interval_no_anomaly",
            "ranked_candidates": [],
        }
    target = sample["target_output"]["fact_check"]
    return {
        "start": int(truth[0]),
        "end": int(truth[1]),
        "proposal_score": 1.0,
        "max_point_score": 1.0,
        "predicted_root_cause_channel": target.get("root_cause_channel"),
        "channel_hints": [
            {"channel_id": ch, "score": 1.0, "encoder_prob": 1.0}
            for ch in target.get("affected_channels", [])[:3]
        ],
        "is_anomalous_proposal": True,
        "source": "oracle_truth_interval_upper_bound",
        "ranked_candidates": [],
    }


def _load_interval_proposer_checkpoint(path: str, config: Dict[str, Any]) -> tuple[AXISMultivariateIntervalProposer, Dict[str, Any]]:
    resolved = relative_to_root(ROOT, path)
    try:
        model = AXISMultivariateIntervalProposer.load(resolved)
        return model, {"checkpoint_path": resolved, "checkpoint_format": "mvaxis"}
    except Exception as exc:
        model, info = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
            resolved,
            config,
            threshold=float((config.get("interval_proposal") or {}).get("threshold", 0.5)),
        )
        info = dict(info)
        info["checkpoint_format"] = "timercd_huggingface"
        info["mvaxis_load_error"] = str(exc).splitlines()[0]
        return model, info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_smoke.json")
    parser.add_argument("--llm-config", default="configs/deepseek_llm.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--interval-source", choices=["proposal", "truth"], default="proposal")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--output", default="outputs/llm_proposals.jsonl")
    parser.add_argument("--report", default="outputs/llm_proposals_report.json")
    parser.add_argument("--raw-log-examples", type=int, default=2)
    parser.add_argument("--raw-log-path", default=None)
    parser.add_argument("--dump-prompt-examples", type=int, default=0)
    parser.add_argument("--prompt-log-path", default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--progress-report-output", default=None)
    parser.add_argument("--partial-output", default=None)
    parser.add_argument("--prompt-variant", choices=["current", "compact_label_json"], default="current")
    parser.add_argument("--disable-axis-embedding-hints", action="store_true")
    parser.add_argument("--force-soft-hints", action="store_true")
    parser.add_argument("--hide-global-hints", action="store_true")
    parser.add_argument("--hide-channel-hints", action="store_true")
    parser.add_argument("--hide-local-hints", action="store_true")
    parser.add_argument("--hide-interval-proposal-hint", action="store_true")
    parser.add_argument("--hide-evidence-card", action="store_true")
    parser.add_argument("--question-provider", default="bank", choices=["bank", "none"])
    parser.add_argument("--question-seed", type=int, default=520)
    parser.add_argument("--overwrite-questions", action="store_true")
    parser.add_argument(
        "--include-normal-counterpart",
        action="store_true",
        help="Oracle ablation only: expose the synthetic normal counterpart window to the LLM.",
    )
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    config["train_encoder"]["checkpoint_path"] = relative_to_root(ROOT, config["train_encoder"]["checkpoint_path"])
    config["train_hint_tuner"]["checkpoint_path"] = relative_to_root(ROOT, config["train_hint_tuner"]["checkpoint_path"])
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
    checkpoint_load_info: Dict[str, Any] = {}
    if args.interval_proposer_checkpoint:
        # Prefer the checkpoint-embedded model config here. The data config may
        # intentionally point at the same JSONL split while using a smaller
        # smoke encoder shape, which cannot load the production AXIS head.
        encoder, checkpoint_load_info = _load_interval_proposer_checkpoint(args.interval_proposer_checkpoint, config)
        if (encoder.config.get("model") or {}).get("top_k_channels") is not None:
            config["model"] = encoder.config["model"]
        hint_tuner = None
    elif "train_interval_proposal" in config:
        proposal_ckpt = relative_to_root(ROOT, config["train_interval_proposal"]["checkpoint_path"])
        if Path(proposal_ckpt).exists():
            encoder, checkpoint_load_info = _load_interval_proposer_checkpoint(proposal_ckpt, config)
            if (encoder.config.get("model") or {}).get("top_k_channels") is not None:
                config["model"] = encoder.config["model"]
            hint_tuner = None
        else:
            encoder = load_encoder(config)
            hint_tuner = load_hint_tuner(config)
    else:
        encoder = load_encoder(config)
        hint_tuner = load_hint_tuner(config)

    llm_config = load_json(ROOT / args.llm_config)
    real_client = None if args.mock else create_llm_client(llm_config)
    mock_client = MockLLMClient("target")
    records: List[Dict[str, Any]] = []
    output_path = ROOT / args.output
    partial_path = None
    if args.partial_output:
        partial_path = Path(args.partial_output)
        if not partial_path.is_absolute():
            partial_path = ROOT / partial_path
    else:
        partial_path = output_path.with_name(output_path.stem + "_partial.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.write_text("", encoding="utf-8")
    if args.progress_report_output:
        progress_report_path = Path(args.progress_report_output)
        if not progress_report_path.is_absolute():
            progress_report_path = ROOT / progress_report_path
    else:
        progress_report_path = output_path.with_name(output_path.stem + "_progress.jsonl")
    progress_report_path.parent.mkdir(parents=True, exist_ok=True)
    progress_report_path.write_text("", encoding="utf-8")
    for idx, sample in enumerate(rows):
        if args.interval_source == "truth":
            proposal = _truth_interval_proposal(sample)
        else:
            proposal = propose_interval(
                sample,
                encoder,
                hint_tuner,
                window_size=args.window_size,
                stride=args.stride,
                top_k_channels=int(config["model"]["top_k_channels"]),
            )
        if proposal["start"] is None:
            prompt_sample = sample
        else:
            prompt_sample = sample_with_interval(sample, int(proposal["start"]), int(proposal["end"]))
        mi = convert_to_model_input(prompt_sample)
        if proposal.get("soft_hints") or proposal.get("channel_hints") or proposal.get("global_hint"):
            hints = _proposal_soft_hints(proposal)
        elif hint_tuner is not None:
            hints = hint_tuner.make_soft_hint_summary(encoder, mi, int(config["model"]["top_k_channels"]))
        else:
            hints = {}
        hints = _filter_hints(hints, args)
        axis_embedding_mode = (
            False
            if args.mock or args.disable_axis_embedding_hints
            else _uses_axis_embedding_hints(real_client, encoder, proposal, llm_config)
        )
        hint_bundle = (
            _make_axis_embedding_hint_bundle(
                encoder,
                sample,
                proposal,
                llm_config,
                include_global=not bool(args.hide_global_hints),
                include_channel=not bool(args.hide_channel_hints),
            )
            if axis_embedding_mode
            else None
        )
        if axis_embedding_mode:
            interval_context = {
                "interval_source": "oracle_truth_interval_upper_bound"
                if args.interval_source == "truth"
                else "encoder_or_anomaly_head_sliding_window_proposal",
                "instruction": "Analyze this proposed interval, then decide whether it is truly anomalous.",
                "hint_delivery": "AXIS embedding hints are injected at the input-embedding level.",
            }
        else:
            interval_context = {
                "interval_source": "oracle_truth_interval_upper_bound"
                if args.interval_source == "truth"
                else "encoder_or_anomaly_head_sliding_window_proposal",
                "proposal_score": proposal["proposal_score"],
                "hint_anomaly_probability": proposal.get("hint_anomaly_probability", proposal.get("proposal_score")),
                "encoder_max_channel_probability": proposal.get("encoder_max_channel_probability", proposal.get("max_point_score")),
                "predicted_root_cause_channel": proposal["predicted_root_cause_channel"],
                "instruction": "Analyze this proposed interval, then decide whether it is truly anomalous.",
            }
        prompt = build_prompt(
            mi,
            hints,
            interval_context=interval_context,
            include_soft_hints=bool(args.force_soft_hints or not axis_embedding_mode),
            include_evidence_card=not bool(args.hide_evidence_card),
            include_normal_counterpart=bool(args.include_normal_counterpart),
            embedding_hint_trace=hint_bundle["trace"] if hint_bundle else None,
        )
        prompt = _apply_prompt_variant(prompt, args.prompt_variant)
        if args.mock:
            response = mock_client.complete_with_target(sample["target_output"])
        else:
            response = _complete_with_optional_axis_hints(
                real_client,
                _messages(prompt, _system_prompt_for_variant(args.prompt_variant)),
                encoder,
                sample,
                proposal,
                llm_config,
                use_axis_hints=bool(axis_embedding_mode),
                hint_bundle=hint_bundle,
            )
        parsed = _parse_or_empty(response.content)
        comparison = score_explanation_against_truth(sample, proposal, parsed, min_iou=args.min_iou)
        records.append(
            record := {
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
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if args.progress_every > 0 and ((idx + 1) % int(args.progress_every) == 0 or idx == len(rows) - 1):
            partial_report = _aggregate(records)
            partial_report.update(
                {
                    "progress": idx + 1,
                    "total": len(rows),
                    "sample_id": sample["sample_id"],
                    "partial_output": str(partial_path),
                }
            )
            with progress_report_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(partial_report, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    partial_report,
                    ensure_ascii=False,
                ),
                flush=True,
            )
    write_jsonl(records, output_path)
    report = _aggregate(records)
    report["output_path"] = str(output_path)
    report["mock"] = bool(args.mock)
    report["window_size"] = args.window_size
    report["stride"] = args.stride
    report["min_iou"] = args.min_iou
    report["interval_source"] = args.interval_source
    report["interval_proposer_checkpoint"] = args.interval_proposer_checkpoint or config.get("train_interval_proposal", {}).get("checkpoint_path")
    report["interval_proposer_checkpoint_load_info"] = checkpoint_load_info
    report["prompt_variant"] = args.prompt_variant
    report["question_provider"] = args.question_provider
    report["question_seed"] = int(args.question_seed)
    report["overwrite_questions"] = bool(args.overwrite_questions)
    report["llm_config"] = {
        "provider": llm_config.get("provider"),
        "base_url": llm_config.get("base_url"),
        "model": llm_config.get("model"),
    }
    save_json(report, ROOT / args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    examples = _raw_examples(records, int(args.raw_log_examples))
    if examples:
        raw_path = Path(args.raw_log_path) if args.raw_log_path else output_path.with_name(output_path.stem + "_raw_examples.json")
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        save_json({"examples": examples}, raw_path)
        print("== raw_response_examples ==")
        print(json.dumps({"raw_log_path": str(raw_path), "examples": examples}, ensure_ascii=False, indent=2))
    prompt_examples = _prompt_examples(records, int(args.dump_prompt_examples))
    if prompt_examples:
        prompt_path = Path(args.prompt_log_path) if args.prompt_log_path else output_path.with_name(output_path.stem + "_prompt_examples.json")
        if not prompt_path.is_absolute():
            prompt_path = ROOT / prompt_path
        save_json({"examples": prompt_examples}, prompt_path)
        first_txt = prompt_path.with_suffix(".txt")
        first_txt.write_text(prompt_examples[0]["prompt"], encoding="utf-8")
        print("== prompt_examples ==")
        print(json.dumps({"prompt_log_path": str(prompt_path), "first_prompt_txt": str(first_txt)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
