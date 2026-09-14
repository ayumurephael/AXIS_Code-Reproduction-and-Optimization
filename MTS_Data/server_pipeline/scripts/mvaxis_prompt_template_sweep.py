from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from run_llm_on_proposals import (
    _aggregate,
    _complete_with_optional_axis_hints,
    _filter_hints,
    _make_axis_embedding_hint_bundle,
    _parse_or_empty,
    _proposal_soft_hints,
    _uses_axis_embedding_hints,
)
from mvaxis_run_truth_interval_with_model_hints import _truth_interval_model_hints
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.llm_client import MockLLMClient, create_llm_client
from src.mvaxis.prompts import build_prompt
from src.mvaxis.proposal import propose_interval, sample_with_interval, score_explanation_against_truth
from src.mvaxis.question_provider import ensure_questions
from src.mvaxis.training import load_encoder, load_hint_tuner
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


SYSTEM_PROMPTS = {
    "current": (
        "Return exactly one valid json object following the requested schema. "
        "The first generated character must be { and the last generated character must be }. "
        "Do not include markdown fences, bullet lists, prose before json, or prose after json. "
        "The target interval was proposed by an anomaly head; verify it against evidence and do not assume it is anomalous."
    ),
    "strict_json_minimal": (
        "You are a strict JSON API. Output one complete JSON object only. "
        "No markdown, no analysis outside JSON, no refusal text, no schema copying. "
        "If evidence is insufficient, put null or [] in fact_check and explain uncertainty inside JSON string fields."
    ),
    "question_first_json": (
        "Output exactly one valid JSON object. The field question_answer must directly answer the user's current question type. "
        "Do not refuse; use uncertainty fields inside JSON when needed. Do not add text before or after the JSON object."
    ),
    "relaxed_json": (
        "You are a helpful multivariate time-series analyst. Return a valid JSON object."
    ),
    "relaxed_question_json": (
        "You are a helpful multivariate time-series analyst. Return valid JSON, and make question_answer directly answer the user's question."
    ),
    "compact_json": (
        "You are a concise JSON API for multivariate time-series QA. Return one valid compact JSON object only."
    ),
}


def _messages(system_prompt: str, prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _apply_variant(prompt: str, variant: str) -> str:
    if variant == "current":
        return prompt
    if variant == "strict_json_minimal":
        return (
            prompt
            + "\n\n### JSON Stability Override\n"
            "Return a complete JSON object only. If a field is unknown, use null, [], or a short uncertainty sentence inside the requested field. "
            "Never output ellipses, apologies, refusals, markdown fences, or schema placeholders."
        )
    if variant == "question_first_json":
        return (
            prompt
            + "\n\n### Question Matching Override\n"
            "Before finalizing, check that question_answer and final_answer answer the exact Question above. "
            "Keep fact_check structured and place all reasoning inside reasoning_summary/evidence_chain."
        )
    if variant == "relaxed_json":
        return (
            prompt
            + "\n\n### Practical Output Note\n"
            "Return JSON with the requested keys. If uncertain, write uncertainty inside the JSON fields rather than refusing."
        )
    if variant == "relaxed_question_json":
        return (
            prompt
            + "\n\n### Practical Output Note\n"
            "Return JSON with the requested keys. Make question_answer and final_answer answer the specific question and choices above."
        )
    if variant == "compact_json":
        head = prompt.split("### Answer Format", 1)[0].rstrip()
        return (
            head
            + "\n\n### Answer Format\n"
            "Return exactly one compact JSON object and no markdown. Keep every string short.\n"
            "Required top-level keys: fact_check, evidence_chain, reasoning_summary, question_answer, final_answer, abnormality_score, answer_confidence.\n"
            "fact_check keys: is_anomalous, root_cause_channel, root_cause_channels, affected_channels, anomaly_type, anomaly_scope, abnormal_edges, causal_path.\n"
            "Use null or [] when unsupported. evidence_chain must contain 1 or 2 short complete sentences. "
            "reasoning_summary must be one short sentence. question_answer must answer the exact question or choice. "
            "abnormality_score and answer_confidence must be numbers from 0 to 1."
        )
    raise ValueError(f"Unknown prompt variant: {variant}")


def _load_proposer(config: Dict[str, Any], checkpoint: str | None):
    if checkpoint:
        return AXISMultivariateIntervalProposer.load(relative_to_root(ROOT, checkpoint), config), None
    proposal_cfg = config.get("train_interval_proposal") or {}
    if proposal_cfg.get("checkpoint_path") and Path(relative_to_root(ROOT, proposal_cfg["checkpoint_path"])).exists():
        return AXISMultivariateIntervalProposer.load(relative_to_root(ROOT, proposal_cfg["checkpoint_path"]), config), None
    return load_encoder(config), load_hint_tuner(config)


def _run_variant(
    variant: str,
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    llm_config: Dict[str, Any],
    client: Any,
    mock_client: MockLLMClient,
    encoder: Any,
    hint_tuner: Any,
    args: argparse.Namespace,
    stream_path: Path | None = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if stream_path is not None:
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_path.write_text("", encoding="utf-8")
    for idx, sample in enumerate(rows):
        if args.interval_source == "truth_model_hints":
            prompt_sample, proposal = _truth_interval_model_hints(
                sample,
                encoder,
                int(config["model"]["top_k_channels"]),
            )
        else:
            proposal = propose_interval(
                sample,
                encoder,
                hint_tuner,
                window_size=int(args.window_size),
                stride=int(args.stride),
                top_k_channels=int(config["model"]["top_k_channels"]),
            )
            prompt_sample = (
                sample
                if proposal["start"] is None
                else sample_with_interval(sample, int(proposal["start"]), int(proposal["end"]))
            )
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
            else _uses_axis_embedding_hints(client, encoder, proposal, llm_config)
        )
        hint_bundle = (
            _make_axis_embedding_hint_bundle(encoder, sample, proposal, llm_config)
            if axis_embedding_mode
            else None
        )
        interval_source = (
            "oracle_truth_interval_model_hints"
            if args.interval_source == "truth_model_hints"
            else "encoder_or_anomaly_head_sliding_window_proposal"
        )
        if axis_embedding_mode:
            # Clean AXIS-style evaluation: do not expose detector decisions,
            # root/channel predictions, or anomaly scores as readable text.
            # The hint information is delivered only through hidden input
            # embeddings injected at the special hint token positions.
            interval_context = {
                "interval_source": "target_interval",
                "instruction": "Analyze this target interval using observed window evidence and AXIS embedding hints.",
                "hint_delivery": "AXIS embedding hints are injected at the input-embedding level.",
            }
        else:
            interval_context = {
                "interval_source": interval_source,
                "proposal_score": proposal.get("proposal_score"),
                "hint_anomaly_probability": proposal.get("hint_anomaly_probability", proposal.get("proposal_score")),
                "encoder_max_channel_probability": proposal.get("encoder_max_channel_probability", proposal.get("max_point_score")),
                "predicted_root_cause_channel": proposal.get("predicted_root_cause_channel"),
                "instruction": "Analyze this proposed interval, then decide whether it is truly anomalous.",
                "hint_delivery": "textual soft hints",
            }
        prompt = build_prompt(
            mi,
            hints,
            interval_context=interval_context,
            include_soft_hints=bool(args.force_soft_hints or not axis_embedding_mode),
            include_evidence_card=not bool(args.hide_evidence_card),
            include_normal_counterpart=False,
            embedding_hint_trace=hint_bundle["trace"] if hint_bundle else None,
        )
        prompt = _apply_variant(prompt, variant)
        if args.mock:
            response = mock_client.complete_with_target(sample["target_output"])
        else:
            response = _complete_with_optional_axis_hints(
                client,
                _messages(SYSTEM_PROMPTS[variant], prompt),
                encoder,
                sample,
                proposal,
                llm_config,
                use_axis_hints=bool(axis_embedding_mode),
                hint_bundle=hint_bundle,
            )
        parsed = _parse_or_empty(response.content)
        comparison = score_explanation_against_truth(sample, proposal, parsed, min_iou=float(args.min_iou))
        record = {
            "index": idx,
            "sample_id": sample["sample_id"],
            "prompt_variant": variant,
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
        records.append(record)
        if stream_path is not None:
            with stream_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale_nocf_schema.json")
    parser.add_argument("--llm-config", default="configs/local_qwen25_7b_cuda_remote_semantic_truth_bridge.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--interval-source", choices=["proposal", "truth_model_hints"], default="proposal")
    parser.add_argument("--prompt-variants", default="current,compact_json,relaxed_json,relaxed_question_json,strict_json_minimal,question_first_json")
    parser.add_argument("--output-dir", default="outputs/runs/prompt_template_sweep_v1")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--raw-log-examples", type=int, default=3)
    parser.add_argument("--dump-prompt-examples", type=int, default=1)
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
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    rows = read_jsonl(Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{args.split}.jsonl")
    rows = rows[: int(args.limit)]
    rows = ensure_questions(
        rows,
        seed=int(args.question_seed),
        provider=args.question_provider,
        overwrite=bool(args.overwrite_questions),
        overwrite_fixed_placeholder=True,
        copy_samples=False,
    )
    encoder, hint_tuner = _load_proposer(config, args.interval_proposer_checkpoint)
    llm_config = load_json(ROOT / args.llm_config)
    client = None if args.mock else create_llm_client(llm_config)
    mock_client = MockLLMClient("target")
    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "variants": {},
        "num_samples": len(rows),
        "question_provider": args.question_provider,
        "question_seed": int(args.question_seed),
        "overwrite_questions": bool(args.overwrite_questions),
    }
    for variant in [v.strip() for v in args.prompt_variants.split(",") if v.strip()]:
        records_path = out_dir / f"{variant}.jsonl"
        records = _run_variant(
            variant,
            rows,
            config,
            llm_config,
            client,
            mock_client,
            encoder,
            hint_tuner,
            args,
            stream_path=records_path,
        )
        report = _aggregate(records)
        report["prompt_variant"] = variant
        report["records_path"] = str(records_path)
        save_json(report, out_dir / f"{variant}_report.json")
        save_json(
            {"examples": records[: max(0, int(args.raw_log_examples))]},
            out_dir / f"{variant}_raw_examples.json",
        )
        if args.dump_prompt_examples > 0 and records:
            save_json(
                {"examples": [{"sample_id": r["sample_id"], "prompt": r["prompt"]} for r in records[: args.dump_prompt_examples]]},
                out_dir / f"{variant}_prompt_examples.json",
            )
            (out_dir / f"{variant}_first_prompt.txt").write_text(records[0]["prompt"], encoding="utf-8")
        summary["variants"][variant] = report
        print(json.dumps(report, ensure_ascii=False, indent=2))
    save_json(summary, out_dir / "summary.json")
    print(json.dumps({"summary_path": str(out_dir / "summary.json"), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
