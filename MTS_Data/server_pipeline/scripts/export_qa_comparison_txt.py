from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.proposal import true_anomaly_interval
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _fmt_list(values: Iterable[Any] | None) -> str:
    if isinstance(values, str):
        return values
    items = list(values or [])
    return ", ".join(str(x) for x in items) if items else "none"


def _as_list(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, list):
        return values
    if isinstance(values, str):
        return [values]
    return list(values) if isinstance(values, tuple) else [values]


def _sample_lookup(config: Dict[str, Any], split: str) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{split}.jsonl")
    return {str(row["sample_id"]): row for row in rows}


def _question_from_prompt(prompt: str) -> str:
    marker = "### Question\n"
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    return tail.split("\n\n### Answer Format", 1)[0].strip()


def _write_record(lines: List[str], idx: int, record: Dict[str, Any], sample: Dict[str, Any] | None) -> None:
    proposal = record.get("proposal") or {}
    comparison = record.get("comparison") or {}
    target = record.get("target_output") or {}
    truth_fact = target.get("fact_check") or {}
    parsed = record.get("parsed_response") or {}
    pred_fact = parsed.get("fact_check") or {}
    synthetic = (sample or {}).get("synthetic_label") or {}
    causal_graph = (sample or {}).get("causal_graph") or {}
    teacher_reasoning = target.get("reasoning_summary", "")
    teacher_evidence_chain = _as_list(target.get("evidence_chain"))
    teacher_question_answer = target.get("question_answer", "")
    teacher_answer = target.get("final_answer", "")
    question = (sample or {}).get("question") or _question_from_prompt(record.get("prompt", ""))
    truth_interval = true_anomaly_interval(sample) if sample else None

    lines.append("=" * 96)
    lines.append(f"[{idx:02d}] sample_id: {record.get('sample_id')}")
    lines.append(f"question_type: {(sample or {}).get('question_type', 'unknown')}")
    lines.append(f"question: {question}")
    lines.append("")
    lines.append("[DoFlow / Label]")
    lines.append(
        "graph: "
        + f"{causal_graph.get('graph_type', synthetic.get('graph_type'))}; "
        + f"mechanism={causal_graph.get('mechanism', synthetic.get('mechanism'))}; "
        + f"active_nodes={causal_graph.get('active_nodes', 'unknown')}"
    )
    lines.append(
        "truth: "
        + f"is_anomalous={truth_fact.get('is_anomalous')}; "
        + f"interval={list(truth_interval) if truth_interval else None}; "
        + f"root={truth_fact.get('root_cause_channel')}; "
        + f"root_set={_fmt_list(truth_fact.get('root_cause_channels'))}; "
        + f"affected={_fmt_list(truth_fact.get('affected_channels'))}; "
        + f"type={truth_fact.get('anomaly_type')}; "
        + f"scope={truth_fact.get('anomaly_scope')}; "
        + f"edges={_compact_json(truth_fact.get('abnormal_edges') or [])}; "
        + f"path={_fmt_list(truth_fact.get('causal_path'))}"
    )
    lines.append("")
    lines.append("[Teacher Standard Explanation]")
    if teacher_evidence_chain:
        lines.append("evidence_chain:")
        for step_idx, step in enumerate(teacher_evidence_chain, start=1):
            lines.append(f"  {step_idx}. {step}")
    lines.append(f"reasoning_summary: {teacher_reasoning}")
    if teacher_question_answer:
        lines.append(f"question_answer: {teacher_question_answer}")
    lines.append(f"final_answer: {teacher_answer}")
    lines.append("")
    lines.append("[Anomaly Head / Proposal]")
    lines.append(
        f"proposal_interval=[{proposal.get('start')}, {proposal.get('end')}]; "
        + f"score={proposal.get('proposal_score')}; "
        + f"max_point_score={proposal.get('max_point_score')}; "
        + f"predicted_root={proposal.get('predicted_root_cause_channel')}; "
        + f"is_anomalous_proposal={proposal.get('is_anomalous_proposal')}; "
        + f"source={proposal.get('source')}"
    )
    lines.append(f"channel_hints: {_compact_json((proposal.get('channel_hints') or [])[:5])}")
    if proposal.get("ranked_candidates"):
        ranked = [
            {
                "start": item.get("start"),
                "end": item.get("end"),
                "score": item.get("proposal_score"),
                "root": item.get("predicted_root_cause_channel"),
            }
            for item in proposal.get("ranked_candidates", [])[:5]
        ]
        lines.append(f"ranked_candidates: {_compact_json(ranked)}")
    lines.append(
        "proposal_metrics: "
        + f"IoU={comparison.get('proposal_temporal_iou')}; "
        + f"truth_coverage={comparison.get('proposal_truth_coverage')}; "
        + f"close={comparison.get('proposal_is_close')}"
    )
    lines.append("")
    lines.append("[Qwen Parsed Answer]")
    evidence_chain = _as_list(parsed.get("evidence_chain"))
    lines.append(
        "parsed_fact: "
        + f"is_anomalous={pred_fact.get('is_anomalous')}; "
        + f"root={pred_fact.get('root_cause_channel')}; "
        + f"root_set={_fmt_list(pred_fact.get('root_cause_channels'))}; "
        + f"affected={_fmt_list(pred_fact.get('affected_channels'))}; "
        + f"type={pred_fact.get('anomaly_type')}; "
        + f"scope={pred_fact.get('anomaly_scope')}; "
        + f"edges={_compact_json(pred_fact.get('abnormal_edges') or [])}; "
        + f"path={_fmt_list(pred_fact.get('causal_path'))}"
    )
    if evidence_chain:
        lines.append("parsed_evidence_chain:")
        for step_idx, step in enumerate(evidence_chain, start=1):
            lines.append(f"  {step_idx}. {step}")
    lines.append(f"parsed_reasoning: {parsed.get('reasoning_summary')}")
    lines.append(f"parsed_question_answer: {parsed.get('question_answer')}")
    lines.append(f"parsed_final_answer: {parsed.get('final_answer')}")
    lines.append(
        "quality_flags: "
        + f"json_parsed={bool(pred_fact)}; "
        + f"anomaly_match={comparison.get('llm_anomaly_match')}; "
        + f"root_match={comparison.get('llm_root_match')}; "
        + f"affected_f1={comparison.get('llm_affected_f1')}; "
        + f"type_match={comparison.get('llm_type_match')}; "
        + f"axis_embedding_hints_used={record.get('axis_embedding_hints_used')}"
    )
    lines.append("")
    lines.append("[Qwen Raw Answer]")
    lines.append(str(record.get("raw_response", "")).strip())
    lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    samples = _sample_lookup(config, args.split)
    records = read_jsonl(ROOT / args.records)
    if args.limit is not None:
        records = records[: args.limit]

    lines: List[str] = []
    lines.append("AXIS multivariate QA explanation comparison")
    lines.append(f"config: {args.config}")
    lines.append(f"split: {args.split}")
    lines.append(f"records: {args.records}")
    lines.append(f"num_records: {len(records)}")
    lines.append("")
    for idx, record in enumerate(records, start=1):
        sample = samples.get(str(record.get("sample_id")))
        _write_record(lines, idx, record, sample)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "num_records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
