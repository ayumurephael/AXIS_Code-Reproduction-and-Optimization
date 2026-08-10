from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.models.MultiAXIS.response_contracts import (
    open_response_parseable,
    parse_response_label,
    response_contract_error,
)


DATASETS = ("478new", "SMD", "SWaT", "LEMMA-RCA", "VTA")
CHOICE_PATTERNS = [
    re.compile(r"(?im)^\s*answer\s*:\s*\(?\s*([A-F])\s*\)?\b"),
    re.compile(r"(?i)\b(?:answer|final answer|option|choice|selected option|correct option)\s*(?:is|:)?\s*\(?\s*([A-F])\s*\)?\b"),
    re.compile(r"(?i)\bI\s*(?:would\s*)?(?:choose|select|pick)\s*\(?\s*([A-F])\s*\)?\b"),
    re.compile(r"^\s*\(?\s*([A-F])\s*\)?\s*(?:[\.)]|$)", re.I),
]
JUDGMENT_PATTERNS = [
    re.compile(r"(?im)^\s*answer\s*:\s*(?:is\s*)?(yes|no|true|false)\b"),
    re.compile(r"(?i)\b(?:answer|final answer|judgment|label)\s*[:\-]?\s*(?:is\s*)?(yes|no|true|false)\b"),
    re.compile(r"^\s*(yes|no|true|false)\b", re.I),
]


def canonical_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("yes") or lowered.startswith("true"):
        return "yes"
    if lowered.startswith("no") or lowered.startswith("false"):
        return "no"
    if len(text) == 1 and text.upper() in "ABCDEF":
        return text.upper()
    return None


def parse_prediction(text: str, question_group: str) -> Optional[str]:
    return parse_response_label(text, question_group)


def target_label(reference: Dict[str, Any], question_group: str) -> Optional[str]:
    if question_group == "MC":
        output = reference.get("target_output") or {}
        fact = output.get("fact_check") or {}
        return canonical_label(fact.get("choice_answer") or fact.get("answer_label"))
    return canonical_label(reference.get("teacher_short_answer"))


def open_parseable(text: str) -> bool:
    return open_response_parseable(text)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def summarize(rows: Sequence[Dict[str, Any]]):
    def bucket(items):
        counts = collections.Counter(item["question_group"] for item in items)
        mc = [item for item in items if item["question_group"] == "MC"]
        tf = [item for item in items if item["question_group"] == "TF"]
        oe = [item for item in items if item["question_group"] == "OE"]
        def rate(values, field):
            return sum(bool(item[field]) for item in values) / len(values) if values else None
        combined = [item for item in items if item["success"] is not None]
        return {
            "examples": len(items),
            "counts": dict(counts),
            "mc_exact_match_accuracy": rate(mc, "exact_match"),
            "tf_exact_match_accuracy": rate(tf, "exact_match"),
            "oe_parse_rate": rate(oe, "parseable"),
            "combined_task_success_rate": rate(combined, "success"),
            "unparseable_mc_labels": sum(item["predicted_label"] is None for item in mc),
            "unparseable_tf_labels": sum(item["predicted_label"] is None for item in tf),
        }
    return {
        "overall": bucket(rows),
        "by_question_type": {group: bucket([row for row in rows if row["question_group"] == group]) for group in ("MC", "OE", "TF")},
        "by_dataset": {dataset: bucket([row for row in rows if row["dataset"] == dataset]) for dataset in DATASETS},
    }


def markdown(summary: Dict[str, Any]) -> str:
    lines = ["# Label metrics", "", "| Scope | N | MC exact | TF exact | OE parse | Combined |", "|---|---:|---:|---:|---:|---:|"]
    entries = [("Overall", summary["overall"])] + [(dataset, summary["by_dataset"][dataset]) for dataset in DATASETS]
    def fmt(value):
        return "?" if value is None else f"{100 * value:.2f}%"
    for name, item in entries:
        lines.append(f"| {name} | {item['examples']} | {fmt(item['mc_exact_match_accuracy'])} | {fmt(item['tf_exact_match_accuracy'])} | {fmt(item['oe_parse_rate'])} | {fmt(item['combined_task_success_rate'])} |")
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_dir, predictions_dir, output_dir = map(Path, (args.manifest_dir, args.predictions_dir, args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        references = list(read_jsonl(manifest_dir / f"eval_{dataset}.jsonl"))
        predictions = list(read_jsonl(predictions_dir / f"{dataset}.predictions.jsonl"))
        if len(references) != len(predictions):
            raise RuntimeError(f"{dataset}: manifest/prediction count mismatch")
        for index, (reference, prediction) in enumerate(zip(references, predictions)):
            if prediction["index"] != index or prediction["sample_id"] != reference["sample_id"]:
                raise RuntimeError(f"{dataset}: identity mismatch at {index}")
            group = reference["question_group"]
            predicted = parse_prediction(prediction["raw_response"], group) if group in ("MC", "TF") else None
            target = target_label(reference["label_reference"], group) if group in ("MC", "TF") else None
            if group in ("MC", "TF") and target is None:
                raise RuntimeError(f"{dataset}: missing target label at {index}")
            exact = predicted == target if group in ("MC", "TF") else None
            parseable = open_parseable(prediction["raw_response"]) if group == "OE" else None
            rows.append({"dataset": dataset, "index": index, "sample_id": reference["sample_id"], "question_group": group, "predicted_label": predicted, "target_label": target, "exact_match": exact, "parseable": parseable, "contract_error": response_contract_error(prediction["raw_response"], group), "success": exact if group in ("MC", "TF") else parseable})
    with (output_dir / "label_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    (output_dir / "label_metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "label_metrics.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
