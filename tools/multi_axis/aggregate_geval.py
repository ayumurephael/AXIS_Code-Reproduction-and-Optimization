from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tools.multi_axis.datasets import (
    OFFICIAL_BIAS_NEUTRALIZED_DATASETS,
    SUPPORTED_EVALUATION_DATASETS,
    validate_datasets,
)

DATASETS = OFFICIAL_BIAS_NEUTRALIZED_DATASETS
JUDGES = ("deepseek-v4-pro", "gemini-2.5-pro", "gpt-5.4", "qwen3.5-397b-a17b", "qwen3-30b-a3b-instruct-2507")
COLUMNS = ("MC Final", "MC Corr.", "MC Rsn.", "OE Final", "OE Acc.", "OE Comp.", "OE Rel.", "TF Final", "TF Corr.", "TF Justif.")
TYPE_INFO = {
    "multiple_choice": ("MC", {"correctness": "MC Corr.", "reasoning_quality": "MC Rsn."}, {"correctness": 0.7, "reasoning_quality": 0.3}, "MC Final"),
    "open_ended": ("OE", {"accuracy": "OE Acc.", "completeness": "OE Comp.", "relevance": "OE Rel."}, {"accuracy": 0.35, "completeness": 0.35, "relevance": 0.3}, "OE Final"),
    "true_false": ("TF", {"correctness": "TF Corr.", "justification_quality": "TF Justif."}, {"correctness": 0.6, "justification_quality": 0.4}, "TF Final"),
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def example_scores(rows: Sequence[dict]):
    grouped = collections.defaultdict(dict)
    metadata = {}
    for row in rows:
        key = row["record_id"]
        if row["dimension"] in grouped[key]:
            raise RuntimeError(f"Duplicate G-Eval task {key}/{row['dimension']}")
        grouped[key][row["dimension"]] = float(row["score"])
        metadata[key] = row
    examples = []
    for key, scores in grouped.items():
        question_type = metadata[key]["question_type"]
        _, _, weights, _ = TYPE_INFO[question_type]
        if set(scores) != set(weights):
            raise RuntimeError(f"{key}: dimensions {sorted(scores)} != {sorted(weights)}")
        examples.append({"record_id": key, "dataset": metadata[key]["dataset"], "question_type": question_type, "scores": scores, "final": sum(scores[name] * weight for name, weight in weights.items())})
    return examples


def ten_metrics(examples: Sequence[dict]):
    output = {column: None for column in COLUMNS}
    counts = {}
    for question_type, (short, dimensions, _weights, final_name) in TYPE_INFO.items():
        subset = [item for item in examples if item["question_type"] == question_type]
        counts[short] = len(subset)
        if not subset:
            continue
        output[final_name] = statistics.fmean(item["final"] for item in subset)
        for dimension, column in dimensions.items():
            output[column] = statistics.fmean(item["scores"][dimension] for item in subset)
    output["counts"] = counts
    output["examples"] = len(examples)
    return output


def mean_tables(tables: Sequence[dict]):
    return {**{column: statistics.fmean(table[column] for table in tables if table[column] is not None) for column in COLUMNS}, "datasets": len(tables), "counts": {key: sum(table["counts"][key] for table in tables) for key in ("MC", "OE", "TF")}}


def fmt(value):
    return "?" if value is None else f"{value:.4f}"


def table_markdown(title: str, rows: Sequence[tuple[str, dict]]):
    lines = [f"## {title}", "", "| Scope | " + " | ".join(COLUMNS) + " | n(MC/OE/TF) |", "|---|" + "---:|" * (len(COLUMNS) + 1)]
    for name, table in rows:
        counts = table["counts"]
        lines.append("| " + name + " | " + " | ".join(fmt(table[column]) for column in COLUMNS) + f" | {counts['MC']}/{counts['OE']}/{counts['TF']} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, help="Contains <judge>/<dataset>.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_EVALUATION_DATASETS,
        default=list(DATASETS),
    )
    parser.add_argument(
        "--judges", nargs="+", choices=JUDGES, default=list(JUDGES)
    )
    args = parser.parse_args()
    root, output_dir = Path(args.results_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = validate_datasets(args.datasets)
    judges = tuple(args.judges)
    if len(judges) != len(set(judges)):
        raise ValueError("Judge names must be unique")
    result = {"by_judge_dataset": {}, "micro": {}, "macro": {}}
    all_examples = {}
    result["datasets"] = list(datasets)
    result["judges"] = list(judges)
    result["macro_warning"] = (
        "478 and 478new are wording views of the same samples; a macro across both "
        "is a view-level diagnostic, not an independent-dataset macro."
        if {"478", "478new"}.issubset(datasets)
        else None
    )
    markdown_parts = ["# Multi-AXIS G-Eval", "", "All scores are raw G-Eval scores on the 1–5 scale."]
    if result["macro_warning"]:
        markdown_parts.extend(["", f"> {result['macro_warning']}"])
    for judge in judges:
        result["by_judge_dataset"][judge] = {}
        all_examples[judge] = []
        dataset_tables = []
        for dataset in datasets:
            path = root / judge / f"{dataset}.jsonl"
            examples = example_scores(read_jsonl(path))
            if any(item["dataset"] != dataset for item in examples):
                raise RuntimeError(f"{judge}/{dataset}: dataset identity mismatch")
            table = ten_metrics(examples)
            result["by_judge_dataset"][judge][dataset] = table
            dataset_tables.append(table)
            all_examples[judge].extend(examples)
        result["micro"][judge] = ten_metrics(all_examples[judge])
        result["macro"][judge] = mean_tables(dataset_tables)
        markdown_parts.extend(["", table_markdown(judge, [(dataset, result["by_judge_dataset"][judge][dataset]) for dataset in datasets] + [("Micro", result["micro"][judge]), ("Macro", result["macro"][judge])])])
    robustness = {}
    for dataset in datasets + ("micro", "macro"):
        robustness[dataset] = {}
        tables = [result[dataset][judge] if dataset in ("micro", "macro") else result["by_judge_dataset"][judge][dataset] for judge in judges]
        for column in COLUMNS:
            values = [table[column] for table in tables]
            robustness[dataset][column] = {"mean": statistics.fmean(values), "std_population": statistics.pstdev(values), "min": min(values), "max": max(values), "range": max(values) - min(values)}
    result["cross_judge_robustness"] = robustness
    (output_dir / "geval_10metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "geval_10metrics.md").write_text("\n".join(markdown_parts) + "\n", encoding="utf-8")
    print(json.dumps({"judges": len(judges), "datasets": len(datasets), "complete": True}))


if __name__ == "__main__":
    main()
