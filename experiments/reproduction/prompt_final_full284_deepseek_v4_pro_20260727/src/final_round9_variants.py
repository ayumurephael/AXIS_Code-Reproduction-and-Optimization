"""Assemble all exposed-split-qualified Round-9 full284 prompt routes."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round8_pipeline import (
    FINAL_MODE as INTERNAL_JUDGE_MODE,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_pipeline import (
    METRICS,
)
from tools.axis_repro.build_tables import DIMS, aggregate
from tools.axis_repro.common import read_jsonl


MC_OE_MODE = "prompt_final_round9_mc_oe"
TF_OE_MODE = "prompt_final_round9_tf_oe"
JOINT_MODE = "prompt_final_round9_joint"
FINAL_MODES = (MC_OE_MODE, TF_OE_MODE, JOINT_MODE)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def index_mode(rows: list[dict], mode: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if row["mode"] != mode:
            continue
        if row["record_id"] in result:
            raise RuntimeError(f"Duplicate {mode} row: {row['record_id']}")
        result[row["record_id"]] = row
    return result


def route_source(mode: str, baseline: dict, joint: dict) -> dict:
    question_type = baseline["question_type"]
    if mode == JOINT_MODE:
        return joint
    if mode == MC_OE_MODE:
        return baseline if question_type == "true_false" else joint
    if mode == TF_OE_MODE:
        return baseline if question_type == "multiple_choice" else joint
    raise ValueError(f"Unknown final mode: {mode!r}")


def assemble_predictions(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.internal_predictions)
    baseline = index_mode(rows, "base")
    joint = index_mode(rows, INTERNAL_JUDGE_MODE)
    if set(baseline) != set(joint):
        raise RuntimeError("Internal joint prediction IDs do not match Baseline")
    base_rows = [copy.deepcopy(baseline[key]) for key in sorted(baseline)]
    candidates = []
    source_counts = {}
    changed_counts = {}
    provenance = []
    for mode in FINAL_MODES:
        source_counts[mode] = {}
        changed_counts[mode] = {}
        for record_id in sorted(baseline):
            base = baseline[record_id]
            source = route_source(mode, base, joint[record_id])
            candidate = copy.deepcopy(source)
            candidate["mode"] = mode
            changed = candidate["response"].strip() != base["response"].strip()
            candidate["changed_response_vs_baseline"] = changed
            candidate["internal_component_mode"] = source["mode"]
            candidates.append(candidate)
            family = base["question_type"]
            source_counts[mode][family] = source_counts[mode].get(family, 0) + int(
                source is joint[record_id]
            )
            changed_counts[mode][family] = changed_counts[mode].get(family, 0) + int(changed)
            provenance.append(
                {
                    "record_id": record_id,
                    "mode": mode,
                    "question_type": family,
                    "internal_component_mode": source["mode"],
                    "changed_response_vs_baseline": changed,
                }
            )
    all_rows = base_rows + candidates
    all_rows.sort(key=lambda row: (row["record_id"], row["mode"]))
    candidates.sort(key=lambda row: (row["record_id"], row["mode"]))
    write_jsonl(args.output_predictions, all_rows)
    write_jsonl(args.output_candidates, candidates)
    manifest = {
        "internal_judge_mode": INTERNAL_JUDGE_MODE,
        "final_modes": list(FINAL_MODES),
        "records_per_mode": len(base_rows),
        "prediction_rows": len(all_rows),
        "candidate_rows": len(candidates),
        "source_counts": source_counts,
        "changed_response_counts": changed_counts,
        "components": provenance,
    }
    Path(args.provenance).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "components"}))


def assemble_scores(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.predictions)
    baseline_predictions = index_mode(predictions, "base")
    baseline_scores = [
        row for row in read_jsonl(args.baseline_scores) if row["mode"] == "base"
    ]
    base_score_index = {
        (row["record_id"], row["dimension"]): row for row in baseline_scores
    }
    judge_scores = [
        row
        for row in read_jsonl(args.internal_judge_scores)
        if row["mode"] == INTERNAL_JUDGE_MODE
    ]
    judge_index = {
        (row["record_id"], row["dimension"]): row for row in judge_scores
    }
    output = [copy.deepcopy(row) for row in baseline_scores]
    reused = 0
    judged = 0
    for prediction in predictions:
        if prediction["mode"] not in FINAL_MODES:
            continue
        record_id = prediction["record_id"]
        base_prediction = baseline_predictions[record_id]
        reuse = prediction["response"].strip() == base_prediction["response"].strip()
        for dimension in DIMS[prediction["question_type"]]:
            key = (record_id, dimension)
            source = base_score_index[key] if reuse else judge_index[key]
            row = copy.deepcopy(source)
            row["mode"] = prediction["mode"]
            row["component_score_source_mode"] = "base" if reuse else INTERNAL_JUDGE_MODE
            row["exact_response_score_reuse"] = reuse
            output.append(row)
            reused += int(reuse)
            judged += int(not reuse)
    output.sort(key=lambda row: (row["record_id"], row["mode"], row["dimension"]))
    expected = len(baseline_scores) * (1 + len(FINAL_MODES))
    if len(output) != expected:
        raise RuntimeError(f"Score count mismatch: {len(output)} != {expected}")
    write_jsonl(args.output, output)
    print(json.dumps({"score_rows": len(output), "reused_dimensions": reused, "judged_dimensions": judged}))


def displayed(value: float) -> float:
    return float(f"{value:.4f}")


def analyze(args: argparse.Namespace) -> None:
    models, _ = aggregate(read_jsonl(args.scores))
    baseline = models["base"]
    candidates = {}
    for mode in FINAL_MODES:
        values = models[mode]
        metrics = {}
        for label, key in METRICS:
            base_value = baseline[key]
            candidate_value = values[key]
            delta = candidate_value - base_value
            metrics[label] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "absolute_delta": delta,
                "relative_percent": 100 * delta / base_value,
                "nonlower_4dp": displayed(candidate_value) >= displayed(base_value),
                "strict_improvement_4dp": displayed(candidate_value) > displayed(base_value),
            }
        candidates[mode] = {
            "all_ten_nonlower": all(row["nonlower_4dp"] for row in metrics.values()),
            "strict_improvements": sum(row["strict_improvement_4dp"] for row in metrics.values()),
            "metrics": metrics,
        }
    payload = {
        "table_i_display_precision": 4,
        "baseline": {label: baseline[key] for label, key in METRICS},
        "candidates": candidates,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    header = ["Prompt"] + [label for label, _ in METRICS] + ["10/10", "Strict +"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.append("| Baseline | " + " | ".join(f"{baseline[key]:.4f}" for _, key in METRICS) + " | — | — |")
    for mode in FINAL_MODES:
        row = candidates[mode]
        cells = [
            f"{row['metrics'][label]['candidate']:.4f} ({row['metrics'][label]['absolute_delta']:+.4f})"
            for label, _ in METRICS
        ]
        lines.append(
            f"| {mode} | " + " | ".join(cells)
            + f" | {row['all_ten_nonlower']} | {row['strict_improvements']} |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({mode: {key: value for key, value in row.items() if key != "metrics"} for mode, row in candidates.items()}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    pred = subparsers.add_parser("assemble-predictions")
    pred.add_argument("--internal-predictions", required=True)
    pred.add_argument("--output-predictions", required=True)
    pred.add_argument("--output-candidates", required=True)
    pred.add_argument("--provenance", required=True)
    pred.set_defaults(function=assemble_predictions)
    score = subparsers.add_parser("assemble-scores")
    score.add_argument("--predictions", required=True)
    score.add_argument("--baseline-scores", required=True)
    score.add_argument("--internal-judge-scores", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(function=assemble_scores)
    analysis = subparsers.add_parser("analyze")
    analysis.add_argument("--scores", required=True)
    analysis.add_argument("--output-json", required=True)
    analysis.add_argument("--output-md", required=True)
    analysis.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()