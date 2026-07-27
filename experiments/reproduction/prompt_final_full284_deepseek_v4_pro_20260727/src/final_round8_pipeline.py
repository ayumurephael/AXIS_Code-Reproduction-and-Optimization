"""Assemble and audit the frozen full284 Round-8 joint prompt route."""

from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_pipeline import (
    METRICS,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round7_pipeline import (
    ROUND5_SOURCE_MODE,
    ROUND8_MODE,
    selected_for_mode,
)
from src.models.AXIS.prompt_stage_a import _has_explicit_tf_negative_cue
from tools.axis_repro.build_tables import aggregate
from tools.axis_repro.common import read_jsonl


FINAL_MODE = "prompt_final_round8_joint"
JOINT_MODE = "lit_r5_04_joint_status_tf_neg_re2"


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def unique_by_record(rows: list[dict], mode: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        if row["mode"] != mode:
            continue
        record_id = row["record_id"]
        if record_id in result:
            raise RuntimeError(f"Duplicate {mode} prediction: {record_id}")
        result[record_id] = row
    return result


def component_source(
    baseline: dict,
    joint: dict,
    oe_revision: dict,
) -> tuple[str, dict]:
    question_type = baseline["question_type"]
    if question_type == "multiple_choice":
        return JOINT_MODE, joint
    if question_type == "true_false":
        if _has_explicit_tf_negative_cue(baseline["question"]):
            return JOINT_MODE, joint
        return "base", baseline
    if question_type == "open_ended":
        if selected_for_mode(
            ROUND8_MODE,
            baseline["response"],
            oe_revision["response"],
        ):
            return ROUND5_SOURCE_MODE, oe_revision
        return "base", baseline
    raise ValueError(f"Unknown question type: {question_type!r}")


def assemble_predictions(args: argparse.Namespace) -> None:
    baseline = unique_by_record(read_jsonl(args.baseline_predictions), "base")
    joint = unique_by_record(read_jsonl(args.joint_predictions), JOINT_MODE)
    oe = unique_by_record(
        read_jsonl(args.oe_revision_predictions),
        ROUND5_SOURCE_MODE,
    )
    if set(baseline) != set(joint):
        raise RuntimeError("Joint prediction IDs do not match Baseline full284")
    oe_ids = {
        record_id
        for record_id, row in baseline.items()
        if row["question_type"] == "open_ended"
    }
    if not oe_ids <= set(oe):
        raise RuntimeError(f"Missing OE revisions: {sorted(oe_ids - set(oe))[:5]}")

    base_rows = [copy.deepcopy(baseline[key]) for key in sorted(baseline)]
    candidates = []
    judge_rows = []
    provenance = []
    source_counts = collections.Counter()
    changed_counts = collections.Counter()
    for record_id in sorted(baseline):
        base = baseline[record_id]
        oe_revision = oe.get(record_id, base)
        source_mode, source = component_source(
            base,
            joint[record_id],
            oe_revision,
        )
        candidate = copy.deepcopy(source)
        candidate["mode"] = FINAL_MODE
        candidate["component_source_mode"] = source_mode
        changed_response = candidate["response"].strip() != base["response"].strip()
        candidate["changed_response_vs_baseline"] = changed_response
        candidates.append(candidate)
        source_counts[(base["question_type"], source_mode)] += 1
        if changed_response:
            judge_rows.append(copy.deepcopy(candidate))
            changed_counts[base["question_type"]] += 1
        provenance.append(
            {
                "record_id": record_id,
                "question_type": base["question_type"],
                "component_source_mode": source_mode,
                "changed_response_vs_baseline": changed_response,
            }
        )

    all_rows = base_rows + candidates
    all_rows.sort(key=lambda row: (row["record_id"], row["mode"]))
    judge_rows.sort(key=lambda row: row["record_id"])
    write_jsonl(args.output_predictions, all_rows)
    write_jsonl(args.output_candidates, candidates)
    write_jsonl(args.output_judge_predictions, judge_rows)
    manifest = {
        "mode": FINAL_MODE,
        "record_count": len(base_rows),
        "prediction_rows": len(all_rows),
        "candidate_rows": len(candidates),
        "judge_prediction_rows": len(judge_rows),
        "source_counts": {
            f"{family}|{source}": count
            for (family, source), count in sorted(source_counts.items())
        },
        "changed_response_counts": dict(sorted(changed_counts.items())),
        "components": provenance,
    }
    Path(args.provenance).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "components"}
        )
    )


def merge_scores(args: argparse.Namespace) -> None:
    baseline = [
        row for row in read_jsonl(args.baseline_scores) if row["mode"] == "base"
    ]
    candidate = [
        row
        for row in read_jsonl(args.candidate_scores)
        if row["mode"] == FINAL_MODE
    ]
    base_keys = {(row["record_id"], row["dimension"]) for row in baseline}
    candidate_keys = {
        (row["record_id"], row["dimension"]) for row in candidate
    }
    if base_keys != candidate_keys:
        raise RuntimeError("Candidate score keys do not match Baseline full284")
    rows = baseline + candidate
    rows.sort(key=lambda row: (row["record_id"], row["mode"], row["dimension"]))
    write_jsonl(args.output, rows)
    print(json.dumps({"baseline_scores": len(baseline), "candidate_scores": len(candidate)}))


def displayed(value: float) -> float:
    return float(f"{value:.4f}")


def analyze(args: argparse.Namespace) -> None:
    models, _ = aggregate(read_jsonl(args.scores))
    baseline = models["base"]
    candidate = models[FINAL_MODE]
    metrics = {}
    all_nonlower = True
    strict = 0
    for label, key in METRICS:
        base_value = baseline[key]
        candidate_value = candidate[key]
        delta = candidate_value - base_value
        relative = 100.0 * delta / base_value if base_value else None
        base_display = displayed(base_value)
        candidate_display = displayed(candidate_value)
        nonlower = candidate_display >= base_display
        improved = candidate_display > base_display
        all_nonlower = all_nonlower and nonlower
        strict += int(improved)
        metrics[label] = {
            "baseline": base_value,
            "candidate": candidate_value,
            "absolute_delta": delta,
            "relative_percent": relative,
            "baseline_display_4dp": base_display,
            "candidate_display_4dp": candidate_display,
            "nonlower_4dp": nonlower,
            "strict_improvement_4dp": improved,
        }
    payload = {
        "mode": FINAL_MODE,
        "table_i_display_precision": 4,
        "all_ten_nonlower": all_nonlower,
        "strict_improvements": strict,
        "metrics": metrics,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    header = ["Metric", "Baseline", "Candidate", "Abs. Δ", "Rel. Δ", "Nonlower"]
    lines = ["| " + " | ".join(header) + " |", "|---|---:|---:|---:|---:|:---:|"]
    for label, _ in METRICS:
        row = metrics[label]
        relative = row["relative_percent"]
        lines.append(
            f"| {label} | {row['baseline']:.4f} | {row['candidate']:.4f} | "
            f"{row['absolute_delta']:+.4f} | {relative:+.2f}% | "
            f"{'✓' if row['nonlower_4dp'] else '✗'} |"
        )
    lines.extend(
        [
            "",
            f"- All ten nonlower at 4 decimals: `{all_nonlower}`",
            f"- Strict improvements at 4 decimals: `{strict}`",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_ten_nonlower": all_nonlower, "strict_improvements": strict}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble-predictions")
    assemble_parser.add_argument("--baseline-predictions", required=True)
    assemble_parser.add_argument("--joint-predictions", required=True)
    assemble_parser.add_argument("--oe-revision-predictions", required=True)
    assemble_parser.add_argument("--output-predictions", required=True)
    assemble_parser.add_argument("--output-candidates", required=True)
    assemble_parser.add_argument("--output-judge-predictions", required=True)
    assemble_parser.add_argument("--provenance", required=True)
    assemble_parser.set_defaults(function=assemble_predictions)
    merge_parser = subparsers.add_parser("merge-scores")
    merge_parser.add_argument("--baseline-scores", required=True)
    merge_parser.add_argument("--candidate-scores", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(function=merge_scores)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--scores", required=True)
    analyze_parser.add_argument("--output-json", required=True)
    analyze_parser.add_argument("--output-md", required=True)
    analyze_parser.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()