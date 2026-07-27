"""Build and analyze a paired repeat Judge audit for the Round-9 finalists.

The locked primary full-284 run remains the formal result.  This module
re-judges both the Baseline and candidate responses for every response changed
by the union Round-9 route so that Judge stochasticity cancels as much as
possible in a same-run comparison.  Unchanged responses reuse the canonical
Baseline score exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round8_pipeline import (
    FINAL_MODE as INTERNAL_JUDGE_MODE,
)
from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round9_variants import (
    FINAL_MODES,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_pipeline import (
    METRICS,
)
from tools.axis_repro.build_tables import DIMS, aggregate
from tools.axis_repro.common import read_jsonl


PAIRED_BASE_MODE = "prompt_final_round9_paired_baseline"
PAIRED_CANDIDATE_MODE = "prompt_final_round9_paired_candidate"


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
        record_id = row["record_id"]
        if record_id in result:
            raise RuntimeError(f"Duplicate {mode} row: {record_id}")
        result[record_id] = row
    return result


def build_predictions(args: argparse.Namespace) -> None:
    baseline = index_mode(read_jsonl(args.baseline_predictions), "base")
    changed = index_mode(read_jsonl(args.changed_predictions), INTERNAL_JUDGE_MODE)
    if not changed:
        raise RuntimeError("No changed Round-9 candidate predictions were found")
    if not set(changed) <= set(baseline):
        raise RuntimeError("Changed candidate IDs are not a subset of Baseline")

    rows = []
    family_counts = {}
    for record_id in sorted(changed):
        base = copy.deepcopy(baseline[record_id])
        candidate = copy.deepcopy(changed[record_id])
        if base["response"].strip() == candidate["response"].strip():
            raise RuntimeError(f"Unchanged response in changed set: {record_id}")
        family = base["question_type"]
        family_counts[family] = family_counts.get(family, 0) + 1
        base["mode"] = PAIRED_BASE_MODE
        candidate["mode"] = PAIRED_CANDIDATE_MODE
        rows.extend((base, candidate))

    rows.sort(key=lambda row: (row["record_id"], row["mode"]))
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "changed_records": len(changed),
                "prediction_rows": len(rows),
                "family_counts": family_counts,
            }
        )
    )


def displayed(value: float) -> float:
    return float(f"{value:.4f}")


def analyze(args: argparse.Namespace) -> None:
    final_predictions = read_jsonl(args.final_predictions)
    baseline_predictions = index_mode(final_predictions, "base")
    canonical_scores = [
        row for row in read_jsonl(args.canonical_scores) if row["mode"] == "base"
    ]
    canonical_index = {
        (row["record_id"], row["dimension"]): row for row in canonical_scores
    }
    paired_scores = read_jsonl(args.paired_scores)
    paired_base = {
        (row["record_id"], row["dimension"]): row
        for row in paired_scores
        if row["mode"] == PAIRED_BASE_MODE
    }
    paired_candidate = {
        (row["record_id"], row["dimension"]): row
        for row in paired_scores
        if row["mode"] == PAIRED_CANDIDATE_MODE
    }
    expected_paired_keys = set(paired_base)
    if expected_paired_keys != set(paired_candidate):
        raise RuntimeError("Paired Baseline and candidate score keys do not match")

    output_rows = []
    route_metadata = {}
    for route in FINAL_MODES:
        candidate_predictions = index_mode(final_predictions, route)
        if set(candidate_predictions) != set(baseline_predictions):
            raise RuntimeError(f"{route} prediction IDs do not match Baseline")
        base_mode = f"{route}__paired_baseline"
        candidate_mode = f"{route}__paired_candidate"
        changed_records = 0
        repeated_dimensions = 0
        reused_dimensions = 0
        for record_id in sorted(baseline_predictions):
            base_prediction = baseline_predictions[record_id]
            candidate_prediction = candidate_predictions[record_id]
            changed = (
                base_prediction["response"].strip()
                != candidate_prediction["response"].strip()
            )
            changed_records += int(changed)
            for dimension in DIMS[base_prediction["question_type"]]:
                key = (record_id, dimension)
                if changed:
                    base_source = paired_base[key]
                    candidate_source = paired_candidate[key]
                    repeated_dimensions += 1
                else:
                    base_source = canonical_index[key]
                    candidate_source = canonical_index[key]
                    reused_dimensions += 1
                base_row = copy.deepcopy(base_source)
                candidate_row = copy.deepcopy(candidate_source)
                base_row["mode"] = base_mode
                candidate_row["mode"] = candidate_mode
                base_row["paired_repeat_source"] = changed
                candidate_row["paired_repeat_source"] = changed
                output_rows.extend((base_row, candidate_row))
        route_metadata[route] = {
            "changed_records": changed_records,
            "paired_repeat_dimensions": repeated_dimensions,
            "canonical_reuse_dimensions": reused_dimensions,
            "baseline_mode": base_mode,
            "candidate_mode": candidate_mode,
        }

    output_rows.sort(
        key=lambda row: (row["record_id"], row["mode"], row["dimension"])
    )
    write_jsonl(args.output_scores, output_rows)
    models, _ = aggregate(output_rows)

    results = {}
    for route, metadata in route_metadata.items():
        baseline = models[metadata["baseline_mode"]]
        candidate = models[metadata["candidate_mode"]]
        metrics = {}
        for label, key in METRICS:
            base_value = baseline[key]
            candidate_value = candidate[key]
            delta = candidate_value - base_value
            metrics[label] = {
                "baseline": base_value,
                "candidate": candidate_value,
                "absolute_delta": delta,
                "relative_percent": 100.0 * delta / base_value,
                "nonlower_4dp": displayed(candidate_value)
                >= displayed(base_value),
                "strict_improvement_4dp": displayed(candidate_value)
                > displayed(base_value),
            }
        results[route] = {
            **metadata,
            "all_ten_nonlower": all(
                row["nonlower_4dp"] for row in metrics.values()
            ),
            "strict_improvements": sum(
                row["strict_improvement_4dp"] for row in metrics.values()
            ),
            "metrics": metrics,
        }

    payload = {
        "role": "secondary paired Judge-variance audit",
        "formal_result_source": "locked primary full284 run",
        "table_i_display_precision": 4,
        "routes": results,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    header = ["Prompt"] + [label for label, _ in METRICS] + ["10/10", "Strict +"]
    lines = [
        "# Round-9 paired repeat Judge audit",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for route, result in results.items():
        cells = [
            (
                f"{result['metrics'][label]['candidate']:.4f} "
                f"({result['metrics'][label]['absolute_delta']:+.4f})"
            )
            for label, _ in METRICS
        ]
        lines.append(
            f"| {route} | "
            + " | ".join(cells)
            + f" | {result['all_ten_nonlower']} | "
            + f"{result['strict_improvements']} |"
        )
    lines.extend(
        [
            "",
            "Unchanged responses reuse the canonical Baseline score on both "
            "sides. Changed responses are re-judged as a Baseline/candidate "
            "pair in this repeat. This is a secondary Judge-variance audit; "
            "it does not replace the locked primary formal result.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                route: {
                    "all_ten_nonlower": result["all_ten_nonlower"],
                    "strict_improvements": result["strict_improvements"],
                }
                for route, result in results.items()
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-predictions")
    build.add_argument("--baseline-predictions", required=True)
    build.add_argument("--changed-predictions", required=True)
    build.add_argument("--output", required=True)
    build.set_defaults(function=build_predictions)
    audit = subparsers.add_parser("analyze")
    audit.add_argument("--final-predictions", required=True)
    audit.add_argument("--canonical-scores", required=True)
    audit.add_argument("--paired-scores", required=True)
    audit.add_argument("--output-scores", required=True)
    audit.add_argument("--output-json", required=True)
    audit.add_argument("--output-md", required=True)
    audit.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
