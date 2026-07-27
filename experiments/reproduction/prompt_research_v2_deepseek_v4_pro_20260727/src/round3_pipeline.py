"""Select, assemble, and analyze Round-3 balanced-evidence OE routers."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_pipeline import (
    METRICS,
    allowed_series,
    baseline_rows,
    unique_index,
    write_jsonl,
)
from src.models.AXIS.prompt_stage_a import (
    V2_R3_MAIN_MODES,
    build_question_prompt,
)
from tools.axis_repro.build_tables import DIMS, aggregate
from tools.axis_repro.common import read_jsonl


SOURCE_MODE = "v2_r1_03_oe_evidence_router"


def _render(row: dict, mode: str) -> str:
    return build_question_prompt(
        question=row["question"],
        question_type=row["question_type"],
        start=row["start_index"],
        end=row["end_index"],
        serialized_values="0",
        local_hint_tokens="<|local_hint|>",
        fixed_hint_tokens="<|fixed_hint|>" * 30,
        mode=mode,
        aligned_rows="",
    )


def route_changed(row: dict, mode: str) -> bool:
    if mode not in V2_R3_MAIN_MODES:
        raise ValueError(f"Unknown Round-3 mode: {mode}")
    routed = _render(row, mode)
    baseline = _render(row, "base")
    if routed == baseline:
        return False
    source = _render(row, SOURCE_MODE)
    if routed != source:
        raise RuntimeError(
            f"{mode} does not reuse the frozen source prompt for "
            f"{row['record_id']}"
        )
    return True


def select(args: argparse.Namespace) -> None:
    modes = tuple(args.modes or V2_R3_MAIN_MODES)
    unknown = set(modes) - set(V2_R3_MAIN_MODES)
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    rows = [
        row
        for row in read_jsonl(args.source_predictions)
        if row["mode"] == SOURCE_MODE
    ]
    by_id = {row["record_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise RuntimeError("Source component prediction IDs are not unique")
    selected_ids = {
        record_id
        for record_id, row in by_id.items()
        if any(route_changed(row, mode) for mode in modes)
    }
    selected = [by_id[record_id] for record_id in sorted(selected_ids)]
    write_jsonl(args.output, selected)
    payload = {
        "source_mode": SOURCE_MODE,
        "modes": list(modes),
        "source_components": len(rows),
        "selected_unique_components": len(selected),
        "selected_record_ids": sorted(selected_ids),
        "matches_by_mode": {
            mode: sum(route_changed(row, mode) for row in rows)
            for mode in modes
        },
    }
    Path(args.provenance).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "selected_record_ids"}))


def assemble(args: argparse.Namespace) -> None:
    modes = tuple(args.modes or V2_R3_MAIN_MODES)
    unknown = set(modes) - set(V2_R3_MAIN_MODES)
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    allowed = allowed_series(args.split_manifest, args.split_key)
    baseline_predictions = baseline_rows(args.baseline_predictions, allowed)
    record_ids = {row["record_id"] for row in baseline_predictions}
    baseline_scores = [
        row
        for row in read_jsonl(args.baseline_scores)
        if row["mode"] == "base" and row["record_id"] in record_ids
    ]
    component_predictions = [
        row
        for row in read_jsonl(args.component_predictions)
        if row["mode"] == SOURCE_MODE and row["record_id"] in record_ids
    ]
    component_scores = [
        row
        for row in read_jsonl(args.component_scores)
        if row["mode"] == SOURCE_MODE and row["record_id"] in record_ids
    ]
    pred_index = unique_index(component_predictions)
    score_index = unique_index(component_scores, scores=True)
    baseline_score_index = unique_index(baseline_scores, scores=True)

    output_predictions = [copy.deepcopy(row) for row in baseline_predictions]
    output_scores = [copy.deepcopy(row) for row in baseline_scores]
    provenance = []
    for mode in modes:
        for base in baseline_predictions:
            changed = route_changed(base, mode)
            source_mode = SOURCE_MODE if changed else "base"
            if changed:
                key = (base["record_id"], SOURCE_MODE)
                if key not in pred_index:
                    raise RuntimeError(f"Missing prediction component: {key}")
                source_prediction = pred_index[key]
            else:
                source_prediction = base
            candidate = copy.deepcopy(source_prediction)
            candidate["mode"] = mode
            candidate["component_source_mode"] = source_mode
            output_predictions.append(candidate)
            for dimension in DIMS[base["question_type"]]:
                source_scores = score_index if changed else baseline_score_index
                key = (base["record_id"], source_mode, dimension)
                if key not in source_scores:
                    raise RuntimeError(f"Missing score component: {key}")
                score = copy.deepcopy(source_scores[key])
                score["mode"] = mode
                score["component_source_mode"] = source_mode
                output_scores.append(score)
            provenance.append(
                {
                    "record_id": base["record_id"],
                    "mode": mode,
                    "question_type": base["question_type"],
                    "component_source_mode": source_mode,
                }
            )

    output_predictions.sort(key=lambda row: (row["record_id"], row["mode"]))
    output_scores.sort(
        key=lambda row: (row["record_id"], row["mode"], row["dimension"])
    )
    if len(unique_index(output_predictions)) != len(output_predictions):
        raise RuntimeError("Assembled prediction keys are not unique")
    if len(unique_index(output_scores, scores=True)) != len(output_scores):
        raise RuntimeError("Assembled score keys are not unique")
    write_jsonl(args.output_predictions, output_predictions)
    write_jsonl(args.output_scores, output_scores)
    manifest = {
        "split_key": args.split_key,
        "modes": list(modes),
        "records_per_mode": len(baseline_predictions),
        "prediction_rows": len(output_predictions),
        "score_rows": len(output_scores),
        "match_counts": dict(
            Counter(
                item["mode"]
                for item in provenance
                if item["component_source_mode"] == SOURCE_MODE
            )
        ),
        "components": provenance,
    }
    Path(args.provenance).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "components"}))


def analyze(args: argparse.Namespace) -> None:
    models, _ = aggregate(read_jsonl(args.scores))
    baseline = models["base"]
    summary = {}
    for mode in sorted(models):
        if mode == "base":
            continue
        values = models[mode]
        deltas = {
            label: values[key] - baseline[key] for label, key in METRICS
        }
        family_deltas = {
            label: deltas[label]
            for label, key in METRICS
            if key.startswith("open_ended/")
        }
        summary[mode] = {
            "metrics": {label: values[key] for label, key in METRICS},
            "deltas": deltas,
            "all_ten_nonlower": all(
                value >= -1e-12 for value in deltas.values()
            ),
            "strict_improvements": sum(
                value > 1e-12 for value in deltas.values()
            ),
            "changed_family": "open_ended",
            "changed_family_nonlower": all(
                value >= -1e-12 for value in family_deltas.values()
            ),
            "changed_family_worst_delta": min(family_deltas.values()),
        }
    payload = {
        "baseline": {label: baseline[key] for label, key in METRICS},
        "candidates": summary,
    }
    Path(args.output_json).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    header = ["Mode"] + [label for label, _ in METRICS] + [
        "All 10 nonlower", "Strict +", "OE worst delta"
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        "| "
        + " | ".join(
            ["Baseline"]
            + [f"{baseline[key]:.4f}" for _, key in METRICS]
            + ["—", "—", "—"]
        )
        + " |",
    ]
    for mode, row in summary.items():
        lines.append(
            "| "
            + " | ".join(
                [mode]
                + [
                    f"{row['metrics'][label]:.4f} "
                    f"({row['deltas'][label]:+.4f})"
                    for label, _ in METRICS
                ]
                + [
                    str(row["all_ten_nonlower"]),
                    str(row["strict_improvements"]),
                    f"{row['changed_family_worst_delta']:+.4f}",
                ]
            )
            + " |"
        )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(summary)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--source-predictions", required=True)
    select_parser.add_argument("--modes", nargs="+")
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--provenance", required=True)
    select_parser.set_defaults(function=select)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--baseline-predictions", required=True)
    assemble_parser.add_argument("--baseline-scores", required=True)
    assemble_parser.add_argument("--component-predictions", required=True)
    assemble_parser.add_argument("--component-scores", required=True)
    assemble_parser.add_argument("--split-manifest", required=True)
    assemble_parser.add_argument("--split-key", required=True)
    assemble_parser.add_argument("--modes", nargs="+")
    assemble_parser.add_argument("--output-predictions", required=True)
    assemble_parser.add_argument("--output-scores", required=True)
    assemble_parser.add_argument("--provenance", required=True)
    assemble_parser.set_defaults(function=assemble)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--scores", required=True)
    analyze_parser.add_argument("--output-json", required=True)
    analyze_parser.add_argument("--output-md", required=True)
    analyze_parser.set_defaults(function=analyze)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
