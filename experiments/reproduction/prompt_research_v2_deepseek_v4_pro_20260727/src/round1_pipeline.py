"""Prepare, assemble, and analyze Round-1 component experiments."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_selection import (
    ACTIVE_FAMILY,
    prompt_changed,
)
from tools.axis_repro.build_tables import DIMS, aggregate
from tools.axis_repro.common import read_jsonl


METRICS = (
    ("MC Final", "multiple_choice/final"),
    ("MC Corr.", "multiple_choice/correctness"),
    ("MC Rsn.", "multiple_choice/reasoning_quality"),
    ("OE Final", "open_ended/final"),
    ("OE Acc.", "open_ended/accuracy"),
    ("OE Comp.", "open_ended/completeness"),
    ("OE Rel.", "open_ended/relevance"),
    ("TF Final", "true_false/final"),
    ("TF Corr.", "true_false/correctness"),
    ("TF Justif.", "true_false/justification_quality"),
)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def allowed_series(path: str | Path, key: str) -> set[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if key == "search_pool_series":
        return set(
            payload["screening_series"]
            + payload["validation_series"]
            + payload["holdout_series"]
        )
    return set(payload[key])


def baseline_rows(path: str | Path, allowed: set[str]) -> list[dict]:
    rows = [
        row
        for row in read_jsonl(path)
        if row["mode"] == "base" and row["series_file"] in allowed
    ]
    rows.sort(key=lambda row: row["record_id"])
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("Baseline prediction IDs are not unique")
    return rows


def selection_record(row: dict) -> SimpleNamespace:
    return SimpleNamespace(
        question=row["question"],
        question_type=row["question_type"],
        start_index=row["start_index"],
        end_index=row["end_index"],
    )


def prepare(args: argparse.Namespace) -> None:
    allowed = allowed_series(args.split_manifest, args.split_key)
    baseline = baseline_rows(args.baseline_predictions, allowed)
    raw = [
        row
        for row in read_jsonl(args.raw_predictions)
        if row["series_file"] in allowed and row.get("selected_component")
    ]
    raw_index = {(row["record_id"], row["mode"]): row for row in raw}
    if len(raw_index) != len(raw):
        raise RuntimeError("Raw selected component keys are not unique")

    expected = set()
    for row in baseline:
        record = selection_record(row)
        for mode in ACTIVE_FAMILY:
            if (
                row["question_type"] == ACTIVE_FAMILY[mode]
                and prompt_changed(record, mode)
            ):
                expected.add((row["record_id"], mode))
    actual = set(raw_index)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"Component mismatch: missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )

    selected = [raw_index[key] for key in sorted(expected)]
    write_jsonl(args.output, selected)
    provenance = {
        "split_key": args.split_key,
        "series_count": len(allowed),
        "baseline_records": len(baseline),
        "selected_components": len(selected),
        "mode_counts": dict(Counter(row["mode"] for row in selected)),
        "question_type_counts": dict(
            Counter(row["question_type"] for row in selected)
        ),
        "raw_support_rows_excluded": len(read_jsonl(args.raw_predictions))
        - len(raw),
    }
    Path(args.provenance).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance))


def unique_index(rows: list[dict], *, scores: bool = False) -> dict:
    result = {}
    for row in rows:
        key = (row["record_id"], row["mode"])
        if scores:
            key += (row["dimension"],)
        if key in result:
            raise RuntimeError(f"Duplicate key: {key}")
        result[key] = row
    return result


def assemble(args: argparse.Namespace) -> None:
    allowed = allowed_series(args.split_manifest, args.split_key)
    baseline_predictions = baseline_rows(args.baseline_predictions, allowed)
    record_ids = {row["record_id"] for row in baseline_predictions}
    baseline_scores = [
        row
        for row in read_jsonl(args.baseline_scores)
        if row["mode"] == "base" and row["record_id"] in record_ids
    ]
    component_predictions = read_jsonl(args.component_predictions)
    component_scores = read_jsonl(args.component_scores)
    pred_index = unique_index(component_predictions)
    score_index = unique_index(component_scores, scores=True)
    baseline_score_index = unique_index(baseline_scores, scores=True)
    modes = tuple(args.modes or ACTIVE_FAMILY)
    unknown = set(modes) - set(ACTIVE_FAMILY)
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")

    output_predictions = [copy.deepcopy(row) for row in baseline_predictions]
    output_scores = [copy.deepcopy(row) for row in baseline_scores]
    provenance = []
    for mode in modes:
        for base in baseline_predictions:
            changed = (
                base["question_type"] == ACTIVE_FAMILY[mode]
                and prompt_changed(selection_record(base), mode)
            )
            source_mode = mode if changed else "base"
            if changed:
                key = (base["record_id"], mode)
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
        "components": provenance,
    }
    Path(args.provenance).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: manifest[key] for key in manifest if key != "components"}))


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
        family = ACTIVE_FAMILY[mode]
        family_deltas = {
            label: deltas[label]
            for label, key in METRICS
            if key.startswith(family + "/")
        }
        summary[mode] = {
            "metrics": {label: values[key] for label, key in METRICS},
            "deltas": deltas,
            "all_ten_nonlower": all(value >= -1e-12 for value in deltas.values()),
            "strict_improvements": sum(value > 1e-12 for value in deltas.values()),
            "changed_family": family,
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
        "All 10 nonlower",
        "Strict +",
        "Family worst Δ",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    base_cells = ["Baseline"] + [
        f"{baseline[key]:.4f}" for _, key in METRICS
    ] + ["—", "—", "—"]
    lines.append("| " + " | ".join(base_cells) + " |")
    for mode, row in summary.items():
        cells = [mode] + [
            f"{row['metrics'][label]:.4f} ({row['deltas'][label]:+.4f})"
            for label, _ in METRICS
        ] + [
            str(row["all_ten_nonlower"]),
            str(row["strict_improvements"]),
            f"{row['changed_family_worst_delta']:+.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(summary)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--baseline-predictions", required=True)
    prepare_parser.add_argument("--raw-predictions", required=True)
    prepare_parser.add_argument("--split-manifest", required=True)
    prepare_parser.add_argument("--split-key", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--provenance", required=True)
    prepare_parser.set_defaults(function=prepare)

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
