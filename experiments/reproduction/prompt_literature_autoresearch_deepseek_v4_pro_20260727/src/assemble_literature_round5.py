"""Prepare and assemble preregistered Literature Round-5 components."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.models.AXIS.prompt_stage_a import _has_explicit_tf_negative_cue
from tools.axis_repro.common import read_jsonl


BASE = "base"
TF_SOURCE = "lit_r4_01_tf_neg_re2"
MC_STRUCTURED = "lit_r5_01_mc_structured_guard"
MC_STATUS = "lit_r5_02_mc_status_then_shape"
ROUTES = (
    TF_SOURCE,
    MC_STRUCTURED,
    MC_STATUS,
    "lit_r5_03_joint_structured_tf_neg_re2",
    "lit_r5_04_joint_status_tf_neg_re2",
)


def source_mode(route: str, row: dict) -> str:
    question_type = row["question_type"]
    if route not in ROUTES:
        raise ValueError(f"Unknown route: {route}")
    if question_type == "open_ended":
        return BASE
    if question_type == "multiple_choice":
        if route in {MC_STRUCTURED, "lit_r5_03_joint_structured_tf_neg_re2"}:
            return MC_STRUCTURED
        if route in {MC_STATUS, "lit_r5_04_joint_status_tf_neg_re2"}:
            return MC_STATUS
        return BASE
    if question_type == "true_false":
        joint_or_tf = route in {
            TF_SOURCE,
            "lit_r5_03_joint_structured_tf_neg_re2",
            "lit_r5_04_joint_status_tf_neg_re2",
        }
        if joint_or_tf and _has_explicit_tf_negative_cue(row["question"]):
            return TF_SOURCE
        return BASE
    raise ValueError(f"Unknown question type: {question_type}")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


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


def prepare(args: argparse.Namespace) -> None:
    baseline = [
        row for row in read_jsonl(args.baseline_predictions)
        if row["mode"] == BASE
    ]
    if len(baseline) != args.expected_records:
        raise RuntimeError(
            f"Expected {args.expected_records} Baseline records, found {len(baseline)}"
        )
    raw = read_jsonl(args.raw_predictions)
    raw_index = unique_index(raw)
    selected = []
    requirements = []
    for row in baseline:
        for source in (MC_STRUCTURED, MC_STATUS, TF_SOURCE):
            required = (
                row["question_type"] == "multiple_choice"
                if source in {MC_STRUCTURED, MC_STATUS}
                else row["question_type"] == "true_false"
                and _has_explicit_tf_negative_cue(row["question"])
            )
            if not required:
                continue
            key = (row["record_id"], source)
            if key not in raw_index:
                raise RuntimeError(f"Missing raw component: {key}")
            selected.append(raw_index[key])
            requirements.append({
                "record_id": row["record_id"],
                "question_type": row["question_type"],
                "source_mode": source,
                "prompt_sha256": raw_index[key].get("prompt_sha256"),
            })
    selected.sort(key=lambda row: (row["record_id"], row["mode"]))
    write_jsonl(args.output_predictions, selected)
    Path(args.provenance).write_text(
        json.dumps({
            "baseline_records": len(baseline),
            "component_rows": len(selected),
            "requirements": requirements,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "baseline_records": len(baseline),
        "component_rows": len(selected),
    }))


def assemble(args: argparse.Namespace) -> None:
    baseline_predictions = [
        row for row in read_jsonl(args.baseline_predictions)
        if row["mode"] == BASE
    ]
    baseline_scores = [
        row for row in read_jsonl(args.baseline_scores)
        if row["mode"] == BASE
    ]
    component_predictions = read_jsonl(args.component_predictions)
    component_scores = read_jsonl(args.component_scores)
    pred_index = unique_index(baseline_predictions + component_predictions)
    score_index = unique_index(baseline_scores + component_scores, scores=True)
    base_rows = sorted(baseline_predictions, key=lambda row: row["record_id"])
    requested = tuple(args.routes or ROUTES)
    unknown = set(requested) - set(ROUTES)
    if unknown:
        raise ValueError(f"Unknown routes: {sorted(unknown)}")

    assembled_predictions = []
    assembled_scores = []
    provenance = []
    for route in requested:
        for base_row in base_rows:
            source = source_mode(route, base_row)
            key = (base_row["record_id"], source)
            if key not in pred_index:
                raise RuntimeError(f"Missing prediction component: {key}")
            source_prediction = pred_index[key]
            candidate = copy.deepcopy(source_prediction)
            candidate["mode"] = route
            candidate["component_source_mode"] = source
            assembled_predictions.append(candidate)
            expected_dims = sorted(
                dimension
                for record_id, mode, dimension in score_index
                if record_id == base_row["record_id"] and mode == BASE
            )
            source_scores = [
                score_index[(base_row["record_id"], source, dimension)]
                for dimension in expected_dims
                if (base_row["record_id"], source, dimension) in score_index
            ]
            if len(source_scores) != len(expected_dims):
                raise RuntimeError(
                    f"Missing score component(s) for {key}: "
                    f"{len(source_scores)}/{len(expected_dims)}"
                )
            for source_score in source_scores:
                score = copy.deepcopy(source_score)
                score["mode"] = route
                score["component_source_mode"] = source
                assembled_scores.append(score)
            provenance.append({
                "record_id": base_row["record_id"],
                "route": route,
                "question_type": base_row["question_type"],
                "source_mode": source,
                "prompt_sha256": source_prediction.get("prompt_sha256"),
            })

    if len(unique_index(assembled_predictions)) != len(assembled_predictions):
        raise RuntimeError("Duplicate assembled prediction key")
    if len(unique_index(assembled_scores, scores=True)) != len(assembled_scores):
        raise RuntimeError("Duplicate assembled score key")
    write_jsonl(args.output_predictions, assembled_predictions)
    write_jsonl(args.output_scores, assembled_scores)
    Path(args.provenance).write_text(
        json.dumps({
            "routes": list(requested),
            "records": len(base_rows),
            "prediction_rows": len(assembled_predictions),
            "score_rows": len(assembled_scores),
            "components": provenance,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "routes": list(requested),
        "records": len(base_rows),
        "prediction_rows": len(assembled_predictions),
        "score_rows": len(assembled_scores),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--baseline-predictions", required=True)
    prepare_parser.add_argument("--raw-predictions", required=True)
    prepare_parser.add_argument("--expected-records", type=int, required=True)
    prepare_parser.add_argument("--output-predictions", required=True)
    prepare_parser.add_argument("--provenance", required=True)
    prepare_parser.set_defaults(func=prepare)

    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--baseline-predictions", required=True)
    assemble_parser.add_argument("--baseline-scores", required=True)
    assemble_parser.add_argument("--component-predictions", required=True)
    assemble_parser.add_argument("--component-scores", required=True)
    assemble_parser.add_argument("--output-predictions", required=True)
    assemble_parser.add_argument("--output-scores", required=True)
    assemble_parser.add_argument("--provenance", required=True)
    assemble_parser.add_argument("--routes", nargs="+")
    assemble_parser.set_defaults(func=assemble)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
