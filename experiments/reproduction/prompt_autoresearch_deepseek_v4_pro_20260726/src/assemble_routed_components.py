"""Assemble question-type-routed candidates from audited component artifacts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


ROUTES = {
    "route_r3_01_mc_f0_safe": {
        "multiple_choice": "route_r2_01_minimal",
        "true_false": "base",
        "open_ended": "base",
    },
    "route_r3_02_mc_f1_safe": {
        "multiple_choice": "route_r2_02_mc_stable",
        "true_false": "base",
        "open_ended": "base",
    },
    "route_r3_03_mc_f0_tf_p0": {
        "multiple_choice": "route_r2_01_minimal",
        "true_false": "route_r2_01_minimal",
        "open_ended": "base",
    },
    "route_r3_04_mc_f1_tf_p0": {
        "multiple_choice": "route_r2_02_mc_stable",
        "true_false": "route_r2_01_minimal",
        "open_ended": "base",
    },
    "route_r3_05_oe_q1": {
        "multiple_choice": "route_r2_01_minimal",
        "true_false": "route_r2_01_minimal",
        "open_ended": "route_r3_05_oe_q1",
    },
    "route_r3_06_oe_q2": {
        "multiple_choice": "route_r2_01_minimal",
        "true_false": "route_r2_01_minimal",
        "open_ended": "route_r3_06_oe_q2",
    },
    "route_r3_07_oe_q3": {
        "multiple_choice": "route_r2_01_minimal",
        "true_false": "route_r2_01_minimal",
        "open_ended": "route_r3_07_oe_q3",
    },
    "route_r3_08_historical": {
        "multiple_choice": "route_r2_01_minimal",
        "true_false": "base",
        "open_ended": "route_r2_04_oe_old_contract",
    },
}


def _unique(rows: list[dict], fields: tuple[str, ...], name: str) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise RuntimeError(f"Duplicate {name} key: {key}")
        result[key] = row
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--existing-predictions", required=True)
    parser.add_argument("--existing-scores", required=True)
    parser.add_argument("--new-predictions", required=True)
    parser.add_argument("--new-scores", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    record_ids = manifest.get("record_ids")
    if not isinstance(record_ids, list) or not record_ids:
        raise RuntimeError("Manifest must contain a non-empty record_ids list")

    existing_predictions = read_jsonl(args.existing_predictions)
    existing_scores = read_jsonl(args.existing_scores)
    new_predictions = read_jsonl(args.new_predictions)
    new_scores = read_jsonl(args.new_scores)
    all_predictions = existing_predictions + new_predictions
    all_scores = existing_scores + new_scores
    prediction_by_key = _unique(
        all_predictions, ("record_id", "mode"), "prediction"
    )
    score_by_key = _unique(
        all_scores, ("record_id", "mode", "dimension"), "score"
    )

    baseline_rows = []
    baseline_scores = []
    qtypes = {}
    for record_id in record_ids:
        key = (record_id, "base")
        if key not in prediction_by_key:
            raise RuntimeError(f"Missing Baseline prediction: {key}")
        row = copy.deepcopy(prediction_by_key[key])
        baseline_rows.append(row)
        qtypes[record_id] = row["question_type"]
        for dimension in DIMS[row["question_type"]]:
            score_key = (record_id, "base", dimension)
            if score_key not in score_by_key:
                raise RuntimeError(f"Missing Baseline score: {score_key}")
            baseline_scores.append(copy.deepcopy(score_by_key[score_key]))

    candidate_rows = []
    candidate_scores = []
    component_counts = {}
    for target_mode, route in ROUTES.items():
        counts = {}
        for record_id in record_ids:
            qtype = qtypes[record_id]
            source_mode = route[qtype]
            source_key = (record_id, source_mode)
            if source_key not in prediction_by_key:
                raise RuntimeError(f"Missing component prediction: {source_key}")
            prediction = copy.deepcopy(prediction_by_key[source_key])
            if prediction["question_type"] != qtype:
                raise RuntimeError(
                    f"Question-type mismatch for {source_key}: "
                    f"{prediction['question_type']} != {qtype}"
                )
            prediction["mode"] = target_mode
            candidate_rows.append(prediction)
            counts[source_mode] = counts.get(source_mode, 0) + 1

            for dimension in DIMS[qtype]:
                score_key = (record_id, source_mode, dimension)
                if score_key not in score_by_key:
                    raise RuntimeError(f"Missing component score: {score_key}")
                score = copy.deepcopy(score_by_key[score_key])
                score["mode"] = target_mode
                candidate_scores.append(score)
        component_counts[target_mode] = counts

    predictions = baseline_rows + candidate_rows
    scores = baseline_scores + candidate_scores
    expected_predictions = len(record_ids) * (1 + len(ROUTES))
    expected_scores = len(baseline_scores) * (1 + len(ROUTES))
    if len(predictions) != expected_predictions:
        raise RuntimeError(
            f"Prediction count {len(predictions)} != {expected_predictions}"
        )
    if len(scores) != expected_scores:
        raise RuntimeError(f"Score count {len(scores)} != {expected_scores}")
    _unique(predictions, ("record_id", "mode"), "assembled prediction")
    _unique(scores, ("record_id", "mode", "dimension"), "assembled score")

    _write_jsonl(Path(args.output_predictions), predictions)
    _write_jsonl(Path(args.output_scores), scores)
    provenance = {
        "record_count": len(record_ids),
        "candidate_count": len(ROUTES),
        "prediction_rows": len(predictions),
        "score_rows": len(scores),
        "routes": ROUTES,
        "component_counts": component_counts,
        "new_component_modes": sorted(
            {row["mode"] for row in new_predictions}
        ),
    }
    Path(args.provenance).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
