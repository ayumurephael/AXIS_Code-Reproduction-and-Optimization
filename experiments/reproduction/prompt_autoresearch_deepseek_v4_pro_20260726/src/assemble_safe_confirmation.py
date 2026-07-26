"""Assemble conservative MC-only routes for holdout and final confirmation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl

ALLOWED_MODES = {
    "route_r3_01_mc_f0_safe",
    "route_r3_02_mc_f1_safe",
}


def _index(rows: list[dict], fields: tuple[str, ...], name: str) -> dict:
    indexed = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in indexed:
            raise RuntimeError(f"Duplicate {name} key: {key}")
        indexed[key] = row
    return indexed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--candidate-modes", nargs="+", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()

    candidate_modes = tuple(args.candidate_modes)
    unsupported = set(candidate_modes) - ALLOWED_MODES
    if unsupported:
        raise RuntimeError(f"Unsupported safe candidate modes: {unsupported}")
    if len(set(candidate_modes)) != len(candidate_modes):
        raise RuntimeError("Duplicate candidate mode")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    record_ids = manifest.get("record_ids")
    if not isinstance(record_ids, list) or not record_ids:
        raise RuntimeError("Manifest must contain record_ids")

    predictions = [
        row for path in args.predictions for row in read_jsonl(path)
    ]
    scores = [row for path in args.scores for row in read_jsonl(path)]
    prediction_by_key = _index(
        predictions, ("record_id", "mode"), "prediction"
    )
    score_by_key = _index(
        scores, ("record_id", "mode", "dimension"), "score"
    )

    assembled_predictions = []
    assembled_scores = []
    source_counts = {}
    for target_mode in ("base",) + candidate_modes:
        counts = {}
        for record_id in record_ids:
            baseline_key = (record_id, "base")
            if baseline_key not in prediction_by_key:
                raise RuntimeError(f"Missing Baseline prediction: {baseline_key}")
            qtype = prediction_by_key[baseline_key]["question_type"]
            source_mode = (
                target_mode
                if target_mode != "base" and qtype == "multiple_choice"
                else "base"
            )
            source_key = (record_id, source_mode)
            if source_key not in prediction_by_key:
                raise RuntimeError(f"Missing component prediction: {source_key}")
            prediction = copy.deepcopy(prediction_by_key[source_key])
            prediction["mode"] = target_mode
            assembled_predictions.append(prediction)
            counts[source_mode] = counts.get(source_mode, 0) + 1

            for dimension in DIMS[qtype]:
                score_key = (record_id, source_mode, dimension)
                if score_key not in score_by_key:
                    raise RuntimeError(f"Missing component score: {score_key}")
                score = copy.deepcopy(score_by_key[score_key])
                score["mode"] = target_mode
                assembled_scores.append(score)
        source_counts[target_mode] = counts

    expected_predictions = len(record_ids) * (1 + len(candidate_modes))
    baseline_dimensions = sum(
        len(DIMS[prediction_by_key[(record_id, "base")]["question_type"]])
        for record_id in record_ids
    )
    expected_scores = baseline_dimensions * (1 + len(candidate_modes))
    if len(assembled_predictions) != expected_predictions:
        raise RuntimeError("Assembled prediction count mismatch")
    if len(assembled_scores) != expected_scores:
        raise RuntimeError("Assembled score count mismatch")
    _index(
        assembled_predictions, ("record_id", "mode"), "assembled prediction"
    )
    _index(
        assembled_scores,
        ("record_id", "mode", "dimension"),
        "assembled score",
    )

    _write_jsonl(Path(args.output_predictions), assembled_predictions)
    _write_jsonl(Path(args.output_scores), assembled_scores)
    provenance = {
        "records": len(record_ids),
        "candidate_modes": candidate_modes,
        "prediction_rows": len(assembled_predictions),
        "score_rows": len(assembled_scores),
        "source_counts": source_counts,
    }
    Path(args.provenance).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False))


if __name__ == "__main__":
    main()
