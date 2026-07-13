"""Fail-closed integrity audit for AXIS prediction and G-Eval JSONL files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .build_tables import DIMS
from .common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--modes", nargs="+",
        default=["base", "wo_fixed_hint", "wo_local_hint", "wo_windows"],
    )
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    errors = []
    prediction_keys = [(x["record_id"], x["mode"]) for x in predictions]
    if len(prediction_keys) != len(set(prediction_keys)):
        errors.append("duplicate prediction (record_id, mode) keys")
    if any(not str(x.get("response", "")).strip() for x in predictions):
        errors.append("empty generated response")

    record_ids = sorted({x["record_id"] for x in predictions})
    if args.manifest:
        expected_ids = set(json.loads(Path(args.manifest).read_text(encoding="utf-8"))["record_ids"])
        if set(record_ids) != expected_ids:
            errors.append(
                f"record-id mismatch: got={len(record_ids)} expected={len(expected_ids)}"
            )
    expected_prediction_keys = {
        (record_id, mode) for record_id in record_ids for mode in args.modes
    }
    missing_predictions = expected_prediction_keys - set(prediction_keys)
    unexpected_predictions = set(prediction_keys) - expected_prediction_keys
    if missing_predictions:
        errors.append(f"missing predictions: {len(missing_predictions)}")
    if unexpected_predictions:
        errors.append(f"unexpected predictions: {len(unexpected_predictions)}")

    summary = {
        "prediction_rows": len(predictions),
        "records": len(record_ids),
        "prediction_modes": dict(Counter(x["mode"] for x in predictions)),
    }

    if args.scores:
        scores = read_jsonl(args.scores)
        score_keys = [(x["record_id"], x["mode"], x["dimension"]) for x in scores]
        if len(score_keys) != len(set(score_keys)):
            errors.append("duplicate score (record_id, mode, dimension) keys")
        qtypes = {(x["record_id"], x["mode"]): x["question_type"] for x in predictions}
        expected_score_keys = {
            (record_id, mode, dimension)
            for record_id, mode in expected_prediction_keys
            for dimension in DIMS[qtypes[(record_id, mode)]]
        }
        missing_scores = expected_score_keys - set(score_keys)
        unexpected_scores = set(score_keys) - expected_score_keys
        if missing_scores:
            errors.append(f"missing scores: {len(missing_scores)}")
        if unexpected_scores:
            errors.append(f"unexpected scores: {len(unexpected_scores)}")
        for row in scores:
            method = str(row.get("method", ""))
            if method.startswith("exact_sample_mean_") and len(row.get("fallback_scores", [])) != 20:
                errors.append(
                    f"non-20 fallback at {row['record_id']} {row['mode']} {row['dimension']}"
                )
                break
        summary.update({
            "score_rows": len(scores),
            "score_methods": dict(Counter(x["method"] for x in scores)),
        })

    summary["errors"] = errors
    summary["ok"] = not errors
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
