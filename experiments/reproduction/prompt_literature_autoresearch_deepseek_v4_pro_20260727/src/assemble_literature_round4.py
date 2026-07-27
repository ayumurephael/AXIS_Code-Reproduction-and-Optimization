"""Assemble preregistered Literature Round-4 lexical-router routes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from src.models.AXIS.prompt_stage_a import (
    _has_explicit_tf_negative_cue,
    _has_tf_nonanomaly_cue,
)
from tools.axis_repro.common import read_jsonl


REUSE_ROUTES = (
    "lit_r4_01_tf_neg_re2",
    "lit_r4_02_joint_semantic_tf_neg_re2",
    "lit_r4_03_tf_nonanomaly_re2",
    "lit_r4_04_joint_semantic_tf_nonanomaly_re2",
)
PREFIX_ROUTES = (
    "lit_r4_05_tf_neg_prefix",
    "lit_r4_06_joint_semantic_tf_neg_prefix",
)
BASE = "base"
MC_SEMANTIC = "lit_r3_02_mc_semantic_qual"
TF_RE2 = "lit_r3_05_tf_re2_qual_verdict"
TF_PREFIX = "lit_r4_05_tf_neg_prefix"


def source_mode(route: str, row: dict) -> str:
    question_type = row["question_type"]
    question = row["question"]
    if question_type == "multiple_choice":
        if route in {
            "lit_r4_02_joint_semantic_tf_neg_re2",
            "lit_r4_04_joint_semantic_tf_nonanomaly_re2",
            "lit_r4_06_joint_semantic_tf_neg_prefix",
        }:
            return MC_SEMANTIC
        return BASE
    if question_type == "open_ended":
        return BASE
    if route in {
        "lit_r4_01_tf_neg_re2",
        "lit_r4_02_joint_semantic_tf_neg_re2",
    }:
        return TF_RE2 if _has_explicit_tf_negative_cue(question) else BASE
    if route in {
        "lit_r4_03_tf_nonanomaly_re2",
        "lit_r4_04_joint_semantic_tf_nonanomaly_re2",
    }:
        return TF_RE2 if _has_tf_nonanomaly_cue(question) else BASE
    if route in PREFIX_ROUTES:
        return TF_PREFIX if _has_explicit_tf_negative_cue(question) else BASE
    raise ValueError(f"Unknown route: {route}")


def index_unique(rows: list[dict], dimensions: bool = False) -> dict:
    def key(row: dict):
        base = (row["record_id"], row["mode"])
        return base + (row["dimension"],) if dimensions else base

    result = {}
    for row in rows:
        item_key = key(row)
        if item_key in result:
            raise RuntimeError(f"Duplicate source key: {item_key}")
        result[item_key] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3-predictions", required=True)
    parser.add_argument("--r3-scores", required=True)
    parser.add_argument("--prefix-predictions")
    parser.add_argument("--prefix-scores")
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-scores", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--routes", nargs="+")
    args = parser.parse_args()

    predictions = read_jsonl(args.r3_predictions)
    scores = read_jsonl(args.r3_scores)
    requested_routes = args.routes
    known_routes = set(REUSE_ROUTES) | set(PREFIX_ROUTES)
    if requested_routes:
        unknown_routes = set(requested_routes) - known_routes
        if unknown_routes:
            raise ValueError(f"Unknown routes: {sorted(unknown_routes)}")
    routes = [
        route for route in REUSE_ROUTES
        if requested_routes is None or route in requested_routes
    ]
    if bool(args.prefix_predictions) != bool(args.prefix_scores):
        raise ValueError("Prefix predictions and scores must be supplied together")
    if args.prefix_predictions:
        predictions.extend(read_jsonl(args.prefix_predictions))
        scores.extend(read_jsonl(args.prefix_scores))
        routes.extend(
            route for route in PREFIX_ROUTES
            if requested_routes is None or route in requested_routes
        )
    elif requested_routes and set(requested_routes) & set(PREFIX_ROUTES):
        raise ValueError("Requested prefix routes require prefix inputs")

    pred_index = index_unique(predictions)
    score_index = index_unique(scores, dimensions=True)
    base_rows = sorted(
        (row for row in predictions if row["mode"] == BASE),
        key=lambda row: row["record_id"],
    )
    if not base_rows:
        raise RuntimeError("No Baseline rows found")

    assembled_predictions = []
    assembled_scores = []
    provenance = []
    for route in routes:
        for base_row in base_rows:
            source = source_mode(route, base_row)
            pred_key = (base_row["record_id"], source)
            if pred_key not in pred_index:
                raise RuntimeError(f"Missing prediction component: {pred_key}")
            source_prediction = pred_index[pred_key]
            candidate = copy.deepcopy(source_prediction)
            candidate["mode"] = route
            assembled_predictions.append(candidate)
            provenance.append({
                "record_id": base_row["record_id"],
                "route": route,
                "question_type": base_row["question_type"],
                "source_mode": source,
                "prompt_sha256": source_prediction.get("prompt_sha256"),
            })
            source_scores = [
                row for (record_id, mode, _), row in score_index.items()
                if record_id == base_row["record_id"] and mode == source
            ]
            if not source_scores:
                raise RuntimeError(f"Missing score component: {pred_key}")
            for source_score in source_scores:
                score = copy.deepcopy(source_score)
                score["mode"] = route
                assembled_scores.append(score)

    prediction_keys = {
        (row["record_id"], row["mode"]) for row in assembled_predictions
    }
    score_keys = {
        (row["record_id"], row["mode"], row["dimension"])
        for row in assembled_scores
    }
    if len(prediction_keys) != len(assembled_predictions):
        raise RuntimeError("Duplicate assembled prediction key")
    if len(score_keys) != len(assembled_scores):
        raise RuntimeError("Duplicate assembled score key")

    def write_jsonl(path: str, rows: list[dict]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    write_jsonl(args.output_predictions, assembled_predictions)
    write_jsonl(args.output_scores, assembled_scores)
    provenance_path = Path(args.provenance)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps({
            "routes": routes,
            "records": len(base_rows),
            "prediction_rows": len(assembled_predictions),
            "score_rows": len(assembled_scores),
            "components": provenance,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "routes": routes,
        "records": len(base_rows),
        "prediction_rows": len(assembled_predictions),
        "score_rows": len(assembled_scores),
    }))


if __name__ == "__main__":
    main()
