"""Paired comparison of two AXIS G-Eval runs with judge-noise controls."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl


def _prediction_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["record_id"]), str(row.get("mode", "base"))


def _score_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (*_prediction_key(row), str(row["dimension"]))


def _unique(rows: list[dict[str, Any]], key_fn, label: str) -> dict[tuple, dict[str, Any]]:
    result = {}
    for row in rows:
        key = key_fn(row)
        if key in result:
            raise ValueError(f"duplicate {label} key {key!r}")
        result[key] = row
    return result


def _bootstrap_delta(values: list[float], samples: int, seed: int) -> dict[str, float] | None:
    if not values:
        return None
    mean = float(statistics.mean(values))
    if len(values) == 1 or samples <= 0:
        return {"mean": mean, "low": mean, "high": mean}
    rng = random.Random(seed)
    means = sorted(
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "low": float(means[int(0.025 * (samples - 1))]),
        "high": float(means[int(0.975 * (samples - 1))]),
    }


def _summarize(pairs: list[tuple[dict[str, Any], dict[str, Any]]], samples: int, seed: int) -> dict[str, Any]:
    left = [float(item[0]["score"]) for item in pairs]
    right = [float(item[1]["score"]) for item in pairs]
    deltas = [b - a for a, b in zip(left, right)]
    return {
        "count": len(pairs),
        "left_mean": None if not left else float(statistics.mean(left)),
        "right_mean": None if not right else float(statistics.mean(right)),
        "delta_right_minus_left": _bootstrap_delta(deltas, samples, seed),
        "mean_absolute_score_delta": None if not deltas else float(statistics.mean(abs(x) for x in deltas)),
        "exact_score_count": sum(a == b for a, b in zip(left, right)),
    }


def compare_geval_runs(
    left_scores_path: str | Path,
    right_scores_path: str | Path,
    left_predictions_path: str | Path,
    right_predictions_path: str | Path,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 72,
) -> dict[str, Any]:
    left_scores = _unique(read_jsonl(left_scores_path), _score_key, "left score")
    right_scores = _unique(read_jsonl(right_scores_path), _score_key, "right score")
    if left_scores.keys() != right_scores.keys():
        raise ValueError("score key mismatch")
    left_predictions = _unique(read_jsonl(left_predictions_path), _prediction_key, "left prediction")
    right_predictions = _unique(read_jsonl(right_predictions_path), _prediction_key, "right prediction")
    if left_predictions.keys() != right_predictions.keys():
        raise ValueError("prediction key mismatch")

    buckets: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    changed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unchanged: list[tuple[dict[str, Any], dict[str, Any]]] = []
    all_pairs = []
    unchanged_prompt_hash_equal = 0
    for key in sorted(left_scores):
        pair = (left_scores[key], right_scores[key])
        all_pairs.append(pair)
        buckets[key[2]].append(pair)
        prediction_key = key[:2]
        is_changed = (
            left_predictions[prediction_key].get("response", "")
            != right_predictions[prediction_key].get("response", "")
        )
        (changed if is_changed else unchanged).append(pair)
        if not is_changed and pair[0].get("prompt_sha256") == pair[1].get("prompt_sha256"):
            unchanged_prompt_hash_equal += 1

    return {
        "left_scores": str(Path(left_scores_path)),
        "right_scores": str(Path(right_scores_path)),
        "overall": _summarize(all_pairs, bootstrap_samples, seed),
        "changed_response_tasks": _summarize(changed, bootstrap_samples, seed + 1),
        "unchanged_response_tasks": _summarize(unchanged, bootstrap_samples, seed + 2),
        "unchanged_prompt_hash_equal": unchanged_prompt_hash_equal,
        "per_dimension": {
            name: _summarize(items, bootstrap_samples, seed + index + 10)
            for index, (name, items) in enumerate(sorted(buckets.items()))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-scores", required=True)
    parser.add_argument("--right-scores", required=True)
    parser.add_argument("--left-predictions", required=True)
    parser.add_argument("--right-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args()
    report = compare_geval_runs(
        args.left_scores, args.right_scores,
        args.left_predictions, args.right_predictions,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()