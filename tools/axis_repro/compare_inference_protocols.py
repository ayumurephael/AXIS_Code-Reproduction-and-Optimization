"""Compare two AXIS prediction files under different inference protocols."""
from __future__ import annotations

import argparse
import difflib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_jsonl, score_prediction


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["record_id"]), str(row.get("mode", "base"))


def _unique_rows(path: str | Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = _key(row)
        if key in result:
            raise ValueError(f"duplicate prediction key {key!r} in {path}")
        result[key] = row
    return result


def _mean(values: list[float]) -> float | None:
    return None if not values else float(statistics.mean(values))


def _summarize_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    similarities = [
        difflib.SequenceMatcher(None, left.get("response", ""), right.get("response", "")).ratio()
        for left, right in pairs
    ]
    left_scores = [score_prediction(row).get("primary_score") for row, _ in pairs]
    right_scores = [score_prediction(row).get("primary_score") for _, row in pairs]
    left_clean = [float(value) for value in left_scores if value is not None]
    right_clean = [float(value) for value in right_scores if value is not None]
    return {
        "count": len(pairs),
        "exact_response_count": sum(
            left.get("response", "") == right.get("response", "")
            for left, right in pairs
        ),
        "changed_response_count": sum(
            left.get("response", "") != right.get("response", "")
            for left, right in pairs
        ),
        "mean_text_similarity": _mean(similarities),
        "median_text_similarity": None if not similarities else float(statistics.median(similarities)),
        "left_mean_response_characters": _mean([float(len(left.get("response", ""))) for left, _ in pairs]),
        "right_mean_response_characters": _mean([float(len(right.get("response", ""))) for _, right in pairs]),
        "left_mean_proxy_score": _mean(left_clean),
        "right_mean_proxy_score": _mean(right_clean),
        "proxy_score_delta_right_minus_left": (
            None if not left_clean or not right_clean
            else float(statistics.mean(right_clean) - statistics.mean(left_clean))
        ),
    }


def compare_predictions(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left = _unique_rows(left_path)
    right = _unique_rows(right_path)
    if left.keys() != right.keys():
        missing_right = sorted(set(left) - set(right))
        missing_left = sorted(set(right) - set(left))
        raise ValueError(
            f"prediction key mismatch: missing_right={missing_right[:5]!r}, "
            f"missing_left={missing_left[:5]!r}"
        )
    pairs = [(left[key], right[key]) for key in sorted(left)]
    by_type: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        left_type = str(pair[0].get("question_type", "unknown"))
        right_type = str(pair[1].get("question_type", "unknown"))
        if left_type != right_type:
            raise ValueError(f"question type mismatch for {_key(pair[0])!r}")
        by_type[left_type].append(pair)
    return {
        "left": str(Path(left_path)),
        "right": str(Path(right_path)),
        "overall": _summarize_pairs(pairs),
        "per_question_type": {
            question_type: _summarize_pairs(items)
            for question_type, items in sorted(by_type.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare_predictions(args.left, args.right)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()