"""Select the Phase-II checkpoint by full validation teacher-forced loss."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import read_jsonl


def summarize(path: str) -> tuple[dict, set[tuple[str, str]]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"empty predictions: {path}")
    if any("loss" not in row for row in rows):
        raise ValueError(f"prediction without loss: {path}")
    keys = {(row["record_id"], row["mode"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError(f"duplicate prediction key: {path}")
    losses = [float(row["loss"]) for row in rows]
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(float(row["loss"]))
    return {
        "predictions": path,
        "rows": len(rows),
        "mean_loss": sum(losses) / len(losses),
        "median_loss": statistics.median(losses),
        "mean_loss_by_question_type": {
            key: sum(values) / len(values)
            for key, values in sorted(by_type.items())
        },
    }, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = []
    expected_keys = None
    for path in args.predictions:
        summary, keys = summarize(path)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"validation key mismatch: {path}")
        candidates.append(summary)

    best = min(candidates, key=lambda row: row["mean_loss"])
    payload = {
        "selection_metric": "minimum full-validation mean teacher-forced loss",
        "candidates": candidates,
        "best": best,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(best))


if __name__ == "__main__":
    main()
