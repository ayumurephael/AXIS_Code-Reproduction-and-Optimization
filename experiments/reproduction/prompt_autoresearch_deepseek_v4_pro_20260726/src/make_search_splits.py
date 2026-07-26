"""Create deterministic series-disjoint AXIS prompt-search manifests."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from tools.axis_repro.common import load_axis_records

SPLIT_SIZES = {
    "screening": 12,
    "validation": 36,
    "holdout": 24,
}


def label_name(value: object) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "none"


def record_features(record) -> tuple[str, ...]:
    label = label_name(record.has_anomaly)
    return (
        f"qtype::{record.question_type}",
        f"label::{label}",
        f"joint::{record.question_type}::{label}",
    )


def count_features(series_names, by_series):
    counts = collections.Counter()
    for name in series_names:
        for record in by_series[name]:
            counts.update(record_features(record))
    return counts


def score_assignment(groups, by_series, total_counts):
    score = 0.0
    total_series = sum(SPLIT_SIZES.values())
    for split_name, names in groups.items():
        observed = count_features(names, by_series)
        fraction = SPLIT_SIZES[split_name] / total_series
        for feature, total in total_counts.items():
            expected = total * fraction
            score += ((observed[feature] - expected) ** 2) / max(expected, 1.0)
        qtype_counts = [
            observed[f"qtype::{qtype}"]
            for qtype in ("multiple_choice", "open_ended", "true_false")
        ]
        if min(qtype_counts) == 0:
            score += 1_000_000
    return score


def split_permutation(permutation):
    cursor = 0
    groups = {}
    for split_name in ("screening", "validation", "holdout"):
        size = SPLIT_SIZES[split_name]
        groups[split_name] = tuple(permutation[cursor : cursor + size])
        cursor += size
    return groups


def sha256_json(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def summarize(split_name, names, by_series):
    records = [record for name in names for record in by_series[name]]
    qtypes = collections.Counter(record.question_type for record in records)
    labels = collections.Counter(label_name(record.has_anomaly) for record in records)
    joints = collections.Counter(
        f"{record.question_type}::{label_name(record.has_anomaly)}"
        for record in records
    )
    record_ids = sorted(record.record_id for record in records)
    series_files = sorted(names)
    return {
        "name": split_name,
        "series_count": len(series_files),
        "record_count": len(record_ids),
        "series_files": series_files,
        "record_ids": record_ids,
        "question_type_counts": dict(sorted(qtypes.items())),
        "anomaly_label_counts": dict(sorted(labels.items())),
        "question_type_label_counts": dict(sorted(joints.items())),
        "series_sha256": sha256_json(series_files),
        "record_ids_sha256": sha256_json(record_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/AXIS_qa_test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--trials", type=int, default=100_000)
    args = parser.parse_args()

    full = load_axis_records(args.data, subset="full")
    paper = load_axis_records(args.data, subset="paper140")
    paper_series = {record.series_file for record in paper}
    pool = [record for record in full if record.series_file not in paper_series]

    by_series = collections.defaultdict(list)
    for record in pool:
        by_series[record.series_file].append(record)
    if len(full) != 284 or len(paper) != 140 or len(pool) != 144:
        raise RuntimeError(
            f"Unexpected dataset counts full={len(full)} paper={len(paper)} "
            f"pool={len(pool)}"
        )
    if len(by_series) != 72 or any(len(rows) != 2 for rows in by_series.values()):
        raise RuntimeError("Expected 72 search-pool series with exactly two QA each")

    names = sorted(by_series)
    total_counts = count_features(names, by_series)
    rng = random.Random(args.seed)
    best_score = float("inf")
    best_groups = None
    candidate = names[:]
    for _ in range(args.trials):
        rng.shuffle(candidate)
        groups = split_permutation(candidate)
        value = score_assignment(groups, by_series, total_counts)
        if value < best_score:
            best_score = value
            best_groups = {key: tuple(value) for key, value in groups.items()}
    if best_groups is None:
        raise RuntimeError("No split assignment was generated")

    summaries = {
        name: summarize(name, split_names, by_series)
        for name, split_names in best_groups.items()
    }
    all_selected = set().union(
        *(set(summary["series_files"]) for summary in summaries.values())
    )
    if len(all_selected) != 72 or all_selected & paper_series:
        raise RuntimeError("Series isolation audit failed")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    combined = {
        "seed": args.seed,
        "trials": args.trials,
        "objective": best_score,
        "data": str(Path(args.data).resolve()),
        "full_record_count": len(full),
        "paper140_record_count": len(paper),
        "paper140_series_count": len(paper_series),
        "search_pool_record_count": len(pool),
        "search_pool_series_count": len(by_series),
        "paper140_series_sha256": sha256_json(sorted(paper_series)),
        "screening_series": summaries["screening"]["series_files"],
        "validation_series": summaries["validation"]["series_files"],
        "holdout_series": summaries["holdout"]["series_files"],
        "splits": summaries,
        "disjoint_ok": True,
    }
    (output / "search_splits.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for name, summary in summaries.items():
        (output / f"{name}.json").write_text(
            json.dumps(
                {
                    "subset": "full",
                    "record_count": summary["record_count"],
                    "record_ids": summary["record_ids"],
                    "series_files": summary["series_files"],
                    "record_ids_sha256": summary["record_ids_sha256"],
                    "series_sha256": summary["series_sha256"],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(combined, ensure_ascii=False))


if __name__ == "__main__":
    main()