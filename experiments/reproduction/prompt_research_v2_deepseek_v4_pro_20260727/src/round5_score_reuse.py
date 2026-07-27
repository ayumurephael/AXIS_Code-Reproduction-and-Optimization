"""Reuse canonical Baseline scores for unchanged Round-5 responses."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


def normalized_response(value: str) -> str:
    return value.strip()


def unique_index(rows: list[dict], fields: tuple[str, ...]) -> dict:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise RuntimeError(f"Duplicate key: {key}")
        result[key] = row
    return result


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--baseline-scores", required=True)
    parser.add_argument("--component-predictions", required=True)
    parser.add_argument("--component-scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()

    baseline_predictions = [
        row
        for row in read_jsonl(args.baseline_predictions)
        if row["mode"] == "base"
    ]
    baseline_scores = [
        row
        for row in read_jsonl(args.baseline_scores)
        if row["mode"] == "base"
    ]
    component_predictions = read_jsonl(args.component_predictions)
    component_scores = read_jsonl(args.component_scores)

    base_pred_index = unique_index(
        baseline_predictions,
        ("record_id",),
    )
    base_score_index = unique_index(
        baseline_scores,
        ("record_id", "dimension"),
    )
    component_score_index = unique_index(
        component_scores,
        ("record_id", "mode", "dimension"),
    )

    output = []
    reused_responses = 0
    judged_responses = 0
    reused_dimensions = 0
    judged_dimensions = 0
    reused_keys = []
    for prediction in component_predictions:
        record_id = prediction["record_id"]
        baseline_key = (record_id,)
        mode = prediction["mode"]
        if baseline_key not in base_pred_index:
            raise RuntimeError(f"Missing Baseline prediction: {record_id}")
        baseline = base_pred_index[baseline_key]
        reuse = normalized_response(prediction["response"]) == (
            normalized_response(baseline["response"])
        )
        if reuse:
            reused_responses += 1
            reused_keys.append({"record_id": record_id, "mode": mode})
        else:
            judged_responses += 1
        for dimension in DIMS[prediction["question_type"]]:
            if reuse:
                source = base_score_index[(record_id, dimension)]
                reused_dimensions += 1
            else:
                source = component_score_index[
                    (record_id, mode, dimension)
                ]
                judged_dimensions += 1
            row = copy.deepcopy(source)
            row["mode"] = mode
            row["component_score_source_mode"] = "base" if reuse else mode
            row["exact_response_score_reuse"] = reuse
            output.append(row)

    expected_dimensions = sum(
        len(DIMS[row["question_type"]]) for row in component_predictions
    )
    if len(output) != expected_dimensions:
        raise RuntimeError(
            f"Score count mismatch: {len(output)} != {expected_dimensions}"
        )
    output.sort(
        key=lambda row: (row["record_id"], row["mode"], row["dimension"])
    )
    write_jsonl(args.output, output)
    provenance = {
        "normalization": "str.strip",
        "component_responses": len(component_predictions),
        "reused_responses": reused_responses,
        "newly_judged_responses": judged_responses,
        "score_dimensions": len(output),
        "reused_dimensions": reused_dimensions,
        "newly_judged_dimensions": judged_dimensions,
        "reused_keys": reused_keys,
    }
    Path(args.provenance).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in provenance.items() if key != "reused_keys"}))


if __name__ == "__main__":
    main()
