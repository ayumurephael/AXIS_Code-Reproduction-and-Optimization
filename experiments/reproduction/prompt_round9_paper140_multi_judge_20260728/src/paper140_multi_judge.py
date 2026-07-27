"""Prepare and expand the locked Round-9 paper140 multi-Judge evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable

from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


MODES = (
    "base",
    "prompt_final_round9_mc_oe",
    "prompt_final_round9_tf_oe",
    "prompt_final_round9_joint",
)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def response_sha256(row: dict) -> str:
    return hashlib.sha256(row["response"].encode("utf-8")).hexdigest()


def load_manifest_ids(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record_ids = payload["record_ids"]
    if payload.get("record_count") != 140 or len(record_ids) != 140:
        raise RuntimeError("The locked paper140 manifest must contain 140 records")
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError("Duplicate record IDs in paper140 manifest")
    return record_ids


def prepare(args: argparse.Namespace) -> None:
    record_ids = load_manifest_ids(args.manifest)
    allowed_ids = set(record_ids)
    source_rows = read_jsonl(args.source)
    index: dict[tuple[str, str], dict] = {}
    for row in source_rows:
        key = (row["record_id"], row["mode"])
        if row["record_id"] not in allowed_ids or row["mode"] not in MODES:
            continue
        if key in index:
            raise RuntimeError(f"Duplicate source prediction: {key}")
        index[key] = row

    expected_keys = {(record_id, mode) for record_id in record_ids for mode in MODES}
    if set(index) != expected_keys:
        missing = sorted(expected_keys - set(index))
        extra = sorted(set(index) - expected_keys)
        raise RuntimeError(f"Prediction key mismatch: missing={missing[:5]}, extra={extra[:5]}")

    all_rows: list[dict] = []
    unique_rows: list[dict] = []
    mapping: list[dict] = []
    changed_by_mode = {mode: 0 for mode in MODES}
    question_type_counts: dict[str, int] = {}
    for record_id in record_ids:
        base = index[(record_id, "base")]
        question_type_counts[base["question_type"]] = (
            question_type_counts.get(base["question_type"], 0) + 1
        )
        canonical_by_response: dict[str, str] = {}
        for mode in MODES:
            row = copy.deepcopy(index[(record_id, mode)])
            digest = response_sha256(row)
            all_rows.append(row)
            if row["response"] != base["response"]:
                changed_by_mode[mode] += 1
            canonical_mode = canonical_by_response.get(digest)
            if canonical_mode is None:
                canonical_mode = mode
                canonical_by_response[digest] = canonical_mode
                unique = copy.deepcopy(row)
                unique["response_sha256"] = digest
                unique["canonical_judge_mode"] = canonical_mode
                unique_rows.append(unique)
            mapping.append(
                {
                    "record_id": record_id,
                    "target_mode": mode,
                    "canonical_judge_mode": canonical_mode,
                    "response_sha256": digest,
                    "exact_response_score_reuse": mode != canonical_mode,
                }
            )

    expected_dimensions = sum(
        len(DIMS[row["question_type"]]) for row in unique_rows
    )
    manifest_payload = {
        "source_predictions": str(Path(args.source)),
        "paper140_manifest": str(Path(args.manifest)),
        "modes": list(MODES),
        "records_per_mode": len(record_ids),
        "all_prediction_rows": len(all_rows),
        "unique_prediction_rows": len(unique_rows),
        "unique_expected_score_dimensions": expected_dimensions,
        "expanded_expected_score_dimensions": 335 * len(MODES),
        "question_type_counts": question_type_counts,
        "changed_responses_by_mode": changed_by_mode,
        "mapping": mapping,
    }
    if len(all_rows) != 560:
        raise RuntimeError(f"Expected 560 expanded predictions, got {len(all_rows)}")

    write_jsonl(args.output_all, all_rows)
    write_jsonl(args.output_unique, unique_rows)
    Path(args.output_mapping).write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in manifest_payload.items() if key != "mapping"},
            ensure_ascii=False,
        )
    )


def expand_scores(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.predictions)
    mapping_payload = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    mapping_index = {
        (row["record_id"], row["target_mode"]): row
        for row in mapping_payload["mapping"]
    }
    raw_scores = read_jsonl(args.raw_scores)
    raw_index: dict[tuple[str, str, str], dict] = {}
    for row in raw_scores:
        key = (row["record_id"], row["mode"], row["dimension"])
        if key in raw_index:
            raise RuntimeError(f"Duplicate raw score: {key}")
        raw_index[key] = row

    output: list[dict] = []
    used_raw_keys: set[tuple[str, str, str]] = set()
    direct = 0
    reused = 0
    for prediction in predictions:
        map_key = (prediction["record_id"], prediction["mode"])
        route = mapping_index[map_key]
        if response_sha256(prediction) != route["response_sha256"]:
            raise RuntimeError(f"Response hash mismatch for {map_key}")
        for dimension in DIMS[prediction["question_type"]]:
            raw_key = (
                prediction["record_id"],
                route["canonical_judge_mode"],
                dimension,
            )
            source = raw_index.get(raw_key)
            if source is None:
                raise RuntimeError(f"Missing raw Judge score: {raw_key}")
            used_raw_keys.add(raw_key)
            row = copy.deepcopy(source)
            row["mode"] = prediction["mode"]
            row["canonical_judge_mode"] = route["canonical_judge_mode"]
            row["response_sha256"] = route["response_sha256"]
            row["exact_response_score_reuse"] = route[
                "exact_response_score_reuse"
            ]
            output.append(row)
            direct += int(not row["exact_response_score_reuse"])
            reused += int(row["exact_response_score_reuse"])

    if set(raw_index) != used_raw_keys:
        unused = sorted(set(raw_index) - used_raw_keys)
        raise RuntimeError(f"Raw score file has unused keys: {unused[:5]}")
    if len(output) != 1340:
        raise RuntimeError(f"Expected 1,340 expanded scores, got {len(output)}")
    output.sort(key=lambda row: (row["record_id"], row["mode"], row["dimension"]))
    write_jsonl(args.output, output)
    print(
        json.dumps(
            {
                "raw_score_dimensions": len(raw_scores),
                "expanded_score_dimensions": len(output),
                "direct_dimensions": direct,
                "exact_reuse_dimensions": reused,
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--source", required=True)
    prepare_parser.add_argument("--manifest", required=True)
    prepare_parser.add_argument("--output-all", required=True)
    prepare_parser.add_argument("--output-unique", required=True)
    prepare_parser.add_argument("--output-mapping", required=True)
    prepare_parser.set_defaults(function=prepare)

    expand_parser = subparsers.add_parser("expand-scores")
    expand_parser.add_argument("--predictions", required=True)
    expand_parser.add_argument("--mapping", required=True)
    expand_parser.add_argument("--raw-scores", required=True)
    expand_parser.add_argument("--output", required=True)
    expand_parser.set_defaults(function=expand_scores)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

