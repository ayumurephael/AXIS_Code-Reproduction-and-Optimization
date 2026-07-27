"""Assemble the two frozen full-284 candidates from exact prompt components."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.AXIS.prompt_stage_a import (  # noqa: E402
    LITERATURE_R1_PROFILES,
    build_question_prompt,
)
from tools.axis_repro.build_tables import DIMS  # noqa: E402
from tools.axis_repro.common import read_jsonl  # noqa: E402


BASE = "base"
OE_ONLY = "lit_r1_08_oe_re2"
TRIPLET = "lit_r1_10_triplet_re2"
TARGET_MODES = (BASE, OE_ONLY, TRIPLET)
QUESTION_TYPES = ("multiple_choice", "open_ended", "true_false")
SAMPLE_QUESTIONS = {
    "multiple_choice": (
        "Which option best describes the window?\n\n"
        "A) Normal.\n\nB) Anomalous."
    ),
    "open_ended": "What evidence supports or challenges this assessment?",
    "true_false": "True or False: There is no anomaly in the window.",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def response_sha256(row: dict) -> str:
    return hashlib.sha256(row["response"].encode("utf-8")).hexdigest()


def representative_prompt(mode: str, question_type: str) -> str:
    return build_question_prompt(
        question=SAMPLE_QUESTIONS[question_type],
        question_type=question_type,
        start=10,
        end=13,
        serialized_values="100, 110, 90",
        local_hint_tokens="<|local_hint|>" * 3,
        fixed_hint_tokens="<|fixed_hint|>" * 30,
        mode=mode,
        aligned_rows=(
            "Step 0010 | value=+00100 | context=<|local_hint|>\n"
            "Step 0011 | value=+00110 | context=<|local_hint|>\n"
            "Step 0012 | value=+00090 | context=<|local_hint|>"
        ),
    )


def assert_component_identity() -> dict[str, dict[str, str]]:
    expected = {
        OE_ONLY: {
            "multiple_choice": BASE,
            "open_ended": TRIPLET,
            "true_false": BASE,
        },
        TRIPLET: {question_type: TRIPLET for question_type in QUESTION_TYPES},
    }
    hashes: dict[str, dict[str, str]] = {}
    for target, routes in expected.items():
        hashes[target] = {}
        for question_type, source in routes.items():
            target_component = LITERATURE_R1_PROFILES[target][question_type]
            if source == BASE:
                source_component = "base"
            else:
                source_component = LITERATURE_R1_PROFILES[source][question_type]
            if target_component != source_component:
                raise RuntimeError(
                    f"Component mismatch: {target}/{question_type} "
                    f"({target_component}) != {source} ({source_component})"
                )
            target_prompt = representative_prompt(target, question_type)
            source_prompt = representative_prompt(source, question_type)
            if target_prompt != source_prompt:
                raise RuntimeError(
                    f"Prompt mismatch: {target}/{question_type} != {source}"
                )
            hashes[target][question_type] = hashlib.sha256(
                target_prompt.encode("utf-8")
            ).hexdigest()
    return hashes


def source_mode(target_mode: str, question_type: str) -> str:
    if target_mode == BASE:
        return BASE
    if target_mode == TRIPLET:
        return TRIPLET
    if target_mode == OE_ONLY:
        return TRIPLET if question_type == "open_ended" else BASE
    raise ValueError(target_mode)


def assemble_predictions(args: argparse.Namespace) -> None:
    prompt_hashes = assert_component_identity()
    raw_rows = read_jsonl(args.raw_predictions)
    raw = {(row["record_id"], row["mode"]): row for row in raw_rows}
    if len(raw) != len(raw_rows):
        raise RuntimeError("Duplicate raw prediction keys")
    record_ids = sorted(
        record_id for record_id, mode in raw if mode == BASE
    )
    if len(record_ids) != args.expected_records:
        raise RuntimeError(
            f"Expected {args.expected_records} Baseline records, "
            f"found {len(record_ids)}"
        )
    expected_raw = {
        (record_id, mode)
        for record_id in record_ids
        for mode in (BASE, TRIPLET)
    }
    if set(raw) != expected_raw:
        raise RuntimeError(
            f"Raw key mismatch: missing={len(expected_raw - set(raw))}, "
            f"unexpected={len(set(raw) - expected_raw)}"
        )

    assembled: list[dict] = []
    provenance: list[dict] = []
    for record_id in record_ids:
        baseline = raw[(record_id, BASE)]
        question_type = baseline["question_type"]
        if question_type not in QUESTION_TYPES:
            raise RuntimeError(f"Unknown question type {question_type!r}")
        for target_mode in TARGET_MODES:
            source = source_mode(target_mode, question_type)
            source_row = raw[(record_id, source)]
            if source_row["question_type"] != question_type:
                raise RuntimeError("Question-type mismatch")
            output = copy.deepcopy(source_row)
            output["mode"] = target_mode
            output["component_source_mode"] = source
            output["component_response_sha256"] = response_sha256(source_row)
            if target_mode != BASE:
                output["component_prompt_sha256"] = prompt_hashes[target_mode][
                    question_type
                ]
            assembled.append(output)
            provenance.append(
                {
                    "record_id": record_id,
                    "question_type": question_type,
                    "target_mode": target_mode,
                    "source_mode": source,
                    "response_sha256": response_sha256(source_row),
                    "reuse": target_mode != source,
                }
            )

    assembled.sort(key=lambda row: (row["record_id"], row["mode"]))
    provenance.sort(
        key=lambda row: (row["record_id"], row["target_mode"])
    )
    expected_rows = args.expected_records * len(TARGET_MODES)
    if len(assembled) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} assembled rows, found {len(assembled)}"
        )
    write_jsonl(Path(args.output), assembled)
    Path(args.provenance).write_text(
        json.dumps(
            {
                "raw_rows": len(raw_rows),
                "assembled_rows": len(assembled),
                "mode_counts": dict(Counter(row["mode"] for row in assembled)),
                "question_type_counts": dict(
                    Counter(row["question_type"] for row in assembled)
                ),
                "prompt_hashes": prompt_hashes,
                "routes": provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_rows": len(raw_rows),
                "assembled_rows": len(assembled),
                "reused_rows": sum(row["reuse"] for row in provenance),
            }
        )
    )


def assemble_scores(args: argparse.Namespace) -> None:
    assert_component_identity()
    predictions = read_jsonl(args.assembled_predictions)
    raw_rows = read_jsonl(args.raw_scores)
    raw = {
        (row["record_id"], row["mode"], row["dimension"]): row
        for row in raw_rows
    }
    if len(raw) != len(raw_rows):
        raise RuntimeError("Duplicate raw score keys")

    assembled: list[dict] = []
    provenance: list[dict] = []
    for prediction in predictions:
        record_id = prediction["record_id"]
        target_mode = prediction["mode"]
        question_type = prediction["question_type"]
        source = source_mode(target_mode, question_type)
        for dimension in DIMS[question_type]:
            key = (record_id, source, dimension)
            try:
                source_row = raw[key]
            except KeyError as exc:
                raise RuntimeError(f"Missing source score {key}") from exc
            output = copy.deepcopy(source_row)
            output["mode"] = target_mode
            output["component_source_mode"] = source
            assembled.append(output)
            provenance.append(
                {
                    "record_id": record_id,
                    "question_type": question_type,
                    "dimension": dimension,
                    "target_mode": target_mode,
                    "source_mode": source,
                    "reuse": target_mode != source,
                }
            )

    assembled.sort(
        key=lambda row: (row["record_id"], row["mode"], row["dimension"])
    )
    expected = sum(
        len(DIMS[row["question_type"]]) for row in predictions
    )
    if len(assembled) != expected:
        raise RuntimeError(
            f"Expected {expected} assembled scores, found {len(assembled)}"
        )
    if len(
        {
            (row["record_id"], row["mode"], row["dimension"])
            for row in assembled
        }
    ) != len(assembled):
        raise RuntimeError("Duplicate assembled score keys")
    write_jsonl(Path(args.output), assembled)
    Path(args.provenance).write_text(
        json.dumps(
            {
                "raw_score_rows": len(raw_rows),
                "assembled_score_rows": len(assembled),
                "mode_counts": dict(Counter(row["mode"] for row in assembled)),
                "routes": provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_score_rows": len(raw_rows),
                "assembled_score_rows": len(assembled),
                "reused_score_rows": sum(row["reuse"] for row in provenance),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prediction_parser = subparsers.add_parser("predictions")
    prediction_parser.add_argument("--raw-predictions", required=True)
    prediction_parser.add_argument("--output", required=True)
    prediction_parser.add_argument("--provenance", required=True)
    prediction_parser.add_argument("--expected-records", type=int, default=284)
    prediction_parser.set_defaults(func=assemble_predictions)

    score_parser = subparsers.add_parser("scores")
    score_parser.add_argument("--assembled-predictions", required=True)
    score_parser.add_argument("--raw-scores", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--provenance", required=True)
    score_parser.set_defaults(func=assemble_scores)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

