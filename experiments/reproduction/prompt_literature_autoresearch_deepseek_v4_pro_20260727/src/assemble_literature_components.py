"""Assemble literature routes from byte-identical, audited task components."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.models.AXIS.prompt_stage_a import (
    LITERATURE_R1_PROFILES,
    LITERATURE_R2_PROFILES,
    LITERATURE_R3_PROFILES,
    build_question_prompt,
)
from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


QUESTION_TYPES = ("multiple_choice", "open_ended", "true_false")
QUESTIONS = {
    "multiple_choice": (
        "Which description best matches the window?\n"
        "A) Stable\nB) Spike\nC) Trend\nD) Cyclic"
    ),
    "open_ended": "What evidence supports or refutes an anomaly in this window?",
    "true_false": "True or False: The window contains an anomalous deviation.",
}

COMPONENT_SOURCES = {
    "base": {
        "multiple_choice": "base",
        "open_ended": "base",
        "true_false": "base",
    },
    "mc_semantic_bind": {
        "multiple_choice": "lit_r1_01_mc_semantic_bind",
    },
    "mc_pointwise": {
        "multiple_choice": "lit_r1_02_mc_pointwise",
    },
    "re2": {
        "multiple_choice": "lit_r1_03_mc_re2",
        "true_false": "lit_r1_06_tf_re2",
        "open_ended": "lit_r1_08_oe_re2",
    },
    "tf_minimal": {
        "true_false": "lit_r1_04_tf_minimal",
    },
    "tf_clause": {
        "true_false": "lit_r1_05_tf_clause",
    },
    "oe_direct": {
        "open_ended": "lit_r1_07_oe_direct",
    },
    "mc_decoupled": {
        "multiple_choice": "lit_r1_11_decoupled",
    },
    "tf_decoupled": {
        "true_false": "lit_r1_11_decoupled",
    },
    "oe_decoupled": {
        "open_ended": "lit_r1_11_decoupled",
    },
    "mc_semantic_bind_re2": {
        "multiple_choice": "lit_r1_12_mc_bind_re2",
    },
    "mc_semantic_qual": {
        "multiple_choice": "lit_r3_02_mc_semantic_qual",
    },
    "mc_pointwise_qual": {
        "multiple_choice": "lit_r3_01_mc_pointwise_qual",
    },
    "mc_salient_aftermath": {
        "multiple_choice": "lit_r3_03_mc_salient_aftermath",
    },
    "tf_re2_verdict": {
        "true_false": "lit_r3_04_tf_re2_verdict",
    },
    "tf_re2_qual_verdict": {
        "true_false": "lit_r3_05_tf_re2_qual_verdict",
    },
}

ALL_LITERATURE_PROFILES = {
    **LITERATURE_R1_PROFILES,
    **LITERATURE_R2_PROFILES,
    **LITERATURE_R3_PROFILES,
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def representative_prompt(mode: str, question_type: str) -> str:
    return build_question_prompt(
        question=QUESTIONS[question_type],
        question_type=question_type,
        start=7,
        end=10,
        serialized_values="10, 11, 42",
        local_hint_tokens="<|local_hint|>" * 3,
        fixed_hint_tokens="<|fixed_hint|>" * 30,
        mode=mode,
        aligned_rows=None,
    )


def prompt_sha(mode: str, question_type: str) -> str:
    return hashlib.sha256(
        representative_prompt(mode, question_type).encode("utf-8")
    ).hexdigest()


def source_mode(component: str, question_type: str) -> str:
    try:
        return COMPONENT_SOURCES[component][question_type]
    except KeyError as exc:
        raise RuntimeError(
            f"No source registered for {component!r}/{question_type!r}"
        ) from exc


def required_source_pairs(profiles: dict) -> set[tuple[str, str]]:
    pairs = set()
    for profile in profiles.values():
        for question_type, component in profile.items():
            source = source_mode(component, question_type)
            if source != "base":
                pairs.add((source, question_type))
    return pairs


def prepare(args: argparse.Namespace) -> None:
    baseline_rows = [
        row for row in read_jsonl(args.baseline_predictions)
        if row["mode"] == "base"
    ]
    candidate_rows = read_jsonl(args.candidate_predictions)
    baseline = {row["record_id"]: row for row in baseline_rows}
    candidate = {
        (row["record_id"], row["mode"]): row for row in candidate_rows
    }
    record_ids = sorted(baseline)
    if len(record_ids) != args.expected_records:
        raise RuntimeError(
            f"Expected {args.expected_records} Baseline records, "
            f"found {len(record_ids)}"
        )
    if len(candidate) != len(candidate_rows):
        raise RuntimeError("Candidate prediction keys are not unique")
    routes = args.routes or list(LITERATURE_R1_PROFILES)
    try:
        profiles = {route: ALL_LITERATURE_PROFILES[route] for route in routes}
    except KeyError as exc:
        raise RuntimeError(f"Unknown route {exc.args[0]!r}") from exc

    component_rows = []
    component_provenance = {}
    for source, question_type in sorted(required_source_pairs(profiles)):
        selected = [
            row for row in candidate_rows
            if row["mode"] == source and row["question_type"] == question_type
        ]
        expected = sum(
            row["question_type"] == question_type for row in baseline_rows
        )
        if len(selected) != expected:
            raise RuntimeError(
                f"Source component {source}/{question_type} has "
                f"{len(selected)} rows; expected {expected}"
            )
        component_rows.extend(selected)
        component_provenance[f"{source}/{question_type}"] = {
            "rows": len(selected),
            "representative_prompt_sha256": prompt_sha(source, question_type),
        }

    assembled = []
    route_provenance = {}
    for route, profile in profiles.items():
        route_provenance[route] = {}
        for record_id in record_ids:
            question_type = baseline[record_id]["question_type"]
            component = profile[question_type]
            source = source_mode(component, question_type)
            source_row = (
                baseline[record_id]
                if source == "base"
                else candidate[(record_id, source)]
            )
            if source_row["question_type"] != question_type:
                raise RuntimeError("Question-type mismatch in component source")
            route_hash = prompt_sha(route, question_type)
            source_hash = prompt_sha(source, question_type)
            if route_hash != source_hash:
                raise RuntimeError(
                    f"Prompt mismatch: {route}/{question_type} != "
                    f"{source}/{question_type}"
                )
            output = copy.deepcopy(source_row)
            output["mode"] = route
            output["component"] = component
            output["component_source_mode"] = source
            output["component_prompt_sha256"] = source_hash
            assembled.append(output)
            route_provenance[route][question_type] = {
                "component": component,
                "source_mode": source,
                "representative_prompt_sha256": source_hash,
            }

    component_rows.sort(key=lambda row: (row["record_id"], row["mode"]))
    assembled.sort(key=lambda row: (row["record_id"], row["mode"]))
    expected_component_rows = sum(
        sum(row["question_type"] == question_type for row in baseline_rows)
        for _, question_type in required_source_pairs(profiles)
    )
    expected_assembled_rows = len(record_ids) * len(profiles)
    if len(component_rows) != expected_component_rows:
        raise RuntimeError(
            f"Expected {expected_component_rows} unique component rows, "
            f"found {len(component_rows)}"
        )
    if len(assembled) != expected_assembled_rows:
        raise RuntimeError(
            f"Expected {expected_assembled_rows} assembled rows, "
            f"found {len(assembled)}"
        )
    if len({(row["record_id"], row["mode"]) for row in assembled}) != len(assembled):
        raise RuntimeError("Duplicate assembled prediction key")

    write_jsonl(Path(args.component_output), component_rows)
    write_jsonl(Path(args.assembled_output), assembled)
    provenance = {
        "baseline_rows": len(baseline_rows),
        "raw_candidate_rows": len(candidate_rows),
        "unique_new_component_rows": len(component_rows),
        "assembled_route_rows": len(assembled),
        "component_question_types": dict(
            Counter(row["question_type"] for row in component_rows)
        ),
        "component_modes": dict(Counter(row["mode"] for row in component_rows)),
        "component_provenance": component_provenance,
        "route_provenance": route_provenance,
    }
    Path(args.provenance).write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "component_rows": len(component_rows),
        "assembled_rows": len(assembled),
    }))


def filter_scores(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.component_predictions)
    qtypes = {
        (row["record_id"], row["mode"]): row["question_type"]
        for row in predictions
    }
    allowed = {
        (record_id, mode, dimension)
        for (record_id, mode), question_type in qtypes.items()
        for dimension in DIMS[question_type]
    }
    by_key = {}
    for row in read_jsonl(args.existing_scores):
        key = (row["record_id"], row["mode"], row["dimension"])
        if key in allowed:
            by_key[key] = row
    rows = [by_key[key] for key in sorted(by_key)]
    write_jsonl(Path(args.output), rows)
    print(json.dumps({
        "reusable_scores": len(rows),
        "expected_component_scores": len(allowed),
    }))


def assemble_scores(args: argparse.Namespace) -> None:
    predictions = read_jsonl(args.assembled_predictions)
    baseline_scores = {
        (row["record_id"], row["mode"], row["dimension"]): row
        for row in read_jsonl(args.baseline_scores)
        if row["mode"] == "base"
    }
    component_scores = {
        (row["record_id"], row["mode"], row["dimension"]): row
        for row in read_jsonl(args.component_scores)
    }
    rows = []
    for prediction in predictions:
        source = prediction["component_source_mode"]
        question_type = prediction["question_type"]
        source_scores = baseline_scores if source == "base" else component_scores
        for dimension in DIMS[question_type]:
            key = (prediction["record_id"], source, dimension)
            if key not in source_scores:
                raise RuntimeError(f"Missing source score {key}")
            row = copy.deepcopy(source_scores[key])
            row["mode"] = prediction["mode"]
            row["component"] = prediction["component"]
            row["component_source_mode"] = source
            rows.append(row)
    rows.sort(key=lambda row: (row["record_id"], row["mode"], row["dimension"]))
    expected_rows = sum(len(DIMS[row["question_type"]]) for row in predictions)
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} assembled scores, found {len(rows)}")
    if len({
        (row["record_id"], row["mode"], row["dimension"]) for row in rows
    }) != len(rows):
        raise RuntimeError("Duplicate assembled score key")
    write_jsonl(Path(args.output), rows)
    print(json.dumps({"assembled_scores": len(rows)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--baseline-predictions", required=True)
    prepare_parser.add_argument("--candidate-predictions", required=True)
    prepare_parser.add_argument("--component-output", required=True)
    prepare_parser.add_argument("--assembled-output", required=True)
    prepare_parser.add_argument("--provenance", required=True)
    prepare_parser.add_argument("--routes", nargs="+")
    prepare_parser.add_argument("--expected-records", type=int, default=24)
    prepare_parser.set_defaults(function=prepare)

    filter_parser = subparsers.add_parser("filter-scores")
    filter_parser.add_argument("--component-predictions", required=True)
    filter_parser.add_argument("--existing-scores", required=True)
    filter_parser.add_argument("--output", required=True)
    filter_parser.set_defaults(function=filter_scores)

    score_parser = subparsers.add_parser("assemble-scores")
    score_parser.add_argument("--assembled-predictions", required=True)
    score_parser.add_argument("--baseline-scores", required=True)
    score_parser.add_argument("--component-scores", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.set_defaults(function=assemble_scores)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
