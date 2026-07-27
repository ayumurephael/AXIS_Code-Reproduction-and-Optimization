"""Assemble conservative OE repairs for malformed answer boundaries."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src import (
    round1_pipeline,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_pipeline import (
    allowed_series,
    baseline_rows,
    unique_index,
    write_jsonl,
)
from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


ROUND5_SOURCE_MODE = "v2_r5_02_oe_verbatim_or_add"
ROUND7_MODE = "v2_r7_01_oe_unclosed_think_answer_repair"
ROUND8_MODE = "v2_r8_01_oe_no_anomaly_boundary_repair"
BOUNDARY_REPAIR_MODES = (ROUND7_MODE, ROUND8_MODE)
ANSWER_MARKER = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:final\s+)?answer\s*:",
    flags=re.IGNORECASE | re.MULTILINE,
)
NO_ANOMALY_CONCLUSION = re.compile(
    r"\b(?:no|without)\b[^.\n]{0,80}\b"
    r"(?:anomal(?:y|ies|ous)?|irregularit(?:y|ies))\b",
    flags=re.IGNORECASE,
)
round1_pipeline.ACTIVE_FAMILY = {
    mode: "open_ended" for mode in BOUNDARY_REPAIR_MODES
}


def select_repaired_answer(
    baseline_response: str,
    candidate_response: str,
    require_no_anomaly: bool = False,
) -> bool:
    """Use the revision only for an objectively malformed Baseline answer."""
    baseline_lower = baseline_response.lower()
    candidate_lower = candidate_response.lower()
    baseline_is_unclosed_reasoning = (
        "<think>" in baseline_lower
        and "</think>" not in baseline_lower
        and ANSWER_MARKER.search(baseline_response) is None
    )
    candidate_has_clean_answer_boundary = (
        "<think>" not in candidate_lower
        and "</think>" not in candidate_lower
        and ANSWER_MARKER.search(candidate_response) is not None
    )
    if not (baseline_is_unclosed_reasoning and candidate_has_clean_answer_boundary):
        return False
    if require_no_anomaly:
        return (
            NO_ANOMALY_CONCLUSION.search(baseline_response) is not None
            and NO_ANOMALY_CONCLUSION.search(candidate_response) is not None
        )
    return True


def selected_for_mode(
    mode: str,
    baseline_response: str,
    candidate_response: str,
) -> bool:
    if mode == ROUND7_MODE:
        return select_repaired_answer(baseline_response, candidate_response)
    if mode == ROUND8_MODE:
        return select_repaired_answer(
            baseline_response,
            candidate_response,
            require_no_anomaly=True,
        )
    raise ValueError(f"Unknown boundary-repair mode: {mode!r}")


def analyze(args: argparse.Namespace) -> None:
    round1_pipeline.analyze(args)


def assemble(args: argparse.Namespace) -> None:
    modes = tuple(args.modes or BOUNDARY_REPAIR_MODES)
    unknown = set(modes) - set(BOUNDARY_REPAIR_MODES)
    if unknown:
        raise ValueError(f"Unknown boundary-repair modes: {sorted(unknown)}")
    allowed = allowed_series(args.split_manifest, args.split_key)
    base_predictions = baseline_rows(args.baseline_predictions, allowed)
    record_ids = {row["record_id"] for row in base_predictions}
    base_scores = [
        row
        for row in read_jsonl(args.baseline_scores)
        if row["mode"] == "base" and row["record_id"] in record_ids
    ]
    candidate_predictions = [
        row
        for row in read_jsonl(args.candidate_predictions)
        if row["mode"] == ROUND5_SOURCE_MODE and row["record_id"] in record_ids
    ]
    candidate_scores = [
        row
        for row in read_jsonl(args.candidate_scores)
        if row["mode"] == ROUND5_SOURCE_MODE and row["record_id"] in record_ids
    ]
    base_pred_index = unique_index(base_predictions)
    base_score_index = unique_index(base_scores, scores=True)
    candidate_pred_index = unique_index(candidate_predictions)
    candidate_score_index = unique_index(candidate_scores, scores=True)

    output_predictions = [copy.deepcopy(row) for row in base_predictions]
    output_scores = [copy.deepcopy(row) for row in base_scores]
    provenance = []
    selected_counts = {mode: 0 for mode in modes}
    for mode in modes:
        for base in base_predictions:
            selected = False
            source_mode = "base"
            source_prediction = base_pred_index[(base["record_id"], "base")]
            score_source = base_score_index
            if base["question_type"] == "open_ended":
                candidate = candidate_pred_index[
                    (base["record_id"], ROUND5_SOURCE_MODE)
                ]
                selected = selected_for_mode(
                    mode,
                    base["response"],
                    candidate["response"],
                )
                if selected:
                    selected_counts[mode] += 1
                    source_mode = ROUND5_SOURCE_MODE
                    source_prediction = candidate
                    score_source = candidate_score_index
            prediction = copy.deepcopy(source_prediction)
            prediction["mode"] = mode
            prediction["component_source_mode"] = source_mode
            prediction["boundary_repair_selected"] = selected
            output_predictions.append(prediction)
            for dimension in DIMS[base["question_type"]]:
                score = copy.deepcopy(
                    score_source[(base["record_id"], source_mode, dimension)]
                )
                score["mode"] = mode
                score["component_source_mode"] = source_mode
                output_scores.append(score)
            provenance.append(
                {
                    "record_id": base["record_id"],
                    "mode": mode,
                    "question_type": base["question_type"],
                    "selected": selected,
                    "component_source_mode": source_mode,
                }
            )

    output_predictions.sort(key=lambda row: (row["record_id"], row["mode"]))
    output_scores.sort(
        key=lambda row: (row["record_id"], row["mode"], row["dimension"])
    )
    write_jsonl(args.output_predictions, output_predictions)
    write_jsonl(args.output_scores, output_scores)
    manifest = {
        "split_key": args.split_key,
        "modes": list(modes),
        "records_per_mode": len(base_predictions),
        "selected_repairs": selected_counts,
        "prediction_rows": len(output_predictions),
        "score_rows": len(output_scores),
        "components": provenance,
    }
    Path(args.provenance).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "components"}
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--baseline-predictions", required=True)
    assemble_parser.add_argument("--baseline-scores", required=True)
    assemble_parser.add_argument("--candidate-predictions", required=True)
    assemble_parser.add_argument("--candidate-scores", required=True)
    assemble_parser.add_argument("--split-manifest", required=True)
    assemble_parser.add_argument("--split-key", required=True)
    assemble_parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(BOUNDARY_REPAIR_MODES),
    )
    assemble_parser.add_argument("--output-predictions", required=True)
    assemble_parser.add_argument("--output-scores", required=True)
    assemble_parser.add_argument("--provenance", required=True)
    assemble_parser.set_defaults(function=assemble)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--scores", required=True)
    analyze_parser.add_argument("--output-json", required=True)
    analyze_parser.add_argument("--output-md", required=True)
    analyze_parser.set_defaults(function=analyze)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()