"""Assemble Round-6 selector routes from exact source components."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src import (
    round1_pipeline,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round1_pipeline import (
    analyze,
    allowed_series,
    baseline_rows,
    unique_index,
    write_jsonl,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.run_round6_selector import (
    ROUND5_SOURCE_MODE,
    ROUND6_MODES,
)
from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


round1_pipeline.ACTIVE_FAMILY = {
    mode: "open_ended" for mode in ROUND6_MODES
}


def assemble(args: argparse.Namespace) -> None:
    modes = tuple(args.modes or ROUND6_MODES)
    unknown = set(modes) - set(ROUND6_MODES)
    if unknown:
        raise ValueError(f"Unknown Round-6 modes: {sorted(unknown)}")
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
    selector_predictions = [
        row
        for row in read_jsonl(args.selector_predictions)
        if row["mode"] in modes and row.get("selected_component")
    ]

    base_score_index = unique_index(base_scores, scores=True)
    candidate_pred_index = unique_index(candidate_predictions)
    candidate_score_index = unique_index(candidate_scores, scores=True)
    selector_index = unique_index(selector_predictions)
    base_pred_index = unique_index(base_predictions)

    output_predictions = [copy.deepcopy(row) for row in base_predictions]
    output_scores = [copy.deepcopy(row) for row in base_scores]
    provenance = []
    for mode in modes:
        for base in base_predictions:
            if base["question_type"] == "open_ended":
                selector = selector_index[(base["record_id"], mode)]
                source_mode = selector["selected_source_mode"]
                if source_mode == "base":
                    source_prediction = base_pred_index[
                        (base["record_id"], "base")
                    ]
                    score_source = base_score_index
                elif source_mode == ROUND5_SOURCE_MODE:
                    source_prediction = candidate_pred_index[
                        (base["record_id"], ROUND5_SOURCE_MODE)
                    ]
                    score_source = candidate_score_index
                else:
                    raise RuntimeError(f"Unknown source mode: {source_mode}")
            else:
                selector = None
                source_mode = "base"
                source_prediction = base_pred_index[
                    (base["record_id"], "base")
                ]
                score_source = base_score_index
            prediction = copy.deepcopy(source_prediction)
            prediction["mode"] = mode
            prediction["component_source_mode"] = source_mode
            if selector is not None:
                prediction["selector_response"] = selector["selector_response"]
                prediction["selector_choice"] = selector["selector_choice"]
                prediction["selector_parse_valid"] = selector[
                    "selector_parse_valid"
                ]
            output_predictions.append(prediction)
            for dimension in DIMS[base["question_type"]]:
                key = (base["record_id"], source_mode, dimension)
                score = copy.deepcopy(score_source[key])
                score["mode"] = mode
                score["component_source_mode"] = source_mode
                output_scores.append(score)
            provenance.append(
                {
                    "record_id": base["record_id"],
                    "mode": mode,
                    "question_type": base["question_type"],
                    "component_source_mode": source_mode,
                }
            )

    output_predictions.sort(key=lambda row: (row["record_id"], row["mode"]))
    output_scores.sort(
        key=lambda row: (row["record_id"], row["mode"], row["dimension"])
    )
    write_jsonl(args.output_predictions, output_predictions)
    write_jsonl(args.output_scores, output_scores)
    Path(args.provenance).write_text(
        json.dumps(
            {
                "split_key": args.split_key,
                "modes": list(modes),
                "records_per_mode": len(base_predictions),
                "prediction_rows": len(output_predictions),
                "score_rows": len(output_scores),
                "components": provenance,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble")
    assemble_parser.add_argument("--baseline-predictions", required=True)
    assemble_parser.add_argument("--baseline-scores", required=True)
    assemble_parser.add_argument("--candidate-predictions", required=True)
    assemble_parser.add_argument("--candidate-scores", required=True)
    assemble_parser.add_argument("--selector-predictions", required=True)
    assemble_parser.add_argument("--split-manifest", required=True)
    assemble_parser.add_argument("--split-key", required=True)
    assemble_parser.add_argument("--modes", nargs="+")
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
