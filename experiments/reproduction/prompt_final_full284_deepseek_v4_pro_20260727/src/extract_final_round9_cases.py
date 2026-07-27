"""Extract the largest paired answer-quality improvement for each QA family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.reproduction.prompt_final_full284_deepseek_v4_pro_20260727.src.final_round8_pipeline import (
    FINAL_MODE,
)
from tools.axis_repro.build_tables import DIMS
from tools.axis_repro.common import read_jsonl


FAMILY_LABELS = {
    "multiple_choice": "MC",
    "open_ended": "OE",
    "true_false": "TF",
}


def select_best_cases(predictions: list[dict], scores: list[dict]) -> list[dict]:
    pred_index = {
        (row["record_id"], row["mode"]): row for row in predictions
    }
    score_index = {
        (row["record_id"], row["mode"], row["dimension"]): row
        for row in scores
    }
    candidates = []
    for (record_id, mode), candidate in pred_index.items():
        if mode != FINAL_MODE:
            continue
        baseline = pred_index[(record_id, "base")]
        if candidate["response"].strip() == baseline["response"].strip():
            continue
        dimensions = {}
        final_delta = 0.0
        for dimension in DIMS[candidate["question_type"]]:
            base_score = score_index[(record_id, "base", dimension)]
            candidate_score = score_index[(record_id, FINAL_MODE, dimension)]
            delta = float(candidate_score["score"]) - float(base_score["score"])
            weight = float(candidate_score["weight"])
            dimensions[dimension] = {
                "baseline": float(base_score["score"]),
                "candidate": float(candidate_score["score"]),
                "delta": delta,
                "weight": weight,
            }
            final_delta += delta * weight
        candidates.append(
            {
                "record_id": record_id,
                "question_type": candidate["question_type"],
                "family": FAMILY_LABELS[candidate["question_type"]],
                "component_source_mode": candidate.get("component_source_mode"),
                "question": baseline["question"],
                "gold_answer": baseline["answer"],
                "baseline_response": baseline["response"],
                "candidate_response": candidate["response"],
                "dimension_scores": dimensions,
                "final_delta": final_delta,
            }
        )
    result = []
    for question_type in FAMILY_LABELS:
        family = [
            row for row in candidates if row["question_type"] == question_type
        ]
        if not family:
            raise RuntimeError(f"No changed responses for {question_type}")
        result.append(max(family, key=lambda row: (row["final_delta"], row["record_id"])))
    return result


def fenced(value: str) -> str:
    return value.replace("```", "` ` `").strip()


def render_markdown(rows: list[dict]) -> str:
    lines = ["# Final Round-9 paired improvement cases", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['family']}: `{row['record_id']}`",
                "",
                f"- Weighted Final Δ: `{row['final_delta']:+.4f}`",
                f"- Component: `{row['component_source_mode']}`",
                f"- Dimension deltas: "
                + ", ".join(
                    f"{key} {value['delta']:+.4f}"
                    for key, value in row["dimension_scores"].items()
                ),
                "",
                "### Question",
                "",
                fenced(row["question"]),
                "",
                "### Gold answer",
                "",
                fenced(row["gold_answer"]),
                "",
                "### Baseline response",
                "",
                "```text",
                fenced(row["baseline_response"]),
                "```",
                "",
                "### Improved response",
                "",
                "```text",
                fenced(row["candidate_response"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    rows = select_best_cases(
        read_jsonl(args.predictions),
        read_jsonl(args.scores),
    )
    Path(args.output_json).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.output_md).write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"cases": [row["record_id"] for row in rows]}))


if __name__ == "__main__":
    main()