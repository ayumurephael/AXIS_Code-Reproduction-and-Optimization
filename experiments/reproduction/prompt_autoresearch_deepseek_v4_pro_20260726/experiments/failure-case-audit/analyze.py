from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.axis_repro.common import extract_choice, extract_true_false


REPO = Path(__file__).resolve().parents[5]
STAGE_A = REPO / "experiments/reproduction/prompt_stage_a_deepseek_v4_pro_20260725"
REVISED = REPO / "experiments/reproduction/prompt_revised_contract_deepseek_v4_pro_20260726"
OUTPUT = Path(__file__).resolve().parent

DIMS = {
    "multiple_choice": ("correctness", "reasoning_quality"),
    "open_ended": ("accuracy", "completeness", "relevance"),
    "true_false": ("correctness", "justification_quality"),
}
COMPARISONS = {
    "old_contract_minus_fixed_role": (
        "fixed_role",
        "fixed_role_evidence_contract",
    ),
    "revised_contract_minus_fixed_role": (
        "fixed_role",
        "fixed_role_evidence_contract_revised",
    ),
    "revised_contract_minus_old_contract": (
        "fixed_role_evidence_contract",
        "fixed_role_evidence_contract_revised",
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decision(row: dict) -> str | bool | None:
    if row["question_type"] == "multiple_choice":
        return extract_choice(row["response"])
    if row["question_type"] == "true_false":
        return extract_true_false(row["response"])
    return None


def expected_decision(row: dict) -> str | bool | None:
    if row["question_type"] == "multiple_choice":
        return extract_choice(row["answer"])
    if row["question_type"] == "true_false":
        return extract_true_false(row["answer"])
    return None


def compact(text: str, limit: int = 500) -> str:
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "…"


def main() -> None:
    score_rows = read_jsonl(STAGE_A / "all_scores.jsonl") + read_jsonl(
        REVISED / "scores.jsonl"
    )
    prediction_rows = read_jsonl(STAGE_A / "predictions.jsonl") + read_jsonl(
        REVISED / "predictions.jsonl"
    )

    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    weights: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    judge_notes: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in score_rows:
        key = (row["record_id"], row["mode"])
        scores[key][row["dimension"]] = float(row["score"])
        weights[key][row["dimension"]] = float(row["weight"])
        judge_notes[key][row["dimension"]] = str(row.get("judge_content", ""))

    predictions = {
        (row["record_id"], row["mode"]): row for row in prediction_rows
    }

    comparison_results = {}
    detailed_rows = []
    for comparison_name, (control_mode, treatment_mode) in COMPARISONS.items():
        records = []
        for (record_id, mode), treatment_prediction in predictions.items():
            if mode != treatment_mode:
                continue
            control_key = (record_id, control_mode)
            treatment_key = (record_id, treatment_mode)
            if control_key not in predictions:
                continue
            qtype = treatment_prediction["question_type"]
            dimensions = DIMS[qtype]
            if not all(
                dimension in scores[control_key]
                and dimension in scores[treatment_key]
                for dimension in dimensions
            ):
                continue

            control_final = sum(
                scores[control_key][dimension] * weights[control_key][dimension]
                for dimension in dimensions
            )
            treatment_final = sum(
                scores[treatment_key][dimension]
                * weights[treatment_key][dimension]
                for dimension in dimensions
            )
            control_prediction = predictions[control_key]
            control_decision = decision(control_prediction)
            treatment_decision = decision(treatment_prediction)
            expected = expected_decision(treatment_prediction)
            record = {
                "comparison": comparison_name,
                "record_id": record_id,
                "question_type": qtype,
                "question": treatment_prediction["question"],
                "answer": treatment_prediction["answer"],
                "has_anomaly": treatment_prediction.get("has_anomaly"),
                "start_index": treatment_prediction.get("start_index"),
                "end_index": treatment_prediction.get("end_index"),
                "control_mode": control_mode,
                "treatment_mode": treatment_mode,
                "control_final": control_final,
                "treatment_final": treatment_final,
                "final_delta": treatment_final - control_final,
                "dimension_scores": {
                    dimension: {
                        "control": scores[control_key][dimension],
                        "treatment": scores[treatment_key][dimension],
                        "delta": (
                            scores[treatment_key][dimension]
                            - scores[control_key][dimension]
                        ),
                    }
                    for dimension in dimensions
                },
                "expected_decision": expected,
                "control_decision": control_decision,
                "treatment_decision": treatment_decision,
                "control_correct": (
                    control_decision == expected if expected is not None else None
                ),
                "treatment_correct": (
                    treatment_decision == expected
                    if expected is not None
                    else None
                ),
                "control_response": control_prediction["response"],
                "treatment_response": treatment_prediction["response"],
                "response_length_delta": (
                    len(treatment_prediction["response"])
                    - len(control_prediction["response"])
                ),
                "judge_notes": {
                    dimension: {
                        "control": judge_notes[control_key][dimension],
                        "treatment": judge_notes[treatment_key][dimension],
                    }
                    for dimension in dimensions
                },
            }
            records.append(record)
            detailed_rows.append(record)

        by_type = {}
        for question_type in DIMS:
            typed = [
                record
                for record in records
                if record["question_type"] == question_type
            ]
            deltas = [record["final_delta"] for record in typed]
            transitions = Counter()
            if question_type != "open_ended":
                for record in typed:
                    transitions[
                        (
                            f"{record['control_correct']}"
                            f"->{record['treatment_correct']}"
                        )
                    ] += 1
            by_type[question_type] = {
                "count": len(typed),
                "mean_final_delta": sum(deltas) / len(deltas),
                "wins": sum(delta > 1e-12 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
                "losses": sum(delta < -1e-12 for delta in deltas),
                "decision_correctness_transitions": dict(transitions),
                "mean_response_length_delta": (
                    sum(record["response_length_delta"] for record in typed)
                    / len(typed)
                ),
                "largest_losses": [
                    {
                        "record_id": record["record_id"],
                        "final_delta": record["final_delta"],
                        "dimension_deltas": {
                            dimension: values["delta"]
                            for dimension, values in record[
                                "dimension_scores"
                            ].items()
                        },
                        "question": compact(record["question"], 300),
                        "answer": compact(record["answer"], 300),
                        "control_decision": record["control_decision"],
                        "treatment_decision": record["treatment_decision"],
                        "control_response": compact(
                            record["control_response"], 500
                        ),
                        "treatment_response": compact(
                            record["treatment_response"], 500
                        ),
                    }
                    for record in sorted(typed, key=lambda item: item["final_delta"])[
                        :10
                    ]
                ],
                "largest_gains": [
                    {
                        "record_id": record["record_id"],
                        "final_delta": record["final_delta"],
                        "dimension_deltas": {
                            dimension: values["delta"]
                            for dimension, values in record[
                                "dimension_scores"
                            ].items()
                        },
                        "question": compact(record["question"], 300),
                        "answer": compact(record["answer"], 300),
                        "control_decision": record["control_decision"],
                        "treatment_decision": record["treatment_decision"],
                        "control_response": compact(
                            record["control_response"], 500
                        ),
                        "treatment_response": compact(
                            record["treatment_response"], 500
                        ),
                    }
                    for record in sorted(
                        typed,
                        key=lambda item: item["final_delta"],
                        reverse=True,
                    )[:10]
                ],
            }
        comparison_results[comparison_name] = {
            "control_mode": control_mode,
            "treatment_mode": treatment_mode,
            "record_count": len(records),
            "by_question_type": by_type,
        }

    (OUTPUT / "failure_case_summary.json").write_text(
        json.dumps(comparison_results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "failure_case_details.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for record in detailed_rows
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                name: {
                    qtype: {
                        key: value
                        for key, value in summary.items()
                        if key
                        in {
                            "count",
                            "mean_final_delta",
                            "wins",
                            "ties",
                            "losses",
                            "decision_correctness_transitions",
                            "mean_response_length_delta",
                        }
                    }
                    for qtype, summary in result["by_question_type"].items()
                }
                for name, result in comparison_results.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
