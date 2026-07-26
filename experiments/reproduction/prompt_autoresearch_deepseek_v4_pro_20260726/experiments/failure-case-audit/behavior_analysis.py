from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[5]
STAGE_A = REPO / "experiments/reproduction/prompt_stage_a_deepseek_v4_pro_20260725"
REVISED = REPO / "experiments/reproduction/prompt_revised_contract_deepseek_v4_pro_20260726"
HERE = Path(__file__).resolve().parent

STEP_PATTERN = re.compile(r"\bstep(?:s)?\s+(\d+)", flags=re.I)
ANSWER_PATTERN = re.compile(r"\banswer\s*:", flags=re.I)
TF_PATTERN = re.compile(r"\b(True|False)\.", flags=re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) != len(left):
        return None
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def features(response: str, start: int, end: int) -> dict:
    steps = [int(value) for value in STEP_PATTERN.findall(response)]
    truth_values = [value.lower() for value in TF_PATTERN.findall(response)]
    return {
        "chars": len(response),
        "shorter_than_200": len(response) < 200,
        "outside_step_reference": any(
            value < start or value >= end for value in steps
        ),
        "outside_step_values": [
            value for value in steps if value < start or value >= end
        ],
        "multiple_answer_markers": len(ANSWER_PATTERN.findall(response)) >= 2,
        "tf_self_contradiction": (
            len(truth_values) >= 2
            and truth_values[0] != truth_values[-1]
        ),
        "mentions_scaling_operation": bool(
            re.search(
                r"(?:scaled\s+by\s+100|divide(?:d)?\s+by\s+100|rescal)",
                response,
                flags=re.I,
            )
        ),
        "injects_new_mc_format": bool(
            re.search(r"(?m)^\s*[a-d]\)\s+", response, flags=re.I)
        ),
    }


def main() -> None:
    details = read_jsonl(HERE / "failure_case_details.jsonl")
    diagnostic_rows = read_jsonl(STAGE_A / "output_diagnostics.jsonl") + read_jsonl(
        REVISED / "output_diagnostics.jsonl"
    )
    diagnostics = {
        (row["record_id"], row["mode"]): row for row in diagnostic_rows
    }

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in details:
        grouped[(row["comparison"], row["question_type"])].append(row)

    result = {}
    for (comparison, question_type), rows in grouped.items():
        control_mode = rows[0]["control_mode"]
        treatment_mode = rows[0]["treatment_mode"]
        control_features = [
            features(
                row["control_response"],
                int(row["start_index"]),
                int(row["end_index"]),
            )
            for row in rows
        ]
        treatment_features = [
            features(
                row["treatment_response"],
                int(row["start_index"]),
                int(row["end_index"]),
            )
            for row in rows
        ]
        unsupported_control = [
            diagnostics[(row["record_id"], control_mode)].get(
                "unsupported_number_fraction"
            )
            for row in rows
        ]
        unsupported_treatment = [
            diagnostics[(row["record_id"], treatment_mode)].get(
                "unsupported_number_fraction"
            )
            for row in rows
        ]
        unsupported_pairs = [
            (float(control), float(treatment))
            for control, treatment in zip(
                unsupported_control, unsupported_treatment
            )
            if control is not None and treatment is not None
        ]

        def rate(items: list[dict], field: str) -> float:
            return sum(bool(item[field]) for item in items) / len(items)

        feature_summary = {}
        for field in (
            "shorter_than_200",
            "outside_step_reference",
            "multiple_answer_markers",
            "tf_self_contradiction",
            "mentions_scaling_operation",
            "injects_new_mc_format",
        ):
            control_rate = rate(control_features, field)
            treatment_rate = rate(treatment_features, field)
            feature_summary[field] = {
                "control_rate": control_rate,
                "treatment_rate": treatment_rate,
                "delta": treatment_rate - control_rate,
            }
        result.setdefault(comparison, {})[question_type] = {
            "count": len(rows),
            "control_mode": control_mode,
            "treatment_mode": treatment_mode,
            "response_length_delta_mean": mean(
                [row["response_length_delta"] for row in rows]
            ),
            "response_length_delta_vs_final_delta_pearson": pearson(
                [float(row["response_length_delta"]) for row in rows],
                [float(row["final_delta"]) for row in rows],
            ),
            "features": feature_summary,
            "unsupported_number_fraction": {
                "paired_count": len(unsupported_pairs),
                "control_mean": mean(
                    [control for control, _ in unsupported_pairs]
                ),
                "treatment_mean": mean(
                    [treatment for _, treatment in unsupported_pairs]
                ),
                "mean_delta": mean(
                    [
                        treatment - control
                        for control, treatment in unsupported_pairs
                    ]
                ),
            },
        }

    (HERE / "behavior_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
