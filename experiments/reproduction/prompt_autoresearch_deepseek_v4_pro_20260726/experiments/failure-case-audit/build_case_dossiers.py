from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[5]
STAGE_A = REPO / "experiments/reproduction/prompt_stage_a_deepseek_v4_pro_20260725"
REVISED = REPO / "experiments/reproduction/prompt_revised_contract_deepseek_v4_pro_20260726"
OUTPUT = Path(__file__).resolve().parent

MODES = (
    "fixed_role",
    "fixed_role_evidence_contract",
    "fixed_role_evidence_contract_revised",
)
MODE_LABELS = {
    "fixed_role": "Fixed role",
    "fixed_role_evidence_contract": "Old Contract",
    "fixed_role_evidence_contract_revised": "Revised Contract",
}
CASE_IDS = (
    "series_000072:1",
    "series_000094:0",
    "series_000064:0",
    "series_000135:1",
    "series_000090:1",
    "series_000051:0",
    "series_000087:1",
    "series_000050:1",
    "series_000085:1",
    "series_000125:1",
    "series_000127:1",
)
DIMS = {
    "multiple_choice": ("correctness", "reasoning_quality"),
    "open_ended": ("accuracy", "completeness", "relevance"),
    "true_false": ("correctness", "justification_quality"),
}
STEP_PATTERN = re.compile(r"\bstep(?:s)?\s+(\d+)", flags=re.I)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def compact(text: str, limit: int = 1200) -> str:
    normalized = " ".join(str(text).split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "…"


def main() -> None:
    prediction_rows = read_jsonl(STAGE_A / "predictions.jsonl") + read_jsonl(
        REVISED / "predictions.jsonl"
    )
    score_rows = read_jsonl(STAGE_A / "all_scores.jsonl") + read_jsonl(
        REVISED / "scores.jsonl"
    )
    predictions = {
        (row["record_id"], row["mode"]): row
        for row in prediction_rows
        if row["mode"] in MODES and row["record_id"] in CASE_IDS
    }
    scores: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    weights: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    notes: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in score_rows:
        key = (row["record_id"], row["mode"])
        if key not in predictions:
            continue
        scores[key][row["dimension"]] = float(row["score"])
        weights[key][row["dimension"]] = float(row["weight"])
        notes[key][row["dimension"]] = str(row.get("judge_content", ""))

    lines = [
        "# Contract failure-case dossiers",
        "",
        "Each dossier holds the question, reference, observed window, and all "
        "three prompt outputs constant except for prompt condition.",
        "",
    ]
    structured = []
    for record_id in CASE_IDS:
        representative = predictions[(record_id, "fixed_role")]
        start = int(representative["start_index"])
        end = int(representative["end_index"])
        window = representative["time_series"][start:end]
        qtype = representative["question_type"]
        lines.extend(
            [
                f"## {record_id} — {qtype}",
                "",
                f"- Window: [{start}, {end}); anomaly label: "
                f"`{representative.get('has_anomaly')}`",
                f"- Question: {representative['question']}",
                f"- Reference: {representative['answer']}",
                "- Window values: "
                + ", ".join(
                    f"{start + offset}:{float(value):+.4f}"
                    for offset, value in enumerate(window)
                ),
                "",
            ]
        )
        case = {
            "record_id": record_id,
            "question_type": qtype,
            "start": start,
            "end": end,
            "has_anomaly": representative.get("has_anomaly"),
            "question": representative["question"],
            "answer": representative["answer"],
            "window": window,
            "modes": {},
        }
        for mode in MODES:
            key = (record_id, mode)
            row = predictions[key]
            dimensions = DIMS[qtype]
            final = sum(
                scores[key][dimension] * weights[key][dimension]
                for dimension in dimensions
            )
            step_references = [
                int(match) for match in STEP_PATTERN.findall(row["response"])
            ]
            outside = [
                step
                for step in step_references
                if step < start or step >= end
            ]
            mode_result = {
                "scores": {
                    **{
                        dimension: scores[key][dimension]
                        for dimension in dimensions
                    },
                    "final": final,
                },
                "response": row["response"],
                "step_references": step_references,
                "outside_window_step_references": outside,
                "judge_notes": {
                    dimension: notes[key][dimension]
                    for dimension in dimensions
                },
            }
            case["modes"][mode] = mode_result
            score_text = ", ".join(
                f"{dimension}={scores[key][dimension]:.3f}"
                for dimension in dimensions
            )
            lines.extend(
                [
                    f"### {MODE_LABELS[mode]}",
                    "",
                    f"- Scores: Final={final:.3f}; {score_text}",
                    f"- Explicit step references: {step_references or 'none'}; "
                    f"outside window: {outside or 'none'}",
                    f"- Response: {row['response']}",
                    "",
                ]
            )
            for dimension in dimensions:
                lines.extend(
                    [
                        f"- Judge ({dimension}): "
                        f"{compact(notes[key][dimension])}",
                        "",
                    ]
                )
        structured.append(case)

    (OUTPUT / "case_dossiers.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "case_dossiers.json").write_text(
        json.dumps(structured, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(structured),
                "modes_per_case": len(MODES),
                "output": str(OUTPUT / "case_dossiers.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
