from __future__ import annotations

import re
from typing import Any, Dict, Optional


EXACT_LABEL_ANSWER_FORMAT = (
    "Return exactly one minified JSON object and no markdown. "
    "Required top-level keys: fact_check, final_answer. "
    "fact_check keys: is_anomalous, anomaly_type, anomaly_scope, "
    "answer_label, choice_answer, question_answer_type, question_id. "
    "Do not output root_cause_channel, root_cause_channels, affected_channels, "
    "abnormal_edges, or causal_path in this run. "
    "For choice questions, answer_label and choice_answer must be the same exact "
    "single option letter such as A, B, C, or D, and final_answer must be that same letter. "
    "For judgment questions, answer_label must be yes or no, choice_answer must be null, "
    "and final_answer must be the same yes or no. "
    "For open questions, answer_label and choice_answer must be null, and final_answer "
    "must directly answer the exact question in one short sentence."
)


def normalize_answer_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    text = str(label).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    match = re.match(r"^\s*([A-Ea-e])(?:\s*[\.)：:]|\s+|$)", text)
    if match:
        return match.group(1).upper()
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D", "E"}:
        return text.upper()
    return text


def target_answer_label(target_output: Dict[str, Any]) -> Optional[str]:
    fact = target_output.get("fact_check") or {}
    answer_type = fact.get("question_answer_type")
    if answer_type == "choice":
        return normalize_answer_label(fact.get("choice_answer") or fact.get("answer_label"))
    if answer_type == "judgment":
        return normalize_answer_label(
            target_output.get("question_answer")
            or target_output.get("final_answer")
            or fact.get("answer_label")
        )
    return None


def schema_label_exact_teacher(target_output: Dict[str, Any]) -> Dict[str, Any]:
    fact = target_output.get("fact_check") or {}
    answer_type = fact.get("question_answer_type")
    answer_label = target_answer_label(target_output)
    is_anomalous = bool(fact.get("is_anomalous", False))

    if answer_type == "choice":
        choice_answer = answer_label
        final_answer = answer_label or ""
    elif answer_type == "judgment":
        choice_answer = None
        final_answer = answer_label or ""
    else:
        choice_answer = None
        answer_label = None
        final_answer = str(
            target_output.get("question_answer")
            or target_output.get("final_answer")
            or ""
        ).strip()

    return {
        "fact_check": {
            "is_anomalous": is_anomalous,
            "anomaly_type": fact.get("anomaly_type") if is_anomalous else None,
            "anomaly_scope": fact.get("anomaly_scope") if is_anomalous else None,
            "answer_label": answer_label,
            "choice_answer": choice_answer,
            "question_answer_type": answer_type,
            "question_id": fact.get("question_id"),
        },
        "final_answer": final_answer,
    }
