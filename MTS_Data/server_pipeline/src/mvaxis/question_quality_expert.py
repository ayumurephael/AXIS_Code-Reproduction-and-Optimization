from __future__ import annotations

"""
Question quality expert for generated QA prompts.

Task scenario:
- The LLM has already generated a draft question from a template and image/text context.
- Before we accept that draft into the dataset, we want a lightweight expert pass that catches
  common failure modes such as multi-task questions, explanation tails inside judgment prompts,
  or scope questions that enumerate many candidate answers in the stem.
- The expert should prefer small, local repairs (truncate or delete secondary clauses) and only
  recommend a retry/fallback when the draft still looks structurally bad after repair.
"""

from dataclasses import dataclass
import re
from typing import List


JUDGMENT_RETRY_MARKERS = ("because", "given", "despite", "while", "rather than")


@dataclass
class QuestionQualityReview:
    original_text: str
    revised_text: str
    signals: List[str]
    should_retry: bool
    should_fallback: bool

    @property
    def changed(self) -> bool:
        return self.revised_text != self.original_text


def _ensure_question(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = cleaned.rstrip(" .")
    if cleaned and not cleaned.endswith("?"):
        cleaned += "?"
    return cleaned


def _truncate_before_marker(text: str, marker: str) -> str:
    pattern = re.compile(rf"\s+(?:,?\s*){re.escape(marker)}\b", flags=re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip(" ,;:")


def _strip_open_secondary_clause(text: str) -> str:
    patterns = [
        r",?\s+and what [^?]+$",
        r",?\s+and why [^?]+$",
        r",?\s+and which [^?]+$",
        r",?\s+what evidence supports that call[^?]*$",
        r",?\s+which evidence supports that call[^?]*$",
    ]
    revised = text
    for pattern in patterns:
        revised = re.sub(pattern, "", revised, flags=re.IGNORECASE)
    return revised.rstrip(" ,;:")


def _strip_scope_enumeration(text: str) -> str:
    revised = re.sub(
        r",?\s*system-wide,\s*channel-group,\s*single-channel,\s*or\s*normal variation",
        "",
        text,
        flags=re.IGNORECASE,
    )
    revised = re.sub(
        r",?\s*system-wide,\s*small-group,\s*single-channel,\s*or\s*normal variation",
        "",
        revised,
        flags=re.IGNORECASE,
    )
    return revised.rstrip(" ,;:")


def review_generated_question(
    text: str,
    *,
    answer_type: str,
    family: str,
    template_fallback: str,
) -> QuestionQualityReview:
    original = _ensure_question(text)
    revised = original
    signals: List[str] = []
    should_retry = False
    should_fallback = False

    if answer_type == "judgment":
        for marker in JUDGMENT_RETRY_MARKERS:
            if re.search(rf"\b{re.escape(marker)}\b", revised, flags=re.IGNORECASE):
                signals.append(f"judgment_marker:{marker}")
                should_retry = True
                revised = _truncate_before_marker(revised, marker)
        revised = _ensure_question(revised)
        if revised.count("?") > 1 or len(revised.split()) < 4:
            should_fallback = True

    elif answer_type == "open":
        lower = revised.lower()
        if "which" in lower and "why" in lower:
            signals.append("open_multitask:which+why")
            revised = _strip_open_secondary_clause(revised)
        if "which" in lower and "what evidence" in lower:
            signals.append("open_multitask:which+what_evidence")
            revised = _strip_open_secondary_clause(revised)
        if "scope" in lower and "evidence supports that call" in lower:
            signals.append("open_multitask:scope+supports_that_call")
            revised = _strip_open_secondary_clause(revised)
        if re.search(
            r"system-wide,\s*(channel-group|small-group),\s*single-channel,\s*or",
            revised,
            flags=re.IGNORECASE,
        ):
            signals.append("open_multitask:scope_enumeration")
            revised = _strip_scope_enumeration(revised)
        revised = _ensure_question(revised)
        if revised.count("?") > 1 or len(revised.split()) < 4:
            should_fallback = True

    if should_fallback:
        revised = _ensure_question(template_fallback)

    return QuestionQualityReview(
        original_text=original,
        revised_text=revised,
        signals=signals,
        should_retry=should_retry,
        should_fallback=should_fallback,
    )
