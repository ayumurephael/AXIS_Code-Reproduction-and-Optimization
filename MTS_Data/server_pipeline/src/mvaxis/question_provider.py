from __future__ import annotations

import base64
import copy
import io
import json
import mimetypes
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .data_schema import attach_channel_scales
from .question_template import (
    ANSWER_TYPE_TO_QUESTION_KIND,
    NORMAL_FOCUS_BANK,
    QUESTION_KIND_TO_FORMAT,
    QUESTION_FAMILIES,
    QUESTION_FOCUS_BANK,
    QuestionSpec,
    fixed_question_specs,
    render_question,
    sample_question_axes,
)
from .question_template_hard import HARD_FRAME_TEMPLATE_BANK, sample_hard_question_axes
from .utils import strip_think_blocks


FIXED_PLACEHOLDER_QUESTION_TYPE = "legacy_axis_explanation"
SAMPLES_PER_SERIES = 2
MIN_WINDOW_SIZE = 15
MAX_WINDOW_SIZE = 60
ANOMALY_RATIO = 0.5
WINDOW_CANDIDATE_MULTIPLIER = 10
MAX_WINDOW_CANDIDATES = 1000
QUESTION_GENERATION_PROVIDERS = {"llm", "visual", "image", "question", "generated"}
QUESTION_BANK_NAMES = {"regular", "hard"}


def _fmt(values: Iterable[Any] | None) -> str:
    """Format optional channel/value collections for natural-language answers."""

    items = [str(x) for x in (values or []) if x is not None]
    return ", ".join(items) if items else "none"


def _scope_label(scope: Any) -> str:
    """Convert internal anomaly scope labels into answer-friendly text."""

    return {"node": "one-channel", "edge": "small multi-channel", "subgraph": "broad multi-channel"}.get(str(scope), str(scope))


def _choice_letter(choice_text: Any) -> str | None:
    """Extract a leading multiple-choice option letter when present."""

    match = re.match(r"^\s*([A-Z])\s*[\.\):]", str(choice_text or "").strip())
    return match.group(1) if match else None


def _structured_choice_answer_letter(spec: QuestionSpec, fact: Dict[str, Any]) -> str | None:
    """Return the gold option letter for balanced hard-style choice specs."""

    is_anom = bool(fact.get("is_anomalous"))
    scope = str(fact.get("anomaly_scope") or "")
    family = str(spec.family or "")
    if family == "source_claim":
        if not is_anom:
            return "A"
        if scope == "node":
            return "B"
        if scope == "subgraph":
            return "E" if len(spec.choices) >= 5 else "D"
        return "D" if len(spec.choices) >= 5 else "C"
    if family == "root_scope":
        if not is_anom:
            return "A"
        if scope == "node":
            return "B"
        if scope == "subgraph":
            return "D"
        return "C"
    if family in {"anomaly_presence", "detector_interpretation", "false_alarm_interpretation"}:
        if not is_anom:
            return "A"
        if scope == "node":
            return "B"
        return "C"
    if family in {"anomaly_scope", "affected_scope", "claim_strength", "root_claim_sufficiency"}:
        if not is_anom:
            return "A"
        if scope == "node":
            return "B"
        if scope == "subgraph":
            return "D"
        return "C"
    if family in {"stability_profile", "channel_stability"}:
        if not is_anom:
            return "A"
        if family == "channel_stability":
            if scope == "node":
                return "C"
            return "D"
        if scope == "node":
            return "B"
        if scope == "subgraph":
            return "E" if len(spec.choices) >= 5 else "D"
        return "C"
    if family in {"context_fit", "fluctuation_level"}:
        return "A" if not is_anom else "D"
    return None


def _neutral_interval_fact_answer(fact: Dict[str, Any]) -> str:
    """Build a neutral ground-truth fact sentence without yes/no wording."""

    if not bool(fact.get("is_anomalous")):
        return (
            "The interval is not anomalous. No root-cause channel or affected channel should be assigned; "
            "observed channel movements should be treated as normal/background behavior."
        )
    root = fact.get("root_cause_channel")
    roots = list(fact.get("root_cause_channels") or ([] if root is None else [root]))
    affected = list(fact.get("affected_channels") or [])
    anomaly_type = fact.get("anomaly_type")
    root_anomaly_name = fact.get("root_anomaly_name")
    scope_text = _scope_label(fact.get("anomaly_scope"))
    root_text = _fmt(roots)
    affected_text = _fmt(affected)
    if anomaly_type and anomaly_type != "legacy_pattern":
        morphology_sentence = f"The true anomaly type is {anomaly_type}."
    elif root_anomaly_name:
        morphology_sentence = f"The true root-channel local morphology is {root_anomaly_name}."
    else:
        morphology_sentence = "The exact legacy local morphology is not available in this stored sample."
    return (
        f"The interval is anomalous. The root-cause channel is {root_text}. "
        f"The affected/abnormal channels are {affected_text}. "
        f"The scope is {scope_text}. {morphology_sentence}"
    )


def _choice_answer(spec: QuestionSpec, fact: Dict[str, Any]) -> str | None:
    """Choose the most compatible option text for a multiple-choice question."""

    if not spec.choices:
        return None
    structured_letter = _structured_choice_answer_letter(spec, fact)
    if structured_letter is not None:
        return structured_letter
    if not bool(fact.get("is_anomalous")):
        return _choice_letter(spec.choices[0]) or str(spec.choices[0])
    scope = str(fact.get("anomaly_scope"))
    anomaly_type = str(fact.get("anomaly_type"))
    affected = list(fact.get("affected_channels") or [])
    affected_count = len(affected)
    choice_text = " ".join(spec.choices).lower()
    if (
        "one-channel" in choice_text
        or "multi-channel" in choice_text
        or spec.family
        in {
            "scope_choice",
            "affected_channels",
            "multi_channel_change",
            "direct_component_choice",
            "minimal_explanation",
            "domain_explanation",
        }
    ):
        if scope == "node" or affected_count <= 1:
            selected = next(
                (
                    c
                    for c in spec.choices
                    if "one" in c.lower() or "isolated" in c.lower() or "local channel" in c.lower()
                ),
                str(spec.choices[min(1, len(spec.choices) - 1)]),
            )
            return _choice_letter(selected) or str(selected)
        if scope == "subgraph" or affected_count > 3:
            selected = next(
                (c for c in spec.choices if "broad" in c.lower() or "many" in c.lower() or "whole" in c.lower()),
                str(spec.choices[-1]),
            )
            return _choice_letter(selected) or str(selected)
        selected = next(
            (c for c in spec.choices if "small" in c.lower() or "group" in c.lower()),
            str(spec.choices[min(2, len(spec.choices) - 1)]),
        )
        return _choice_letter(selected) or str(selected)
    if spec.family in {"morphology", "node_type_choice", "morphology_family"}:
        if anomaly_type in {"spike", "variance_burst"}:
            selected = next((c for c in spec.choices if "spike" in c.lower() or "burst" in c.lower()), str(spec.choices[-1]))
            return _choice_letter(selected) or str(selected)
        if anomaly_type in {"level_shift", "trend_reversal", "stuck", "saturation"}:
            selected = next((c for c in spec.choices if "level" in c.lower() or "trend" in c.lower()), str(spec.choices[-1]))
            return _choice_letter(selected) or str(selected)
        if anomaly_type in {"oscillation", "phase_delay", "lag_shift"}:
            selected = next((c for c in spec.choices if "oscillation" in c.lower() or "timing" in c.lower()), str(spec.choices[-1]))
            return _choice_letter(selected) or str(selected)
    if spec.family in {
        "anomaly_presence",
        "false_positive",
        "normality",
        "detector_interpretation",
        "normality_label",
        "false_alarm_interpretation",
    }:
        selected = next(
            (
                c
                for c in spec.choices
                if "clear anomaly" in c.lower() or "true anomaly" in c.lower() or "anomalous" in c.lower()
            ),
            str(spec.choices[-1]),
        )
        return _choice_letter(selected) or str(selected)
    if spec.family.startswith("what_if") or spec.family in {"counterfactual", "intervention_plan"}:
        selected = next(
            (c for c in spec.choices if "affected" in c.lower() or "weaker" in c.lower() or "group weakens" in c.lower()),
            str(spec.choices[-1]),
        )
        return _choice_letter(selected) or str(selected)
    fallback = str(spec.choices[min(2, len(spec.choices) - 1)])
    return _choice_letter(fallback) or fallback


def answer_for_question(sample: Dict[str, Any], spec: QuestionSpec) -> Dict[str, Any]:
    """Generate the standard answer payload for a sample/question pair."""

    previous = sample.get("target_output") or {}
    fact = dict(previous.get("fact_check") or {})
    fact.pop("evidence_used", None)
    is_anom = bool(fact.get("is_anomalous"))
    root = fact.get("root_cause_channel")
    roots = list(fact.get("root_cause_channels") or ([] if root is None else [root]))
    affected = list(fact.get("affected_channels") or [])
    anomaly_type = fact.get("anomaly_type")
    scope = fact.get("anomaly_scope")
    scope_text = _scope_label(scope)
    affected_text = _fmt(affected)
    root_text = _fmt(roots)
    question_answer = _neutral_interval_fact_answer(fact)

    if not is_anom:
        reasoning = "The interval is labeled normal for this QA window; no root-cause or affected-channel attribution is assigned."
        evidence_chain = [
            "The sampled interval has no active anomaly label.",
            "No direct abnormal channel or affected set is available for attribution.",
            "The answer is based on the sampled QA window label and channel attribution fields.",
        ]
        reasoning_process = [
            "Step 1: Check the sampled QA window label and find no active anomaly in the interval.",
            "Step 2: Keep root-cause and affected-channel fields empty because no abnormal source is labeled for this window.",
            f"Step 3: Answer the {spec.answer_type} question family '{spec.family}' as a normal or unsupported-anomaly case.",
        ]
    else:
        reasoning = (
            f"The sampled interval is labeled anomalous with direct component {root_text}. "
            f"The anomaly is {scope_text} ({anomaly_type}) and affects {affected_text}."
        )
        evidence_chain = [
            f"The direct abnormal component is {root_text}.",
            f"The labeled scope/type is {scope_text} / {anomaly_type}.",
            f"Affected channels are {affected_text}.",
        ]
        reasoning_process = [
            "Step 1: Check the sampled QA window label and confirm that the interval is anomalous.",
            f"Step 2: Read the direct abnormal component label as {root_text}.",
            f"Step 3: Read the affected-channel set as {affected_text} and classify the scope/type as {scope_text} / {anomaly_type}.",
            f"Step 4: Map those facts to the requested {spec.answer_type} question family '{spec.family}'.",
        ]

    fact.update(
        {
            "question_id": spec.question_id,
            "question_difficulty": spec.difficulty,
            "question_answer_type": spec.answer_type,
            "question_family": spec.family,
            "choice_answer": _choice_answer(spec, fact),
        }
    )
    return {
        "fact_check": fact,
        "evidence_chain": evidence_chain,
        "reasoning_process": reasoning_process,
        "reasoning_summary": reasoning,
        "question_answer": question_answer,
        "final_answer": question_answer,
        "abnormality_score": float(previous.get("abnormality_score", 1.0 if is_anom else 0.0)),
        "answer_confidence": float(previous.get("answer_confidence", 0.70 if is_anom else 0.75)),
    }


def apply_question_spec(sample: Dict[str, Any], spec: QuestionSpec) -> Dict[str, Any]:
    """Attach a question spec to a sample and refresh its answer metadata."""

    sample["question_id"] = spec.question_id
    sample["question_difficulty"] = spec.difficulty
    sample["question_answer_type"] = spec.answer_type
    sample["question_family"] = spec.family
    sample["question_choices"] = list(spec.choices)
    sample["question_type"] = spec.question_id
    sample["question"] = render_question(spec)
    sample["target_output"] = answer_for_question(sample, spec)
    for window in sample.get("windows") or []:
        window["question"] = sample["question"]
        window["answer"] = sample["target_output"].get("final_answer", "")
        window["question_type"] = sample["question_type"]
        window["question_difficulty"] = spec.difficulty
        window["question_answer_type"] = spec.answer_type
    return sample

# Lightweight style pools used by the AXIS-style visual question workflow.

PLAIN_USER_QUESTION_STYLES = [
    "You are a curious beginner looking at a chart and asking simple questions about what looks wrong. Use everyday language, avoid technical jargon, and ask questions like a regular person would.",
    "You are a regular user who wants to know if there are any weird bumps or drops in the data. Use simple, conversational language without technical terms.",
    "You are a non-technical person asking straightforward questions about the data patterns. Use plain language, avoid jargon, and ask questions in a friendly, approachable way.",
    "You are a student learning about time series, asking basic questions to understand potential errors. Use simple language, ask questions naturally, and avoid complex terminology.",
    "You are a casual observer pointing out things that stand out in the graph. Use everyday language, ask questions conversationally, and avoid technical terms.",
    "You are someone who just wants to know if the data looks normal or if something is off. Use simple, direct language without technical jargon.",
    "You are a beginner asking simple questions like 'Is this normal?' or 'What's wrong here?'. Use everyday language, be conversational, and avoid technical terminology.",
    "You are a regular person looking at a graph and wondering if anything looks strange. Use plain, friendly language without technical jargon.",
    "You are a curious user asking basic questions about unusual patterns you notice. Use simple, conversational language and avoid complex technical terms.",
]

BUSINESS_USER_QUESTION_STYLES = [
    "You are a busy project manager who just wants to know if everything is running smoothly or if there's a crisis. Use direct, action-oriented language focused on business impact and urgency.",
    "You are a business executive asking high-level questions about potential risks in the data. Use concise, strategic language focused on business outcomes and decision-making.",
    "You are an operations manager checking for immediate alerts or problems in the system. Use urgent, action-focused language emphasizing operational impact.",
    "You are a decision maker asking if the current trend requires any action or attention. Use direct, executive-level language focused on actionable insights.",
    "You are a manager who needs quick answers about whether there are any problems that need attention. Use brief, results-oriented language emphasizing urgency and action.",
    "You are a business leader asking if the data shows any red flags or concerns. Use strategic, high-level language focused on business risks and implications.",
    "You are an operations director checking for any issues that might affect business operations. Use direct, operational language emphasizing business impact and urgency.",
]

SKEPTIC_USER_QUESTION_STYLES = [
    "You are a skeptical auditor double-checking if the data is actually normal or if something is hidden. Use questioning, critical language that challenges assumptions and probes for hidden issues.",
    "You are a cautious inspector asking probing questions to ensure nothing has gone wrong. Use careful, investigative language that questions and verifies claims.",
    "You are a quality assurance tester trying to break the system by asking about edge cases and glitches. Use critical, testing-oriented language that challenges and probes for weaknesses.",
    "You are a critical reviewer questioning whether the data patterns are truly normal or if anomalies are being missed. Use skeptical, analytical language that questions and verifies.",
    "You are a vigilant monitor asking tough questions to catch any potential problems. Use alert, probing language that challenges assumptions and seeks verification.",
    "You are a careful analyst who doesn't trust the data at face value and asks probing questions. Use skeptical, investigative language that questions and verifies claims.",
]

QUESTION_SYSTEM_STYLE_GROUPS = [
    PLAIN_USER_QUESTION_STYLES,
    BUSINESS_USER_QUESTION_STYLES,
    SKEPTIC_USER_QUESTION_STYLES,
]

POSITION_AND_CHANNEL_FOCUS_INSTRUCTION = (
    "Use global positions from the full time series for every position reference in the question, "
    "options, and judgment; do not renumber positions relative to the highlighted window. "
    "When the question focuses on channel-level behavior, explicitly name the relevant channel ids/names "
    "in the question and, when applicable, in the generated options or true/false judgment statement. "
    "Avoid vague wording such as 'one channel' or 'some channels' when specific channels are available. "
    "Do not focus on only one preselected anomalous-looking channel as if it were already the answer. "
    "Instead, when channel-level attribution is involved, name a small comparison set of plausible candidate channels "
    "or candidate channel groups, including distractor channels when possible. "
    "The question may mention concrete channels and positions, but it must not describe their specific behavior "
    "or reveal which one is truly abnormal."
)


def get_random_question_system_style(*, rng: random.Random | None = None) -> str:
    """Sample a system persona for question generation."""

    sampler = rng or random
    return sampler.choice(sampler.choice(QUESTION_SYSTEM_STYLE_GROUPS))


def get_question_specs(bank, level=None, fmt=None, focus_type=None):
    """Filter question-template entries by level, format, and optional focus type."""

    specs = []
    levels = [level] if level else bank.keys()
    for lv in levels:
        if lv not in bank:
            continue
        formats = [fmt] if fmt else bank[lv].keys()
        for fm in formats:
            if fm not in bank[lv]:
                continue
            for item in bank[lv][fm]:
                if focus_type is None or item.get("focus_type") == focus_type:
                    specs.append(item)
    return specs


def answer_type_to_question_kind(answer_type):
    """Map internal answer types to visual question generation kinds."""

    try:
        return ANSWER_TYPE_TO_QUESTION_KIND[answer_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported answer_type for question generation: {answer_type}") from exc


def question_frame_for_sample(sample: Dict[str, Any]) -> str:
    """Return the question frame implied by the sampled QA-window truth label."""

    override = str(sample.get("question_frame_override") or "").strip()
    if override in {"anomaly_frame", "normal_frame"}:
        return override
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return "anomaly_frame" if bool(fact.get("is_anomalous")) else "normal_frame"


def _normalize_question_bank(question_bank: str | None) -> str:
    """Normalize the requested question bank name."""

    bank_name = str(question_bank or "regular").strip().lower()
    if bank_name not in QUESTION_BANK_NAMES:
        raise ValueError(f"Unsupported question bank: {question_bank}")
    return bank_name


def sample_question_axes_for_bank(n: int, seed: int = 0, *, question_bank: str = "regular") -> List[Dict[str, str]]:
    """Sample balanced answer-type axes for the requested question bank."""

    bank_name = _normalize_question_bank(question_bank)
    if bank_name == "hard":
        return sample_hard_question_axes(n, seed)
    return sample_question_axes(n, seed)


def _focus_bank_for_question_bank(question_bank: str, frame: str) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Return the focus bank dictionary for the requested bank/frame pair."""

    bank_name = _normalize_question_bank(question_bank)
    if bank_name == "hard":
        return HARD_FRAME_TEMPLATE_BANK
    return QUESTION_FOCUS_BANK if frame == "anomaly_frame" else NORMAL_FOCUS_BANK


def sample_question_focus_item(
    question_type,
    has_anomaly=True,
    difficulty="anomaly_frame",
    rng=None,
    *,
    question_bank: str = "regular",
):
    """Sample one focus item for the requested visual question kind and truth frame."""

    if rng is None:
        rng = random
    try:
        fmt = QUESTION_KIND_TO_FORMAT[question_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported question_type: {question_type}") from exc

    frame = str(difficulty or ("anomaly_frame" if has_anomaly else "normal_frame"))
    bank = _focus_bank_for_question_bank(question_bank, frame)
    pool = get_question_specs(bank, level=frame, fmt=fmt)
    if not pool:
        pool = get_question_specs(bank, fmt=fmt)
    if not pool:
        raise ValueError(
            f"No MVAXIS question template for question_type={question_type}, "
            f"has_anomaly={has_anomaly}, difficulty={difficulty}, question_bank={question_bank}"
        )
    return rng.choice(pool)


def build_visual_question_prompt(
    *,
    focus_item,
    question_type,
    difficulty,
    window_size,
    window_start,
    window_end,
    has_anomaly,
    anomaly_description,
    num_options=None,
):
    """Build the text prompt used by the visual question generator."""

    window_end_inclusive = max(int(window_start), int(window_end) - 1)
    anomaly_status_line = "- The window contains anomalies\n" if has_anomaly else "- The window may not contain anomalies\n"
    description = anomaly_description or ("No anomaly detected" if not has_anomaly else "An anomalous pattern is present.")
    if num_options is None:
        num_options = 4
    focus_template = str(focus_item.get("template") or "").strip()
    focus_instruction = str(focus_item.get("instruction") or "").strip()
    anchor_choices = tuple(str(c).strip() for c in (focus_item.get("choices") or ()) if str(c).strip())
    focus_key = str(focus_item.get("key") or "").strip()
    is_normal_frame = str(difficulty or "").strip() == "normal_frame"
    anchor_block = ""
    if anchor_choices:
        anchor_block = (
            "Semantic option anchors (keep this order, but rewrite them concisely and evenly rather than copying them verbatim):\n"
            + "\n".join(f"- {choice}" for choice in anchor_choices)
            + "\n\n"
        )
    extra_choice_guidance = ""
    if focus_key in {"anomaly_presence", "detector_interpretation", "false_alarm_interpretation"}:
        extra_choice_guidance += (
            "13. For this family, use exactly one benign or no-alert option, exactly one manual-review or unresolved option, "
            "and two positive alert interpretations that differ by candidate channel or by single-channel versus multi-channel scope. "
            "Do not create three near-negative options such as 'no anomaly', 'likely false alarm', and 'too mixed' in the same set.\n"
        )
    if focus_key in {"root_scope", "affected_scope", "anomaly_scope", "root_claim_sufficiency"}:
        extra_choice_guidance += (
            "14. For this family, at least one distractor should name a plausible alternative channel or alternative small group, "
            "and another distractor should get the scope wrong by being too narrow or too broad. Do not make every non-gold option revolve around the same focal channel only.\n"
        )
    normal_frame_choice_guidance = ""
    normal_frame_judgment_guidance = ""
    normal_frame_open_guidance = ""
    if is_normal_frame:
        normal_frame_choice_guidance = (
            "15. For normal-frame questions, do not summarize benign or background evidence in the stem. "
            "Do not preview phrases such as staying near 0, remaining in usual ranges, mild variation, "
            "or plausible timing in the question text. Leave those details to the answer.\n"
        )
        normal_frame_judgment_guidance = (
            "12. For normal-frame questions, do not summarize benign/background evidence in the question text. "
            "Do not use phrasing such as 'based on traces staying near ...', 'remaining in usual ranges', "
            "'mild variation', or 'preserving a plausible timing pattern'. Ask only the claim or decision and leave the supporting evidence to the answer.\n"
        )
        normal_frame_open_guidance = (
            "12. For normal-frame questions, do not preview the benign evidence in the question text. "
            "Ask only for the decision, explanation target, or missing-evidence target, and leave supporting details such as staying near 0, "
            "remaining in range, mild variation, or plausible timing to the answer.\n"
        )

    if question_type == "multiple_choice":
        return (
            "Generate a multiple choice question about anomaly detection in this multivariate time series window.\n\n"
            "Context:\n"
            f"- Analysis window from position {window_start} to {window_end} in the full time series\n"
            f"- Within this window, there are {window_size} data points "
            f"(global positions {window_start} to {window_end_inclusive})\n"
            "- IMPORTANT: All position references in your question must use global positions "
            "in the full time series, NOT positions relative to the highlighted window\n"
            f"{anomaly_status_line}"
            f"- Reference anomaly context for writing a valid question, not for direct disclosure: {description}\n\n"
            "Selected focus:\n"
            f"- Template: {focus_template}\n"
            f"- Instruction: {focus_instruction}\n\n"
            f"{POSITION_AND_CHANNEL_FOCUS_INSTRUCTION}\n\n"
            f"{anchor_block}"
            "Requirements:\n"
            f"1. Generate ONLY one concise multiple-choice question stem plus exactly {num_options} options using letters A, B, C, D, E, F as needed. Put the stem first, then each option on its own new line.\n"
            "2. Keep the stem brief and natural, ideally one sentence, and keep it on a single phenomenon or decision.\n"
            "3. Use the selected focus as the semantic skeleton; do not drift into another task.\n"
            "4. Keep options parallel, concise, and comparable in specificity. Do not let one option contain a long descriptive clause while the others stay short.\n"
            "5. Do not overload the options with morphology stories, long evidence descriptions, or multi-sentence explanations. Short labels or short claims are preferred.\n"
            "6. If one option names a specific channel or channel group, competing options should stay similarly concrete rather than vague.\n"
            "7. Preserve the semantic roles and order of the option anchors when they are provided, but rewrite them in concise language rather than copying them verbatim.\n"
            "8. The stem should mainly identify the global-position interval and ask the decision. Do not narrate detailed local events such as spikes, dips, oscillations, bursts, reversals, timing stories, or rough amplitude summaries before the question.\n"
            "9. After naming the interval, move directly to the question. Do not append a second descriptive clause introduced by words such as where, with, given, based on, showing, or while.\n"
            "10. When the task is about source, affected-channel scope, channel stability, anomaly scope, or detector interpretation, prefer naming concrete channel ids or concrete channel groups in the stem and relevant options when the interval supports that level of specificity.\n"
            "11. Do not build the stem around only one presumed anomalous channel. Mention a small set of plausible candidate channels or candidate channel groups whenever possible, so the question tests multivariate discrimination rather than simply pointing at the gold channel.\n"
            "12. If channels are named, name them briefly without describing their detailed local behavior or performance in the question text. Do not reveal which candidate is the true abnormal one.\n"
            "12a. When possible, let one non-gold option focus on a different plausible candidate channel or small candidate group, and let another non-gold option differ by scope rather than by repeating the same focal channel.\n"
            "13. Do not include rough numeric cues in the stem unless the task itself is explicitly about a numeric threshold. Avoid dense exact decimals and do not answer the question.\n"
            f"{extra_choice_guidance}"
            f"{normal_frame_choice_guidance}"
        )

    if question_type == "true_false":
        return (
            "Generate a true/false style question about anomaly detection in this multivariate time series window.\n\n"
            "Context:\n"
            f"- Analysis window from position {window_start} to {window_end} in the full time series\n"
            f"- Window contains {window_size} data points "
            f"(global positions {window_start} to {window_end_inclusive})\n"
            "- IMPORTANT: Position references must use global positions in the full time series only\n"
            f"- Reference anomaly context for writing a valid proposition, not for direct disclosure: {description}\n\n"
            "Selected focus:\n"
            f"- Template: {focus_template}\n"
            f"- Instruction: {focus_instruction}\n\n"
            f"{POSITION_AND_CHANNEL_FOCUS_INSTRUCTION}\n\n"
            "Requirements:\n"
            "1. Generate ONLY one natural user-style question, not the answer. Match the persona in your system message.\n"
            "2. Use the selected focus as the semantic skeleton and write exactly one verifiable proposition.\n"
            "3. Do not combine anomaly status, root cause, affected channels, anomaly type, and scope in the same question "
            "unless the selected focus explicitly asks for that combination.\n"
            "3a. When the selected focus is channel-level, root-cause-related, affected-channel-related, timing-alignment-related, or stability-related, explicitly name the relevant channel ids when the interval supports that level of specificity. Do not fall back to vague wording like 'one channel' or 'some channels' when a concrete channel or channel set is available.\n"
            "3b. For anomaly-oriented judgment questions, do not frame the proposition around only one presumed anomalous channel as if it were already the strongest cue. If the question is about root cause, affected channels, scope, stability, or detector trust, mention a small comparison set of plausible candidate channels or candidate groups whenever possible.\n"
            "3c. The judgment should ask whether a claim is defensible among those candidates, not assume that one named channel is already the focal anomaly. The question may mention concrete channels and global positions, but it must not describe their local behavior or indicate which candidate is the true answer.\n"
            "4. If mentioning possible distractor or comparison channels, describe them as possible comparison/background channels, "
            "not as confirmed abnormal, root-cause channels, or affected channels.\n"
            "5. For normal windows, avoid strong anomaly language such as 'clear spike', 'obvious red flag', "
            "'sustained level shift', or 'abnormal lag' unless the question asks whether such a pattern is absent.\n"
            "6. For false-positive questions, make the polarity unambiguous: True means the proposal is likely a false alarm; "
            "False means the proposal is a real anomaly.\n"
            "7. Ask naturally, such as 'Is there...', 'Does the data show...', or 'Are there any...'; "
            "do NOT write 'True or False: ...' and do not reveal hidden labels or provided metadata.\n"
            "8. Keep the question focused on the interval and the claim itself. Do not preface the question with a local event description before asking.\n"
            "9. Explicitly avoid constructions such as 'such as ...', 'even with ...', and clauses introduced by 'where', 'with', 'given', 'while', 'despite', or 'because'.\n"
            "10. Do not describe a concrete spike, dip, oscillation, burst, reversal, or similar local phenomenon in the question text. Leave those details for the answer.\n"
            "11. Ground the proposition in evidence from the plotted traces and the interval's rough numeric behavior; "
            "one or two rough quantitative cues are allowed when helpful, but avoid dense exact decimals."
            f"\n{normal_frame_judgment_guidance}"
        )
    if question_type != "open_ended":
        raise ValueError(f"Unsupported question_type: {question_type}")

    return (
        "Generate an open-ended question about anomaly detection in this multivariate time series window.\n\n"
        "Context:\n"
        f"- Analysis window from position {window_start} to {window_end} in the full time series\n"
        f"- Window contains {window_size} data points "
        f"(global positions {window_start} to {window_end_inclusive})\n"
        "- IMPORTANT: All position references must use global positions in the full time series only\n"
        f"{anomaly_status_line}"
        f"- Reference anomaly context for writing a valid question, not for direct disclosure: {description}\n\n"
        "Selected focus:\n"
        f"- Template: {focus_template}\n"
        f"- Instruction: {focus_instruction}\n\n"
        f"{POSITION_AND_CHANNEL_FOCUS_INSTRUCTION}\n\n"
        "Requirements:\n"
        "1. Generate ONLY the question text. Match the persona in your system message.\n"
        "2. Use the selected focus as the semantic skeleton; ask for analysis of the highlighted window, not a different task.\n"
        "3. Ask about exactly one phenomenon, one claim, or one decision in the highlighted interval. Do not combine multiple sub-questions in the same prompt.\n"
        "4. Choose one open-ended style only: a why-question, an evidence question, a which-channel question, or a single analysis request. Do not mix these styles in one question.\n"
        "4a. When the selected focus is channel-level, root-cause-related, affected-channel-related, timing-alignment-related, or stability-related, explicitly name the relevant channel ids when the interval supports that level of specificity. Do not fall back to vague wording like 'one channel' or 'some channels' when a concrete channel or channel set is available.\n"
        "4b. For anomaly-oriented open questions that involve root-cause, affected-channel, scope, stability, or detector interpretation, do not ask about only one presumed anomalous channel. Instead, ask about a small set of plausible candidate channels or candidate groups.\n"
        "4c. The question may ask which candidate is strongest, which channels should be treated as directly abnormal, or which scope best fits, but it must not pre-identify the gold channel as the focus.\n"
        "4d. The question may mention concrete channels and global positions, but it must not describe their specific behavior or imply which candidate is already the strongest cue.\n"
        "5. Encourage discussion of evidence from the plotted traces and the interval's rough numeric behavior, "
        "including channels, global timing, morphology, scope, or affected-channel reasoning only when directly relevant to that one selected focus.\n"
        "6. Let the answer provide the evidence or justification. Do not append a second clause such as 'based on ...', 'and why ...', 'and what evidence ...', or 'rather than ...' after the main question.\n"
        "7. Do not preface the question with a local event description before asking. Explicitly avoid constructions such as 'such as ...', 'even with ...', and clauses introduced by 'where', 'with', 'given', 'while', 'despite', or 'because'.\n"
        "8. Do not describe a concrete spike, dip, oscillation, burst, reversal, or similar local phenomenon in the question text. Leave those details for the answer.\n"
        "9. If mentioning possible distractor or comparison channels, ask the model to distinguish them from root/affected channels "
        "rather than treating them as confirmed abnormal.\n"
        "10. Keep the question short and clean, ideally one sentence, and avoid asking for too many unrelated facts at once.\n"
        "11. Focus on trace-level and rough quantitative evidence rather than dense exact numerical values, and do not reveal hidden labels, provided labels, or template metadata.\n"
        f"{normal_frame_open_guidance}"
    )


def _expected_choice_count(focus_item: Dict[str, Any]) -> int:
    """Return the expected multiple-choice option count for a focus item."""

    anchors = tuple(str(c).strip() for c in (focus_item.get("choices") or ()) if str(c).strip())
    return len(anchors) if anchors else 4


def _parse_generated_multiple_choice(text: str, *, expected_count: int | None = None) -> Tuple[str, Tuple[str, ...]] | None:
    """Parse an LLM-generated multiple-choice question into stem and option lines."""

    normalized = str(text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return None
    normalized = re.sub(r"^\s*Question\s*:\s*", "", normalized, flags=re.IGNORECASE)
    candidate = normalized
    pattern = re.compile(r"(?m)^\s*([A-F])[\.\):]\s*(.+?)\s*$")
    matches = list(pattern.finditer(candidate))
    if len(matches) < 3:
        candidate = re.sub(r"\s+(?=([A-F])[\.\):]\s)", "\n", normalized)
        matches = list(pattern.finditer(candidate))
    if len(matches) < 3:
        return None
    if expected_count is not None and len(matches) < expected_count:
        return None
    if expected_count is not None and len(matches) > expected_count:
        matches = matches[:expected_count]
    stem = candidate[: matches[0].start()].strip()
    stem = re.sub(r"(?:\bChoices?\s*:?\s*)$", "", stem, flags=re.IGNORECASE).strip()
    if not stem:
        return None
    choices = tuple(f"{match.group(1)}. {match.group(2).strip()}" for match in matches)
    return stem, choices


def is_fixed_placeholder_question(sample: Dict[str, Any]) -> bool:
    """Return whether a sample still has the placeholder question."""

    return sample.get("question_type") == FIXED_PLACEHOLDER_QUESTION_TYPE


def _labels(sample: Dict[str, Any]) -> np.ndarray:
    """Read sample labels as an integer numpy array."""

    return np.asarray(sample["series"]["labels"], dtype=int)


def _values(sample: Dict[str, Any]) -> np.ndarray:
    """Read sample time-series values as a float numpy array."""

    return np.asarray(sample["series"]["values"], dtype=float)


def _has_anomaly(labels: np.ndarray, start: int, end: int) -> bool:
    """Check whether any anomaly label intersects the half-open window."""

    if labels.ndim == 1:
        return bool(labels[start:end].max() > 0)
    return bool(labels[start:end].max() > 0)


def _affected_channels(sample: Dict[str, Any], start: int, end: int) -> List[str]:
    """Return channel ids whose labels are active inside the target window."""

    labels = _labels(sample)
    if labels.ndim == 1:
        if labels[start:end].max() <= 0:
            return []
        return [ch["channel_id"] for ch in sample["channels"] if ch.get("active", True)]
    affected = np.flatnonzero(labels[start:end].max(axis=0) > 0)
    return [sample["channels"][int(i)]["channel_id"] for i in affected]


def _window_variance(sample: Dict[str, Any], start: int, end: int) -> float:
    """Compute total value variance for ranking normal hard-negative windows."""

    window = _values(sample)[start:end]
    return float(np.var(window)) if window.size else 0.0


def _candidate_windows(
    sample: Dict[str, Any],
    rng: random.Random,
    *,
    min_window_size: int,
    max_window_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate random analysis-window candidates split by anomaly presence."""

    values = _values(sample)
    labels = _labels(sample)
    length = int(values.shape[0])
    min_size = max(1, min(int(min_window_size), length))
    max_size = max(min_size, min(int(max_window_size), length))

    attempts = min(MAX_WINDOW_CANDIDATES, max(1, (length - min_size + 1) * WINDOW_CANDIDATE_MULTIPLIER))#采样数
    anomalous: List[Dict[str, Any]] = []
    normal: List[Dict[str, Any]] = []
    for _ in range(attempts):
        window_size = rng.randint(min_size, max_size)
        max_start = max(0, length - window_size)
        start = rng.randint(0, max_start) if max_start > 0 else 0
        end = min(length, start + window_size)
        has_anom = _has_anomaly(labels, start, end)
        item = {
            "start": int(start),
            "end": int(end),
            "has_anomaly": has_anom,
            "variance": _window_variance(sample, start, end),#维护variance是为了可以按方差选正常序列
        }
        if has_anom:
            anomalous.append(item)
        else:
            normal.append(item)
    normal.sort(key=lambda x: float(x["variance"]), reverse=True)
    return anomalous, normal


def sample_analysis_windows(
    sample: Dict[str, Any],
    *,
    seed: int = 520,
    samples_per_series: int = SAMPLES_PER_SERIES,
    min_window_size: int = MIN_WINDOW_SIZE,
    max_window_size: int = MAX_WINDOW_SIZE,
    anomaly_ratio: float = ANOMALY_RATIO,
) -> List[Dict[str, Any]]:
    """Sample QA windows for visual question generation.

    Anomalous windows are sampled from windows intersecting labels. Normal
    windows are hard negatives: high-variance normal windows are preferred.
    """
    rng = random.Random(seed)
    anomalous, normal = _candidate_windows(
        sample,
        rng,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
    )
    # Keep the requested anomaly/normal mix at the window level, not only at the series level.
    target_anom = int(samples_per_series * float(anomaly_ratio))#采样数，按照比例设定
    target_normal = max(0, samples_per_series - target_anom)#采样数
    selected: List[Dict[str, Any]] = []

    for _ in range(target_anom):
        if anomalous:
            selected.append(dict(rng.choice(anomalous)))#随机抽取
    for _ in range(target_normal):
        if normal:
            pool_size = max(1, len(normal) // 3)
            pool = normal[:pool_size] if rng.random() < 0.7 else normal
             # 70% from top third variance, 30% from the rest
            selected.append(dict(rng.choice(pool)))

    all_candidates = anomalous + normal
    while len(selected) < samples_per_series and all_candidates:
        selected.append(dict(rng.choice(all_candidates)))
    if not selected:
        interval = sample.get("target_interval") or {"start": 0, "end": min(max_window_size, _values(sample).shape[0])}
        selected.append(
            {
                "start": int(interval["start"]),
                "end": int(interval["end"]),
                "has_anomaly": _has_anomaly(_labels(sample), int(interval["start"]), int(interval["end"])),
                "variance": _window_variance(sample, int(interval["start"]), int(interval["end"])),
            }
        )
    return selected[:samples_per_series]


def _window_fact_construct(sample: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    """Rewrite fact-check labels so they describe the sampled QA window."""

    previous_fact = dict((sample.get("target_output") or {}).get("fact_check") or {})
    previous_fact.pop("evidence_used", None)
    affected = _affected_channels(sample, start, end)
    is_anom = bool(affected)
    old_roots = list(previous_fact.get("root_cause_channels") or [])
    root = previous_fact.get("root_cause_channel")
    if root not in affected:
        root = old_roots[0] if old_roots and old_roots[0] in affected else (affected[0] if affected else None)  #维护一定有root的设定，优先原root，其次受影响通道中原root，最后受影响通道中任一，一般不会错

    roots = [root] if root else []
    scope = None
    if is_anom:
        scope = "node" if len(affected) == 1 else ("edge" if len(affected) <= 3 else "subgraph")#异常等级仍然是单变量，双变量，还是多变量
    previous_fact.update(
        {
            "is_anomalous": is_anom,
            "root_cause_channel": root,
            "root_cause_channels": roots,
            "affected_channels": affected,
            "anomaly_type": previous_fact.get("anomaly_type") if is_anom else None,
            "root_anomaly_name": previous_fact.get("root_anomaly_name") if is_anom else None,
            "anomaly_scope": scope,
            "abnormal_edges": previous_fact.get("abnormal_edges") if is_anom else [],
            "causal_path": affected if is_anom else [],
        }
    )
    return previous_fact


def construct_window_label(
    sample: Dict[str, Any],
    start: int,
    end: int,
    *,
    window_index: int = 0,
    copy_sample: bool = True,
) -> Dict[str, Any]:
    """Clone or update a sample so its target interval is the chosen QA window."""

    out = copy.deepcopy(sample) if copy_sample else sample
    base_id = out.get("base_sample_id") or out.get("sample_id") or "sample"
    out["base_sample_id"] = base_id
    out["source_sample_id"] = out.get("sample_id")
    out["sample_id"] = f"{base_id}_w{window_index:03d}_{int(start):04d}_{int(end):04d}"
    out["target_interval"] = {"start": int(start), "end": int(end)}
    fact = _window_fact_construct(out, int(start), int(end))
    out["root_cause"] = {
        "channel_id": fact.get("root_cause_channel"),
        "time_index": int(start) if fact.get("root_cause_channel") else None,
        "interval": [int(start), int(end)] if fact.get("root_cause_channel") else None,
        "provided_by": "window_label",
    }
    label = dict(out.get("synthetic_label") or {})
    label.update(
        {
            "affected_channels": fact.get("affected_channels", []),
            "root_cause_channels": fact.get("root_cause_channels", []),
            "anomaly_type": fact.get("anomaly_type"),
            "root_anomaly_name": fact.get("root_anomaly_name"),
            "anomaly_scope": fact.get("anomaly_scope"),
            "causal_path": fact.get("causal_path", []),
            "abnormal_edges": fact.get("abnormal_edges", []),
        }
    )
    out["synthetic_label"] = label
    attach_channel_scales(out)
    out.pop("evidence_card", None)
    previous = dict(out.get("target_output") or {})
    previous["fact_check"] = fact
    previous["abnormality_score"] = 1.0 if fact.get("is_anomalous") else 0.0
    previous["answer_confidence"] = previous.get("answer_confidence", 0.70 if fact.get("is_anomalous") else 0.75)
    out["target_output"] = previous
    out["windows"] = [
        {
            "window_range": [int(start), int(end)],
            "question": out.get("question"),
            "answer": previous.get("final_answer", ""),
            "question_type": out.get("question_type"),
            "has_anomaly": bool(fact.get("is_anomalous")),
            "labels": _labels(out)[int(start):int(end)].max(axis=1).astype(int).tolist()
            if _labels(out).ndim == 2
            else _labels(out)[int(start):int(end)].astype(int).tolist(),
            "mv_label": {
                "root_cause_channel": fact.get("root_cause_channel"),
                "affected_channels": fact.get("affected_channels", []),
                "anomaly_type": fact.get("anomaly_type"),
                "root_anomaly_name": fact.get("root_anomaly_name"),
                "anomaly_scope": fact.get("anomaly_scope"),
            },
        }
    ]
    return out


def retarget_sample_to_window(
    sample: Dict[str, Any],
    start: int,
    end: int,
    *,
    window_index: int = 0,
    copy_sample: bool = True,
) -> Dict[str, Any]:
    """Backward-compatible name for retargeting a sample to a QA window."""

    return construct_window_label(
        sample,
        start,
        end,
        window_index=window_index,
        copy_sample=copy_sample,
    )


def expand_samples_with_analysis_windows(
    samples: Sequence[Dict[str, Any]],
    *,
    seed: int = 520,
    samples_per_series: int = SAMPLES_PER_SERIES,
    min_window_size: int = MIN_WINDOW_SIZE,
    max_window_size: int = MAX_WINDOW_SIZE,
    anomaly_ratio: float = ANOMALY_RATIO,
) -> List[Dict[str, Any]]:
    """Expand base series into one or more retargeted QA-window samples."""

    expanded: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        windows = sample_analysis_windows(
            sample,
            seed=seed + sample_idx * 7919,
            samples_per_series=samples_per_series,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
            anomaly_ratio=anomaly_ratio,
        )
        for window_idx, window in enumerate(windows):
            expanded.append(
                construct_window_label(
                    sample,
                    int(window["start"]),
                    int(window["end"]),
                    window_index=window_idx + 1,
                    copy_sample=True,
                )
            )
    return expanded


def duplicate_samples_with_question_frames(
    samples: Sequence[Dict[str, Any]],
    *,
    frames: Sequence[str] = ("anomaly_frame", "normal_frame"),
) -> List[Dict[str, Any]]:
    """Duplicate each sampled window into one row per requested question frame."""

    duplicated: List[Dict[str, Any]] = []
    for pair_index, sample in enumerate(samples, start=1):
        window_sample_id = str(sample.get("sample_id") or f"window_{pair_index:05d}")
        for frame_index, frame in enumerate(frames, start=1):
            if frame not in {"anomaly_frame", "normal_frame"}:
                raise ValueError(f"Unsupported question frame: {frame}")
            out = copy.deepcopy(sample)
            out["question_pair_id"] = window_sample_id
            out["question_frame_override"] = frame
            out["question_frame_index"] = frame_index
            out["source_window_sample_id"] = window_sample_id
            out["sample_id"] = f"{window_sample_id}__{frame}"
            duplicated.append(out)
    return duplicated


def assign_question(
    sample: Dict[str, Any],
    spec: QuestionSpec,
    *,
    index: int | None = None,
    copy_sample: bool = True,
    preserve_source_question: bool = True,
) -> Dict[str, Any]:
    """Apply one selected question spec to a sample."""

    out = copy.deepcopy(sample) if copy_sample else sample
    if preserve_source_question:
        out.setdefault("source_question_type", out.get("question_type"))
        out.setdefault("source_question", out.get("question"))
    if index is not None:
        base_id = out.get("base_sample_id") or out.get("sample_id") or f"sample_{index:05d}"
        out["base_sample_id"] = base_id
        out["sample_id"] = f"{base_id}_q{index:03d}_{spec.question_id}"
    return apply_question_spec(out, spec)


def _clean_generated_question(text: str) -> str:
    """Normalize raw LLM output into a bare question string."""

    cleaned = strip_think_blocks(text)
    cleaned = re.sub(r"^```(?:text)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = re.sub(r"^(question|q)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.replace("〞", ",").replace("〝", ",").replace("—", ", ")
    cleaned = re.sub(r"(?<=\d)\s*[每–-]\s*(?=\d)", " to ", cleaned)
    cleaned = cleaned.replace("–", "-")
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _anomaly_description(sample: Dict[str, Any], start: int, end: int) -> str:
    """Summarize hidden interval facts in natural language for question design."""

    del start, end
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return _neutral_interval_fact_answer(fact)


def _channel_names(sample: Dict[str, Any], count: int) -> List[str]:
    """Return stable channel display names for plotting."""

    channels = sample.get("channels") or []
    names = [str(ch.get("channel_id", f"ch_{idx}")) for idx, ch in enumerate(channels[:count])]
    if len(names) < count:
        names.extend(f"ch_{idx}" for idx in range(len(names), count))
    return names


def _generate_multichannel_time_series_image(
    sample: Dict[str, Any],
    *,
    window_start: int,
    window_end: int,
) -> str:
    """Render all channels with the analysis window highlighted as a base64 PNG."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    values = _values(sample)
    if values.ndim == 1:
        values = values[:, None]
    length, num_channels = int(values.shape[0]), int(values.shape[1])
    if length <= 0 or num_channels <= 0:
        return ""
    start = max(0, min(int(window_start), length - 1))
    end = max(start + 1, min(int(window_end), length))
    names = _channel_names(sample, num_channels)

    fig_height = max(4.5, min(1.6 + 0.9 * num_channels, 24.0))
    fig, axes = plt.subplots(
        num_channels,
        1,
        figsize=(12, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]
    x = np.arange(length)
    try:
        for ch_idx, ax in enumerate(axes):
            raw = np.asarray(values[:, ch_idx], dtype=float)
            if np.isfinite(raw).any():
                mean = float(np.nanmean(raw))
                std = float(np.nanstd(raw))
                plotted = (raw - mean) / std if std > 1e-8 else raw - mean
                scale_note = "z" if std > 1e-8 else "centered"
            else:
                plotted = raw
                scale_note = "value"
            ax.plot(x, plotted, color="#2563eb", linewidth=1.0, alpha=0.75)
            ax.plot(x[start:end], plotted[start:end], color="#dc2626", linewidth=1.8, alpha=0.95)
            ax.axvspan(start, end - 1, color="#facc15", alpha=0.22)
            ax.axvline(start, color="#92400e", linewidth=0.8, alpha=0.65)
            ax.axvline(end - 1, color="#92400e", linewidth=0.8, alpha=0.65)
            ax.set_ylabel(f"{names[ch_idx]}\n{scale_note}", rotation=0, ha="right", va="center", fontsize=8)
            ax.grid(True, alpha=0.22, linestyle="--")
        axes[0].set_title(
            f"Multivariate Time Series with Highlighted Analysis Window ({start}-{end})",
            fontsize=13,
            fontweight="bold",
        )
        axes[-1].set_xlabel("Time step")
        fig.tight_layout(h_pad=0.25)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception:
        return ""
    finally:
        plt.close(fig)


def _image_data_url(path: str) -> str | None:
    """Load a saved image file as a data URL when present."""

    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _prepare_visual_question_image(
    sample: Dict[str, Any],
    *,
    window_start: int,
    window_end: int,
) -> str | None:
    """Prefer a saved window PNG and fall back to rendering from series values."""

    artifacts = sample.setdefault("question_generation_artifacts", {})
    image_path = str(artifacts.get("image_path") or "")
    data_url = _image_data_url(image_path)
    if data_url:
        artifacts["visual_question_used_image"] = True
        artifacts["visual_question_image_source"] = "saved_window_image"
        artifacts["visual_question_image_path"] = image_path
        return data_url

    image_base64 = _generate_multichannel_time_series_image(
        sample,
        window_start=window_start,
        window_end=window_end,
    )
    if image_base64:
        artifacts["visual_question_used_image"] = True
        artifacts["visual_question_image_source"] = "rendered_from_series"
        if image_path:
            artifacts["visual_question_image_path"] = image_path
        return f"data:image/png;base64,{image_base64}"

    artifacts["visual_question_used_image"] = False
    artifacts["visual_question_image_source"] = "text_only_fallback"
    if image_path:
        artifacts["visual_question_image_path"] = image_path
    return None


def _make_visual_question_generator(client: Any, *, difficulty: str, seed: int, question_bank: str = "regular") -> Any:
    """Create a visual question generator backed by mvaxis templates."""

    class VisualQuestionGenerator:
        """Adapter that sends visual-question prompts through the current LLM client."""

        def __init__(self, llm_client: Any, *, question_difficulty: str, generator_seed: int) -> None:
            """Store the injected client and deterministic template sampler."""

            self.llm_client = llm_client
            self.question_difficulty = question_difficulty
            self.rng = random.Random(generator_seed)
            self.question_bank = _normalize_question_bank(question_bank)

        def _make_api_request(self, messages: List[Dict[str, Any]], temperature: float = 0.3, max_tokens: int = 200) -> str:
            """Delegate chat messages to the configured mvaxis LLM client."""

            del temperature, max_tokens
            response = self.llm_client.complete(messages)
            return str(response.content).strip()

        def _generate_single_question(self, data: Dict[str, Any], *, focus_item: Dict[str, Any] | None = None) -> str:
            """Generate one visual question from a sampled template and highlighted plot."""

            sample = data["sample"]
            question_type = data["question_type"]
            window_start = int(data["window_start"])
            window_end = int(data["window_end"])
            anomaly_description = str(data.get("anomaly_description") or "")
            has_anomaly = bool(data["has_anomaly"])
            window_size = window_end - window_start
            if focus_item is None:
                focus_item = sample_question_focus_item(
                    question_type,
                    has_anomaly=has_anomaly,
                    difficulty=self.question_difficulty,
                    rng=self.rng,
                    question_bank=self.question_bank,
                )
            num_options = _expected_choice_count(focus_item) if question_type == "multiple_choice" else None
            prompt = build_visual_question_prompt(
                focus_item=focus_item,
                question_type=question_type,
                difficulty=self.question_difficulty,
                window_size=window_size,
                window_start=window_start,
                window_end=window_end,
                has_anomaly=has_anomaly,
                anomaly_description=anomaly_description,
                num_options=num_options,
            )
            system_message = get_random_question_system_style(rng=self.rng)
            image_data_url = _prepare_visual_question_image(
                sample,
                window_start=window_start,
                window_end=window_end,
            )
            # Visual question generation uses text instructions plus a highlighted time-series image.
            if image_data_url:
                messages = [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ]
            question = self._make_api_request(messages, temperature=0.5, max_tokens=300)
            return question

    return VisualQuestionGenerator(client, question_difficulty=difficulty, generator_seed=seed)


def sample_bank_question_spec(
    *,
    difficulty: str,
    answer_type: str,
    seed: int = 0,
    index: int = 0,
    question_bank: str = "regular",
) -> QuestionSpec:
    """Sample a fixed question spec from the requested regular or hard bank."""

    question_type = answer_type_to_question_kind(answer_type)
    rng = random.Random(seed + index * 37)
    focus_item = sample_question_focus_item(
        question_type,
        difficulty=difficulty,
        rng=rng,
        question_bank=question_bank,
    )
    bank_name = _normalize_question_bank(question_bank)
    family = str(focus_item["key"])
    return QuestionSpec(
        question_id=f"{bank_name}_{difficulty}_{answer_type}_{index:04d}_{family}",
        difficulty=difficulty,
        answer_type=answer_type,
        family=family,
        text=str(focus_item.get("template") or "").strip(),
        choices=tuple(str(c).strip() for c in (focus_item.get("choices") or ()) if str(c).strip()),
    )


def generate_question_spec_with_visual_workflow(
    sample: Dict[str, Any],
    client: Any,
    *,
    difficulty: str,
    answer_type: str,
    seed: int = 520,
    index: int = 0,
    question_bank: str = "regular",
) -> QuestionSpec | None:
    """Use the visual question workflow to produce one frame-aware QuestionSpec."""

    interval = sample["target_interval"]
    start, end = int(interval["start"]), int(interval["end"])
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    has_anomaly = bool(fact.get("is_anomalous"))
    question_type = answer_type_to_question_kind(answer_type)
    rng = random.Random(seed + index * 104729)
    focus_item = sample_question_focus_item(
        question_type,
        has_anomaly=has_anomaly,
        difficulty=difficulty,
        rng=rng,
        question_bank=question_bank,
    )
    generator = _make_visual_question_generator(
        client,
        difficulty=difficulty,
        seed=seed + index * 104729,
        question_bank=question_bank,
    )
    request_payload = {
        "sample": sample,
        "question_type": question_type,
        "window_start": start,
        "window_end": end,
        "anomaly_description": _anomaly_description(sample, start, end),
        "has_anomaly": has_anomaly,
    }
    if answer_type == "choice":
        expected_count = _expected_choice_count(focus_item)
        text = generator._generate_single_question(request_payload, focus_item=focus_item)
        cleaned = _clean_generated_question(text)
        parsed = _parse_generated_multiple_choice(cleaned, expected_count=expected_count)
        if parsed is None:
            return None
        stem, choices = parsed
        return QuestionSpec(
            question_id=f"{_normalize_question_bank(question_bank)}_visual_{difficulty}_{answer_type}_{index:04d}_{focus_item['key']}",
            difficulty=difficulty,
            answer_type=answer_type,
            family=str(focus_item["key"]),
            text=stem,
            choices=choices,
        )

    text = generator._generate_single_question(request_payload, focus_item=focus_item)
    text = _clean_generated_question(text)
    if not text:
        return None
    return QuestionSpec(
        question_id=f"{_normalize_question_bank(question_bank)}_visual_{difficulty}_{answer_type}_{index:04d}_{focus_item['key']}",
        difficulty=difficulty,
        answer_type=answer_type,
        family=str(focus_item["key"]),
        text=text,
        choices=(),
    )


def assign_questions(
    samples: Sequence[Dict[str, Any]],
    *,
    seed: int = 520,
    copy_samples: bool = True,
    preserve_source_question: bool = True,
    annotate_sample_id: bool = False,
    provider: str = "bank",
    llm_client: Any = None,
    question_bank: str = "regular",
    axes: Sequence[Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    """Assign generated  questions to a sequence of samples."""

    sampled_axes = list(axes) if axes is not None else sample_question_axes_for_bank(len(samples), seed, question_bank=question_bank)
    out: List[Dict[str, Any]] = []
    for idx, (sample, axis) in enumerate(zip(samples, sampled_axes), start=1):
        difficulty = question_frame_for_sample(sample)
        provider_name = (provider or "bank").lower()
        # Provider selection is the main switch between fixed templates and visual generation.
        if provider_name in QUESTION_GENERATION_PROVIDERS:
            if llm_client is None:
                raise ValueError("llm_client is required when provider uses visual question generation")
            spec = generate_question_spec_with_visual_workflow(
                sample,
                llm_client,
                difficulty=difficulty,
                answer_type=axis["answer_type"],
                seed=seed,
                index=idx,
                question_bank=question_bank,
            )
            if spec is None:
                continue
        elif provider_name == "bank":
            spec = sample_bank_question_spec(
                difficulty=difficulty,
                answer_type=axis["answer_type"],
                seed=seed,
                index=idx,
                question_bank=question_bank,
            )
        else:
            raise ValueError(f"Unsupported question provider: {provider}")
        out.append(
            assign_question(
                sample,
                spec,
                index=idx if annotate_sample_id else None,
                copy_sample=copy_samples,
                preserve_source_question=preserve_source_question,
            )
        )
    return out


def build_qa_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    seed: int = 520,
    question_provider: str = "bank",
    llm_client: Any = None,
    question_bank: str = "regular",
    sample_windows: bool = True,
    samples_per_series: int = SAMPLES_PER_SERIES,
    min_window_size: int = MIN_WINDOW_SIZE,
    max_window_size: int = MAX_WINDOW_SIZE,
    anomaly_ratio: float = ANOMALY_RATIO,
    annotate_sample_id: bool = False,
) -> List[Dict[str, Any]]:
    """Build final QA rows by sampling windows and attaching questions."""

    rows = (
        expand_samples_with_analysis_windows(
            samples,
            seed=seed,
            samples_per_series=samples_per_series,
            min_window_size=min_window_size,
            max_window_size=max_window_size,
            anomaly_ratio=anomaly_ratio,
        )
        if sample_windows
        else [copy.deepcopy(sample) for sample in samples]
    )
    return assign_questions(
        rows,
        seed=seed + 17,
        provider=question_provider,
        llm_client=llm_client,
        question_bank=question_bank,
        copy_samples=False,
        preserve_source_question=True,
        annotate_sample_id=annotate_sample_id,
    )


def ensure_question(
    sample: Dict[str, Any],
    *,
    index: int = 0,
    seed: int = 520,
    provider: str = "bank",
    overwrite: bool = False,
    overwrite_fixed_placeholder: bool = True,
    copy_sample: bool = True,
    llm_client: Any = None,
    question_bank: str = "regular",
) -> Dict[str, Any]:
    """Ensure a sample has a usable question, optionally replacing old placeholders."""

    provider = (provider or "bank").lower()
    if provider in {"none", "off", "keep"}:
        return copy.deepcopy(sample) if copy_sample else sample
    if provider not in {"bank", *QUESTION_GENERATION_PROVIDERS}:
        raise ValueError(f"Unsupported question provider: {provider}")

    has_question = bool(sample.get("question"))
    should_replace = bool(overwrite)
    should_replace = should_replace or (overwrite_fixed_placeholder and is_fixed_placeholder_question(sample))
    should_replace = should_replace or not has_question
    if not should_replace:
        return copy.deepcopy(sample) if copy_sample else sample

    axis = sample_question_axes_for_bank(index + 1, seed, question_bank=question_bank)[index]
    difficulty = question_frame_for_sample(sample)
    # Use the same provider switch as assign_questions so inference scripts can refresh questions lazily.
    if provider in QUESTION_GENERATION_PROVIDERS:
        if llm_client is None:
            raise ValueError("llm_client is required when provider uses visual question generation")
        spec = generate_question_spec_with_visual_workflow(
            sample,
            llm_client,
            difficulty=difficulty,
            answer_type=axis["answer_type"],
            seed=seed,
            index=index,
            question_bank=question_bank,
        )
        if spec is None:
            raise ValueError(
                f"Visual question generation did not produce a valid {axis['answer_type']} question for sample "
                f"{sample.get('sample_id') or index} in bank={question_bank}"
            )
    else:
        spec = sample_bank_question_spec(
            difficulty=difficulty,
            answer_type=axis["answer_type"],
            seed=seed,
            index=index,
            question_bank=question_bank,
        )
    return assign_question(sample, spec, copy_sample=copy_sample)


def ensure_questions(
    samples: Iterable[Dict[str, Any]],
    *,
    seed: int = 520,
    provider: str = "bank",
    overwrite: bool = False,
    overwrite_fixed_placeholder: bool = True,
    copy_samples: bool = True,
    llm_client: Any = None,
    question_bank: str = "regular",
) -> List[Dict[str, Any]]:
    """Ensure every sample in an iterable has a question."""

    rows = list(samples)
    out: List[Dict[str, Any]] = []
    for idx, sample in enumerate(rows):
        out.append(
            ensure_question(
                sample,
                index=idx,
                seed=seed,
                provider=provider,
                overwrite=overwrite,
                overwrite_fixed_placeholder=overwrite_fixed_placeholder,
                copy_sample=copy_samples,
                llm_client=llm_client,
                question_bank=question_bank,
            )
        )
    return out


def dump_fixed_question_specs(path: str) -> None:
    """Write the fixed template bank to JSON for inspection."""

    specs = fixed_question_specs()
    with open(path, "w", encoding="utf-8") as f:
        json.dump([spec.__dict__ for spec in specs], f, ensure_ascii=False, indent=2)
