from __future__ import annotations

from typing import Any, Dict, List, Optional

from .student_answer_provider import (
    ANSWER_CHANNEL_INSPECTION_INSTRUCTION,
    STUDENT_OUTPUT_CLEANLINESS_RULE,
    _contextual_hint_block,
    _global_information,
    _question_text,
    _student_system_format_rule,
    _target_interval,
    _template_question_type,
    format_observed_data_str,
    get_random_student_answer_system_style,
    parse_student_answer,
    prompt_from_messages,
)


EXPERIMENTAL_PROMPT_STYLES = {
    "default_strict",
    "question_first_hints_data",
}


def _choice_output_rules() -> str:
    return "\n".join(
        [
            "Output rules for this multiple-choice question:",
            '1. The FIRST line must be exactly "Answer: [option letter]".',
            '2. The SECOND line must be exactly "Analysis:".',
            "3. Write one short paragraph after Analysis in at most 120 words.",
            '4. Do not write any sentence before the "Answer:" line.',
            "5. Explain why the chosen option fits better than the main alternatives.",
        ]
    )


def _judgment_output_rules() -> str:
    return "\n".join(
        [
            "Output rules for this judgment question:",
            '1. The FIRST line must be exactly "Answer: True" or "Answer: False".',
            '2. The SECOND line must be exactly "Analysis:".',
            "3. Write one short paragraph after Analysis in at most 120 words.",
            '4. Do not write any sentence before the "Answer:" line.',
            "5. Explain why the observed interval supports or rejects the claim.",
        ]
    )


def _open_output_rules() -> str:
    return "\n".join(
        [
            "Output rules for this open-ended question:",
            "1. Use exactly three sections in this order:",
            "   Decision:",
            "   Main evidence:",
            "   Interpretation:",
            '2. The first sentence under Decision must be exactly "This interval is anomalous." or "This interval is normal."',
            "3. Main evidence should name the strongest relevant channels and global positions.",
            "4. The first sentence under Interpretation must directly answer the question.",
            "5. Keep the full answer within about 150 words.",
        ]
    )


def _render_question_first_prompt(
    sample: Dict[str, Any],
    *,
    embedding_hint_trace: Optional[Dict[str, Any]],
    use_axis_hints: bool,
    include_global_hints: bool,
    include_channel_hints: bool,
    include_anomaly_score_text: bool,
    include_anomaly_score_image_note: bool,
    include_raw_image_note: bool,
    max_window_rows: Optional[int],
    digits: int,
) -> str:
    start, end = _target_interval(sample)
    question = _question_text(sample)
    question_type = _template_question_type(sample)
    hint_block = _contextual_hint_block(
        embedding_hint_trace,
        use_axis_hints=use_axis_hints,
        include_global_hints=include_global_hints,
        include_channel_hints=include_channel_hints,
    )
    data_str = format_observed_data_str(
        sample,
        digits=digits,
        max_window_rows=max_window_rows,
        include_anomaly_scores=include_anomaly_score_text,
    )
    window_end = max(start, end - 1)
    image_notes: List[str] = []
    if include_anomaly_score_image_note:
        image_notes.append(
            "Visual anomaly-score reference: the companion figure shows the anomaly-score curve and variables ch_0 through ch_9; the highlighted region is the target interval."
        )
    if include_raw_image_note:
        image_notes.append(
            "Visual time-series reference: the companion figure shows variables ch_0 through ch_9 with the highlighted target interval and no anomaly-score curve."
        )
    if question_type == "multiple_choice":
        output_rules = _choice_output_rules()
    elif question_type == "true_false":
        output_rules = _judgment_output_rules()
    else:
        output_rules = _open_output_rules()
    parts = [
        "You are analyzing a time-series window for anomaly QA.",
        "",
        "Question:",
        question,
        "",
        "Support Bundle (read these before answering):",
        f"- Analysis window in the full time series: {start} to {window_end}",
        f"- Window size: {max(0, end - start)} points",
        f"- Global time series context: {_global_information(sample)}",
        hint_block,
        "Observed target-window data:",
        f"Data [current_value]:\n{data_str}",
    ]
    if image_notes:
        parts.extend(["", "Visual notes:"] + [f"- {note}" for note in image_notes])
    parts.extend(
        [
            "",
            "Answering guidance:",
            "- Use global positions from the full time series for every location reference.",
            "- Do not renumber positions relative to the highlighted window.",
            f"- {ANSWER_CHANNEL_INSPECTION_INSTRUCTION}",
            output_rules,
            STUDENT_OUTPUT_CLEANLINESS_RULE,
        ]
    )
    return "\n".join(parts)


def build_student_answer_prompt_experimental(
    sample: Dict[str, Any],
    *,
    prompt_style: str = "default_strict",
    embedding_hint_trace: Optional[Dict[str, Any]] = None,
    use_axis_hints: bool = False,
    include_global_hints: bool = True,
    include_channel_hints: bool = True,
    include_evidence_card: bool = True,
    include_anomaly_score_text: bool = False,
    include_anomaly_score_image_note: bool = False,
    include_raw_image_note: bool = False,
    max_window_rows: Optional[int] = None,
    digits: int = 3,
) -> str:
    from .student_answer_provider import build_student_answer_prompt

    if prompt_style == "default_strict":
        return build_student_answer_prompt(
            sample,
            embedding_hint_trace=embedding_hint_trace,
            use_axis_hints=use_axis_hints,
            include_global_hints=include_global_hints,
            include_channel_hints=include_channel_hints,
            include_evidence_card=include_evidence_card,
            include_anomaly_score_text=include_anomaly_score_text,
            include_anomaly_score_image_note=include_anomaly_score_image_note,
            include_raw_image_note=include_raw_image_note,
            max_window_rows=max_window_rows,
            digits=digits,
        )
    if prompt_style == "question_first_hints_data":
        return _render_question_first_prompt(
            sample,
            embedding_hint_trace=embedding_hint_trace,
            use_axis_hints=use_axis_hints,
            include_global_hints=include_global_hints,
            include_channel_hints=include_channel_hints,
            include_anomaly_score_text=include_anomaly_score_text,
            include_anomaly_score_image_note=include_anomaly_score_image_note,
            include_raw_image_note=include_raw_image_note,
            max_window_rows=max_window_rows,
            digits=digits,
        )
    raise ValueError(f"Unknown experimental prompt style: {prompt_style}")


def build_student_answer_messages_experimental(
    sample: Dict[str, Any],
    *,
    prompt_style: str = "default_strict",
    embedding_hint_trace: Optional[Dict[str, Any]] = None,
    use_axis_hints: bool = False,
    include_global_hints: bool = True,
    include_channel_hints: bool = True,
    include_evidence_card: bool = True,
    include_anomaly_score_text: bool = False,
    include_anomaly_score_image_note: bool = False,
    include_raw_image_note: bool = False,
    max_window_rows: Optional[int] = None,
    digits: int = 3,
) -> List[Dict[str, str]]:
    question_type = _template_question_type(sample)
    prompt = build_student_answer_prompt_experimental(
        sample,
        prompt_style=prompt_style,
        embedding_hint_trace=embedding_hint_trace,
        use_axis_hints=use_axis_hints,
        include_global_hints=include_global_hints,
        include_channel_hints=include_channel_hints,
        include_evidence_card=include_evidence_card,
        include_anomaly_score_text=include_anomaly_score_text,
        include_anomaly_score_image_note=include_anomaly_score_image_note,
        include_raw_image_note=include_raw_image_note,
        max_window_rows=max_window_rows,
        digits=digits,
    )
    system_content = (
        get_random_student_answer_system_style()
        + "\n\n"
        + _student_system_format_rule(question_type)
        + "\n"
        + STUDENT_OUTPUT_CLEANLINESS_RULE
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


__all__ = [
    "EXPERIMENTAL_PROMPT_STYLES",
    "build_student_answer_messages_experimental",
    "build_student_answer_prompt_experimental",
    "parse_student_answer",
    "prompt_from_messages",
]
