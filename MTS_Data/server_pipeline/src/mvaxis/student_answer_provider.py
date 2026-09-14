from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from .answer_schema import normalize_answer_label
from .teacher_answer_provider import (
    ANSWER_CHANNEL_INSPECTION_INSTRUCTION,
    ANSWER_SYSTEM_STYLES,
)


AXIS_HINT_TOKEN_PLACEHOLDER = "[[AXIS_HINT_TOKENS]]"
AXIS_GLOBAL_HINT_TOKEN_PLACEHOLDER = "[[AXIS_GLOBAL_HINT_TOKENS]]"
AXIS_CHANNEL_HINT_TOKEN_PLACEHOLDER = "[[AXIS_CHANNEL_HINT_TOKENS]]"
STUDENT_OUTPUT_CLEANLINESS_RULE = (
    "Output cleanliness rule: Answer in English unless the question explicitly requests another language. "
    "Do not include local file paths, Markdown image links, image tags, or embedded image references in your answer."
)

STUDENT_MULTIPLE_CHOICE_ANSWER_TEMPLATES = [
    """You are analyzing a time series window for anomaly detection.

Context:
- Analysis window in the full time series: {window_start} to {window_end}
- Window size: {window_size} points
- Global time series context: {global_information}
- Data [current_value]: [{data_str}]

Question: {question}

Answer this multiple-choice question using only the observed target-window values, the visual reference if provided, and the hidden AXIS hints when available.

CRITICAL Output Format:
1. The FIRST line must be exactly: "Answer: [option letter]".
2. The SECOND line must be exactly: "Analysis:"
3. After "Analysis:", write one short paragraph in at most 120 words.
4. Commit to one option letter immediately. Do not delay the answer label until the end.
5. Do not write any sentence, hedge, or explanation before the "Answer:" line.
6. Do not use formats such as "The best answer is A", "I choose B", or "This looks like C" in place of the first line.
7. Explain why the chosen option fits the observed interval better than the main alternatives.
8. Use visible pattern language such as "sharp rise", "broad dip", "sustained shift", "noisy fluctuations", or "coordinated movement".
9. You may include one or two approximate numeric cues such as rough ranges, rough duration, or rough change size, but do not quote exact decimals.
10. Use global positions in the full time series for any location reference.
11. Valid example:
Answer: B
Analysis:
The interval shows ...
12. {channel_instruction}""",
]

STUDENT_TRUE_FALSE_ANSWER_TEMPLATES = [
    """You are answering a judgment question about a time series window.

Context:
- Analysis window in the full time series: {window_start} to {window_end}
- Window size: {window_size} points
- Global time series context: {global_information}
- Data [current_value]: [{data_str}]

Question: {question}

Answer this judgment question using only the observed target-window values, the visual reference if provided, and the hidden AXIS hints when available.

CRITICAL Output Format:
1. The FIRST line must be exactly either "Answer: True" or "Answer: False".
2. The SECOND line must be exactly: "Analysis:"
3. After "Analysis:", write one short paragraph in at most 120 words.
4. Commit to True or False immediately. Do not delay the answer label until the end.
5. Do not write any sentence, hedge, or explanation before the "Answer:" line.
6. Do not use formats such as "Yes.", "No.", "This is true", or "This claim is false" in place of the first line.
7. Explain why the observed interval supports or rejects the claim in the question.
8. Use visual evidence language such as "sustained drop", "isolated spike", "coordinated shift", or "background fluctuation".
9. You may include one or two approximate numeric cues such as rough ranges, rough duration, or rough change size, but do not quote exact decimals.
10. Use global positions in the full time series for any location reference.
11. Valid example:
Answer: False
Analysis:
The interval does not show ...
12. {channel_instruction}""",
]

STUDENT_OPEN_ENDED_ANSWER_TEMPLATES = [
    """You are analyzing a time series window to answer an open-ended anomaly question.

Context:
- Analysis window in the full time series: {window_start} to {window_end}
- Window size: {window_size} points
- Global time series context: {global_information}
- Data [current_value]: [{data_str}]

Question: {question}

Answer this open-ended question using only the observed target-window values, the visual reference if provided, and the hidden AXIS hints when available.

CRITICAL Output Format:
1. Use exactly three sections with these headings in this order:
   Decision:
   Main evidence:
   Interpretation:
2. The first sentence under Decision must be exactly one of:
   - "This interval is anomalous."
   - "This interval is normal."
3. Main evidence should summarize the strongest visible patterns, relevant channels, and global positions.
4. The first sentence under Interpretation must directly answer the question in one or two clear sentences.
5. Keep the full answer within about 150 words.
6. You may include one or two approximate numeric cues such as rough ranges, rough duration, or rough change size, but do not quote exact decimals.
7. Use global positions in the full time series for any location reference.
8. {channel_instruction}""",
]


def get_random_student_answer_system_style() -> str:
    """Return an AXIS-style system role for student answer generation."""

    return random.choice(ANSWER_SYSTEM_STYLES)


def _student_system_format_rule(question_type: str) -> str:
    if question_type == "multiple_choice":
        return (
            "Mandatory response schema for this multiple-choice question: "
            "the first line must be exactly 'Answer: <single option letter>', "
            "the second line must be exactly 'Analysis:', and no explanation may appear before the Answer line."
        )
    if question_type == "true_false":
        return (
            "Mandatory response schema for this judgment question: "
            "the first line must be exactly 'Answer: True' or 'Answer: False', "
            "the second line must be exactly 'Analysis:', and no explanation may appear before the Answer line."
        )
    return (
        "Mandatory response schema for this open-ended question: "
        "use exactly three sections in this order: Decision:, Main evidence:, Interpretation:. "
        "The first sentence under Decision must be exactly 'This interval is anomalous.' or 'This interval is normal.'."
    )


def _question_text(sample: Dict[str, Any]) -> str:
    question = sample.get("question")
    if question is None:
        windows = sample.get("windows") or []
        if windows and isinstance(windows[0], dict):
            question = windows[0].get("question")
    return "" if question is None else str(question).strip()


def _question_answer_type(sample: Dict[str, Any]) -> str:
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return str(
        sample.get("question_answer_type")
        or fact.get("question_answer_type")
        or sample.get("question_type")
        or "open"
    ).lower()


def _template_question_type(sample: Dict[str, Any]) -> str:
    answer_type = _question_answer_type(sample)
    question_type = str(sample.get("question_type") or "").lower()
    question = _question_text(sample).lower()
    if "choice" in answer_type or "multiple" in answer_type or "choice" in question_type or "choices:" in question:
        return "multiple_choice"
    if (
        "judgment" in answer_type
        or "true_false" in answer_type
        or "true-false" in answer_type
        or "true_false" in question_type
        or "judgment" in question_type
        or "yes or no" in question
    ):
        return "true_false"
    return "open_ended"


def get_student_answer_template(question_type: str) -> str:
    """Return one legacy AXIS answer template for a student-style answer."""

    if question_type == "multiple_choice":
        return random.choice(STUDENT_MULTIPLE_CHOICE_ANSWER_TEMPLATES)
    if question_type == "true_false":
        return random.choice(STUDENT_TRUE_FALSE_ANSWER_TEMPLATES)
    if question_type == "open_ended":
        return random.choice(STUDENT_OPEN_ENDED_ANSWER_TEMPLATES)
    raise ValueError(f"Unknown question type: {question_type}")


def _target_interval(sample: Dict[str, Any]) -> Tuple[int, int]:
    interval = sample.get("target_interval") or {}
    if isinstance(interval, dict) and "start" in interval and "end" in interval:
        return int(interval.get("start", 0)), int(interval.get("end", 0))
    windows = sample.get("windows") or []
    if windows and isinstance(windows[0], dict):
        window = windows[0]
        if "start" in window and "end" in window:
            return int(window.get("start", 0)), int(window.get("end", 0))
        if "window_start" in window and "window_end" in window:
            return int(window.get("window_start", 0)), int(window.get("window_end", 0))
    return 0, 0


def _values(sample: Dict[str, Any]) -> List[List[float]]:
    series = sample.get("series") or {}
    values = series.get("values")
    if values is None:
        values = (sample.get("original_data") or {}).get("time_series")
    if values is None:
        raise ValueError("Sample does not contain series.values or original_data.time_series")
    return values


def _channel_ids(sample: Dict[str, Any], width: int) -> List[str]:
    channels = sample.get("channels") or []
    ids = [str(ch.get("channel_id", f"ch_{idx}")) for idx, ch in enumerate(channels[:width])]
    while len(ids) < width:
        ids.append(f"ch_{len(ids)}")
    return ids


def _round_value(value: Any, digits: int) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _row_indices(start: int, end: int, max_rows: Optional[int]) -> List[int]:
    if max_rows is None or max_rows <= 0 or end - start <= max_rows:
        return list(range(start, end))
    if max_rows == 1:
        return [start]
    span = end - start - 1
    return sorted({start + round(i * span / (max_rows - 1)) for i in range(max_rows)})


def format_observed_data_str(
    sample: Dict[str, Any],
    *,
    digits: int = 3,
    max_window_rows: Optional[int] = None,
    include_anomaly_scores: bool = False,
    score_digits: int = 3,
) -> str:
    """Format only observed target-window values, without normal counterfactuals."""

    values = _values(sample)
    start, end = _target_interval(sample)
    end = min(end, len(values))
    if start < 0 or start >= end:
        raise ValueError(f"Invalid target interval: {sample.get('target_interval')}")

    width = len(values[0]) if values and isinstance(values[0], list) else 1
    channel_ids = _channel_ids(sample, width)
    row_indices = _row_indices(start, end, max_window_rows)
    score_context = sample.get("anomaly_score_context") or {}
    lines = []
    if max_window_rows and end - start > max_window_rows:
        lines.append(f"evenly sampled rows from target window [{start}, {end})")

    for idx in row_indices:
        observed_row = values[idx]
        if not isinstance(observed_row, list):
            observed_row = [observed_row]
        cells = [
            f"{channel_ids[ch_idx]}={_round_value(observed, digits)}"
            for ch_idx, observed in enumerate(observed_row)
        ]
        lines.append(f"t={idx}: " + "; ".join(cells))
    return "\n".join(lines)


def format_anomaly_score_text(
    sample: Dict[str, Any],
    *,
    max_window_rows: Optional[int] = None,
    score_digits: int = 3,
) -> str:
    """Format anomaly-score text as a separate detector-evidence block."""

    score_context = sample.get("anomaly_score_context") or {}
    score_values = score_context.get("scores") or []
    if not score_values:
        return ""

    start, end = _target_interval(sample)
    end = min(end, len(score_values))
    if start < 0 or start >= end:
        return ""

    row_indices = _row_indices(start, end, max_window_rows)
    aggregation = str(score_context.get("aggregation") or "max")
    lines = [
        "### Detector anomaly-score estimate",
        (
            "This anomaly score is produced by the AXIS/TimeRCD detector using "
            f"{aggregation} aggregation across channels. Higher anomaly score means the detector considers that time point more suspicious."
        ),
        "Treat this score as auxiliary detector evidence, not as an additional sensor channel.",
        "The channel ids in the Data block use the same naming as the channel labels in the figure.",
    ]
    if max_window_rows and end - start > max_window_rows:
        lines.append(f"Sampled anomaly scores from target window [{start}, {end}):")
    else:
        lines.append(f"Anomaly scores from target window [{start}, {end}):")
    for idx in row_indices:
        lines.append(f"t={idx}: anomaly_score={float(score_values[idx]):.{score_digits}f}")
    return "\n".join(lines)


def _global_information(sample: Dict[str, Any]) -> str:
    start, end = _target_interval(sample)
    values = _values(sample)
    width = len(values[0]) if values and isinstance(values[0], list) else 1
    descriptor = (sample.get("original_data") or {}).get("global_descriptor")
    if descriptor:
        return str(descriptor)
    return (
        f"Multivariate time series with {width} channels. "
        f"The highlighted analysis window spans global positions {start} to {max(start, end - 1)}."
    )


def _format_hint_trace(
    trace: Optional[Dict[str, Any]],
    *,
    include_global_hints: bool = True,
    include_channel_hints: bool = True,
) -> str:
    if not trace:
        return "- Hint source trace is unavailable; hidden hint tokens are still injected when provided."
    global_trace = trace.get("global") or {}
    channel_trace = trace.get("channel") or {}
    sampled_times = global_trace.get("sampled_time_indices") or []
    sampled_positions = channel_trace.get("sampled_positions") or []
    position_parts = []
    for item in sampled_positions[:24]:
        time_index = item.get("time_index")
        channel_id = item.get("channel_id")
        rank = item.get("rank")
        if rank is None:
            position_parts.append(f"(t={time_index}, ch={channel_id})")
        else:
            position_parts.append(f"(t={time_index}, ch={channel_id}, rank={rank})")
    if len(sampled_positions) > 24:
        position_parts.append(f"... {len(sampled_positions) - 24} more positions hidden")
    position_text = ", ".join(position_parts) if position_parts else "none"
    lines = []
    if include_global_hints:
        lines.append(
            (
                "- Overall Summary Hints source: "
                f"interval={global_trace.get('source_interval')}, "
                f"sampled_time_indices={sampled_times}, "
                f"num_tokens={int(global_trace.get('num_global_tokens') or 0)}"
            )
        )
    if include_channel_hints:
        lines.append(
            (
                "- Per-Channel Analysis Hints source: "
                f"interval={channel_trace.get('source_interval')}, "
                f"sampled_positions={position_text}, "
                f"num_tokens={int(channel_trace.get('num_channel_tokens') or 0)}"
            )
        )
    return "\n".join(lines) if lines else "- AXIS embedding hints are disabled for this ablation."


def _contextual_hint_block(
    trace: Optional[Dict[str, Any]],
    *,
    use_axis_hints: bool,
    include_global_hints: bool = True,
    include_channel_hints: bool = True,
) -> str:
    if use_axis_hints and include_global_hints:
        global_tokens = AXIS_GLOBAL_HINT_TOKEN_PLACEHOLDER
    else:
        global_tokens = "(not provided)"
    if use_axis_hints and include_channel_hints:
        channel_tokens = AXIS_CHANNEL_HINT_TOKEN_PLACEHOLDER
    else:
        channel_tokens = "(not provided)"
    return "\n".join(
        [
            "### Contextual Hints",
            f"- **Overall Summary Hints:** {global_tokens}",
            f"- **Per-Channel Analysis Hints:** {channel_tokens}",
            "",
            "Hint Source Trace:",
            _format_hint_trace(
                trace,
                include_global_hints=include_global_hints,
                include_channel_hints=include_channel_hints,
            ),
        ]
    )


def _globalize_position_guidance(text: str, *, window_start: int, window_end: int) -> str:
    start = int(window_start)
    end_inclusive = max(start, int(window_end) - 1)
    global_span = f"global positions {start} to {end_inclusive}"
    replacements = [
        (r"positions 0 to \d+ within this window", global_span),
        (r"positions 0 to \d+ within window", global_span),
        (r"window positions 0-\d+", global_span),
        (r"window \(0 to \d+\)", "the full time series"),
        (r"window-relative positions \(0 to \d+\)", "global positions in the full time series"),
        (r"window positions \(0 to \d+\)", "global positions in the full time series"),
        (r"relative to window \(0 to \d+\)", "using global positions in the full time series"),
        (r"Use window-relative positions", "Use global positions in the full time series"),
        (r"Reference window positions", "Reference global positions in the full time series"),
        (r"All position references must be relative to window", "All position references must use global positions in the full time series"),
    ]
    rewritten = text
    for pattern, replacement in replacements:
        rewritten = re.sub(pattern, replacement, rewritten)
    return (
        rewritten
        + "\n\nCRITICAL Global Position and Channel Rule:\n"
        + f"- Use global positions from the full time series for every location reference; the target window is {global_span}.\n"
        + "- Do not renumber locations relative to the highlighted window.\n"
        + "- When discussing channel-level behavior, explicitly name the relevant channel ids/names whenever they are available.\n"
        + f"- {ANSWER_CHANNEL_INSPECTION_INSTRUCTION}"
    )


def _studentize_legacy_prompt(text: str, hint_block: str, *, window_start: int, window_end: int) -> str:
    """Remove teacher-only anomaly labels and insert AXIS-style hidden hint slots."""

    inserted_hint_block = False
    lines = []
    truth_prefixes = (
        "- Anomaly Description:",
        "- Anomaly info:",
        "- Anomaly details:",
    )
    for line in text.splitlines():
        if line.strip().startswith(truth_prefixes):
            if not inserted_hint_block:
                lines.extend(hint_block.splitlines())
                inserted_hint_block = True
            continue
        lines.append(line)
    rewritten = "\n".join(lines)
    if not inserted_hint_block:
        rewritten = rewritten.replace("\nQuestion:", f"\n\n{hint_block}\n\nQuestion:", 1)

    replacements = {
        "Data [current_value(normal_value)]": "Data [current_value]",
        "Data values [current(normal)]": "Data values [current]",
        "Values [current(normal)]": "Values [current]",
        "according to the Anomaly Description field": "based on the observed data and AXIS hints",
        "according to the Anomaly info field": "based on the observed data and AXIS hints",
        "according to the Anomaly details field": "based on the observed data and AXIS hints",
        "according to the Anomaly Description": "based on the observed data and AXIS hints",
        "according to the Anomaly info": "based on the observed data and AXIS hints",
        "according to the Anomaly details": "based on the observed data and AXIS hints",
        "Anomaly Description field": "observed data and AXIS hints",
        "Anomaly info field": "observed data and AXIS hints",
        "Anomaly details field": "observed data and AXIS hints",
    }
    for old, new in replacements.items():
        rewritten = rewritten.replace(old, new)
    rewritten = _globalize_position_guidance(rewritten, window_start=window_start, window_end=window_end)
    return rewritten


def build_student_answer_prompt(
    sample: Dict[str, Any],
    *,
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
    """Build an AXIS-style student-answer prompt without truth anomaly labels.

    Evidence cards are intentionally not inserted into student prompts. The
    include_evidence_card argument is retained only for compatibility with
    older scripts.
    """

    start, end = _target_interval(sample)
    question = _question_text(sample)
    question_type = _template_question_type(sample)
    template = get_student_answer_template(question_type)
    hint_block = _contextual_hint_block(
        embedding_hint_trace,
        use_axis_hints=use_axis_hints,
        include_global_hints=include_global_hints,
        include_channel_hints=include_channel_hints,
    )
    format_dict = {
        "window_size": max(0, end - start),
        "window_size_minus_one": max(0, end - start - 1),
        "global_information": _global_information(sample),
        "anomaly_description": "Hidden for student inference; use only observed values and AXIS hints.",
        "data_str": format_observed_data_str(
            sample,
            digits=digits,
            max_window_rows=max_window_rows,
            include_anomaly_scores=include_anomaly_score_text,
        ),
        "question": question,
        "window_start": start,
        "window_end": max(start, end - 1),
        "channel_instruction": ANSWER_CHANNEL_INSPECTION_INSTRUCTION,
    }
    rewritten = _studentize_legacy_prompt(
        template.format(**format_dict),
        hint_block,
        window_start=start,
        window_end=end,
    )
    if include_anomaly_score_text:
        score_block = format_anomaly_score_text(
            sample,
            max_window_rows=max_window_rows,
            score_digits=digits,
        )
        if score_block:
            rewritten = rewritten.replace("\nQuestion:", f"\n\n{score_block}\n\nQuestion:", 1)
            if score_block not in rewritten:
                rewritten = rewritten + "\n\n" + score_block
    if include_anomaly_score_image_note:
        image_note = (
            "Visual anomaly-score reference: The companion figure shows different channel trajectories "
            "using the same channel ids as the Data block, plus a detector anomaly-score curve for the "
            "highlighted interval. Higher anomaly score means more suspicious according to the detector. "
            "Treat the score curve as auxiliary detector evidence rather than as a sensor channel, and do not "
            "quote or reproduce any image path."
        )
        rewritten = rewritten.replace("\nQuestion:", f"\n\n{image_note}\n\nQuestion:", 1)
        if image_note not in rewritten:
            rewritten = rewritten + "\n\n" + image_note
    if include_raw_image_note:
        image_note = (
            "Visual time-series reference: The companion figure shows different channel trajectories using "
            "the same channel ids as the Data block. The yellow region and red curve segments mark the target "
            "interval; no anomaly-score curve is included. Use the figure only as visual support for the target-window "
            "values in Data, and do not quote or reproduce any image path."
        )
        rewritten = rewritten.replace("\nQuestion:", f"\n\n{image_note}\n\nQuestion:", 1)
        if image_note not in rewritten:
            rewritten = rewritten + "\n\n" + image_note
    rewritten = rewritten + "\n\n" + STUDENT_OUTPUT_CLEANLINESS_RULE
    return rewritten


def build_student_answer_messages(
    sample: Dict[str, Any],
    *,
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
    """Build chat messages for local/API student answer generation."""

    question_type = _template_question_type(sample)
    prompt = build_student_answer_prompt(
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


def prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Return a readable preview of text chat messages."""

    return "\n\n".join(
        f"{str(message.get('role', 'user')).upper()}:\n{str(message.get('content', ''))}"
        for message in messages
    )


def _answer_style(question_type: Optional[str], question: str) -> str:
    question_type = (question_type or "").lower()
    lower_question = question.lower()
    if "choice" in question_type or "choices:" in lower_question:
        return "choice"
    if "judgment" in question_type or "true_false" in question_type or "yes or no" in lower_question:
        return "judgment"
    return "open"


def _extract_final_answer(text: str) -> str:
    matches = list(re.finditer(r"Answer\s*:\s*(.+)", text, flags=re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip()
    return text.strip()


_CHOICE_LABEL_PATTERNS = [
    re.compile(
        r"(?i)\b(?:answer|final answer|option|choice|selected option|correct option)\s*(?:is|:)?\s*\(?\s*([A-F])\s*\)?\b"
    ),
    re.compile(
        r"(?i)\b(?:the\s+)?(?:answer|correct answer|best answer|best option)\s+(?:should\s+be|is)\s*\(?\s*([A-F])\s*\)?\b"
    ),
    re.compile(r"(?im)^\s*(?:final answer\s*[:\-]\s*)?\(?\s*([A-F])\s*\)?\s*[\.)]\s+"),
    re.compile(r"(?i)\bI\s*(?:would\s*)?(?:choose|select|pick)\s*\(?\s*([A-F])\s*\)?\b"),
]


def _scan_choice_label(text: str) -> Optional[str]:
    answer_line = re.search(
        r"(?im)^\s*answer\s*:\s*\(?\s*([A-F])\s*\)?\b",
        text or "",
    )
    if answer_line:
        return normalize_answer_label(answer_line.group(1))
    matches = []
    for pattern in _CHOICE_LABEL_PATTERNS:
        matches.extend(pattern.finditer(text or ""))
    if matches:
        match = sorted(matches, key=lambda item: item.start())[0]
        return normalize_answer_label(match.group(1))
    first = re.match(r"\s*\(?\s*([A-Fa-f])\s*\)?\s*(?:[\.)]|$)", text or "")
    return normalize_answer_label(first.group(1)) if first else None


def _scan_judgment_label(text: str) -> Optional[str]:
    answer_line = re.search(
        r"(?im)^\s*answer\s*:\s*(?:is\s*)?(yes|no|true|false)\b",
        text or "",
    )
    if answer_line:
        return normalize_answer_label(answer_line.group(1))
    matches = list(
        re.finditer(
            r"(?i)\b(?:answer|final answer|judgment|label)\s*[:\-]?\s*(?:is\s*)?(yes|no|true|false)\b",
            text or "",
        )
    )
    matches.extend(
        re.finditer(
            r"(?i)\b(?:the\s+)?(?:answer|judgment)\s+(?:should\s+be|is)\s+(yes|no|true|false)\b",
            text or "",
        )
    )
    if matches:
        match = sorted(matches, key=lambda item: item.start())[0]
        return normalize_answer_label(match.group(1))
    direct = re.match(r"\s*(yes|no|true|false)\b", text or "", flags=re.IGNORECASE)
    return normalize_answer_label(direct.group(1)) if direct else None


def parse_student_answer(
    text: str,
    *,
    question_type: Optional[str] = None,
    question: str = "",
) -> Dict[str, Any]:
    """Parse an AXIS-style natural-language student answer into a light schema."""

    answer_text = text.strip()
    final_answer = _extract_final_answer(answer_text)
    style = _answer_style(question_type, question)
    answer_label = None
    if style == "choice":
        answer_label = _scan_choice_label(answer_text) or normalize_answer_label(final_answer)
    elif style == "judgment":
        answer_label = _scan_judgment_label(answer_text) or normalize_answer_label(final_answer)
    return {
        "answer_text": answer_text,
        "final_answer": final_answer,
        "answer_label": answer_label,
        "question_answer_type": style,
    }
