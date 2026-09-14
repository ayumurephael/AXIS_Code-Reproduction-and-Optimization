from __future__ import annotations

import base64
import json
import mimetypes
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FIELD_MEANINGS: Dict[str, str] = {
    "target_interval": "The global time-index range [start, end) selected for this question. The data_str below contains only this target window.",
    "is_anomalous": "Whether the target window overlaps a true labeled anomaly.",
    "root_cause_channel": "The direct injected or primary abnormal channel for this target window, if known.",
    "root_cause_channels": "All direct injected or primary abnormal channels for this target window.",
    "affected_channels": "Channels whose labels indicate abnormal behavior inside the target window.",
    "anomaly_type": "The generator-side anomaly type or legacy morphology label.",
    "anomaly_scope": "The labeled scale of the anomaly: node means one channel, edge means a small relation/local group, and subgraph means a broader multi-channel event.",
    "abnormal_edges": "Teacher-only relation labels attached to the anomaly. Use them only to write the ground-truth teacher answer, not as visible evidence for student inference.",
    "causal_path": "Teacher-only ordered channel set/path associated with the labeled abnormal event, when available.",
    "root_cause": "The root-cause metadata copied from the sample.",
}

FACT_CHECK_TEACHER_KEYS = [
    "is_anomalous",
    "root_cause_channel",
    "root_cause_channels",
    "affected_channels",
    "anomaly_type",
    "anomaly_scope",
    "abnormal_edges",
    "causal_path",
]

ANSWER_CHANNEL_INSPECTION_INSTRUCTION = (
    "Do not assume that channels named in the question are abnormal. "
    "Inspect all channels and try to distinguish root channels, affected channels, "
    "distractor channels, and normal background trends, but it is unnecessary to "
    "always report them unless you are asked to answer root cause or affected channels."
)

ANSWER_SYSTEM_STYLES = [
    "You are a helpful assistant that analyzes time series patterns and generates question-answer pairs.",
    "You are an expert in time series analysis specializing in providing clear and accurate answers about anomalies.",
    "You are a data science assistant that provides insightful analysis of time series anomalies.",
    "You are a technical analyst explaining time series patterns and anomaly characteristics.",
    "You are a machine learning specialist providing detailed answers about time series anomaly detection.",
    "You are an analytical assistant that explains time series patterns and anomaly indicators.",
    "You are a domain expert in time series analysis, providing comprehensive answers about anomalies.",
    "You are a research assistant explaining time series anomaly detection findings.",
    "You are a data analyst specializing in interpreting time series patterns and anomalies.",
    "You are a technical assistant that provides thorough analysis of anomalies in time series data.",
    "You are an AI assistant expert in time series analysis, explaining anomaly patterns and characteristics.",
    "You are a specialist in time series data, providing detailed explanations of anomalous patterns.",
    "You are a professional analyst for time series anomaly detection, offering clear and accurate explanations.",
    "You are an assistant with expertise in time series analysis, explaining how to identify and understand anomalies.",
    "You are a data scientist providing comprehensive answers about anomaly detection in time series.",
]


MULTIPLE_CHOICE_ANSWER_TEMPLATES = [
    """You are analyzing a time series window for anomaly detection.

Context:
- Analysis window in the full time series: {window_start} to {window_end}
- Window size: {window_size} points
- Global time series context: {global_information}
- Ground-truth anomaly description for teacher answering: {anomaly_description}
- Data [current_value(normal_value)]: [{data_str}]

Question: {question}

Generate a structured teacher answer for this multiple choice question.

Internal reasoning goals:
1. Inspect the plotted traces and the interval's rough numeric behavior in the highlighted interval and nearby context.
2. Compare the answer options against the multichannel trace behavior and any useful rough quantitative cues.
3. Choose the single option that best matches the interval.

CRITICAL Output Format:
1. The FIRST line must be exactly: "Answer: [option letter]".
2. After that, write one short reasoning paragraph in at most 120 words.
3. Do not repeat multiple options. Commit to one option letter immediately.
4. Use time-series evidence language such as "sharp rise", "broad dip", "sustained shift", "noisy fluctuations", or "coordinated movement", and add one or two rough numeric cues when they sharpen the explanation.
5. You may include one or two approximate numeric cues such as rough ranges, rough duration, or rough change size, but do not quote exact decimals.
6. Do NOT mention "expected value" or "normal value" in the answer body.
7. Explain why the chosen option fits better than the main alternatives using evidence from the plotted traces and, when helpful, rough numeric interval cues.
8. Use global positions in the full time series for any location reference.
9. {channel_instruction}""",
]


TRUE_FALSE_ANSWER_TEMPLATES = [
    """You are answering a yes/no anomaly-analysis question about this time series window.

Context:
- Analysis window in the full time series: {window_start} to {window_end}
- Window size: {window_size} points
- Global time series context: {global_information}
- Ground-truth anomaly description for teacher answering: {anomaly_description}
- Data [current_value(normal_value)]: [{data_str}]

Question: {question}

Generate a structured teacher answer for this judgment question.

Internal reasoning goals:
1. Understand the claim made by the question.
2. Compare that claim with the multichannel trace evidence and the interval's rough numeric behavior.
3. Decide whether the correct answer is Yes or No.

CRITICAL Output Format:
1. The FIRST line must be exactly either "Yes." or "No.".
2. After that, write one short reasoning paragraph in at most 120 words.
3. Do not hedge before the first line.
4. Use time-series evidence language such as "sustained drop", "isolated spike", "coordinated shift", or "background fluctuation", and add one or two rough numeric cues when they sharpen the explanation.
5. You may include one or two approximate numeric cues such as rough ranges, rough duration, or rough change size, but do not quote exact decimals.
6. Do NOT mention "expected value" or "normal value" in the answer body.
7. Explain why the interval supports or rejects the claim in the question.
8. Use global positions in the full time series for any location reference.
9. {channel_instruction}""",
]


OPEN_ENDED_ANSWER_TEMPLATES = [
    """You are analyzing a time series window to answer an open-ended anomaly question.

Context:
- Analysis window in the full time series: {window_start} to {window_end}
- Window size: {window_size} points
- Global time series context: {global_information}
- Ground-truth anomaly description for teacher answering: {anomaly_description}
- Data [current_value(normal_value)]: [{data_str}]

Question: {question}

Generate a structured teacher answer for this open-ended question.

Internal reasoning goals:
1. Decide whether the interval should be treated as anomalous or normal.
2. Identify the strongest trace, timing, shape, and rough-magnitude evidence in the relevant channels and positions.
3. Explain the most plausible interpretation of the pattern in a stable, reusable format.

CRITICAL Output Format:
1. Use exactly three sections with these headings in this order:
   Decision:
   Main evidence:
   Interpretation:
2. The first sentence under Decision must be exactly one of:
   - "This interval is anomalous."
   - "This interval is normal."
3. Keep the full answer within about 140 words.
4. Main evidence should focus on the clearest trace, timing, shape, and rough-magnitude cues, along with the relevant channels and global positions.
5. Interpretation should explain the likely meaning of those trace and magnitude cues, not restate all evidence.
6. You may include one or two approximate numeric cues such as rough ranges, rough duration, or rough change size, but do not quote exact decimals.
7. Do NOT mention "expected value" or "normal value" in the answer body.
8. Use global positions in the full time series for any location reference.
9. {channel_instruction}""",
]


def build_anomaly_description_json(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Return the current truth JSON as the teacher-side anomaly description."""

    target_output = sample.get("target_output") or {}
    fact_check = target_output.get("fact_check") or {}
    return {
        "target_interval": sample.get("target_interval"),
        "fact_check": {
            key: fact_check.get(key)
            for key in FACT_CHECK_TEACHER_KEYS
            if key in fact_check and fact_check.get(key) is not None
        },
        "root_cause": sample.get("root_cause") or {},
    }


def format_field_meanings(anomaly_json: Dict[str, Any]) -> str:
    keys: List[str] = ["target_interval"]
    fact = anomaly_json.get("fact_check") or {}
    keys.extend(str(key) for key in fact.keys())
    keys.extend(["root_cause"])

    seen = set()
    lines = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        meaning = FIELD_MEANINGS.get(key, "Dataset field copied from the current sample.")
        lines.append(f"- {key}: {meaning}")
    return "\n".join(lines)


def get_random_answer_system_style() -> str:
    """Get a random system message style for answer generation."""

    return random.choice(ANSWER_SYSTEM_STYLES)


def _template_question_type(sample: Dict[str, Any]) -> str:
    answer_type = _question_answer_type(sample)
    question_type = str(sample.get("question_type") or "").lower()
    if "choice" in answer_type or "multiple" in answer_type or "choice" in question_type:
        return "multiple_choice"
    if (
        "judgment" in answer_type
        or "true_false" in answer_type
        or "true-false" in answer_type
        or "true_false" in question_type
        or "judgment" in question_type
    ):
        return "true_false"
    return "open_ended"


def get_answer_template(question_type: str) -> str:
    """Get a random answer template for the specified question type."""

    if question_type == "multiple_choice":
        return random.choice(MULTIPLE_CHOICE_ANSWER_TEMPLATES)
    if question_type == "true_false":
        return random.choice(TRUE_FALSE_ANSWER_TEMPLATES)
    if question_type == "open_ended":
        return random.choice(OPEN_ENDED_ANSWER_TEMPLATES)
    raise ValueError(f"Unknown question type: {question_type}")


def _globalize_position_guidance(text: str, *, window_start: Optional[int], window_end: Optional[int]) -> str:
    """Rewrite legacy window-relative position guidance to global-position guidance."""

    if window_start is None or window_end is None:
        return text
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


def format_answer_prompt(
    template: str,
    window_size: int,
    global_information: str,
    anomaly_description: str,
    data_str: str,
    question: str,
    has_anomaly: bool,
    window_start: Optional[int] = None,
    window_end: Optional[int] = None,
) -> str:
    """Format an AXIS answer template with the provided parameters."""

    window_size_minus_one = window_size - 1
    if not has_anomaly:
        anomaly_description = "目标序列实际上没有异常"

    format_dict = {
        "window_size": window_size,
        "window_size_minus_one": window_size_minus_one,
        "global_information": global_information,
        "anomaly_description": anomaly_description,
        "data_str": data_str,
        "question": question,
        "channel_instruction": ANSWER_CHANNEL_INSPECTION_INSTRUCTION,
    }
    if window_start is not None:
        format_dict["window_start"] = window_start
    if window_end is not None:
        format_dict["window_end"] = window_end
    return _globalize_position_guidance(
        template.format(**format_dict),
        window_start=window_start,
        window_end=window_end,
    )


def _target_interval(sample: Dict[str, Any]) -> Tuple[int, int]:
    interval = sample.get("target_interval") or {}
    return int(interval.get("start", 0)), int(interval.get("end", 0))


def _values(sample: Dict[str, Any]) -> List[List[float]]:
    series = sample.get("series") or {}
    values = series.get("values")
    if values is None:
        values = (sample.get("original_data") or {}).get("time_series")
    if values is None:
        raise ValueError("Sample does not contain series.values or original_data.time_series")
    return values


def _normal_values(sample: Dict[str, Any]) -> Optional[List[List[float]]]:
    normal = sample.get("normal_series")
    if normal is None:
        original = sample.get("original_data") or {}
        normal = original.get("normal_series")
        if normal is None:
            normal = original.get("normal_time_series")
    return normal


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


def format_data_str(
    sample: Dict[str, Any],
    *,
    digits: int = 2,
    max_window_rows: Optional[int] = None,
) -> str:
    """Format target-window values as observed(normal_counterfactual), AXIS-style."""

    values = _values(sample)
    normal = _normal_values(sample)
    start, end = _target_interval(sample)
    end = min(end, len(values))
    if start < 0 or start >= end:
        raise ValueError(f"Invalid target interval: {sample.get('target_interval')}")

    width = len(values[0]) if values and isinstance(values[0], list) else 1
    channel_ids = _channel_ids(sample, width)
    row_indices = _row_indices(start, end, max_window_rows)
    lines = []
    if max_window_rows and end - start > max_window_rows:
        lines.append(f"evenly sampled rows from target window [{start}, {end})")

    for idx in row_indices:
        observed_row = values[idx]
        if not isinstance(observed_row, list):
            observed_row = [observed_row]
        normal_row = None
        if normal is not None and idx < len(normal):
            normal_row = normal[idx]
            if not isinstance(normal_row, list):
                normal_row = [normal_row]
        cells = []
        for ch_idx, observed in enumerate(observed_row):
            observed_s = _round_value(observed, digits)
            if normal_row is not None and ch_idx < len(normal_row):
                normal_s = _round_value(normal_row[ch_idx], digits)
                cells.append(f"{channel_ids[ch_idx]}={observed_s}({normal_s})")
            else:
                cells.append(f"{channel_ids[ch_idx]}={observed_s}")
        lines.append(f"t={idx}: " + "; ".join(cells))
    return "\n".join(lines)


def _question_answer_type(sample: Dict[str, Any]) -> str:
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return str(
        sample.get("question_answer_type")
        or fact.get("question_answer_type")
        or sample.get("question_type")
        or "open"
    ).lower()


def format_anomaly_description(sample: Dict[str, Any]) -> str:
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    if not bool(fact.get("is_anomalous")):
        return "目标序列实际上没有异常"
    anomaly_json = build_anomaly_description_json(sample)
    return "\n".join(
        [
            "Ground-truth anomaly_description JSON:",
            json.dumps(anomaly_json, ensure_ascii=False, indent=2),
            "",
            "Field meanings for the anomaly_description JSON:",
            format_field_meanings(anomaly_json),
        ]
    )


def _global_information(sample: Dict[str, Any]) -> str:
    start, end = _target_interval(sample)
    values = _values(sample)
    width = len(values[0]) if values and isinstance(values[0], list) else 1
    descriptor = (sample.get("original_data") or {}).get("global_descriptor")
    if descriptor:
        return str(descriptor)
    return (
        f"Multivariate time series with {width} channels. "
        f"The highlighted analysis window spans global positions {start} to {end - 1}."
    )


def build_teacher_answer_prompt(
    sample: Dict[str, Any],
    *,
    max_window_rows: Optional[int] = None,
    digits: int = 2,
) -> str:
    start, end = _target_interval(sample)
    question = str(sample.get("question", "")).strip()
    question_type = _template_question_type(sample)
    template = get_answer_template(question_type)
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return format_answer_prompt(
        template=template,
        window_size=max(0, end - start),
        global_information=_global_information(sample),
        anomaly_description=format_anomaly_description(sample),
        data_str=format_data_str(sample, digits=digits, max_window_rows=max_window_rows),
        question=question,
        has_anomaly=bool(fact.get("is_anomalous")),
        window_start=start,
        window_end=end,
    )


def _image_data_url(path: str) -> Optional[str]:
    if not path:
        return None
    image_path = Path(path)
    if not image_path.exists():
        return None
    mime = mimetypes.guess_type(str(image_path))[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_teacher_answer_messages(
    sample: Dict[str, Any],
    *,
    include_image: bool = True,
    max_window_rows: Optional[int] = None,
    digits: int = 2,
) -> List[Dict[str, Any]]:
    prompt = build_teacher_answer_prompt(sample, max_window_rows=max_window_rows, digits=digits)
    if not include_image:
        return [
            {"role": "system", "content": get_random_answer_system_style()},
            {"role": "user", "content": prompt},
        ]

    image_path = str((sample.get("question_generation_artifacts") or {}).get("image_path") or "")
    data_url = _image_data_url(image_path)
    if data_url is None:
        return [
            {"role": "system", "content": get_random_answer_system_style()},
            {"role": "user", "content": prompt},
        ]
    return [
        {"role": "system", "content": get_random_answer_system_style()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]


def prompt_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Return a readable preview without embedding base64 image bytes."""

    rendered = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    parts.append("[image_url: base64 image attached]")
                else:
                    parts.append(f"[{item.get('type', 'unknown content')}]")
            content_text = "\n".join(parts)
        else:
            content_text = str(content)
        rendered.append(f"{str(message.get('role', 'user')).upper()}:\n{content_text}")
    return "\n\n".join(rendered)
