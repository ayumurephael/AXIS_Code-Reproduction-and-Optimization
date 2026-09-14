from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .data_schema import ModelInput

AXIS_HINT_TOKEN_PLACEHOLDER = "[[AXIS_HINT_TOKENS]]"


def _active_indices(channels: Sequence[Dict[str, Any]]) -> List[int]:
    idx = [i for i, ch in enumerate(channels) if ch.get("active", True)]
    return idx or list(range(len(channels)))


def _channel_brief(channels: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for ch in channels:
        if not ch.get("active", True):
            continue
        unit = ch.get("unit") or "a.u."
        parts.append(f"{ch['channel_id']}({unit})")
    return ", ".join(parts) if parts else "no active channel metadata"


def _sample_rows(start: int, end: int, max_rows: int = 10) -> List[int]:
    if end <= start:
        return []
    if end - start <= max_rows:
        return list(range(start, end))
    return [int(x) for x in np.linspace(start, end - 1, max_rows)]


def _format_window_values(
    values: np.ndarray,
    channels: Sequence[Dict[str, Any]],
    start: int,
    end: int,
    *,
    max_rows: int = 10,
    scale: float = 100.0,
) -> str:
    active = _active_indices(channels)
    header = "t | " + " | ".join(channels[i]["channel_id"] for i in active)
    lines = [header]
    for t in _sample_rows(start, end, max_rows=max_rows):
        scaled = [int(round(float(values[t, i]) * scale)) for i in active]
        lines.append(f"{t} | " + " | ".join(str(x) for x in scaled))
    if end - start > max_rows:
        lines.append(f"... sampled {max_rows} of {end - start} steps")
    return "\n".join(lines)


def _format_evidence(evidence: Dict[str, Any]) -> str:
    if not evidence:
        return "No handcrafted evidence card is available."
    flags = evidence.get("summary_flags") or {}
    visible_flags = {
        key: value
        for key, value in flags.items()
        if key
        in {
            "has_high_freq_oscillation",
            "has_level_shift",
            "has_over_fluctuation",
        }
    }
    lines = [
        "Summary flags: "
        + ", ".join(f"{key}={value}" for key, value in visible_flags.items())
    ]
    for key in [
        "peaks_and_troughs",
        "level_shifts",
        "over_fluctuation",
    ]:
        items = evidence.get(key) or []
        if items:
            lines.append(f"{key}: {json.dumps(items[:3], ensure_ascii=False, separators=(',', ':'))}")
    if len(lines) == 1:
        lines.append("No strong handcrafted local evidence was recognized.")
    return "\n".join(lines)


def _format_interval_context(interval_context: Optional[Dict[str, Any]]) -> str:
    if not interval_context:
        return "No extra proposal context."
    # Keep proposal context readable but decision-neutral. Scores, predicted
    # roots, thresholded anomaly decisions, and channel rankings are hint
    # content and must be delivered either as explicit debug text hints or as
    # AXIS embedding hints, not through this always-visible context field.
    keep = {
        k: v
        for k, v in interval_context.items()
        if k
        in {
            "interval_source",
            "instruction",
            "hint_delivery",
        }
    }
    return json.dumps(keep, ensure_ascii=False, separators=(",", ":"))


def _question_answer_guidance(question_type: Optional[str]) -> str:
    question_type = question_type or ""
    guidance = {
        "anomaly_detection": (
            "Answer whether the observed target interval is anomalous, then cite the strongest evidence from the target window, evidence card, and hints."
        ),
        "root_cause": (
            "Name the most likely directly abnormal channel first, then distinguish it from other affected channels."
        ),
        "affected_channels": (
            "List all affected channels and state whether the observed pattern appears to spread across channels."
        ),
        "mechanism": (
            "Classify whether the abnormality is localized to one channel or spans multiple channels, and name the likely anomaly type."
        ),
        "relation_consistency": (
            "Use only the target window, evidence card, and embedding hints to describe any abnormal channel behavior if supported."
        ),
        "counterfactual_relation": (
            "Use only the target window, evidence card, and embedding hints to describe any abnormal channel behavior if supported."
        ),
        "legacy_axis_explanation": (
            "Answer in AXIS style: judge anomaly presence, locate the interval evidence, and describe the abnormal pattern."
        ),
    }
    if question_type in guidance:
        return guidance[question_type]
    if question_type.startswith("simple_"):
        return "Answer the simple multivariate question directly: decide anomaly/affected channels/local morphology from the observed interval, evidence card, and model hints."
    if question_type.startswith("medium_"):
        return "Answer the exact medium multivariate question directly. For choice questions, choose one listed option; for judgment questions, answer yes or no; for open questions, explain root cause, affected channels, scope, or type only when the question asks for them."
    if question_type.startswith("complex_"):
        return "Answer the complex multivariate question conservatively: use only observed interval evidence and model hints, and keep uncertainty when direct support is weak."
    return "Answer the user's exact question directly, using the observed interval evidence, metadata, and model hints."


def _answer_style_from_question(question_type: Optional[str], question: str) -> str:
    question_type = question_type or ""
    lower_question = question.lower()
    if "_choice_" in question_type or "choices:" in lower_question:
        return "choice"
    if "_judgment_" in question_type or "yes or no" in lower_question:
        return "judgment"
    if "_open_" in question_type:
        return "open"
    return "open"


def _format_soft_hints(hint_summary: Dict[str, Any]) -> str:
    if not hint_summary:
        return "No textual or embedding hints are provided in this run."
    compact: Dict[str, Any] = {}
    if "global_hint" in hint_summary:
        compact["global_hint"] = hint_summary.get("global_hint") or {}
    if "channel_hints" in hint_summary:
        compact["channel_hints"] = (hint_summary.get("channel_hints") or [])[:5]
    if "local_hints" in hint_summary:
        compact["local_hints"] = (hint_summary.get("local_hints") or [])[:5]
    if "interval_proposal" in hint_summary:
        compact["interval_proposal"] = hint_summary.get("interval_proposal") or {}
    if not compact:
        return "All soft textual hint sections are hidden for this ablation."
    return json.dumps(compact, ensure_ascii=False, indent=2)


def _format_embedding_hint_trace(trace: Optional[Dict[str, Any]]) -> str:
    if not trace:
        return (
            "Embedding hint trace is unavailable. Hidden hint embeddings are still injected at "
            f"{AXIS_HINT_TOKEN_PLACEHOLDER} if provided by the caller."
        )
    global_trace = trace.get("global") or {}
    channel_trace = trace.get("channel") or {}
    sampled_times = global_trace.get("sampled_time_indices") or []
    sampled_positions = channel_trace.get("sampled_positions") or []
    global_count = int(global_trace.get("num_global_tokens") or 0)
    channel_count = int(channel_trace.get("num_channel_tokens") or 0)
    position_parts = []
    for item in sampled_positions[:24]:
        time_index = item.get("time_index")
        channel_id = item.get("channel_id")
        rank = item.get("rank")
        if rank is None:
            position_parts.append(f"(t={time_index},ch={channel_id})")
        else:
            position_parts.append(f"(t={time_index},ch={channel_id},rank={rank})")
    if len(sampled_positions) > 24:
        position_parts.append(f"... {len(sampled_positions) - 24} more positions hidden")
    position_text = ", ".join(position_parts) if position_parts else "none"
    global_line = (
        "global: omitted for this ablation"
        if global_count <= 0
        else (
            "global: "
            f"interval={global_trace.get('source_interval')}, "
            f"sampled_time_indices={sampled_times}, "
            f"num_global_tokens={global_count}"
        )
    )
    channel_line = (
        "channel: omitted for this ablation"
        if channel_count <= 0
        else (
            "channel: "
            f"interval={channel_trace.get('source_interval')}, "
            f"num_channel_tokens={channel_count}, "
            f"sampled_positions={position_text}"
        )
    )
    return "\n".join(
        [
            global_line,
            channel_line,
            f"hidden_embedding_placeholder: {AXIS_HINT_TOKEN_PLACEHOLDER}",
        ]
    )


def _embedding_hint_description(trace: Optional[Dict[str, Any]]) -> str:
    global_count = int(((trace or {}).get("global") or {}).get("num_global_tokens") or 0)
    channel_count = int(((trace or {}).get("channel") or {}).get("num_channel_tokens") or 0)
    lines = []
    if global_count > 0:
        lines.append("- Global hidden hints: interval-level embeddings generated from multivariate sequence context.")
    if channel_count > 0:
        lines.append("- Channel hidden hints: per-variable embeddings sampled from the target interval.")
    if not lines:
        lines.append("- Embedding hints: no hidden hint embeddings are injected in this run.")
    lines.append("- The readable prompt shows only source time/channel positions; the hidden vectors are injected at placeholder tokens.")
    lines.append("- Hints are model-derived signals, not ground-truth labels; verify them against the observed target window and evidence card.")
    return "\n".join(lines) + "\n"


def build_prompt(
    model_input: ModelInput,
    hint_summary: Dict[str, Any],
    interval_context: Optional[Dict[str, Any]] = None,
    include_causal_graph: bool = False,
    include_channel_names: bool = False,
    include_channel_roles: bool = False,
    include_channel_relations: bool = False,
    include_soft_hints: bool = True,
    include_evidence_card: bool = True,
    include_normal_counterpart: bool = False,
    embedding_hint_trace: Optional[Dict[str, Any]] = None,
) -> str:
    del include_channel_names, include_channel_roles, include_channel_relations
    start, end = model_input.interval
    question_type = model_input.question_type or "unspecified"
    answer_style = _answer_style_from_question(model_input.question_type, model_input.question)
    values = np.asarray(model_input.values, dtype=float)
    normal_values = (
        np.asarray(model_input.normal_values, dtype=float)
        if model_input.normal_values is not None
        else None
    )
    normal_block = (
        _format_window_values(normal_values, model_input.channels, start, end)
        if include_normal_counterpart and normal_values is not None
        else "Normal counterpart is hidden for realistic inference; use only the observed target window, metadata, hints, and evidence card."
    )
    normal_section = (
        "#### Normal Counterpart Window Values\n"
        f"{normal_block}\n\n"
        if include_normal_counterpart
        else ""
    )
    soft_hint_block = (
        _format_soft_hints(hint_summary)
        if include_soft_hints
        else _format_embedding_hint_trace(embedding_hint_trace)
    )
    hint_line_label = "Textual soft hint summary" if include_soft_hints else "Embedding hint trace"
    if include_soft_hints and not hint_summary:
        hint_description = (
            "- Model hints: no textual or embedding hints are provided in this run; rely on the observed target window and evidence card.\n"
        )
    elif not include_soft_hints:
        hint_description = _embedding_hint_description(embedding_hint_trace)
    else:
        hint_description = (
            "- Global hints: generated by the AXIS time-series encoder and hint tuner.\n"
            "- Channel hints: generated by the AXIS time-series encoder and hint tuner.\n"
            "- Hints are model-derived signals, not ground-truth labels; verify them against the observed target window and evidence card.\n"
        )
    graph_note = ""
    evidence_block = (
        _format_evidence(model_input.evidence_card)
        if include_evidence_card
        else "Evidence card is intentionally hidden for this ablation."
    )
    return (
        "You are an expert multivariate time series analyst. Analyze the provided data and answer the question.\n\n"
        "### Time Series Data\n"
        f"- Window: Steps {start} to {end}\n"
        "- Values are scaled by 100 and shown in channel order.\n"
        "- Counterfactual normal values are not provided in realistic inference.\n"
        f"- Active Channels: {_channel_brief(model_input.channels)}\n\n"
        "#### Target Window Values\n"
        f"{_format_window_values(values, model_input.channels, start, end)}\n\n"
        f"{normal_section}"
        "### Contextual Hints\n"
        f"{hint_description}"
        f"- Proposal context: {_format_interval_context(interval_context)}\n"
        f"- {hint_line_label}: {soft_hint_block}\n\n"
        "### Evidence Card\n"
        f"{evidence_block}"
        f"{graph_note}\n\n"
        "### Question\n"
        f"- Question type: {question_type}\n"
        f"- Expected answer style: {answer_style}\n"
        f"- Required focus: {_question_answer_guidance(model_input.question_type)}\n"
        f"{model_input.question}\n\n"
        "### Answer Format\n"
        "Return exactly one nested JSON object and no markdown. The first character must be { and the last character must be }.\n"
        "Do not use ellipses, placeholders, copied schema text, or truncated phrases. Every text field must be a complete sentence.\n"
        "The JSON object must contain these top-level keys: fact_check, evidence_chain, reasoning_process, reasoning_summary, question_answer, final_answer, abnormality_score, answer_confidence.\n"
        "fact_check must contain these keys: is_anomalous, root_cause_channel, root_cause_channels, affected_channels, anomaly_type, anomaly_scope.\n"
        "Use booleans for is_anomalous; use channel ids such as ch_0 only when supported by the observed target window and hints.\n"
        "If is_anomalous is false, root_cause_channel, anomaly_type, and anomaly_scope must be null, and root_cause_channels and affected_channels must be empty arrays.\n"
        "If is_anomalous is false, question_answer and final_answer must explicitly say that no anomaly/root/affected mechanism is supported.\n"
        "Except for evidence_chain and reasoning_process, explanation fields must be JSON strings, not arrays of fragments.\n"
        "evidence_chain must be an array of 3 to 5 complete sentences based only on the observed target window, metadata, evidence card, and model hints; do not invent or reference hidden normal-counterpart values.\n"
        "reasoning_process must be an array of 3 to 5 concise observable reasoning steps. Keep it as a short rationale summary, not hidden chain-of-thought, and do not reveal labels or normal-counterpart values.\n"
        "reasoning_summary must be 2 to 4 complete sentences that explain the evidence without exposing hidden labels or assuming hints are ground truth.\n"
        "question_answer must directly answer the specific Question above.\n"
        "final_answer must be a concise direct answer to the user's question, not a generic yes/no answer unless the question only asks for detection.\n"
        "abnormality_score and answer_confidence must be numbers between 0 and 1."
    )


def normalize_json_output(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    normalized = dict(parsed)
    fact_check = normalized.get("fact_check")
    if not isinstance(fact_check, dict):
        fact_check = {}
    for key, value in list(normalized.items()):
        if not isinstance(key, str) or not key.startswith("fact_check."):
            continue
        fact_check[key.split(".", 1)[1]] = value
        normalized.pop(key, None)
    if fact_check:
        normalized["fact_check"] = fact_check
    return normalized


def parse_json_output(text: str) -> Dict[str, Any]:
    candidates: List[Dict[str, Any]] = []
    stack: List[int] = []
    for idx, char in enumerate(text):
        if char == "{":
            stack.append(idx)
        elif char == "}" and stack:
            start = stack.pop()
            snippet = text[start : idx + 1]
            try:
                parsed = json.loads(snippet)
            except Exception:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)
    for parsed in candidates:
        parsed = normalize_json_output(parsed)
        if isinstance(parsed.get("fact_check"), dict):
            return parsed
    raise ValueError("No JSON object found in model output")
