from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .data_schema import SCHEMA_VERSION, attach_channel_scales, make_channel, validate_sample
from .evidence import recognize_evidence


def _interval_from_labels(labels: np.ndarray) -> Dict[str, int]:
    if labels.ndim == 2:
        point_labels = labels.max(axis=1)
    else:
        point_labels = labels
    idx = np.where(point_labels > 0)[0]
    if len(idx) == 0:
        n = len(point_labels)
        return {"start": max(0, n // 2 - n // 10), "end": min(n, n // 2 + n // 10)}
    return {"start": int(max(0, idx[0] - 2)), "end": int(min(len(point_labels), idx[-1] + 3))}


def _label_matrix_from_legacy(sample: Dict[str, Any], values: np.ndarray) -> np.ndarray:
    raw_labels = np.asarray(sample.get("labels", np.zeros(values.shape[0])), dtype=int)
    if raw_labels.ndim == 2 and raw_labels.shape == values.shape:
        return raw_labels.astype(int)
    normal = sample.get("normal_time_series")
    if normal is not None:
        normal_values = np.asarray(normal, dtype=float)
        if normal_values.ndim == 1:
            normal_values = normal_values[:, None]
        if normal_values.shape == values.shape:
            diff = np.abs(values - normal_values)
            scale = np.std(normal_values, axis=0, keepdims=True) + 1e-6
            threshold = np.maximum(1e-4, 0.05 * scale)
            diff_labels = diff > threshold
            if raw_labels.ndim == 1 and raw_labels.shape[0] == values.shape[0] and raw_labels.max() > 0:
                diff_labels &= raw_labels[:, None] > 0
            if diff_labels.any():
                return diff_labels.astype(int)
    if raw_labels.ndim == 1:
        return np.repeat(raw_labels[:, None], values.shape[1], axis=1).astype(int)
    return np.zeros_like(values, dtype=int)


def _legacy_anomaly_channels(attribute: Dict[str, Any], num_channels: int) -> List[int]:
    attr_list = attribute.get("attribute_list") or []
    channels = []
    for idx, attr in enumerate(attr_list[:num_channels]):
        if isinstance(attr, dict) and attr.get("anomalies"):
            channels.append(idx)
    return channels


_LEGACY_LOCAL_ANOMALY_TYPES = {
    "shake",
    "upward spike",
    "downward spike",
    "continuous upward spike",
    "continuous downward spike",
    "upward convex",
    "downward convex",
    "sudden increase",
    "sudden decrease",
    "rapid rise followed by slow decline",
    "slow rise followed by rapid decline",
    "rapid decline followed by slow rise",
    "slow decline followed by rapid rise",
    "decrease after upward spike",
    "increase after downward spike",
    "increase after upward spike",
    "decrease after downward spike",
    "wide upward spike",
    "wide downward spike",
    "outlier",
}


def _normalize_anomaly_name(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"^\d+[_:\-\s]+", "", name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _slug_anomaly_name(name: str) -> Optional[str]:
    clean = _normalize_anomaly_name(name)
    slug = re.sub(r"[^a-z0-9]+", "_", clean).strip("_")
    return slug or None


def _legacy_anomaly_names(raw: Any) -> List[str]:
    names: List[str] = []
    if raw is None:
        return names
    if isinstance(raw, dict):
        for key in ("type", "anomaly_type", "name"):
            value = raw.get(key)
            if value:
                names.append(str(value))
        for key, value in raw.items():
            if key in {"type", "anomaly_type", "name", "start", "end", "position_start", "position_end"}:
                continue
            names.append(str(key))
            if isinstance(value, dict):
                names.extend(_legacy_anomaly_names(value))
        return names
    if isinstance(raw, (list, tuple)):
        for item in raw:
            names.extend(_legacy_anomaly_names(item))
        return names
    names.append(str(raw))
    return names


def _canonical_legacy_anomaly_type(raw_name: str) -> Optional[str]:
    clean = _normalize_anomaly_name(raw_name)
    if not clean:
        return None
    for anomaly_name in sorted(_LEGACY_LOCAL_ANOMALY_TYPES, key=len, reverse=True):
        if clean == anomaly_name or anomaly_name in clean:
            return _slug_anomaly_name(anomaly_name)
    mapping = [
        ("spike", "spike"),
        ("level", "level_shift"),
        ("shift", "level_shift"),
        ("variance", "variance_burst"),
        ("fluct", "variance_burst"),
        ("noise", "variance_burst"),
        ("oscillat", "oscillation"),
        ("period", "oscillation"),
        ("frequency", "oscillation"),
        ("trend", "trend_reversal"),
        ("saturat", "saturation"),
    ]
    for key, name in mapping:
        if key in clean:
            return name
    return _slug_anomaly_name(clean)


def _legacy_anomaly_type(attribute: Dict[str, Any], root_idx: Optional[int]) -> Optional[str]:
    attr_list = attribute.get("attribute_list") or []
    candidates: List[str] = []
    if root_idx is not None and root_idx < len(attr_list) and isinstance(attr_list[root_idx], dict):
        candidates.extend(_legacy_anomaly_names(attr_list[root_idx].get("anomalies")))
    if not candidates:
        for attr in attr_list:
            if isinstance(attr, dict):
                candidates.extend(_legacy_anomaly_names(attr.get("anomalies")))
    for candidate in candidates:
        canonical = _canonical_legacy_anomaly_type(candidate)
        if canonical:
            return canonical
    return None


def _legacy_root_anomaly_name(attribute: Dict[str, Any], root_idx: Optional[int]) -> Optional[str]:
    attr_list = attribute.get("attribute_list") or []
    if root_idx is None or root_idx >= len(attr_list) or not isinstance(attr_list[root_idx], dict):
        return None
    for candidate in _legacy_anomaly_names(attr_list[root_idx].get("anomalies")):
        clean = _normalize_anomaly_name(candidate)
        if clean:
            return clean
    return None


def _parse_legacy_edges(attribute: Dict[str, Any]) -> List[List[str]]:
    raw = attribute.get("dag")
    if raw is None:
        return []
    try:
        edges = ast.literal_eval(str(raw))
    except Exception:
        return []
    parsed = []
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            parsed.append([f"ch_{int(edge[0])}", f"ch_{int(edge[1])}"])
    return parsed


def _legacy_scope_label(scope: Optional[str]) -> str:
    return {"node": "one-channel", "edge": "small multi-channel", "subgraph": "broad multi-channel"}.get(
        str(scope), str(scope)
    )


def _legacy_interval_answer(
    *,
    affected: List[str],
    root_channel: Optional[str],
    anomaly_type: Optional[str],
    root_anomaly_name: Optional[str],
    anomaly_scope: Optional[str],
) -> str:
    if not affected:
        return (
            "The interval is not anomalous. No root-cause channel or affected channel should be assigned; "
            "observed channel movements should be treated as normal/background behavior."
        )
    root_text = root_channel or "unknown"
    affected_text = ", ".join(affected) if affected else "none"
    scope_text = _legacy_scope_label(anomaly_scope)
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


def _channels_from_legacy(attribute: Dict[str, Any], num_channels: int) -> List[Dict[str, Any]]:
    attr_list = attribute.get("attribute_list") or []
    channels: List[Dict[str, Any]] = []
    for c in range(num_channels):
        legacy_attr = attr_list[c] if c < len(attr_list) and isinstance(attr_list[c], dict) else {}
        name = legacy_attr.get("metric") or legacy_attr.get("name") or f"legacy_channel_{c}"
        channels.append(
            make_channel(
                f"ch_{c}",
                str(name),
                "sensor",
                "continuous",
                "a.u.",
                "Converted from legacy TSAD generator.",
                [],
            )
        )
    return channels


def convert_legacy_sample(sample: Dict[str, Any], sample_id: str, base_sample_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert TSAD_dataset_gen sample dict into MultivariateAnomalyQASample JSON.

    Expected legacy keys are normal_time_series, time_series, labels, and attribute.
    This adapter is intentionally conservative: unknown root cause fields are left
    null unless the old sample has an endogenous channel hint.
    """
    values = np.asarray(sample["time_series"], dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    attribute = sample.get("attribute") or {}
    labels = _label_matrix_from_legacy(sample, values)
    channels = _channels_from_legacy(attribute, values.shape[1])
    interval = _interval_from_labels(labels)
    affected = [f"ch_{i}" for i in np.where(labels[interval["start"]:interval["end"]].max(axis=0) > 0)[0]]
    legacy_roots = _legacy_anomaly_channels(attribute, values.shape[1])
    root_idx = legacy_roots[0] if legacy_roots else (int(affected[0].split("_", 1)[1]) if affected else None)
    root_channel = f"ch_{root_idx}" if root_idx is not None else None
    anomaly_type = _legacy_anomaly_type(attribute, root_idx)
    root_anomaly_name = _legacy_root_anomaly_name(attribute, root_idx) if affected else None
    normal_values = sample.get("normal_time_series")
    if normal_values is not None:
        normal_values = np.asarray(normal_values, dtype=float)
        if normal_values.ndim == 1:
            normal_values = normal_values[:, None]
    causal_edges = _parse_legacy_edges(attribute)
    abnormal_edges = [edge for edge in causal_edges if root_channel in edge] if root_channel else []
    anomaly_scope = None
    if affected:
        anomaly_scope = "node" if len(affected) == 1 else "subgraph"
    natural_answer = _legacy_interval_answer(
        affected=affected,
        root_channel=root_channel,
        anomaly_type=anomaly_type if affected else None,
        root_anomaly_name=root_anomaly_name,
        anomaly_scope=anomaly_scope,
    )
    converted = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "base_sample_id": base_sample_id or sample_id,
        "source": "legacy_generator",
        "original_data": {
            "time_series": values.round(6).tolist(),
            "normal_series": normal_values.round(6).tolist() if normal_values is not None else None,
            "global_descriptor": "Converted from the original AXIS TSAD_dataset_gen generator.",
            "legacy_root_anomaly_name": root_anomaly_name,
            "legacy_root_anomaly_type": anomaly_type if affected else None,
        },
        "normal_series": normal_values.round(6).tolist() if normal_values is not None else None,
        "series": {
            "shape": [int(values.shape[0]), int(values.shape[1])],
            "values": values.round(6).tolist(),
            "labels": labels.astype(int).tolist(),
            "timestamps": None,
        },
        "channels": channels,
        "causal_graph": {
            "graph_type": "legacy_random_dag",
            "edges": causal_edges,
            "is_endogenous": attribute.get("is_endogenous"),
        },
        "target_interval": interval,
        "question_type": "legacy_axis_explanation",
        "question": (
            "Analyze whether the converted legacy AXIS interval is anomalous. "
            "If anomalous, identify the directly injected/root channel, the affected channels, "
            "and the most likely local morphology."
        ),
        "root_cause": {
            "channel_id": root_channel,
            "time_index": interval["start"] if root_channel else None,
            "interval": [interval["start"], interval["end"]] if root_channel else None,
            "provided_by": "legacy_label" if root_channel else "unknown",
        },
        "synthetic_label": {
            "is_synthetic": True,
            "base_sample_id": base_sample_id or sample_id,
            "abnormality_strength_alpha": None,
            "anomaly_type": anomaly_type if affected else None,
            "root_anomaly_name": root_anomaly_name,
            "anomaly_scope": anomaly_scope,
            "affected_channels": affected,
            "abnormal_edges": abnormal_edges,
            "root_cause_channels": [root_channel] if root_channel else [],
            "causal_path": affected,
            "counterfactual_normal_sample_id": None,
        },
        "evidence_card": {},
        "target_output": {},
    }
    attach_channel_scales(converted)
    converted["evidence_card"] = recognize_evidence(converted)
    converted["target_output"] = {
        "fact_check": {
            "is_anomalous": bool(affected),
            "root_cause_channel": root_channel,
            "root_cause_channels": [root_channel] if root_channel else [],
            "affected_channels": affected,
            "anomaly_type": anomaly_type if affected else None,
            "root_anomaly_name": root_anomaly_name,
            "anomaly_scope": anomaly_scope,
            "abnormal_edges": abnormal_edges,
            "causal_path": affected,
            "graph_type": "legacy_random_dag",
            "has_high_freq_oscillation": converted["evidence_card"]["summary_flags"]["has_high_freq_oscillation"],
            "has_over_fluctuation": converted["evidence_card"]["summary_flags"]["has_over_fluctuation"],
            "has_relation_break": converted["evidence_card"]["summary_flags"]["has_relation_break"],
            "left_right_gap_level": converted["evidence_card"]["summary_flags"]["max_gap_level"],
            "state_response_consistency": "uncertain",
            "evidence_used": ["legacy_labels", "normal_counterpart_diff", "legacy_attribute", "evidence_card"],
        },
        "reasoning_summary": (
            f"Converted from the original AXIS generator. The target interval is "
            f"{'anomalous' if affected else 'not anomalous'} by the legacy labels and target-normal difference. "
            f"The direct injected/root channel is {root_channel}; affected channels are "
            f"{', '.join(affected) if affected else 'none'}. The root local anomaly type is {anomaly_type}."
        ),
        "final_answer": natural_answer,
        "abnormality_score": 1.0 if affected else 0.0,
        "answer_confidence": 0.70 if affected else 0.75,
    }
    converted["windows"] = [
        {
            "window_range": [interval["start"], interval["end"]],
            "question": converted["question"],
            "answer": converted["target_output"]["final_answer"],
            "question_type": converted["question_type"],
            "has_anomaly": bool(affected),
            "anomaly_descriptions": [converted["target_output"]["reasoning_summary"]],
            "mv_label": {
                "root_cause_channel": root_channel,
                "affected_channels": affected,
                "anomaly_type": anomaly_type,
                "root_anomaly_name": root_anomaly_name,
                "anomaly_scope": anomaly_scope,
                "abnormal_edges": abnormal_edges,
                "causal_path": affected,
                "graph_type": "legacy_random_dag",
            },
        }
    ]
    validate_sample(converted)
    return converted
