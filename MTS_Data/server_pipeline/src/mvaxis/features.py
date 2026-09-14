from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .data_schema import ModelInput, convert_to_model_input


ROLES = ["cause", "response", "state", "request", "control_input", "sensor", "unknown"]
TYPES = ["continuous", "discrete_state", "binary_flag", "categorical", "unknown"]


def channel_window_features(model_input: ModelInput) -> np.ndarray:
    values = model_input.values
    start, end = model_input.interval
    window = values[start:end]
    outside = np.concatenate([values[:start], values[end:]], axis=0)
    if outside.size == 0:
        outside = values
    mean = values.mean(axis=0)
    std = values.std(axis=0) + 1e-8
    zw = (window - mean) / std
    out = (outside - mean) / std
    t = np.linspace(-1.0, 1.0, len(window))
    feats = []
    for c in range(values.shape[1]):
        x = zw[:, c]
        xo = out[:, c]
        diff = np.diff(x) if len(x) > 1 else np.zeros(1)
        slope = float(np.dot(t, x - x.mean()) / (np.dot(t, t) + 1e-8)) if len(x) > 1 else 0.0
        sign_changes = float(np.sum(np.sign(diff[1:]) * np.sign(diff[:-1]) < 0)) if len(diff) > 1 else 0.0
        feats.append([
            float(np.mean(x)),
            float(np.std(x)),
            float(np.max(x)),
            float(np.min(x)),
            float(np.max(np.abs(x))),
            float(np.ptp(x)),
            slope,
            float(np.std(diff)),
            float(np.mean(np.abs(x)) - np.mean(np.abs(xo))),
            sign_changes / max(1.0, len(x) / 10.0),
        ])
    return np.asarray(feats, dtype=float)


def metadata_vectors(channels: List[Dict[str, Any]]) -> np.ndarray:
    rows = []
    for ch in channels:
        role = ch.get("role", "unknown")
        ctype = ch.get("type", "unknown")
        role_vec = [1.0 if role == r else 0.0 for r in ROLES]
        type_vec = [1.0 if ctype == t else 0.0 for t in TYPES]
        scale = ch.get("scale") or {}
        span = float((scale.get("max") or 0.0) - (scale.get("min") or 0.0))
        std = float(scale.get("std") or 0.0)
        rows.append(role_vec[:4] + type_vec[:3] + [np.tanh(span), np.tanh(std), 0.0])
    return np.asarray(rows, dtype=float)


def evidence_vector(evidence_card: Dict[str, Any]) -> np.ndarray:
    flags = evidence_card.get("summary_flags", {})
    gap_map = {"none": 0.0, "slight": 0.33, "moderate": 0.66, "severe": 1.0}
    return np.asarray([
        float(flags.get("has_high_freq_oscillation", False)),
        float(flags.get("has_over_fluctuation", False)),
        0.0,
        gap_map.get(flags.get("max_gap_level", "none"), 0.0),
        min(1.0, len(evidence_card.get("peaks_and_troughs", [])) / 3.0),
        0.0,
    ], dtype=float)


def channel_targets(model_input: ModelInput) -> np.ndarray:
    start, end = model_input.interval
    return (model_input.labels[start:end].max(axis=0) > 0).astype(float)


def root_target_index(model_input: ModelInput) -> int:
    ids = [ch["channel_id"] for ch in model_input.channels]
    if model_input.root_cause_channel in ids:
        return ids.index(model_input.root_cause_channel)
    return len(ids)


def samples_to_inputs(samples: List[Dict[str, Any]]) -> List[ModelInput]:
    return [convert_to_model_input(s) for s in samples]
