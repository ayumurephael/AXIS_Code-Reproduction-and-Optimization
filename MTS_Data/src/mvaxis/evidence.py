from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-8)


def _gap_level(score: float) -> str:
    if score < 0.75:
        return "none"
    if score < 1.25:
        return "slight"
    if score < 2.0:
        return "moderate"
    return "severe"


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 6) -> Tuple[int, float, float]:
    zero_corr = _safe_corr(a, b)
    best_lag = 0
    best_corr = zero_corr
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = _safe_corr(a[-lag:], b[:lag])
        elif lag > 0:
            corr = _safe_corr(a[:-lag], b[lag:])
        else:
            corr = zero_corr
        if abs(corr) > abs(best_corr):
            best_lag = lag
            best_corr = corr
    return best_lag, best_corr, zero_corr


def _trend_label(x: np.ndarray) -> str:
    if len(x) < 2:
        return "flat"
    t = np.linspace(-1.0, 1.0, len(x))
    slope = float(np.dot(t, x - x.mean()) / (np.dot(t, t) + 1e-8))
    if slope > 0.08:
        return "increasing"
    if slope < -0.08:
        return "decreasing"
    return "flat"


def _local_extrema_count(x: np.ndarray) -> int:
    if len(x) < 3:
        return 0
    d = np.diff(x)
    signs = np.sign(d)
    return int(np.sum((signs[1:] * signs[:-1]) < 0))


def recognize_evidence(sample: Dict[str, Any]) -> Dict[str, Any]:
    values = np.asarray(sample["series"]["values"], dtype=float)
    channels = sample["channels"]
    active_indices = [idx for idx, ch in enumerate(channels) if ch.get("active", True)]
    active_count = max(1, len(active_indices))
    start = int(sample["target_interval"]["start"])
    end = int(sample["target_interval"]["end"])
    z = _zscore(values)
    zw = z[start:end]
    window = values[start:end]

    local_extrema: List[Dict[str, Any]] = []
    peaks_and_troughs: List[Dict[str, Any]] = []
    over_fluctuation: List[Dict[str, Any]] = []
    level_shifts: List[Dict[str, Any]] = []
    trend_groups: Dict[str, List[str]] = {"increasing": [], "decreasing": [], "flat": []}
    state_changes: List[Dict[str, Any]] = []

    for c in active_indices:
        ch = channels[c]
        cid = ch["channel_id"]
        xw = window[:, c]
        zwc = zw[:, c]
        extrema_count = _local_extrema_count(zwc)
        if extrema_count:
            local_extrema.append({"channel_id": cid, "count": extrema_count})
        max_abs = float(np.max(np.abs(zwc))) if len(zwc) else 0.0
        if max_abs > 2.0:
            local_idx = int(np.argmax(np.abs(zwc)))
            peaks_and_troughs.append({
                "channel_id": cid,
                "time_index": start + local_idx,
                "z_abs": max_abs,
                "kind": "peak" if zwc[local_idx] > 0 else "trough",
            })
        outside = np.concatenate([values[:start, c], values[end:, c]])
        if outside.size == 0:
            outside = values[:, c]
        outside_std = float(np.std(outside) + 1e-8)
        mean_shift_z = float(abs(np.mean(xw) - np.mean(outside)) / outside_std)
        if ch.get("type") not in {"discrete_state", "binary_flag", "categorical"} and mean_shift_z > 1.3:
            level_shifts.append({
                "channel_id": cid,
                "mean_shift_z": mean_shift_z,
                "direction": "up" if float(np.mean(xw) - np.mean(outside)) > 0 else "down",
                "strength": "strong" if mean_shift_z > 1.8 else "moderate",
            })
        ratio = float(np.std(xw) / outside_std)
        if ratio > 1.6:
            over_fluctuation.append({"channel_id": cid, "std_ratio": ratio})
        trend_groups[_trend_label(zwc)].append(cid)
        if ch.get("type") in {"discrete_state", "binary_flag", "categorical"}:
            changes = int(np.sum(np.diff(xw) != 0))
            if changes:
                state_changes.append({"channel_id": cid, "change_count": changes})

    has_high_freq = any(x["count"] >= max(4, (end - start) // 8) for x in local_extrema)
    evidence_confidence = {
        "peak": min(1.0, len(peaks_and_troughs) / active_count),
        "fluctuation": min(1.0, len(over_fluctuation) / active_count),
    }
    return {
        "phase_segments": [],
        "state_changes": state_changes,
        "local_extrema": local_extrema,
        "peaks_and_troughs": peaks_and_troughs,
        "level_shifts": level_shifts,
        "phase_lags": [],
        "over_fluctuation": over_fluctuation,
        "trend_groups": [
            {"trend": k, "channels": v} for k, v in trend_groups.items() if v
        ],
        "left_right_gap": [],
        "front_rear_gap": [],
        "channel_relation_breaks": [],
        "root_response_order": [],
        "evidence_confidence": evidence_confidence,
        "summary_flags": {
            "has_high_freq_oscillation": bool(has_high_freq),
            "has_level_shift": bool(level_shifts),
            "has_phase_lag": False,
            "has_over_fluctuation": bool(over_fluctuation),
            "has_relation_break": False,
            "max_gap_level": "none",
        },
    }
