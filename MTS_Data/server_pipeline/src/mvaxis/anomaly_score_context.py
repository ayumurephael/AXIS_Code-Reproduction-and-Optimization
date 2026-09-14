from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def aggregate_anomaly_scores(
    channel_probs: np.ndarray,
    channels: Optional[List[Dict[str, Any]]] = None,
    *,
    aggregation: str = "max",
) -> np.ndarray:
    """Aggregate per-channel anomaly probabilities into one score per time step."""

    probs = np.asarray(channel_probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("channel_probs must have shape [T, C]")
    active = []
    if channels:
        active = [idx for idx, ch in enumerate(channels) if ch.get("active", True)]
    if not active:
        active = list(range(probs.shape[1]))
    active_probs = probs[:, active]
    if aggregation == "mean":
        return active_probs.mean(axis=1)
    if aggregation == "max":
        return active_probs.max(axis=1)
    raise ValueError(f"Unsupported anomaly score aggregation: {aggregation}")


def score_stats(scores: Iterable[float], start: int, end: int) -> Dict[str, Any]:
    arr = np.asarray(list(scores), dtype=float)
    if arr.size == 0:
        return {"mean": None, "max": None, "interval_mean": None, "interval_max": None}
    start = max(0, min(int(start), arr.size))
    end = max(start, min(int(end), arr.size))
    window = arr[start:end]
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "interval_mean": float(window.mean()) if window.size else None,
        "interval_max": float(window.max()) if window.size else None,
    }


def attach_anomaly_score_context(
    sample: Dict[str, Any],
    scores: Iterable[float],
    *,
    aggregation: str,
    image_path: Optional[str] = None,
    copy_sample: bool = True,
) -> Dict[str, Any]:
    """Attach model-produced anomaly scores to a sample without changing labels."""

    row = copy.deepcopy(sample) if copy_sample else sample
    interval = row.get("target_interval") or {}
    start = int(interval.get("start", 0))
    end = int(interval.get("end", 0))
    score_list = [float(x) for x in scores]
    row["anomaly_score_context"] = {
        "scores": score_list,
        "aggregation": aggregation,
        "source": "AXIS/TimeRCD anomaly_head",
        "target_interval": {"start": start, "end": end},
        "stats": score_stats(score_list, start, end),
    }
    if image_path:
        row["anomaly_score_context"]["image_path"] = str(image_path)
    return row

