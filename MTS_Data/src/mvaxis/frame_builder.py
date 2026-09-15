from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .data_schema import attach_channel_scales
from .question_provider import duplicate_samples_with_question_frames


QUESTION_FRAMES = ("anomaly_frame", "normal_frame")


def normalize_source_range(total: int, start: int, end: int | None) -> Tuple[int, int]:
    """Validate and normalize the half-open raw-series index range [start, end)."""

    range_start = max(0, int(start))
    range_end = total if end is None else min(total, max(range_start, int(end)))
    if total <= 0:
        raise ValueError("The window metadata file is empty.")
    if range_start >= total:
        raise ValueError(
            f"source-start-index {range_start} is outside the available range "
            f"0..{max(total - 1, 0)}"
        )
    return range_start, range_end


def filter_windows_by_source_range(
    rows: Sequence[Dict[str, Any]],
    *,
    source_start_index: int = 0,
    source_end_index: int | None = None,
) -> List[Dict[str, Any]]:
    """Keep windows whose source_index is inside the requested raw-series range."""

    if not rows:
        raise ValueError("The window metadata file is empty.")
    max_source_index = max(int(row.get("source_index", -1)) for row in rows) + 1
    range_start, range_end = normalize_source_range(
        max_source_index,
        source_start_index,
        source_end_index,
    )
    filtered = [
        copy.deepcopy(row)
        for row in rows
        if range_start <= int(row.get("source_index", -1)) < range_end
    ]
    if not filtered:
        raise ValueError(
            f"No windows remain after filtering source_index in [{range_start}, {range_end})."
        )
    return filtered


def choose_window_rows(rows: Sequence[Dict[str, Any]], n: int | None, seed: int) -> List[Dict[str, Any]]:
    """Shuffle deterministically and choose at most n physical windows."""

    shuffled = [copy.deepcopy(row) for row in rows]
    random.Random(int(seed)).shuffle(shuffled)
    if n is None:
        return shuffled
    if int(n) <= 0:
        raise ValueError("num-windows must be greater than zero.")
    if len(shuffled) < int(n):
        raise ValueError(f"Requested {int(n)} windows, but only {len(shuffled)} are available.")
    return shuffled[: int(n)]


def load_raw_rows_by_index(path: Path, wanted_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Read only the raw JSONL rows referenced by selected window metadata."""

    wanted = {int(idx) for idx in wanted_indices}
    found: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx not in wanted:
                continue
            found[idx] = json.loads(line)
            if len(found) == len(wanted):
                break
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f"Missing raw series rows for source indices: {missing[:10]}")
    return found


def rebuild_window_sample(raw_row: Dict[str, Any], window_row: Dict[str, Any]) -> Dict[str, Any]:
    """Combine a long-series row and one saved window into the QA sample schema."""

    out = copy.deepcopy(raw_row)
    interval = window_row.get("target_interval") or {}
    start = int(interval.get("start", 0))
    end = int(interval.get("end", 0))
    source_window_id = str(
        window_row.get("sample_id") or f"window_{int(window_row.get('source_index', 0)):05d}"
    )
    root_channel = window_row.get("root_cause_channel")
    affected = list(window_row.get("affected_channels") or [])
    is_anomalous = bool(window_row.get("has_anomaly"))
    synthetic_label = out.get("synthetic_label") or {}
    fact = {
        "is_anomalous": is_anomalous,
        "root_cause_channel": root_channel,
        "root_cause_channels": [root_channel] if root_channel else [],
        "affected_channels": affected,
        "anomaly_type": window_row.get("anomaly_type") if is_anomalous else None,
        "root_anomaly_name": synthetic_label.get("root_anomaly_name") if is_anomalous else None,
        "anomaly_scope": window_row.get("anomaly_scope") if is_anomalous else None,
        "abnormal_edges": list(synthetic_label.get("abnormal_edges") or []) if is_anomalous else [],
        "causal_path": list(synthetic_label.get("causal_path") or []) if is_anomalous else [],
    }

    out["base_sample_id"] = str(out.get("base_sample_id") or out.get("sample_id") or source_window_id)
    out["source_sample_id"] = out.get("sample_id")
    out["source_window_sample_id"] = source_window_id
    out["sample_id"] = source_window_id
    out["target_interval"] = {"start": start, "end": end}
    out["root_cause"] = {
        "channel_id": root_channel,
        "time_index": start if root_channel else None,
        "interval": [start, end] if root_channel else None,
        "provided_by": "saved_window_metadata",
    }
    target_output = dict(out.get("target_output") or {})
    target_output["fact_check"] = fact
    target_output["abnormality_score"] = 1.0 if is_anomalous else 0.0
    target_output["answer_confidence"] = target_output.get(
        "answer_confidence",
        0.72 if is_anomalous else 0.76,
    )
    out["target_output"] = target_output
    out["windows"] = [
        {
            "window_range": [start, end],
            "question": out.get("question"),
            "answer": target_output.get("final_answer", ""),
            "question_type": out.get("question_type"),
            "has_anomaly": is_anomalous,
            "mv_label": {
                "root_cause_channel": root_channel,
                "affected_channels": affected,
                "anomaly_type": fact.get("anomaly_type"),
                "root_anomaly_name": fact.get("root_anomaly_name"),
                "anomaly_scope": fact.get("anomaly_scope"),
            },
        }
    ]
    out["question_generation_artifacts"] = {
        "image_path": str(window_row.get("image_path") or ""),
    }
    attach_channel_scales(out)
    return out


def build_question_frames(
    window_rows: Sequence[Dict[str, Any]],
    raw_series_path: Path,
    *,
    num_windows: int | None,
    seed: int,
    source_start_index: int = 0,
    source_end_index: int | None = None,
    frames: Sequence[str] = QUESTION_FRAMES,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Select physical windows, rebuild them, and expand each into question frames."""

    filtered = filter_windows_by_source_range(
        window_rows,
        source_start_index=source_start_index,
        source_end_index=source_end_index,
    )
    selected = choose_window_rows(filtered, num_windows, seed)
    source_indices = [int(row["source_index"]) for row in selected]
    raw_by_index = load_raw_rows_by_index(raw_series_path, source_indices)
    rebuilt = [rebuild_window_sample(raw_by_index[int(row["source_index"])], row) for row in selected]
    return duplicate_samples_with_question_frames(rebuilt, frames=frames), selected
