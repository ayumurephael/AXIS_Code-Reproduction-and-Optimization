from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.utils import save_json, write_jsonl


REPO_ROOT = ROOT.parent
RAW_NORMAL = REPO_ROOT / "data" / "raw" / "swat" / "SWaT_Normal.csv"
RAW_ABNORMAL = REPO_ROOT / "data" / "raw" / "swat" / "SWaT_Abnormal.csv"
RAW_LABEL = REPO_ROOT / "data" / "raw" / "swat" / "SWaT_label.csv"


def _canonicalize_signal_name(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _repair_label_start_times(start_times: pd.Series) -> pd.Series:
    """Repair the Jan-02 rollover typo and keep event times monotonic."""

    repaired = []
    year_offset = 0
    previous: pd.Timestamp | None = None
    for value in pd.to_datetime(start_times, errors="coerce"):
        ts = value + pd.DateOffset(years=year_offset)
        if previous is not None and ts < previous - pd.Timedelta(days=30):
            year_offset += 1
            ts = value + pd.DateOffset(years=year_offset)
        repaired.append(ts)
        previous = ts
    return pd.Series(repaired, index=start_times.index)


def _load_label_rows(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["Start Time", "Adjusted End Time", "Attack Point"]).copy()
    df["Start Time"] = _repair_label_start_times(df["Start Time"])
    end_times = pd.to_datetime(df["Adjusted End Time"], errors="coerce")
    df["Adjusted End Time"] = [
        pd.to_datetime(start.strftime("%Y-%m-%d") + " " + end.strftime("%H:%M:%S"))
        if pd.notna(start) and pd.notna(end)
        else pd.NaT
        for start, end in zip(df["Start Time"], end_times)
    ]
    df = df.dropna(subset=["Start Time", "Adjusted End Time"]).copy()
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "attack_id": int(row["Attack #"]),
                "start_time": pd.Timestamp(row["Start Time"]),
                "end_time": pd.Timestamp(row["Adjusted End Time"]),
                "attack_points": [part.strip() for part in str(row["Attack Point"]).split(",") if part.strip()],
            }
        )
    return rows


def _read_swat_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return _clean_columns(df)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    drop = {"", "Timestamp", "Normal/Attack"}
    return [col for col in df.columns if col not in drop and not str(col).startswith("Unnamed:")]


def _coerce_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        df[col] = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)


def _downsample_with_labels(
    values: np.ndarray,
    point_labels: np.ndarray,
    node_labels: np.ndarray,
    timestamps: Sequence[pd.Timestamp],
    factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    usable = (len(values) // factor) * factor
    values = values[:usable]
    point_labels = point_labels[:usable]
    node_labels = node_labels[:usable]
    ts = list(timestamps[:usable])
    values = values.reshape(-1, factor, values.shape[1])
    point_labels = point_labels.reshape(-1, factor)
    node_labels = node_labels.reshape(-1, factor, node_labels.shape[1])
    ts_ds = [ts[idx] for idx in range(0, usable, factor)]
    return (
        np.median(values, axis=1).astype(np.float32),
        (point_labels.max(axis=1) > 0).astype(np.uint8),
        (node_labels.max(axis=1) > 0).astype(np.uint8),
        ts_ds,
    )


def _fit_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_scaler(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


def _channel_specs(
    feature_names: Sequence[str],
    normalized_normal_values: np.ndarray,
) -> List[Dict[str, Any]]:
    channels: List[Dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        col = normalized_normal_values[:, idx]
        channels.append(
            {
                "channel_id": f"ch_{idx}",
                "name": str(name),
                "unit": "a.u.",
                "scale": {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col)),
                    "min": float(np.min(col)),
                    "max": float(np.max(col)),
                },
                "role": "sensor",
                "type": "continuous",
                "description": f"SWaT signal {name}.",
                "prior_relations": [],
                "active": True,
            }
        )
    return channels


def _scope_from_channels(channels: Sequence[str]) -> str | None:
    if not channels:
        return None
    if len(channels) == 1:
        return "node"
    if len(channels) <= 3:
        return "edge"
    return "subgraph"


def _fit_window_bounds(total_len: int, desired_start: int, seq_len: int) -> tuple[int, int]:
    if total_len <= seq_len:
        return 0, total_len
    start = max(0, min(int(desired_start), total_len - seq_len))
    return start, start + seq_len


def _attack_variant_starts(event_start: int, event_end: int, total_len: int, seq_len: int) -> List[tuple[str, int]]:
    event_len = max(1, event_end - event_start)
    long_event = event_len > 420
    variants: List[tuple[str, int]] = [
        ("onset", event_start - 160),
        ("center", ((event_start + event_end) // 2) - (seq_len // 2)),
        ("offset", event_end - 340),
    ]
    if long_event:
        variants.insert(1, ("early_middle", event_start + int(0.30 * event_len) - (seq_len // 2)))
    out: List[tuple[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for role, desired_start in variants:
        bounds = _fit_window_bounds(total_len, desired_start, seq_len)
        if bounds in seen:
            continue
        seen.add(bounds)
        out.append((role, bounds[0]))
    return out


def _window_timestamps(timestamps: Sequence[pd.Timestamp], start: int, end: int) -> List[str]:
    return [pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps[start:end]]


def _series_block(
    values: np.ndarray,
    node_labels: np.ndarray,
    timestamps: Sequence[pd.Timestamp],
    start: int,
    end: int,
) -> Dict[str, Any]:
    block_values = values[start:end].astype(np.float32)
    block_labels = node_labels[start:end].astype(np.uint8)
    return {
        "shape": [int(block_values.shape[0]), int(block_values.shape[1])],
        "values": block_values.tolist(),
        "labels": block_labels.tolist(),
        "timestamps": _window_timestamps(timestamps, start, end),
    }


def _empty_label_block(length: int, num_channels: int) -> List[List[int]]:
    return np.zeros((length, num_channels), dtype=np.uint8).tolist()


@dataclass
class AttackEvent:
    attack_id: int
    attack_points: List[str]
    raw_indices: np.ndarray
    ds_start: int
    ds_end: int
    attacked_channel_ids: List[str]
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def _extract_attack_events(
    label_rows: Sequence[Dict[str, Any]],
    attack_timestamps: Sequence[pd.Timestamp],
    attack_point_labels: np.ndarray,
    feature_columns: Sequence[str],
    downsample_factor: int,
) -> List[AttackEvent]:
    attack_col_map = {_canonicalize_signal_name(col): f"ch_{idx}" for idx, col in enumerate(feature_columns)}
    events: List[AttackEvent] = []
    for row in label_rows:
        st = row["start_time"]
        et = row["end_time"]
        indices = np.where(
            (attack_timestamps >= st)
            & (attack_timestamps <= et)
            & (attack_point_labels == 1)
        )[0]
        if len(indices) == 0:
            indices = np.where((attack_timestamps >= st) & (attack_timestamps <= et))[0]
        if len(indices) == 0:
            continue
        attacked_channels = []
        for point in row["attack_points"]:
            channel_id = attack_col_map.get(_canonicalize_signal_name(point))
            if channel_id and channel_id not in attacked_channels:
                attacked_channels.append(channel_id)
        if not attacked_channels:
            continue
        ds_start = int(indices[0] // downsample_factor)
        ds_end = int(indices[-1] // downsample_factor) + 1
        events.append(
            AttackEvent(
                attack_id=int(row["attack_id"]),
                attack_points=list(row["attack_points"]),
                raw_indices=np.asarray(indices, dtype=np.int64),
                ds_start=ds_start,
                ds_end=ds_end,
                attacked_channel_ids=attacked_channels,
                start_time=st,
                end_time=et,
            )
        )
    return events


def _make_anomaly_rows(
    *,
    values_ds: np.ndarray,
    node_labels_ds: np.ndarray,
    timestamps_ds: Sequence[pd.Timestamp],
    channels: Sequence[Dict[str, Any]],
    events: Sequence[AttackEvent],
    seq_len: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    event_index: List[Dict[str, Any]] = []
    total_len = int(values_ds.shape[0])
    global_idx = 0
    for event in events:
        variants = _attack_variant_starts(event.ds_start, event.ds_end, total_len, seq_len)
        event_record = {
            "attack_id": int(event.attack_id),
            "attack_points": list(event.attack_points),
            "start_time": str(event.start_time),
            "end_time": str(event.end_time),
            "raw_start_index": int(event.raw_indices[0]),
            "raw_end_index_inclusive": int(event.raw_indices[-1]),
            "downsampled_start_index": int(event.ds_start),
            "downsampled_end_index_exclusive": int(event.ds_end),
            "attacked_channel_ids": list(event.attacked_channel_ids),
            "variants": [],
        }
        for role, start in variants:
            left, right = _fit_window_bounds(total_len, start, seq_len)
            local_event_start = max(0, event.ds_start - left)
            local_event_end = min(right - left, event.ds_end - left)
            root_channel = event.attacked_channel_ids[0]
            sample_id = f"swat_attack_{event.attack_id:03d}_{role}"
            row = {
                "schema_version": "axis_swat_v1",
                "sample_id": sample_id,
                "base_sample_id": f"swat_attack_{event.attack_id:03d}",
                "source": {
                    "dataset": "SWaT",
                    "source_file": "SWaT_Abnormal.csv",
                    "split": "attack",
                    "event_id": int(event.attack_id),
                    "event_role": role,
                    "downsample_factor": 2,
                    "raw_start_index": int(left * 2),
                    "raw_end_index_exclusive": int(right * 2),
                    "downsampled_start_index": int(left),
                    "downsampled_end_index_exclusive": int(right),
                },
                "original_data": {
                    "timestamp_start": timestamps_ds[left].strftime("%Y-%m-%d %H:%M:%S"),
                    "timestamp_end": timestamps_ds[right - 1].strftime("%Y-%m-%d %H:%M:%S"),
                    "attack_points": list(event.attack_points),
                    "label_source": "SWaT_label.csv",
                    "attack_window_start_time": str(event.start_time),
                    "attack_window_end_time": str(event.end_time),
                },
                "normal_series": None,
                "series": _series_block(values_ds, node_labels_ds, timestamps_ds, left, right),
                "channels": list(channels),
                "causal_graph": {},
                "root_cause": {
                    "channel_id": root_channel,
                    "time_index": int(local_event_start),
                    "interval": [int(local_event_start), int(local_event_end)],
                    "provided_by": "attack_label_projection",
                },
                "synthetic_label": {
                    "is_synthetic": False,
                    "base_sample_id": f"swat_attack_{event.attack_id:03d}",
                    "anomaly_type": "attack_event",
                    "root_anomaly_name": "SWaT attack event",
                    "anomaly_scope": _scope_from_channels(event.attacked_channel_ids),
                    "affected_channels": list(event.attacked_channel_ids),
                    "root_cause_channels": list(event.attacked_channel_ids),
                    "causal_path": list(event.attacked_channel_ids),
                    "abnormal_edges": [],
                },
                "sequence_length": int(right - left),
                "global_sample_index": int(global_idx),
            }
            rows.append(row)
            event_record["variants"].append(
                {
                    "sample_id": sample_id,
                    "event_role": role,
                    "series_start_index": int(left),
                    "series_end_index_exclusive": int(right),
                    "local_attack_interval": [int(local_event_start), int(local_event_end)],
                }
            )
            global_idx += 1
        event_index.append(event_record)
    return rows, event_index


def _dilate_positive_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    if radius <= 0:
        return out
    pos = np.flatnonzero(mask > 0)
    for idx in pos:
        left = max(0, int(idx) - radius)
        right = min(len(mask), int(idx) + radius + 1)
        out[left:right] = True
    return out


def _contiguous_false_segments(mask: np.ndarray) -> List[tuple[int, int]]:
    segments: List[tuple[int, int]] = []
    start = None
    for idx, val in enumerate(mask.astype(bool)):
        if not val and start is None:
            start = idx
        if start is not None and (val or idx == len(mask) - 1):
            end = idx if val else idx + 1
            segments.append((int(start), int(end)))
            start = None
    return segments


def _evenly_spaced_starts(total_len: int, seq_len: int, count: int) -> List[int]:
    if total_len <= seq_len:
        return [0] * count
    max_start = total_len - seq_len
    if count == 1:
        return [max_start // 2]
    return [int(round(pos)) for pos in np.linspace(0, max_start, num=count)]


def _safe_abnormal_normal_starts(
    point_labels_ds: np.ndarray,
    *,
    seq_len: int,
    count: int,
    buffer_points: int,
) -> List[int]:
    forbidden = _dilate_positive_mask(point_labels_ds, buffer_points)
    segments = [(left, right) for left, right in _contiguous_false_segments(forbidden) if right - left >= seq_len]
    if not segments:
        return []
    candidates: List[int] = []
    for left, right in segments:
        seg_len = right - left
        local_count = max(1, int(round(count * (seg_len / sum(r - l for l, r in segments)))))
        starts = _evenly_spaced_starts(seg_len, seq_len, local_count)
        for start in starts:
            candidates.append(left + start)
    candidates = sorted(dict.fromkeys(candidates))
    if len(candidates) <= count:
        return candidates
    picks = np.linspace(0, len(candidates) - 1, num=count)
    return [candidates[int(round(idx))] for idx in picks]


def _make_normal_rows(
    *,
    pure_normal_values_ds: np.ndarray,
    pure_normal_timestamps_ds: Sequence[pd.Timestamp],
    abnormal_values_ds: np.ndarray,
    abnormal_timestamps_ds: Sequence[pd.Timestamp],
    abnormal_point_labels_ds: np.ndarray,
    channels: Sequence[Dict[str, Any]],
    seq_len: int,
    pure_count: int,
    abnormal_safe_count: int,
    global_index_start: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    zero_labels = _empty_label_block(seq_len, len(channels))
    global_idx = int(global_index_start)

    pure_starts = _evenly_spaced_starts(len(pure_normal_values_ds), seq_len, pure_count)
    for local_idx, start in enumerate(pure_starts):
        end = start + seq_len
        sample_id = f"swat_normal_pure_{local_idx:03d}"
        row = {
            "schema_version": "axis_swat_v1",
            "sample_id": sample_id,
            "base_sample_id": sample_id,
            "source": {
                "dataset": "SWaT",
                "source_file": "SWaT_Normal.csv",
                "split": "normal",
                "normal_source": "pure_normal",
                "downsample_factor": 2,
                "raw_start_index": int(start * 2),
                "raw_end_index_exclusive": int(end * 2),
                "downsampled_start_index": int(start),
                "downsampled_end_index_exclusive": int(end),
            },
            "original_data": {
                "timestamp_start": pure_normal_timestamps_ds[start].strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_end": pure_normal_timestamps_ds[end - 1].strftime("%Y-%m-%d %H:%M:%S"),
            },
            "normal_series": None,
            "series": {
                "shape": [seq_len, len(channels)],
                "values": pure_normal_values_ds[start:end].astype(np.float32).tolist(),
                "labels": zero_labels,
                "timestamps": _window_timestamps(pure_normal_timestamps_ds, start, end),
            },
            "channels": list(channels),
            "causal_graph": {},
            "root_cause": {
                "channel_id": None,
                "time_index": None,
                "interval": None,
                "provided_by": "none",
            },
            "synthetic_label": {
                "is_synthetic": False,
                "base_sample_id": sample_id,
                "anomaly_type": None,
                "root_anomaly_name": None,
                "anomaly_scope": None,
                "affected_channels": [],
                "root_cause_channels": [],
                "causal_path": [],
                "abnormal_edges": [],
            },
            "sequence_length": seq_len,
            "global_sample_index": global_idx,
        }
        rows.append(row)
        index_rows.append(
            {
                "sample_id": sample_id,
                "normal_source": "pure_normal",
                "series_start_index": int(start),
                "series_end_index_exclusive": int(end),
            }
        )
        global_idx += 1

    safe_starts = _safe_abnormal_normal_starts(
        abnormal_point_labels_ds,
        seq_len=seq_len,
        count=abnormal_safe_count,
        buffer_points=150,
    )
    for local_idx, start in enumerate(safe_starts):
        end = start + seq_len
        sample_id = f"swat_normal_abfile_{local_idx:03d}"
        row = {
            "schema_version": "axis_swat_v1",
            "sample_id": sample_id,
            "base_sample_id": sample_id,
            "source": {
                "dataset": "SWaT",
                "source_file": "SWaT_Abnormal.csv",
                "split": "normal",
                "normal_source": "abnormal_nonattack",
                "downsample_factor": 2,
                "raw_start_index": int(start * 2),
                "raw_end_index_exclusive": int(end * 2),
                "downsampled_start_index": int(start),
                "downsampled_end_index_exclusive": int(end),
            },
            "original_data": {
                "timestamp_start": abnormal_timestamps_ds[start].strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp_end": abnormal_timestamps_ds[end - 1].strftime("%Y-%m-%d %H:%M:%S"),
            },
            "normal_series": None,
            "series": {
                "shape": [seq_len, len(channels)],
                "values": abnormal_values_ds[start:end].astype(np.float32).tolist(),
                "labels": zero_labels,
                "timestamps": _window_timestamps(abnormal_timestamps_ds, start, end),
            },
            "channels": list(channels),
            "causal_graph": {},
            "root_cause": {
                "channel_id": None,
                "time_index": None,
                "interval": None,
                "provided_by": "none",
            },
            "synthetic_label": {
                "is_synthetic": False,
                "base_sample_id": sample_id,
                "anomaly_type": None,
                "root_anomaly_name": None,
                "anomaly_scope": None,
                "affected_channels": [],
                "root_cause_channels": [],
                "causal_path": [],
                "abnormal_edges": [],
            },
            "sequence_length": seq_len,
            "global_sample_index": global_idx,
        }
        rows.append(row)
        index_rows.append(
            {
                "sample_id": sample_id,
                "normal_source": "abnormal_nonattack",
                "series_start_index": int(start),
                "series_end_index_exclusive": int(end),
            }
        )
        global_idx += 1
    return rows, index_rows


def _summary(
    *,
    output_dir: Path,
    rows: Sequence[Dict[str, Any]],
    anomaly_rows: Sequence[Dict[str, Any]],
    normal_rows: Sequence[Dict[str, Any]],
    downsample_factor: int,
    seq_len: int,
    channels: Sequence[Dict[str, Any]],
    event_index: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "dataset": "SWaT",
        "schema_version": "axis_swat_v1",
        "output_dir": str(output_dir),
        "downsample_factor": int(downsample_factor),
        "sequence_length": int(seq_len),
        "num_features": int(len(channels)),
        "num_total_series": int(len(rows)),
        "num_anomaly_series": int(len(anomaly_rows)),
        "num_normal_series": int(len(normal_rows)),
        "num_attack_labels_raw": 25,
        "num_attack_events_used": int(len(event_index)),
        "channel_names": [str(ch["name"]) for ch in channels],
        "example_sample_ids": [row["sample_id"] for row in rows[:12]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a SWaT dataset in Axis raw_series.jsonl format.")
    parser.add_argument("--output-dir", default="data/swat_axis_v1")
    parser.add_argument("--downsample-factor", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=500)
    parser.add_argument("--pure-normal-count", type=int, default=40)
    parser.add_argument("--abnormal-normal-count", type=int, default=40)
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    label_rows = _load_label_rows(RAW_LABEL)
    normal_df = _read_swat_csv(RAW_NORMAL)
    abnormal_df = _read_swat_csv(RAW_ABNORMAL)
    normal_features = _feature_columns(normal_df)
    abnormal_features = _feature_columns(abnormal_df)
    if normal_features != abnormal_features:
        raise ValueError("SWaT normal/abnormal feature columns do not match after cleaning.")
    feature_columns = list(normal_features)

    _coerce_numeric_columns(normal_df, feature_columns)
    _coerce_numeric_columns(abnormal_df, feature_columns)

    normal_values = normal_df[feature_columns].values.astype(np.float32)
    abnormal_values = abnormal_df[feature_columns].values.astype(np.float32)
    normal_ts = pd.to_datetime(normal_df["Timestamp"].astype(str).str.strip(), errors="coerce").tolist()
    abnormal_ts = pd.to_datetime(
        abnormal_df["Timestamp"].astype(str).str.strip(),
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce",
    ).tolist()

    abnormal_point_labels = (
        abnormal_df["Normal/Attack"]
        .astype(str)
        .str.strip()
        .map({"Normal": 0, "Attack": 1, "A ttack": 1})
        .fillna(0)
        .astype(np.uint8)
        .values
    )
    abnormal_node_labels = np.zeros((len(abnormal_df), len(feature_columns)), dtype=np.uint8)

    attack_events = _extract_attack_events(
        label_rows,
        np.asarray(abnormal_ts, dtype="datetime64[ns]"),
        abnormal_point_labels,
        feature_columns,
        int(args.downsample_factor),
    )
    feature_to_idx = {_canonicalize_signal_name(col): idx for idx, col in enumerate(feature_columns)}
    for event in attack_events:
        for channel_id in event.attacked_channel_ids:
            idx = int(channel_id.split("_")[1])
            abnormal_node_labels[event.raw_indices, idx] = 1

    normal_values_ds, _, _, normal_ts_ds = _downsample_with_labels(
        normal_values,
        np.zeros(len(normal_values), dtype=np.uint8),
        np.zeros((len(normal_values), len(feature_columns)), dtype=np.uint8),
        normal_ts,
        int(args.downsample_factor),
    )
    abnormal_values_ds, abnormal_point_labels_ds, abnormal_node_labels_ds, abnormal_ts_ds = _downsample_with_labels(
        abnormal_values,
        abnormal_point_labels,
        abnormal_node_labels,
        abnormal_ts,
        int(args.downsample_factor),
    )

    mean, std = _fit_scaler(normal_values_ds)
    normal_values_ds = _apply_scaler(normal_values_ds, mean, std)
    abnormal_values_ds = _apply_scaler(abnormal_values_ds, mean, std)
    channels = _channel_specs(feature_columns, normal_values_ds)

    anomaly_rows, event_index = _make_anomaly_rows(
        values_ds=abnormal_values_ds,
        node_labels_ds=abnormal_node_labels_ds,
        timestamps_ds=abnormal_ts_ds,
        channels=channels,
        events=attack_events,
        seq_len=int(args.sequence_length),
    )
    normal_rows, normal_index = _make_normal_rows(
        pure_normal_values_ds=normal_values_ds,
        pure_normal_timestamps_ds=normal_ts_ds,
        abnormal_values_ds=abnormal_values_ds,
        abnormal_timestamps_ds=abnormal_ts_ds,
        abnormal_point_labels_ds=abnormal_point_labels_ds,
        channels=channels,
        seq_len=int(args.sequence_length),
        pure_count=int(args.pure_normal_count),
        abnormal_safe_count=int(args.abnormal_normal_count),
        global_index_start=len(anomaly_rows),
    )
    rows = anomaly_rows + normal_rows

    raw_series_path = output_dir / "raw_series.jsonl"
    attack_event_index_path = output_dir / "attack_event_index.json"
    normal_segment_index_path = output_dir / "normal_segment_index.json"
    summary_path = output_dir / "dataset_summary.json"

    write_jsonl(rows, raw_series_path)
    save_json(event_index, attack_event_index_path)
    save_json(normal_index, normal_segment_index_path)
    save_json(
        _summary(
            output_dir=output_dir,
            rows=rows,
            anomaly_rows=anomaly_rows,
            normal_rows=normal_rows,
            downsample_factor=int(args.downsample_factor),
            seq_len=int(args.sequence_length),
            channels=channels,
            event_index=event_index,
        ),
        summary_path,
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "raw_series": str(raw_series_path),
                "num_total_series": len(rows),
                "num_anomaly_series": len(anomaly_rows),
                "num_normal_series": len(normal_rows),
                "attack_events_used": len(event_index),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
