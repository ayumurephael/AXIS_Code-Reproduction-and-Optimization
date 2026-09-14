from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.question_provider import expand_samples_with_analysis_windows
from src.mvaxis.utils import read_jsonl, save_json, write_jsonl


DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

HAI_MEDIA_BASE = "https://media.githubusercontent.com/media/icsdataset/hai/master"
EXATHLON_TREE_URL = "https://api.github.com/repos/exathlonbenchmark/exathlon/git/trees/master?recursive=1"
EXATHLON_BLOB_URL = "https://api.github.com/repos/exathlonbenchmark/exathlon/git/blobs/{sha}"


def _canonicalize_channel_name(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum() or ch == "_")


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
    seen: set[tuple[int, int]] = set()
    out: List[tuple[str, int]] = []
    for role, desired_start in variants:
        bounds = _fit_window_bounds(total_len, desired_start, seq_len)
        if bounds in seen:
            continue
        seen.add(bounds)
        out.append((role, bounds[0]))
    return out


def _evenly_spaced_starts(total_len: int, seq_len: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if total_len <= seq_len:
        return [0] * count
    max_start = total_len - seq_len
    if count == 1:
        return [max_start // 2]
    return [int(round(pos)) for pos in np.linspace(0, max_start, num=count)]


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


def _safe_normal_starts(labels: np.ndarray, seq_len: int, count: int, buffer_points: int) -> List[int]:
    forbidden = _dilate_positive_mask(labels.astype(np.uint8), buffer_points)
    segments = [(left, right) for left, right in _contiguous_false_segments(forbidden) if right - left >= seq_len]
    if not segments:
        return []
    total = sum(right - left for left, right in segments)
    starts: List[int] = []
    for left, right in segments:
        seg_len = right - left
        local_count = max(1, int(round(count * (seg_len / total))))
        for local_start in _evenly_spaced_starts(seg_len, seq_len, local_count):
            starts.append(left + local_start)
    starts = sorted(dict.fromkeys(starts))
    if len(starts) <= count:
        return starts
    picks = np.linspace(0, len(starts) - 1, num=count)
    return [starts[int(round(idx))] for idx in picks]


def _find_attack_ranges(labels: Sequence[int]) -> List[tuple[int, int]]:
    arr = np.asarray(labels, dtype=np.uint8)
    ranges: List[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(arr):
        if value > 0 and start is None:
            start = idx
        if start is not None and (value == 0 or idx == len(arr) - 1):
            end = idx if value == 0 else idx + 1
            ranges.append((int(start), int(end)))
            start = None
    return ranges


def _download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def _channel_role(name: str) -> str:
    upper = str(name).upper()
    if upper.endswith("D") or upper.endswith("R") or upper.endswith("GO") or upper.endswith("SD"):
        return "actuator"
    return "sensor"


def _make_channels(feature_names: Sequence[str], normal_values: np.ndarray) -> List[Dict[str, Any]]:
    channels: List[Dict[str, Any]] = []
    for idx, name in enumerate(feature_names):
        col = normal_values[:, idx]
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
                "role": _channel_role(name),
                "type": "continuous",
                "description": f"HAI signal {name}.",
                "prior_relations": [],
                "active": True,
            }
            )
    return channels


def _github_get_json(url: str) -> Dict[str, Any]:
    resp = requests.get(url, headers={"Accept": "application/vnd.github+json"}, timeout=180)
    resp.raise_for_status()
    return resp.json()


def _github_get_blob_bytes(sha: str) -> bytes:
    cache_path = RAW_DIR / "exathlon" / "blob_cache" / f"{sha}.bin"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    url = EXATHLON_BLOB_URL.format(sha=sha)
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            raw_resp = requests.get(url, headers={"Accept": "application/vnd.github.raw"}, timeout=300)
            raw_resp.raise_for_status()
            content_type = raw_resp.headers.get("Content-Type", "")
            if "application/json" not in content_type.lower():
                cache_path.write_bytes(raw_resp.content)
                return raw_resp.content
            obj = raw_resp.json()
            if obj.get("encoding") == "base64":
                import base64

                decoded = base64.b64decode(obj["content"])
                cache_path.write_bytes(decoded)
                return decoded
            raise RuntimeError(f"Unexpected GitHub blob response for {sha}: {content_type}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= 5:
                break
            time.sleep(2.0 * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to download blob {sha}")


def _canonicalize_exathlon_feature_name(name: str) -> str:
    # Spark traces prefix many metrics with the app id (for example "1_" or "10_").
    # Stripping that prefix aligns app-specific traces to a single shared channel schema.
    return re.sub(r"^\d+_", "", str(name))


def _load_exathlon_tree(cache_path: Path) -> List[Dict[str, Any]]:
    if cache_path.exists():
        obj = json.loads(cache_path.read_text(encoding="utf-8"))
        return list(obj.get("tree", []))
    obj = _github_get_json(EXATHLON_TREE_URL)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return list(obj.get("tree", []))


def _exathlon_part_order(path: str) -> tuple[int, str]:
    suffix = Path(path).suffix.lower()
    if suffix == ".zip":
        return (999, path)
    if suffix.startswith(".z") and len(suffix) == 4 and suffix[2:].isdigit():
        return (int(suffix[2:]), path)
    return (1000, path)


def _exathlon_trace_parts(tree: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in tree:
        path = str(item.get("path", ""))
        if not path.startswith("data/raw/app"):
            continue
        if not any(path.endswith(ext) for ext in (".zip", ".z01", ".z02", ".z03", ".z04", ".z05", ".z06", ".z07", ".z08", ".z09")):
            continue
        parts = path.split("/")
        if len(parts) == 4:
            trace_name = Path(parts[-1]).stem
        elif len(parts) == 5:
            trace_name = parts[-2]
        else:
            continue
        groups.setdefault(trace_name, []).append(item)
    for part_list in groups.values():
        part_list.sort(key=lambda obj: _exathlon_part_order(str(obj.get("path", ""))))
    return groups


def _download_exathlon_ground_truth(raw_dir: Path) -> pd.DataFrame:
    out_zip = raw_dir / "exathlon_ground_truth.zip"
    if not out_zip.exists():
        tree = _load_exathlon_tree(raw_dir / "exathlon_tree.json")
        matches = [item for item in tree if item.get("path") == "data/raw/ground_truth.zip"]
        if not matches:
            raise FileNotFoundError("ground_truth.zip not found in Exathlon tree")
        out_zip.write_bytes(_github_get_blob_bytes(str(matches[0]["sha"])))
    with zipfile.ZipFile(out_zip, "r") as zf:
        name = zf.namelist()[0]
        with zf.open(name) as f:
            df = pd.read_csv(f)
    df["trace_name"] = df["trace_name"].astype(str)
    return df


def _read_exathlon_trace_frame(parts: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    def _read_csv_with_schema(path_or_buffer: Any) -> pd.DataFrame:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        header = pd.read_csv(path_or_buffer, nrows=0)
        columns = list(header.columns)
        dtype_map = {col: np.float32 for col in columns if col != "t"}
        dtype_map["t"] = np.int64
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        return pd.read_csv(path_or_buffer, dtype=dtype_map)

    blob = bytearray()
    for item in parts:
        blob.extend(_github_get_blob_bytes(str(item["sha"])))
    archive_bytes = bytes(blob)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV found inside Exathlon archive")
            with zf.open(csv_names[0]) as f:
                df = _read_csv_with_schema(f)
    except zipfile.BadZipFile:
        with tempfile.TemporaryDirectory(prefix="exathlon_zip_") as tmpdir:
            tmp_path = Path(tmpdir)
            archive_path = tmp_path / "trace.zip"
            archive_path.write_bytes(archive_bytes)
            list_cmd = ["tar", "-tf", str(archive_path)]
            listed = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
            csv_names = [line.strip() for line in listed.stdout.splitlines() if line.strip().lower().endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV found inside Exathlon archive (tar fallback)")
            subprocess.run(["tar", "-xf", str(archive_path), "-C", str(tmp_path), csv_names[0]], check=True)
            df = _read_csv_with_schema(tmp_path / csv_names[0])
    if "t" not in df.columns:
        raise KeyError("Exathlon trace is missing timestamp column 't'")
    raw_metric_cols = [col for col in df.columns if col != "t"]
    metric_values = df[raw_metric_cols].to_numpy(dtype=np.float32, copy=True)
    canonical_cols = [_canonicalize_exathlon_feature_name(col) for col in raw_metric_cols]
    if pd.Index(canonical_cols).duplicated().any():
        unique_names = sorted(set(canonical_cols))
        name_to_idx = {name: idx for idx, name in enumerate(unique_names)}
        merged = np.zeros((metric_values.shape[0], len(unique_names)), dtype=np.float32)
        counts = np.zeros(len(unique_names), dtype=np.float32)
        for col_idx, name in enumerate(canonical_cols):
            target_idx = name_to_idx[name]
            merged[:, target_idx] += metric_values[:, col_idx]
            counts[target_idx] += 1.0
        counts = np.where(counts <= 0, 1.0, counts)
        merged = merged / counts
        metrics = pd.DataFrame(merged, columns=unique_names)
    else:
        sort_order = np.argsort(np.asarray(canonical_cols, dtype=object))
        unique_names = [canonical_cols[idx] for idx in sort_order]
        metrics = pd.DataFrame(metric_values[:, sort_order], columns=unique_names)
    df = pd.concat([df[["t"]].reset_index(drop=True), metrics.reset_index(drop=True)], axis=1)
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"]).reset_index(drop=True)
    return df


def _allocate_normal_series_counts(lengths: Sequence[int], target_total: int, seq_len: int) -> List[int]:
    if target_total <= 0 or not lengths:
        return [0] * len(lengths)
    eligible = [max(0, int(length) - seq_len + 1) for length in lengths]
    total_mass = sum(max(1, value) for value in eligible)
    if total_mass <= 0:
        return [0] * len(lengths)
    counts = [max(1, int(round(target_total * (max(1, value) / total_mass)))) for value in eligible]
    while sum(counts) > target_total:
        idx = max(range(len(counts)), key=lambda i: counts[i])
        if counts[idx] > 1:
            counts[idx] -= 1
        else:
            break
    while sum(counts) < target_total:
        idx = min(range(len(counts)), key=lambda i: counts[i])
        counts[idx] += 1
    return counts


def _coalesce_event_time(*values: Any) -> int:
    for value in values:
        if value is None:
            continue
        if pd.isna(value):
            continue
        return int(float(value))
    raise ValueError("No usable event time value was provided")


@dataclass
class HaiPrepared:
    raw_dir: Path
    raw_series_path: Path
    windows_path: Path
    raw_count: int
    window_count: int
    anomaly_raw_count: int
    normal_raw_count: int


@dataclass
class ExathlonPrepared:
    raw_dir: Path
    raw_series_path: Path
    windows_path: Path
    raw_count: int
    window_count: int
    anomaly_raw_count: int
    normal_raw_count: int
    feature_count: int


def prepare_hai_2305_axis_dataset(
    *,
    output_dir: Path,
    windows_dir: Path,
    seq_len: int,
    samples_per_series: int,
    min_window_size: int,
    max_window_size: int,
    anomaly_ratio: float,
    seed: int,
) -> HaiPrepared:
    raw_dir = RAW_DIR / "hai-23.05"
    files = [
        "hai-train1.csv",
        "hai-train2.csv",
        "hai-train3.csv",
        "hai-train4.csv",
        "hai-test1.csv",
        "hai-test2.csv",
        "label-test1.csv",
        "label-test2.csv",
        "summary_label1.txt",
        "summary_label2.txt",
    ]
    for name in files:
        _download_file(f"{HAI_MEDIA_BASE}/hai-23.05/{name}", raw_dir / name)

    train_frames = [pd.read_csv(raw_dir / f"hai-train{i}.csv") for i in range(1, 5)]
    test_frames = [pd.read_csv(raw_dir / f"hai-test{i}.csv") for i in range(1, 3)]
    label_frames = [pd.read_csv(raw_dir / f"label-test{i}.csv") for i in range(1, 3)]

    feature_names = [c for c in train_frames[0].columns if c != "timestamp"]
    train_values = np.vstack([df[feature_names].to_numpy(dtype=np.float32) for df in train_frames])
    mean = train_values.mean(axis=0, keepdims=True)
    std = train_values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    channels = _make_channels(feature_names, ((train_values - mean) / std).astype(np.float32))

    rows: List[Dict[str, Any]] = []
    attack_index: List[Dict[str, Any]] = []
    global_sample_index = 0

    anomaly_target_count = 0
    for test_idx, (test_df, label_df) in enumerate(zip(test_frames, label_frames), start=1):
        timestamps = [str(v) for v in test_df["timestamp"].tolist()]
        values = test_df[feature_names].to_numpy(dtype=np.float32)
        values = ((values - mean) / std).astype(np.float32)
        labels = label_df["label"].to_numpy(dtype=np.uint8)
        attack_ranges = _find_attack_ranges(labels)
        for attack_idx, (start, end) in enumerate(attack_ranges, start=1):
            variants = _attack_variant_starts(start, end, len(values), seq_len)
            attack_record = {
                "test_file": f"hai-test{test_idx}.csv",
                "attack_index": attack_idx,
                "interval_start": int(start),
                "interval_end_exclusive": int(end),
                "variants": [],
            }
            for role, desired_start in variants:
                left, right = _fit_window_bounds(len(values), desired_start, seq_len)
                local_start = max(0, start - left)
                local_end = min(right - left, end - left)
                sample_id = f"hai2305_attack_t{test_idx:02d}_{attack_idx:03d}_{role}"
                rows.append(
                    {
                        "schema_version": "axis_hai2305_v1",
                        "sample_id": sample_id,
                        "base_sample_id": f"hai2305_attack_t{test_idx:02d}_{attack_idx:03d}",
                        "source": {
                            "dataset": "HAI",
                            "dataset_version": "23.05",
                            "source_file": f"hai-test{test_idx}.csv",
                            "split": "attack",
                            "attack_index": attack_idx,
                            "event_role": role,
                            "raw_start_index": int(left),
                            "raw_end_index_exclusive": int(right),
                            "label_source": f"label-test{test_idx}.csv",
                        },
                        "original_data": {
                            "timestamp_start": timestamps[left],
                            "timestamp_end": timestamps[right - 1],
                            "label_granularity": "global_attack_interval",
                            "attack_window_start_index": int(start),
                            "attack_window_end_exclusive": int(end),
                        },
                        "normal_series": None,
                        "series": {
                            "shape": [int(right - left), int(len(feature_names))],
                            "values": values[left:right].tolist(),
                            "labels": labels[left:right].astype(int).tolist(),
                            "timestamps": timestamps[left:right],
                        },
                        "channels": channels,
                        "causal_graph": {},
                        "root_cause": {
                            "channel_id": None,
                            "time_index": int(local_start),
                            "interval": [int(local_start), int(local_end)],
                            "provided_by": "hai_global_attack_label",
                        },
                        "synthetic_label": {
                            "is_synthetic": False,
                            "base_sample_id": f"hai2305_attack_t{test_idx:02d}_{attack_idx:03d}",
                            "anomaly_type": "attack_event",
                            "root_anomaly_name": "HAI attack event",
                            "anomaly_scope": "global",
                            "affected_channels": [],
                            "root_cause_channels": [],
                            "causal_path": [],
                            "abnormal_edges": [],
                            "label_granularity": "global_attack_interval",
                        },
                        "sequence_length": int(right - left),
                        "global_sample_index": int(global_sample_index),
                    }
                )
                attack_record["variants"].append(
                    {
                        "sample_id": sample_id,
                        "event_role": role,
                        "series_start_index": int(left),
                        "series_end_index_exclusive": int(right),
                        "local_attack_interval": [int(local_start), int(local_end)],
                    }
                )
                global_sample_index += 1
                anomaly_target_count += 1
            attack_index.append(attack_record)

    normal_rows: List[Dict[str, Any]] = []
    train_target = max(1, anomaly_target_count // 2)
    test_target = max(1, anomaly_target_count - train_target)

    train_segments = _evenly_spaced_starts(len(train_values), seq_len, train_target)
    train_timestamps = [str(v) for frame in train_frames for v in frame["timestamp"].tolist()]
    zero_labels = [0] * seq_len
    for idx, start in enumerate(train_segments, start=1):
        end = min(len(train_values), start + seq_len)
        sample_id = f"hai2305_normal_train_{idx:03d}"
        normal_rows.append(
            {
                "schema_version": "axis_hai2305_v1",
                "sample_id": sample_id,
                "base_sample_id": sample_id,
                "source": {
                    "dataset": "HAI",
                    "dataset_version": "23.05",
                    "source_file": "hai-train*.csv",
                    "split": "normal",
                    "normal_source": "train_even",
                    "raw_start_index": int(start),
                    "raw_end_index_exclusive": int(end),
                },
                "original_data": {
                    "timestamp_start": train_timestamps[start],
                    "timestamp_end": train_timestamps[end - 1],
                    "label_granularity": "global_attack_interval",
                },
                "normal_series": None,
                "series": {
                    "shape": [int(end - start), int(len(feature_names))],
                    "values": train_values[start:end].astype(np.float32).tolist(),
                    "labels": zero_labels[: end - start],
                    "timestamps": train_timestamps[start:end],
                },
                "channels": channels,
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
                    "label_granularity": "global_attack_interval",
                },
                "sequence_length": int(end - start),
                "global_sample_index": int(global_sample_index),
            }
        )
        global_sample_index += 1

    safe_test_rows_needed = max(0, anomaly_target_count - len(normal_rows))
    for test_idx, (test_df, label_df) in enumerate(zip(test_frames, label_frames), start=1):
        if safe_test_rows_needed <= 0:
            break
        timestamps = [str(v) for v in test_df["timestamp"].tolist()]
        values = test_df[feature_names].to_numpy(dtype=np.float32)
        values = ((values - mean) / std).astype(np.float32)
        labels = label_df["label"].to_numpy(dtype=np.uint8)
        safe_starts = _safe_normal_starts(labels, seq_len=seq_len, count=safe_test_rows_needed, buffer_points=300)
        for idx, start in enumerate(safe_starts, start=1):
            if safe_test_rows_needed <= 0:
                break
            end = min(len(values), start + seq_len)
            sample_id = f"hai2305_normal_test{test_idx}_{idx:03d}"
            normal_rows.append(
                {
                    "schema_version": "axis_hai2305_v1",
                    "sample_id": sample_id,
                    "base_sample_id": sample_id,
                    "source": {
                        "dataset": "HAI",
                        "dataset_version": "23.05",
                        "source_file": f"hai-test{test_idx}.csv",
                        "split": "normal",
                        "normal_source": "test_safe",
                        "raw_start_index": int(start),
                        "raw_end_index_exclusive": int(end),
                    },
                    "original_data": {
                        "timestamp_start": timestamps[start],
                        "timestamp_end": timestamps[end - 1],
                        "label_granularity": "global_attack_interval",
                    },
                    "normal_series": None,
                    "series": {
                        "shape": [int(end - start), int(len(feature_names))],
                        "values": values[start:end].tolist(),
                        "labels": zero_labels[: end - start],
                        "timestamps": timestamps[start:end],
                    },
                    "channels": channels,
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
                        "label_granularity": "global_attack_interval",
                    },
                    "sequence_length": int(end - start),
                    "global_sample_index": int(global_sample_index),
                }
            )
            global_sample_index += 1
            safe_test_rows_needed -= 1

    all_rows = rows + normal_rows
    raw_series_path = output_dir / "raw_series.jsonl"
    write_jsonl(all_rows, raw_series_path)

    sampled_windows = expand_samples_with_analysis_windows(
        all_rows,
        seed=seed,
        samples_per_series=samples_per_series,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        anomaly_ratio=anomaly_ratio,
    )
    windows_path = windows_dir / f"windows_{len(sampled_windows)}.jsonl"
    write_jsonl(sampled_windows, windows_path)

    save_json(
        {
            "dataset": "HAI",
            "dataset_version": "23.05",
            "schema_version": "axis_hai2305_v1",
            "raw_series": str(raw_series_path),
            "windows": str(windows_path),
            "num_features": len(feature_names),
            "sequence_length": seq_len,
            "raw_series_count": len(all_rows),
            "anomaly_raw_series_count": len(rows),
            "normal_raw_series_count": len(normal_rows),
            "window_count": len(sampled_windows),
            "min_window_size": min_window_size,
            "max_window_size": max_window_size,
            "samples_per_series": samples_per_series,
            "anomaly_ratio": anomaly_ratio,
            "seed": seed,
            "notes": [
                "HAI 23.05 provides global attack-interval labels but not channel-level root-cause labels.",
                "series.labels is stored as a 1D anomaly mask aligned to timestamps.",
                "Downstream multi-axis QA should treat this dataset as global-label supervision unless extra channel annotations are added later.",
            ],
        },
        output_dir / "dataset_summary.json",
    )
    save_json({"attacks": attack_index}, output_dir / "attack_event_index.json")
    save_json(
        {
            "source_status": "ready",
            "download_method": "media.githubusercontent.com",
            "raw_files": files,
        },
        output_dir / "source_status.json",
    )

    return HaiPrepared(
        raw_dir=output_dir,
        raw_series_path=raw_series_path,
        windows_path=windows_path,
        raw_count=len(all_rows),
        window_count=len(sampled_windows),
        anomaly_raw_count=len(rows),
        normal_raw_count=len(normal_rows),
    )


def prepare_exathlon_axis_dataset(
    *,
    output_dir: Path,
    windows_dir: Path,
    seq_len: int,
    samples_per_series: int,
    min_window_size: int,
    max_window_size: int,
    anomaly_ratio: float,
    seed: int,
    max_features: int = 64,
) -> ExathlonPrepared:
    raw_dir = RAW_DIR / "exathlon"
    tree = _load_exathlon_tree(raw_dir / "exathlon_tree.json")
    trace_parts = _exathlon_trace_parts(tree)
    if not trace_parts:
        raise RuntimeError("No Exathlon raw trace archives were found")

    ground_truth = _download_exathlon_ground_truth(raw_dir)
    gt_by_trace: Dict[str, List[Dict[str, Any]]] = {}
    for record in ground_truth.to_dict(orient="records"):
        gt_by_trace.setdefault(str(record["trace_name"]), []).append(record)

    common_cols: set[str] | None = None
    normal_sum: Dict[str, float] = {}
    normal_sumsq: Dict[str, float] = {}
    normal_count: Dict[str, int] = {}
    trace_meta: List[Dict[str, Any]] = []

    for idx, trace_name in enumerate(sorted(trace_parts), start=1):
        df = _read_exathlon_trace_frame(trace_parts[trace_name])
        feature_cols = [col for col in df.columns if col != "t"]
        feature_set = set(feature_cols)
        common_cols = feature_set if common_cols is None else (common_cols & feature_set)
        is_normal = trace_name not in gt_by_trace
        if is_normal:
            feature_values = df[feature_cols].to_numpy(dtype=np.float64)
            for col_idx, col_name in enumerate(feature_cols):
                col = feature_values[:, col_idx]
                normal_sum[col_name] = normal_sum.get(col_name, 0.0) + float(np.sum(col))
                normal_sumsq[col_name] = normal_sumsq.get(col_name, 0.0) + float(np.sum(col * col))
                normal_count[col_name] = normal_count.get(col_name, 0) + int(col.shape[0])
        trace_meta.append(
            {
                "trace_name": trace_name,
                "length": int(len(df)),
                "is_normal": bool(is_normal),
                "event_count": len(gt_by_trace.get(trace_name, [])),
            }
        )
        if idx % 10 == 0:
            print(f"[exathlon pass1] traces={idx}/{len(trace_parts)} common_cols={len(common_cols or [])}", flush=True)

    if not common_cols:
        raise RuntimeError("Exathlon common feature intersection is empty")

    common_cols_sorted = sorted(common_cols)
    variance_rows: List[tuple[str, float]] = []
    for col_name in common_cols_sorted:
        count = float(normal_count.get(col_name, 0))
        if count <= 0:
            continue
        mean = normal_sum[col_name] / count
        var = max(0.0, (normal_sumsq[col_name] / count) - (mean * mean))
        variance_rows.append((col_name, var))
    variance_rows.sort(key=lambda item: item[1], reverse=True)
    selected_features = [name for name, _ in variance_rows[: max(1, int(max_features))]]
    selected_features = sorted(selected_features)

    if not selected_features:
        raise RuntimeError("No Exathlon features were selected")

    normal_means = np.array([normal_sum[name] / max(1, normal_count[name]) for name in selected_features], dtype=np.float64)
    normal_stds = np.array(
        [
            max(1e-6, ((normal_sumsq[name] / max(1, normal_count[name])) - ((normal_sum[name] / max(1, normal_count[name])) ** 2)) ** 0.5)
            for name in selected_features
        ],
        dtype=np.float64,
    )

    anomaly_plan_rows: List[Dict[str, Any]] = []
    for meta in trace_meta:
        trace_name = str(meta["trace_name"])
        for event_idx, event in enumerate(gt_by_trace.get(trace_name, []), start=1):
            event_start = _coalesce_event_time(event.get("root_cause_start"))
            event_end = _coalesce_event_time(event.get("extended_effect_end"), event.get("root_cause_end"), event_start)
            desired_start = ((event_start + event_end) // 2) - (seq_len // 2)
            left, right = _fit_window_bounds(int(meta["length"]), desired_start, seq_len)
            anomaly_plan_rows.append(
                {
                    "trace_name": trace_name,
                    "event_index": event_idx,
                    "series_start": int(left),
                    "series_end": int(right),
                    "event_start": event_start,
                    "event_root_end": _coalesce_event_time(event.get("root_cause_end"), event_end),
                    "event_effect_end": event_end,
                    "trace_type": str(event.get("trace_type") or "disturbed"),
                    "anomaly_type": str(event.get("anomaly_type") or "unknown"),
                    "anomaly_details": None if pd.isna(event.get("anomaly_details")) else str(event.get("anomaly_details")),
                }
            )

    normal_meta = [meta for meta in trace_meta if bool(meta["is_normal"])]
    normal_counts = _allocate_normal_series_counts([int(meta["length"]) for meta in normal_meta], len(anomaly_plan_rows), seq_len)
    normal_plan_rows: Dict[str, List[Dict[str, Any]]] = {}
    for meta, count in zip(normal_meta, normal_counts):
        starts = _evenly_spaced_starts(int(meta["length"]), seq_len, count)
        if not starts:
            continue
        normal_plan_rows[str(meta["trace_name"])] = [{"series_start": int(start), "series_end": int(min(int(meta["length"]), start + seq_len))} for start in starts]

    channels = [
        {
            "channel_id": f"ch_{idx}",
            "name": str(name),
            "unit": "z-score",
            "scale": {
                "mean": 0.0,
                "std": 1.0,
                "min": None,
                "max": None,
            },
            "role": "sensor",
            "type": "continuous",
            "description": f"Exathlon Spark metric {name}.",
            "prior_relations": [],
            "active": True,
        }
        for idx, name in enumerate(selected_features)
    ]
    rows: List[Dict[str, Any]] = []
    anomaly_index: List[Dict[str, Any]] = []
    trace_meta_by_name = {str(meta["trace_name"]): meta for meta in trace_meta}
    global_sample_index = 0

    anomaly_plans_by_trace: Dict[str, List[Dict[str, Any]]] = {}
    for plan in anomaly_plan_rows:
        anomaly_plans_by_trace.setdefault(str(plan["trace_name"]), []).append(plan)

    for idx, trace_name in enumerate(sorted(trace_parts), start=1):
        df = _read_exathlon_trace_frame(trace_parts[trace_name])
        missing = [name for name in selected_features if name not in df.columns]
        if missing:
            raise KeyError(f"Trace {trace_name} is missing selected Exathlon features: {missing[:5]}")
        ts_arr = pd.to_numeric(df["t"], errors="coerce").to_numpy(dtype=np.int64)
        timestamps = ts_arr.tolist()
        values = df[selected_features].to_numpy(dtype=np.float32)
        values = ((values - normal_means.astype(np.float32)) / normal_stds.astype(np.float32)).astype(np.float32)
        labels = np.zeros(len(df), dtype=np.uint8)
        for event in gt_by_trace.get(trace_name, []):
            left = _coalesce_event_time(event.get("root_cause_start"))
            right = _coalesce_event_time(event.get("extended_effect_end"), event.get("root_cause_end"), left)
            labels[(ts_arr >= left) & (ts_arr <= right)] = 1

        for plan in anomaly_plans_by_trace.get(trace_name, []):
            left = int(plan["series_start"])
            right = int(plan["series_end"])
            segment_ts = ts_arr[left:right]
            local_root_start = int(np.searchsorted(segment_ts, int(plan["event_start"]), side="left"))
            local_root_end = int(np.searchsorted(segment_ts, int(plan["event_root_end"]), side="right"))
            local_effect_end = int(np.searchsorted(segment_ts, int(plan["event_effect_end"]), side="right"))
            local_root_start = max(0, min(local_root_start, right - left - 1))
            local_root_end = max(local_root_start + 1, min(local_root_end, right - left))
            local_effect_end = max(local_root_end, min(local_effect_end, right - left))
            sample_id = f"exathlon_{trace_name}_event{int(plan['event_index']):02d}"
            rows.append(
                {
                    "schema_version": "axis_exathlon_v1",
                    "sample_id": sample_id,
                    "base_sample_id": f"exathlon_{trace_name}",
                    "source": {
                        "dataset": "Exathlon",
                        "source_file": trace_name,
                        "split": "attack",
                        "trace_type": plan["trace_type"],
                        "event_index": int(plan["event_index"]),
                        "raw_start_index": left,
                        "raw_end_index_exclusive": right,
                    },
                    "original_data": {
                        "timestamp_start": int(timestamps[left]),
                        "timestamp_end": int(timestamps[right - 1]),
                        "label_source": "ground_truth.csv",
                        "trace_name": trace_name,
                        "anomaly_details": plan["anomaly_details"],
                    },
                    "normal_series": None,
                    "series": {
                        "shape": [int(right - left), int(len(selected_features))],
                        "values": values[left:right].tolist(),
                        "labels": labels[left:right].astype(int).tolist(),
                        "timestamps": timestamps[left:right],
                    },
                    "channels": channels,
                    "causal_graph": {},
                    "root_cause": {
                        "channel_id": None,
                        "time_index": int(local_root_start),
                        "interval": [int(local_root_start), int(local_root_end)],
                        "provided_by": "exathlon_ground_truth_interval",
                    },
                    "synthetic_label": {
                        "is_synthetic": False,
                        "base_sample_id": f"exathlon_{trace_name}",
                        "anomaly_type": plan["anomaly_type"],
                        "root_anomaly_name": plan["trace_type"],
                        "anomaly_scope": "global",
                        "affected_channels": [],
                        "root_cause_channels": [],
                        "causal_path": [],
                        "abnormal_edges": [],
                        "label_granularity": "global_root_cause_and_effect_interval",
                        "effect_interval": [int(local_root_start), int(local_effect_end)],
                    },
                    "sequence_length": int(right - left),
                    "global_sample_index": int(global_sample_index),
                }
            )
            anomaly_index.append(
                {
                    "sample_id": sample_id,
                    "trace_name": trace_name,
                    "event_index": int(plan["event_index"]),
                    "series_start": left,
                    "series_end_exclusive": right,
                    "anomaly_type": plan["anomaly_type"],
                    "trace_type": plan["trace_type"],
                }
            )
            global_sample_index += 1

        for normal_idx, plan in enumerate(normal_plan_rows.get(trace_name, []), start=1):
            left = int(plan["series_start"])
            right = int(plan["series_end"])
            sample_id = f"exathlon_{trace_name}_normal{normal_idx:02d}"
            rows.append(
                {
                    "schema_version": "axis_exathlon_v1",
                    "sample_id": sample_id,
                    "base_sample_id": f"exathlon_{trace_name}",
                    "source": {
                        "dataset": "Exathlon",
                        "source_file": trace_name,
                        "split": "normal",
                        "trace_type": "undisturbed",
                        "raw_start_index": left,
                        "raw_end_index_exclusive": right,
                    },
                    "original_data": {
                        "timestamp_start": int(timestamps[left]),
                        "timestamp_end": int(timestamps[right - 1]),
                        "label_source": "ground_truth.csv",
                        "trace_name": trace_name,
                    },
                    "normal_series": None,
                    "series": {
                        "shape": [int(right - left), int(len(selected_features))],
                        "values": values[left:right].tolist(),
                        "labels": labels[left:right].astype(int).tolist(),
                        "timestamps": timestamps[left:right],
                    },
                    "channels": channels,
                    "causal_graph": {},
                    "root_cause": {
                        "channel_id": None,
                        "time_index": None,
                        "interval": None,
                        "provided_by": "none",
                    },
                    "synthetic_label": {
                        "is_synthetic": False,
                        "base_sample_id": f"exathlon_{trace_name}",
                        "anomaly_type": None,
                        "root_anomaly_name": None,
                        "anomaly_scope": None,
                        "affected_channels": [],
                        "root_cause_channels": [],
                        "causal_path": [],
                        "abnormal_edges": [],
                        "label_granularity": "global_root_cause_and_effect_interval",
                    },
                    "sequence_length": int(right - left),
                    "global_sample_index": int(global_sample_index),
                }
            )
            global_sample_index += 1

        if idx % 10 == 0:
            print(f"[exathlon pass2] traces={idx}/{len(trace_parts)} rows={len(rows)}", flush=True)

    raw_series_path = output_dir / "raw_series.jsonl"
    write_jsonl(rows, raw_series_path)

    sampled_windows = expand_samples_with_analysis_windows(
        rows,
        seed=seed,
        samples_per_series=samples_per_series,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        anomaly_ratio=anomaly_ratio,
    )
    windows_path = windows_dir / f"windows_{len(sampled_windows)}.jsonl"
    write_jsonl(sampled_windows, windows_path)

    summary = {
        "dataset": "Exathlon",
        "schema_version": "axis_exathlon_v1",
        "raw_series": str(raw_series_path),
        "windows": str(windows_path),
        "raw_trace_count": len(trace_parts),
        "ground_truth_event_rows": len(ground_truth),
        "common_feature_count": len(common_cols_sorted),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "sequence_length": seq_len,
        "raw_series_count": len(rows),
        "anomaly_raw_series_count": len(anomaly_plan_rows),
        "normal_raw_series_count": len(rows) - len(anomaly_plan_rows),
        "window_count": len(sampled_windows),
        "min_window_size": min_window_size,
        "max_window_size": max_window_size,
        "samples_per_series": samples_per_series,
        "anomaly_ratio": anomaly_ratio,
        "seed": seed,
        "notes": [
            "Exathlon raw Spark traces expose 2,283+ metrics; this conversion keeps the highest-variance common features on undisturbed traces for multi-axis tractability.",
            "Normalization statistics are fit only on undisturbed traces.",
            "series.labels is a global anomaly mask built from root_cause_start to extended_effect_end.",
        ],
    }
    save_json(summary, output_dir / "dataset_summary.json")
    save_json({"events": anomaly_index}, output_dir / "attack_event_index.json")
    save_json(
        {
            "dataset": "Exathlon",
            "source_status": "ready",
            "source": "https://github.com/exathlonbenchmark/exathlon",
            "download_method": "github_blob_api",
        },
        output_dir / "source_status.json",
    )

    return ExathlonPrepared(
        raw_dir=output_dir,
        raw_series_path=raw_series_path,
        windows_path=windows_path,
        raw_count=len(rows),
        window_count=len(sampled_windows),
        anomaly_raw_count=len(anomaly_plan_rows),
        normal_raw_count=len(rows) - len(anomaly_plan_rows),
        feature_count=len(selected_features),
    )


def _prepare_swat_windows(
    *,
    samples_per_series: int,
    min_window_size: int,
    max_window_size: int,
    anomaly_ratio: float,
    seed: int,
) -> Dict[str, Any]:
    raw_path = DATA_DIR / "swat_axis_v1" / "raw_series.jsonl"
    rows = read_jsonl(raw_path)
    out_dir = DATA_DIR / "swat_axis_v1_windows_qa"
    windows = expand_samples_with_analysis_windows(
        rows,
        seed=seed,
        samples_per_series=samples_per_series,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        anomaly_ratio=anomaly_ratio,
    )
    windows_path = out_dir / f"windows_{len(windows)}.jsonl"
    write_jsonl(windows, windows_path)
    summary = {
        "source": str(raw_path),
        "output_dir": str(out_dir),
        "windows_jsonl": str(windows_path),
        "num_series": len(rows),
        "samples_per_series": samples_per_series,
        "num_windows": len(windows),
        "anomalous_windows": int(sum(1 for row in windows if bool((row.get("target_output") or {}).get("fact_check", {}).get("is_anomalous")))),
        "normal_windows": int(sum(1 for row in windows if not bool((row.get("target_output") or {}).get("fact_check", {}).get("is_anomalous")))),
        "min_window_size": min_window_size,
        "max_window_size": max_window_size,
        "anomaly_ratio": anomaly_ratio,
        "seed": seed,
    }
    save_json(summary, out_dir / "summary.json")
    return {
        "dataset": "SWaT",
        "status": "ready",
        "raw_series_path": str(raw_path),
        "raw_series_count": len(rows),
        "sequence_length_values": sorted({int(r.get("sequence_length", 0)) for r in rows}),
        "windows_path": str(windows_path),
        "window_summary": summary,
        "notes": [
            "SWaT raw series are normalized and segmented to length 500.",
            "This swat_axis_v1_windows_qa folder is re-sampled with the current 15-60 QA-window setting.",
        ],
    }


def _swat_registry_entry() -> Dict[str, Any]:
    raw_path = DATA_DIR / "swat_axis_v1" / "raw_series.jsonl"
    windows_summary_path = DATA_DIR / "swat_axis_v1_windows" / "summary.json"
    rows = read_jsonl(raw_path)
    summary = json.loads(windows_summary_path.read_text(encoding="utf-8"))
    return {
        "dataset": "SWaT",
        "status": "ready_existing",
        "raw_series_path": str(raw_path),
        "raw_series_count": len(rows),
        "sequence_length_values": sorted({int(r.get("sequence_length", 0)) for r in rows}),
        "existing_window_summary": summary,
        "notes": [
            "Existing SWaT raw series are already normalized and segmented to length 500.",
            "Existing swat_axis_v1_windows were sampled with 80-160 windows; they are kept for reference.",
            "This legacy window folder is preserved only for reference; the QA-aligned one is written separately.",
        ],
    }


def _blocked_entry(name: str, reason: str, source: str) -> Dict[str, Any]:
    return {
        "dataset": name,
        "status": "blocked",
        "source": source,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real multivariate benchmarks into multi-axis compatible data folders.")
    parser.add_argument("--sequence-length", type=int, default=500)
    parser.add_argument("--samples-per-series", type=int, default=2)
    parser.add_argument("--min-window-size", type=int, default=15)
    parser.add_argument("--max-window-size", type=int, default=60)
    parser.add_argument("--anomaly-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--exathlon-max-features", type=int, default=64)
    args = parser.parse_args()

    registry: List[Dict[str, Any]] = []
    registry.append(_swat_registry_entry())
    registry.append(
        _prepare_swat_windows(
            samples_per_series=int(args.samples_per_series),
            min_window_size=int(args.min_window_size),
            max_window_size=int(args.max_window_size),
            anomaly_ratio=float(args.anomaly_ratio),
            seed=int(args.seed),
        )
    )

    hai = prepare_hai_2305_axis_dataset(
        output_dir=DATA_DIR / "hai_axis_v1",
        windows_dir=DATA_DIR / "hai_axis_v1_windows",
        seq_len=int(args.sequence_length),
        samples_per_series=int(args.samples_per_series),
        min_window_size=int(args.min_window_size),
        max_window_size=int(args.max_window_size),
        anomaly_ratio=float(args.anomaly_ratio),
        seed=int(args.seed),
    )
    registry.append(
        {
            "dataset": "HAI",
            "status": "ready",
            "version": "23.05",
            "raw_series_path": str(hai.raw_series_path),
            "windows_path": str(hai.windows_path),
            "raw_series_count": int(hai.raw_count),
            "window_count": int(hai.window_count),
            "anomaly_raw_series_count": int(hai.anomaly_raw_count),
            "normal_raw_series_count": int(hai.normal_raw_count),
        }
    )

    exathlon = prepare_exathlon_axis_dataset(
        output_dir=DATA_DIR / "exathlon_axis_v1",
        windows_dir=DATA_DIR / "exathlon_axis_v1_windows",
        seq_len=int(args.sequence_length),
        samples_per_series=int(args.samples_per_series),
        min_window_size=int(args.min_window_size),
        max_window_size=int(args.max_window_size),
        anomaly_ratio=float(args.anomaly_ratio),
        seed=int(args.seed),
        max_features=int(args.exathlon_max_features),
    )
    registry.append(
        {
            "dataset": "Exathlon",
            "status": "ready",
            "raw_series_path": str(exathlon.raw_series_path),
            "windows_path": str(exathlon.windows_path),
            "raw_series_count": int(exathlon.raw_count),
            "window_count": int(exathlon.window_count),
            "anomaly_raw_series_count": int(exathlon.anomaly_raw_count),
            "normal_raw_series_count": int(exathlon.normal_raw_count),
            "selected_feature_count": int(exathlon.feature_count),
        }
    )

    registry.append(
        _blocked_entry(
            "CISS 2020-OL",
            "Official source is access-controlled and requires a manual dataset request/approval.",
            "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/ciss/",
        )
    )
    registry.append(
        _blocked_entry(
            "WADI",
            "Official source is access-controlled and requires a manual dataset request/approval.",
            "https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/",
        )
    )
    registry.append(
        _blocked_entry(
            "LEMMA-RCA",
            "Primary public hosting is on HuggingFace, which is currently unreachable from this network path.",
            "https://lemma-rca.github.io/",
        )
    )

    save_json({"datasets": registry}, DATA_DIR / "real_benchmark_registry.json")
    print(json.dumps({"datasets": registry}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
