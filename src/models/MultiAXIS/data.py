from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


TEACHER_ANSWER_FIELDS = ("model_answer", "teacher_answer_llm", "answer")


def teacher_answer(row: Dict[str, Any]) -> str:
    for field in TEACHER_ANSWER_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def question_type_group(row: Dict[str, Any]) -> str:
    value = str(row.get("question_answer_type", "")).strip().lower()
    mapping = {
        "choice": "MC",
        "multiple_choice": "MC",
        "mc": "MC",
        "open": "OE",
        "open_ended": "OE",
        "oe": "OE",
        "judgment": "TF",
        "true_false": "TF",
        "tf": "TF",
    }
    if value in mapping:
        return mapping[value]
    question_type = str(row.get("question_type", "")).lower()
    if "choice" in question_type:
        return "MC"
    if "judg" in question_type or "true_false" in question_type:
        return "TF"
    if "open" in question_type:
        return "OE"
    raise ValueError(f"Cannot map question type: {value!r} / {question_type!r}")


def extract_values(row: Dict[str, Any]) -> np.ndarray:
    series = row.get("series")
    if not isinstance(series, dict) or "values" not in series:
        raise ValueError("Question record lacks series.values")
    values = np.asarray(series["values"], dtype=np.float32)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError(f"series.values must be non-empty [T,C], got {values.shape}")
    declared = series.get("shape")
    if declared is not None and tuple(declared) != tuple(values.shape):
        raise ValueError(f"Declared series shape {declared} != decoded shape {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("series.values contains NaN or infinity")
    return values


def extract_interval(row: Dict[str, Any], steps: int) -> Tuple[int, int]:
    interval = row.get("target_interval")
    if not isinstance(interval, dict):
        raise ValueError("Question record lacks target_interval")
    start, end = int(interval["start"]), int(interval["end"])
    if not 0 <= start < end <= steps:
        raise ValueError(f"Invalid target interval [{start}, {end}) for T={steps}")
    return start, end


def normalize_and_serialize(
    values: np.ndarray,
    interval: Tuple[int, int],
    epsilon: float = 1e-5,
    scale: int = 100,
) -> Tuple[np.ndarray, List[List[int]]]:
    mean = values.mean(axis=0, keepdims=True, dtype=np.float64)
    std = values.std(axis=0, keepdims=True, dtype=np.float64)
    normalized = ((values.astype(np.float64) - mean) / (std + epsilon)).astype(np.float32)
    start, end = interval
    serialized = np.rint(normalized[start:end].T * scale).astype(np.int64).tolist()
    if len(serialized) != values.shape[1] or any(len(x) != end - start for x in serialized):
        raise AssertionError("Channel-major serialized window shape is inconsistent")
    return normalized, serialized


class ManifestDataset(Dataset):
    """Random-access sanitized manifest; question JSONL is read by byte offset."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        window_epsilon: float = 1e-5,
        window_scale: int = 100,
    ):
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.window_epsilon = window_epsilon
        self.window_scale = window_scale
        index_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".idx.json")
        self.offsets = json.loads(index_path.read_text(encoding="utf-8"))
        groups_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".groups.json")
        self.groups = (
            json.loads(groups_path.read_text(encoding="utf-8"))
            if groups_path.exists() else [[index] for index in range(len(self.offsets))]
        )
        self._manifest_handle = None
        self._question_handles: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_manifest_handle"] = None
        state["_question_handles"] = {}
        return state

    def _manifest_record(self, index: int) -> Dict[str, Any]:
        if self._manifest_handle is None:
            self._manifest_handle = self.manifest_path.open("rb")
        self._manifest_handle.seek(self.offsets[index])
        return json.loads(self._manifest_handle.readline())

    def _question_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        relative = record["question_path"]
        handle = self._question_handles.get(relative)
        if handle is None:
            handle = (self.data_root / relative).open("rb")
            self._question_handles[relative] = handle
        handle.seek(record["question_offset"])
        raw = handle.read(record["question_length"])
        row = json.loads(raw)
        if str(row.get("question")) != record["question"]:
            raise RuntimeError("Manifest/question text mismatch; source data changed")
        return row

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self._manifest_record(index)
        row = self._question_record(record)
        values = extract_values(row)
        interval = extract_interval(row, values.shape[0])
        if list(interval) != record["interval"]:
            raise RuntimeError("Manifest/question interval mismatch; source data changed")
        normalized, serialized = normalize_and_serialize(
            values,
            interval,
            epsilon=self.window_epsilon,
            scale=self.window_scale,
        )
        sample = {
            "normalized_series": normalized,
            "question": record["question"],
            "answer": record.get("teacher_answer"),
            "interval": interval,
            "window_values": serialized,
            "time_count": values.shape[0],
            "channel_count": values.shape[1],
            "sample_id": record["sample_id"],
            "base_sample_id": record["base_sample_id"],
            "question_group": record["question_group"],
            "dataset": record.get("dataset", "train"),
        }
        if "label_reference" in record:
            sample["label_reference"] = record["label_reference"]
        return sample


def collate_multiaxis(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    max_time = max(sample["time_count"] for sample in samples)
    max_channels = max(sample["channel_count"] for sample in samples)
    batch = len(samples)
    series = torch.zeros(batch, max_time, max_channels, dtype=torch.float32)
    time_mask = torch.zeros(batch, max_time, dtype=torch.bool)
    channel_mask = torch.zeros(batch, max_channels, dtype=torch.bool)
    for index, sample in enumerate(samples):
        steps, channels = sample["time_count"], sample["channel_count"]
        value = torch.from_numpy(sample["normalized_series"])
        if value.shape != (steps, channels):
            raise AssertionError("Normalized series shape changed during collation")
        series[index, :steps, :channels] = value
        time_mask[index, :steps] = True
        channel_mask[index, :channels] = True
    return {
        "normalized_series": series,
        "time_mask": time_mask,
        "channel_mask": channel_mask,
        "questions": [sample["question"] for sample in samples],
        "answers": [sample.get("answer") for sample in samples],
        "intervals": [sample["interval"] for sample in samples],
        "window_values": [sample["window_values"] for sample in samples],
        "time_counts": [sample["time_count"] for sample in samples],
        "channel_counts": [sample["channel_count"] for sample in samples],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "base_sample_ids": [sample["base_sample_id"] for sample in samples],
        "question_groups": [sample["question_group"] for sample in samples],
        "datasets": [sample["dataset"] for sample in samples],
        "label_references": [sample.get("label_reference") for sample in samples],
    }
