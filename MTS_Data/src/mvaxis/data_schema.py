from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


SCHEMA_VERSION = "mv-axis-0.1"

REQUIRED_TOP_LEVEL = {
    "schema_version",
    "sample_id",
    "base_sample_id",
    "source",
    "series",
    "channels",
    "target_interval",
    "question",
    "root_cause",
    "synthetic_label",
    "evidence_card",
    "target_output",
}


@dataclass
class ModelInput:
    values: np.ndarray
    labels: np.ndarray
    normal_values: Optional[np.ndarray]
    interval: Tuple[int, int]
    channels: List[Dict[str, Any]]
    causal_graph: Dict[str, Any]
    question_type: Optional[str]
    question: str
    target_output: Dict[str, Any]
    evidence_card: Dict[str, Any]
    root_cause_channel: Optional[str]
    is_anomalous: bool


def make_channel(
    channel_id: str,
    name: str,
    role: str,
    channel_type: str = "continuous",
    unit: Optional[str] = None,
    description: str = "",
    prior_relations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "channel_id": channel_id,
        "name": name,
        "unit": unit,
        "scale": {"mean": None, "std": None, "min": None, "max": None},
        "role": role,
        "type": channel_type,
        "description": description,
        "prior_relations": prior_relations or [],
    }


def attach_channel_scales(sample: Dict[str, Any]) -> None:
    values = np.asarray(sample["series"]["values"], dtype=float)
    for idx, channel in enumerate(sample["channels"]):
        x = values[:, idx]
        channel["scale"] = {
            "mean": float(np.mean(x)),
            "std": float(np.std(x) + 1e-8),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
        }


def validate_sample(sample: Dict[str, Any]) -> None:
    missing = REQUIRED_TOP_LEVEL.difference(sample)
    if missing:
        raise ValueError(f"Missing required sample fields: {sorted(missing)}")
    values = np.asarray(sample["series"].get("values"), dtype=float)
    if values.ndim != 2:
        raise ValueError("series.values must have shape [T, C]")
    labels = np.asarray(sample["series"].get("labels"), dtype=int)
    if labels.shape != values.shape:
        raise ValueError(f"series.labels shape {labels.shape} must match values {values.shape}")
    normal_values = sample.get("normal_series")
    if normal_values is None:
        original_data = sample.get("original_data") or {}
        normal_values = original_data.get("normal_series")
    if normal_values is not None and np.asarray(normal_values, dtype=float).shape != values.shape:
        raise ValueError("normal_series shape must match series.values")
    if len(sample["channels"]) != values.shape[1]:
        raise ValueError("channel metadata length must match series channel count")
    interval = sample["target_interval"]
    start, end = int(interval["start"]), int(interval["end"])
    if not (0 <= start < end <= values.shape[0]):
        raise ValueError(f"invalid target interval: {interval}")
    channel_ids = {c["channel_id"] for c in sample["channels"]}
    root_channel = sample["root_cause"].get("channel_id")
    if root_channel is not None and root_channel not in channel_ids:
        raise ValueError(f"root cause channel {root_channel} not found")


def convert_to_model_input(sample: Dict[str, Any]) -> ModelInput:
    validate_sample(sample)
    interval = sample["target_interval"]
    normal_values = sample.get("normal_series")
    if normal_values is None:
        normal_values = (sample.get("original_data") or {}).get("normal_series")
    return ModelInput(
        values=np.asarray(sample["series"]["values"], dtype=float),
        labels=np.asarray(sample["series"]["labels"], dtype=int),
        normal_values=np.asarray(normal_values, dtype=float) if normal_values is not None else None,
        interval=(int(interval["start"]), int(interval["end"])),
        channels=sample["channels"],
        causal_graph=sample.get("causal_graph", {}),
        question_type=sample.get("question_type"),
        question=sample["question"],
        target_output=sample["target_output"],
        evidence_card=sample["evidence_card"],
        root_cause_channel=sample["root_cause"].get("channel_id"),
        is_anomalous=bool(sample["target_output"]["fact_check"]["is_anomalous"]),
    )
