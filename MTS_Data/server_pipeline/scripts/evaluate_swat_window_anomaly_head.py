from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from _bootstrap import add_project_root

ROOT = add_project_root()
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_cf_ts.datasets import create_sliding_windows
from causal_cf_ts.engines import AnomalyEngine
from causal_cf_ts.models import GraphCounterfactualFlow
from src.mvaxis.utils import save_json


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_raw_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        rows[str(row["sample_id"])] = row
    return rows


def _load_model(checkpoint_path: Path, stats_path: Path) -> AnomalyEngine:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adjacency = state["adjacency"].detach().cpu().numpy()
    model = GraphCounterfactualFlow(
        adjacency=adjacency,
        horizon=1,
        hidden_dim=32,
        context_dim=32,
        flow_transforms=3,
        flow_hidden_features=(32, 32),
    ).to("cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    stats = np.load(stats_path)
    node_mean = torch.from_numpy(stats["node_mean"]).float()
    node_std = torch.from_numpy(stats["node_std"]).float()
    return AnomalyEngine(
        model=model,
        node_mean=node_mean,
        node_std=node_std,
        aggregate="sum",
        topk=5,
    )


def _score_sequence(anomaly_engine: AnomalyEngine, values: np.ndarray, batch_size: int = 256) -> np.ndarray:
    point_node_labels = np.zeros_like(values, dtype=np.int64)
    point_labels = np.zeros(values.shape[0], dtype=np.int64)
    windows = create_sliding_windows(
        values.astype(np.float32),
        L=8,
        H=1,
        point_labels=point_labels,
        point_node_labels=point_node_labels,
    )
    if len(windows.x_past) == 0:
        return np.zeros((0,), dtype=np.float32)
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows.x_past), batch_size):
            xb = torch.from_numpy(windows.x_past[start : start + batch_size]).float()
            yb = torch.from_numpy(windows.y_future[start : start + batch_size]).float()
            scores = anomaly_engine.score_windows(xb, yb).cpu().numpy().astype(np.float32)
            outputs.append(scores)
    return np.concatenate(outputs, axis=0).astype(np.float32)


def _roc_auc(labels: List[int], scores: List[float]) -> float | None:
    if not labels:
        return None
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    pos_ranks = ranks[y == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _confusion(labels: np.ndarray, preds: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.logical_and(labels == 1, preds == 1).sum())
    tn = int(np.logical_and(labels == 0, preds == 0).sum())
    fp = int(np.logical_and(labels == 0, preds == 1).sum())
    fn = int(np.logical_and(labels == 1, preds == 0).sum())
    return tp, tn, fp, fn


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _find_best_threshold(labels: List[int], scores: List[float]) -> Dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    thresholds = sorted(set(float(x) for x in s))
    if not thresholds:
        return {
            "threshold": None,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "coverage": 0.0,
            "f1": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    best: Dict[str, Any] | None = None
    for threshold in thresholds:
        preds = (s >= threshold).astype(np.int64)
        tp, tn, fp, fn = _confusion(y, preds)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        accuracy = _safe_div(tp + tn, len(y))
        candidate = {
            "threshold": float(threshold),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "coverage": recall,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
        if best is None:
            best = candidate
            continue
        if candidate["f1"] > best["f1"] + 1e-12:
            best = candidate
            continue
        if abs(candidate["f1"] - best["f1"]) <= 1e-12 and candidate["accuracy"] > best["accuracy"] + 1e-12:
            best = candidate
            continue
    return best or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SWaT anomaly head on current rich windows.")
    parser.add_argument("--raw-series", default="data/swat_axis_v1/raw_series.jsonl")
    parser.add_argument("--windows", default="data/swat_axis_v1_windows_rich/windows_320.jsonl")
    parser.add_argument("--checkpoint", default="outputs/rootcause_swat_paper_protocol/ourmodel_swat_paper_protocol.pt")
    parser.add_argument("--stats", default="outputs/rootcause_swat_paper_protocol/ourmodel_swat_paper_protocol_stats.npz")
    parser.add_argument("--output-dir", default="data/swat_axis_v1_windows_rich/anomaly_head_eval")
    args = parser.parse_args()

    raw_series_path = ROOT / args.raw_series
    windows_path = ROOT / args.windows
    checkpoint_path = REPO_ROOT / args.checkpoint
    stats_path = REPO_ROOT / args.stats
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_raw_rows(raw_series_path)
    window_rows = list(_iter_jsonl(windows_path))
    anomaly_engine = _load_model(checkpoint_path, stats_path)

    scores_by_sample: Dict[str, np.ndarray] = {}
    detailed_rows: List[Dict[str, Any]] = []
    interval_max_scores: List[float] = []
    interval_mean_scores: List[float] = []
    labels: List[int] = []
    valid_interval_count = 0

    for row in window_rows:
        source_sample_id = str(row["source_sample_id"])
        if source_sample_id not in scores_by_sample:
            sample = raw_rows[source_sample_id]
            values = np.asarray(sample["series"]["values"], dtype=np.float32)
            scores_by_sample[source_sample_id] = _score_sequence(anomaly_engine, values)
        seq_scores = scores_by_sample[source_sample_id]
        interval = row.get("target_interval") or {}
        start = int(interval.get("start", 0))
        end = int(interval.get("end", 0))
        score_start = max(0, start - 8)
        score_end = max(score_start, end - 8)
        score_slice = seq_scores[score_start:score_end]
        if score_slice.size == 0 and seq_scores.size > 0:
            nearest = min(max(0, start - 8), seq_scores.size - 1)
            score_slice = seq_scores[nearest : nearest + 1]
        has_valid = bool(score_slice.size > 0)
        valid_interval_count += int(has_valid)
        interval_max = float(np.max(score_slice)) if has_valid else float("-inf")
        interval_mean = float(np.mean(score_slice)) if has_valid else float("-inf")
        label = int(bool(row.get("has_anomaly")))
        labels.append(label)
        interval_max_scores.append(interval_max)
        interval_mean_scores.append(interval_mean)
        detailed_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "source_sample_id": source_sample_id,
                "has_anomaly": bool(label),
                "target_interval": {"start": start, "end": end},
                "interval_max_score": interval_max,
                "interval_mean_score": interval_mean,
                "valid_interval_score": has_valid,
                "image_path": row.get("image_path"),
                "root_cause_channel": row.get("root_cause_channel"),
                "affected_channels": row.get("affected_channels"),
            }
        )

    best_max = _find_best_threshold(labels, interval_max_scores)
    best_mean = _find_best_threshold(labels, interval_mean_scores)
    summary = {
        "checkpoint": str(checkpoint_path),
        "stats_file": str(stats_path),
        "windows_file": str(windows_path),
        "raw_series_file": str(raw_series_path),
        "num_windows": len(window_rows),
        "num_anomalous_windows": int(sum(labels)),
        "num_normal_windows": int(len(labels) - sum(labels)),
        "score_coverage": _safe_div(valid_interval_count, len(window_rows)),
        "interval_max_auroc": _roc_auc(labels, interval_max_scores),
        "interval_mean_auroc": _roc_auc(labels, interval_mean_scores),
        "best_interval_max_threshold_metrics": best_max,
        "best_interval_mean_threshold_metrics": best_mean,
    }

    details_path = output_dir / "window_scores.jsonl"
    with details_path.open("w", encoding="utf-8") as handle:
        for row in detailed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = output_dir / "summary.json"
    save_json(summary, summary_path)
    print(json.dumps({"summary": summary, "details_jsonl": str(details_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
