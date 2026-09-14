from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

from _bootstrap import add_project_root

ROOT = add_project_root()
REPO_ROOT = ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
TIMERCd_ROOT = ROOT / "references" / "TimeRCD"

for path in (SRC_ROOT, TIMERCd_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from causal_cf_ts.datasets import create_sliding_windows  # noqa: E402
from causal_cf_ts.engines import AnomalyEngine  # noqa: E402
from causal_cf_ts.models import GraphCounterfactualFlow  # noqa: E402
from evaluation.basic_metrics import basic_metricor, generate_curve  # noqa: E402
from src.mvaxis.utils import save_json  # noqa: E402


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def _time_label_and_score(
    anomaly_engine: AnomalyEngine,
    row: Dict[str, Any],
    score_reduction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(row["series"]["values"], dtype=np.float32)
    labels_2d = np.asarray(row["series"]["labels"], dtype=np.int64)
    if labels_2d.ndim == 1:
        labels_2d = labels_2d[:, None]

    point_scores = _score_sequence(anomaly_engine, values)
    point_labels = (labels_2d.max(axis=1) > 0).astype(np.int64)

    # Scores correspond to future points starting at absolute index 8.
    aligned_labels = point_labels[8:]
    effective_len = min(len(point_scores), len(aligned_labels))
    if effective_len <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

    aligned_labels = aligned_labels[:effective_len]
    aligned_scores = point_scores[:effective_len].astype(np.float64)

    if score_reduction not in {"max", "mean"}:
        raise ValueError(f"Unsupported score_reduction: {score_reduction}")
    return aligned_labels, aligned_scores


def _concatenate_with_gap(
    labels: List[np.ndarray],
    scores: List[np.ndarray],
    gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not labels:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    label_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    for index, (label, score) in enumerate(zip(labels, scores)):
        if index and gap > 0:
            label_parts.append(np.zeros(gap, dtype=np.int64))
            score_parts.append(np.zeros(gap, dtype=np.float64))
        label_parts.append(label.astype(np.int64))
        score_parts.append(score.astype(np.float64))
    return np.concatenate(label_parts), np.concatenate(score_parts)


def _compute_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    sliding_window: int,
    vus_version: str,
    vus_thresholds: int,
    workers: int,
    chunk_size: int,
) -> Dict[str, float]:
    grader = basic_metricor()
    metrics: Dict[str, float] = {}

    try:
        standard = grader.metric_standard_F1_chunked(
            labels,
            scores,
            chunk_size=max(1, int(chunk_size)),
            num_workers=max(1, int(workers)),
        )
    except Exception as exc:
        print(f"Chunked Standard-F1 failed ({exc}); falling back to sequential implementation.")
        standard = grader.metric_standard_F1(labels, scores)
    metrics["Standard-F1"] = float(standard["F1"])
    metrics["Standard-Precision"] = float(standard["Precision"])
    metrics["Standard-Recall"] = float(standard["Recall"])

    try:
        f1_t = grader.metric_F1_T(
            labels,
            scores,
            use_parallel=True,
            parallel_method="chunked",
            chunk_size=max(1, int(chunk_size)),
            max_workers=max(1, int(workers)),
        )
    except Exception as exc:
        print(f"Parallel F1_T failed ({exc}); falling back to sequential implementation.")
        f1_t = grader.metric_F1_T(labels, scores, use_parallel=False)
    metrics["F1_T"] = float(f1_t["F1_T"])
    metrics["Precision_T"] = float(f1_t["P_T"])
    metrics["Recall_T"] = float(f1_t["R_T"])
    metrics["Threshold_T"] = float(f1_t["thre_T"])

    try:
        aff_f, aff_p, aff_r = grader.metric_Affiliation_chunked(
            labels,
            scores,
            chunk_size=max(1, int(chunk_size)),
            num_workers=max(1, int(workers)),
        )
    except Exception as exc:
        print(f"Chunked Affiliation-F failed ({exc}); falling back to sequential implementation.")
        aff_f, aff_p, aff_r = grader.metric_Affiliation(labels, scores)
    metrics["Affiliation-F"] = float(aff_f)
    metrics["Affiliation-P"] = float(aff_p)
    metrics["Affiliation-R"] = float(aff_r)

    _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(
        labels.astype(int),
        scores.astype(float),
        int(sliding_window),
        vus_version,
        thre=int(vus_thresholds),
    )
    metrics["VUS-PR"] = float(vus_pr)
    metrics["VUS-ROC"] = float(vus_roc)
    metrics["AUC-PR"] = float(grader.metric_PR(labels, scores))
    metrics["AUC-ROC"] = float(grader.metric_ROC(labels, scores))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TimeRCD-style TSAD metrics on SWaT rich series.")
    parser.add_argument("--raw-series", default="data/swat_axis_v1/raw_series.jsonl")
    parser.add_argument("--checkpoint", default="outputs/rootcause_swat_paper_protocol/ourmodel_swat_paper_protocol.pt")
    parser.add_argument("--stats", default="outputs/rootcause_swat_paper_protocol/ourmodel_swat_paper_protocol_stats.npz")
    parser.add_argument("--output", default="data/swat_axis_v1_windows_rich/anomaly_head_eval/reference_tsad_metrics.json")
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--sliding-window", type=int, default=100)
    parser.add_argument("--vus-version", default="opt")
    parser.add_argument("--vus-thresholds", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--score-reduction", choices=["max", "mean"], default="max")
    args = parser.parse_args()

    raw_series_path = ROOT / args.raw_series
    checkpoint_path = REPO_ROOT / args.checkpoint
    stats_path = REPO_ROOT / args.stats
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_jsonl(raw_series_path))
    anomaly_engine = _load_model(checkpoint_path, stats_path)

    labels_by_series: List[np.ndarray] = []
    scores_by_series: List[np.ndarray] = []
    per_series_preview: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index % 25 == 0:
            print(f"Scoring series {index}/{len(rows)}")
        labels_i, scores_i = _time_label_and_score(anomaly_engine, row, str(args.score_reduction))
        labels_by_series.append(labels_i)
        scores_by_series.append(scores_i)
        if len(per_series_preview) < 10:
            per_series_preview.append(
                {
                    "index": index,
                    "sample_id": row.get("sample_id"),
                    "length": int(len(labels_i)),
                    "positive_points": int(labels_i.sum()),
                    "max_score": float(scores_i.max()) if len(scores_i) else 0.0,
                    "mean_score": float(scores_i.mean()) if len(scores_i) else 0.0,
                }
            )

    labels, scores = _concatenate_with_gap(labels_by_series, scores_by_series, int(args.gap))
    metrics = _compute_metrics(
        labels,
        scores,
        sliding_window=int(args.sliding_window),
        vus_version=str(args.vus_version),
        vus_thresholds=int(args.vus_thresholds),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
    )

    report = {
        "raw_series_file": str(raw_series_path),
        "checkpoint": str(checkpoint_path),
        "stats_file": str(stats_path),
        "num_series": len(rows),
        "concat_length": int(len(labels)),
        "positive_points": int(labels.sum()),
        "negative_points": int((labels == 0).sum()),
        "gap_between_series": int(args.gap),
        "score_alignment": "future-point scores aligned to absolute indices 8..end-1",
        "score_reduction": str(args.score_reduction),
        "sliding_window": int(args.sliding_window),
        "vus_version": str(args.vus_version),
        "vus_thresholds": int(args.vus_thresholds),
        "metrics": metrics,
        "per_series_preview": per_series_preview,
    }
    save_json(report, output_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
