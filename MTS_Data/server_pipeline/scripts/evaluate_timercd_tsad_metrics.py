from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()
TIMercd_ROOT = ROOT / "references" / "TimeRCD"
if str(TIMercd_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMercd_ROOT))

from evaluation.basic_metrics import basic_metricor, generate_curve  # noqa: E402
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer  # noqa: E402
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json  # noqa: E402


def _as_float_dict(payload: Dict[str, Any]) -> Dict[str, float]:
    return {key: float(value) for key, value in payload.items()}


def _time_label_and_score(
    model: AXISMultivariateIntervalProposer,
    row: Dict[str, Any],
    score_reduction: str,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.anomaly_probabilities(row)
    labels = np.asarray(row["series"]["labels"], dtype=np.int64)
    length = int(row["series"]["shape"][0])
    if score_reduction == "max":
        time_score = probs[:length].max(axis=1).astype(np.float64)
    elif score_reduction == "mean":
        time_score = probs[:length].mean(axis=1).astype(np.float64)
    else:
        raise ValueError(f"Unsupported score_reduction: {score_reduction}")
    time_label = (labels[:length].max(axis=1) > 0).astype(np.int64)
    return time_label, time_score


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


def _compute_reference_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    sliding_window: int,
    vus_version: str,
    vus_thresholds: int,
    workers: int,
    chunk_size: int,
) -> Dict[str, Any]:
    grader = basic_metricor()
    result: Dict[str, Any] = {}

    print("Computing Standard-F1...")
    try:
        standard = grader.metric_standard_F1_chunked(
            labels,
            scores,
            chunk_size=max(1, int(chunk_size)),
            num_workers=max(1, int(workers)),
        )
    except Exception as exc:
        print(f"Chunked Standard-F1 failed ({exc}); falling back to reference sequential implementation.")
        standard = grader.metric_standard_F1(labels, scores)
    result["Standard-F1"] = float(standard["F1"])
    result["Standard-Precision"] = float(standard["Precision"])
    result["Standard-Recall"] = float(standard["Recall"])

    print("Computing F1-T...")
    f1_t = grader.metric_F1_T(
        labels,
        scores,
        use_parallel=True,
        parallel_method="chunked",
        chunk_size=max(1, int(chunk_size)),
        max_workers=max(1, int(workers)),
    )
    result["F1-T"] = float(f1_t["F1_T"])
    result["Precision-T"] = float(f1_t["P_T"])
    result["Recall-T"] = float(f1_t["R_T"])
    result["Threshold-T"] = float(f1_t["thre_T"])

    print("Computing Affiliation-F...")
    try:
        aff_f, aff_p, aff_r = grader.metric_Affiliation_chunked(
            labels,
            scores,
            chunk_size=max(1, int(chunk_size)),
            num_workers=max(1, int(workers)),
        )
    except Exception as exc:
        print(f"Chunked Affiliation-F failed ({exc}); falling back to reference sequential implementation.")
        aff_f, aff_p, aff_r = grader.metric_Affiliation(labels, scores)
    result["Affiliation-F"] = float(aff_f)
    result["Affiliation-Precision"] = float(aff_p)
    result["Affiliation-Recall"] = float(aff_r)

    print("Computing VUS-PR...")
    try:
        _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(
            labels.astype(int),
            scores.astype(float),
            int(sliding_window),
            vus_version,
            thre=int(vus_thresholds),
        )
    except Exception as exc:
        print(f"VUS computation failed ({exc}); returning 0 for VUS metrics.")
        vus_roc, vus_pr = 0.0, 0.0
    result["VUS-PR"] = float(vus_pr)
    result["VUS-ROC"] = float(vus_roc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale_nocf_schema.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gap", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sliding-window", type=int, default=100)
    parser.add_argument("--vus-version", default="opt")
    parser.add_argument("--vus-thresholds", type=int, default=250)
    parser.add_argument("--score-reduction", choices=["max", "mean"], default="max")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--score-cache", default=None)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    checkpoint_path = relative_to_root(ROOT, args.checkpoint)
    data_path = Path(relative_to_root(ROOT, args.data_file))
    rows = read_jsonl(data_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]

    print(f"Loading TimeRCD checkpoint: {checkpoint_path}")
    model, load_info = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
        checkpoint_path,
        config,
        threshold=float(args.threshold),
    )

    labels_by_series: List[np.ndarray] = []
    scores_by_series: List[np.ndarray] = []
    per_series: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index % 50 == 0:
            print(f"Scoring sample {index}/{len(rows)}")
        label, score = _time_label_and_score(model, row, str(args.score_reduction))
        labels_by_series.append(label)
        scores_by_series.append(score)
        per_series.append(
            {
                "index": index,
                "sample_id": row.get("sample_id"),
                "length": int(len(label)),
                "positive_points": int(label.sum()),
                "max_score": float(score.max()) if len(score) else 0.0,
                "mean_score": float(score.mean()) if len(score) else 0.0,
            }
        )

    labels, scores = _concatenate_with_gap(labels_by_series, scores_by_series, int(args.gap))
    if args.score_cache:
        cache_path = ROOT / args.score_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            labels=labels,
            scores=scores,
            gap=np.asarray([int(args.gap)], dtype=np.int64),
        )

    metrics = _compute_reference_metrics(
        labels,
        scores,
        sliding_window=int(args.sliding_window),
        vus_version=str(args.vus_version),
        vus_thresholds=int(args.vus_thresholds),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
    )
    report = {
        "data_file": str(data_path),
        "checkpoint": checkpoint_path,
        "load_info": load_info,
        "num_series": len(rows),
        "concat_length": int(len(labels)),
        "positive_points": int(labels.sum()),
        "negative_points": int((labels == 0).sum()),
        "gap_between_series": int(args.gap),
        "score_reduction": f"{args.score_reduction} over channels per time step",
        "label_reduction": "any anomalous channel per time step",
        "sliding_window": int(args.sliding_window),
        "vus_version": str(args.vus_version),
        "vus_thresholds": int(args.vus_thresholds),
        "metrics": metrics,
        "per_series_preview": per_series[:10],
    }
    save_json(report, ROOT / args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
