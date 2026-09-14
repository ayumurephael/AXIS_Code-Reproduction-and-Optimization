from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import (
    AXISMultivariateIntervalProposer,
    evaluate_interval_proposer,
    tune_threshold,
)
from src.mvaxis.proposal import (
    affected_f1,
    interval_coverage,
    interval_iou,
    true_affected_channels,
    true_anomaly_interval,
)
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def _metrics_payload(metrics: Any) -> Dict[str, Any]:
    return dict(metrics.__dict__)


def _slim_channel_hints(proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "channel_id": item.get("channel_id"),
            "score": item.get("score"),
            "encoder_prob": item.get("encoder_prob"),
        }
        for item in proposal.get("channel_hints", [])
    ]


def _proposal_records(
    model: AXISMultivariateIntervalProposer,
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    proposal_cfg = config.get("interval_proposal", {})
    window_size = int(proposal_cfg.get("window_size", 32))
    stride = int(proposal_cfg.get("stride", 8))
    min_iou = float(proposal_cfg.get("min_iou", 0.25))
    records: List[Dict[str, Any]] = []
    truth_anom_count = 0
    pred_anom_count = 0
    false_positive = 0
    false_negative = 0
    non_overlap = 0
    for index, row in enumerate(rows):
        probs = model.anomaly_probabilities(row)
        proposal = model.proposal_from_probs(row, probs, window_size, stride, threshold=threshold)
        truth_interval = true_anomaly_interval(row)
        proposal_interval: Optional[Tuple[int, int]] = (
            None if proposal.get("start") is None else (int(proposal["start"]), int(proposal["end"]))
        )
        truth_anom = truth_interval is not None
        pred_anom = bool(proposal.get("is_anomalous_proposal", False))
        truth_anom_count += int(truth_anom)
        pred_anom_count += int(pred_anom)
        false_positive += int((not truth_anom) and pred_anom)
        false_negative += int(truth_anom and (not pred_anom))
        iou = interval_iou(proposal_interval, truth_interval)
        coverage = interval_coverage(proposal_interval, truth_interval)
        if truth_anom and proposal_interval is not None and iou <= 0.0:
            non_overlap += 1
        records.append(
            {
                "index": index,
                "sample_id": row.get("sample_id"),
                "truth_is_anomalous": truth_anom,
                "pred_is_anomalous": pred_anom,
                "truth_interval": list(truth_interval) if truth_interval is not None else None,
                "proposal_interval": list(proposal_interval) if proposal_interval is not None else None,
                "proposal_score": proposal.get("proposal_score"),
                "max_point_score": proposal.get("max_point_score"),
                "temporal_iou": iou,
                "truth_coverage": coverage,
                "is_close": bool((not truth_anom and not pred_anom) or (truth_anom and iou >= min_iou)),
                "truth_root_cause_channel": (row.get("target_output", {}).get("fact_check") or {}).get("root_cause_channel"),
                "predicted_root_cause_channel": proposal.get("predicted_root_cause_channel"),
                "truth_affected_channels": true_affected_channels(row),
                "predicted_channel_hints": _slim_channel_hints(proposal),
                "affected_f1": affected_f1(
                    [x.get("channel_id") for x in proposal.get("channel_hints", []) if x.get("channel_id")],
                    true_affected_channels(row),
                ),
                "top_ranked_candidates": [
                    {
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "proposal_score": item.get("proposal_score"),
                        "max_point_score": item.get("max_point_score"),
                        "predicted_root_cause_channel": item.get("predicted_root_cause_channel"),
                    }
                    for item in proposal.get("ranked_candidates", [])[:5]
                ],
                "max_anomaly_probability": float(np.max(probs)) if probs.size else 0.0,
                "mean_anomaly_probability": float(np.mean(probs)) if probs.size else 0.0,
            }
        )
    summary = {
        "num_samples": len(rows),
        "truth_anomalous_count": truth_anom_count,
        "truth_normal_count": len(rows) - truth_anom_count,
        "pred_anomalous_count": pred_anom_count,
        "pred_normal_count": len(rows) - pred_anom_count,
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "truth_anomaly_non_overlap_count": non_overlap,
        "truth_anomaly_non_overlap_rate": non_overlap / max(1, truth_anom_count),
        "window_size": window_size,
        "stride": stride,
        "threshold": float(threshold),
    }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_relation_stress_v1.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--tune-split", default="val")
    parser.add_argument("--report", default="outputs/runs/relation_stress_v1/timercd_interval_eval.json")
    parser.add_argument("--details-jsonl", default=None)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    checkpoint_path = relative_to_root(ROOT, args.checkpoint)
    model, load_info = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
        checkpoint_path,
        config,
        threshold=float(args.threshold),
    )
    if args.data_file:
        data_path = Path(relative_to_root(ROOT, args.data_file))
    else:
        data_path = Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{args.split}.jsonl"
    rows = read_jsonl(data_path)
    raw_metrics = evaluate_interval_proposer(model, rows, config, threshold=float(args.threshold))
    records, detection_summary = _proposal_records(model, rows, config, threshold=float(args.threshold))
    payload: Dict[str, Any] = {
        "split": args.split,
        "data_path": str(data_path),
        "checkpoint_path": checkpoint_path,
        "load_info": load_info,
        "raw_threshold": float(args.threshold),
        "raw_metrics": _metrics_payload(raw_metrics),
        "detection_summary": detection_summary,
    }
    if args.details_jsonl:
        details_path = ROOT / args.details_jsonl
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        payload["details_jsonl"] = str(details_path)

    if args.tune_threshold:
        tune_rows = read_jsonl(Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{args.tune_split}.jsonl")
        tuned_threshold = tune_threshold(model, tune_rows, config)
        model.threshold = tuned_threshold
        tuned_metrics = evaluate_interval_proposer(model, rows, config, threshold=tuned_threshold)
        payload["tune_split"] = args.tune_split
        payload["tuned_threshold"] = float(tuned_threshold)
        payload["tuned_metrics"] = _metrics_payload(tuned_metrics)

    save_json(payload, ROOT / args.report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
