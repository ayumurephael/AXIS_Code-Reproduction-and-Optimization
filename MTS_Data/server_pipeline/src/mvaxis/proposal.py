from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .data_schema import convert_to_model_input
from .evidence import recognize_evidence
from .prompts import normalize_json_output
from .answer_schema import normalize_answer_label, target_answer_label


def sample_with_interval(sample: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    proposed = copy.deepcopy(sample)
    proposed["target_interval"] = {"start": int(start), "end": int(end)}
    proposed["evidence_card"] = recognize_evidence(proposed)
    return proposed


def true_anomaly_interval(sample: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    labels = np.asarray(sample["series"]["labels"], dtype=int)
    any_label = labels.max(axis=1) > 0
    idx = np.flatnonzero(any_label)
    if len(idx) == 0:
        return None
    return int(idx[0]), int(idx[-1] + 1)


def true_affected_channels(sample: Dict[str, Any]) -> List[str]:
    labels = np.asarray(sample["series"]["labels"], dtype=int)
    channels = sample["channels"]
    affected = np.flatnonzero(labels.max(axis=0) > 0)
    return [channels[int(i)]["channel_id"] for i in affected]


def interval_iou(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / union) if union > 0 else 0.0


def interval_coverage(proposed: Optional[Tuple[int, int]], truth: Optional[Tuple[int, int]]) -> float:
    if truth is None:
        return 1.0 if proposed is None else 0.0
    if proposed is None:
        return 0.0
    inter = max(0, min(proposed[1], truth[1]) - max(proposed[0], truth[0]))
    return float(inter / max(1, truth[1] - truth[0]))


def affected_f1(predicted: List[str], truth: List[str]) -> float:
    p = set(predicted)
    t = set(truth)
    if not p and not t:
        return 1.0
    if not p or not t:
        return 0.0
    tp = len(p & t)
    precision = tp / len(p)
    recall = tp / len(t)
    return float(2 * precision * recall / max(1e-8, precision + recall))


def _answer_label_from_target(target_output: Dict[str, Any]) -> Optional[str]:
    label = target_answer_label(target_output)
    return str(label).strip().lower() if label is not None else None


def _answer_label_from_prediction(pred_fact: Dict[str, Any], parsed_output: Dict[str, Any]) -> Optional[str]:
    label = pred_fact.get("answer_label") or pred_fact.get("choice_answer")
    if label is not None:
        normalized = normalize_answer_label(label)
        return str(normalized).strip().lower() if normalized is not None else None
    normalized = normalize_answer_label(parsed_output.get("question_answer") or parsed_output.get("final_answer"))
    return str(normalized).strip().lower() if normalized is not None else None


def propose_interval(
    sample: Dict[str, Any],
    encoder: Any,
    hint_tuner: Any,
    window_size: int,
    stride: int,
    top_k_channels: int,
) -> Dict[str, Any]:
    if hasattr(encoder, "propose_interval"):
        return encoder.propose_interval(sample, window_size, stride, top_k_channels)
    length = int(sample["series"]["shape"][0])
    candidates: List[Dict[str, Any]] = []
    temporal_probs = None
    if hasattr(encoder, "predict_temporal_probs"):
        full_sample = sample_with_interval(sample, 0, length)
        temporal_probs = encoder.predict_temporal_probs(convert_to_model_input(full_sample))
        time_score = np.max(temporal_probs, axis=1)
    for start in range(0, max(1, length - window_size + 1), stride):
        end = min(length, start + window_size)
        if end <= start:
            continue
        proposed_sample = sample_with_interval(sample, start, end)
        mi = convert_to_model_input(proposed_sample)
        if temporal_probs is not None:
            window_temporal = temporal_probs[start:end]
            encoder_prob = float(np.max(window_temporal))
            temporal_window_score = float(np.mean(time_score[start:end]))
            score = 0.75 * temporal_window_score + 0.25 * encoder_prob
            hints = {"global_hint": {"anomaly_probability": score, "predicted_root_cause_channel": None}, "channel_hints": []}
            if np.max(window_temporal) > 0:
                channel_order = np.argsort(-window_temporal.max(axis=0))
                hints["channel_hints"] = [
                    {
                        "channel_id": sample["channels"][int(i)]["channel_id"],
                        "score": float(window_temporal[:, int(i)].mean()),
                        "encoder_prob": float(window_temporal[:, int(i)].max()),
                    }
                    for i in channel_order[:top_k_channels]
                ]
                hints["global_hint"]["predicted_root_cause_channel"] = hints["channel_hints"][0]["channel_id"]
            hint_prob = score
        else:
            hints = hint_tuner.make_soft_hint_summary(encoder, mi, top_k_channels)
            enc_probs, _ = encoder.predict_channel_probs(mi)
            hint_prob = float(hints["global_hint"]["anomaly_probability"])
            encoder_prob = float(np.max(enc_probs))
            score = 0.5 * hint_prob + 0.5 * encoder_prob
        candidates.append(
            {
                "start": int(start),
                "end": int(end),
                "proposal_score": score,
                "hint_anomaly_probability": hint_prob,
                "encoder_max_channel_probability": encoder_prob,
                "predicted_root_cause_channel": hints["global_hint"]["predicted_root_cause_channel"],
                "channel_hints": hints["channel_hints"],
            }
        )
    if not candidates:
        return {
            "start": None,
            "end": None,
            "proposal_score": 0.0,
            "hint_anomaly_probability": 0.0,
            "encoder_max_channel_probability": 0.0,
            "predicted_root_cause_channel": None,
            "channel_hints": [],
            "ranked_candidates": [],
        }
    candidates.sort(key=lambda x: x["proposal_score"], reverse=True)
    best = dict(candidates[0])
    best["ranked_candidates"] = candidates[:5]
    return best


def score_explanation_against_truth(
    sample: Dict[str, Any],
    proposal: Dict[str, Any],
    parsed_output: Dict[str, Any],
    min_iou: float = 0.25,
) -> Dict[str, Any]:
    truth_interval = true_anomaly_interval(sample)
    proposal_interval = None
    if proposal.get("start") is not None and proposal.get("end") is not None:
        proposal_interval = (int(proposal["start"]), int(proposal["end"]))
    truth_fact = sample["target_output"]["fact_check"]
    parsed_output = normalize_json_output(parsed_output) if parsed_output else {}
    pred_fact = parsed_output.get("fact_check", {}) if parsed_output else {}
    truth_anom = bool(truth_fact.get("is_anomalous", truth_interval is not None))
    pred_anom = bool(pred_fact.get("is_anomalous", False)) if parsed_output else False
    truth_root = truth_fact.get("root_cause_channel")
    pred_root = pred_fact.get("root_cause_channel")
    truth_affected = true_affected_channels(sample)
    pred_affected = list(pred_fact.get("affected_channels") or [])
    truth_answer_label = _answer_label_from_target(sample.get("target_output") or {})
    pred_answer_label = _answer_label_from_prediction(pred_fact, parsed_output) if parsed_output else None
    iou = interval_iou(proposal_interval, truth_interval)
    coverage = interval_coverage(proposal_interval, truth_interval)
    return {
        "truth_interval": list(truth_interval) if truth_interval else None,
        "proposal_interval": list(proposal_interval) if proposal_interval else None,
        "proposal_temporal_iou": iou,
        "proposal_truth_coverage": coverage,
        "proposal_is_close": bool((not truth_anom and not pred_anom) or (truth_anom and iou >= min_iou)),
        "llm_anomaly_match": bool(pred_anom == truth_anom),
        "llm_root_match": bool(pred_root == truth_root),
        "llm_affected_f1": affected_f1(pred_affected, truth_affected),
        "llm_type_match": bool(pred_fact.get("anomaly_type") == truth_fact.get("anomaly_type")),
        "llm_answer_label_match": (
            None
            if truth_answer_label is None
            else bool((pred_answer_label or "").lower() == str(truth_answer_label).lower())
        ),
        "truth_root_cause_channel": truth_root,
        "pred_root_cause_channel": pred_root,
        "truth_affected_channels": truth_affected,
        "pred_affected_channels": pred_affected,
        "truth_anomaly_type": truth_fact.get("anomaly_type"),
        "pred_anomaly_type": pred_fact.get("anomaly_type"),
        "truth_answer_label": truth_answer_label,
        "pred_answer_label": pred_answer_label,
    }
