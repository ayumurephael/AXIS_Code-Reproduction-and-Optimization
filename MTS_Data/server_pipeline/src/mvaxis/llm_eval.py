from __future__ import annotations

from typing import Any, Dict, List

from .data_schema import ModelInput


def _target_fact(model_input: ModelInput) -> Dict[str, Any]:
    return model_input.target_output.get("fact_check", {})


def score_llm_outputs(inputs: List[ModelInput], parsed_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    parseable = []
    anomaly_ok = []
    root_ok = []
    abnormal_root_ok = []
    false_positive = []
    false_negative = []
    evidence_grounded = []
    for mi, out in zip(inputs, parsed_outputs):
        if not out:
            parseable.append(0.0)
            anomaly_ok.append(0.0)
            root_ok.append(0.0)
            if mi.is_anomalous:
                abnormal_root_ok.append(0.0)
            false_positive.append(0.0)
            false_negative.append(0.0)
            evidence_grounded.append(0.0)
            continue
        parseable.append(1.0)
        fact = out.get("fact_check", {}) if isinstance(out, dict) else {}
        target = _target_fact(mi)
        pred_anom = bool(fact.get("is_anomalous", False))
        gold_anom = bool(target.get("is_anomalous", mi.is_anomalous))
        anomaly_ok.append(float(pred_anom == gold_anom))
        false_positive.append(float(pred_anom and not gold_anom))
        false_negative.append(float((not pred_anom) and gold_anom))

        pred_root = fact.get("root_cause_channel")
        gold_root = target.get("root_cause_channel", mi.root_cause_channel)
        root_match = pred_root == gold_root
        root_ok.append(float(root_match))
        if gold_anom:
            abnormal_root_ok.append(float(root_match))

        summary = " ".join([
            str(out.get("reasoning_summary", "")),
            str(out.get("final_answer", "")),
        ]).lower()
        evidence_terms = ["evidence", "hint", "channel", "relation", "peak", "trend", "gap", "oscillation"]
        evidence_grounded.append(float(any(term in summary for term in evidence_terms)))

    return {
        "json_parse_rate": sum(parseable) / max(1, len(parseable)),
        "llm_anomaly_accuracy": sum(anomaly_ok) / max(1, len(anomaly_ok)),
        "llm_root_accuracy": sum(root_ok) / max(1, len(root_ok)),
        "llm_abnormal_root_accuracy": sum(abnormal_root_ok) / max(1, len(abnormal_root_ok)),
        "llm_false_positive_rate": sum(false_positive) / max(1, len(false_positive)),
        "llm_false_negative_rate": sum(false_negative) / max(1, len(false_negative)),
        "evidence_term_rate": sum(evidence_grounded) / max(1, len(evidence_grounded)),
        "num_samples": len(parsed_outputs),
    }
