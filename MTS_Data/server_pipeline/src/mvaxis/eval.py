from __future__ import annotations

from typing import Any, Dict, List

from .data_schema import ModelInput
from .models import NumpyHintTuner, NumpyMultivariateEncoder


def evaluate_closed_loop(
    encoder: NumpyMultivariateEncoder,
    hint_tuner: NumpyHintTuner,
    inputs: List[ModelInput],
) -> Dict[str, Any]:
    enc = encoder.evaluate(inputs)
    hint = hint_tuner.evaluate(encoder, inputs)
    hallucinated = []
    for mi in inputs:
        pred = hint_tuner.predict(encoder, mi)
        hallucinated.append(float((pred["is_anomalous_prob"] > 0.5) and not mi.is_anomalous))
    return {
        "encoder": {
            "loss": enc.loss,
            "channel_accuracy": enc.channel_accuracy,
            "anomaly_accuracy": enc.anomaly_accuracy,
            "root_hit_rate": enc.root_hit_rate,
        },
        "hint_tuner": {
            "loss": hint.loss,
            "root_accuracy": hint.root_accuracy,
            "abnormal_root_accuracy": hint.abnormal_root_accuracy,
            "anomaly_accuracy": hint.anomaly_accuracy,
        },
        "hallucinated_anomaly_rate": sum(hallucinated) / max(1, len(hallucinated)),
        "num_samples": len(inputs),
    }

