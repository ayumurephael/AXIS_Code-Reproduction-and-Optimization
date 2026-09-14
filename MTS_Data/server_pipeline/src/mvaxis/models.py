from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .data_schema import ModelInput
from .features import channel_targets, channel_window_features, evidence_vector, metadata_vectors, root_target_index
from .utils import ensure_parent, sigmoid, softmax


@dataclass
class EncoderMetrics:
    loss: float
    channel_accuracy: float
    anomaly_accuracy: float
    root_hit_rate: float


@dataclass
class HintMetrics:
    loss: float
    root_accuracy: float
    abnormal_root_accuracy: float
    anomaly_accuracy: float


class NumpyMultivariateEncoder:
    """A tiny trainable channel scorer used for smoke tests."""

    def __init__(self, feature_dim: int, seed: int = 72):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0.0, 0.03, size=(feature_dim,))
        self.b = 0.0

    def predict_channel_logits(self, model_input: ModelInput) -> Tuple[np.ndarray, np.ndarray]:
        feats = channel_window_features(model_input)
        return feats @ self.w + self.b, feats

    def predict_channel_probs(self, model_input: ModelInput) -> Tuple[np.ndarray, np.ndarray]:
        logits, feats = self.predict_channel_logits(model_input)
        return sigmoid(logits), feats

    def train_epoch(self, inputs: List[ModelInput], lr: float, pos_weight: float) -> float:
        total = 0.0
        for mi in inputs:
            logits, feats = self.predict_channel_logits(mi)
            probs = sigmoid(logits)
            y = channel_targets(mi)
            weights = np.where(y > 0.5, pos_weight, 1.0)
            err = (probs - y) * weights
            grad_w = feats.T @ err / max(1, len(y))
            grad_b = float(np.mean(err))
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            loss = -weights * (y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8))
            total += float(np.mean(loss))
        return total / max(1, len(inputs))

    def evaluate(self, inputs: List[ModelInput]) -> EncoderMetrics:
        losses = []
        chan_correct = []
        anom_correct = []
        root_hits = []
        for mi in inputs:
            probs, _ = self.predict_channel_probs(mi)
            y = channel_targets(mi)
            weights = np.where(y > 0.5, 3.0, 1.0)
            losses.append(float(np.mean(-weights * (y * np.log(probs + 1e-8) + (1 - y) * np.log(1 - probs + 1e-8)))))
            chan_correct.extend(((probs > 0.5) == (y > 0.5)).astype(float).tolist())
            anom_correct.append(float((np.max(probs) > 0.5) == mi.is_anomalous))
            if mi.is_anomalous and np.sum(y) > 0:
                root_hits.append(float(y[int(np.argmax(probs))] > 0.5))
        return EncoderMetrics(
            loss=float(np.mean(losses)) if losses else 0.0,
            channel_accuracy=float(np.mean(chan_correct)) if chan_correct else 0.0,
            anomaly_accuracy=float(np.mean(anom_correct)) if anom_correct else 0.0,
            root_hit_rate=float(np.mean(root_hits)) if root_hits else 0.0,
        )

    def save(self, path: str) -> None:
        ensure_parent(path)
        np.savez(path, w=self.w, b=np.asarray([self.b]))

    @classmethod
    def load(cls, path: str) -> "NumpyMultivariateEncoder":
        data = np.load(path)
        obj = cls(feature_dim=len(data["w"]))
        obj.w = data["w"]
        obj.b = float(data["b"][0])
        return obj


class NumpyHintTuner:
    """A small trainable root-cause and anomaly hint scorer."""

    def __init__(self, input_dim: int, seed: int = 72):
        rng = np.random.default_rng(seed + 1000)
        self.root_w = rng.normal(0.0, 0.03, size=(input_dim,))
        self.root_b = 0.0
        self.null_logit = 0.0
        self.anom_w = rng.normal(0.0, 0.03, size=(5,))
        self.anom_b = 0.0

    def _channel_input(self, encoder: NumpyMultivariateEncoder, mi: ModelInput) -> Tuple[np.ndarray, np.ndarray]:
        probs, feats = encoder.predict_channel_probs(mi)
        meta = metadata_vectors(mi.channels)
        return np.concatenate([probs[:, None], feats, meta], axis=1), probs

    def _anom_input(self, probs: np.ndarray, mi: ModelInput) -> np.ndarray:
        ev = evidence_vector(mi.evidence_card)
        return np.asarray([float(np.max(probs)), float(np.mean(probs)), ev[1], ev[2], ev[3]], dtype=float)

    def predict(self, encoder: NumpyMultivariateEncoder, mi: ModelInput) -> Dict[str, Any]:
        x, probs = self._channel_input(encoder, mi)
        root_logits = x @ self.root_w + self.root_b
        all_logits = np.concatenate([root_logits, np.asarray([self.null_logit])])
        root_probs = softmax(all_logits)
        anom_x = self._anom_input(probs, mi)
        anom_prob = float(sigmoid(np.asarray([anom_x @ self.anom_w + self.anom_b]))[0])
        root_idx = int(np.argmax(root_probs))
        root_id = None if root_idx == len(mi.channels) else mi.channels[root_idx]["channel_id"]
        order = np.argsort(-root_probs[: len(mi.channels)])
        return {
            "is_anomalous_prob": anom_prob,
            "root_cause_channel": root_id,
            "root_probs": root_probs,
            "top_channels": [
                {
                    "channel_id": mi.channels[int(i)]["channel_id"],
                    "score": float(root_probs[int(i)]),
                    "encoder_prob": float(probs[int(i)]),
                }
                for i in order
            ],
        }

    def train_epoch(self, encoder: NumpyMultivariateEncoder, inputs: List[ModelInput], lr: float) -> float:
        total = 0.0
        for mi in inputs:
            x, probs = self._channel_input(encoder, mi)
            target = root_target_index(mi)
            root_logits = x @ self.root_w + self.root_b
            all_logits = np.concatenate([root_logits, np.asarray([self.null_logit])])
            p = softmax(all_logits)
            d = p.copy()
            d[target] -= 1.0
            self.root_w -= lr * (x.T @ d[: len(mi.channels)])
            self.root_b -= lr * float(np.sum(d[: len(mi.channels)]))
            self.null_logit -= lr * float(d[-1])
            root_loss = -float(np.log(p[target] + 1e-8))

            anom_x = self._anom_input(probs, mi)
            y = 1.0 if mi.is_anomalous else 0.0
            pa = float(sigmoid(np.asarray([anom_x @ self.anom_w + self.anom_b]))[0])
            err = pa - y
            self.anom_w -= lr * err * anom_x
            self.anom_b -= lr * err
            anom_loss = -(y * np.log(pa + 1e-8) + (1 - y) * np.log(1 - pa + 1e-8))
            total += root_loss + float(anom_loss)
        return total / max(1, len(inputs))

    def evaluate(self, encoder: NumpyMultivariateEncoder, inputs: List[ModelInput]) -> HintMetrics:
        losses = []
        root_ok = []
        ab_root_ok = []
        anom_ok = []
        for mi in inputs:
            pred = self.predict(encoder, mi)
            target = root_target_index(mi)
            root_probs = pred["root_probs"]
            root_idx = int(np.argmax(root_probs))
            losses.append(-float(np.log(root_probs[target] + 1e-8)))
            root_ok.append(float(root_idx == target))
            if mi.is_anomalous:
                ab_root_ok.append(float(root_idx == target))
            anom_ok.append(float((pred["is_anomalous_prob"] > 0.5) == mi.is_anomalous))
        return HintMetrics(
            loss=float(np.mean(losses)) if losses else 0.0,
            root_accuracy=float(np.mean(root_ok)) if root_ok else 0.0,
            abnormal_root_accuracy=float(np.mean(ab_root_ok)) if ab_root_ok else 0.0,
            anomaly_accuracy=float(np.mean(anom_ok)) if anom_ok else 0.0,
        )

    def make_soft_hint_summary(self, encoder: NumpyMultivariateEncoder, mi: ModelInput, top_k: int) -> Dict[str, Any]:
        pred = self.predict(encoder, mi)
        top = pred["top_channels"][:top_k]
        return {
            "global_hint": {
                "anomaly_probability": pred["is_anomalous_prob"],
                "predicted_root_cause_channel": pred["root_cause_channel"],
            },
            "channel_hints": top,
        }

    def save(self, path: str) -> None:
        ensure_parent(path)
        np.savez(
            path,
            root_w=self.root_w,
            root_b=np.asarray([self.root_b]),
            null_logit=np.asarray([self.null_logit]),
            anom_w=self.anom_w,
            anom_b=np.asarray([self.anom_b]),
        )

    @classmethod
    def load(cls, path: str) -> "NumpyHintTuner":
        data = np.load(path)
        obj = cls(input_dim=len(data["root_w"]))
        obj.root_w = data["root_w"]
        obj.root_b = float(data["root_b"][0])
        obj.null_logit = float(data["null_logit"][0])
        obj.anom_w = data["anom_w"]
        obj.anom_b = float(data["anom_b"][0])
        return obj

