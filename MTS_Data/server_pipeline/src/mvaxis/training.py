from __future__ import annotations

from typing import Any, Dict, List

from .data_schema import ModelInput
from .models import NumpyHintTuner, NumpyMultivariateEncoder


def model_backend(config: Dict[str, Any]) -> str:
    return str(config.get("model", {}).get("backend", "numpy")).lower()


def create_encoder(config: Dict[str, Any]):
    if model_backend(config) in {"torch", "torch_axis", "axis"}:
        from .torch_models import TorchAXISEncoder

        return TorchAXISEncoder(config)
    return NumpyMultivariateEncoder(config["model"]["feature_dim"], seed=int(config.get("seed", 72)))


def create_hint_tuner(config: Dict[str, Any]):
    if model_backend(config) in {"torch", "torch_axis", "axis"}:
        from .torch_models import TorchHintTuner

        return TorchHintTuner(config)
    input_dim = 1 + config["model"]["feature_dim"] + config["model"]["metadata_dim"]
    return NumpyHintTuner(input_dim=input_dim, seed=int(config.get("seed", 72)))


def load_encoder(config: Dict[str, Any]):
    path = config["train_encoder"]["checkpoint_path"]
    if model_backend(config) in {"torch", "torch_axis", "axis"}:
        from .torch_models import TorchAXISEncoder

        return TorchAXISEncoder.load(path, config)
    return NumpyMultivariateEncoder.load(path)


def load_hint_tuner(config: Dict[str, Any]):
    path = config["train_hint_tuner"]["checkpoint_path"]
    if model_backend(config) in {"torch", "torch_axis", "axis"}:
        from .torch_models import TorchHintTuner

        return TorchHintTuner.load(path, config)
    return NumpyHintTuner.load(path)


def train_encoder(
    inputs_train: List[ModelInput],
    inputs_val: List[ModelInput],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    train_cfg = config["train_encoder"]
    model = create_encoder(config)
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        loss = model.train_epoch(inputs_train, float(train_cfg["learning_rate"]), float(train_cfg["pos_weight"]))
        tr = model.evaluate(inputs_train)
        va = model.evaluate(inputs_val)
        history.append({
            "epoch": epoch,
            "train_loss_step": loss,
            "train_loss": tr.loss,
            "train_channel_accuracy": tr.channel_accuracy,
            "train_anomaly_accuracy": tr.anomaly_accuracy,
            "train_root_hit_rate": tr.root_hit_rate,
            "val_loss": va.loss,
            "val_channel_accuracy": va.channel_accuracy,
            "val_anomaly_accuracy": va.anomaly_accuracy,
            "val_root_hit_rate": va.root_hit_rate,
        })
    model.save(train_cfg["checkpoint_path"])
    return {"model": model, "history": history, "val_metrics": history[-1] if history else {}}


def train_hint_tuner(
    encoder: NumpyMultivariateEncoder,
    inputs_train: List[ModelInput],
    inputs_val: List[ModelInput],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    train_cfg = config["train_hint_tuner"]
    model = create_hint_tuner(config)
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        loss = model.train_epoch(encoder, inputs_train, float(train_cfg["learning_rate"]))
        tr = model.evaluate(encoder, inputs_train)
        va = model.evaluate(encoder, inputs_val)
        history.append({
            "epoch": epoch,
            "train_loss_step": loss,
            "train_loss": tr.loss,
            "train_root_accuracy": tr.root_accuracy,
            "train_abnormal_root_accuracy": tr.abnormal_root_accuracy,
            "train_anomaly_accuracy": tr.anomaly_accuracy,
            "val_loss": va.loss,
            "val_root_accuracy": va.root_accuracy,
            "val_abnormal_root_accuracy": va.abnormal_root_accuracy,
            "val_anomaly_accuracy": va.anomaly_accuracy,
        })
    model.save(train_cfg["checkpoint_path"])
    return {"model": model, "history": history, "val_metrics": history[-1] if history else {}}
