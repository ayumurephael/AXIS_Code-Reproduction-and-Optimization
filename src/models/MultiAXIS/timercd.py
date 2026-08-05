from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from experiments.configs.axis_config import AXISConfig, TimeSeriesConfig
from src.models.AXIS.Pretrain_ts_encoder import TimeSeriesPretrainModel

from .config import TimeRCDConfig


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenTimeRCD(nn.Module):
    """Checkpoint-compatible frozen TimeRCD encoder and per-(t,c) anomaly head."""

    def __init__(self, config: TimeRCDConfig):
        super().__init__()
        axis_config = AXISConfig(
            ts_config=TimeSeriesConfig(
                d_model=config.d_model,
                d_proj=config.d_proj,
                patch_size=config.patch_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                d_ff_dropout=config.dropout,
                use_rope=True,
                activation=config.activation,
                num_features=config.checkpoint_num_features,
            ),
            dropout=config.dropout,
        )
        self.config = config
        self.backbone = TimeSeriesPretrainModel(axis_config)
        self.checkpoint_sha256: str | None = None
        self.checkpoint_metadata: Dict[str, Any] = {}
        self.freeze()

    @property
    def d_proj(self) -> int:
        return self.config.d_proj

    def freeze(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def load_checkpoint(self, checkpoint_path: str | Path, strict: bool = True) -> Dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("TimeRCD checkpoint must be a dictionary")
        state = payload.get("model_state_dict", payload.get("state_dict", payload))
        if not isinstance(state, dict):
            raise TypeError("TimeRCD checkpoint does not contain a state dictionary")
        incompatible = self.backbone.load_state_dict(state, strict=strict)
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            raise RuntimeError(
                f"Strict TimeRCD load failed: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        self.checkpoint_sha256 = sha256_file(checkpoint_path)
        self.checkpoint_metadata = {
            "epoch": payload.get("epoch"),
            "loss": payload.get("loss"),
            "config_type": type(payload.get("config")).__name__,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        self.freeze()
        return dict(self.checkpoint_metadata)

    def forward(
        self,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if normalized_series.ndim != 3:
            raise ValueError("normalized_series must have shape [B,T,C]")
        batch, steps, channels = normalized_series.shape
        if time_mask.shape != (batch, steps) or channel_mask.shape != (batch, channels):
            raise ValueError("Time/channel masks do not match normalized_series")
        self.backbone.eval()
        with torch.no_grad():
            local = self.backbone(
                normalized_series,
                mask=time_mask,
                channel_mask=channel_mask,
            )
            logits = self.backbone.anomaly_head(local)
        if local.shape != (batch, steps, channels, self.config.d_proj):
            raise AssertionError(f"Unexpected TimeRCD representation shape: {tuple(local.shape)}")
        if logits.shape != (batch, steps, channels, 2):
            raise AssertionError(f"Unexpected TimeRCD anomaly-logit shape: {tuple(logits.shape)}")
        return local, logits
