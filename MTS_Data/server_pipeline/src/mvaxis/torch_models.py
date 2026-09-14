from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_schema import ModelInput
from .features import channel_targets, channel_window_features, metadata_vectors, root_target_index
from .models import EncoderMetrics, HintMetrics
from .utils import ensure_parent


def _resolve_device(config: Dict[str, Any]) -> torch.device:
    requested = str(config.get("model", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_torch_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_values(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-6
    return (values - mean) / std


def _batch_iter(inputs: List[ModelInput], batch_size: int, shuffle: bool) -> List[List[ModelInput]]:
    indices = list(range(len(inputs)))
    if shuffle:
        random.shuffle(indices)
    ordered = [inputs[i] for i in indices]
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def _collate_model_inputs(batch: List[ModelInput], device: torch.device) -> Dict[str, Any]:
    max_len = max(int(mi.values.shape[0]) for mi in batch)
    num_channels = int(batch[0].values.shape[1])
    values = np.zeros((len(batch), max_len, num_channels), dtype=np.float32)
    mask = np.zeros((len(batch), max_len), dtype=np.float32)
    targets = np.zeros((len(batch), num_channels), dtype=np.float32)
    full_labels = np.zeros((len(batch), max_len, num_channels), dtype=np.float32)
    channel_feats = np.zeros((len(batch), num_channels, channel_window_features(batch[0]).shape[1]), dtype=np.float32)
    meta_feats = np.zeros((len(batch), num_channels, metadata_vectors(batch[0].channels).shape[1]), dtype=np.float32)
    starts: List[int] = []
    ends: List[int] = []
    root_targets: List[int] = []
    is_anom: List[float] = []
    for i, mi in enumerate(batch):
        length = int(mi.values.shape[0])
        values[i, :length] = _normalize_values(mi.values).astype(np.float32)
        mask[i, :length] = 1.0
        full_labels[i, :length] = mi.labels.astype(np.float32)
        targets[i] = channel_targets(mi).astype(np.float32)
        channel_feats[i] = channel_window_features(mi).astype(np.float32)
        meta_feats[i] = metadata_vectors(mi.channels).astype(np.float32)
        starts.append(int(mi.interval[0]))
        ends.append(int(mi.interval[1]))
        root_targets.append(int(root_target_index(mi)))
        is_anom.append(1.0 if mi.is_anomalous else 0.0)
    return {
        "values": torch.from_numpy(values).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "targets": torch.from_numpy(targets).to(device),
        "full_labels": torch.from_numpy(full_labels).to(device),
        "channel_feats": torch.from_numpy(channel_feats).to(device),
        "meta_feats": torch.from_numpy(meta_feats).to(device),
        "starts": starts,
        "ends": ends,
        "root_targets": torch.tensor(root_targets, dtype=torch.long, device=device),
        "is_anom": torch.tensor(is_anom, dtype=torch.float32, device=device),
    }


class _RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return (self.scale * x * torch.rsqrt(norm + self.eps)).type_as(x)


class _RotaryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int) -> torch.Tensor:
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        return torch.einsum("i,j->ij", t, self.inv_freq)


class _BinaryAttentionBias(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(2, num_heads)

    def forward(self, query_id: torch.Tensor, kv_id: torch.Tensor) -> torch.Tensor:
        same_channel = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2)).long()
        return self.embedding(same_channel).permute(0, 3, 1, 2)


class _MultiheadAttentionWithRoPE(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_features: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.binary_attention_bias = _BinaryAttentionBias(num_heads) if num_features > 1 else None

    def _apply_rope(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        num_patches = seq_len // self.num_features
        x = x.reshape(batch_size * self.num_features, num_patches, embed_dim)
        paired = x.view(batch_size * self.num_features, num_patches, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)
        sin = freqs.sin().unsqueeze(0)
        rotated = torch.stack(
            [
                paired[..., 0] * cos - paired[..., 1] * sin,
                paired[..., 0] * sin + paired[..., 1] * cos,
            ],
            dim=-1,
        )
        return rotated.view(batch_size, seq_len, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        feature_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self._apply_rope(self.q_proj(x), freqs)
        k = self._apply_rope(self.k_proj(x), freqs)
        v = self.v_proj(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.binary_attention_bias is not None:
            scores = scores + self.binary_attention_bias(feature_ids, feature_ids)
        scores = scores.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(y)


class _AxisEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_features: int, dropout: float) -> None:
        super().__init__()
        self.input_norm = _RMSNorm(d_model)
        self.output_norm = _RMSNorm(d_model)
        self.attn = _MultiheadAttentionWithRoPE(d_model, num_heads, num_features)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, feature_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.input_norm(x), freqs, feature_ids, attn_mask))
        x = x + self.dropout(self.mlp(self.output_norm(x)))
        return x


class _AxisEncoderStack(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_features: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            _AxisEncoderLayer(d_model, num_heads, num_features, dropout) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, feature_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, freqs, feature_ids, attn_mask)
        return x


class AXISStyleTimeSeriesEncoder(nn.Module):
    """Patch + RoPE + binary variate attention encoder with AXIS-compatible I/O."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        d_proj: int = 32,
        patch_size: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_patches: int = 64,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.patch_size = patch_size
        self.d_proj = d_proj
        self.max_patches = max_patches
        self.patch_embedding = nn.Linear(patch_size, d_model)
        self.rope = _RotaryEmbedding(d_model)
        self.encoder = _AxisEncoderStack(d_model, num_heads, num_features, num_layers, dropout)
        self.projection = nn.Linear(d_model, patch_size * d_proj)

    def forward(self, time_series: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)
        batch_size, seq_len, num_features = time_series.shape
        if num_features != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {num_features}")
        padded_len = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_len > seq_len:
            pad_len = padded_len - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_len), value=0.0)
            mask = F.pad(mask, (0, pad_len), value=0.0)
        num_patches = padded_len // self.patch_size
        if num_patches > self.max_patches:
            raise ValueError("Sequence is longer than max_patches * patch_size")

        patches = time_series.view(batch_size, num_patches, self.patch_size, num_features)
        patches = patches.permute(0, 3, 1, 2).contiguous()
        patches = patches.view(batch_size, num_features * num_patches, self.patch_size)

        tokens = self.patch_embedding(patches)
        feature_ids = torch.arange(num_features, device=time_series.device).repeat_interleave(num_patches)
        feature_ids = feature_ids.unsqueeze(0).expand(batch_size, -1)

        patch_mask = mask.view(batch_size, num_patches, self.patch_size).sum(dim=-1) > 0
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1).reshape(batch_size, num_features * num_patches)
        encoded = self.encoder(tokens, self.rope(num_patches).to(time_series.device), feature_ids, full_mask)
        projected = self.projection(encoded)
        local = projected.view(batch_size, num_features, num_patches, self.patch_size, self.d_proj)
        local = local.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, padded_len, num_features, self.d_proj)
        return local[:, :seq_len]


class TorchAXISEncoder:
    def __init__(self, config: Dict[str, Any]):
        _set_torch_seed(int(config.get("seed", 72)))
        self.config = config
        self.device = _resolve_device(config)
        model_cfg = config["model"]
        ts_cfg = model_cfg.get("ts_encoder", {})
        self.d_proj = int(ts_cfg.get("d_proj", 32))
        self.num_channels = int(config["data"]["num_channels"])
        self.feature_dim = int(model_cfg["feature_dim"])
        self.batch_size = int(model_cfg.get("batch_size", 16))
        self.encoder = AXISStyleTimeSeriesEncoder(
            num_features=self.num_channels,
            d_model=int(ts_cfg.get("d_model", 64)),
            d_proj=self.d_proj,
            patch_size=int(ts_cfg.get("patch_size", 16)),
            num_layers=int(ts_cfg.get("num_layers", 2)),
            num_heads=int(ts_cfg.get("num_heads", 4)),
            dropout=float(ts_cfg.get("dropout", 0.1)),
            max_patches=int(ts_cfg.get("max_patches", 64)),
        ).to(self.device)
        self.channel_head = nn.Sequential(
            nn.Linear(self.d_proj * 2 + self.feature_dim, self.d_proj),
            nn.GELU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.1))),
            nn.Linear(self.d_proj, 1),
        ).to(self.device)
        self.temporal_head = nn.Sequential(
            nn.Linear(self.d_proj, self.d_proj // 2),
            nn.GELU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.1))),
            nn.Linear(self.d_proj // 2, 1),
        ).to(self.device)
        self._optimizer: torch.optim.Optimizer | None = None

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.channel_head.parameters()) + list(self.temporal_head.parameters())

    def _ensure_optimizer(self, lr: float) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        return self._optimizer

    def _channel_representations(self, batch_data: Dict[str, Any]) -> torch.Tensor:
        local = self.encoder(batch_data["values"], batch_data["mask"])
        reps = []
        for b, (start, end) in enumerate(zip(batch_data["starts"], batch_data["ends"])):
            window = local[b, start:end]
            mean = window.mean(dim=0)
            std = window.std(dim=0, unbiased=False)
            reps.append(torch.cat([mean, std, batch_data["channel_feats"][b]], dim=-1))
        return torch.stack(reps, dim=0)

    def _local_and_channel_representations(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        local = self.encoder(batch_data["values"], batch_data["mask"])
        reps = []
        for b, (start, end) in enumerate(zip(batch_data["starts"], batch_data["ends"])):
            window = local[b, start:end]
            mean = window.mean(dim=0)
            std = window.std(dim=0, unbiased=False)
            reps.append(torch.cat([mean, std, batch_data["channel_feats"][b]], dim=-1))
        return local, torch.stack(reps, dim=0)

    def predict_channel_logits_batch(self, batch: List[ModelInput]) -> torch.Tensor:
        data = _collate_model_inputs(batch, self.device)
        reps = self._channel_representations(data)
        return self.channel_head(reps).squeeze(-1)

    def predict_channel_probs(self, model_input: ModelInput) -> Tuple[np.ndarray, np.ndarray]:
        self.encoder.eval()
        self.channel_head.eval()
        with torch.no_grad():
            logits = self.predict_channel_logits_batch([model_input])
            probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
        return probs, channel_window_features(model_input)

    def predict_temporal_probs(self, model_input: ModelInput) -> np.ndarray:
        self.encoder.eval()
        self.temporal_head.eval()
        with torch.no_grad():
            data = _collate_model_inputs([model_input], self.device)
            local = self.encoder(data["values"], data["mask"])
            probs = torch.sigmoid(self.temporal_head(local).squeeze(-1))[0]
        return probs.detach().cpu().numpy()

    def train_epoch(self, inputs: List[ModelInput], lr: float, pos_weight: float) -> float:
        self.encoder.train()
        self.channel_head.train()
        opt = self._ensure_optimizer(lr)
        losses: List[float] = []
        for batch in _batch_iter(inputs, self.batch_size, shuffle=True):
            data = _collate_model_inputs(batch, self.device)
            local, reps = self._local_and_channel_representations(data)
            logits = self.channel_head(reps).squeeze(-1)
            channel_loss = F.binary_cross_entropy_with_logits(
                logits,
                data["targets"],
                pos_weight=torch.tensor(float(pos_weight), device=self.device),
            )
            temporal_logits = self.temporal_head(local).squeeze(-1)
            mask = data["mask"].unsqueeze(-1).expand_as(data["full_labels"]) > 0
            temporal_loss = F.binary_cross_entropy_with_logits(
                temporal_logits[mask],
                data["full_labels"][mask],
                pos_weight=torch.tensor(float(pos_weight), device=self.device),
            )
            loss = channel_loss + float(self.config["model"].get("temporal_loss_weight", 1.0)) * temporal_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def evaluate(self, inputs: List[ModelInput]) -> EncoderMetrics:
        self.encoder.eval()
        self.channel_head.eval()
        self.temporal_head.eval()
        losses: List[float] = []
        chan_correct: List[float] = []
        anom_correct: List[float] = []
        root_hits: List[float] = []
        with torch.no_grad():
            for batch in _batch_iter(inputs, self.batch_size, shuffle=False):
                data = _collate_model_inputs(batch, self.device)
                logits = self.channel_head(self._channel_representations(data)).squeeze(-1)
                probs = torch.sigmoid(logits)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    data["targets"],
                    pos_weight=torch.tensor(3.0, device=self.device),
                )
                losses.append(float(loss.detach().cpu()))
                y_np = data["targets"].detach().cpu().numpy()
                p_np = probs.detach().cpu().numpy()
                chan_correct.extend(((p_np > 0.5) == (y_np > 0.5)).astype(float).ravel().tolist())
                for i, mi in enumerate(batch):
                    anom_correct.append(float((np.max(p_np[i]) > 0.5) == mi.is_anomalous))
                    if mi.is_anomalous and np.sum(y_np[i]) > 0:
                        root_hits.append(float(y_np[i, int(np.argmax(p_np[i]))] > 0.5))
        return EncoderMetrics(
            loss=float(np.mean(losses)) if losses else 0.0,
            channel_accuracy=float(np.mean(chan_correct)) if chan_correct else 0.0,
            anomaly_accuracy=float(np.mean(anom_correct)) if anom_correct else 0.0,
            root_hit_rate=float(np.mean(root_hits)) if root_hits else 0.0,
        )

    def save(self, path: str) -> None:
        ensure_parent(path)
        torch.save(
            {
                "config": self.config,
                "encoder": self.encoder.state_dict(),
                "channel_head": self.channel_head.state_dict(),
                "temporal_head": self.temporal_head.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, config: Dict[str, Any]) -> "TorchAXISEncoder":
        obj = cls(config)
        ckpt = torch.load(path, map_location=obj.device, weights_only=False)
        obj.encoder.load_state_dict(ckpt["encoder"])
        obj.channel_head.load_state_dict(ckpt["channel_head"])
        if "temporal_head" in ckpt:
            obj.temporal_head.load_state_dict(ckpt["temporal_head"])
        return obj


class TorchHintTuner:
    def __init__(self, config: Dict[str, Any]):
        _set_torch_seed(int(config.get("seed", 72)) + 1000)
        self.config = config
        self.device = _resolve_device(config)
        model_cfg = config["model"]
        self.batch_size = int(model_cfg.get("batch_size", 16))
        input_dim = 1 + int(model_cfg["feature_dim"]) + int(model_cfg["metadata_dim"])
        hidden_dim = int(model_cfg.get("hint_hidden_dim", 32))
        self.root_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(model_cfg.get("dropout", 0.1))),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self.null_logit = nn.Parameter(torch.zeros(()).to(self.device))
        self.anom_head = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        ).to(self.device)
        self._optimizer: torch.optim.Optimizer | None = None

    def parameters(self):
        return list(self.root_head.parameters()) + [self.null_logit] + list(self.anom_head.parameters())

    def _ensure_optimizer(self, lr: float) -> torch.optim.Optimizer:
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        return self._optimizer

    def _inputs_from_batch(self, encoder: TorchAXISEncoder, batch: List[ModelInput]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = _collate_model_inputs(batch, self.device)
        encoder.encoder.eval()
        encoder.channel_head.eval()
        with torch.no_grad():
            logits = encoder.channel_head(encoder._channel_representations(data)).squeeze(-1)
            probs = torch.sigmoid(logits)
        channel_x = torch.cat([probs.unsqueeze(-1), data["channel_feats"], data["meta_feats"]], dim=-1)
        relation_break = []
        over_fluctuation = []
        gap = []
        for mi in batch:
            flags = mi.evidence_card.get("summary_flags", {})
            relation_break.append(float(flags.get("has_relation_break", False)))
            over_fluctuation.append(float(flags.get("has_over_fluctuation", False)))
            gap_level = {"none": 0.0, "slight": 0.33, "moderate": 0.66, "severe": 1.0}.get(flags.get("max_gap_level", "none"), 0.0)
            gap.append(gap_level)
        anom_x = torch.stack(
            [
                probs.max(dim=1).values,
                probs.mean(dim=1),
                torch.tensor(over_fluctuation, dtype=torch.float32, device=self.device),
                torch.tensor(relation_break, dtype=torch.float32, device=self.device),
                torch.tensor(gap, dtype=torch.float32, device=self.device),
            ],
            dim=-1,
        )
        return channel_x, anom_x, data["root_targets"], data["is_anom"]

    def predict(self, encoder: TorchAXISEncoder, mi: ModelInput) -> Dict[str, Any]:
        self.root_head.eval()
        self.anom_head.eval()
        with torch.no_grad():
            channel_x, anom_x, _, _ = self._inputs_from_batch(encoder, [mi])
            root_logits = self.root_head(channel_x).squeeze(-1)[0]
            all_logits = torch.cat([root_logits, self.null_logit.reshape(1)], dim=0)
            root_probs = F.softmax(all_logits, dim=0).detach().cpu().numpy()
            anom_prob = float(torch.sigmoid(self.anom_head(anom_x).squeeze(-1))[0].detach().cpu())
        root_idx = int(np.argmax(root_probs))
        root_id = None if root_idx == len(mi.channels) else mi.channels[root_idx]["channel_id"]
        encoder_probs, _ = encoder.predict_channel_probs(mi)
        order = np.argsort(-root_probs[: len(mi.channels)])
        return {
            "is_anomalous_prob": anom_prob,
            "root_cause_channel": root_id,
            "root_probs": root_probs,
            "top_channels": [
                {
                    "channel_id": mi.channels[int(i)]["channel_id"],
                    "score": float(root_probs[int(i)]),
                    "encoder_prob": float(encoder_probs[int(i)]),
                }
                for i in order
            ],
        }

    def train_epoch(self, encoder: TorchAXISEncoder, inputs: List[ModelInput], lr: float) -> float:
        self.root_head.train()
        self.anom_head.train()
        opt = self._ensure_optimizer(lr)
        losses: List[float] = []
        for batch in _batch_iter(inputs, self.batch_size, shuffle=True):
            channel_x, anom_x, root_targets, is_anom = self._inputs_from_batch(encoder, batch)
            root_logits = self.root_head(channel_x).squeeze(-1)
            null_logits = self.null_logit.expand(root_logits.shape[0], 1)
            all_logits = torch.cat([root_logits, null_logits], dim=1)
            root_loss = F.cross_entropy(all_logits, root_targets)
            anom_logits = self.anom_head(anom_x).squeeze(-1)
            anom_loss = F.binary_cross_entropy_with_logits(anom_logits, is_anom)
            loss = root_loss + anom_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else 0.0

    def evaluate(self, encoder: TorchAXISEncoder, inputs: List[ModelInput]) -> HintMetrics:
        self.root_head.eval()
        self.anom_head.eval()
        losses: List[float] = []
        root_ok: List[float] = []
        ab_root_ok: List[float] = []
        anom_ok: List[float] = []
        with torch.no_grad():
            for batch in _batch_iter(inputs, self.batch_size, shuffle=False):
                channel_x, anom_x, root_targets, is_anom = self._inputs_from_batch(encoder, batch)
                root_logits = self.root_head(channel_x).squeeze(-1)
                all_logits = torch.cat([root_logits, self.null_logit.expand(root_logits.shape[0], 1)], dim=1)
                anom_logits = self.anom_head(anom_x).squeeze(-1)
                loss = F.cross_entropy(all_logits, root_targets) + F.binary_cross_entropy_with_logits(anom_logits, is_anom)
                losses.append(float(loss.detach().cpu()))
                root_idx = torch.argmax(all_logits, dim=1)
                anom_pred = torch.sigmoid(anom_logits) > 0.5
                for i, mi in enumerate(batch):
                    ok = float(root_idx[i].item() == root_targets[i].item())
                    root_ok.append(ok)
                    if mi.is_anomalous:
                        ab_root_ok.append(ok)
                    anom_ok.append(float(bool(anom_pred[i].item()) == mi.is_anomalous))
        return HintMetrics(
            loss=float(np.mean(losses)) if losses else 0.0,
            root_accuracy=float(np.mean(root_ok)) if root_ok else 0.0,
            abnormal_root_accuracy=float(np.mean(ab_root_ok)) if ab_root_ok else 0.0,
            anomaly_accuracy=float(np.mean(anom_ok)) if anom_ok else 0.0,
        )

    def make_soft_hint_summary(self, encoder: TorchAXISEncoder, mi: ModelInput, top_k: int) -> Dict[str, Any]:
        pred = self.predict(encoder, mi)
        return {
            "global_hint": {
                "anomaly_probability": pred["is_anomalous_prob"],
                "predicted_root_cause_channel": pred["root_cause_channel"],
            },
            "channel_hints": pred["top_channels"][:top_k],
        }

    def save(self, path: str) -> None:
        ensure_parent(path)
        torch.save(
            {
                "config": self.config,
                "root_head": self.root_head.state_dict(),
                "null_logit": self.null_logit.detach().cpu(),
                "anom_head": self.anom_head.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, config: Dict[str, Any]) -> "TorchHintTuner":
        obj = cls(config)
        ckpt = torch.load(path, map_location=obj.device, weights_only=False)
        obj.root_head.load_state_dict(ckpt["root_head"])
        obj.null_logit.data.copy_(ckpt["null_logit"].to(obj.device))
        obj.anom_head.load_state_dict(ckpt["anom_head"])
        return obj
