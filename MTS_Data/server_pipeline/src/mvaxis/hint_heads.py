from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    if weights.ndim == 2:
        weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(-1)
    denom = weights.sum(dim=2, keepdim=False).clamp_min(1.0)
    return (values * weights).sum(dim=2) / denom


def _masked_std(values: torch.Tensor, mask: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    if weights.ndim == 2:
        weights = weights.unsqueeze(1)
    weights = weights.unsqueeze(-1)
    denom = weights.sum(dim=2, keepdim=False).clamp_min(1.0)
    centered = values - mean.unsqueeze(2)
    var = (centered.square() * weights).sum(dim=2) / denom
    return torch.sqrt(var.clamp_min(1e-6))


class FrozenQuestionEmbeddingAdapter(nn.Module):
    """Project frozen LLM input embeddings into a small query space."""

    def __init__(
        self,
        d_query: int = 256,
        bottleneck_dim: int = 64,
        num_question_queries: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_query = int(d_query)
        self.num_question_queries = int(num_question_queries)
        self.down_proj = nn.LazyLinear(int(bottleneck_dim))
        self.up_proj = nn.Linear(int(bottleneck_dim), self.d_query)
        self.norm = nn.LayerNorm(self.d_query)
        self.dropout = nn.Dropout(float(dropout))
        self.question_latents = nn.Parameter(torch.randn(self.num_question_queries, self.d_query) * 0.02)
        self.question_pool = nn.MultiheadAttention(
            embed_dim=self.d_query,
            num_heads=max(1, min(8, self.d_query // 32)),
            dropout=float(dropout),
            batch_first=True,
        )

    def forward(
        self,
        question_embeddings: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> torch.Tensor:
        if question_embeddings.ndim != 3:
            raise ValueError("question_embeddings must have shape [B, Lq, Dllm]")
        if question_mask.ndim != 2:
            raise ValueError("question_mask must have shape [B, Lq]")
        target_device = self.question_latents.device
        target_dtype = self.question_latents.dtype
        question_embeddings = question_embeddings.to(device=target_device, dtype=target_dtype)
        question_mask = question_mask.to(device=target_device)
        projected = self.down_proj(question_embeddings)
        projected = self.up_proj(torch.nn.functional.gelu(projected))
        projected = self.norm(self.dropout(projected))
        queries = self.question_latents.unsqueeze(0).expand(projected.shape[0], -1, -1)
        pooled, _ = self.question_pool(
            query=queries,
            key=projected,
            value=projected,
            key_padding_mask=~question_mask.bool(),
            need_weights=False,
        )
        return pooled

    def materialize_from_input_dim(
        self,
        input_dim: int,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not hasattr(self.down_proj, "has_uninitialized_params"):
            return
        if not self.down_proj.has_uninitialized_params():
            return
        sample = torch.zeros(1, 1, int(input_dim), device=device, dtype=dtype or torch.float32)
        with torch.no_grad():
            _ = self.down_proj(sample)


class AnyVariateChannelPool(nn.Module):
    """Permutation-invariant channel pooling over [B, T, C, d_proj]."""

    def __init__(
        self,
        d_proj: int = 256,
        num_question_queries: int = 2,
        num_base_queries: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_stats_residual: bool = True,
        use_anomaly_probs: bool = True,
    ) -> None:
        super().__init__()
        self.d_proj = int(d_proj)
        self.num_question_queries = int(num_question_queries)
        self.num_base_queries = int(num_base_queries)
        self.use_stats_residual = bool(use_stats_residual)
        self.use_anomaly_probs = bool(use_anomaly_probs)

        extra_dim = 1 if self.use_anomaly_probs else 0
        self.channel_feature_proj = nn.Sequential(
            nn.Linear(self.d_proj + extra_dim, self.d_proj),
            nn.LayerNorm(self.d_proj),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_proj, self.d_proj),
            nn.LayerNorm(self.d_proj),
        )
        self.base_queries = nn.Parameter(torch.randn(self.num_base_queries, self.d_proj) * 0.02)
        self.data_only_queries = nn.Parameter(torch.randn(self.num_question_queries, self.d_proj) * 0.02)
        self.channel_pool = nn.MultiheadAttention(
            embed_dim=self.d_proj,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        total_queries = self.num_base_queries + self.num_question_queries
        self.query_fusion = nn.Sequential(
            nn.Linear(total_queries * self.d_proj, self.d_proj),
            nn.LayerNorm(self.d_proj),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        if self.use_stats_residual:
            self.stats_proj = nn.Sequential(
                nn.Linear(2 * self.d_proj, self.d_proj),
                nn.LayerNorm(self.d_proj),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
        self.out_norm = nn.LayerNorm(self.d_proj)

    def build_queries(
        self,
        batch_size: int,
        device: torch.device,
        *,
        query_mode: str,
        query_mix_alpha: float,
        question_queries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, str]:
        base = self.base_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)
        data_queries = self.data_only_queries.unsqueeze(0).expand(batch_size, -1, -1).to(device=device)
        mode = str(query_mode or "data_only").strip().lower()
        if mode not in {"data_only", "frozen_llm_embedding", "hybrid"}:
            mode = "data_only"
        if mode == "data_only" or question_queries is None:
            return torch.cat([base, data_queries], dim=1), "data_only"
        question_queries = question_queries.to(device=device, dtype=base.dtype)
        if mode == "frozen_llm_embedding":
            mixed = question_queries
            return torch.cat([base, mixed], dim=1), "frozen_llm_embedding"
        alpha = float(max(0.0, min(1.0, query_mix_alpha)))
        mixed = alpha * question_queries + (1.0 - alpha) * data_queries
        return torch.cat([base, mixed], dim=1), "hybrid"

    def forward(
        self,
        local_embeddings: torch.Tensor,
        channel_mask: torch.Tensor,
        *,
        question_queries: Optional[torch.Tensor] = None,
        query_mode: str = "data_only",
        query_mix_alpha: float = 0.5,
        anomaly_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if local_embeddings.ndim != 4:
            raise ValueError("local_embeddings must have shape [B, T, C, d_proj]")
        batch_size, steps, channels, _ = local_embeddings.shape
        valid_channels = channel_mask.bool()
        if valid_channels.ndim != 2 or valid_channels.shape != (batch_size, channels):
            raise ValueError("channel_mask must have shape [B, C]")

        features = local_embeddings
        if self.use_anomaly_probs:
            if anomaly_probs is None:
                anomaly_probs = torch.zeros(
                    batch_size,
                    steps,
                    channels,
                    dtype=local_embeddings.dtype,
                    device=local_embeddings.device,
                )
            features = torch.cat([features, anomaly_probs.unsqueeze(-1).to(local_embeddings.dtype)], dim=-1)
        projected = self.channel_feature_proj(features)

        pooled_queries, mode_used = self.build_queries(
            batch_size,
            projected.device,
            query_mode=query_mode,
            query_mix_alpha=query_mix_alpha,
            question_queries=question_queries,
        )
        repeated_queries = pooled_queries[:, None, :, :].expand(batch_size, steps, -1, -1).reshape(
            batch_size * steps,
            pooled_queries.shape[1],
            self.d_proj,
        )
        flat_values = projected.reshape(batch_size * steps, channels, self.d_proj)
        flat_mask = (~valid_channels)[:, None, :].expand(batch_size, steps, channels).reshape(batch_size * steps, channels)
        pooled, attn = self.channel_pool(
            query=repeated_queries,
            key=flat_values,
            value=flat_values,
            key_padding_mask=flat_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        pooled = pooled.reshape(batch_size, steps, -1, self.d_proj)
        fused = self.query_fusion(pooled.reshape(batch_size, steps, -1))

        if self.use_stats_residual:
            mean_t = _masked_mean(projected, valid_channels)
            std_t = _masked_std(projected, valid_channels, mean_t)
            fused = fused + self.stats_proj(torch.cat([mean_t, std_t], dim=-1))

        diagnostics: Dict[str, torch.Tensor] = {
            "channel_attention": attn.mean(dim=1).reshape(batch_size, steps, pooled.shape[2], channels),
            "queries": pooled_queries,
        }
        return self.out_norm(fused), diagnostics


class TemporalGlobalRefiner(nn.Module):
    """Light temporal encoder over per-step global summaries."""

    def __init__(
        self,
        d_proj: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.d_proj = int(d_proj)
        self.causal = bool(causal)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_proj,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.out_norm = nn.LayerNorm(self.d_proj)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, z: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        src_mask = self._causal_mask(z.shape[1], z.device) if self.causal else None
        refined = self.encoder(
            z,
            mask=src_mask,
            src_key_padding_mask=~time_mask.bool(),
        )
        return self.out_norm(refined)


class GlobalHintHead(nn.Module):
    """Any-variate temporal global hint head producing dense per-time states G_t."""

    def __init__(
        self,
        d_proj: int = 256,
        d_query: int = 256,
        question_bottleneck_dim: int = 64,
        num_question_queries: int = 2,
        num_base_queries: int = 1,
        channel_pool_heads: int = 8,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_ffn_dim: int = 1024,
        dropout: float = 0.1,
        query_mode: str = "data_only",
        query_mix_alpha: float = 0.5,
        use_stats_residual: bool = True,
        use_anomaly_probs: bool = True,
        temporal_causal: bool = False,
    ) -> None:
        super().__init__()
        self.d_proj = int(d_proj)
        self.query_mode = str(query_mode or "data_only")
        self.query_mix_alpha = float(query_mix_alpha)
        self.question_adapter = FrozenQuestionEmbeddingAdapter(
            d_query=d_query,
            bottleneck_dim=question_bottleneck_dim,
            num_question_queries=num_question_queries,
            dropout=dropout,
        )
        self.channel_pool = AnyVariateChannelPool(
            d_proj=d_proj,
            num_question_queries=num_question_queries,
            num_base_queries=num_base_queries,
            num_heads=channel_pool_heads,
            dropout=dropout,
            use_stats_residual=use_stats_residual,
            use_anomaly_probs=use_anomaly_probs,
        )
        self.temporal_refiner = TemporalGlobalRefiner(
            d_proj=d_proj,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            ffn_dim=temporal_ffn_dim,
            dropout=dropout,
            causal=temporal_causal,
        )

    def forward(
        self,
        *,
        local_embeddings: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        question_embeddings: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        anomaly_probs: Optional[torch.Tensor] = None,
        query_mode: Optional[str] = None,
        query_mix_alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        question_queries = None
        if question_embeddings is not None and question_mask is not None:
            question_queries = self.question_adapter(question_embeddings, question_mask.bool())
        pooled, diagnostics = self.channel_pool(
            local_embeddings,
            channel_mask.bool(),
            question_queries=question_queries,
            query_mode=query_mode or self.query_mode,
            query_mix_alpha=self.query_mix_alpha if query_mix_alpha is None else query_mix_alpha,
            anomaly_probs=anomaly_probs,
        )
        global_states = self.temporal_refiner(pooled, time_mask.bool())
        diagnostics["global_states"] = global_states
        if question_queries is not None:
            diagnostics["question_queries"] = question_queries
        return global_states, diagnostics

    def sample_interval_tokens(
        self,
        global_states: torch.Tensor,
        intervals: torch.Tensor,
        max_global_tokens: int,
        strategy: str = "uniform",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if global_states.ndim != 3:
            raise ValueError("global_states must have shape [B, T, d_proj]")
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError("intervals must have shape [B, 2]")
        batch_size, steps, d_proj = global_states.shape
        token_count = max(1, int(max_global_tokens))
        sampled_tokens = []
        sampled_indices = []
        mode = str(strategy or "uniform").strip().lower()
        for batch_idx in range(batch_size):
            start = int(intervals[batch_idx, 0].item())
            end = int(intervals[batch_idx, 1].item())
            start = max(0, min(start, steps - 1))
            end = max(start + 1, min(end, steps))
            window = global_states[batch_idx, start:end]
            if window.shape[0] == 1:
                local_indices = torch.zeros(token_count, dtype=torch.long, device=global_states.device)
            elif mode == "anchors" and token_count >= 4:
                anchor_pos = torch.tensor(
                    [0.0, 0.25, 0.5, 0.75, 1.0],
                    device=global_states.device,
                    dtype=torch.float32,
                )
                if token_count != 5:
                    anchor_pos = torch.linspace(0.0, 1.0, steps=token_count, device=global_states.device)
                local_indices = torch.round(anchor_pos * (window.shape[0] - 1)).long()
            else:
                local_indices = torch.round(
                    torch.linspace(0, window.shape[0] - 1, steps=token_count, device=global_states.device)
                ).long()
            sampled_tokens.append(window.index_select(0, local_indices))
            sampled_indices.append((local_indices + start).to(dtype=torch.long))
        return (
            torch.stack(sampled_tokens, dim=0).reshape(batch_size, token_count, d_proj),
            {"sampled_time_indices": torch.stack(sampled_indices, dim=0)},
        )
