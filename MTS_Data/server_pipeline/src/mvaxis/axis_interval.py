from __future__ import annotations

import importlib.util
import math
import random
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .proposal import affected_f1, interval_coverage, interval_iou, true_affected_channels, true_anomaly_interval
from .hint_heads import GlobalHintHead
from .utils import ensure_parent


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_axis_time_series_encoder():
    root = _repo_root()
    candidates = [
        root / "legacy" / "AXIS_repo" / "src" / "models" / "AXIS" / "ts_encoder_bi_bias.py",
        root.parent / "external" / "AXIS_repo" / "src" / "models" / "AXIS" / "ts_encoder_bi_bias.py",
        root / "src" / "models" / "AXIS" / "ts_encoder_bi_bias.py",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    spec = importlib.util.spec_from_file_location("axis_original_ts_encoder_bi_bias", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AXIS TimeSeriesEncoder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TimeSeriesEncoder


def _device(config: Dict[str, Any]) -> torch.device:
    requested = str(config.get("model", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-6)


def _evenly_spaced_indices(length: int, limit: Optional[int]) -> List[int]:
    if length <= 0:
        return []
    if limit is None or int(limit) <= 0 or length <= int(limit):
        return list(range(length))
    return [int(x) for x in np.linspace(0, length - 1, int(limit))]


def _batches(rows: List[Dict[str, Any]], batch_size: int, shuffle: bool) -> Iterable[List[Dict[str, Any]]]:
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    ordered = [rows[i] for i in idx]
    for i in range(0, len(ordered), batch_size):
        yield ordered[i : i + batch_size]


def _collate(rows: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    max_len = max(int(r["series"]["shape"][0]) for r in rows)
    channels = max(int(r["series"]["shape"][1]) for r in rows)
    values = np.zeros((len(rows), max_len, channels), dtype=np.float32)
    labels = np.full((len(rows), max_len, channels), -1, dtype=np.int64)
    mask = np.zeros((len(rows), max_len), dtype=bool)
    channel_mask = np.zeros((len(rows), channels), dtype=bool)
    for i, row in enumerate(rows):
        v = np.asarray(row["series"]["values"], dtype=np.float32)
        y = np.asarray(row["series"]["labels"], dtype=np.int64)
        length = v.shape[0]
        width = v.shape[1]
        values[i, :length, :width] = _normalize(v)
        labels[i, :length, :width] = y
        mask[i, :length] = True
        channel_mask[i, :width] = True
    return {
        "values": torch.from_numpy(values).to(device),
        "labels": torch.from_numpy(labels).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "channel_mask": torch.from_numpy(channel_mask).to(device),
    }


@dataclass
class IntervalProposalMetrics:
    detection_accuracy: float
    temporal_iou: float
    truth_coverage: float
    abnormal_temporal_iou: float
    abnormal_truth_coverage: float
    close_rate: float
    root_hit_rate: float
    affected_f1: float
    point_f1: float
    threshold: float


class AXISMultivariateIntervalProposer(nn.Module):
    """Original AXIS TimeSeriesEncoder plus the AXIS anomaly head for multivariate interval proposal."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        _set_seed(int(config.get("seed", 72)))
        self.config = config
        self.device = _device(config)
        ts_cfg = config["model"].get("ts_encoder", {})
        TimeSeriesEncoder = _load_axis_time_series_encoder()
        self.ts_encoder = TimeSeriesEncoder(
            d_model=int(ts_cfg.get("d_model", 64)),
            d_proj=int(ts_cfg.get("d_proj", 32)),
            patch_size=int(ts_cfg.get("patch_size", 16)),
            num_layers=int(ts_cfg.get("num_layers", 2)),
            num_heads=int(ts_cfg.get("num_heads", 4)),
            d_ff_dropout=float(ts_cfg.get("dropout", 0.1)),
            max_total_tokens=int(ts_cfg.get("max_total_tokens", 8192)),
            use_rope=bool(ts_cfg.get("use_rope", True)),
            num_features=int(config["data"]["num_channels"]),
            activation=str(ts_cfg.get("activation", "gelu")),
        )
        d_proj = int(ts_cfg.get("d_proj", 32))
        self.anomaly_head = nn.Sequential(
            nn.Linear(d_proj, max(2, d_proj // 2)),
            nn.GELU(),
            nn.Dropout(float(config["model"].get("dropout", 0.1))),
            nn.Linear(max(2, d_proj // 2), 2),
        )
        global_cfg = config["model"].get("global_hint_head", {})
        self.global_hint_head = GlobalHintHead(
            d_proj=d_proj,
            d_query=int(global_cfg.get("d_query", d_proj)),
            question_bottleneck_dim=int(global_cfg.get("question_bottleneck_dim", 64)),
            num_question_queries=int(global_cfg.get("num_question_queries", 2)),
            num_base_queries=int(global_cfg.get("num_base_queries", 1)),
            channel_pool_heads=int(global_cfg.get("channel_pool_heads", int(ts_cfg.get("num_heads", 4)))),
            temporal_layers=int(global_cfg.get("temporal_layers", global_cfg.get("num_layers", 3))),
            temporal_heads=int(global_cfg.get("temporal_heads", global_cfg.get("num_heads", int(ts_cfg.get("num_heads", 4))))),
            temporal_ffn_dim=int(global_cfg.get("temporal_ffn_dim", 4 * d_proj)),
            dropout=float(global_cfg.get("dropout", config["model"].get("dropout", 0.1))),
            query_mode=str(global_cfg.get("query_mode", "data_only")),
            query_mix_alpha=float(global_cfg.get("query_mix_alpha", 0.5)),
            use_stats_residual=bool(global_cfg.get("use_stats_residual", True)),
            use_anomaly_probs=bool(global_cfg.get("use_anomaly_probs", True)),
            temporal_causal=bool(global_cfg.get("temporal_causal", False)),
        )
        self.threshold = float(config.get("interval_proposal", {}).get("threshold", 0.5))
        self.to(self.device)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        local_embeddings = self.ts_encoder(values, mask)
        return self.anomaly_head(local_embeddings)

    def anomaly_probabilities(self, sample: Dict[str, Any]) -> np.ndarray:
        self.eval()
        batch = _collate([sample], self.device)
        with torch.no_grad():
            logits = self.forward(batch["values"], batch["mask"])
            probs = F.softmax(logits, dim=-1)[..., 1][0]
        length = int(sample["series"]["shape"][0])
        return probs[:length].detach().cpu().numpy()

    def _question_embeddings(
        self,
        question_text: Optional[str],
        llm_question_embedder: Any,
        *,
        max_question_tokens: int = 128,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        text = str(question_text or "").strip()
        if not text or llm_question_embedder is None:
            return None, None
        extractor = getattr(llm_question_embedder, "get_input_token_embeddings", None)
        if extractor is None:
            return None, None
        embeddings, question_mask = extractor([text], max_length=int(max_question_tokens))
        if embeddings is None or question_mask is None:
            return None, None
        return embeddings.to(self.device), question_mask.to(self.device)

    def _global_hint_state_and_diagnostics(
        self,
        sample: Dict[str, Any],
        *,
        question_text: Optional[str] = None,
        llm_question_embedder: Any = None,
        query_mode: Optional[str] = None,
        query_mix_alpha: Optional[float] = None,
        max_question_tokens: int = 128,
    ) -> Tuple[torch.Tensor, Dict[str, Any], torch.Tensor, int]:
        self.eval()
        batch = _collate([sample], self.device)
        with torch.no_grad():
            local_embeddings = self.ts_encoder(batch["values"], batch["mask"])
            logits = self.anomaly_head(local_embeddings)
            probs = torch.softmax(logits, dim=-1)[..., 1]
            requested_mode = str(query_mode or self.global_hint_head.query_mode).strip().lower()
            question_embeddings, question_mask = self._question_embeddings(
                question_text,
                llm_question_embedder,
                max_question_tokens=max_question_tokens,
            )
            if requested_mode in {"frozen_llm_embedding", "hybrid"} and question_embeddings is None:
                applied_mode = "data_only"
            else:
                applied_mode = requested_mode
            global_states, diagnostics = self.global_hint_head(
                local_embeddings=local_embeddings,
                time_mask=batch["mask"],
                channel_mask=batch["channel_mask"],
                question_embeddings=question_embeddings,
                question_mask=question_mask,
                anomaly_probs=probs,
                query_mode=applied_mode,
                query_mix_alpha=query_mix_alpha,
            )
        length = int(sample["series"]["shape"][0])
        return global_states, diagnostics, probs, length

    def global_hint_forward_tensors(
        self,
        sample: Dict[str, Any],
        *,
        question_text: Optional[str] = None,
        llm_question_embedder: Any = None,
        query_mode: Optional[str] = None,
        query_mix_alpha: Optional[float] = None,
        max_question_tokens: int = 128,
    ) -> Dict[str, Any]:
        """Return differentiable tensors for global-hint warmup/alignment.

        The original ts_encoder/anomaly_head stay frozen by convention. Their
        outputs are detached here so warmup can optimize only the new
        GlobalHintHead parameters while still reusing AXIS local embeddings and
        anomaly probabilities.
        """

        batch = _collate([sample], self.device)
        with torch.no_grad():
            local_embeddings = self.ts_encoder(batch["values"], batch["mask"]).detach()
            logits = self.anomaly_head(local_embeddings)
            probs = torch.softmax(logits, dim=-1)[..., 1].detach()
            question_embeddings, question_mask = self._question_embeddings(
                question_text,
                llm_question_embedder,
                max_question_tokens=max_question_tokens,
            )
        requested_mode = str(query_mode or self.global_hint_head.query_mode).strip().lower()
        if requested_mode in {"frozen_llm_embedding", "hybrid"} and question_embeddings is None:
            applied_mode = "data_only"
        else:
            applied_mode = requested_mode
        global_states, diagnostics = self.global_hint_head(
            local_embeddings=local_embeddings,
            time_mask=batch["mask"],
            channel_mask=batch["channel_mask"],
            question_embeddings=question_embeddings,
            question_mask=question_mask,
            anomaly_probs=probs,
            query_mode=applied_mode,
            query_mix_alpha=query_mix_alpha,
        )
        labels = batch["labels"]
        time_labels = (labels >= 0).any(dim=-1) & (labels.gt(0).any(dim=-1))
        return {
            "global_states": global_states,
            "diagnostics": diagnostics,
            "anomaly_probs": probs,
            "local_embeddings": local_embeddings,
            "time_mask": batch["mask"].bool(),
            "channel_mask": batch["channel_mask"].bool(),
            "labels": labels,
            "time_labels": time_labels.long(),
            "query_mode_requested": requested_mode,
            "query_mode_used": applied_mode,
        }

    def local_hint_embeddings(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        pool: str = "channel_topk",
        top_k_channels: Optional[int] = None,
    ) -> np.ndarray:
        """Backward-compatible alias for channel hint embeddings."""
        return self.channel_hint_embeddings(
            sample,
            start,
            end,
            pool=pool,
            top_k_channels=top_k_channels,
        )

    def channel_hint_embeddings(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        pool: str = "channel_topk",
        top_k_channels: Optional[int] = None,
        max_channel_tokens: Optional[int] = None,
    ) -> np.ndarray:
        """Return multivariate channel hint embeddings for a proposed interval.

        The default emits time-major top-k channel tokens, preserving channel
        identity instead of averaging all variables into one local token.
        """
        tokens, _ = self._channel_hint_tokens_and_trace(
            sample,
            start,
            end,
            pool=pool,
            top_k_channels=top_k_channels,
            max_channel_tokens=max_channel_tokens,
        )
        return tokens

    def _channel_hint_tokens_and_trace(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        pool: str = "channel_topk",
        top_k_channels: Optional[int] = None,
        max_channel_tokens: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        self.eval()
        batch = _collate([sample], self.device)
        with torch.no_grad():
            local_embeddings = self.ts_encoder(batch["values"], batch["mask"])[0]
            logits = self.anomaly_head(local_embeddings)
            probs = F.softmax(logits, dim=-1)[..., 1]
        length = int(sample["series"]["shape"][0])
        start = max(0, min(int(start), length - 1))
        end = max(start + 1, min(int(end), length))
        active_indices = [idx for idx, ch in enumerate(sample["channels"]) if ch.get("active", True)]
        if not active_indices:
            active_indices = list(range(local_embeddings.shape[1]))
        window_embeddings = local_embeddings[start:end, active_indices, :]
        window_probs = probs[start:end, active_indices]
        if pool == "channel_topk":
            k = top_k_channels or int(self.config["model"].get("top_k_channels", 3))
            k = max(1, min(int(k), len(active_indices)))
            top = torch.topk(window_probs, k=k, dim=1).indices
            gather_idx = top.unsqueeze(-1).expand(-1, -1, window_embeddings.shape[-1])
            tokens = torch.gather(window_embeddings, 1, gather_idx).reshape(-1, window_embeddings.shape[-1])
            trace: List[Dict[str, Any]] = []
            top_cpu = top.detach().cpu().numpy()
            probs_cpu = window_probs.detach().cpu().numpy()
            for rel_t in range(top_cpu.shape[0]):
                for rank in range(top_cpu.shape[1]):
                    active_pos = int(top_cpu[rel_t, rank])
                    ch_idx = int(active_indices[active_pos])
                    trace.append(
                        {
                            "time_index": int(start + rel_t),
                            "relative_index": int(rel_t),
                            "channel_id": sample["channels"][ch_idx]["channel_id"],
                            "rank": int(rank + 1),
                            "score": float(probs_cpu[rel_t, active_pos]),
                        }
                    )
            tokens_np = tokens.detach().cpu().numpy()
        elif pool == "mean":
            pooled = window_embeddings.mean(dim=1)
            trace = [
                {"time_index": int(t), "relative_index": int(t - start), "channel_id": "mean_active_channels"}
                for t in range(start, end)
            ]
            tokens_np = pooled.detach().cpu().numpy()
        else:
            weights = F.softmax(window_probs / 0.25, dim=1).unsqueeze(-1)
            pooled = (weights * window_embeddings).sum(dim=1)
            trace = [
                {"time_index": int(t), "relative_index": int(t - start), "channel_id": "weighted_active_channels"}
                for t in range(start, end)
            ]
            tokens_np = pooled.detach().cpu().numpy()
        keep = _evenly_spaced_indices(len(tokens_np), max_channel_tokens)
        if keep and len(keep) < len(tokens_np):
            tokens_np = tokens_np[keep]
            trace = [trace[i] for i in keep]
        return tokens_np, trace

    def global_hint_embeddings(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        *,
        question_text: Optional[str] = None,
        llm_question_embedder: Any = None,
        query_mode: Optional[str] = None,
        query_mix_alpha: Optional[float] = None,
        global_sampling_strategy: str = "uniform",
        max_question_tokens: int = 128,
        max_global_tokens: Optional[int] = None,
    ) -> np.ndarray:
        global_states, _, _, length = self._global_hint_state_and_diagnostics(
            sample,
            question_text=question_text,
            llm_question_embedder=llm_question_embedder,
            query_mode=query_mode,
            query_mix_alpha=query_mix_alpha,
            max_question_tokens=max_question_tokens,
        )
        token_count = max_global_tokens
        if token_count is None:
            token_count = int(
                (self.config.get("model", {}).get("global_hint_head", {}) or {}).get("num_global_tokens", 4)
            )
        start = max(0, min(int(start), length - 1))
        end = max(start + 1, min(int(end), length))
        intervals = torch.tensor([[start, end]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            tokens, _ = self.global_hint_head.sample_interval_tokens(
                global_states,
                intervals,
                int(token_count),
                strategy=global_sampling_strategy,
            )
            tokens = tokens[0]
        tokens_np = tokens.detach().cpu().numpy()
        return tokens_np

    def embedding_hint_trace(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        *,
        channel_trace: Optional[List[Dict[str, Any]]] = None,
        num_global_tokens: Optional[int] = None,
        global_trace_extras: Optional[Dict[str, Any]] = None,
        max_trace_positions: int = 24,
    ) -> Dict[str, Any]:
        start = int(start)
        end = int(end)
        length = max(0, end - start)
        sampled_rel = _evenly_spaced_indices(length, min(max_trace_positions, length) if length else None)
        sampled_times = [int(start + i) for i in sampled_rel]
        visible_channel_trace = [
            {
                key: item.get(key)
                for key in ["time_index", "relative_index", "channel_id", "rank"]
                if key in item
            }
            for item in list(channel_trace or [])[: int(max_trace_positions)]
        ]
        return {
            "global": {
                "source_interval": [start, end],
                "sampled_time_indices": sampled_times,
                "num_global_tokens": int(num_global_tokens or 0),
                **dict(global_trace_extras or {}),
            },
            "channel": {
                "source_interval": [start, end],
                "num_channel_tokens": len(channel_trace or []),
                "sampled_positions": visible_channel_trace,
            },
        }

    def make_embedding_hints(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        *,
        question_text: Optional[str] = None,
        llm_question_embedder: Any = None,
        query_mode: Optional[str] = None,
        query_mix_alpha: Optional[float] = None,
        global_sampling_strategy: str = "uniform",
        max_question_tokens: int = 128,
        top_k_channels: Optional[int] = None,
        max_channel_tokens: Optional[int] = None,
        max_global_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        channel_tokens, channel_trace = self._channel_hint_tokens_and_trace(
            sample,
            start,
            end,
            top_k_channels=top_k_channels,
            max_channel_tokens=max_channel_tokens,
        )
        global_tokens = self.global_hint_embeddings(
            sample,
            start,
            end,
            question_text=question_text,
            llm_question_embedder=llm_question_embedder,
            query_mode=query_mode,
            query_mix_alpha=query_mix_alpha,
            global_sampling_strategy=global_sampling_strategy,
            max_question_tokens=max_question_tokens,
            max_global_tokens=max_global_tokens,
        )
        requested_mode = str(query_mode or self.global_hint_head.query_mode).strip().lower()
        applied_mode = requested_mode
        if requested_mode in {"frozen_llm_embedding", "hybrid"}:
            can_embed_question = bool(
                str(question_text or "").strip()
                and getattr(llm_question_embedder, "get_input_token_embeddings", None) is not None
            )
            applied_mode = requested_mode if can_embed_question else "data_only"
        return {
            "embeddings": {
                "global": global_tokens,
                "channel": channel_tokens,
            },
            "trace": self.embedding_hint_trace(
                sample,
                start,
                end,
                channel_trace=channel_trace,
                num_global_tokens=len(global_tokens),
                global_trace_extras={
                    "query_mode_requested": requested_mode,
                    "query_mode_used": applied_mode,
                    "sampling_strategy": str(global_sampling_strategy),
                },
            ),
        }

    def make_channel_only_embedding_hints(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        *,
        top_k_channels: Optional[int] = None,
        max_channel_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return only channel hint embeddings and trace information.

        This path intentionally skips global hint generation so datasets whose
        channel count differs from the checkpointed GlobalHintHead can still use
        ts_encoder/anomaly_head-derived channel tokens.
        """
        channel_tokens, channel_trace = self._channel_hint_tokens_and_trace(
            sample,
            start,
            end,
            top_k_channels=top_k_channels,
            max_channel_tokens=max_channel_tokens,
        )
        return {
            "embeddings": {
                "channel": channel_tokens,
            },
            "trace": self.embedding_hint_trace(
                sample,
                start,
                end,
                channel_trace=channel_trace,
                num_global_tokens=0,
            ),
        }

    def make_embedding_hint_tensors(
        self,
        sample: Dict[str, Any],
        start: int,
        end: int,
        *,
        question_text: Optional[str] = None,
        llm_question_embedder: Any = None,
        query_mode: Optional[str] = None,
        query_mix_alpha: Optional[float] = None,
        global_sampling_strategy: str = "uniform",
        max_question_tokens: int = 128,
        top_k_channels: Optional[int] = None,
        max_channel_tokens: Optional[int] = None,
        max_global_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return differentiable global tokens plus frozen channel tokens.

        This is used by bridge training: the original AXIS encoder/anomaly head
        can stay frozen, while the new GlobalHintHead remains trainable.
        """
        batch = _collate([sample], self.device)
        length = int(sample["series"]["shape"][0])
        start = max(0, min(int(start), length - 1))
        end = max(start + 1, min(int(end), length))
        active_indices = [idx for idx, ch in enumerate(sample["channels"]) if ch.get("active", True)]
        if not active_indices:
            active_indices = list(range(int(batch["values"].shape[-1])))
        with torch.no_grad():
            local_embeddings = self.ts_encoder(batch["values"], batch["mask"])[0]
            logits = self.anomaly_head(local_embeddings)
            probs = F.softmax(logits, dim=-1)[..., 1]
            window_embeddings = local_embeddings[start:end, active_indices, :]
            window_probs = probs[start:end, active_indices]
            k = top_k_channels or int(self.config["model"].get("top_k_channels", 3))
            k = max(1, min(int(k), len(active_indices)))
            top = torch.topk(window_probs, k=k, dim=1).indices
            gather_idx = top.unsqueeze(-1).expand(-1, -1, window_embeddings.shape[-1])
            channel_tokens = torch.gather(window_embeddings, 1, gather_idx).reshape(-1, window_embeddings.shape[-1])
            trace: List[Dict[str, Any]] = []
            top_cpu = top.detach().cpu().numpy()
            probs_cpu = window_probs.detach().cpu().numpy()
            for rel_t in range(top_cpu.shape[0]):
                for rank in range(top_cpu.shape[1]):
                    active_pos = int(top_cpu[rel_t, rank])
                    ch_idx = int(active_indices[active_pos])
                    trace.append(
                        {
                            "time_index": int(start + rel_t),
                            "relative_index": int(rel_t),
                            "channel_id": sample["channels"][ch_idx]["channel_id"],
                            "rank": int(rank + 1),
                            "score": float(probs_cpu[rel_t, active_pos]),
                        }
                    )
        keep_channel = _evenly_spaced_indices(int(channel_tokens.shape[0]), max_channel_tokens)
        if keep_channel and len(keep_channel) < int(channel_tokens.shape[0]):
            channel_tokens = channel_tokens.index_select(
                0,
                torch.tensor(keep_channel, dtype=torch.long, device=channel_tokens.device),
            )
            trace = [trace[i] for i in keep_channel]
        question_embeddings, question_mask = self._question_embeddings(
            question_text,
            llm_question_embedder,
            max_question_tokens=max_question_tokens,
        )
        requested_mode = str(query_mode or self.global_hint_head.query_mode).strip().lower()
        if requested_mode in {"frozen_llm_embedding", "hybrid"} and question_embeddings is None:
            applied_mode = "data_only"
        else:
            applied_mode = requested_mode
        global_states, _ = self.global_hint_head(
            local_embeddings=local_embeddings.unsqueeze(0),
            time_mask=batch["mask"],
            channel_mask=batch["channel_mask"],
            question_embeddings=question_embeddings,
            question_mask=question_mask,
            anomaly_probs=probs.unsqueeze(0),
            query_mode=applied_mode,
            query_mix_alpha=query_mix_alpha,
        )
        intervals = torch.tensor([[start, end]], dtype=torch.long, device=self.device)
        global_token_count = max_global_tokens
        if global_token_count is None:
            global_token_count = int(
                (self.config.get("model", {}).get("global_hint_head", {}) or {}).get("num_global_tokens", 4)
            )
        global_tokens, sampled = self.global_hint_head.sample_interval_tokens(
            global_states,
            intervals,
            int(global_token_count),
            strategy=global_sampling_strategy,
        )
        global_tokens = global_tokens[0]
        return {
            "embeddings": {
                "global": global_tokens,
                "channel": channel_tokens.detach(),
            },
            "trace": self.embedding_hint_trace(
                sample,
                start,
                end,
                channel_trace=trace,
                num_global_tokens=int(global_tokens.shape[0]),
                global_trace_extras={
                    "query_mode_requested": requested_mode,
                    "query_mode_used": applied_mode,
                    "sampling_strategy": str(global_sampling_strategy),
                    "sampled_time_indices": [int(x) for x in sampled["sampled_time_indices"][0].detach().cpu().tolist()],
                },
            ),
        }

    def proposal_from_probs(
        self,
        sample: Dict[str, Any],
        probs: np.ndarray,
        window_size: int,
        stride: int,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        threshold = self.threshold if threshold is None else threshold
        length, channels = probs.shape
        active_indices = np.asarray(
            [idx for idx, ch in enumerate(sample["channels"]) if ch.get("active", True)],
            dtype=int,
        )
        if active_indices.size == 0:
            active_indices = np.arange(channels, dtype=int)
        active_probs = probs[:, active_indices]
        time_score = active_probs.max(axis=1)
        candidates: List[Dict[str, Any]] = []
        for start in range(0, max(1, length - window_size + 1), stride):
            end = min(length, start + window_size)
            score = float(time_score[start:end].mean())
            max_score = float(time_score[start:end].max())
            channel_order = active_indices[np.argsort(-probs[start:end][:, active_indices].max(axis=0))]
            candidates.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "proposal_score": score,
                    "max_point_score": max_score,
                    "predicted_root_cause_channel": sample["channels"][int(channel_order[0])]["channel_id"],
                    "channel_hints": [
                        {
                            "channel_id": sample["channels"][int(i)]["channel_id"],
                            "score": float(probs[start:end, int(i)].mean()),
                            "encoder_prob": float(probs[start:end, int(i)].max()),
                        }
                        for i in channel_order[: int(self.config["model"].get("top_k_channels", 3))]
                    ],
                }
            )
        candidates.sort(key=lambda x: x["proposal_score"], reverse=True)
        best = dict(candidates[0]) if candidates else {
            "start": None,
            "end": None,
            "proposal_score": 0.0,
            "max_point_score": 0.0,
            "predicted_root_cause_channel": None,
            "channel_hints": [],
        }
        best["is_anomalous_proposal"] = bool(best["proposal_score"] >= threshold)
        best["threshold"] = float(threshold)
        best["source"] = "axis_original_TimeSeriesEncoder_anomaly_head"
        best["global_hint"] = {
            "anomaly_probability": float(best["proposal_score"]),
            "predicted_interval": [best["start"], best["end"]] if best["start"] is not None else None,
            "predicted_root_cause_channel": best["predicted_root_cause_channel"] if best["is_anomalous_proposal"] else None,
            "candidate_root_cause_channel": best["predicted_root_cause_channel"],
            "is_anomalous_proposal": best["is_anomalous_proposal"],
            "hint_source": "AXIS anomaly_head",
        }
        best["local_hints"] = self._local_hints(sample, probs, best)
        best["soft_hints"] = {
            "interval_proposal": {
                "source": best["source"],
                "start": best["start"],
                "end": best["end"],
                "score": best["proposal_score"],
                "threshold": best["threshold"],
                "is_anomalous_proposal": best["is_anomalous_proposal"],
            },
            "global_hint": best["global_hint"],
            "channel_hints": best["channel_hints"],
            "local_hints": best["local_hints"],
            "usage_policy": "Hints are model outputs, not ground truth. The LLM must verify them against evidence_card.",
        }
        best["ranked_candidates"] = candidates[:5]
        return best

    def _local_hints(self, sample: Dict[str, Any], probs: np.ndarray, proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
        if proposal.get("start") is None:
            return []
        start = int(proposal["start"])
        end = int(proposal["end"])
        active_indices = [idx for idx, ch in enumerate(sample["channels"]) if ch.get("active", True)]
        if not active_indices:
            active_indices = list(range(probs.shape[1]))
        active_probs = probs[start:end][:, active_indices]
        time_score = active_probs.max(axis=1)
        top = np.argsort(-time_score)[: min(5, len(time_score))]
        hints = []
        for rel_idx in top:
            abs_idx = start + int(rel_idx)
            ch_idx = int(active_indices[int(np.argmax(probs[abs_idx, active_indices]))])
            hints.append(
                {
                    "time_index": abs_idx,
                    "relative_index": int(rel_idx),
                    "score": float(probs[abs_idx, ch_idx]),
                    "top_channel_id": sample["channels"][ch_idx]["channel_id"],
                }
            )
        return hints

    def propose_interval(self, sample: Dict[str, Any], window_size: int, stride: int, top_k_channels: int) -> Dict[str, Any]:
        probs = self.anomaly_probabilities(sample)
        return self.proposal_from_probs(sample, probs, window_size, stride)

    def save(self, path: str) -> None:
        ensure_parent(path)
        torch.save(
            {
                "config": self.config,
                "state_dict": self.state_dict(),
                "threshold": self.threshold,
            },
            path,
        )

    @classmethod
    def _maybe_materialize_global_question_adapter(
        cls,
        model: "AXISMultivariateIntervalProposer",
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        key = "global_hint_head.question_adapter.down_proj.weight"
        value = state_dict.get(key)
        value_shape = cls._safe_shape(value)
        if value is None or value_shape is None or len(value_shape) != 2:
            return
        try:
            model.global_hint_head.question_adapter.materialize_from_input_dim(
                int(value_shape[1]),
                device=model.device,
                dtype=value.dtype,
            )
        except Exception:
            return

    @staticmethod
    def _safe_shape(value: Any) -> Optional[Tuple[int, ...]]:
        try:
            return tuple(value.shape)
        except Exception:
            return None

    @classmethod
    def load(cls, path: str, config: Optional[Dict[str, Any]] = None) -> "AXISMultivariateIntervalProposer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = config or checkpoint["config"]
        model = cls(cfg)
        raw_state = checkpoint.get("state_dict", checkpoint)
        state_dict = cls._strip_state_dict_prefix(raw_state)
        cls._maybe_materialize_global_question_adapter(model, state_dict)
        own_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own_state
            and cls._safe_shape(own_state[key]) is not None
            and cls._safe_shape(own_state[key]) == cls._safe_shape(value)
        }
        model.load_state_dict(compatible, strict=False)
        model.threshold = float(checkpoint.get("threshold", model.threshold))
        model.to(model.device)
        return model

    @classmethod
    def load_shape_compatible_checkpoint(
        cls,
        path: str,
        config: Dict[str, Any],
        *,
        threshold: Optional[float] = None,
        num_channels_override: Optional[int] = None,
        skip_global_hint_head: bool = False,
    ) -> Tuple["AXISMultivariateIntervalProposer", Dict[str, Any]]:
        """Load any AXIS-style checkpoint by keeping only shape-compatible weights.

        This is useful when the target dataset changes channel count: the
        ts_encoder/anomaly_head weights can still transfer, while
        GlobalHintHead weights are skipped.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        state_dict = cls._strip_state_dict_prefix(raw_state)
        cfg = copy.deepcopy(config)
        cfg.setdefault("data", {})
        if num_channels_override is not None:
            cfg["data"]["num_channels"] = int(num_channels_override)
        model = cls(cfg)
        cls._maybe_materialize_global_question_adapter(model, state_dict)
        own_state = model.state_dict()
        compatible = {}
        skipped_shape: List[str] = []
        skipped_global: List[str] = []
        for key, value in state_dict.items():
            if skip_global_hint_head and key.startswith("global_hint_head."):
                skipped_global.append(key)
                continue
            if key not in own_state:
                continue
            own_shape = cls._safe_shape(own_state[key])
            value_shape = cls._safe_shape(value)
            if own_shape is not None and own_shape == value_shape:
                compatible[key] = value
            else:
                skipped_shape.append(key)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        model.threshold = float(
            checkpoint.get(
                "threshold",
                threshold if threshold is not None else model.threshold,
            )
        )
        model.to(model.device)
        load_info = {
            "checkpoint_path": str(path),
            "loaded_keys": len(compatible),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "skipped_shape_mismatch_keys": sorted(skipped_shape),
            "skipped_global_hint_head_keys": sorted(skipped_global),
            "num_channels_override": int(num_channels_override) if num_channels_override is not None else None,
            "skip_global_hint_head": bool(skip_global_hint_head),
        }
        return model, load_info

    @staticmethod
    def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]
            cleaned[key] = value
        return cleaned

    @staticmethod
    def infer_timercd_ts_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Infer AXIS/TimeRCD encoder dimensions from a pretrained checkpoint."""
        embedding = state_dict.get("ts_encoder.embedding_layer.weight")
        projection = state_dict.get("ts_encoder.projection_layer.weight")
        anomaly = state_dict.get("anomaly_head.0.weight")
        if embedding is None or projection is None or anomaly is None:
            raise ValueError("Checkpoint does not contain expected ts_encoder/anomaly_head weights.")
        layer_ids = []
        prefix = "ts_encoder.transformer_encoder.layers."
        for key in state_dict:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if "." in suffix:
                    layer_ids.append(int(suffix.split(".", 1)[0]))
        bias = state_dict.get(
            "ts_encoder.transformer_encoder.layers.0.self_attn.binary_attention_bias.emd.weight"
        )
        return {
            "d_model": int(embedding.shape[0]),
            "d_proj": int(anomaly.shape[1]),
            "patch_size": int(embedding.shape[1]),
            "num_layers": int(max(layer_ids) + 1) if layer_ids else 8,
            "num_heads": int(bias.shape[1]) if bias is not None else 8,
            "dropout": 0.0,
            "use_rope": True,
            "activation": "gelu",
            "max_total_tokens": 8192,
            "projection_out_features": int(projection.shape[0]),
        }

    @classmethod
    def load_timercd_checkpoint(
        cls,
        path: str,
        config: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Tuple["AXISMultivariateIntervalProposer", Dict[str, Any]]:
        """Load TimeRCD/AXIS pretrained ts_encoder + anomaly_head into the proposer.

        Hugging Face TimeRCD checkpoints store the full pretrain model
        (`ts_encoder`, `reconstruction_head`, `anomaly_head`). For interval
        proposal we intentionally ignore the reconstruction head and keep the
        same AXIS anomaly-head scoring path used elsewhere in this package.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        raw_state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        state_dict = cls._strip_state_dict_prefix(raw_state)
        inferred = cls.infer_timercd_ts_config(state_dict)
        cfg = copy.deepcopy(config)
        cfg.setdefault("model", {})
        cfg["model"].setdefault("ts_encoder", {})
        for key in ("d_model", "d_proj", "patch_size", "num_layers", "num_heads", "dropout", "use_rope", "activation", "max_total_tokens"):
            cfg["model"]["ts_encoder"][key] = inferred[key]
        model = cls(cfg)
        cls._maybe_materialize_global_question_adapter(model, state_dict)
        own_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own_state
            and cls._safe_shape(own_state[key]) is not None
            and cls._safe_shape(own_state[key]) == cls._safe_shape(value)
        }
        skipped_shape = sorted(
            key
            for key, value in state_dict.items()
            if key in own_state and cls._safe_shape(own_state[key]) != cls._safe_shape(value)
        )
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if threshold is not None:
            model.threshold = float(threshold)
        model.to(model.device)
        load_info = {
            "checkpoint_path": str(path),
            "inferred_ts_config": inferred,
            "loaded_keys": len(compatible),
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
            "skipped_shape_mismatch_keys": skipped_shape,
        }
        return model, load_info


def train_interval_proposer(
    train_rows: List[Dict[str, Any]],
    val_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    model = AXISMultivariateIntervalProposer(config)
    cfg = config["train_interval_proposal"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    batch_size = int(cfg.get("batch_size", config["model"].get("batch_size", 8)))
    pos_weight = float(cfg.get("pos_weight", 8.0))
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses: List[float] = []
        for rows in _batches(train_rows, batch_size, shuffle=True):
            batch = _collate(rows, model.device)
            logits = model(batch["values"], batch["mask"])
            valid = batch["labels"] >= 0
            loss = F.cross_entropy(
                logits[valid],
                batch["labels"][valid],
                weight=torch.tensor([1.0, pos_weight], dtype=torch.float32, device=model.device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics = evaluate_interval_proposer(model, val_rows, config, threshold=model.threshold)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, **metrics.__dict__})
    model.threshold = tune_threshold(model, val_rows, config)
    model.save(cfg["checkpoint_path"])
    final_metrics = evaluate_interval_proposer(model, val_rows, config, threshold=model.threshold)
    return {"model": model, "history": history, "val_metrics": final_metrics.__dict__}


def tune_threshold(model: AXISMultivariateIntervalProposer, rows: List[Dict[str, Any]], config: Dict[str, Any]) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        metrics = evaluate_interval_proposer(model, rows, config, threshold=float(threshold))
        score = metrics.detection_accuracy + metrics.close_rate + metrics.truth_coverage
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate_interval_proposer(
    model: AXISMultivariateIntervalProposer,
    rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    threshold: Optional[float] = None,
) -> IntervalProposalMetrics:
    proposal_cfg = config.get("interval_proposal", {})
    window_size = int(proposal_cfg.get("window_size", 32))
    stride = int(proposal_cfg.get("stride", 8))
    min_iou = float(proposal_cfg.get("min_iou", 0.25))
    threshold = model.threshold if threshold is None else threshold
    detection_ok: List[float] = []
    ious: List[float] = []
    coverages: List[float] = []
    abnormal_ious: List[float] = []
    abnormal_coverages: List[float] = []
    close: List[float] = []
    root_hits: List[float] = []
    affected_scores: List[float] = []
    point_tp = point_fp = point_fn = 0.0
    for row in rows:
        probs = model.anomaly_probabilities(row)
        proposal = model.proposal_from_probs(row, probs, window_size, stride, threshold=threshold)
        truth_interval = true_anomaly_interval(row)
        proposal_interval = None if proposal["start"] is None else (int(proposal["start"]), int(proposal["end"]))
        truth_anom = truth_interval is not None
        pred_anom = bool(proposal.get("is_anomalous_proposal", proposal_interval is not None))
        detection_ok.append(float(pred_anom == truth_anom))
        iou = interval_iou(proposal_interval, truth_interval)
        coverage = interval_coverage(proposal_interval, truth_interval)
        ious.append(iou)
        coverages.append(coverage)
        if truth_anom:
            abnormal_ious.append(iou)
            abnormal_coverages.append(coverage)
        close.append(float((not truth_anom and not pred_anom) or (truth_anom and iou >= min_iou)))
        truth_root = row["target_output"]["fact_check"].get("root_cause_channel")
        root_hits.append(float(proposal.get("predicted_root_cause_channel") == truth_root))
        affected_scores.append(affected_f1([x["channel_id"] for x in proposal.get("channel_hints", [])], true_affected_channels(row)))
        labels = np.asarray(row["series"]["labels"], dtype=int)
        pred_points = probs >= threshold
        point_tp += float(np.logical_and(pred_points, labels > 0).sum())
        point_fp += float(np.logical_and(pred_points, labels <= 0).sum())
        point_fn += float(np.logical_and(~pred_points, labels > 0).sum())
    point_precision = point_tp / max(1.0, point_tp + point_fp)
    point_recall = point_tp / max(1.0, point_tp + point_fn)
    point_f1 = 2 * point_precision * point_recall / max(1e-8, point_precision + point_recall)
    return IntervalProposalMetrics(
        detection_accuracy=float(np.mean(detection_ok)) if detection_ok else 0.0,
        temporal_iou=float(np.mean(ious)) if ious else 0.0,
        truth_coverage=float(np.mean(coverages)) if coverages else 0.0,
        abnormal_temporal_iou=float(np.mean(abnormal_ious)) if abnormal_ious else 0.0,
        abnormal_truth_coverage=float(np.mean(abnormal_coverages)) if abnormal_coverages else 0.0,
        close_rate=float(np.mean(close)) if close else 0.0,
        root_hit_rate=float(np.mean(root_hits)) if root_hits else 0.0,
        affected_f1=float(np.mean(affected_scores)) if affected_scores else 0.0,
        point_f1=float(point_f1),
        threshold=float(threshold),
    )
