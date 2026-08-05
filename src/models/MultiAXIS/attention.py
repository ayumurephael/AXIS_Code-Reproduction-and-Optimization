from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flash_only_context(require_flash: bool):
    if not torch.cuda.is_available():
        return nullcontext()
    if not require_flash:
        return nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
    except (ImportError, AttributeError):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        )


class FlashCrossAttention(nn.Module):
    """Non-causal MHA whose CUDA path is forced onto the FlashAttention-2 SDPA kernel."""

    def __init__(self, embed_dim: int, num_heads: int, require_flash: bool = True):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.require_flash = require_flash
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query/key/value must have shape [B, N, D]")
        if key.shape != value.shape or query.shape[0] != key.shape[0]:
            raise ValueError("Cross-attention batch/key/value shapes do not match")
        batch, q_len, _ = query.shape
        kv_len = key.shape[1]

        def split(x: torch.Tensor, length: int) -> torch.Tensor:
            return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj(query), q_len)
        k = split(self.k_proj(key), kv_len)
        v = split(self.v_proj(value), kv_len)
        attention_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, kv_len):
                raise ValueError("key_padding_mask must have shape [B, K]")
            if not bool(key_padding_mask.any(dim=-1).all()):
                raise ValueError("Every sample must expose at least one valid key")
            attention_mask = key_padding_mask[:, None, None, :].bool()

        try:
            with _flash_only_context(self.require_flash):
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        except RuntimeError as exc:
            if torch.cuda.is_available() and self.require_flash:
                raise RuntimeError(
                    "FlashAttention-2 was required but PyTorch could not dispatch this "
                    "cross-attention operation to the flash SDPA backend."
                ) from exc
            raise
        output = output.transpose(1, 2).contiguous().view(batch, q_len, self.embed_dim)
        return self.out_proj(output)
