"""Configuration and audit helpers for the AXIS architecture redesign."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


ARCHITECTURE_VARIANTS = {
    "loss_only": {
        "qk_norm": False,
        "continuous_bypass": False,
        "direct_task_prompt": False,
    },
    "qk_only": {
        "qk_norm": True,
        "continuous_bypass": False,
        "direct_task_prompt": False,
    },
    "bypass_only": {
        "qk_norm": False,
        "continuous_bypass": True,
        "direct_task_prompt": False,
    },
    "task_prompt_only": {
        "qk_norm": False,
        "continuous_bypass": False,
        "direct_task_prompt": True,
    },
    "qk_bypass": {
        "qk_norm": True,
        "continuous_bypass": True,
        "direct_task_prompt": False,
    },
    "qk_task_prompt": {
        "qk_norm": True,
        "continuous_bypass": False,
        "direct_task_prompt": True,
    },
    "bypass_task_prompt": {
        "qk_norm": False,
        "continuous_bypass": True,
        "direct_task_prompt": True,
    },
    "full": {
        "qk_norm": True,
        "continuous_bypass": True,
        "direct_task_prompt": True,
    },
}


def add_architecture_arguments(parser, *, default_variant: str = "loss_only"):
    """Attach the shared factorial architecture options to a CLI parser."""
    parser.add_argument(
        "--architecture-variant",
        choices=sorted(ARCHITECTURE_VARIANTS),
        default=default_variant,
    )
    parser.add_argument("--qk-norm-seq-len", type=int)
    parser.add_argument("--gate-bias", type=float, default=-2.0)
    return parser


def qk_scale_initial_value(sequence_length: int) -> float:
    """Return the QKNorm paper's module-level learnable scale initialization."""
    if sequence_length < 2:
        raise ValueError("QK-Norm sequence length must be at least 2")
    return math.log2(sequence_length**2 - sequence_length)


def local_length_percentile(series_files: Iterable[str | Path], percentile: float = 97.5) -> int:
    """Calculate ceil(P97.5) over Local hint lengths in the selected training series."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    lengths: list[int] = []
    for path in series_files:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for window in payload["windows"]:
            start = int(window["window_range"]["start"])
            end = int(window["window_range"]["end"])
            if end <= start:
                raise ValueError(f"invalid window range in {path}")
            lengths.append(end - start)
    if not lengths:
        raise ValueError("cannot initialize QK-Norm without training Local lengths")
    return int(math.ceil(float(np.percentile(lengths, percentile, method="linear"))))


def configure_llm_architecture(
    llm_config,
    *,
    variant: str,
    qk_norm_seq_len: int | None,
    gate_bias: float = -2.0,
) -> dict:
    """Apply one factorial architecture variant and return auditable metadata."""
    if variant not in ARCHITECTURE_VARIANTS:
        raise ValueError(f"unknown architecture variant: {variant}")
    flags = dict(ARCHITECTURE_VARIANTS[variant])
    llm_config.architecture_variant = variant
    if flags["qk_norm"]:
        if qk_norm_seq_len is None:
            raise ValueError("QK-Norm variants require qk_norm_seq_len")
        qk_norm_seq_len = int(qk_norm_seq_len)
        qk_initial = qk_scale_initial_value(qk_norm_seq_len)
    else:
        qk_norm_seq_len = None
        qk_initial = None
    llm_config.qk_norm = flags["qk_norm"]
    llm_config.continuous_bypass = flags["continuous_bypass"]
    llm_config.direct_task_prompt = flags["direct_task_prompt"]
    llm_config.qk_norm_seq_len = qk_norm_seq_len
    llm_config.gate_bias = float(gate_bias)
    return {
        "variant": variant,
        **flags,
        "qk_norm_seq_len": qk_norm_seq_len,
        "qk_scale_initial": qk_initial,
        "gate_bias": float(gate_bias),
        "task_prompt_tokens": int(getattr(llm_config, "num_fixed_tokens", 30)),
    }


def audit_architecture_checkpoint(payload: dict, expected_variant: str | None = None) -> dict:
    """Audit metadata and active Perceiver parameters without loading the LLM."""
    metadata = payload.get("reproduction_meta", {}).get("architecture")
    if not metadata:
        raise ValueError("checkpoint is missing architecture reproduction metadata")
    variant = metadata.get("variant")
    if variant not in ARCHITECTURE_VARIANTS:
        raise ValueError(f"unknown checkpoint architecture variant: {variant}")
    if expected_variant is not None and variant != expected_variant:
        raise ValueError(f"checkpoint variant {variant} != expected {expected_variant}")
    flags = ARCHITECTURE_VARIANTS[variant]
    for name, expected in flags.items():
        if bool(metadata.get(name)) != expected:
            raise ValueError(f"metadata flag {name} does not match variant {variant}")

    model_state = payload.get("model_state_dict", payload)
    state = model_state.get("moirai_trainable")
    if state is None:
        prefix = "axis.perceiver."
        state = {
            name[len(prefix):]: value
            for name, value in model_state.items()
            if name.startswith(prefix)
        }
    if not state:
        raise ValueError("checkpoint has no Perceiver state")

    def require(name: str, present: bool) -> None:
        actual = name in state
        if actual != present:
            relation = "missing" if present else "unexpected"
            raise ValueError(f"{relation} Perceiver parameter: {name}")

    require("local_attention.qk_scale", flags["qk_norm"])
    bypass_names = (
        "continuous_proj.weight",
        "continuous_proj.bias",
        "gate_proj.weight",
        "gate_proj.bias",
        "fusion_norm.weight",
        "fusion_norm.bias",
    )
    for name in bypass_names:
        require(name, flags["continuous_bypass"])
    require("task_prompt_embeddings", flags["direct_task_prompt"])
    require("fix_prompt_embeddings", not flags["direct_task_prompt"])

    if flags["qk_norm"]:
        qk_length = metadata.get("qk_norm_seq_len")
        if not isinstance(qk_length, int) or qk_length < 2:
            raise ValueError("QK-Norm checkpoint has invalid sequence-length metadata")
        if state["local_attention.qk_scale"].numel() != 1:
            raise ValueError("QK-Norm scale must be one module-shared scalar")

    prompt_name = (
        "task_prompt_embeddings"
        if flags["direct_task_prompt"]
        else "fix_prompt_embeddings"
    )
    prompt_shape = tuple(state[prompt_name].shape)
    if len(prompt_shape) != 3 or prompt_shape[0] != 1:
        raise ValueError("task/fixed prompt tensor must have shape [1, K, d]")
    expected_tokens = int(metadata.get("task_prompt_tokens", prompt_shape[1]))
    if prompt_shape[1] != expected_tokens:
        raise ValueError("prompt token count does not match checkpoint metadata")

    return {
        "ok": True,
        "variant": variant,
        **flags,
        "qk_norm_seq_len": metadata.get("qk_norm_seq_len"),
        "gate_bias": metadata.get("gate_bias"),
        "task_prompt_tokens": prompt_shape[1],
        "perceiver_parameters": sum(value.numel() for value in state.values()),
    }

