from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import torch


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_model(
    architecture_variant: str = "loss_only",
    qk_norm_seq_len: int | None = None,
    gate_bias: float = -2.0,
):
    import copy

    from experiments.configs.axis_config import default_config
    from src.models.AXIS.AXIS import AXISCombinedModel
    from tools.axis_repro.architecture_redesign import configure_llm_architecture

    config = copy.deepcopy(default_config)
    config.enable_ts_train = False
    configure_llm_architecture(
        config.llm_config,
        variant=architecture_variant,
        qk_norm_seq_len=qk_norm_seq_len,
        gate_bias=gate_bias,
    )
    return AXISCombinedModel(config)


def validate_checkpoint_architecture(payload: dict, llm_config) -> None:
    """Fail closed when checkpoint provenance and instantiated architecture differ."""
    expected_variant = getattr(llm_config, "architecture_variant", "loss_only")
    actual = payload.get("reproduction_meta", {}).get("architecture")
    if not actual:
        if expected_variant != "loss_only":
            raise ValueError("architecture checkpoint is missing reproduction metadata")
        return
    if actual.get("variant") != expected_variant:
        raise ValueError(
            f"checkpoint architecture {actual.get('variant')} != requested {expected_variant}"
        )
    expected_length = getattr(llm_config, "qk_norm_seq_len", None)
    if actual.get("qk_norm_seq_len") != expected_length:
        raise ValueError(
            f"checkpoint QK length {actual.get('qk_norm_seq_len')} != requested {expected_length}"
        )
    expected_gate_bias = float(getattr(llm_config, "gate_bias", -2.0))
    actual_gate_bias = float(actual.get("gate_bias", expected_gate_bias))
    if abs(actual_gate_bias - expected_gate_bias) > 1e-12:
        raise ValueError(
            f"checkpoint gate bias {actual_gate_bias} != requested {expected_gate_bias}"
        )

def load_axis_payload(model, payload: dict, strict: bool = True) -> Dict[str, Any]:
    """Load an already-materialized AXIS checkpoint payload into ``model``."""
    validate_checkpoint_architecture(payload, model.axis.config.llm_config)
    state = payload.get("model_state_dict", payload)
    if "ts_pretrain_model" in state and "moirai_trainable" in state:
        model.ts_pretrain_model.load_state_dict(state["ts_pretrain_model"], strict=strict)
        perceiver_state = state["moirai_trainable"]
        # Author Accelerate checkpoints were saved from the enclosing Moirai
        # module and prefix every perceiver key with ``perceiver.``. Formal
        # reproduction checkpoints save the perceiver state_dict directly.
        # Normalize only if all keys have the prefix so mixed/corrupt inputs
        # continue to fail closed under strict loading.
        legacy_prefix = "perceiver."
        if perceiver_state and all(
            key.startswith(legacy_prefix) for key in perceiver_state
        ):
            perceiver_state = {
                key[len(legacy_prefix):]: value
                for key, value in perceiver_state.items()
            }
        model.axis.perceiver.load_state_dict(perceiver_state, strict=strict)
    else:
        model.load_state_dict(state, strict=strict)
    return payload


def load_axis_checkpoint(model, path: str | Path, strict: bool = True) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return load_axis_payload(model, payload, strict=strict)


def load_phase1_fresh_hint(model, path: str | Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    ts_state = state.get("ts_pretrain_model", state)
    model.ts_pretrain_model.load_state_dict(ts_state, strict=True)
    return payload


def freeze_for_phase2(model) -> int:
    # resize_token_embeddings happens after the original LLM freeze; freeze again.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.axis.perceiver.parameters():
        p.requires_grad = True
    model.ts_pretrain_model.eval()
    model.axis.model.eval()
    model.axis.model.config.use_cache = False
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_payload(model, optimizer, epoch: int, global_step: int, meta: dict) -> dict:
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": {
            "ts_pretrain_model": model.ts_pretrain_model.state_dict(),
            "moirai_trainable": model.axis.perceiver.state_dict(),
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "reproduction_meta": meta,
    }
