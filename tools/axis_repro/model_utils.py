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


def build_model():
    from experiments.configs.axis_config import default_config
    from src.models.AXIS.AXIS import AXISCombinedModel

    default_config.enable_ts_train = False
    return AXISCombinedModel(default_config)


def load_axis_checkpoint(model, path: str | Path, strict: bool = True) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if "ts_pretrain_model" in state and "moirai_trainable" in state:
        model.ts_pretrain_model.load_state_dict(state["ts_pretrain_model"], strict=strict)
        model.axis.perceiver.load_state_dict(state["moirai_trainable"], strict=strict)
    else:
        model.load_state_dict(state, strict=strict)
    return payload


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
