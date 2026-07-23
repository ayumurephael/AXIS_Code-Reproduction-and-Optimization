"""Inference entry point that restores the optional cached F0 tensor."""
from __future__ import annotations

from . import run_inference as implementation
from .loss_e2e_runtime import load_loss_e2e_checkpoint


implementation.load_axis_checkpoint = load_loss_e2e_checkpoint


if __name__ == "__main__":
    implementation.main()
