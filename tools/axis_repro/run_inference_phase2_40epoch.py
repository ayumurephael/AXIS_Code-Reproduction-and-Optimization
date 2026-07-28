"""Inference entry point for the boundary-correct 40-epoch checkpoint."""
from __future__ import annotations

from . import run_inference as implementation
from .loss_e2e_40epoch_runtime import load_phase2_40epoch_checkpoint


implementation.load_axis_checkpoint = load_phase2_40epoch_checkpoint


if __name__ == "__main__":
    implementation.main()
