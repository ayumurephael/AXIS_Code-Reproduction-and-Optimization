"""Multivariate AXIS: frozen TimeRCD + trainable soft-hint bridge + frozen LLM."""

from .config import MultiAxisConfig
from .model import MultiAxisForConditionalGeneration
from .timercd import FrozenTimeRCD

__all__ = ["MultiAxisConfig", "MultiAxisForConditionalGeneration", "FrozenTimeRCD"]
