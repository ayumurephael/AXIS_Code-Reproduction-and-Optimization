from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN3_VL_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
SUPPORTED_LLM_IDS = (DEEPSEEK_MODEL_ID, QWEN35_MODEL_ID, QWEN3_VL_MODEL_ID)
FORMAL_DISTRIBUTED_PROFILES = {
    (4, 1, 8, 20): 32,
    (4, 2, 4, 20): 32,
    (4, 4, 2, 20): 32,
    (5, 1, 6, 40): 30,
    (8, 1, 4, 25): 32,
    (8, 2, 2, 25): 32,
    (8, 4, 1, 25): 32,
    (32, 1, 1, 25): 32,
}


@dataclass
class TimeRCDConfig:
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 16
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    checkpoint_num_features: int = 20


@dataclass
class HintConfig:
    num_prototypes: int = 1024
    prototype_heads: int = 16
    fixed_tokens: int = 30
    joint_heads: int = 4
    representation_epsilon: float = 1e-6
    window_epsilon: float = 1e-5
    window_scale: int = 100
    require_flash_attention: bool = True
    ablation_phase: str = "D"


@dataclass
class LLMConfig:
    model_name: str = DEEPSEEK_MODEL_ID
    torch_dtype: str = "bfloat16"
    max_context_tokens: int = 32768
    trust_remote_code: bool = True
    attention_implementation: str = "flash_attention_2"
    gradient_checkpointing: bool = True
    use_cache_during_training: bool = False

    def validate(self) -> None:
        if self.model_name not in SUPPORTED_LLM_IDS:
            raise ValueError(
                f"Unsupported LLM {self.model_name!r}; supported values are {SUPPORTED_LLM_IDS}."
            )
        if self.torch_dtype != "bfloat16":
            raise ValueError("Formal Multi-AXIS training requires bfloat16.")
        if self.max_context_tokens < 32768:
            raise ValueError(
                "The documented architecture requires at least a 32768-token context."
            )


@dataclass
class VisionConfig:
    """Native Qwen3-VL image path and the official processor pixel budget."""

    enabled: bool = False
    min_pixels: int = 65_536
    max_pixels: int = 16_777_216
    renderer_dpi: int = 600
    renderer_version: str = "multi-axis-vl-render-v1"
    require_mm_token_type_ids: bool = True

    def validate(self, model_name: str) -> None:
        if self.enabled and model_name != QWEN3_VL_MODEL_ID:
            raise ValueError(
                "Image-enabled Multi-AXIS requires Qwen/Qwen3-VL-8B-Instruct."
            )
        if self.min_pixels != 65_536 or self.max_pixels != 16_777_216:
            raise ValueError(
                "The formal VLM run uses the official Qwen3-VL processor budget: "
                "65,536 through 16,777,216 pixels."
            )
        if self.renderer_dpi != 600:
            raise ValueError("The specified VLM renderer requires dpi=600.")


@dataclass
class TrainingConfig:
    epochs: int = 25
    seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    micro_batch_size: int = 2
    accumulation_steps: int = 2
    expected_world_size: int = 8
    expected_nodes: int = 1
    num_workers: int = 2
    validation_fraction: float = 0.10
    select_by: str = "validation_answer_nll"
    early_stopping: bool = False

    @property
    def effective_batch_size(self) -> int:
        return (
            self.micro_batch_size * self.accumulation_steps * self.expected_world_size
        )

    def validate(self) -> None:
        if self.seed != 42 or self.validation_fraction != 0.10:
            raise ValueError("The confirmed grouped split is 90/10 with seed 42.")
        if self.early_stopping:
            raise ValueError(
                "The confirmed protocol runs all configured epochs without early stopping."
            )
        if self.select_by != "validation_answer_nll":
            raise ValueError(
                "Checkpoint selection must use validation answer-token NLL only."
            )
        if self.expected_nodes <= 0 or self.expected_world_size % self.expected_nodes:
            raise ValueError(
                "expected_world_size must be divisible by a positive expected_nodes."
            )
        profile = (
            self.expected_world_size,
            self.micro_batch_size,
            self.accumulation_steps,
            self.epochs,
        )
        if profile not in FORMAL_DISTRIBUTED_PROFILES:
            supported = ", ".join(
                f"{world_size} GPUs / micro {micro_batch} / accumulation {accumulation} "
                f"/ {epochs} epochs / effective batch {batch_size}"
                for (
                    world_size,
                    micro_batch,
                    accumulation,
                    epochs,
                ), batch_size in sorted(FORMAL_DISTRIBUTED_PROFILES.items())
            )
            raise ValueError(
                f"Unsupported formal distributed profile; use {supported}."
            )
        if self.effective_batch_size != FORMAL_DISTRIBUTED_PROFILES[profile]:
            raise ValueError(
                "Configured effective batch size does not match its formal profile."
            )


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1000
    num_beams: int = 5
    do_sample: bool = False
    repetition_penalty: float = 1.15
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0


@dataclass
class MultiAxisConfig:
    timercd: TimeRCDConfig = field(default_factory=TimeRCDConfig)
    hints: HintConfig = field(default_factory=HintConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    schema_version: str = "multi-axis-1.0"

    def validate(self) -> None:
        self.llm.validate()
        self.vision.validate(self.llm.model_name)
        self.training.validate()
        if self.timercd.d_proj % self.hints.joint_heads:
            raise ValueError("TimeRCD d_proj must be divisible by joint_heads.")
        if self.hints.fixed_tokens != 30 or self.hints.num_prototypes != 1024:
            raise ValueError(
                "The specified architecture fixes K_F=30 and N_proto=1024."
            )

        if self.hints.ablation_phase not in {"A", "B", "C", "D"}:
            raise ValueError("Ablation phase must be A, B, C, or D (full model).")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "MultiAxisConfig":
        return cls(
            timercd=TimeRCDConfig(**raw.get("timercd", {})),
            hints=HintConfig(**raw.get("hints", {})),
            llm=LLMConfig(**raw.get("llm", {})),
            vision=VisionConfig(**raw.get("vision", {})),
            training=TrainingConfig(**raw.get("training", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            schema_version=raw.get("schema_version", "multi-axis-1.0"),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "MultiAxisConfig":
        config = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config
