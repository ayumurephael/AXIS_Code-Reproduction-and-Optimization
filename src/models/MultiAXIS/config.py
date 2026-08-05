from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
QWEN35_MODEL_ID = "Qwen/Qwen3.5-9B"
SUPPORTED_LLM_IDS = (DEEPSEEK_MODEL_ID, QWEN35_MODEL_ID)


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
    prototype_heads: int = 8
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
            raise ValueError("The documented architecture requires at least a 32768-token context.")


@dataclass
class TrainingConfig:
    epochs: int = 40
    seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_clip_norm: float = 1.0
    micro_batch_size: int = 1
    accumulation_steps: int = 6
    expected_world_size: int = 5
    num_workers: int = 2
    validation_fraction: float = 0.10
    select_by: str = "validation_answer_nll"
    early_stopping: bool = False

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps * self.expected_world_size

    def validate(self) -> None:
        if self.epochs != 40:
            raise ValueError("The formal experiment is fixed to exactly 40 epochs.")
        if self.seed != 42 or self.validation_fraction != 0.10:
            raise ValueError("The confirmed grouped split is 90/10 with seed 42.")
        if self.early_stopping:
            raise ValueError("The confirmed protocol runs all 40 epochs without early stopping.")
        if self.select_by != "validation_answer_nll":
            raise ValueError("Checkpoint selection must use validation answer-token NLL only.")


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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    schema_version: str = "multi-axis-1.0"

    def validate(self) -> None:
        self.llm.validate()
        self.training.validate()
        if self.timercd.d_proj % self.hints.joint_heads:
            raise ValueError("TimeRCD d_proj must be divisible by joint_heads.")
        if self.hints.fixed_tokens != 30 or self.hints.num_prototypes != 1024:
            raise ValueError("The specified architecture fixes K_F=30 and N_proto=1024.")

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
            training=TrainingConfig(**raw.get("training", {})),
            generation=GenerationConfig(**raw.get("generation", {})),
            schema_version=raw.get("schema_version", "multi-axis-1.0"),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "MultiAxisConfig":
        config = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config
