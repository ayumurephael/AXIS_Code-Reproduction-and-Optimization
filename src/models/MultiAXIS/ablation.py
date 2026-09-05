from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple


FULL_VARIANT = "multi_axis"
ABLATION_VARIANTS: Tuple[str, ...] = (
    FULL_VARIANT,
    "observable_only",
    "contextual_only",
    "wo_visual",
    "wo_numeric",
    "wo_scale_calibration",
    "wo_step",
    "wo_channel",
    "wo_joint",
    "wo_question_conditioning",
    "wo_task_prior",
)

DISPLAY_NAMES = {
    "multi_axis": "Multi-AXIS",
    "observable_only": "Observable only",
    "contextual_only": "Contextual only",
    "wo_visual": "w/o Visual",
    "wo_numeric": "w/o Numeric",
    "wo_scale_calibration": "w/o Scale Calibration",
    "wo_step": "w/o Step",
    "wo_channel": "w/o Channel",
    "wo_joint": "w/o Joint",
    "wo_question_conditioning": "w/o Question Conditioning",
    "wo_task_prior": "w/o Task Prior",
}


@dataclass(frozen=True)
class AblationSpec:
    name: str
    visual: bool
    numeric: bool
    scale_calibration: bool
    step: bool
    channel: bool
    joint: bool
    question_conditioning: bool
    task_prior: bool

    def __post_init__(self) -> None:
        if self.name not in ABLATION_VARIANTS:
            raise ValueError(f"Unknown ablation variant: {self.name}")
        if self.question_conditioning and not self.joint:
            raise ValueError("Question conditioning requires Joint-Local evidence")

    @property
    def contextual(self) -> bool:
        return self.step or self.channel or self.joint

    @property
    def is_full(self) -> bool:
        return self.name == FULL_VARIANT

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


_SPECS = {
    FULL_VARIANT: AblationSpec(FULL_VARIANT, True, True, True, True, True, True, True, True),
    "observable_only": AblationSpec("observable_only", True, True, True, False, False, False, False, True),
    "contextual_only": AblationSpec("contextual_only", False, False, False, True, True, True, True, True),
    "wo_visual": AblationSpec("wo_visual", False, True, True, True, True, True, True, True),
    "wo_numeric": AblationSpec("wo_numeric", True, False, True, True, True, True, True, True),
    "wo_scale_calibration": AblationSpec("wo_scale_calibration", True, True, False, True, True, True, True, True),
    "wo_step": AblationSpec("wo_step", True, True, True, False, True, True, True, True),
    "wo_channel": AblationSpec("wo_channel", True, True, True, True, False, True, True, True),
    "wo_joint": AblationSpec("wo_joint", True, True, True, True, True, False, False, True),
    "wo_question_conditioning": AblationSpec("wo_question_conditioning", True, True, True, True, True, True, False, True),
    "wo_task_prior": AblationSpec("wo_task_prior", True, True, True, True, True, True, True, False),
}


def ablation_spec(name: str = FULL_VARIANT) -> AblationSpec:
    try:
        return _SPECS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ablation variant {name!r}; expected one of {ABLATION_VARIANTS}"
        ) from exc


def ablation_table() -> Dict[str, Dict[str, object]]:
    return {name: _SPECS[name].to_dict() for name in ABLATION_VARIANTS}
