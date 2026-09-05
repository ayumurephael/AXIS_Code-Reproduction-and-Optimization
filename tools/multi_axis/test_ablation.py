from __future__ import annotations

from types import SimpleNamespace

import torch

from src.models.MultiAXIS.ablation import ABLATION_VARIANTS, ablation_spec
from src.models.MultiAXIS.model import MultiAxisForConditionalGeneration, MultiAxisHintTuner
from src.models.MultiAXIS.prompting import (
    CHANNEL_TOKEN,
    FIXED_TOKEN,
    JOINT_TOKEN,
    STEP_TOKEN,
    MultiAxisPromptBuilder,
    TokenizedPrompts,
)
from tools.multi_axis.test_multiaxis import FakeProcessor


EXPECTED_FLAGS = {
    "multi_axis": (1, 1, 1, 1, 1, 1, 1, 1),
    "observable_only": (1, 1, 1, 0, 0, 0, 0, 1),
    "contextual_only": (0, 0, 0, 1, 1, 1, 1, 1),
    "wo_visual": (0, 1, 1, 1, 1, 1, 1, 1),
    "wo_numeric": (1, 0, 1, 1, 1, 1, 1, 1),
    "wo_scale_calibration": (1, 1, 0, 1, 1, 1, 1, 1),
    "wo_step": (1, 1, 1, 0, 1, 1, 1, 1),
    "wo_channel": (1, 1, 1, 1, 0, 1, 1, 1),
    "wo_joint": (1, 1, 1, 1, 1, 0, 0, 1),
    "wo_question_conditioning": (1, 1, 1, 1, 1, 1, 0, 1),
    "wo_task_prior": (1, 1, 1, 1, 1, 1, 1, 0),
}


def _builder(*, use_images: bool = False) -> MultiAxisPromptBuilder:
    return MultiAxisPromptBuilder(
        FakeProcessor(),
        fixed_tokens=3,
        max_context_tokens=4096,
        use_images=use_images,
    )


def _text(builder: MultiAxisPromptBuilder, variant: str) -> str:
    return builder.build_text(
        "What happened in ch0?",
        42,
        44,
        [[135, -7], [4, 9]],
        [1.25, -2.0],
        [0.5, 3.0],
        ["ch_0", "ch_1"],
        "OE",
        include_visual=False,
        ablation_variant=variant,
    )


def test_ablation_matrix_matches_predeclared_study() -> None:
    assert set(ABLATION_VARIANTS) == set(EXPECTED_FLAGS)
    for name, expected in EXPECTED_FLAGS.items():
        spec = ablation_spec(name)
        actual = tuple(
            int(value)
            for value in (
                spec.visual,
                spec.numeric,
                spec.scale_calibration,
                spec.step,
                spec.channel,
                spec.joint,
                spec.question_conditioning,
                spec.task_prior,
            )
        )
        assert actual == expected


def test_every_ablation_removes_placeholders_instead_of_masking() -> None:
    builder = _builder()
    for name in ABLATION_VARIANTS:
        spec = ablation_spec(name)
        text = _text(builder, name)
        assert text.count(STEP_TOKEN) == (4 if spec.step else 0)
        assert text.count(CHANNEL_TOKEN) == (2 if spec.channel else 0)
        assert text.count(JOINT_TOKEN) == (1 if spec.joint else 0)
        assert text.count(FIXED_TOKEN) == (3 if spec.task_prior else 0)
        assert "<MASK>" not in text


def test_contextual_only_retains_addresses_and_removes_observable_evidence() -> None:
    text = _text(_builder(), "contextual_only")
    assert "Global interval: [42, 44)" in text
    assert "Window length: 2" in text
    assert "ch_0:" in text and "t=42: <STEP_HINT>" in text
    for forbidden in (
        "### Visual evidence",
        "value=",
        "mean=",
        "std=",
        "epsilon",
        "raw_value",
        "reconstruction",
    ):
        assert forbidden not in text


def test_wo_numeric_keeps_scale_and_addresses_but_no_local_values() -> None:
    text = _text(_builder(), "wo_numeric")
    assert "### Channel scale calibration" in text
    assert "ch_0: mean=1.25, std=0.5" in text
    assert "t=42: <STEP_HINT>" in text
    assert "value=" not in text


def test_wo_scale_removes_all_scale_calibration_language() -> None:
    text = _text(_builder(), "wo_scale_calibration")
    assert "t=42, value=135 <STEP_HINT>" in text
    for forbidden in ("mean=", "std=", "epsilon", "raw_value", "raw-scale"):
        assert forbidden not in text


def test_wo_question_conditioning_uses_ordinary_joint_title() -> None:
    text = _text(_builder(), "wo_question_conditioning")
    assert "### Full-window Joint-Local evidence" in text
    assert "Question-conditioned" not in text
    assert "question-conditioned" not in text


def test_visual_off_variant_does_not_open_or_process_supplied_image_path() -> None:
    builder = _builder(use_images=True)
    tokenized = builder.tokenize(
        ["What happened?"],
        [(42, 44)],
        [[[135, -7], [4, 9]]],
        [[1.25, -2.0]],
        [[0.5, 3.0]],
        [["ch_0", "ch_1"]],
        ["OE"],
        ["Z:/this/file/must/not/be/opened.png"],
        ablation_variant="wo_visual",
    )
    assert tokenized.visual_token_counts == [0]
    assert tokenized.original_image_sizes == [None]
    assert "pixel_values" not in tokenized.model_inputs
    assert tokenized.ablation_variant == "wo_visual"


def test_joint_pooling_and_question_conditioning_are_independent_switches() -> None:
    config = SimpleNamespace(
        num_prototypes=7,
        prototype_heads=4,
        fixed_tokens=3,
        channel_heads=4,
        joint_heads=2,
        representation_epsilon=1e-6,
        require_flash_attention=False,
        ablation_phase="D",
    )
    tuner = MultiAxisHintTuner(13, 16, 8, config)
    with torch.no_grad():
        tuner.question_projection.weight.fill_(0.1)
    local = torch.randn(1, 4, 2, 8)
    logits = torch.randn(1, 4, 2, 2)
    words = torch.randn(13, 16)
    question = torch.randn(1, 16)
    unconditioned = tuner(
        local,
        logits,
        [(1, 3)],
        [2],
        None,
        words,
        question_conditioning=False,
    )
    conditioned = tuner(local, logits, [(1, 3)], [2], question, words)
    no_joint = tuner(
        local,
        logits,
        [(1, 3)],
        [2],
        None,
        words,
        include_joint=False,
        question_conditioning=False,
    )
    assert unconditioned[2] is not None
    assert conditioned[2] is not None
    assert not torch.allclose(unconditioned[2], conditioned[2])
    assert no_joint[2] is None


def test_embedding_hook_skips_non_injected_hint_families() -> None:
    input_ids = torch.tensor([[3, 4]])
    tokenized = TokenizedPrompts(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        },
        labels=None,
        prompt_lengths=[2],
        step_positions=[torch.empty(0, dtype=torch.long)],
        channel_positions=[torch.empty(0, dtype=torch.long)],
        joint_positions=[torch.empty(0, dtype=torch.long)],
        fixed_positions=[torch.tensor([1])],
        visual_token_counts=[0],
        original_image_sizes=[None],
        padding_side="right",
        ablation_variant="observable_only",
    )
    dummy = SimpleNamespace(
        config=SimpleNamespace(vision=SimpleNamespace(enabled=True))
    )
    step = [torch.randn(4, 5)]
    channel = [torch.randn(2, 5)]
    fixed = torch.full((1, 1, 5), 7.0)
    model_inputs, _labels, hook = (
        MultiAxisForConditionalGeneration._model_inputs_and_hint_hook(
            dummy,
            tokenized,
            step,
            channel,
            None,
            fixed,
            torch.device("cpu"),
        )
    )
    base = torch.zeros(1, 2, 5)
    replaced = hook(None, (model_inputs["input_ids"],), base)
    assert torch.all(replaced[0, 0] == 0)
    assert torch.all(replaced[0, 1] == 7)
