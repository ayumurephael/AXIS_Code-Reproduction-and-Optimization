"""Prompt/boundary protocol for the full Phase-II Treatment retraining.

This module is intentionally opt-in. Released AXIS checkpoints and the
completed two-epoch loss_e2e_0723 experiment retain the author's original
``question + EOS + "Answer: ..."`` construction. New checkpoints whose
manifest declares :data:`SIMPLIFIED_FINAL_ANSWER_V1` use one shared
train/inference prefix and supervise only the answer continuation plus the
terminal EOS.
"""
from __future__ import annotations

import types
from typing import List, Sequence

import torch


SIMPLIFIED_FINAL_ANSWER_V1 = "simplified_1txt_final_answer_boundary_v1"
ANSWER_CONTINUATION_PREFIX = " "


def build_simplified_prompt(
    *,
    question: str,
    time_series: torch.Tensor,
    start_index: int,
    end_index: int,
    num_local_hint_tokens: int,
    num_fixed_hint_tokens: int,
    ablation_mode: str | None = None,
) -> str:
    """Render exactly the user-approved ``1.txt`` prompt."""
    local_hint_tokens = (
        ""
        if ablation_mode == "wo_local_hint"
        else "<|local_hint|>" * num_local_hint_tokens
    )
    fixed_hint_tokens = (
        ""
        if ablation_mode == "wo_fixed_hint"
        else "<|fixed_hint|>" * num_fixed_hint_tokens
    )
    values = ", ".join(
        f"{(value * 100):.0f}"
        for value in time_series[start_index:end_index].tolist()
    )
    if ablation_mode == "wo_windows":
        values = "(removed)"
    return (
        "You are an expert time-series anomaly analyst. Analyze the provided "
        "data and produce one precise, evidence-grounded answer to the "
        "question.\n\n"
        "### Time Series Data\n"
        f"- **Window:** Steps {start_index} to {end_index}\n"
        f"- **Values (scaled by 100):** {values}\n\n"
        "### Contextual Hints\n"
        "**Per-Step Analysis:** \n"
        f"{local_hint_tokens}\n\n"
        "### Learned Task Guidance/Shared Task-Control Tokens\n"
        f" {fixed_hint_tokens}\n\n"
        "### Question\n"
        f"{question}\n\n"
        "### Final Answer\n"
        "Answer:"
    )


def answer_continuations(answers: Sequence[str]) -> list[str]:
    """Return targets that render natural ``Answer: <gold>`` text."""
    return [ANSWER_CONTINUATION_PREFIX + str(answer) for answer in answers]


def assert_additive_tokenization(
    tokenizer,
    prompts: Sequence[str],
    continuations: Sequence[str],
) -> None:
    """Fail closed if separate prefix/target tokenization changes joint IDs."""
    for row, (prompt, continuation) in enumerate(zip(prompts, continuations)):
        prefix_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        continuation_ids = tokenizer(
            continuation,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        joint_ids = tokenizer(
            prompt + continuation,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        if list(prefix_ids) + list(continuation_ids) != list(joint_ids):
            raise RuntimeError(
                "prompt/answer tokenization is not additive at row "
                f"{row}; the answer boundary would be misaligned"
            )


def _generate_input_ids_and_labels(
    self,
    questions: List[str],
    answers: List[str],
    time_series: torch.Tensor,
    start_indices: List[int],
    end_indices: List[int],
    ablation_mode: str | None = None,
):
    batch_size = len(questions)
    if not (
        len(answers)
        == len(start_indices)
        == len(end_indices)
        == time_series.size(0)
        == batch_size
    ):
        raise ValueError("prompt inputs must have one row per question")

    prompts = [
        build_simplified_prompt(
            question=questions[row],
            time_series=time_series[row],
            start_index=start_indices[row],
            end_index=end_indices[row],
            num_local_hint_tokens=end_indices[row] - start_indices[row],
            num_fixed_hint_tokens=self.num_fixed_tokens,
            ablation_mode=ablation_mode,
        )
        for row in range(batch_size)
    ]
    continuations = answer_continuations(answers)
    assert_additive_tokenization(self.tokenizer, prompts, continuations)

    prefix_input = self.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    answer_input = self.tokenizer(
        continuations,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prefix_width = int(prefix_input["input_ids"].shape[1])
    answer_width = int(answer_input["input_ids"].shape[1])
    total_width = prefix_width + answer_width + 1
    model_max_length = int(self.tokenizer.model_max_length)
    if total_width > model_max_length:
        raise RuntimeError(
            f"untruncated AXIS input has {total_width} tokens, exceeding "
            f"tokenizer.model_max_length={model_max_length}"
        )

    eos = torch.full(
        (batch_size, 1),
        int(self.tokenizer.eos_token_id),
        dtype=prefix_input["input_ids"].dtype,
    )
    full_input_ids = torch.cat(
        [prefix_input["input_ids"], answer_input["input_ids"], eos],
        dim=1,
    )
    full_attention_mask = torch.cat(
        [
            prefix_input["attention_mask"],
            answer_input["attention_mask"],
            torch.ones_like(eos),
        ],
        dim=1,
    )
    answer_labels = answer_input["input_ids"].clone()
    answer_labels[answer_input["attention_mask"] == 0] = -100
    full_labels = torch.cat(
        [
            torch.full_like(prefix_input["input_ids"], -100),
            answer_labels,
            eos,
        ],
        dim=1,
    )
    self._loss_e2e_prompt_layout = {
        "protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "answer_block_start": prefix_width,
        "answer_width": answer_width,
        "terminal_eos_index": total_width - 1,
        "answer_prefix": ANSWER_CONTINUATION_PREFIX,
        "prefix_rows": list(prompts),
        "prefix_input_ids": prefix_input["input_ids"].detach().cpu(),
        "prefix_attention_mask": prefix_input["attention_mask"].detach().cpu(),
        "answer_offsets": answer_input["offset_mapping"].detach().cpu(),
    }
    return full_input_ids, full_attention_mask, full_labels, prefix_width


def install_simplified_prompt_runtime(axis: torch.nn.Module) -> None:
    """Install the opt-in boundary-correct prompt implementation."""
    if getattr(axis, "prompt_protocol", None) == SIMPLIFIED_FINAL_ANSWER_V1:
        return
    if not hasattr(axis, "_original_generate_input_ids_and_labels"):
        axis._original_generate_input_ids_and_labels = (
            axis.generate_input_ids_and_labels
        )
    axis.generate_input_ids_and_labels = types.MethodType(
        _generate_input_ids_and_labels,
        axis,
    )
    axis.prompt_protocol = SIMPLIFIED_FINAL_ANSWER_V1


def prompt_protocol_from_checkpoint(payload: dict) -> str | None:
    meta = payload.get("reproduction_meta", {})
    if not isinstance(meta, dict):
        return None
    return meta.get("prompt_protocol")
