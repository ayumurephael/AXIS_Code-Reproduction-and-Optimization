"""Runtime for the full 40-epoch Phase-II Treatment retraining.

The completed two-epoch experiment remains immutable.  This adapter combines
its cached-F0 Treatment objective with the opt-in, boundary-correct ``1.txt``
prompt protocol.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from torch import nn

from .loss_e2e import (
    ANSWER_PREFIX,
    ERROR_ANSWER,
    build_full_segment_ids,
    objective_from_hidden,
)
from .loss_e2e_runtime import (
    fixed_hint_checkpoint_state,
    initialize_fixed_hint_reference,
    install_fixed_hint_runtime,
    set_fixed_hint_reference,
)
from .model_utils import load_axis_checkpoint
from .prompt_boundary import (
    ANSWER_CONTINUATION_PREFIX,
    SIMPLIFIED_FINAL_ANSWER_V1,
    answer_continuations,
    install_simplified_prompt_runtime,
    prompt_protocol_from_checkpoint,
)


def load_phase2_40epoch_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict:
    payload = load_axis_checkpoint(model, path, strict=strict)
    protocol = prompt_protocol_from_checkpoint(payload)
    if protocol != SIMPLIFIED_FINAL_ANSWER_V1:
        raise ValueError(
            "40-epoch checkpoint must declare the simplified boundary "
            f"protocol, got {protocol!r}"
        )
    install_simplified_prompt_runtime(model.axis)
    state = payload.get("model_state_dict", payload)
    reference = state.get("fixed_hint_reference")
    if reference is None:
        raise ValueError("40-epoch Treatment checkpoint has no cached F0")
    set_fixed_hint_reference(model.axis, reference)
    return payload


def _answer_encoding(axis: nn.Module, answers: Sequence[str]):
    if not getattr(axis.tokenizer, "is_fast", False):
        raise RuntimeError(
            "segmented loss requires a fast tokenizer with offset mapping"
        )
    return axis.tokenizer(
        answer_continuations(answers),
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )


def _offsets_for_existing_segment_mapper(offsets: torch.Tensor) -> torch.Tensor:
    """Translate the one-space continuation offsets to the legacy mapper.

    ``build_full_segment_ids`` deliberately remains unchanged for the completed
    two-epoch experiment.  It assumes the literal ``"Answer: "`` prefix.  The
    new target has only one leading space, so shifting all nonempty offsets by
    ``len("Answer: ") - 1`` gives exactly the same answer-character coordinate
    system without changing token IDs.
    """
    adjusted = offsets.clone()
    nonempty = adjusted[..., 1] > adjusted[..., 0]
    delta = len(ANSWER_PREFIX) - len(ANSWER_CONTINUATION_PREFIX)
    adjusted[..., 0][nonempty] += delta
    adjusted[..., 1][nonempty] += delta
    return adjusted


def axis_objective_forward(
    *,
    axis: nn.Module,
    local_embeddings: torch.Tensor,
    time_series: torch.Tensor,
    questions: Sequence[str],
    answers: Sequence[str],
    start_indices: Sequence[int],
    end_indices: Sequence[int],
    question_types: Sequence[str],
    segment_alpha: float,
    valid_rows: Sequence[bool],
    loss_chunk_size: int,
):
    input_ids, attention_mask, labels, question_length = (
        axis.generate_input_ids_and_labels(
            list(questions),
            list(answers),
            time_series,
            list(start_indices),
            list(end_indices),
        )
    )
    layout = getattr(axis, "_loss_e2e_prompt_layout", None)
    if not isinstance(layout, dict):
        raise RuntimeError("simplified prompt runtime did not record layout")
    if layout.get("protocol") != SIMPLIFIED_FINAL_ANSWER_V1:
        raise RuntimeError("unexpected prompt protocol in 40-epoch runtime")
    if int(layout["answer_block_start"]) != int(question_length):
        raise RuntimeError("train/inference prefix width drifted")

    device = next(axis.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    input_embeddings = axis.get_hint_embeddings(
        input_ids,
        local_embeddings,
        list(start_indices),
        list(end_indices),
    )

    encoding = _answer_encoding(axis, answers)
    if int(layout["answer_width"]) != int(encoding["input_ids"].size(1)):
        raise RuntimeError("answer width differs between full and offset encodings")
    segment_ids, _ = build_full_segment_ids(
        full_input_ids=input_ids,
        answer_input_ids=encoding["input_ids"].to(device),
        answer_offsets=_offsets_for_existing_segment_mapper(
            encoding["offset_mapping"]
        ),
        answer_block_start=int(layout["answer_block_start"]),
        answers=answers,
        question_types=question_types,
        terminal_eos_index=int(layout["terminal_eos_index"]),
        terminal_eos_token_id=axis.tokenizer.eos_token_id,
    )

    hidden = axis.model.model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state
    return objective_from_hidden(
        hidden=hidden,
        labels=labels,
        lm_head=axis.model.lm_head,
        mode="treatment",
        valid_rows=torch.tensor(valid_rows, device=device),
        segment_ids=segment_ids,
        question_types=question_types,
        alpha=segment_alpha,
        chunk_size=loss_chunk_size,
    )


class Phase2TreatmentObjectiveModel(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        *,
        segment_alpha: float,
        loss_chunk_size: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.segment_alpha = segment_alpha
        self.loss_chunk_size = loss_chunk_size

    def forward(
        self,
        padded_sequences: torch.Tensor,
        attention_masks: torch.Tensor,
        questions: Sequence[str],
        answers: Sequence[str],
        start_indices: Sequence[int],
        end_indices: Sequence[int],
        question_types: Sequence[str],
        valid_rows: Optional[Sequence[bool]] = None,
    ):
        if valid_rows is None:
            valid_rows = [
                answer.strip() != ERROR_ANSWER
                for answer in answers
            ]
        with torch.no_grad():
            local_embeddings = self.base.ts_pretrain_model(
                padded_sequences,
                mask=attention_masks,
            )
        return axis_objective_forward(
            axis=self.base.axis,
            local_embeddings=local_embeddings,
            time_series=padded_sequences,
            questions=questions,
            answers=answers,
            start_indices=start_indices,
            end_indices=end_indices,
            question_types=question_types,
            segment_alpha=self.segment_alpha,
            valid_rows=valid_rows,
            loss_chunk_size=self.loss_chunk_size,
        )


__all__ = [
    "Phase2TreatmentObjectiveModel",
    "axis_objective_forward",
    "fixed_hint_checkpoint_state",
    "initialize_fixed_hint_reference",
    "install_fixed_hint_runtime",
    "load_phase2_40epoch_checkpoint",
    "set_fixed_hint_reference",
]
