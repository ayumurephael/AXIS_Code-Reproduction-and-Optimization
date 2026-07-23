"""Runtime adapter for the loss_e2e_0723 continuation experiment.

The baseline model remains source-compatible.  This adapter adds the cached F0
path and the memory-safe objective only to the explicit continuation entry
points, which prevents accidental changes to the repository's original
Phase-I -> Phase-II reproduction commands.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch import nn

from .loss_e2e import (
    ANSWER_PREFIX,
    ERROR_ANSWER,
    build_full_segment_ids,
    objective_from_hidden,
)
from .model_utils import load_axis_checkpoint


FIXED_HINT_KEY = "fixed_hint_reference"


def install_fixed_hint_runtime(axis: nn.Module) -> None:
    if hasattr(axis, "_loss_e2e_original_get_hint_embeddings"):
        return
    axis._loss_e2e_original_get_hint_embeddings = axis.get_hint_embeddings
    axis.register_buffer(FIXED_HINT_KEY, torch.empty(0), persistent=True)
    axis.fixed_hint_policy = "dynamic"

    def get_hint_embeddings(
        self,
        input_ids: torch.Tensor,
        local_embeddings: torch.Tensor,
        start_indices: List[int],
        end_indices: List[int],
    ) -> torch.Tensor:
        if self.fixed_hint_policy == "dynamic":
            return self._loss_e2e_original_get_hint_embeddings(
                input_ids,
                local_embeddings,
                start_indices,
                end_indices,
            )
        reference = getattr(self, FIXED_HINT_KEY)
        if reference.numel() == 0:
            raise RuntimeError("cached fixed hint policy has no F0 tensor")

        input_embeddings = self.model.get_input_embeddings()(input_ids).clone()
        word_embeddings = self.model.get_input_embeddings().weight
        source_embeddings = self.perceiver.get_source_embeddings(word_embeddings)
        for row in range(local_embeddings.shape[0]):
            projected_local = self.perceiver.process_local_embeddings(
                local_embeddings[row],
                source_embeddings,
                start_indices[row],
                end_indices[row],
            )
            current_ids = input_ids[row]
            local_positions = (
                current_ids == self.local_hint_token_id
            ).nonzero(as_tuple=True)[0]
            input_embeddings[row, local_positions] = projected_local[
                :len(local_positions)
            ].to(input_embeddings.dtype)

            fixed_positions = (
                current_ids == self.fixed_hint_token_id
            ).nonzero(as_tuple=True)[0]
            if len(fixed_positions) > reference.size(0):
                raise RuntimeError(
                    f"prompt requests {len(fixed_positions)} fixed tokens, "
                    f"but cached F0 has {reference.size(0)}"
                )
            if len(fixed_positions):
                input_embeddings[row, fixed_positions] = reference[
                    :len(fixed_positions)
                ].to(device=input_embeddings.device, dtype=input_embeddings.dtype)
        return input_embeddings

    axis.get_hint_embeddings = types.MethodType(get_hint_embeddings, axis)


@torch.no_grad()
def initialize_fixed_hint_reference(axis: nn.Module) -> torch.Tensor:
    """Compute F0 once under the caller's autocast context."""
    install_fixed_hint_runtime(axis)
    word_embeddings = axis.model.get_input_embeddings().weight
    source_embeddings = axis.perceiver.get_source_embeddings(word_embeddings)
    reference = axis.perceiver.process_fixed_embeddings(
        source_embeddings,
        axis.num_fixed_tokens,
    )
    reference = reference.to(dtype=word_embeddings.dtype).detach().clone()
    axis.fixed_hint_reference = reference
    axis.fixed_hint_policy = "cached"
    return reference


def set_fixed_hint_reference(axis: nn.Module, reference: torch.Tensor) -> None:
    install_fixed_hint_runtime(axis)
    expected = (axis.num_fixed_tokens, axis.model.config.hidden_size)
    if tuple(reference.shape) != expected:
        raise ValueError(
            f"fixed hint reference has shape {tuple(reference.shape)}, "
            f"expected {expected}"
        )
    axis.fixed_hint_reference = reference.detach().clone()
    axis.fixed_hint_policy = "cached"


def fixed_hint_checkpoint_state(axis: nn.Module) -> Optional[torch.Tensor]:
    if not hasattr(axis, FIXED_HINT_KEY):
        return None
    reference = getattr(axis, FIXED_HINT_KEY)
    if reference.numel() == 0:
        return None
    return reference.detach().cpu()


def load_loss_e2e_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict:
    payload = load_axis_checkpoint(model, path, strict=strict)
    state = payload.get("model_state_dict", payload)
    reference = state.get(FIXED_HINT_KEY) if isinstance(state, dict) else None
    if reference is not None:
        set_fixed_hint_reference(model.axis, reference)
    return payload


def _answer_encoding(axis: nn.Module, answers: Sequence[str]):
    if not getattr(axis.tokenizer, "is_fast", False):
        raise RuntimeError(
            "segmented loss requires a fast tokenizer with offset mapping"
        )
    return axis.tokenizer(
        [f"{ANSWER_PREFIX}{answer}" for answer in answers],
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )


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
    objective_mode: str,
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

    segment_ids = None
    if objective_mode == "treatment":
        encoding = _answer_encoding(axis, answers)
        segment_ids, _ = build_full_segment_ids(
            full_input_ids=input_ids,
            answer_input_ids=encoding["input_ids"].to(device),
            answer_offsets=encoding["offset_mapping"],
            answer_block_start=question_length + 1,
            answers=answers,
            question_types=question_types,
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
        mode=objective_mode,
        valid_rows=torch.tensor(valid_rows, device=device),
        segment_ids=segment_ids,
        question_types=question_types,
        alpha=segment_alpha,
        chunk_size=loss_chunk_size,
    )


class ContinuationObjectiveModel(nn.Module):
    """DDP wrapper that leaves the baseline combined model unmodified."""

    def __init__(
        self,
        base: nn.Module,
        *,
        objective_mode: str,
        segment_alpha: float,
        loss_chunk_size: int,
    ) -> None:
        super().__init__()
        if objective_mode not in {"control", "treatment"}:
            raise ValueError(f"unsupported arm: {objective_mode}")
        self.base = base
        self.objective_mode = objective_mode
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
            objective_mode=self.objective_mode,
            segment_alpha=self.segment_alpha,
            valid_rows=valid_rows,
            loss_chunk_size=self.loss_chunk_size,
        )
