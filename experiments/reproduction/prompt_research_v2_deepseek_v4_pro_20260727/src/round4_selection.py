"""Lightweight task-family and prompt-change selection for Round 4."""

from src.models.AXIS.prompt_stage_a import (
    V2_R4_MAIN_MODES,
    build_question_prompt,
)


ACTIVE_FAMILY = {mode: "open_ended" for mode in V2_R4_MAIN_MODES}


def prompt_changed(record, mode: str) -> bool:
    """Return whether ``mode`` changes this record's Baseline prompt."""
    common = {
        "question": record.question,
        "question_type": record.question_type,
        "start": record.start_index,
        "end": record.end_index,
        "serialized_values": "0",
        "local_hint_tokens": "<|local_hint|>",
        "fixed_hint_tokens": "<|fixed_hint|>" * 30,
        "aligned_rows": "",
    }
    return build_question_prompt(mode=mode, **common) != build_question_prompt(
        mode="base",
        **common,
    )
