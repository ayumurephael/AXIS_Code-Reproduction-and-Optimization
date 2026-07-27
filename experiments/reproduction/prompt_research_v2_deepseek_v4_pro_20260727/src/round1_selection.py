"""Lightweight task-family and prompt-change selection for Round 1."""

from src.models.AXIS.prompt_stage_a import build_question_prompt


ACTIVE_FAMILY = {
    "v2_r1_01_oe_context_balanced": "open_ended",
    "v2_r1_02_oe_context_contrast": "open_ended",
    "v2_r1_03_oe_evidence_router": "open_ended",
    "v2_r1_04_oe_context_direct": "open_ended",
    "v2_r1_05_oe_fixed_postnote": "open_ended",
    "v2_r1_06_mc_context_balanced": "multiple_choice",
    "v2_r1_07_mc_content_output": "multiple_choice",
    "v2_r1_08_tf_context_whole": "true_false",
    "v2_r1_09_tf_prefix_context": "true_false",
}


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
