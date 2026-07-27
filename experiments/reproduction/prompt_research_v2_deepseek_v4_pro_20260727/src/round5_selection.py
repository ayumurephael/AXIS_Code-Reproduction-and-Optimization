"""Frozen selection rules for Round-5 two-pass OE refinement."""

ROUND5_MODES = (
    "v2_r5_01_oe_preserve_cover",
    "v2_r5_02_oe_verbatim_or_add",
    "v2_r5_03_oe_conservative_edit",
    "v2_r5_04_oe_evidence_repair",
    "v2_r5_05_oe_minimal_facets",
)

ACTIVE_FAMILY = {mode: "open_ended" for mode in ROUND5_MODES}


def prompt_changed(record, mode: str) -> bool:
    if mode not in ACTIVE_FAMILY:
        raise ValueError(f"Unknown Round-5 mode: {mode!r}")
    return record.question_type == "open_ended"
