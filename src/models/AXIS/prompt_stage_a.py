"""Prompt-only ablations for the released AXIS checkpoint.

Stage A deliberately preserves the released prompt's ordering and its
serialization of window values and local-hint tokens.  Each mode below changes
only the factor named by the experiment specification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


OUTPUT_PROTOCOLS = {
    "multiple_choice": """This is a single-answer multiple-choice question.

Silently evaluate every option against the time-series evidence. Do not use option position, option-letter frequency, or the wording of the question as evidence.

Output requirements:

1. The first line must be exactly: <LETTER>) <EXACT TEXT OF THE SELECTED OPTION>
2. Copy the selected option text verbatim. Do not paraphrase it and do not output only the letter.
3. On the next line, write "Explanation:" followed by 3–5 focused sentences.
4. The explanation must:

   * state the observed pattern that positively supports the selected option;
   * identify its relevant location, shape, persistence, or recovery behavior;
   * explain briefly why the strongest competing option does not fit.
5. Do not enumerate every option. Do not write "the correct answer is" after already giving the answer.""",
    "true_false": """This is a true-or-false question.

Judge the truth of the complete proposition, not merely whether an anomaly exists. Silently identify and evaluate every essential clause, including negation, "all", "only", "no", anomaly shape, temporal position, duration, boundary containment, and conjunctions. If an essential required clause is contradicted, the complete proposition is false.

Output requirements:

1. The first token must be exactly "True." or "False."
2. Continue with 3–5 natural sentences without headings such as "Explanation:" or "Evidence:".
3. If the answer is False, explicitly identify which part of the proposition is incorrect and state the evidence-consistent correction.
4. Support the judgment with the relevant observed pattern, location, duration or recovery behavior, and contextual consistency.
5. Never map anomaly presence mechanically to False or anomaly absence mechanically to True.""",
    "open_ended": """This is an open-ended question.

Answer the actual diagnostic question rather than giving a generic anomaly-detection tutorial. Identify every requested component of the question, such as anomaly status, anomaly type, temporal location, relation to window boundaries, supporting evidence, persistence, recovery, or analytical indicators.

Output requirements:

1. Begin with a direct diagnostic conclusion in the first sentence.
2. Do not begin with A, B, C, D, True, or False unless the question explicitly requests such a format.
3. Write one coherent paragraph of normally 4–6 focused sentences.
4. Cover every requested component, but do not add unrelated background.
5. Ground the answer in the observed shape, location, duration or recovery, and contextual consistency.
6. If the question asks what evidence or techniques should be examined, first assess the supplied window and then name only the indicators relevant to that assessment.
7. End when the requested conclusion and justification are complete; do not pad the answer with generic advice.""",
}


EVIDENCE_CONTRACT = """### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [{start}, {end}), i.e. steps {start} through {end_minus_1}.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable."""


@dataclass(frozen=True)
class PromptCondition:
    description: str
    answer_boundary: bool = False
    remove_fixed_hint: bool = False
    add_output_protocol: bool = False
    rename_fixed_hint: bool = False
    add_evidence_contract: bool = False
    remove_local_hint: bool = False
    remove_window_values: bool = False


MODE_SPECS: Dict[str, PromptCondition] = {
    "base": PromptCondition("Released AXIS prompt and generation boundary."),
    "answer_boundary": PromptCondition(
        "Released prompt; append EOS and tokenized literal Answer: to the generation prefix.",
        answer_boundary=True,
    ),
    "answer_boundary_wo_fixed": PromptCondition(
        "Answer-boundary alignment with all fixed-hint placeholders removed.",
        answer_boundary=True,
        remove_fixed_hint=True,
    ),
    "task_protocol": PromptCondition(
        "Add only the active question-type-specific output protocol.",
        add_output_protocol=True,
    ),
    "fixed_role": PromptCondition(
        "Rename Overall Summary Hints to Learned Task Guidance/Shared Task-Control Tokens.",
        rename_fixed_hint=True,
    ),
    "fixed_role_evidence_contract": PromptCondition(
        "Rename the fixed-hint role and add the specified Evidence Contract.",
        rename_fixed_hint=True,
        add_evidence_contract=True,
    ),
    "combined_234": PromptCondition(
        "Combine task protocol, fixed-hint role rename, and Evidence Contract.",
        add_output_protocol=True,
        rename_fixed_hint=True,
        add_evidence_contract=True,
    ),
    "answer_boundary_combined_234": PromptCondition(
        "Combine EOS + Answer: boundary alignment with all changes in combined_234.",
        answer_boundary=True,
        add_output_protocol=True,
        rename_fixed_hint=True,
        add_evidence_contract=True,
    ),
    # Released-code ablations remain available for compatibility.
    "wo_local_hint": PromptCondition(
        "Released-code ablation without local-hint placeholders.",
        remove_local_hint=True,
    ),
    "wo_fixed_hint": PromptCondition(
        "Released-code ablation without fixed-hint placeholders.",
        remove_fixed_hint=True,
    ),
    "wo_windows": PromptCondition(
        "Released-code ablation with serialized window values removed.",
        remove_window_values=True,
    ),
}


STAGE_A_MODES = (
    "base",
    "answer_boundary",
    "answer_boundary_wo_fixed",
    "task_protocol",
    "fixed_role",
    "fixed_role_evidence_contract",
    "combined_234",
    "answer_boundary_combined_234",
)


def normalize_mode(mode: Optional[str]) -> str:
    return "base" if mode is None else mode


def get_condition(mode: Optional[str]) -> PromptCondition:
    normalized = normalize_mode(mode)
    try:
        return MODE_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AXIS prompt/ablation mode {normalized!r}; "
            f"expected one of {sorted(MODE_SPECS)}"
        ) from exc


def mode_manifest() -> Dict[str, Dict[str, object]]:
    """Return a JSON-serializable definition of every formal Stage-A mode."""
    return {mode: asdict(MODE_SPECS[mode]) for mode in STAGE_A_MODES}


def build_question_prompt(
    *,
    question: str,
    question_type: Optional[str],
    start: int,
    end: int,
    serialized_values: str,
    local_hint_tokens: str,
    fixed_hint_tokens: str,
    mode: Optional[str],
) -> str:
    """Build the textual portion of a Stage-A prompt.

    The base branch intentionally reproduces the released f-string exactly,
    including leading/trailing newlines and indentation.
    """
    condition = get_condition(mode)
    if condition.add_output_protocol:
        if question_type not in OUTPUT_PROTOCOLS:
            raise ValueError(
                f"Unsupported question_type {question_type!r} for mode "
                f"{normalize_mode(mode)!r}; expected one of "
                f"{sorted(OUTPUT_PROTOCOLS)}"
            )
        protocol_section = (
            "\n            ### Active Output Protocol\n"
            f"            {OUTPUT_PROTOCOLS[question_type]}\n"
        )
    else:
        protocol_section = ""

    if condition.add_evidence_contract:
        contract_section = (
            "\n            "
            + EVIDENCE_CONTRACT.format(
                start=start,
                end=end,
                end_minus_1=end - 1,
            )
            + "\n"
        )
    else:
        contract_section = ""

    fixed_label = (
        "Learned Task Guidance/Shared Task-Control Tokens"
        if condition.rename_fixed_hint
        else "Overall Summary Hints"
    )

    return f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.
{contract_section}
            ### Time Series Data
            - **Window:** Steps {start} to {end}
            - **Values (scaled by 100):** {serialized_values}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_hint_tokens}
            - **{fixed_label}:** {fixed_hint_tokens}

            ### Question
            {question}
            {protocol_section}"""
