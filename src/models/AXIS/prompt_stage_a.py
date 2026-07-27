"""Prompt-only ablations for the released AXIS checkpoint.

The original Stage-A modes preserve the released prompt's ordering and value /
local-hint serialization.  Follow-up modes change only the explicitly declared
prompt factors and continue to use the released generation boundary.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence


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


EVIDENCE_CONTRACT_FIRST_THREE = """### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [{start}, {end}), i.e. steps {start} through {end_minus_1}.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous."""


EVIDENCE_CONTRACT_REVISED = """### Evidence Contract
1. Window Values are rounded integer encodings of sample-specific
   observations for the half-open interval [{start}, {end}), listed in
   chronological order from step {start} through step {end_minus_1}.
   Use the displayed scale consistently when comparing shape, direction,
   relative change, and local deviations. Do not infer physical units.
   If the question explicitly requires an approximate unscaled numerical
   value, divide the displayed integer by 100 exactly once; otherwise,
   do not rescale the values.

2. Per-Step Context tokens are sample-specific and align one-to-one with
   Window Values by sequence position, not by text row. The first token
   corresponds to the first value at step {start}; each subsequent token
   corresponds to the next value; and the last token corresponds to the
   last value at step {end_minus_1}. Use each token only with its aligned
   value when judging whether that local behavior is unexpected relative
   to the complete temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable."""


PARETO_TASK_RULES = {
    "multiple_choice": (
        "Compare the complete meaning of every option with the evidence. "
        "The selected option must match anomaly status, shape, direction, "
        "temporal location, persistence or recovery, and boundary relation. "
        "Reject an option that requires an unsupported event."
    ),
    "true_false": (
        "Evaluate the truth of the complete proposition, including negation "
        "and every required clause. Decide the underlying evidence claim "
        "first, then map it to True or False. Anomaly presence does not "
        "mechanically imply either label."
    ),
    "open_ended": (
        "Answer every component requested by the question. Start from a "
        "diagnostic conclusion, then state the observed shape and location "
        "and why they support or refute an anomaly. Discuss boundary "
        "uncertainty only when relevant, and distinguish observed evidence "
        "from evidence that would still be needed."
    ),
}

PARETO_BOUNDARY_RULE = (
    "Use only evidence inside the half-open window [{start}, {end}); never "
    "invent or cite a step outside it."
)

PARETO_CALIBRATION_RULE = (
    "Call a behavior anomalous only when it is unexpected relative to the "
    "complete window and supported by Per-Step Analysis. A large or small "
    "value, sign change, smooth trend, ordinary peak or trough, or variability "
    "is not anomalous by itself; check local contrast, persistence, and "
    "recovery."
)

PARETO_NUMERIC_RULE = (
    "Use displayed values for qualitative shape, order, direction, relative "
    "change, and local contrast. Do not convert or quote exact numbers unless "
    "the question explicitly asks for an approximate numerical value in the "
    "original scale."
)

PARETO_SINGLE_ANSWER_RULE = (
    "Give one final answer only. Do not repeat, revise, or contradict it; stop "
    "after the shortest explanation that fully supports the answer."
)

ROUTED_MC_STABLE_RULE = (
    "Compare the complete meaning of every option with the supplied evidence. "
    "Select exactly one best-supported option, state that option once, and keep "
    "the explanation consistent with it. Do not revise the selected option or "
    "introduce a second answer."
)

ROUTED_TF_POLARITY_RULE = (
    'Judge the truth of the complete proposition exactly as written, preserving '
    'every negation such as "no evidence" and "does not". Answer True when the '
    "evidence supports the proposition as written; answer False when any "
    "essential clause is contradicted. Before finishing, verify that the label "
    "and the first explanatory sentence express the same truth value."
)

ROUTED_TF_BOUNDARY_POLARITY_RULE = (
    "Use only evidence inside the half-open window [{start}, {end}); do not "
    "invent or cite a step outside it. "
    + ROUTED_TF_POLARITY_RULE
)

ROUTED_OE_SPEECH_ACT_RULE = (
    "First determine whether the question asks for a diagnosis, an assessment "
    "method, or evidence that would support or challenge an assessment. Address "
    "the supplied window before general methods. Cover every requested part: "
    "the observed shape and location; what it currently supports or challenges "
    "relative to ordinary variation; relevant boundary, persistence, or recovery "
    "uncertainty; and only the requested additional evidence or indicators. For "
    "a methodological or evidence-seeking question, do not deny its premise "
    "merely to force a normal/anomalous verdict, and do not claim that no further "
    "evidence is needed unless the question and supplied evidence justify that "
    "claim."
)

ROUTED_OE_POSITIVE_EVIDENCE_RULE = (
    "Answer the open-ended question in one coherent paragraph. Include: "
    "(1) a direct assessment of this window; (2) two or three observed "
    "qualitative features that support the assessment, including shape and "
    "persistence or recovery; and (3) the boundary/context limitation or "
    "additional evidence requested by the question. State location as "
    "beginning, middle, or end and describe magnitude relatively unless an "
    "exact value or step is unambiguous in the supplied input."
)

ROUTED_OE_NAMED_EVENT_RULE = (
    "Treat the event or pattern named in the question as the behavior to "
    "evaluate, not as a predetermined anomaly label. Compare its abruptness, "
    "isolation, persistence or recovery, and boundary position with the rest "
    "of the supplied window. Report the evidence supporting your conclusion "
    "and the strongest evidence against it or remaining uncertainty."
)

ROUTED_OE_QUALITATIVE_SYNTHESIS_RULE = (
    "First answer the assessment or analysis requested by the question. Then "
    "apply it to this window using qualitative evidence: local contrast, "
    "persistence or recovery, and boundary position. State what supports the "
    "conclusion and what observation would refute it or require outside-window "
    "context. Use beginning, middle, or end for location when exact step "
    "alignment is not explicit."
)

LITERATURE_MC_SEMANTIC_BIND_RULE = (
    "Choose the option by its complete text before mapping it to A, B, C, "
    "or D. Report that option once and give a brief reason from the supplied "
    "evidence."
)

LITERATURE_MC_POINTWISE_RULE = (
    "Test each option's complete claim independently against the supplied "
    "evidence. Choose the best-supported option by its text, then report its "
    "attached letter and a brief reason."
)

LITERATURE_MC_SEMANTIC_QUALITATIVE_RULE = (
    "Choose the option by its complete text before mapping it to A, B, C, "
    "or D. Explain the observed qualitative pattern and its persistence or "
    "recovery that distinguish the selected option from the strongest "
    "alternative. Do not quote exact values or step numbers unless the "
    "question explicitly asks for them."
)

LITERATURE_MC_POINTWISE_QUALITATIVE_RULE = (
    "Test each option's complete claim independently against the supplied "
    "evidence, then choose the best-supported option by its text. Explain "
    "which observed pattern and continuation or recovery support it and "
    "which part of the strongest alternative is absent. Use qualitative "
    "evidence; do not quote exact values or step numbers unless the question "
    "explicitly asks for them."
)

LITERATURE_MC_SALIENT_AFTERMATH_RULE = (
    "Match the options to the most salient local change and what happens "
    "immediately afterward. Select the option whose complete description "
    "matches both, then explain the local change and its continuation or "
    "recovery using qualitative evidence. Do not quote exact values or step "
    "numbers unless the question explicitly asks for them."
)

LITERATURE_TF_MINIMAL_RULE = (
    "Judge whether the complete statement as written is true. Report True or "
    "False once, followed by a brief reason that supports the same truth value."
)

LITERATURE_TF_CLAUSE_RULE = (
    "Check every required clause and negation in the statement. It is false "
    "if any required clause is contradicted; report one truth label and a "
    "brief consistent reason."
)

LITERATURE_TF_RE2_VERDICT_RULE = (
    'Evaluate the complete proposition exactly as written. End with exactly '
    '"Your answer: True." or "Your answer: False.", and keep the explanation '
    "consistent with that verdict."
)

LITERATURE_TF_RE2_QUALITATIVE_VERDICT_RULE = (
    "Evaluate the complete proposition exactly as written. Use qualitative "
    "shape, direction, persistence, and recovery; do not quote exact values "
    "or step numbers unless the question explicitly asks for them. End with "
    'exactly "Your answer: True." or "Your answer: False.", and keep the '
    "explanation consistent with that verdict."
)

LITERATURE_TF_PREFIX_RULE = (
    "Preserve the proposition's polarity exactly as written. Decide whether "
    "the complete statement is supported, and begin with exactly "
    '"Answer: True." or "Answer: False." Then give a qualitative reason '
    "consistent with that label. Do not quote exact values or step numbers "
    "unless the question asks for them."
)


LITERATURE_OE_DIRECT_RULE = (
    "Answer exactly what the question asks, using evidence from this window. "
    "Include the requested conclusion or assessment and the brief reason "
    "needed to support it."
)

LITERATURE_DECOUPLED_RULES = {
    "mc_decoupled": (
        "First determine which complete option text is best supported by the "
        "evidence. Then report its attached letter once and give one brief reason."
    ),
    "tf_decoupled": (
        "First determine whether the complete statement is supported. Then "
        "report True or False once and give one brief reason consistent with it."
    ),
    "oe_decoupled": (
        "First determine the analysis requested by the question. Then answer "
        "it directly with a brief reason from the supplied evidence."
    ),
}

LITERATURE_R1_PROFILES = {
    "lit_r1_01_mc_semantic_bind": {
        "multiple_choice": "mc_semantic_bind",
        "true_false": "base",
        "open_ended": "base",
    },
    "lit_r1_02_mc_pointwise": {
        "multiple_choice": "mc_pointwise",
        "true_false": "base",
        "open_ended": "base",
    },
    "lit_r1_03_mc_re2": {
        "multiple_choice": "re2",
        "true_false": "base",
        "open_ended": "base",
    },
    "lit_r1_04_tf_minimal": {
        "multiple_choice": "base",
        "true_false": "tf_minimal",
        "open_ended": "base",
    },
    "lit_r1_05_tf_clause": {
        "multiple_choice": "base",
        "true_false": "tf_clause",
        "open_ended": "base",
    },
    "lit_r1_06_tf_re2": {
        "multiple_choice": "base",
        "true_false": "re2",
        "open_ended": "base",
    },
    "lit_r1_07_oe_direct": {
        "multiple_choice": "base",
        "true_false": "base",
        "open_ended": "oe_direct",
    },
    "lit_r1_08_oe_re2": {
        "multiple_choice": "base",
        "true_false": "base",
        "open_ended": "re2",
    },
    "lit_r1_09_triplet_minimal": {
        "multiple_choice": "mc_semantic_bind",
        "true_false": "tf_minimal",
        "open_ended": "oe_direct",
    },
    "lit_r1_10_triplet_re2": {
        "multiple_choice": "re2",
        "true_false": "re2",
        "open_ended": "re2",
    },
    "lit_r1_11_decoupled": {
        "multiple_choice": "mc_decoupled",
        "true_false": "tf_decoupled",
        "open_ended": "oe_decoupled",
    },
    "lit_r1_12_mc_bind_re2": {
        "multiple_choice": "mc_semantic_bind_re2",
        "true_false": "base",
        "open_ended": "base",
    },
}

LITERATURE_R2_PROFILES = {
    "lit_r2_01_mc_tf_re2_safe": {
        "multiple_choice": "re2",
        "true_false": "re2",
        "open_ended": "base",
    },
}

LITERATURE_R3_PROFILES = {
    "lit_r3_01_mc_pointwise_qual": {
        "multiple_choice": "mc_pointwise_qual",
        "true_false": "base",
        "open_ended": "base",
    },
    "lit_r3_02_mc_semantic_qual": {
        "multiple_choice": "mc_semantic_qual",
        "true_false": "base",
        "open_ended": "base",
    },
    "lit_r3_03_mc_salient_aftermath": {
        "multiple_choice": "mc_salient_aftermath",
        "true_false": "base",
        "open_ended": "base",
    },
    "lit_r3_04_tf_re2_verdict": {
        "multiple_choice": "base",
        "true_false": "tf_re2_verdict",
        "open_ended": "base",
    },
    "lit_r3_05_tf_re2_qual_verdict": {
        "multiple_choice": "base",
        "true_false": "tf_re2_qual_verdict",
        "open_ended": "base",
    },
    "lit_r3_06_joint_pointwise_tf_qual": {
        "multiple_choice": "mc_pointwise_qual",
        "true_false": "tf_re2_qual_verdict",
        "open_ended": "base",
    },
    "lit_r3_07_joint_semantic_tf_qual": {
        "multiple_choice": "mc_semantic_qual",
        "true_false": "tf_re2_qual_verdict",
        "open_ended": "base",
    },
}

TF_EXPLICIT_NEGATIVE_CUES = frozenset({
    "no", "not", "without", "absence", "lack", "neither", "nor",
    "cannot", "can't", "doesn't", "isn't", "aren't", "wasn't", "weren't",
})
TF_NONANOMALY_CUES = frozenset({
    "normal", "stable", "consistent", "regular", "expected", "typical",
})


def _tf_question_tokens(question: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z]+(?:'[a-z]+)?", question.lower()))


def _has_explicit_tf_negative_cue(question: str) -> bool:
    return bool(_tf_question_tokens(question) & TF_EXPLICIT_NEGATIVE_CUES)


def _has_tf_nonanomaly_cue(question: str) -> bool:
    tokens = _tf_question_tokens(question)
    return bool(tokens & (TF_EXPLICIT_NEGATIVE_CUES | TF_NONANOMALY_CUES))


LITERATURE_R4_PROFILES = {
    "lit_r4_01_tf_neg_re2": {
        "multiple_choice": "base",
        "true_false": "tf_neg_re2_router",
        "open_ended": "base",
    },
    "lit_r4_02_joint_semantic_tf_neg_re2": {
        "multiple_choice": "mc_semantic_qual",
        "true_false": "tf_neg_re2_router",
        "open_ended": "base",
    },
    "lit_r4_03_tf_nonanomaly_re2": {
        "multiple_choice": "base",
        "true_false": "tf_nonanomaly_re2_router",
        "open_ended": "base",
    },
    "lit_r4_04_joint_semantic_tf_nonanomaly_re2": {
        "multiple_choice": "mc_semantic_qual",
        "true_false": "tf_nonanomaly_re2_router",
        "open_ended": "base",
    },
    "lit_r4_05_tf_neg_prefix": {
        "multiple_choice": "base",
        "true_false": "tf_neg_prefix_router",
        "open_ended": "base",
    },
    "lit_r4_06_joint_semantic_tf_neg_prefix": {
        "multiple_choice": "mc_semantic_qual",
        "true_false": "tf_neg_prefix_router",
        "open_ended": "base",
    },
}


ROUTED_R2_PROFILES = {
    "r2_01_minimal": {
        "multiple_choice": "mc_f0",
        "true_false": "tf_p0",
        "open_ended": "oe_s0",
    },
    "r2_02_mc_stable": {
        "multiple_choice": "mc_f1",
        "true_false": "tf_p0",
        "open_ended": "oe_s0",
    },
    "r2_03_tf_boundary": {
        "multiple_choice": "mc_f0",
        "true_false": "tf_p1",
        "open_ended": "oe_s0",
    },
    "r2_04_oe_old_contract": {
        "multiple_choice": "mc_f0",
        "true_false": "tf_p0",
        "open_ended": "oe_c0",
    },
    "r2_05_oe_contract_coverage": {
        "multiple_choice": "mc_f0",
        "true_false": "tf_p0",
        "open_ended": "oe_c1",
    },
    "r2_06_full_routed": {
        "multiple_choice": "mc_f1",
        "true_false": "tf_p1",
        "open_ended": "oe_c1",
    },
}

ROUTED_R3_PROFILES = {
    "r3_01_mc_f0_safe": {
        "multiple_choice": "mc_f0", "true_false": "base", "open_ended": "base",
    },
    "r3_02_mc_f1_safe": {
        "multiple_choice": "mc_f1", "true_false": "base", "open_ended": "base",
    },
    "r3_03_mc_f0_tf_p0": {
        "multiple_choice": "mc_f0", "true_false": "tf_p0", "open_ended": "base",
    },
    "r3_04_mc_f1_tf_p0": {
        "multiple_choice": "mc_f1", "true_false": "tf_p0", "open_ended": "base",
    },
    "r3_05_oe_q1": {
        "multiple_choice": "mc_f0", "true_false": "tf_p0", "open_ended": "oe_q1",
    },
    "r3_06_oe_q2": {
        "multiple_choice": "mc_f0", "true_false": "tf_p0", "open_ended": "oe_q2",
    },
    "r3_07_oe_q3": {
        "multiple_choice": "mc_f0", "true_false": "tf_p0", "open_ended": "oe_q3",
    },
    "r3_08_historical": {
        "multiple_choice": "mc_f0", "true_false": "base", "open_ended": "oe_c0",
    },
}

EXPERT_ANOMALY_ANALYST_OPENING = (
    "You are an expert time-series anomaly analyst. Produce one precise, "
    "evidence-grounded answer to the question."
)


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
    use_expert_anomaly_analyst_opening: bool = False
    use_first_three_contract_items: bool = False
    use_revised_evidence_contract: bool = False
    standalone_fixed_hint: bool = False
    interleave_time_series_evidence: bool = False
    pareto_factors: str = ""
    routed_profile: str = ""
    literature_profile: str = ""


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
    "fixed_role_evidence_contract_revised": PromptCondition(
        "Rename the fixed-hint role and add the revised four-item Evidence "
        "Contract with explicit rounded-value and positional-alignment rules.",
        rename_fixed_hint=True,
        use_revised_evidence_contract=True,
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
    "expert_contract3_fixed_section": PromptCondition(
        "Use the anomaly-analyst opening, the first three Evidence Contract "
        "items, and a standalone renamed fixed-hint section.",
        rename_fixed_hint=True,
        use_expert_anomaly_analyst_opening=True,
        use_first_three_contract_items=True,
        standalone_fixed_hint=True,
    ),
    "expert_contract3_interleaved": PromptCondition(
        "Apply expert_contract3_fixed_section and replace the released "
        "value/local blocks with step-interleaved time-series evidence.",
        rename_fixed_hint=True,
        use_expert_anomaly_analyst_opening=True,
        use_first_three_contract_items=True,
        standalone_fixed_hint=True,
        interleave_time_series_evidence=True,
    ),
    "pareto_p01_boundary": PromptCondition(
        "P01: fixed role plus boundary guard (A).",
        rename_fixed_hint=True,
        pareto_factors="A",
    ),
    "pareto_p02_calibration": PromptCondition(
        "P02: fixed role plus anomaly calibration (B).",
        rename_fixed_hint=True,
        pareto_factors="B",
    ),
    "pareto_p03_task_rule": PromptCondition(
        "P03: fixed role plus question-type-conditioned decision rule (C).",
        rename_fixed_hint=True,
        pareto_factors="C",
    ),
    "pareto_p04_numeric_guard": PromptCondition(
        "P04: fixed role plus qualitative numeric guard (D).",
        rename_fixed_hint=True,
        pareto_factors="D",
    ),
    "pareto_p05_single_answer": PromptCondition(
        "P05: fixed role plus single-answer concise guard (E).",
        rename_fixed_hint=True,
        pareto_factors="E",
    ),
    "pareto_p06_abc": PromptCondition(
        "P06: boundary, calibration, and task rules (ABC).",
        rename_fixed_hint=True,
        pareto_factors="ABC",
    ),
    "pareto_p07_abd": PromptCondition(
        "P07: boundary, calibration, and numeric rules (ABD).",
        rename_fixed_hint=True,
        pareto_factors="ABD",
    ),
    "pareto_p08_acde": PromptCondition(
        "P08: boundary, task, numeric, and answer rules (ACDE).",
        rename_fixed_hint=True,
        pareto_factors="ACDE",
    ),
    "pareto_p09_bce": PromptCondition(
        "P09: calibration, task, and answer rules (BCE).",
        rename_fixed_hint=True,
        pareto_factors="BCE",
    ),
    "pareto_p10_abcde": PromptCondition(
        "P10: full Pareto-v1 rules (ABCDE).",
        rename_fixed_hint=True,
        pareto_factors="ABCDE",
    ),
    "pareto_p11_abce": PromptCondition(
        "P11: Pareto-v1 without the numeric guard (ABCE).",
        rename_fixed_hint=True,
        pareto_factors="ABCE",
    ),
    "route_r2_01_minimal": PromptCondition(
        "Round 2: MC fixed-only, TF direct polarity, OE speech-act coverage.",
        rename_fixed_hint=True,
        routed_profile="r2_01_minimal",
    ),
    "route_r2_02_mc_stable": PromptCondition(
        "Round 2: add stable MC selection to the minimal routed profile.",
        rename_fixed_hint=True,
        routed_profile="r2_02_mc_stable",
    ),
    "route_r2_03_tf_boundary": PromptCondition(
        "Round 2: add boundary discipline to the TF routed profile.",
        rename_fixed_hint=True,
        routed_profile="r2_03_tf_boundary",
    ),
    "route_r2_04_oe_old_contract": PromptCondition(
        "Round 2: route the historical five-item Contract to OE only.",
        rename_fixed_hint=True,
        routed_profile="r2_04_oe_old_contract",
    ),
    "route_r2_05_oe_contract_coverage": PromptCondition(
        "Round 2: OE-only historical Contract plus speech-act coverage.",
        rename_fixed_hint=True,
        routed_profile="r2_05_oe_contract_coverage",
    ),
    "route_r2_06_full_routed": PromptCondition(
        "Round 2: stable MC, boundary/polarity TF, Contract/coverage OE.",
        rename_fixed_hint=True,
        routed_profile="r2_06_full_routed",
    ),
    "route_r3_01_mc_f0_safe": PromptCondition(
        "Round 3: fixed-role MC with exact Baseline TF and OE.",
        routed_profile="r3_01_mc_f0_safe",
    ),
    "route_r3_02_mc_f1_safe": PromptCondition(
        "Round 3: stable-selection MC with exact Baseline TF and OE.",
        routed_profile="r3_02_mc_f1_safe",
    ),
    "route_r3_03_mc_f0_tf_p0": PromptCondition(
        "Round 3: fixed-role MC, direct-polarity TF, exact Baseline OE.",
        routed_profile="r3_03_mc_f0_tf_p0",
    ),
    "route_r3_04_mc_f1_tf_p0": PromptCondition(
        "Round 3: stable-selection MC, direct-polarity TF, exact Baseline OE.",
        routed_profile="r3_04_mc_f1_tf_p0",
    ),
    "route_r3_05_oe_q1": PromptCondition(
        "Round 3: positive OE evidence obligation.",
        routed_profile="r3_05_oe_q1",
    ),
    "route_r3_06_oe_q2": PromptCondition(
        "Round 3: named-event OE test.",
        routed_profile="r3_06_oe_q2",
    ),
    "route_r3_07_oe_q3": PromptCondition(
        "Round 3: compact qualitative OE synthesis.",
        routed_profile="r3_07_oe_q3",
    ),
    "route_r3_08_historical": PromptCondition(
        "Round 3: fixed-role MC, Baseline TF, historical OE Contract.",
        routed_profile="r3_08_historical",
    ),
    "lit_r1_01_mc_semantic_bind": PromptCondition(
        "Literature Round 1: MC semantic option binding on the released scaffold.",
        literature_profile="lit_r1_01_mc_semantic_bind",
    ),
    "lit_r1_02_mc_pointwise": PromptCondition(
        "Literature Round 1: MC pointwise option verification.",
        literature_profile="lit_r1_02_mc_pointwise",
    ),
    "lit_r1_03_mc_re2": PromptCondition(
        "Literature Round 1: MC question re-reading (RE2).",
        literature_profile="lit_r1_03_mc_re2",
    ),
    "lit_r1_04_tf_minimal": PromptCondition(
        "Literature Round 1: minimal TF polarity-consistent response.",
        literature_profile="lit_r1_04_tf_minimal",
    ),
    "lit_r1_05_tf_clause": PromptCondition(
        "Literature Round 1: TF clause and negation verification.",
        literature_profile="lit_r1_05_tf_clause",
    ),
    "lit_r1_06_tf_re2": PromptCondition(
        "Literature Round 1: TF question re-reading (RE2).",
        literature_profile="lit_r1_06_tf_re2",
    ),
    "lit_r1_07_oe_direct": PromptCondition(
        "Literature Round 1: direct OE answering rule.",
        literature_profile="lit_r1_07_oe_direct",
    ),
    "lit_r1_08_oe_re2": PromptCondition(
        "Literature Round 1: OE question re-reading (RE2).",
        literature_profile="lit_r1_08_oe_re2",
    ),
    "lit_r1_09_triplet_minimal": PromptCondition(
        "Literature Round 1: task-specific minimal rules for all task types.",
        literature_profile="lit_r1_09_triplet_minimal",
    ),
    "lit_r1_10_triplet_re2": PromptCondition(
        "Literature Round 1: question re-reading for all task types.",
        literature_profile="lit_r1_10_triplet_re2",
    ),
    "lit_r1_11_decoupled": PromptCondition(
        "Literature Round 1: decision/report decoupling for all task types.",
        literature_profile="lit_r1_11_decoupled",
    ),
    "lit_r1_12_mc_bind_re2": PromptCondition(
        "Literature Round 1: MC semantic binding plus question re-reading.",
        literature_profile="lit_r1_12_mc_bind_re2",
    ),
    "lit_r2_01_mc_tf_re2_safe": PromptCondition(
        "Literature Round 2: MC and TF re-reading with exact Baseline OE.",
        literature_profile="lit_r2_01_mc_tf_re2_safe",
    ),
    "lit_r3_01_mc_pointwise_qual": PromptCondition(
        "Literature Round 3: MC pointwise comparison with qualitative support.",
        literature_profile="lit_r3_01_mc_pointwise_qual",
    ),
    "lit_r3_02_mc_semantic_qual": PromptCondition(
        "Literature Round 3: MC semantic binding with qualitative support.",
        literature_profile="lit_r3_02_mc_semantic_qual",
    ),
    "lit_r3_03_mc_salient_aftermath": PromptCondition(
        "Literature Round 3: MC salient-event and aftermath matching.",
        literature_profile="lit_r3_03_mc_salient_aftermath",
    ),
    "lit_r3_04_tf_re2_verdict": PromptCondition(
        "Literature Round 3: TF re-reading with an explicit final verdict.",
        literature_profile="lit_r3_04_tf_re2_verdict",
    ),
    "lit_r3_05_tf_re2_qual_verdict": PromptCondition(
        "Literature Round 3: TF re-reading, qualitative evidence, and verdict.",
        literature_profile="lit_r3_05_tf_re2_qual_verdict",
    ),
    "lit_r3_06_joint_pointwise_tf_qual": PromptCondition(
        "Literature Round 3: pointwise MC plus qualitative-verdict TF.",
        literature_profile="lit_r3_06_joint_pointwise_tf_qual",
    ),
    "lit_r3_07_joint_semantic_tf_qual": PromptCondition(
        "Literature Round 3: semantic MC plus qualitative-verdict TF.",
        literature_profile="lit_r3_07_joint_semantic_tf_qual",
    ),
    "lit_r4_01_tf_neg_re2": PromptCondition(
        "Literature Round 4: negative-cue TF router with qualitative RE2.",
        literature_profile="lit_r4_01_tf_neg_re2",
    ),
    "lit_r4_02_joint_semantic_tf_neg_re2": PromptCondition(
        "Literature Round 4: semantic MC plus negative-cue TF RE2 router.",
        literature_profile="lit_r4_02_joint_semantic_tf_neg_re2",
    ),
    "lit_r4_03_tf_nonanomaly_re2": PromptCondition(
        "Literature Round 4: non-anomaly-cue TF router with qualitative RE2.",
        literature_profile="lit_r4_03_tf_nonanomaly_re2",
    ),
    "lit_r4_04_joint_semantic_tf_nonanomaly_re2": PromptCondition(
        "Literature Round 4: semantic MC plus non-anomaly TF RE2 router.",
        literature_profile="lit_r4_04_joint_semantic_tf_nonanomaly_re2",
    ),
    "lit_r4_05_tf_neg_prefix": PromptCondition(
        "Literature Round 4: negative-cue TF router with no-reread prefix.",
        literature_profile="lit_r4_05_tf_neg_prefix",
    ),
    "lit_r4_06_joint_semantic_tf_neg_prefix": PromptCondition(
        "Literature Round 4: semantic MC plus negative-cue TF prefix router.",
        literature_profile="lit_r4_06_joint_semantic_tf_neg_prefix",
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

PARETO_SCREEN_MODES = (
    "fixed_role",
    "pareto_p01_boundary",
    "pareto_p02_calibration",
    "pareto_p03_task_rule",
    "pareto_p04_numeric_guard",
    "pareto_p05_single_answer",
    "pareto_p06_abc",
    "pareto_p07_abd",
    "pareto_p08_acde",
    "pareto_p09_bce",
    "pareto_p10_abcde",
    "pareto_p11_abce",
)

ROUTED_R2_MODES = (
    "route_r2_01_minimal",
    "route_r2_02_mc_stable",
    "route_r2_03_tf_boundary",
    "route_r2_04_oe_old_contract",
    "route_r2_05_oe_contract_coverage",
    "route_r2_06_full_routed",
)

ROUTED_R3_MODES = (
    "route_r3_01_mc_f0_safe",
    "route_r3_02_mc_f1_safe",
    "route_r3_03_mc_f0_tf_p0",
    "route_r3_04_mc_f1_tf_p0",
    "route_r3_05_oe_q1",
    "route_r3_06_oe_q2",
    "route_r3_07_oe_q3",
    "route_r3_08_historical",
)

LITERATURE_R1_MODES = (
    "lit_r1_01_mc_semantic_bind",
    "lit_r1_02_mc_pointwise",
    "lit_r1_03_mc_re2",
    "lit_r1_04_tf_minimal",
    "lit_r1_05_tf_clause",
    "lit_r1_06_tf_re2",
    "lit_r1_07_oe_direct",
    "lit_r1_08_oe_re2",
    "lit_r1_09_triplet_minimal",
    "lit_r1_10_triplet_re2",
    "lit_r1_11_decoupled",
    "lit_r1_12_mc_bind_re2",
)

LITERATURE_R2_MODES = (
    "lit_r2_01_mc_tf_re2_safe",
)

LITERATURE_R3_MODES = (
    "lit_r3_01_mc_pointwise_qual",
    "lit_r3_02_mc_semantic_qual",
    "lit_r3_03_mc_salient_aftermath",
    "lit_r3_04_tf_re2_verdict",
    "lit_r3_05_tf_re2_qual_verdict",
    "lit_r3_06_joint_pointwise_tf_qual",
    "lit_r3_07_joint_semantic_tf_qual",
)

LITERATURE_R4_MODES = (
    "lit_r4_01_tf_neg_re2",
    "lit_r4_02_joint_semantic_tf_neg_re2",
    "lit_r4_03_tf_nonanomaly_re2",
    "lit_r4_04_joint_semantic_tf_nonanomaly_re2",
    "lit_r4_05_tf_neg_prefix",
    "lit_r4_06_joint_semantic_tf_neg_prefix",
)


FOLLOWUP_PROMPT_MODES = (
    "expert_contract3_fixed_section",
    "expert_contract3_interleaved",
    "fixed_role_evidence_contract_revised",
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


def mode_manifest(
    modes: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    """Return JSON-serializable definitions for the requested formal modes."""
    selected_modes = STAGE_A_MODES if modes is None else tuple(modes)
    return {mode: asdict(get_condition(mode)) for mode in selected_modes}


def build_aligned_rows(start: int, window_values: Iterable[float]) -> str:
    """Serialize one scaled observation beside one local-hint placeholder."""
    rows = []
    for offset, value in enumerate(window_values):
        # Preserve the released prompt's ``(x * 100):.0f`` rounding rule, then
        # serialize the resulting integer with the approved signed width.
        scaled_value = int(f"{float(value) * 100:.0f}")
        rows.append(
            f"Step {start + offset:04d} | value={scaled_value:+06d} | "
            "context=<|local_hint|>"
        )
    return "\n".join(rows)


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
    aligned_rows: Optional[str] = None,
) -> str:
    """Build the textual portion of a Stage-A prompt.

    The base branch intentionally reproduces the released f-string exactly,
    including leading/trailing newlines and indentation.
    """
    condition = get_condition(mode)
    if condition.interleave_time_series_evidence and aligned_rows is None:
        raise ValueError(
            f"Mode {normalize_mode(mode)!r} requires step-aligned evidence rows"
        )
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

    if condition.use_revised_evidence_contract:
        contract_text = EVIDENCE_CONTRACT_REVISED
    elif condition.use_first_three_contract_items:
        contract_text = EVIDENCE_CONTRACT_FIRST_THREE
    elif condition.add_evidence_contract:
        contract_text = EVIDENCE_CONTRACT
    else:
        contract_text = None

    if contract_text is not None:
        contract_section = (
            "\n            "
            + contract_text.format(
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

    if condition.literature_profile:
        try:
            literature_profiles = {
                **LITERATURE_R1_PROFILES,
                **LITERATURE_R2_PROFILES,
                **LITERATURE_R3_PROFILES,
                **LITERATURE_R4_PROFILES,
            }
            literature_component = literature_profiles[
                condition.literature_profile][question_type]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported literature profile/question type: "
                f"{condition.literature_profile!r}/{question_type!r}"
            ) from exc

        if question_type == "true_false":
            if literature_component == "tf_neg_re2_router":
                literature_component = (
                    "tf_re2_qual_verdict"
                    if _has_explicit_tf_negative_cue(question)
                    else "base"
                )
            elif literature_component == "tf_nonanomaly_re2_router":
                literature_component = (
                    "tf_re2_qual_verdict"
                    if _has_tf_nonanomaly_cue(question)
                    else "base"
                )
            elif literature_component == "tf_neg_prefix_router":
                literature_component = (
                    "tf_prefix"
                    if _has_explicit_tf_negative_cue(question)
                    else "base"
                )

        if literature_component == "base":
            return build_question_prompt(
                question=question,
                question_type=question_type,
                start=start,
                end=end,
                serialized_values=serialized_values,
                local_hint_tokens=local_hint_tokens,
                fixed_hint_tokens=fixed_hint_tokens,
                mode="base",
                aligned_rows=aligned_rows,
            )

        reread_question = literature_component in {
            "re2",
            "mc_semantic_bind_re2",
            "tf_re2_verdict",
            "tf_re2_qual_verdict",
        }

        literature_rules = {
            "mc_semantic_bind": LITERATURE_MC_SEMANTIC_BIND_RULE,
            "mc_semantic_bind_re2": LITERATURE_MC_SEMANTIC_BIND_RULE,
            "mc_pointwise": LITERATURE_MC_POINTWISE_RULE,
            "mc_semantic_qual": LITERATURE_MC_SEMANTIC_QUALITATIVE_RULE,
            "mc_pointwise_qual": LITERATURE_MC_POINTWISE_QUALITATIVE_RULE,
            "mc_salient_aftermath": LITERATURE_MC_SALIENT_AFTERMATH_RULE,
            "tf_minimal": LITERATURE_TF_MINIMAL_RULE,
            "tf_clause": LITERATURE_TF_CLAUSE_RULE,
            "tf_re2_verdict": LITERATURE_TF_RE2_VERDICT_RULE,
            "tf_re2_qual_verdict": (
                LITERATURE_TF_RE2_QUALITATIVE_VERDICT_RULE
            ),
            "tf_prefix": LITERATURE_TF_PREFIX_RULE,
            "oe_direct": LITERATURE_OE_DIRECT_RULE,
            **LITERATURE_DECOUPLED_RULES,
        }
        literature_task_rule = literature_rules.get(literature_component)
        if literature_component != "re2" and literature_task_rule is None:
            raise ValueError(
                f"Unsupported literature component {literature_component!r}"
            )

        if literature_task_rule is None:
            literature_task_section = ""
        else:
            literature_task_section = (
                "\n            ### Answering Rule\n"
                f"            {literature_task_rule}\n"
            )

        if reread_question:
            literature_reread_section = (
                "\n\n            Read the question again:\n"
                f"            {question}"
            )
        else:
            literature_reread_section = ""

        return f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps {start} to {end}
            - **Values (scaled by 100):** {serialized_values}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_hint_tokens}
            - **Overall Summary Hints:** {fixed_hint_tokens}
{literature_task_section}
            ### Question
            {question}{literature_reread_section}
            """

    if condition.routed_profile:
        routed_profiles = {**ROUTED_R2_PROFILES, **ROUTED_R3_PROFILES}
        try:
            routed_component = routed_profiles[condition.routed_profile][
                question_type
            ]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported routed profile/question type: "
                f"{condition.routed_profile!r}/{question_type!r}"
            ) from exc

        if routed_component == "base":
            return build_question_prompt(
                question=question,
                question_type=question_type,
                start=start,
                end=end,
                serialized_values=serialized_values,
                local_hint_tokens=local_hint_tokens,
                fixed_hint_tokens=fixed_hint_tokens,
                mode="base",
                aligned_rows=aligned_rows,
            )

        if routed_component in {"oe_q1", "oe_q2", "oe_q3"}:
            routed_fixed_label = "Overall Summary Hints"
        else:
            routed_fixed_label = (
                "Learned Task Guidance/Shared Task-Control Tokens"
            )

        routed_contract_section = ""
        if routed_component in {"oe_c0", "oe_c1"}:
            routed_contract_section = (
                "\n            "
                + EVIDENCE_CONTRACT.format(
                    start=start,
                    end=end,
                    end_minus_1=end - 1,
                )
                + "\n"
            )

        if routed_component == "mc_f0" or routed_component == "oe_c0":
            routed_task_rule = None
        elif routed_component == "mc_f1":
            routed_task_rule = ROUTED_MC_STABLE_RULE
        elif routed_component == "tf_p0":
            routed_task_rule = ROUTED_TF_POLARITY_RULE
        elif routed_component == "tf_p1":
            routed_task_rule = ROUTED_TF_BOUNDARY_POLARITY_RULE.format(
                start=start,
                end=end,
            )
        elif routed_component in {"oe_s0", "oe_c1"}:
            routed_task_rule = ROUTED_OE_SPEECH_ACT_RULE
        elif routed_component == "oe_q1":
            routed_task_rule = ROUTED_OE_POSITIVE_EVIDENCE_RULE
        elif routed_component == "oe_q2":
            routed_task_rule = ROUTED_OE_NAMED_EVENT_RULE
        elif routed_component == "oe_q3":
            routed_task_rule = ROUTED_OE_QUALITATIVE_SYNTHESIS_RULE
        else:
            raise ValueError(f"Unsupported routed component {routed_component!r}")

        if routed_task_rule is None:
            routed_task_section = ""
        else:
            routed_task_section = (
                "\n            ### Task Rule\n"
                f"            {routed_task_rule}\n"
            )

        return f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.
{routed_contract_section}
            ### Time Series Data
            - **Window:** Steps {start} to {end}
            - **Values (scaled by 100):** {serialized_values}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_hint_tokens}
            - **{routed_fixed_label}:** {fixed_hint_tokens}
{routed_task_section}
            ### Question
            {question}
            """

    if condition.pareto_factors:
        unknown = set(condition.pareto_factors) - set("ABCDE")
        if unknown:
            raise ValueError(
                f"Unknown Pareto prompt factor(s): {sorted(unknown)}"
            )
        if "C" in condition.pareto_factors:
            if question_type not in PARETO_TASK_RULES:
                raise ValueError(
                    f"Unsupported question_type {question_type!r} for Pareto "
                    f"task rule; expected one of {sorted(PARETO_TASK_RULES)}"
                )
            task_rule_section = (
                "\n            ### Task Rule\n"
                f"            {PARETO_TASK_RULES[question_type]}\n"
            )
        else:
            task_rule_section = ""

        evidence_rules = []
        if "A" in condition.pareto_factors:
            evidence_rules.append(
                PARETO_BOUNDARY_RULE.format(start=start, end=end)
            )
        if "B" in condition.pareto_factors:
            evidence_rules.append(PARETO_CALIBRATION_RULE)
        if "D" in condition.pareto_factors:
            evidence_rules.append(PARETO_NUMERIC_RULE)
        if evidence_rules:
            evidence_use_section = (
                "\n            ### Evidence Use\n"
                + "\n".join(
                    f"            - {rule}" for rule in evidence_rules
                )
                + "\n"
            )
        else:
            evidence_use_section = ""

        if "E" in condition.pareto_factors:
            response_rule_section = (
                "\n            ### Response Rule\n"
                f"            {PARETO_SINGLE_ANSWER_RULE}\n"
            )
        else:
            response_rule_section = ""

        return f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.
{evidence_use_section}
            ### Time Series Data
            - **Window:** Steps {start} to {end}
            - **Values (scaled by 100):** {serialized_values}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_hint_tokens}
            - **{fixed_label}:** {fixed_hint_tokens}
{task_rule_section}{response_rule_section}
            ### Question
            {question}
            """
    if condition.use_expert_anomaly_analyst_opening:
        if not condition.standalone_fixed_hint:
            raise ValueError(
                "The approved expert-anomaly prompt requires a standalone "
                "fixed-hint section"
            )

        if condition.interleave_time_series_evidence:
            aligned_rows_indented = aligned_rows.replace(
                "\n", "\n            "
            )
            evidence_section = f"""### Time-Series Evidence

            Each row has the format:
            step | observed value | aligned per-step latent evidence

            {aligned_rows_indented}"""
        else:
            evidence_section = f"""### Time Series Data
            - **Window:** Steps {start} to {end}
            - **Values (scaled by 100):** {serialized_values}

            ### Contextual Hints
            - **Per-Step Analysis:** {local_hint_tokens}"""

        return f"""
            {EXPERT_ANOMALY_ANALYST_OPENING}
{contract_section}
            {evidence_section}

            ### {fixed_label}

            {fixed_hint_tokens}

            ### Question
            {question}
            """

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
