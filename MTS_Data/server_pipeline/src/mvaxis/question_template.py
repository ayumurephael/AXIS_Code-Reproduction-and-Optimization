from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

DIFFICULTIES = ("anomaly_frame", "normal_frame")
ANSWER_TYPES = ("judgment", "choice", "open")


@dataclass(frozen=True)
class QuestionSpec:
    """Structured description of one question template and its answer style."""

    question_id: str
    difficulty: str
    answer_type: str
    family: str
    text: str
    choices: Sequence[str] = ()


QUESTION_FAMILIES: Dict[str, Dict[str, str]] = {
    "anomaly_frame": {
        "definition": "Ask from an anomaly-oriented perspective while still grounding the question in evidence from the plotted traces and the interval's rough numeric behavior.",
        "focus": "Prefer anomaly, root-cause, affected-channel, morphology, or detector-support framing, but keep the question answerable from the highlighted interval.",
    },
    "normal_frame": {
        "definition": "Ask from a normality-oriented perspective while still grounding the question in evidence from the plotted traces and the interval's rough numeric behavior.",
        "focus": "Prefer normality, benign variation, false-alarm, context-fit, or non-over-attribution framing, but keep the question answerable from the highlighted interval.",
    },
}


FRAME_TEMPLATE_BANK: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "anomaly_frame": {
        "judgment": [
            {
                "key": "anomaly_presence",
                "focus_type": "existence",
                "instruction": "Ask a yes or no question about whether the highlighted interval supports an anomalous interpretation. The user should need to cite the strongest trace, timing, shape, or rough-magnitude cue rather than rely on vague suspicion.",
                "template": "Does the highlighted interval contain enough evidence from the plotted traces and rough interval behavior to support calling it anomalous? Answer yes or no and cite the strongest cue.",
            },
            {
                "key": "root_cause_candidate",
                "focus_type": "root_source",
                "instruction": "Ask whether any channel can be defended as the primary abnormal source candidate from timing, shape, and rough-magnitude cues. Do not force the question to assume a root candidate exists.",
                "template": "Is there enough evidence from the plotted traces and rough interval behavior to defend one channel as the primary abnormal source candidate in this interval? Answer yes or no and justify the choice if one exists.",
            },
            {
                "key": "multi_channel_scope",
                "focus_type": "scope_classification",
                "instruction": "Ask whether the interval supports a multi-channel anomaly rather than an isolated one-channel event. The question should stay observational and not require a hidden graph.",
                "template": "Does the highlighted interval support a multi-channel anomaly rather than an isolated one-channel event? Answer yes or no and explain the main trace or magnitude pattern.",
            },
            {
                "key": "morphology_match",
                "focus_type": "morphology",
                "instruction": "Ask whether a recognizable anomaly morphology can be defended from the highlighted interval, such as spike, level shift, oscillation, burst, or timing mismatch.",
                "template": "Can a recognizable anomaly morphology be defended from the highlighted interval? Answer yes or no and name the best candidate shape if possible.",
            },
            {
                "key": "detector_support",
                "focus_type": "detector_reliability",
                "instruction": "Ask whether the highlighted interval provides enough evidence to support an anomaly detector proposal. The answer should weigh both signal and uncertainty.",
                "template": "Does the highlighted interval provide enough evidence to support an anomaly detector proposal here? Answer yes or no and cite the main evidence for or against it.",
            },
            {
                "key": "root_claim_sufficiency",
                "focus_type": "root_claim",
                "instruction": "Ask exactly one short yes/no question about whether one channel can be justified as the primary abnormal source. Use a single-clause judgment only. Do not add a reason clause and do not use 'because', 'given', 'despite', 'while', or 'rather than'.",
                "template": "Is one channel justified as the primary abnormal source in this interval?",
            },
            {
                "key": "anomaly_scope",
                "focus_type": "scope",
                "instruction": "Ask exactly one short yes/no question about whether the evidence supports a broad or multi-channel anomaly claim. The question must still be answerable with 'No' when no anomaly is present. Keep the question on scope only and do not append justification clauses.",
                "template": "Does the evidence support a multi-channel anomaly claim here?",
            },
            {
                "key": "affected_scope",
                "focus_type": "affected_scope",
                "instruction": "Ask exactly one short yes/no question about whether more than one channel should be treated as affected. Do not mix this with root-cause, anomaly-type, or stability judgments. Do not add a second clause or a reason tail.",
                "template": "Should more than one channel be treated as affected in this interval?",
            },
            {
                "key": "stability_profile",
                "focus_type": "channel_stability",
                "instruction": "Ask exactly one short yes/no question about whether the instability stays localized or becomes coordinated across several channels. Keep the question on stability profile only. Do not add extra comparisons, explanations, or fallback clauses.",
                "template": "Does the instability stay localized in this interval?",
            },
        ],
        "choice": [
            {
                "key": "anomaly_presence",
                "focus_type": "existence",
                "instruction": "Ask a multiple-choice question about the safest anomaly-versus-alert interpretation. Include exactly one benign/no-alert option, exactly one manual-review option, and two positive anomaly interpretations that differ by candidate channel or by single-channel versus multi-channel scope.",
                "template": "Choose the safest anomaly-support interpretation for the highlighted interval.",
                "choices": (
                    "A. the interval is better treated as normal or benign",
                    "B. the anomaly is best defended as a one-channel event around one named channel",
                    "C. the anomaly is best defended as a multi-channel event around a concrete named group",
                    "D. the interval should stay under manual review",
                ),
            },
            {
                "key": "root_scope",
                "focus_type": "root_source",
                "instruction": "Ask a multiple-choice question about the safest primary abnormal-source claim. When the interval supports it, use concrete channel ids or channel groups rather than vague phrases. At least one distractor should name a plausible alternative channel or small alternative group, and another should overstate the scope.",
                "template": "Choose the safest primary abnormal-source claim for the highlighted interval.",
                "choices": (
                    "A. no defensible primary source is visible",
                    "B. one named channel is the strongest primary-source candidate",
                    "C. a different named channel or small alternative named set is more plausible than the focal one-channel claim",
                    "D. a broader named channel group is more plausible than a single primary source",
                ),
            },
            {
                "key": "affected_scope",
                "focus_type": "channel_scope",
                "instruction": "Ask a multiple-choice question about how many channels look involved if the interval is treated as anomalous. When the interval supports it, use concrete channel ids or channel groups rather than vague phrases. Include at least one distractor that names a plausible alternative channel or small group, and one that gets the scope too broad.",
                "template": "Choose the best visible affected-channel scope for this interval.",
                "choices": (
                    "A. no channel shows convincing abnormal involvement",
                    "B. one named channel looks directly involved",
                    "C. a small named channel group looks directly involved",
                    "D. a broader named channel group looks involved",
                ),
            },
            {
                "key": "morphology_family",
                "focus_type": "morphology",
                "instruction": "Ask a multiple-choice question about the dominant anomaly family. The options should be observable shape families rather than abstract severity words.",
                "template": "Choose the dominant anomaly-family explanation for the highlighted interval.",
                "choices": (
                    "A. no defensible abnormal shape",
                    "B. spike or abrupt local excursion",
                    "C. level shift, drift, or reversal",
                    "D. oscillation, burst, or timing inconsistency",
                ),
            },
            {
                "key": "detector_interpretation",
                "focus_type": "detector_reliability",
                "instruction": "Ask a multiple-choice question about how the highlighted evidence should affect detector trust. When a specific channel is central to the judgment, name it directly. Include exactly one no-alert option, exactly one manual-review option, and two valid-alert interpretations that differ by candidate channel or by single-channel versus multi-channel scope.",
                "template": "Choose the safest detector interpretation for the highlighted interval.",
                "choices": (
                    "A. no alert should be trusted from this evidence",
                    "B. a valid alert should stay local to one named channel",
                    "C. a valid alert should cover a concrete named channel group",
                    "D. the evidence should stay under manual review",
                ),
            },
            {
                "key": "anomaly_scope",
                "focus_type": "scope",
                "instruction": "Choose the safest anomaly-scope interpretation using concrete channel names or groups rather than vague phrases. Include one too-broad distractor and one plausible alternative small-group distractor rather than making every option orbit the same focal channel only.",
                "template": "Which anomaly-scope interpretation is best supported by the highlighted interval?",
                "choices": (
                    "A. no anomaly",
                    "B. one-channel local anomaly around one named channel",
                    "C. small multi-channel anomaly around a concrete named group or alternative candidate cluster",
                    "D. broad multi-channel anomaly across a larger named group",
                ),
            },
            {
                "key": "affected_scope",
                "focus_type": "affected_scope",
                "instruction": "Choose the safest affected-channel claim. Options should stay balanced in specificity and name concrete channels or channel groups. Include a plausible alternative small-group distractor and a too-broad distractor.",
                "template": "Which affected-channel claim is best supported by the highlighted interval?",
                "choices": (
                    "A. no channel shows convincing abnormal involvement",
                    "B. only one named channel looks directly involved",
                    "C. the affected set is concentrated in a concrete named group or alternative candidate cluster",
                    "D. the affected set is broad across a larger named group",
                ),
            },
            {
                "key": "stability_profile",
                "focus_type": "channel_stability",
                "instruction": "Choose the best channel-stability description. Make all options equally concrete, with named channels or channel sets instead of one vivid answer and several vague distractors.",
                "template": "Which channel-stability description best withstands scrutiny?",
                "choices": (
                    "A. all channels remain broadly stable",
                    "B. one named channel is the main unstable channel",
                    "C. a concrete small named group shares the clearest instability",
                    "D. instability is broad across a larger named group",
                ),
            },
            {
                "key": "root_claim_sufficiency",
                "focus_type": "claim_strength",
                "instruction": "Choose how strong the source/causal claim should be. The options should distinguish no claim, local claim, small-group claim, and broad-event claim at the same granularity, with at least one distractor naming an alternative candidate channel or cluster.",
                "template": "How strong a source-level claim is justified by the current evidence?",
                "choices": (
                    "A. no root-cause claim is justified",
                    "B. only a local claim about one named channel is justified",
                    "C. only a small-group claim about a concrete named group or alternative candidate cluster is justified",
                    "D. only a broad-event claim about a larger named group is justified",
                ),
            },
        ],
        "open": [
            {
                "key": "anomaly_evidence",
                "focus_type": "existence",
                "instruction": "Ask exactly one open question about the strongest evidence for an anomalous interpretation. Keep the task to one phenomenon only and do not add a second request about uncertainty, counterarguments, or extra evidence.",
                "template": "What evidence in the highlighted interval most strongly supports an anomalous interpretation?",
            },
            {
                "key": "root_candidate_comparison",
                "focus_type": "root_source",
                "instruction": "Ask exactly one open question about which channel is the strongest primary abnormal-source candidate. The question may ask for a brief justification, but it must not ask for a second comparison task.",
                "template": "Which channel, if any, is the strongest primary abnormal-source candidate in the highlighted interval, and why?",
            },
            {
                "key": "root_vs_affected",
                "focus_type": "root_affected_separation",
                "instruction": "Ask exactly one open question about which channels should be treated as directly abnormal. Do not also ask for affected-channel analysis, distractor analysis, or multiple separate classifications.",
                "template": "If the interval is treated as anomalous, which channels look directly abnormal?",
            },
            {
                "key": "morphology_and_timing",
                "focus_type": "morphology",
                "instruction": "Ask exactly one open question about the main anomaly morphology in the highlighted interval. Keep the task to describing one main pattern rather than multiple sub-questions.",
                "template": "How would you describe the main anomaly morphology in the highlighted interval?",
            },
            {
                "key": "anomaly_vs_benign",
                "focus_type": "context_comparison",
                "instruction": "Ask exactly one open question about the main evidence separating an anomaly interpretation from a benign/background explanation. Do not also ask what would weaken the claim.",
                "template": "What evidence most clearly separates an anomaly interpretation from a benign/background explanation here?",
            },
            {
                "key": "root_claim_sufficiency",
                "focus_type": "root_claim",
                "instruction": "Ask exactly one short open question about the strongest root-cause candidate, if any. Keep the task to primary source attribution only. Do not append a second request for evidence, comparisons, or exclusions.",
                "template": "Which channel, if any, is the strongest root-cause candidate here?",
            },
            {
                "key": "affected_scope",
                "focus_type": "affected_scope",
                "instruction": "Ask exactly one short open question about which channels should be treated as affected. Keep the task to affected-channel scope only. Do not add a second clause asking why, what evidence, or which channels to exclude.",
                "template": "Which channels, if any, should be treated as affected here?",
            },
            {
                "key": "anomaly_scope",
                "focus_type": "scope",
                "instruction": "Ask exactly one short open question about what anomaly scope, if any, is supported. The question must remain answerable when no anomaly is present. Keep the task to scope only. Do not enumerate candidate scope labels and do not append a second request for evidence.",
                "template": "What anomaly scope, if any, is supported here?",
            },
            {
                "key": "stability_profile",
                "focus_type": "channel_stability",
                "instruction": "Ask exactly one short open question about which channels contribute most to the instability pattern. Keep the task to stability profile only. Do not append extra comparisons, scope questions, or evidence requests.",
                "template": "Which channels look most unstable here?",
            },
        ],
    },
    "normal_frame": {
        "judgment": [
            {
                "key": "normality_presence",
                "focus_type": "normality_confirmation",
                "instruction": "Ask a yes or no question about whether the interval can reasonably be treated as normal or benign. Keep the question on the claim itself and leave the supporting evidence to the answer.",
                "template": "Can the highlighted interval reasonably be treated as normal or benign?",
            },
            {
                "key": "context_consistency",
                "focus_type": "context_comparison",
                "instruction": "Ask whether the interval remains consistent with surrounding or broader context rather than creating a distinct departure. Keep the question on that decision and leave the evidence to the answer.",
                "template": "Is the highlighted interval consistent with its surrounding or broader context rather than creating a distinct departure?",
            },
            {
                "key": "absence_of_local_anomaly",
                "focus_type": "anomaly_absence",
                "instruction": "Ask whether obvious local anomaly signals are absent in the highlighted interval. Do not list concrete signal types inside the question text.",
                "template": "Are clear local anomaly signals absent in the highlighted interval?",
            },
            {
                "key": "false_alarm_check",
                "focus_type": "false_positive_check",
                "instruction": "Ask whether a detector flag would more likely be a false alarm than a true anomaly. Keep the question on that judgment and leave the supporting evidence to the answer.",
                "template": "If a detector flagged this interval, would a false-alarm interpretation be more defensible than a true-anomaly interpretation?",
            },
            {
                "key": "timing_alignment",
                "focus_type": "timing_alignment",
                "instruction": "Ask whether related channels remain temporally aligned or preserve plausible lag structure. Keep the question on the temporal-alignment claim and leave the evidence to the answer.",
                "template": "Do related channels remain temporally aligned or preserve plausible lag structure in the highlighted interval?",
            },
            {
                "key": "root_claim_sufficiency",
                "focus_type": "root_claim",
                "instruction": "Ask exactly one short yes/no question about whether the evidence is too weak to justify any root-cause claim. Keep the question on root-claim sufficiency only. Do not append competing explanations, reason clauses, or a second judgment.",
                "template": "Is the evidence too weak to justify any root-cause claim in this interval?",
            },
            {
                "key": "anomaly_scope",
                "focus_type": "scope",
                "instruction": "Ask exactly one short yes/no question about whether the evidence fails to support a broad or multi-channel anomaly claim. Keep the question on scope only and do not append a second clause.",
                "template": "Does the evidence fail to support a broad multi-channel anomaly claim here?",
            },
            {
                "key": "affected_scope",
                "focus_type": "affected_scope",
                "instruction": "Ask exactly one short yes/no question about whether the evidence is too narrow or too ambiguous to call several channels affected. Do not add any second clause or reason tail.",
                "template": "Is the evidence too narrow or too ambiguous to call several channels affected?",
            },
            {
                "key": "stability_profile",
                "focus_type": "channel_stability",
                "instruction": "Ask exactly one short yes/no question about whether the instability still looks too mild or local for a stronger multi-channel claim. Keep the question on stability profile only. Do not add extra comparisons or explanation clauses.",
                "template": "Is the instability here too mild or too local for a stronger multi-channel claim?",
            },
        ],
        "choice": [
            {
                "key": "normality_label",
                "focus_type": "normality_confirmation",
                "instruction": "Ask a multiple-choice question about how safely the interval can be described as normal. Include both suspicious and anomalous alternatives so the wording does not force a normal answer.",
                "template": "Choose the safest normality-oriented interpretation of the highlighted interval.",
                "choices": (
                    "A. clearly consistent with normal behavior",
                    "B. mostly benign with mild variation",
                    "C. suspicious but not enough evidence for a strong anomaly claim",
                    "D. better treated as anomalous",
                ),
            },
            {
                "key": "channel_stability",
                "focus_type": "channel_stability",
                "instruction": "Ask a multiple-choice question about whether channels stay stable, mildly variable, or clearly unstable. When the interval supports it, use concrete channel ids or channel groups rather than vague phrases.",
                "template": "Choose the best channel-stability description for the highlighted interval.",
                "choices": (
                    "A. channels are broadly stable",
                    "B. channels show mild but still plausible variation",
                    "C. one named channel looks noticeably unstable",
                    "D. a concrete named channel group looks unstable",
                ),
            },
            {
                "key": "context_fit",
                "focus_type": "context_comparison",
                "instruction": "Ask a multiple-choice question about how the interval fits surrounding context, including an option for clear deviation.",
                "template": "Choose how the highlighted interval fits the surrounding context.",
                "choices": (
                    "A. fully consistent with surrounding context",
                    "B. mostly consistent with minor differences",
                    "C. context is too incomplete to judge safely",
                    "D. clearly departs from surrounding context",
                ),
            },
            {
                "key": "false_alarm_interpretation",
                "focus_type": "false_positive_check",
                "instruction": "Ask a multiple-choice question about the safest alert handling in a normality-oriented review. Include exactly one no-alert option, exactly one manual-review option, and two alert interpretations that differ by candidate channel or by single-channel versus multi-channel scope.",
                "template": "Choose the safest alert interpretation for this normality-oriented review.",
                "choices": (
                    "A. no alert should be raised here",
                    "B. if escalated, only a one-channel alert would be defensible",
                    "C. if escalated, only a multi-channel alert would be defensible",
                    "D. the interval should stay under manual review",
                ),
            },
            {
                "key": "fluctuation_level",
                "focus_type": "noise_variability",
                "instruction": "Ask a multiple-choice question about whether the interval looks smooth, mildly noisy, still plausible, or too bursty to be comfortably normal.",
                "template": "Choose the best fluctuation-level description for the highlighted interval.",
                "choices": (
                    "A. smooth or low-variance behavior",
                    "B. mild normal-looking fluctuation",
                    "C. moderate but still plausible variation",
                    "D. bursty or abnormal-looking fluctuation",
                ),
            },
            {
                "key": "anomaly_scope",
                "focus_type": "scope",
                "instruction": "Choose the safest scope interpretation in a conservative review. The options should remain concrete and balanced, with one plausible alternative small-group distractor and one too-broad distractor.",
                "template": "Which anomaly-scope interpretation is safest under a conservative review?",
                "choices": (
                    "A. no anomaly",
                    "B. one-channel local anomaly around one named channel",
                    "C. small multi-channel anomaly around a concrete named group or alternative candidate cluster",
                    "D. broad multi-channel anomaly across a larger named group",
                ),
            },
            {
                "key": "affected_scope",
                "focus_type": "affected_scope",
                "instruction": "Choose the safest affected-channel claim under a conservative review, using concrete channel sets instead of vague phrases. Include a plausible alternative small-group distractor and a too-broad distractor.",
                "template": "Which affected-channel claim is safest under a conservative review?",
                "choices": (
                    "A. no channel shows convincing abnormal involvement",
                    "B. only one named channel looks directly involved",
                    "C. the affected set is concentrated in a concrete named group or alternative candidate cluster",
                    "D. the affected set is broad across a larger named group",
                ),
            },
            {
                "key": "stability_profile",
                "focus_type": "channel_stability",
                "instruction": "Choose the safest channel-stability description, keeping all options equally concrete and plausible.",
                "template": "Which channel-stability description is safest under a conservative review?",
                "choices": (
                    "A. all channels remain broadly stable",
                    "B. one named channel is the main unstable channel",
                    "C. a concrete small named group shares the clearest instability",
                    "D. instability is broad across a larger named group",
                ),
            },
            {
                "key": "root_claim_sufficiency",
                "focus_type": "claim_strength",
                "instruction": "Choose how strong a source/causal claim is justified under a conservative review, distinguishing no claim, local claim, small-group claim, and broad-event claim, with one distractor naming an alternative candidate channel or cluster.",
                "template": "How strong a source-level claim is justified under a conservative review?",
                "choices": (
                    "A. no root-cause claim is justified",
                    "B. only a local claim about one named channel is justified",
                    "C. only a small-group claim about a concrete named group or alternative candidate cluster is justified",
                    "D. only a broad-event claim about a larger named group is justified",
                ),
            },
        ],
        "open": [
            {
                "key": "normality_explanation",
                "focus_type": "normality_confirmation",
                "instruction": "Ask exactly one open question about why a normal or benign interpretation is defensible. Do not preview the benign evidence in the question text and do not combine it with a second request.",
                "template": "Why is a normal or benign interpretation defensible for the highlighted interval?",
            },
            {
                "key": "channel_stability_explanation",
                "focus_type": "channel_stability",
                "instruction": "Ask exactly one open question about which channel or channels look least stable in the highlighted interval. Keep the task to one stability judgment only.",
                "template": "Which channel or channels look least stable in the highlighted interval?",
            },
            {
                "key": "context_fit_explanation",
                "focus_type": "context_comparison",
                "instruction": "Ask exactly one open question about why the interval fits the surrounding or broader pattern. Do not ask for an additional comparison task or preview the evidence in the question text.",
                "template": "Why does the highlighted interval fit the surrounding or broader pattern?",
            },
            {
                "key": "false_alarm_reasoning",
                "focus_type": "false_positive_check",
                "instruction": "Ask exactly one open question about why an anomaly alert on the interval could be treated as a false alarm. Do not add a second request about counter-evidence or preview the benign evidence in the question text.",
                "template": "If someone flagged this interval as anomalous, why might that flag be treated as a false alarm?",
            },
            {
                "key": "normal_uncertainty",
                "focus_type": "uncertainty",
                "instruction": "Ask exactly one open question about what evidence is missing for a strong anomaly claim. Do not combine benign support, uncertainty, and missing-evidence requests in the same question.",
                "template": "What evidence is missing for a strong anomaly claim in the highlighted interval?",
            },
            {
                "key": "root_claim_sufficiency",
                "focus_type": "root_claim",
                "instruction": "Ask exactly one short open question about what limits a strong root-cause claim. Keep the task to root-claim sufficiency only. Do not turn it into a debate question and do not append extra requests about evidence, distractors, or affected channels.",
                "template": "What limits a strong root-cause claim here?",
            },
            {
                "key": "affected_scope",
                "focus_type": "affected_scope",
                "instruction": "Ask exactly one short open question about what limits the affected-channel claim. Keep the task to affected-channel scope only. Do not append root-cause, anomaly-type, or evidence requests.",
                "template": "What limits the affected-channel claim here?",
            },
            {
                "key": "anomaly_scope",
                "focus_type": "scope",
                "instruction": "Ask exactly one short open question about what keeps the interval from supporting a broad or multi-channel anomaly claim. Keep the task to scope only. Do not append extra questions about timing, channel lists, or supporting evidence.",
                "template": "What keeps this interval from supporting a broad multi-channel anomaly claim?",
            },
            {
                "key": "stability_profile",
                "focus_type": "channel_stability",
                "instruction": "Ask exactly one short open question about what keeps the interval from supporting a stronger multi-channel instability claim. Keep the task to stability profile only. Do not append extra scope, attribution, or evidence requests.",
                "template": "What keeps this interval from supporting a stronger multi-channel instability claim?",
            },
        ],
    },
}


def _with_level(level: str, fmt: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(entry)
    item["level"] = level
    item["format"] = fmt
    return item


QUESTION_FOCUS_BANK: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "anomaly_frame": {
        fmt: [_with_level("anomaly_frame", fmt, item) for item in items]
        for fmt, items in FRAME_TEMPLATE_BANK["anomaly_frame"].items()
    }
}

NORMAL_FOCUS_BANK: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "normal_frame": {
        fmt: [_with_level("normal_frame", fmt, item) for item in items]
        for fmt, items in FRAME_TEMPLATE_BANK["normal_frame"].items()
    }
}


QUESTION_KIND_TO_FORMAT = {
    "multiple_choice": "choice",
    "true_false": "judgment",
    "open_ended": "open",
}

ANSWER_TYPE_TO_QUESTION_KIND = {
    "choice": "multiple_choice",
    "judgment": "true_false",
    "open": "open_ended",
}


def _make_specs() -> List[QuestionSpec]:
    """Build fixed question specifications from the frame-based template bank."""

    specs: List[QuestionSpec] = []
    for difficulty in DIFFICULTIES:
        for answer_type in ANSWER_TYPES:
            entries = FRAME_TEMPLATE_BANK[difficulty][answer_type]
            for idx, entry in enumerate(entries, start=1):
                specs.append(
                    QuestionSpec(
                        question_id=f"{difficulty}_{answer_type}_{idx:02d}_{entry['key']}",
                        difficulty=difficulty,
                        answer_type=answer_type,
                        family=str(entry["key"]),
                        text=str(entry["template"]),
                        choices=tuple(entry.get("choices") or ()),
                    )
                )
    return specs


QUESTION_SPECS: List[QuestionSpec] = _make_specs()


def fixed_question_specs() -> List[QuestionSpec]:
    """Return a copy of all fixed question specifications."""

    return list(QUESTION_SPECS)


def grouped_question_counts() -> Dict[str, Dict[str, int]]:
    """Count fixed question templates by frame and answer type."""

    counts: Dict[str, Dict[str, int]] = {}
    for spec in QUESTION_SPECS:
        counts.setdefault(spec.difficulty, {}).setdefault(spec.answer_type, 0)
        counts[spec.difficulty][spec.answer_type] += 1
    return counts


def sample_question_axes(n: int, seed: int = 0) -> List[Dict[str, str]]:
    """Sample balanced frame/answer-type axes for n examples."""

    rng = random.Random(seed)
    axes = [{"difficulty": d, "answer_type": a} for d in DIFFICULTIES for a in ANSWER_TYPES]
    selected: List[Dict[str, str]] = []
    while len(selected) < n:
        batch = list(axes)
        rng.shuffle(batch)
        selected.extend(batch)
    selected = selected[:n]
    rng.shuffle(selected)
    return selected


def sample_question_specs(
    n: int,
    seed: int = 0,
    *,
    difficulty: str | None = None,
    answer_type: str | None = None,
) -> List[QuestionSpec]:
    """Sample question templates, optionally constrained to one frame and answer type."""

    rng = random.Random(seed)
    by_bucket: Dict[tuple[str, str], List[QuestionSpec]] = {}
    for spec in QUESTION_SPECS:
        if difficulty and spec.difficulty != difficulty:
            continue
        if answer_type and spec.answer_type != answer_type:
            continue
        by_bucket.setdefault((spec.difficulty, spec.answer_type), []).append(spec)
    if not by_bucket:
        raise ValueError(f"No question specs match difficulty={difficulty}, answer_type={answer_type}")
    buckets = sorted(by_bucket)
    specs: List[QuestionSpec] = []
    while len(specs) < n:
        for bucket in buckets:
            specs.append(rng.choice(by_bucket[bucket]))
            if len(specs) >= n:
                break
    rng.shuffle(specs)
    return specs


def render_question(spec: QuestionSpec) -> str:
    """Render a question and append choices when the template has them."""

    if not spec.choices:
        return spec.text
    return spec.text + "\nChoices:\n" + "\n".join(spec.choices)
