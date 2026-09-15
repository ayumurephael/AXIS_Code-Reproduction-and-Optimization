from __future__ import annotations

import random
from typing import Any, Dict, List


DIFFICULTIES = ("anomaly_frame", "normal_frame")
ANSWER_TYPES = ("judgment", "choice", "open")


HARD_FRAME_TEMPLATE_BANK: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "anomaly_frame": {
        "judgment": [
            {
                "key": "lead_lag",
                "focus_type": "lead_lag",
                "instruction": "Ask exactly one short yes/no question about whether one named channel clearly deviates before another named channel responds. Keep the question on temporal ordering only. Do not append cause explanations or a second request for evidence.",
                "template": "Does one channel clearly deviate first, with another channel responding later in this interval?",
            },
            {
                "key": "propagation_path",
                "focus_type": "propagation_path",
                "instruction": "Ask exactly one short yes/no question about whether the anomaly looks like a single-point disturbance that spreads, rather than several channels becoming unstable at once. Keep the question on propagation pattern only.",
                "template": "Does this interval look like a disturbance that starts locally and then spreads across channels?",
            },
            {
                "key": "relation_break",
                "focus_type": "relation_break",
                "instruction": "Ask exactly one short yes/no question about whether a previously stable cross-channel relation appears broken. Do not turn it into a magnitude-only or root-cause-only question.",
                "template": "Does this interval show a break in an otherwise stable cross-channel relation?",
            },
            {
                "key": "counterfactual_exclusion",
                "focus_type": "counterfactual_exclusion",
                "instruction": "Ask exactly one short yes/no question about whether the anomaly judgment would still stand if one named focal channel were ignored. Keep the task to counterfactual support only.",
                "template": "Would the anomaly judgment still be defensible if the main focal channel were ignored?",
            },
        ],
        "choice": [
            {
                "key": "minimal_affected_set",
                "focus_type": "minimal_affected_set",
                "instruction": "Choose the smallest affected-channel set that is still defensible. Keep all options concrete and nested by scope, with one too-broad distractor.",
                "template": "Which minimal affected-channel set is best supported by the highlighted interval?",
                "choices": (
                    "A. no channel needs to be marked as affected",
                    "B. only one named channel needs to be marked as affected",
                    "C. a concrete small named set needs to be marked as affected",
                    "D. a broader named group needs to be marked as affected",
                ),
            },
            {
                "key": "root_vs_follower",
                "focus_type": "root_vs_follower",
                "instruction": "Choose the safest root-versus-follower interpretation. Options should separate one leading source, a small leading group, follower-only channels, and a no-clear-leader fallback.",
                "template": "Which root-versus-follower interpretation is best supported here?",
                "choices": (
                    "A. no clear leader can be separated from the rest",
                    "B. one named channel is the clearest leading/root candidate",
                    "C. a concrete small named group shares the strongest leading role",
                    "D. the named channels mostly look like followers rather than primary sources",
                ),
            },
            {
                "key": "stage_localization",
                "focus_type": "stage_localization",
                "instruction": "Choose the safest stage-localization summary, distinguishing onset, peak, and recovery involvement rather than only asking which channel is abnormal overall.",
                "template": "Which stage-localization summary best fits onset, peak, and recovery across the channels?",
                "choices": (
                    "A. the same one named channel dominates onset, peak, and recovery",
                    "B. one named channel leads onset, but a small named group is most visible at peak",
                    "C. a small named group is already involved at onset and remains involved through peak and recovery",
                    "D. stage boundaries are too mixed to localize confidently across channels",
                ),
            },
            {
                "key": "cross_channel_consistency",
                "focus_type": "cross_channel_consistency",
                "instruction": "Choose whether the channel evidence is mutually reinforcing or internally conflicting. Keep options concrete and balanced rather than making the conflict option obviously extreme.",
                "template": "Which cross-channel consistency description is best supported by the interval?",
                "choices": (
                    "A. the channel evidence is largely absent or too weak overall",
                    "B. one named channel gives the main support while the rest stay mostly neutral",
                    "C. a concrete small named group provides mutually consistent support",
                    "D. the channel evidence conflicts enough that a clean multichannel story is hard to defend",
                ),
            },
        ],
        "open": [
            {
                "key": "lead_lag",
                "focus_type": "lead_lag",
                "instruction": "Ask exactly one short open question about which channel moves first and which channels respond later. Keep the task to temporal ordering only. Do not append extra requests about root cause confidence or anomaly scope.",
                "template": "Which channel appears to move first here, and which channels respond later?",
            },
            {
                "key": "propagation_path",
                "focus_type": "propagation_path",
                "instruction": "Ask exactly one short open question about whether the pattern looks like a local disturbance that spreads or a near-simultaneous multichannel failure. Keep the task to propagation pattern only.",
                "template": "Does this pattern look more like local spread or near-simultaneous multichannel failure?",
            },
            {
                "key": "relation_break",
                "focus_type": "relation_break",
                "instruction": "Ask exactly one short open question about which channel relation, if any, appears broken. Keep the task to relation stability only. Do not append a second clause about anomaly type or affected scope.",
                "template": "Which cross-channel relation, if any, looks broken here?",
            },
            {
                "key": "minimal_affected_set",
                "focus_type": "minimal_affected_set",
                "instruction": "Ask exactly one short open question about the smallest channel set that still needs to be marked as affected. Keep the task to minimal set selection only.",
                "template": "What is the smallest channel set that still needs to be treated as affected here?",
            },
            {
                "key": "counterfactual_exclusion",
                "focus_type": "counterfactual_exclusion",
                "instruction": "Ask exactly one short open question about whether the anomaly judgment survives after removing one named focal channel from consideration. Keep the task to counterfactual support only.",
                "template": "If the main focal channel were ignored, would the remaining channels still support an anomaly judgment?",
            },
            {
                "key": "root_vs_follower",
                "focus_type": "root_vs_follower",
                "instruction": "Ask exactly one short open question about which channels look like primary drivers and which look like followers. Keep the task to role separation only.",
                "template": "Which channels look like primary drivers here, and which look more like followers?",
            },
            {
                "key": "stage_localization",
                "focus_type": "stage_localization",
                "instruction": "Ask exactly one short open question about which channels are most involved at onset, peak, and recovery. Keep the task to stage localization only.",
                "template": "Which channels are most involved at onset, peak, and recovery in this interval?",
            },
            {
                "key": "cross_channel_consistency",
                "focus_type": "cross_channel_consistency",
                "instruction": "Ask exactly one short open question about whether the channels tell a consistent multichannel story or conflicting stories. Keep the task to consistency only.",
                "template": "Do the channels tell a consistent multichannel story here, or do they conflict?",
            },
        ],
    },
    "normal_frame": {
        "judgment": [
            {
                "key": "lead_lag",
                "focus_type": "lead_lag",
                "instruction": "Ask exactly one short yes/no question about whether the evidence is too weak or too ambiguous to support a clean lead-lag ordering. Keep the task to temporal ordering only.",
                "template": "Is the evidence too weak or too ambiguous to support a clear lead-lag ordering here?",
            },
            {
                "key": "propagation_path",
                "focus_type": "propagation_path",
                "instruction": "Ask exactly one short yes/no question about whether the evidence fails to support a local-spread story across channels. Keep the task to propagation pattern only.",
                "template": "Does the evidence fail to support a clear local-spread story across channels here?",
            },
            {
                "key": "relation_break",
                "focus_type": "relation_break",
                "instruction": "Ask exactly one short yes/no question about whether the cross-channel relation still looks too intact for a strong relation-break claim. Keep the task to relation stability only.",
                "template": "Does the cross-channel relation still look too intact for a strong relation-break claim here?",
            },
            {
                "key": "counterfactual_exclusion",
                "focus_type": "counterfactual_exclusion",
                "instruction": "Ask exactly one short yes/no question about whether removing one named focal channel leaves too little support for an anomaly judgment. Keep the task to counterfactual support only.",
                "template": "If the main focal channel were ignored, would the remaining evidence become too weak for an anomaly judgment?",
            },
        ],
        "choice": [
            {
                "key": "minimal_affected_set",
                "focus_type": "minimal_affected_set",
                "instruction": "Choose the safest minimal affected set under a conservative review. Keep options concrete and nested by scope, with a no-channel option and a too-broad distractor.",
                "template": "Which minimal affected-channel set is safest under a conservative review?",
                "choices": (
                    "A. no channel needs to be marked as affected",
                    "B. only one named channel might still need to be marked",
                    "C. a concrete small named set might still need to be marked",
                    "D. a broader named group needs to be marked",
                ),
            },
            {
                "key": "root_vs_follower",
                "focus_type": "root_vs_follower",
                "instruction": "Choose the safest root-versus-follower interpretation under a conservative review. Keep options balanced and concrete rather than making the no-leader option trivial.",
                "template": "Which root-versus-follower interpretation is safest under a conservative review?",
                "choices": (
                    "A. no clear leader can be separated from the rest",
                    "B. one named channel is still the safest leading/root candidate",
                    "C. a concrete small named group still looks like the safest shared leader set",
                    "D. the named channels look more like followers or ambiguous bystanders than leaders",
                ),
            },
            {
                "key": "stage_localization",
                "focus_type": "stage_localization",
                "instruction": "Choose the safest stage-localization summary under a conservative review, distinguishing onset, peak, and recovery while allowing for uncertainty.",
                "template": "Which stage-localization summary is safest under a conservative review?",
                "choices": (
                    "A. no stage can be localized cleanly across channels",
                    "B. one named channel is the safest onset and peak anchor, with weak recovery evidence",
                    "C. a concrete small named group is still the safest stage-localized interpretation",
                    "D. different channels look mixed enough that stage localization remains ambiguous",
                ),
            },
            {
                "key": "cross_channel_consistency",
                "focus_type": "cross_channel_consistency",
                "instruction": "Choose the safest cross-channel consistency description under a conservative review. Keep options concrete and balanced, including a real conflict option.",
                "template": "Which cross-channel consistency description is safest under a conservative review?",
                "choices": (
                    "A. the channel evidence is too weak overall",
                    "B. one named channel is mildly suggestive while the rest stay mostly neutral",
                    "C. a concrete small named group is still mutually consistent",
                    "D. the channel evidence conflicts enough that no clean multichannel story is safe",
                ),
            },
        ],
        "open": [
            {
                "key": "lead_lag",
                "focus_type": "lead_lag",
                "instruction": "Ask exactly one short open question about what prevents a confident lead-lag conclusion. Keep the task to temporal ordering only.",
                "template": "What prevents a confident lead-lag conclusion here?",
            },
            {
                "key": "propagation_path",
                "focus_type": "propagation_path",
                "instruction": "Ask exactly one short open question about what keeps the interval from supporting a clear spread pattern across channels. Keep the task to propagation pattern only.",
                "template": "What keeps this interval from supporting a clear spread pattern across channels?",
            },
            {
                "key": "relation_break",
                "focus_type": "relation_break",
                "instruction": "Ask exactly one short open question about what keeps the interval from supporting a strong relation-break claim. Keep the task to relation stability only.",
                "template": "What keeps this interval from supporting a strong cross-channel relation-break claim?",
            },
            {
                "key": "minimal_affected_set",
                "focus_type": "minimal_affected_set",
                "instruction": "Ask exactly one short open question about what limits the minimal affected set under a conservative review. Keep the task to affected-set sufficiency only.",
                "template": "What limits the minimal affected-channel set that can be defended here?",
            },
            {
                "key": "counterfactual_exclusion",
                "focus_type": "counterfactual_exclusion",
                "instruction": "Ask exactly one short open question about what happens to the anomaly judgment if one named focal channel is removed. Keep the task to counterfactual support only.",
                "template": "If the main focal channel were removed, what would be left of the anomaly case?",
            },
            {
                "key": "root_vs_follower",
                "focus_type": "root_vs_follower",
                "instruction": "Ask exactly one short open question about what limits a confident root-versus-follower separation. Keep the task to role separation only.",
                "template": "What limits a confident separation between root channels and followers here?",
            },
            {
                "key": "stage_localization",
                "focus_type": "stage_localization",
                "instruction": "Ask exactly one short open question about what limits confident stage localization across onset, peak, and recovery. Keep the task to stage localization only.",
                "template": "What limits confident stage localization across onset, peak, and recovery here?",
            },
            {
                "key": "cross_channel_consistency",
                "focus_type": "cross_channel_consistency",
                "instruction": "Ask exactly one short open question about what makes the channel evidence look consistent or inconsistent under a conservative review. Keep the task to consistency only.",
                "template": "What makes the channel evidence look inconsistent or too weak for a clean multichannel story here?",
            },
        ],
    },
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


def sample_hard_question_axes(n: int, seed: int = 0) -> List[Dict[str, str]]:
    """Sample balanced answer-type axes for the hard workflow."""

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


def sample_hard_focus_item(question_type: str, difficulty: str, *, rng: random.Random | None = None) -> Dict[str, Any]:
    """Sample one hard focus item for the requested question kind and frame."""

    sampler = rng or random.Random()
    fmt = QUESTION_KIND_TO_FORMAT[question_type]
    frame = str(difficulty or "anomaly_frame")
    pool = HARD_FRAME_TEMPLATE_BANK[frame][fmt]
    item = dict(sampler.choice(pool))
    item["format"] = fmt
    item["level"] = frame
    return item
