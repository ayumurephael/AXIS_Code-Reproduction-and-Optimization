"""Deterministic supervision rules for the final MC/TF/OE joint objective.

The rules deliberately prefer precision over coverage. They decide only whether
a counterfactual pair receives an auxiliary loss; every row keeps the factual
full-answer loss.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch


SUPERVISION_CACHE_VERSION = 1

_TYPE_ALIASES = {
    "multiple_choice": "multiple_choice",
    "multiplechoice": "multiple_choice",
    "mc": "multiple_choice",
    "true_false": "true_false",
    "truefalse": "true_false",
    "tf": "true_false",
    "open_ended": "open_ended",
    "openended": "open_ended",
    "oe": "open_ended",
}

_TF_ATTRIBUTE_WORDS = {
    "upward", "downward", "spike", "drop", "shift", "drift",
    "amplitude", "periodic", "center", "centre", "boundary",
    "sustained", "consecutive", "duration", "magnitude", "step",
    "point", "position", "location", "increase", "decrease",
}

_NEGATIVE_STATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno\s+(?:clear\s+|significant\s+|obvious\s+)?anomal(?:y|ies|ous\s+(?:behavior|behaviour|pattern))\b",
        r"\bno\s+evidence\s+of\s+(?:an\s+)?anomal",
        r"\babsence\s+of\s+(?:an\s+)?anomal",
        r"\bfree\s+from\s+anomal",
        r"\bdoes\s+not\s+(?:contain|show|exhibit|indicate)\b[^.!?]{0,50}\banomal",
        r"\bnot\s+anomalous\b",
        r"\bconsistent\s+with\s+normal\b",
        r"\bnormal\s+(?:behavior|behaviour|fluctuations?|pattern)\b",
        r"\bappears?\s+(?:to\s+be\s+)?normal\b",
        r"\bwindow\s+is\s+normal\b",
    )
)

_POSITIVE_STATE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bthere\s+(?:is|are)\b[^.!?]{0,45}\banomal(?:y|ies)\b",
        r"\b(?:an\s+)?anomal(?:y|ies)\s+(?:is|are)\s+present\b",
        r"\b(?:contains?|shows?|exhibits?|indicates?|reveals?)\b[^.!?]{0,45}\banomal(?:y|ies|ous)\b",
        r"\bevidence\s+of\s+(?:an\s+)?anomal",
        r"\banomal(?:y|ies)\s+(?:is|are|was|were)\s+detected\b",
        r"\bdetected\s+(?:an\s+)?anomal",
        r"\bsignificant\s+anomal(?:y|ies|ous\s+(?:behavior|behaviour|pattern))\b",
        r"\babnormal\s+(?:behavior|behaviour|pattern|fluctuation)\b",
    )
)

_TF_NEGATIVE_PREDICATES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bno\s+(?:anomalous\s+(?:behavior|behaviour|pattern)|anomal(?:y|ies))\b",
        r"\babsence\s+of\s+anomal",
        r"\bfree\s+from\s+anomal",
        r"\bdoes\s+not\s+contain\b[^.!?]{0,30}\banomal",
        r"\bno\s+evidence\s+of\s+anomalous\s+(?:behavior|behaviour|patterns?)\b",
        r"\bdoes\s+not\s+(?:display|show|exhibit)\b[^.!?]{0,40}\bevidence\s+of\s+anomal",
        r"\bconsistent\s+with\s+normal\b",
        r"\bnormal\s+(?:behavior|behaviour|fluctuations?|pattern)\b",
        r"\bwindow\s+is\s+normal\b",
    )
)

_TF_POSITIVE_PREDICATES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bcontains?\b[^.!?]{0,30}\banomal(?:y|ies)\b",
        r"\bpresence\s+of\s+(?:an\s+)?anomal",
        r"\bevidence\s+of\s+anomalous\s+(?:behavior|behaviour)\b",
        r"\banomal(?:y|ies)\s+(?:is|are)\s+present\b",
        r"\bthere\s+(?:is|are)\b[^.!?]{0,30}\banomal(?:y|ies)\b",
        r"\bindicating\s+(?:an\s+)?anomal",
        r"\bshows?\s+anomalous\s+(?:behavior|behaviour)\b",
    )
)

_TF_AMBIGUOUS_SCOPE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:no|does\s+not)\b[^.!?]{0,120}\b(?:indicat(?:e|es|ing|ive)|suggests?)\b[^.!?]{0,80}\b(?:presence\s+of\s+)?(?:an\s+)?anomal",
        r"\bno\s+evidence\b[^.!?]{0,100}\bpresence\s+of\s+(?:an\s+)?anomal",
        r"\brather\s+than\b[^.!?]{0,100}\bnormal\b",
        r"\bunlikely\s+to\s+be\s+explained\s+by\s+normal\b",
    )
)

_OE_QUESTION_ASSERTIONS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bconsidering\s+the\s+presence\b",
        r"\bgiven\s+the\s+presence\b",
        r"\b(?:the|this|that)\s+(?:clear\s+|observed\s+|detected\s+|local\s+)?anomal(?:y|ies)\b",
        r"\b(?:the|this|that)\s+(?:upward|downward|spike|drop|shift|drift)\b",
    )
)

_OE_NEUTRAL_QUESTION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwhether\b[^?]{0,120}\banomal",
        r"\b(?:is|are)\s+there\b[^?]{0,100}\banomal",
        r"\bdoes\b[^?]{0,100}\bcontain\b[^?]{0,50}\banomal",
        r"\b(?:detect|identify|find)\b[^?]{0,80}\bany\s+anomal",
        r"\b(?:assess|determine|evaluate|analy[sz]e)\b[^?]{0,120}\b(?:anomal|normal)",
        r"\b(?:anomalies|anomaly)\s+(?:are|is)\s+present\b",
    )
)


def canonical_question_type(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    compact = normalized.replace("_", "")
    if normalized in _TYPE_ALIASES:
        return _TYPE_ALIASES[normalized]
    if compact in _TYPE_ALIASES:
        return _TYPE_ALIASES[compact]
    return "unknown"


def parse_tf_target(answer: str) -> int | None:
    text = re.sub(r"^\s*answer\s*:\s*", "", str(answer), flags=re.IGNORECASE)
    match = re.match(r"^\s*(true|false)\b", text, flags=re.IGNORECASE)
    return None if not match else int(match.group(1).lower() == "true")


def _matches_any(patterns: Iterable[re.Pattern[str]], text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _mask_matches(patterns: Iterable[re.Pattern[str]], text: str) -> str:
    """Blank matched spans so a negated assertion is not also read as positive."""
    characters = list(text)
    for pattern in patterns:
        for match in pattern.finditer(text):
            characters[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


def parse_tf_state_predicate(question: str) -> int | None:
    text = str(question).lower()
    if _matches_any(_TF_AMBIGUOUS_SCOPE, text):
        return None
    negative = _matches_any(_TF_NEGATIVE_PREDICATES, text)
    positive = _matches_any(
        _TF_POSITIVE_PREDICATES,
        _mask_matches(_TF_NEGATIVE_PREDICATES, text),
    )
    if negative == positive:
        return None
    return 0 if negative else 1


def first_two_sentences(text: str) -> str:
    pieces = [piece.strip() for piece in re.split(r"(?<=[.!?])\s+", str(text).strip()) if piece.strip()]
    return " ".join(pieces[:2])


def parse_oe_answer_state(answer: str) -> int | None:
    text = re.sub(
        r"^\s*answer\s*:\s*", "", first_two_sentences(answer), flags=re.IGNORECASE
    )
    if not text:
        return None
    negative = _matches_any(_NEGATIVE_STATE_PATTERNS, text)
    positive = _matches_any(
        _POSITIVE_STATE_PATTERNS,
        _mask_matches(_NEGATIVE_STATE_PATTERNS, text),
    )
    if negative == positive:
        return None
    return 0 if negative else 1


def oe_answer_format_valid(answer: str) -> bool:
    text = str(answer).strip()
    if not text or re.match(r"^(?:true|false)\b", text, flags=re.IGNORECASE):
        return False
    return re.match(r"^[A-D](?:\s*[.):]|\s*$)", text, flags=re.IGNORECASE) is None


def oe_question_counterfactual_valid(question: str) -> bool:
    text = str(question).strip()
    if not text or _matches_any(_OE_QUESTION_ASSERTIONS, text):
        return False
    return _matches_any(_OE_NEUTRAL_QUESTION_PATTERNS, text)


def build_supervision_record(window: dict[str, Any], counterfactual: dict[str, Any]) -> dict[str, Any]:
    question_type = canonical_question_type(window.get("question_type", "unknown"))
    state_target = int(bool(window.get("has_anomaly", False)))
    counterfactual_valid = bool(counterfactual.get("valid", False))
    kind = str(counterfactual.get("kind", "invalid"))
    record: dict[str, Any] = {
        "question_type": question_type,
        "state_target": state_target,
        "counterfactual_valid": counterfactual_valid,
        "counterfactual_kind": kind,
        "mc_pair_valid": question_type == "multiple_choice" and counterfactual_valid,
        "tf_target": None,
        "tf_predicate_polarity": None,
        "tf_format_valid": False,
        "tf_predicate_valid": False,
        "tf_metadata_consistent": False,
        "tf_pair_valid": False,
        "oe_answer_state": None,
        "oe_format_valid": False,
        "oe_question_valid": False,
        "oe_metadata_consistent": False,
        "oe_depend_valid": False,
        "oe_pair_valid": False,
    }
    if question_type == "true_false":
        target = parse_tf_target(window.get("answer", ""))
        polarity = parse_tf_state_predicate(window.get("question", ""))
        consistent = target is not None and polarity is not None and target == int(state_target == polarity)
        record.update({
            "tf_target": target,
            "tf_predicate_polarity": polarity,
            "tf_format_valid": target is not None,
            "tf_predicate_valid": polarity is not None,
            "tf_metadata_consistent": consistent,
            "tf_pair_valid": bool(counterfactual_valid and consistent),
        })
    if question_type == "open_ended":
        answer_state = parse_oe_answer_state(window.get("answer", ""))
        format_valid = oe_answer_format_valid(window.get("answer", ""))
        question_valid = oe_question_counterfactual_valid(window.get("question", ""))
        consistent = answer_state is not None and answer_state == state_target
        depend_valid = answer_state is not None
        deletion_only = state_target == 1 and kind == "self_normal_patch"
        record.update({
            "oe_answer_state": answer_state,
            "oe_format_valid": format_valid,
            "oe_question_valid": question_valid,
            "oe_metadata_consistent": consistent,
            "oe_depend_valid": depend_valid,
            "oe_pair_valid": bool(
                counterfactual_valid and deletion_only and format_valid
                and question_valid and consistent and depend_valid
            ),
        })
    return record


@dataclass(frozen=True)
class TFVerbalizers:
    false_text: str
    true_text: str
    false_id: int
    true_id: int

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


def select_tf_verbalizers(tokenizer) -> TFVerbalizers:
    for false_text, true_text in ((" False", " True"), ("False", "True")):
        false_ids = tokenizer.encode(false_text, add_special_tokens=False)
        true_ids = tokenizer.encode(true_text, add_special_tokens=False)
        if len(false_ids) == len(true_ids) == 1 and false_ids[0] != true_ids[0]:
            return TFVerbalizers(false_text, true_text, int(false_ids[0]), int(true_ids[0]))
    raise RuntimeError("True/False are not distinct single-token verbalizers")


def ddp_global_mean(local_sum: torch.Tensor, local_count: int | torch.Tensor) -> torch.Tensor:
    """Return a loss whose DDP-averaged gradient is the exact global mean."""
    count = torch.as_tensor(local_count, dtype=torch.float32, device=local_sum.device).detach().clone()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    if float(count.item()) == 0.0:
        return local_sum * 0.0
    return local_sum * float(world_size) / count
