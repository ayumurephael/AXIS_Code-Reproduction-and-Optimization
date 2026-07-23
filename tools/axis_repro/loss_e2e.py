"""Deterministic answer segmentation and memory-safe AXIS objectives.

This module contains the only intentional objective change used by the
``loss_e2e_0723`` experiment.  It is deliberately independent of the dataset
loader so that every boundary rule can be unit tested without loading the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


SEGMENT_IGNORE = 0
SEGMENT_CONCLUSION = 1
SEGMENT_EXPLANATION = 2
ANSWER_PREFIX = "Answer: "
ERROR_ANSWER = "Error generating answer."

_CONTENT_WORD = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?")
_SENTENCE_END = re.compile(r"[.!?](?:[\"')\]]*)(?=\s|$)")
_MC_MARKER = re.compile(r"\b(?:Explanation|Reasoning)\s*:\s*", re.IGNORECASE)
_BLANK_LINE = re.compile(r"\r?\n[ \t]*\r?\n")
_NEWLINE = re.compile(r"\r?\n")
_TF_LABEL = re.compile(r"^\s*(?:True|False)\b(?:\s*[.:;\-])?", re.IGNORECASE)
_MC_LABEL = re.compile(r"^\s*[A-D]\s*[).:\-]", re.IGNORECASE)


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True)
class AnswerSegmentation:
    question_type: str
    conclusion: Optional[Span]
    explanation: Optional[Span]
    rule: str
    fallback: bool


def normalize_question_type(value: str, answer: str = "") -> str:
    normalized = re.sub(r"[^a-z]+", "_", str(value).lower()).strip("_")
    if normalized in {"multiple_choice", "multiplechoice", "mc"} or (
        "multiple" in normalized and "choice" in normalized
    ):
        return "multiple_choice"
    if normalized in {"true_false", "truefalse", "tf"} or (
        "true" in normalized and "false" in normalized
    ):
        return "true_false"
    if normalized in {"open_ended", "openended", "oe"} or "open" in normalized:
        return "open_ended"

    # Fail-soft inference is deterministic and is reported as a fallback rule.
    if _TF_LABEL.match(answer):
        return "true_false"
    if _MC_LABEL.match(answer):
        return "multiple_choice"
    return "open_ended"


def _trim_span(text: str, start: int, end: int) -> Optional[Span]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return Span(start, end) if end > start else None


def _sentence_spans(text: str, start: int = 0, end: Optional[int] = None) -> List[Span]:
    end = len(text) if end is None else end
    spans: List[Span] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        span = _trim_span(text, cursor, match.end())
        if span is not None:
            spans.append(span)
        cursor = match.end()
    tail = _trim_span(text, cursor, end)
    if tail is not None:
        spans.append(tail)
    return spans


def _split_at(text: str, split_start: int, split_end: int, rule: str) -> AnswerSegmentation:
    return AnswerSegmentation(
        question_type="multiple_choice",
        conclusion=_trim_span(text, 0, split_start),
        explanation=_trim_span(text, split_end, len(text)),
        rule=rule,
        fallback=rule in {"mc_first_sentence", "mc_all"},
    )


def _segment_multiple_choice(answer: str) -> AnswerSegmentation:
    marker = _MC_MARKER.search(answer)
    if marker:
        return _split_at(answer, marker.start(), marker.end(), "mc_marker")

    blank = _BLANK_LINE.search(answer)
    if blank:
        return _split_at(answer, blank.start(), blank.end(), "mc_blank_line")

    newline = _NEWLINE.search(answer)
    if newline:
        candidate = _split_at(answer, newline.start(), newline.end(), "mc_first_line")
        if candidate.conclusion is not None and candidate.explanation is not None:
            return candidate

    sentences = _sentence_spans(answer)
    if len(sentences) >= 2:
        return _split_at(
            answer,
            sentences[0].end,
            sentences[0].end,
            "mc_first_sentence",
        )
    return AnswerSegmentation(
        "multiple_choice",
        _trim_span(answer, 0, len(answer)),
        None,
        "mc_all",
        True,
    )


def _segment_true_false(answer: str) -> AnswerSegmentation:
    """Put the label and first semantic statement in the conclusion.

    The explanation starts at the following semantic statement, so conclusion
    and explanation never overlap.
    """
    label = _TF_LABEL.match(answer)
    inferred = label is None
    if label is None:
        sentences = _sentence_spans(answer)
        if len(sentences) >= 2:
            return AnswerSegmentation(
                "true_false",
                sentences[0],
                _trim_span(answer, sentences[0].end, len(answer)),
                "tf_missing_label_first_sentence",
                True,
            )
        return AnswerSegmentation(
            "true_false",
            _trim_span(answer, 0, len(answer)),
            None,
            "tf_missing_label_all",
            True,
        )

    statements = _sentence_spans(answer, label.end())
    if statements:
        conclusion = _trim_span(answer, label.start(), statements[0].end)
        explanation = _trim_span(answer, statements[0].end, len(answer))
        return AnswerSegmentation(
            "true_false",
            conclusion,
            explanation,
            "tf_label_plus_first_statement",
            inferred,
        )
    return AnswerSegmentation(
        "true_false",
        _trim_span(answer, label.start(), label.end()),
        None,
        "tf_label_only",
        False,
    )


def _segment_open_ended(answer: str, short_threshold: int) -> AnswerSegmentation:
    sentences = _sentence_spans(answer)
    if not sentences:
        return AnswerSegmentation("open_ended", None, None, "oe_empty", True)

    conclusion_end = sentences[0].end
    rule = "oe_first_sentence"
    if len(_CONTENT_WORD.findall(answer[sentences[0].start:sentences[0].end])) < short_threshold:
        if len(sentences) >= 2:
            conclusion_end = sentences[1].end
            rule = "oe_short_merge_second"
        else:
            rule = "oe_short_single"
    return AnswerSegmentation(
        "open_ended",
        _trim_span(answer, sentences[0].start, conclusion_end),
        _trim_span(answer, conclusion_end, len(answer)),
        rule,
        rule == "oe_short_single",
    )


def segment_answer(
    answer: str,
    question_type: str,
    *,
    short_threshold: int = 5,
) -> AnswerSegmentation:
    kind = normalize_question_type(question_type, answer)
    if kind == "multiple_choice":
        return _segment_multiple_choice(answer)
    if kind == "true_false":
        return _segment_true_false(answer)
    return _segment_open_ended(answer, short_threshold)


def _overlap_length(token: Tuple[int, int], span: Optional[Span]) -> int:
    if span is None:
        return 0
    return max(0, min(token[1], span.end) - max(token[0], span.start))


def build_full_segment_ids(
    *,
    full_input_ids: torch.Tensor,
    answer_input_ids: torch.Tensor,
    answer_offsets: torch.Tensor,
    answer_block_start: int,
    answers: Sequence[str],
    question_types: Sequence[str],
    short_threshold: int = 5,
) -> Tuple[torch.Tensor, List[AnswerSegmentation]]:
    """Map deterministic character spans to the exact tokenized answer block."""
    batch_size, answer_width = answer_input_ids.shape
    if batch_size != len(answers) or batch_size != len(question_types):
        raise ValueError("answers/question_types must match the tokenized batch")
    block = full_input_ids[:, answer_block_start:answer_block_start + answer_width]
    if block.shape != answer_input_ids.shape or not torch.equal(
        block.detach().cpu(), answer_input_ids.detach().cpu()
    ):
        raise RuntimeError("answer tokenization no longer matches the full AXIS input")

    segment_ids = torch.zeros_like(full_input_ids, dtype=torch.long)
    segmentations: List[AnswerSegmentation] = []
    offsets_cpu = answer_offsets.detach().cpu().tolist()
    for row, (answer, question_type) in enumerate(zip(answers, question_types)):
        segmentation = segment_answer(
            answer,
            question_type,
            short_threshold=short_threshold,
        )
        segmentations.append(segmentation)
        conclusion = (
            None
            if segmentation.conclusion is None
            else Span(
                segmentation.conclusion.start + len(ANSWER_PREFIX),
                segmentation.conclusion.end + len(ANSWER_PREFIX),
            )
        )
        explanation = (
            None
            if segmentation.explanation is None
            else Span(
                segmentation.explanation.start + len(ANSWER_PREFIX),
                segmentation.explanation.end + len(ANSWER_PREFIX),
            )
        )
        for column, raw_offset in enumerate(offsets_cpu[row]):
            token_span = (int(raw_offset[0]), int(raw_offset[1]))
            if token_span[1] <= token_span[0]:
                continue  # BOS/EOS/padding/special separators have (0, 0).
            con_overlap = _overlap_length(token_span, conclusion)
            exp_overlap = _overlap_length(token_span, explanation)
            if con_overlap == 0 and exp_overlap == 0:
                continue  # Includes the literal ``Answer:`` prefix.
            segment_ids[row, answer_block_start + column] = (
                SEGMENT_CONCLUSION
                if con_overlap >= exp_overlap
                else SEGMENT_EXPLANATION
            )
    return segment_ids, segmentations


def _chunk_token_nll(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    lm_head: torch.nn.Module,
    chunk_size: int,
) -> torch.Tensor:
    """Return per-token FP32 NLL without retaining full-vocabulary logits."""
    chunks: List[torch.Tensor] = []
    for start in range(0, hidden.size(1), chunk_size):
        h = hidden[:, start:start + chunk_size, :]
        y = targets[:, start:start + chunk_size]

        def chunk_loss(hh: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
            logits = lm_head(hh).float()
            flat = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                yy.reshape(-1),
                ignore_index=-100,
                reduction="none",
            )
            return flat.view_as(yy)

        chunks.append(checkpoint(chunk_loss, h, y, use_reentrant=False))
    return torch.cat(chunks, dim=1)


def objective_from_hidden(
    *,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: torch.nn.Module,
    mode: str,
    valid_rows: torch.Tensor,
    segment_ids: Optional[torch.Tensor] = None,
    question_types: Optional[Sequence[str]] = None,
    alpha: float = 0.40,
    chunk_size: int = 64,
) -> Mapping[str, torch.Tensor]:
    """Build either the original global-NLL or row-balanced segmented objective.

    ``objective_sum`` keeps gradients.  All counts and diagnostic sums are
    tensors so the caller can aggregate them exactly across DDP ranks.
    """
    if mode not in {"control", "treatment"}:
        raise ValueError(f"unsupported objective mode: {mode}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    shifted_hidden = hidden[:, :-1, :]
    targets = labels[:, 1:]
    row_valid = valid_rows.to(device=hidden.device, dtype=torch.bool)
    if row_valid.numel() != hidden.size(0):
        raise ValueError("valid_rows must contain one value per QA row")

    token_nll = _chunk_token_nll(shifted_hidden, targets, lm_head, chunk_size)
    target_valid = targets.ne(-100) & row_valid[:, None]
    zero = token_nll.new_zeros(())
    token_sum = token_nll.masked_fill(~target_valid, 0).sum()
    token_count = target_valid.sum().to(torch.float32)

    result = {
        "token_nll_sum": token_sum,
        "token_count": token_count,
        "valid_row_count": row_valid.sum().to(torch.float32),
        "conclusion_mean_sum": zero,
        "conclusion_row_count": token_count.new_zeros(()),
        "explanation_mean_sum": zero,
        "explanation_row_count": token_count.new_zeros(()),
        "both_segment_row_count": token_count.new_zeros(()),
        "conclusion_only_row_count": token_count.new_zeros(()),
        "explanation_only_row_count": token_count.new_zeros(()),
        "empty_segment_row_count": token_count.new_zeros(()),
    }
    for kind in ("multiple_choice", "true_false", "open_ended"):
        result[f"{kind}_objective_sum"] = zero
        result[f"{kind}_row_count"] = token_count.new_zeros(())

    if mode == "control":
        result["objective_sum"] = token_sum
        result["objective_count"] = token_count
        return result

    if segment_ids is None:
        raise ValueError("treatment objective requires segment_ids")
    shifted_segments = segment_ids[:, 1:].to(hidden.device)
    con_mask = target_valid & shifted_segments.eq(SEGMENT_CONCLUSION)
    exp_mask = target_valid & shifted_segments.eq(SEGMENT_EXPLANATION)
    con_count = con_mask.sum(dim=1)
    exp_count = exp_mask.sum(dim=1)
    con_sum = token_nll.masked_fill(~con_mask, 0).sum(dim=1)
    exp_sum = token_nll.masked_fill(~exp_mask, 0).sum(dim=1)
    con_mean = con_sum / con_count.clamp_min(1)
    exp_mean = exp_sum / exp_count.clamp_min(1)
    has_con = con_count.gt(0)
    has_exp = exp_count.gt(0)
    both = has_con & has_exp & row_valid
    con_only = has_con & ~has_exp & row_valid
    exp_only = has_exp & ~has_con & row_valid
    usable = both | con_only | exp_only

    row_objective = torch.where(
        both,
        alpha * con_mean + (1.0 - alpha) * exp_mean,
        torch.where(con_only, con_mean, exp_mean),
    )
    result["objective_sum"] = row_objective.masked_fill(~usable, 0).sum()
    result["objective_count"] = usable.sum().to(torch.float32)
    result["conclusion_mean_sum"] = con_mean.masked_fill(~has_con, 0).sum()
    result["conclusion_row_count"] = (has_con & row_valid).sum().to(torch.float32)
    result["explanation_mean_sum"] = exp_mean.masked_fill(~has_exp, 0).sum()
    result["explanation_row_count"] = (has_exp & row_valid).sum().to(torch.float32)
    result["both_segment_row_count"] = both.sum().to(torch.float32)
    result["conclusion_only_row_count"] = con_only.sum().to(torch.float32)
    result["explanation_only_row_count"] = exp_only.sum().to(torch.float32)
    result["empty_segment_row_count"] = (row_valid & ~usable).sum().to(torch.float32)

    kinds = [
        normalize_question_type(value)
        for value in (question_types or [""] * hidden.size(0))
    ]
    for kind in ("multiple_choice", "true_false", "open_ended"):
        kind_mask = torch.tensor(
            [value == kind for value in kinds],
            dtype=torch.bool,
            device=hidden.device,
        ) & usable
        result[f"{kind}_objective_sum"] = row_objective.masked_fill(
            ~kind_mask, 0
        ).sum()
        result[f"{kind}_row_count"] = kind_mask.sum().to(torch.float32)
    return result


def count_segmentation_rules(
    answers: Iterable[str],
    question_types: Iterable[str],
) -> Mapping[str, int]:
    counts: dict[str, int] = {}
    for answer, question_type in zip(answers, question_types):
        item = segment_answer(answer, question_type)
        key = f"{item.question_type}/{item.rule}"
        counts[key] = counts.get(key, 0) + 1
        if item.fallback:
            fallback_key = f"{item.question_type}/fallback"
            counts[fallback_key] = counts.get(fallback_key, 0) + 1
    return counts
