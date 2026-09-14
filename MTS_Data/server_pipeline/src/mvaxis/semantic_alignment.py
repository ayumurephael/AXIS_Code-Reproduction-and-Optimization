from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def compact_text(text: Any, max_chars: Optional[int] = None) -> str:
    value = " ".join(str(text or "").replace("\u3000", " ").split())
    if max_chars is not None and max_chars > 0 and len(value) > max_chars:
        return value[: max_chars - 3].rstrip() + "..."
    return value


def extract_teacher_answer(row: Dict[str, Any]) -> str:
    teacher = row.get("teacher_answer_llm") or {}
    if isinstance(teacher, dict) and teacher.get("answer"):
        return compact_text(teacher.get("answer"))
    for key in ("model_answer", "windows_0_answer", "answer", "final_answer"):
        if row.get(key):
            return compact_text(row.get(key))
    target = row.get("target_output") or {}
    return compact_text(
        target.get("question_answer")
        or target.get("final_answer")
        or target.get("reasoning_summary")
        or ""
    )


def extract_student_answer(row: Dict[str, Any]) -> str:
    if row.get("raw_response"):
        return compact_text(row.get("raw_response"))
    parsed = row.get("parsed_student_answer") or row.get("parsed_response") or {}
    pieces: List[str] = []
    if isinstance(parsed, dict):
        for key in ("reasoning_process", "reasoning_summary", "analysis", "final_answer"):
            value = parsed.get(key)
            if isinstance(value, list):
                pieces.extend(str(item) for item in value)
            elif value:
                pieces.append(str(value))
    if pieces:
        return compact_text(" ".join(pieces))
    return compact_text(row.get("answer") or row.get("model_answer") or "")


def extract_final_answer_text(row: Dict[str, Any]) -> str:
    parsed = row.get("parsed_student_answer") or row.get("parsed_response") or {}
    if isinstance(parsed, dict) and parsed.get("final_answer"):
        return compact_text(parsed.get("final_answer"))
    return compact_text(row.get("final_answer") or row.get("answer_label") or row.get("raw_response") or "")


def tokenize_words(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def lexical_cosine(a: str, b: str) -> float:
    words_a = Counter(tokenize_words(a))
    words_b = Counter(tokenize_words(b))
    if not words_a or not words_b:
        return 0.0
    shared = set(words_a) & set(words_b)
    dot = sum(words_a[key] * words_b[key] for key in shared)
    norm_a = math.sqrt(sum(value * value for value in words_a.values()))
    norm_b = math.sqrt(sum(value * value for value in words_b.values()))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def summarize_scores(scores: Sequence[float]) -> Dict[str, float]:
    values = sorted(float(x) for x in scores if x is not None)
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    def pct(q: float) -> float:
        if len(values) == 1:
            return values[0]
        pos = q * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        frac = pos - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    return {
        "count": float(len(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(pct(0.5)),
        "p10": float(pct(0.1)),
        "p25": float(pct(0.25)),
        "p75": float(pct(0.75)),
        "p90": float(pct(0.9)),
        "min": float(values[0]),
        "max": float(values[-1]),
    }


def differentiable_hidden_semantic_loss(
    *,
    hidden_states: Any,
    input_ids: Any,
    labels: Any,
    embedding_weight: Any,
    torch_module: Any,
) -> Any:
    """Differentiable COT/answer semantic proxy for hint training.

    During teacher forcing, the hidden state at position t-1 predicts the target
    token at position t. This loss compares those prediction-side hidden states
    with the frozen target token embeddings by cosine distance. It is not a
    replacement for next-token CE; it is a smooth auxiliary signal that can send
    gradients through injected hint embeddings while the LLM weights stay frozen.
    """

    import torch.nn.functional as F

    target_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
    if target_positions.numel() == 0:
        return hidden_states.new_tensor(0.0)
    pred_positions = target_positions - 1
    keep = pred_positions >= 0
    if keep.sum().item() == 0:
        return hidden_states.new_tensor(0.0)
    pred_positions = pred_positions[keep]
    target_positions = target_positions[keep]
    pred_hidden = hidden_states[0, pred_positions].float()
    target_emb = embedding_weight[input_ids[0, target_positions]].detach().float()
    token_loss = 1.0 - F.cosine_similarity(pred_hidden, target_emb, dim=-1)
    pooled_loss = 1.0 - F.cosine_similarity(
        pred_hidden.mean(dim=0, keepdim=True),
        target_emb.mean(dim=0, keepdim=True),
        dim=-1,
    )
    return 0.5 * token_loss.mean() + 0.5 * pooled_loss.mean()
