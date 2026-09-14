from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .semantic_alignment import compact_text, lexical_cosine, summarize_scores


_CHOICE_HINT_RE = re.compile(r"(?im)^\s*[A-E][\)\.]\s+")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class EvalExample:
    example_id: str
    sample_id: Optional[str]
    source_format: str
    question: str
    student_answer: str
    teacher_answer: str
    question_type: str = "unknown"
    answer_type: str = "open"
    difficulty: str = "unknown"
    frame: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


def canonical_answer_type(value: Optional[str], *, question: str = "", teacher_answer: str = "") -> str:
    text = (value or "").strip().lower()
    if text in {"multiple_choice", "multiple-choice", "choice", "mcq"}:
        return "choice"
    if text in {"judgment", "judge", "boolean", "bool", "yes_no", "yes-no", "true_false", "true-false"}:
        return "judgment"
    if text in {"open", "open_ended", "open-ended", "freeform"}:
        return "open"

    lowered_question = question.lower()
    if "choices:" in lowered_question or _CHOICE_HINT_RE.search(question):
        return "choice"
    teacher_head = teacher_answer.strip().lower()
    if teacher_head.startswith("yes.") or teacher_head.startswith("no.") or teacher_head.startswith("yes ") or teacher_head.startswith("no "):
        return "judgment"
    if teacher_head.startswith("answer:") and re.search(r"(?i)\b[a-e]\b", teacher_head[:16]):
        return "choice"
    if "yes or no" in lowered_question:
        return "judgment"
    return "open"


def infer_difficulty(question_type: Optional[str], explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    text = str(question_type or "").lower()
    for key in ("simple", "medium", "complex"):
        if key in text:
            return key
    if "anomaly_frame" in text:
        return "anomaly_frame"
    if "normal_frame" in text:
        return "normal_frame"
    return "unknown"


def infer_frame(question_type: Optional[str], explicit: Optional[str] = None) -> str:
    if explicit:
        return str(explicit)
    text = str(question_type or "").lower()
    if "anomaly_frame" in text:
        return "anomaly_frame"
    if "normal_frame" in text:
        return "normal_frame"
    return "unknown"


def load_rubric_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if "rubrics" not in config:
        raise ValueError("Rubric config must contain a top-level 'rubrics' object.")
    return config


def get_rubric(config: Dict[str, Any], answer_type: str) -> Dict[str, Any]:
    rubrics = dict(config.get("rubrics") or {})
    key = canonical_answer_type(answer_type)
    rubric = rubrics.get(key)
    if rubric is None:
        raise KeyError(f"No rubric configured for answer_type={key}")
    return rubric


def build_geval_messages(
    example: EvalExample,
    rubric_config: Dict[str, Any],
) -> List[Dict[str, str]]:
    rubric = get_rubric(rubric_config, example.answer_type)
    scale = dict(rubric_config.get("scale") or {"min": 1, "max": 5})
    dimensions = list(rubric.get("dimensions") or [])
    dim_lines = []
    for item in dimensions:
        dim_lines.append(
            f"- {item['key']}: {item['description']} "
            f"(score {scale['min']} to {scale['max']})"
        )
    dim_text = "\n".join(dim_lines)
    metadata = {
        "example_id": example.example_id,
        "sample_id": example.sample_id,
        "source_format": example.source_format,
        "question_type": example.question_type,
        "answer_type": example.answer_type,
        "difficulty": example.difficulty,
        "frame": example.frame,
        "metadata": example.metadata,
    }
    system = (
        "You are a careful LLM-as-a-judge for multivariate time-series QA. "
        "Judge only the student answer quality relative to the question and the teacher/reference answer. "
        "Return exactly one valid JSON object and no markdown."
    )
    user = (
        "Score the student answer with a G-Eval style rubric.\n\n"
        "Evaluation metadata:\n"
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        "Question:\n"
        f"{example.question}\n\n"
        "Teacher/reference answer:\n"
        f"{example.teacher_answer}\n\n"
        "Student answer:\n"
        f"{example.student_answer}\n\n"
        "Rubric dimensions:\n"
        f"{dim_text}\n\n"
        "Scoring requirements:\n"
        f"- Use integer scores from {scale['min']} to {scale['max']}.\n"
        "- Judge factual and inferential quality, not style similarity alone.\n"
        "- If the student answer is incomplete, lower completeness or relevance rather than inventing missing content.\n"
        "- If the answer format is choice/judgment, correctness should dominate the final overall score.\n"
        "- Treat hidden model hints as unavailable judge-side artifacts; judge only what appears in the visible question/teacher/student texts.\n\n"
        "Return JSON with this schema:\n"
        "{"
        "\"example_id\": string, "
        "\"answer_type\": string, "
        "\"question_type\": string, "
        "\"scores\": {\"<dimension_key>\": integer, ...}, "
        "\"overall_score\": integer, "
        "\"dimension_comments\": {\"<dimension_key>\": string, ...}, "
        "\"summary\": string"
        "}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _extract_json_candidates(text: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    fenced = _JSON_FENCE_RE.findall(text or "")
    snippets = list(fenced)
    raw = text or ""
    stack: List[int] = []
    for idx, char in enumerate(raw):
        if char == "{":
            stack.append(idx)
        elif char == "}" and stack:
            start = stack.pop()
            snippets.append(raw[start : idx + 1])
    for snippet in snippets:
        try:
            parsed = json.loads(snippet)
        except Exception:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
    return candidates


def parse_geval_response(text: str) -> Dict[str, Any]:
    for candidate in _extract_json_candidates(text):
        if isinstance(candidate.get("scores"), dict):
            return candidate
    raise ValueError("No valid G-Eval JSON object found in judge response.")


def _normalize_score(value: Any, minimum: int, maximum: int) -> Optional[float]:
    try:
        raw = float(value)
    except Exception:
        return None
    if maximum <= minimum:
        return raw
    return (raw - minimum) / float(maximum - minimum)


def score_summary_payload(score_map: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    return {
        key: summarize_scores([float(v) for v in values if v is not None])
        for key, values in score_map.items()
    }


def aggregate_geval_records(
    rows: Sequence[Dict[str, Any]],
    *,
    group_fields: Sequence[str] = ("answer_type", "difficulty", "frame", "question_type"),
) -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[float]]] = {"overall": {}}
    all_fields = ["overall_score_norm"]
    for row in rows:
        for key in row.get("scores_norm", {}).keys():
            if key not in all_fields:
                all_fields.append(key)
    for key in all_fields:
        grouped["overall"].setdefault(key, [])

    buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = {field: {} for field in group_fields}
    for row in rows:
        grouped["overall"]["overall_score_norm"].append(float(row.get("overall_score_norm", 0.0)))
        for key, value in (row.get("scores_norm") or {}).items():
            grouped["overall"].setdefault(key, []).append(float(value))
        for field in group_fields:
            bucket_name = str(row.get(field) or "unknown")
            bucket = buckets[field].setdefault(bucket_name, {"overall_score_norm": []})
            bucket["overall_score_norm"].append(float(row.get("overall_score_norm", 0.0)))
            for key, value in (row.get("scores_norm") or {}).items():
                bucket.setdefault(key, []).append(float(value))

    return {
        "overall": score_summary_payload(grouped["overall"]),
        "by_group": {
            field: {
                bucket_name: score_summary_payload(metric_map)
                for bucket_name, metric_map in bucket_map.items()
            }
            for field, bucket_map in buckets.items()
        },
    }


def lexical_baseline(examples: Iterable[EvalExample]) -> Dict[str, Any]:
    rows = list(examples)
    overall_scores: List[float] = []
    by_answer_type: Dict[str, List[float]] = {}
    for item in rows:
        score = lexical_cosine(item.teacher_answer, item.student_answer)
        overall_scores.append(float(score))
        by_answer_type.setdefault(item.answer_type, []).append(float(score))
    return {
        "overall": summarize_scores(overall_scores),
        "by_answer_type": {
            key: summarize_scores(values) for key, values in sorted(by_answer_type.items())
        },
    }


def make_judged_record(
    *,
    example: EvalExample,
    parsed: Dict[str, Any],
    raw_response: str,
    latency_seconds: float,
    scale_min: int,
    scale_max: int,
) -> Dict[str, Any]:
    scores = dict(parsed.get("scores") or {})
    scores_norm = {
        key: _normalize_score(value, scale_min, scale_max)
        for key, value in scores.items()
    }
    scores_norm = {key: value for key, value in scores_norm.items() if value is not None}
    overall_score = parsed.get("overall_score")
    overall_norm = _normalize_score(overall_score, scale_min, scale_max)
    record = asdict(example)
    record.update(
        {
            "judge_raw_response": raw_response,
            "judge_latency_seconds": latency_seconds,
            "judge_summary": parsed.get("summary"),
            "dimension_comments": parsed.get("dimension_comments") or {},
            "scores": scores,
            "scores_norm": scores_norm,
            "overall_score": overall_score,
            "overall_score_norm": overall_norm,
        }
    )
    return record


def format_summary_table(summary: Dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    lines = ["G-Eval summary", ""]
    for metric, values in overall.items():
        if not isinstance(values, dict):
            continue
        lines.append(
            f"- {metric}: mean={values.get('mean')} median={values.get('median')} "
            f"p25={values.get('p25')} p75={values.get('p75')} n={values.get('count')}"
        )
    return "\n".join(lines)

