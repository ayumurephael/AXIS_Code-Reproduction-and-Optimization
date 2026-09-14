from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.answer_schema import normalize_answer_label, target_answer_label
from src.mvaxis.geval import (
    EvalExample,
    aggregate_geval_records,
    build_geval_messages,
    canonical_answer_type,
    format_summary_table,
    infer_difficulty,
    infer_frame,
    lexical_baseline,
    load_rubric_config,
    make_judged_record,
    parse_geval_response,
)
from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.semantic_alignment import (
    compact_text,
    extract_student_answer,
    extract_teacher_answer,
    lexical_cosine,
    summarize_scores,
)
from src.mvaxis.student_answer_provider import parse_student_answer
from src.mvaxis.utils import load_json, read_jsonl, save_json, write_jsonl


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _sample_id(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("sample_id")
    if value is None:
        return None
    text = str(value)
    if ":" in text and text.rsplit(":", 1)[-1].isdigit():
        return None
    return text


def _build_teacher_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = _sample_id(row)
        if sid:
            lookup[sid] = row
    return lookup


def _row_by_index(rows: List[Dict[str, Any]], index: Any) -> Optional[Dict[str, Any]]:
    try:
        idx = int(index)
    except Exception:
        return None
    if 0 <= idx < len(rows):
        return rows[idx]
    return None


def _find_teacher(
    student_row: Dict[str, Any],
    teacher_rows: List[Dict[str, Any]],
    teacher_by_sample: Dict[str, Dict[str, Any]],
    question_rows: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    idx = student_row.get("index")
    if question_rows:
        question_row = _row_by_index(question_rows, idx)
        if question_row:
            sid = _sample_id(question_row)
            if sid and sid in teacher_by_sample:
                return teacher_by_sample[sid]
    sid = _sample_id(student_row)
    if sid and sid in teacher_by_sample:
        return teacher_by_sample[sid]
    return _row_by_index(teacher_rows, idx)


def _question_row(
    student_row: Dict[str, Any],
    question_rows: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if question_rows is None:
        return None
    return _row_by_index(question_rows, student_row.get("index"))


def _question_text(student_row: Dict[str, Any], question_row: Optional[Dict[str, Any]]) -> str:
    value = student_row.get("question")
    if value is None and question_row is not None:
        value = question_row.get("question")
    return compact_text(value or "")


def _extract_answer_type(
    student_row: Dict[str, Any],
    teacher_row: Optional[Dict[str, Any]],
    question_row: Optional[Dict[str, Any]],
    question: str,
) -> str:
    explicit = (
        (student_row.get("parsed_student_answer") or {}).get("question_answer_type")
        or student_row.get("question_answer_type")
    )
    if question_row is not None:
        fact = (question_row.get("target_output") or {}).get("fact_check") or {}
        explicit = explicit or question_row.get("question_answer_type") or fact.get("question_answer_type")
    return canonical_answer_type(explicit, question=question, teacher_answer=extract_teacher_answer(teacher_row or {}))


def _extract_question_type(student_row: Dict[str, Any], question_row: Optional[Dict[str, Any]]) -> str:
    return str(
        student_row.get("question_type")
        or student_row.get("question_id")
        or (question_row or {}).get("question_type")
        or (question_row or {}).get("question_id")
        or "unknown"
    )


def _normalized_label(value: Any) -> Optional[str]:
    normalized = normalize_answer_label(value)
    if normalized is None:
        return None
    text = str(normalized).strip().lower()
    return text or None


def _hard_metrics_from_examples(
    examples: Sequence[EvalExample],
    *,
    student_rows: Optional[Sequence[Dict[str, Any]]] = None,
    question_rows: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    comparable = 0
    correct = 0
    parsed = 0
    choice_judgment_total = 0
    open_lexical: List[float] = []
    by_answer_type: Dict[str, Dict[str, Any]] = {}

    if student_rows is None:
        student_rows = [{} for _ in examples]
    if question_rows is None:
        question_rows = [None for _ in examples]

    for example, student_row, question_row in zip(examples, student_rows, question_rows):
        bucket = by_answer_type.setdefault(
            example.answer_type,
            {"n": 0, "parsed": 0, "correct": 0, "lexical_scores": []},
        )
        bucket["n"] += 1
        if example.answer_type == "open":
            score = compact_text(example.student_answer)
            if score:
                lexical = float(lexical_cosine(example.teacher_answer, example.student_answer))
                open_lexical.append(lexical)
                bucket["lexical_scores"].append(lexical)
            continue

        choice_judgment_total += 1
        parsed_answer = student_row.get("parsed_student_answer") or {}
        pred = _normalized_label(parsed_answer.get("answer_label"))
        if pred is None:
            raw_student_text = str(
                student_row.get("raw_response")
                or student_row.get("generated_response")
                or student_row.get("answer")
                or ""
            )
            reparsed = parse_student_answer(
                raw_student_text,
                question_type=example.question_type,
                question=example.question,
            )
            pred = _normalized_label(reparsed.get("answer_label"))
        if pred is not None:
            parsed += 1
            bucket["parsed"] += 1

        gold = None
        if question_row is not None:
            gold = _normalized_label(target_answer_label((question_row.get("target_output") or {})))
        if gold is None:
            teacher_parsed = parse_student_answer(
                example.teacher_answer,
                question_type=example.question_type,
                question=example.question,
            )
            gold = _normalized_label(teacher_parsed.get("answer_label"))
        if gold is not None:
            comparable += 1
            if pred == gold:
                correct += 1
                bucket["correct"] += 1

    summary: Dict[str, Any] = {
        "choice_judgment_parse_rate": parsed / max(1, choice_judgment_total),
        "choice_judgment_accuracy": correct / max(1, comparable),
        "choice_judgment_n": choice_judgment_total,
        "choice_judgment_label_n": comparable,
        "open_lexical_similarity": {
            "overall": summarize_scores(open_lexical),
            "by_answer_type": {},
        },
        "by_answer_type": {},
    }
    for key, bucket in sorted(by_answer_type.items()):
        summary["by_answer_type"][key] = {
            "n": bucket["n"],
            "parsed": bucket["parsed"],
            "correct": bucket["correct"],
            "parse_rate": bucket["parsed"] / max(1, bucket["n"]) if key != "open" else None,
            "accuracy": bucket["correct"] / max(1, bucket["n"]) if key != "open" else None,
            "lexical_similarity": summarize_scores(bucket["lexical_scores"]) if bucket["lexical_scores"] else None,
        }
    return summary


def _load_multi_examples(
    predictions_path: Path,
    *,
    teacher_path: Path,
    questions_path: Optional[Path],
    limit: Optional[int],
) -> Tuple[List[EvalExample], List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
    student_rows = read_jsonl(predictions_path)
    teacher_rows = read_jsonl(teacher_path)
    question_rows = read_jsonl(questions_path) if questions_path else None
    if limit is not None:
        student_rows = student_rows[: int(limit)]
    teacher_by_sample = _build_teacher_lookup(teacher_rows)
    examples: List[EvalExample] = []
    aligned_questions: List[Optional[Dict[str, Any]]] = []
    for idx, student in enumerate(student_rows):
        question_row = _question_row(student, question_rows)
        teacher = _find_teacher(student, teacher_rows, teacher_by_sample, question_rows)
        question = _question_text(student, question_row)
        teacher_answer = extract_teacher_answer(teacher or {})
        student_answer = extract_student_answer(student)
        question_type = _extract_question_type(student, question_row)
        answer_type = _extract_answer_type(student, teacher, question_row, question)
        difficulty = infer_difficulty(question_type, (question_row or {}).get("question_difficulty"))
        frame = infer_frame(question_type, (question_row or {}).get("question_frame"))
        examples.append(
            EvalExample(
                example_id=str(student.get("index", idx)),
                sample_id=str(student.get("sample_id")) if student.get("sample_id") is not None else None,
                source_format="multi_jsonl",
                question=question,
                student_answer=student_answer,
                teacher_answer=teacher_answer,
                question_type=question_type,
                answer_type=answer_type,
                difficulty=difficulty,
                frame=frame,
                metadata={
                    "prediction_path": str(predictions_path),
                    "teacher_path": str(teacher_path),
                    "student_index": student.get("index", idx),
                },
            )
        )
        aligned_questions.append(question_row)
    return examples, student_rows, aligned_questions


def _iter_legacy_yaml_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*.yml")):
        yield item
    for item in sorted(path.rglob("*.yaml")):
        yield item


def _load_legacy_examples(
    predictions_path: Path,
    *,
    limit: Optional[int],
) -> Tuple[List[EvalExample], List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
    rows: List[Dict[str, Any]] = []
    for path in _iter_legacy_yaml_files(predictions_path):
        rows.append(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    if limit is not None:
        rows = rows[: int(limit)]
    examples: List[EvalExample] = []
    for idx, row in enumerate(rows):
        question = compact_text(row.get("question") or "")
        teacher_answer = compact_text(row.get("expected_answer") or "")
        student_answer = compact_text(row.get("generated_response") or "")
        question_type = str(row.get("question_type") or "unknown")
        answer_type = canonical_answer_type(question_type, question=question, teacher_answer=teacher_answer)
        examples.append(
            EvalExample(
                example_id=str(row.get("global_question_id") or idx),
                sample_id=str(row.get("global_question_id") or idx),
                source_format="legacy_yaml",
                question=question,
                student_answer=student_answer,
                teacher_answer=teacher_answer,
                question_type=question_type,
                answer_type=answer_type,
                difficulty=infer_difficulty(question_type),
                frame=infer_frame(question_type),
                metadata={
                    "batch_id": row.get("batch_id"),
                    "question_in_batch": row.get("question_in_batch"),
                    "window_range": row.get("window_range"),
                },
            )
        )
    return examples, rows, [None for _ in rows]


def _format_metrics_tsv(judged_rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["example_id", "sample_id", "answer_type", "difficulty", "frame", "question_type", "overall_score", "overall_score_norm"]
    dynamic = sorted({key for row in judged_rows for key in (row.get("scores") or {}).keys()})
    lines = ["\t".join(headers + dynamic)]
    for row in judged_rows:
        cells = [
            str(row.get("example_id")),
            str(row.get("sample_id")),
            str(row.get("answer_type")),
            str(row.get("difficulty")),
            str(row.get("frame")),
            str(row.get("question_type")),
            str(row.get("overall_score")),
            str(row.get("overall_score_norm")),
        ]
        for key in dynamic:
            cells.append(str((row.get("scores") or {}).get(key)))
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def _select_examples(
    fmt: str,
    predictions_path: Path,
    *,
    teacher_path: Optional[Path],
    questions_path: Optional[Path],
    limit: Optional[int],
) -> Tuple[List[EvalExample], List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
    if fmt == "multi_jsonl":
        if teacher_path is None:
            raise ValueError("--teacher is required for format=multi_jsonl")
        return _load_multi_examples(
            predictions_path,
            teacher_path=teacher_path,
            questions_path=questions_path,
            limit=limit,
        )
    if fmt == "legacy_yaml":
        return _load_legacy_examples(predictions_path, limit=limit)
    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score legacy AXIS or MVAXIS answers with G-Eval.")
    parser.add_argument("--format", choices=["multi_jsonl", "legacy_yaml"], required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--questions", default=None)
    parser.add_argument("--llm-config", default=None, help="Required unless --skip-judge is set.")
    parser.add_argument("--rubric-config", default="configs/mvaxis_geval_rubrics.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--prompt-examples", type=int, default=3)
    args = parser.parse_args()

    predictions_path = _resolve(args.predictions)
    teacher_path = _resolve(args.teacher) if args.teacher else None
    questions_path = _resolve(args.questions) if args.questions else None
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rubric_config = load_rubric_config(str(_resolve(args.rubric_config)))
    examples, source_rows, aligned_questions = _select_examples(
        args.format,
        predictions_path,
        teacher_path=teacher_path,
        questions_path=questions_path,
        limit=args.limit,
    )

    hard_metrics = _hard_metrics_from_examples(
        examples,
        student_rows=source_rows,
        question_rows=aligned_questions,
    )
    lexical_metrics = lexical_baseline(examples)
    prompt_examples: List[Dict[str, Any]] = []
    judged_rows: List[Dict[str, Any]] = []
    judge_summary: Dict[str, Any] = {}

    if not args.skip_judge:
        if not args.llm_config:
            raise ValueError("--llm-config is required unless --skip-judge is set.")
        llm_config = load_json(_resolve(args.llm_config))
        client = create_llm_client(llm_config)
        scale = dict(rubric_config.get("scale") or {"min": 1, "max": 5})
        for idx, example in enumerate(examples):
            messages = build_geval_messages(example, rubric_config)
            if len(prompt_examples) < int(args.prompt_examples):
                prompt_examples.append(
                    {
                        "example_id": example.example_id,
                        "answer_type": example.answer_type,
                        "messages": messages,
                    }
                )
            response = client.complete(messages)
            parsed = parse_geval_response(response.content)
            judged_rows.append(
                make_judged_record(
                    example=example,
                    parsed=parsed,
                    raw_response=response.content,
                    latency_seconds=response.latency_seconds,
                    scale_min=int(scale["min"]),
                    scale_max=int(scale["max"]),
                )
            )
            print(f"[geval] {idx + 1}/{len(examples)} example_id={example.example_id}", flush=True)
        judge_summary = aggregate_geval_records(judged_rows)
        write_jsonl(judged_rows, output_dir / "geval_per_example.jsonl")
        (output_dir / "geval_metrics.tsv").write_text(_format_metrics_tsv(judged_rows), encoding="utf-8")
        (output_dir / "geval_summary.txt").write_text(format_summary_table(judge_summary), encoding="utf-8")

    summary = {
        "format": args.format,
        "predictions": str(predictions_path),
        "teacher": str(teacher_path) if teacher_path else None,
        "questions": str(questions_path) if questions_path else None,
        "num_examples": len(examples),
        "hard_metrics": hard_metrics,
        "lexical_baseline": lexical_metrics,
        "judge_summary": judge_summary,
        "prompt_examples": prompt_examples,
        "rubric_config": str(_resolve(args.rubric_config)),
    }
    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
