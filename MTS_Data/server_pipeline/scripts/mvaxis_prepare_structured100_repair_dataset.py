from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.utils import read_jsonl, write_jsonl, save_json


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _teacher_answer(row: Dict[str, Any]) -> str:
    for key in ("repaired_answer", "model_answer", "answer", "windows_0_answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _question_text(row: Dict[str, Any]) -> str:
    return str(row.get("question") or "").strip()


def _sample_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = {}
    for key in (
        "sample_id",
        "question_pair_id",
        "source_window_sample_id",
        "base_sample_id",
    ):
        value = row.get(key)
        if value is not None:
            text = str(value).strip()
            if text:
                keys[key] = text
    return keys


def _build_teacher_row(
    question_row: Dict[str, Any],
    *,
    teacher_row: Dict[str, Any],
    repaired_answer: str | None,
    source: str,
) -> Dict[str, Any]:
    answer_text = (repaired_answer or _teacher_answer(teacher_row)).strip()
    payload: Dict[str, Any] = {
        "schema_version": question_row.get("schema_version"),
        "sample_id": question_row.get("sample_id"),
        "base_sample_id": question_row.get("base_sample_id"),
        "source_sample_id": question_row.get("source_sample_id"),
        "question_pair_id": question_row.get("question_pair_id"),
        "source_window_sample_id": question_row.get("source_window_sample_id"),
        "question_id": question_row.get("question_id"),
        "question_type": question_row.get("question_type"),
        "question_answer_type": question_row.get("question_answer_type"),
        "question_difficulty": question_row.get("question_difficulty"),
        "question_family": question_row.get("question_family"),
        "question": question_row.get("question"),
        "target_interval": question_row.get("target_interval"),
        "target_output": question_row.get("target_output"),
        "evidence_card": question_row.get("evidence_card") or {},
        "teacher_answer_llm": {
            "answer": answer_text,
            "source": source,
            "original_model_answer": teacher_row.get("model_answer"),
            "reference_answer": teacher_row.get("windows_0_answer"),
        },
        "answer": answer_text,
        "model_answer": answer_text,
        "windows_0_answer": teacher_row.get("windows_0_answer"),
        "audit_repaired": repaired_answer is not None,
        "question_generation_artifacts": question_row.get("question_generation_artifacts"),
    }
    payload.update(_sample_keys(question_row))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="outputs/question_final_530/questions_600.jsonl")
    parser.add_argument(
        "--teacher",
        default="outputs/teacheranswer_question_final_530_structured100/teacher_gpt55.answers.jsonl",
    )
    parser.add_argument(
        "--repair-results",
        default="mvaxis_answer_audit/results/teacheranswer_question_final_530_structured100_repair/repair_results.jsonl",
    )
    parser.add_argument(
        "--output-teacher",
        default="outputs/teacheranswer_question_final_530_structured100_repair/teacher_gpt55.repaired_aligned.jsonl",
    )
    parser.add_argument(
        "--output-questions",
        default="outputs/question_final_530/questions_structured100_aligned.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="outputs/teacheranswer_question_final_530_structured100_repair/summary_aligned.json",
    )
    args = parser.parse_args()

    question_rows = read_jsonl(_resolve(args.questions))
    teacher_rows = read_jsonl(_resolve(args.teacher))
    repair_rows = read_jsonl(_resolve(args.repair_results))

    question_by_text = {}
    for idx, row in enumerate(question_rows):
        question_by_text[_question_text(row)] = (idx, row)

    repair_by_question = {}
    for row in repair_rows:
        question_text = _question_text(row)
        repaired_answer = row.get("repaired_answer")
        if question_text and isinstance(repaired_answer, str) and repaired_answer.strip():
            repair_by_question[question_text] = row

    aligned_questions: List[Dict[str, Any]] = []
    aligned_teachers: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for idx, teacher_row in enumerate(teacher_rows):
        question_text = _question_text(teacher_row)
        mapped = question_by_text.get(question_text)
        if mapped is None:
            missing.append({"teacher_index": idx, "question": question_text})
            continue
        question_index, question_row = mapped
        repair_row = repair_by_question.get(question_text)
        aligned_questions.append(question_row)
        if "evidence_card" not in aligned_questions[-1]:
            aligned_questions[-1] = dict(aligned_questions[-1])
            aligned_questions[-1]["evidence_card"] = aligned_questions[-1].get("evidence_card") or {}
        aligned_teachers.append(
            _build_teacher_row(
                question_row,
                teacher_row=teacher_row,
                repaired_answer=(repair_row or {}).get("repaired_answer"),
                source="structured100_repair" if repair_row else "structured100_original",
            )
        )

    output_teacher = _resolve(args.output_teacher)
    output_questions = _resolve(args.output_questions)
    output_summary = _resolve(args.summary)
    output_teacher.parent.mkdir(parents=True, exist_ok=True)
    output_questions.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(aligned_teachers, output_teacher)
    write_jsonl(aligned_questions, output_questions)
    save_json(
        {
            "questions_path": str(_resolve(args.questions)),
            "teacher_path": str(_resolve(args.teacher)),
            "repair_results_path": str(_resolve(args.repair_results)),
            "output_teacher": str(output_teacher),
            "output_questions": str(output_questions),
            "teacher_rows": len(teacher_rows),
            "question_rows": len(question_rows),
            "aligned_rows": len(aligned_teachers),
            "repair_applied_rows": sum(1 for row in aligned_teachers if row.get("audit_repaired")),
            "missing_rows": missing,
        },
        output_summary,
    )
    print(
        json.dumps(
            {
                "aligned_rows": len(aligned_teachers),
                "repair_applied_rows": sum(1 for row in aligned_teachers if row.get("audit_repaired")),
                "output_teacher": str(output_teacher),
                "output_questions": str(output_questions),
                "summary": str(output_summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
