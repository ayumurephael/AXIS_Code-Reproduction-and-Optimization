from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.semantic_alignment import (
    compact_text,
    extract_student_answer,
    extract_teacher_answer,
    lexical_cosine,
    summarize_scores,
)
from src.mvaxis.utils import read_jsonl, save_json, write_jsonl


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
    question_rows: Optional[List[Dict[str, Any]]] = None,
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


class HFTextSemanticScorer:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        max_length: int = 768,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.torch = torch
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        dtype = None
        if torch_dtype.lower() in {"float16", "fp16"}:
            dtype = torch.float16
        elif torch_dtype.lower() in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif torch_dtype.lower() in {"float32", "fp32"}:
            dtype = torch.float32
        elif torch_dtype.lower() == "auto":
            dtype = torch.float16 if self.device.type == "cuda" else None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(self.device)
        self.model.eval()
        self.max_length = int(max_length)

    def embed(self, text: str) -> torch.Tensor:
        with self.torch.no_grad():
            encoded = self.tokenizer(
                compact_text(text),
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            outputs = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = outputs.hidden_states[-1][0].float()
        mask = encoded["attention_mask"][0].bool()
        if mask.any():
            hidden = hidden[mask]
        return hidden.mean(dim=0).detach().cpu()

    def similarity(self, a: str, b: str) -> float:
        emb_a = self.embed(a)
        emb_b = self.embed(b)
        return float(self.torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item())


def _write_text_report(summary: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "Teacher/student semantic alignment report",
        f"student_path: {summary.get('student_path')}",
        f"teacher_path: {summary.get('teacher_path')}",
        f"model: {summary.get('model') or 'lexical_fallback'}",
        "",
        "Scores:",
    ]
    for key, value in (summary.get("score_summary") or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "Lowest examples:"])
    for item in summary.get("lowest_examples", []):
        lines.append(
            f"- index={item.get('index')} score={item.get('semantic_similarity'):.4f} "
            f"question={compact_text(item.get('question'), 180)}"
        )
        lines.append(f"  teacher={compact_text(item.get('teacher_answer'), 220)}")
        lines.append(f"  student={compact_text(item.get('student_answer'), 220)}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _evaluate_rows(
    *,
    student_rows: List[Dict[str, Any]],
    student_path: Path,
    teacher_rows: List[Dict[str, Any]],
    teacher_by_sample: Dict[str, Dict[str, Any]],
    question_rows: Optional[List[Dict[str, Any]]],
    teacher_path: Path,
    questions_path: Optional[Path],
    output_dir: Path,
    scorer: Optional[HFTextSemanticScorer],
    model_name: Optional[str],
    progress_step: int,
    progress_prefix: str = "semantic-align",
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    total = len(student_rows)
    for idx, student in enumerate(student_rows):
        teacher = _find_teacher(student, teacher_rows, teacher_by_sample, question_rows)
        teacher_answer = extract_teacher_answer(teacher or {})
        student_answer = extract_student_answer(student)
        if scorer is None:
            score = lexical_cosine(teacher_answer, student_answer)
            score_kind = "lexical_cosine"
        else:
            score = scorer.similarity(teacher_answer, student_answer)
            score_kind = "hf_hidden_mean_cosine"
        question = student.get("question")
        if question is None and question_rows:
            qrow = _row_by_index(question_rows, student.get("index"))
            question = None if qrow is None else qrow.get("question")
        records.append(
            {
                "index": student.get("index", idx),
                "question_type": student.get("question_type"),
                "question": question,
                "semantic_similarity": float(score),
                "score_kind": score_kind,
                "teacher_answer": teacher_answer,
                "student_answer": student_answer,
                "student_label": ((student.get("parsed_student_answer") or student.get("parsed_response") or {}).get("answer_label")),
                "teacher_sample_id": None if teacher is None else teacher.get("sample_id"),
                "student_sample_id": student.get("sample_id"),
            }
        )
        if progress_step > 0 and ((idx + 1) % int(progress_step) == 0 or idx + 1 == total):
            print(f"[{progress_prefix}] {idx + 1}/{total}", flush=True)

    scores = [float(row["semantic_similarity"]) for row in records]
    lowest = sorted(records, key=lambda x: float(x["semantic_similarity"]))[:10]
    summary = {
        "student_path": str(student_path),
        "teacher_path": str(teacher_path),
        "questions_path": str(questions_path) if questions_path else None,
        "model": model_name,
        "score_kind": "lexical_cosine" if scorer is None else "hf_hidden_mean_cosine",
        "num_pairs": len(records),
        "score_summary": summarize_scores(scores),
        "lowest_examples": lowest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, output_dir / "semantic_alignment.jsonl")
    save_json(summary, output_dir / "semantic_alignment_summary.json")
    _write_text_report(summary, output_dir / "semantic_alignment_summary.txt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default=None, help="Path to qwen_raw_answers.jsonl")
    parser.add_argument("--student-root", default=None, help="Root directory containing group/qwen_raw_answers.jsonl outputs")
    parser.add_argument(
        "--groups",
        default="",
        help="Comma-separated group names under --student-root. If omitted, all subdirs with qwen_raw_answers.jsonl are used.",
    )
    parser.add_argument("--teacher", required=True, help="Path to teacher jsonl")
    parser.add_argument("--questions", default=None, help="Optional question jsonl used for sample_id alignment")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None, help="HF causal LM path/name for semantic hidden-state scoring")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--progress-step", type=int, default=25)
    args = parser.parse_args()

    teacher_path = _resolve(args.teacher)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = _resolve(args.questions) if args.questions else None
    question_rows = read_jsonl(questions_path) if questions_path else None
    teacher_rows = read_jsonl(teacher_path)
    teacher_by_sample = _build_teacher_lookup(teacher_rows)

    scorer = None
    if args.model:
        scorer = HFTextSemanticScorer(
            args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_length=int(args.max_length),
        )

    if args.student_root:
        root = _resolve(args.student_root)
        if args.groups.strip():
            groups = [item.strip() for item in args.groups.split(",") if item.strip()]
        else:
            groups = sorted(
                item.name
                for item in root.iterdir()
                if item.is_dir() and (item / "qwen_raw_answers.jsonl").exists()
            )
        group_summaries: List[Dict[str, Any]] = []
        for group in groups:
            student_path = root / group / "qwen_raw_answers.jsonl"
            if not student_path.exists():
                print(f"[semantic-align:{group}] skipped missing {student_path}", flush=True)
                continue
            student_rows = read_jsonl(student_path)
            if args.limit is not None:
                student_rows = student_rows[: int(args.limit)]
            summary = _evaluate_rows(
                student_rows=student_rows,
                student_path=student_path,
                teacher_rows=teacher_rows,
                teacher_by_sample=teacher_by_sample,
                question_rows=question_rows,
                teacher_path=teacher_path,
                questions_path=questions_path,
                output_dir=output_dir / group / "semantic_alignment_qwen7b",
                scorer=scorer,
                model_name=args.model,
                progress_step=int(args.progress_step),
                progress_prefix=f"semantic-align:{group}",
            )
            summary["group"] = group
            group_summaries.append(summary)
        table_lines = [
            "\t".join(["group", "num_pairs", "mean", "median", "p10", "p25", "p75", "p90", "min", "max"])
        ]
        for summary in group_summaries:
            scores = summary.get("score_summary") or {}
            table_lines.append(
                "\t".join(
                    [
                        str(summary.get("group")),
                        str(summary.get("num_pairs")),
                        str(scores.get("mean")),
                        str(scores.get("median")),
                        str(scores.get("p10")),
                        str(scores.get("p25")),
                        str(scores.get("p75")),
                        str(scores.get("p90")),
                        str(scores.get("min")),
                        str(scores.get("max")),
                    ]
                )
            )
        (output_dir / "semantic_alignment_metrics.tsv").write_text("\n".join(table_lines), encoding="utf-8")
        save_json({"groups": group_summaries}, output_dir / "semantic_alignment_metrics.json")
        print(json.dumps({"groups": group_summaries}, ensure_ascii=False, indent=2))
        return

    if not args.student:
        raise ValueError("Either --student or --student-root must be provided")
    student_path = _resolve(args.student)
    student_rows = read_jsonl(student_path)
    if args.limit is not None:
        student_rows = student_rows[: int(args.limit)]
    summary = _evaluate_rows(
        student_rows=student_rows,
        student_path=student_path,
        teacher_rows=teacher_rows,
        teacher_by_sample=teacher_by_sample,
        question_rows=question_rows,
        teacher_path=teacher_path,
        questions_path=questions_path,
        output_dir=output_dir,
        scorer=scorer,
        model_name=args.model,
        progress_step=int(args.progress_step),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
