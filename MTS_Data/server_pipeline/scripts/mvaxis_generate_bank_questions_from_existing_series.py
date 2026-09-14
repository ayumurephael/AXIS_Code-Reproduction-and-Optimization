from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.html_reports import write_question_generation_html
from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.question_provider import assign_questions
from src.mvaxis.utils import read_jsonl, save_json, write_jsonl


def _dated_output_dir(prefix: str, count: int) -> str:
    """Return a standard dated output directory name."""

    from datetime import datetime

    return f"outputs/{prefix}_{int(count)}_{datetime.now().strftime('%m%d')}"


def _summary(
    *,
    source_path: Path,
    output_dir: Path,
    questions_path: Path,
    txt_path: Path,
    summary_path: Path,
    html_path: Path,
    rows: List[Dict[str, Any]],
    requested_questions: int,
    seed: int,
    started: float,
) -> Dict[str, Any]:
    label_counts = {"anomalous": 0, "normal": 0}
    answer_type_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        fact = (row.get("target_output") or {}).get("fact_check") or {}
        has_anomaly = bool(fact.get("is_anomalous"))
        label_counts["anomalous" if has_anomaly else "normal"] += 1
        answer_type = str(row.get("question_answer_type") or "unknown")
        difficulty = str(row.get("question_difficulty") or "unknown")
        answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        if len(examples) < 10:
            examples.append(
                {
                    "idx": idx,
                    "sample_id": row.get("sample_id"),
                    "interval": row.get("target_interval"),
                    "question_type": row.get("question_type"),
                    "question_difficulty": row.get("question_difficulty"),
                    "question_answer_type": row.get("question_answer_type"),
                    "question": row.get("question"),
                    "has_anomaly": has_anomaly,
                }
            )
    return {
        "source_jsonl": str(source_path),
        "source_rows": len(rows),
        "generated_questions": len(rows),
        "requested_questions": int(requested_questions),
        "output_dir": str(output_dir),
        "output_jsonl": str(questions_path),
        "output_txt": str(txt_path),
        "output_html": str(html_path),
        "summary": str(summary_path),
        "question_provider": "bank(question_provider.py, bank=regular)",
        "seed": int(seed),
        "label_counts": label_counts,
        "answer_type_counts": answer_type_counts,
        "difficulty_counts": difficulty_counts,
        "elapsed_seconds": time.perf_counter() - started,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign regular-bank question templates through question_provider.py to an existing sample/window JSONL."
    )
    parser.add_argument(
        "--source-series",
        default="outputs/question_hard_1000_0601/questions_1000.jsonl",
        help="Existing sample/window JSONL whose sample/window order should be preserved.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-questions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=602)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = _dated_output_dir("question_bank_1000", int(args.num_questions))

    source_path = Path(args.source_series)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_path = output_dir / f"questions_{int(args.num_questions)}.jsonl"
    txt_path = output_dir / f"questions_{int(args.num_questions)}.txt"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / f"questions_{int(args.num_questions)}.html"

    started = time.perf_counter()
    source_rows = read_jsonl(source_path)
    selected_rows = source_rows[: int(args.num_questions)]
    if len(selected_rows) < int(args.num_questions):
        raise ValueError(
            f"Requested {int(args.num_questions)} questions, but only {len(selected_rows)} rows are available in {source_path}"
        )

    progress = ProgressPrinter(len(selected_rows), label="question-bank", step_percent=5.0)
    progress.start(extra=f"source_rows={len(source_rows)} selected={len(selected_rows)}")
    rows = assign_questions(
        selected_rows,
        seed=int(args.seed),
        copy_samples=True,
        preserve_source_question=True,
        annotate_sample_id=False,
        provider="bank",
        question_bank="regular",
    )
    progress.update(len(rows), extra="assigned all bank questions")

    write_jsonl(rows, questions_path)
    with txt_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            answer = ((row.get("target_output") or {}).get("final_answer") or "").strip()
            f.write(f"[{idx:04d}] {row.get('sample_id')}\n")
            f.write(f"Question: {row.get('question', '')}\n")
            f.write(f"Answer: {answer}\n\n")

    write_question_generation_html(
        questions_path,
        html_path,
        title=f"Regular Bank Questions: {questions_path.name}",
    )
    summary = _summary(
        source_path=source_path,
        output_dir=output_dir,
        questions_path=questions_path,
        txt_path=txt_path,
        summary_path=summary_path,
        html_path=html_path,
        rows=rows,
        requested_questions=int(args.num_questions),
        seed=int(args.seed),
        started=started,
    )
    save_json(summary, summary_path)
    progress.finish(extra=f"output={questions_path.name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
