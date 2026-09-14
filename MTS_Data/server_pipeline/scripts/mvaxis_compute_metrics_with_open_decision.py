from __future__ import annotations

"""
Default student-answer eval metrics.

Use this script to maintain the core accuracy report for teacher-aligned runs:

- overall_with_open_acc
- explicit_answer_rate
- choice_acc
- judgment_acc
- open_decision_match

Companion anomaly-sensitive metrics such as judgment_true_f1,
open_decision_anomalous_f1, and interval_max_auroc are computed by
`mvaxis_compute_f1_score_auc.py` and should be reported alongside this file in
regular eval summaries.

Operational note:

- keep `explicit_answer_rate` from the teacher-aligned summary in the default
  eval bundle because formatting drift is a first-class failure mode
- for hint-alignment smoke checks, compare real hints against shuffled hints
  under the same `ct` / `scale` settings rather than changing the loss
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from _bootstrap import add_project_root

ROOT = add_project_root()

from scripts.mvaxis_eval_teacher_aligned_student_runs import parse_bool, parse_choice, _teacher_answer


GROUPS = [
    "all_hints",
    "no_global_hints",
    "no_channel_hints",
    "score_image_note_all_hints",
    "score_text_all_hints",
    "raw_image",
    "score_text_image_no_global_hints",
    "score_text_image_no_channel_hints",
    "score_text_nohints",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_answer_type(row: Dict[str, Any]) -> str:
    fact = (row.get("target_output") or {}).get("fact_check") or {}
    return str(
        row.get("question_answer_type")
        or fact.get("question_answer_type")
        or row.get("question_type")
        or ""
    ).lower()


def _parse_open_decision(text: str) -> Optional[bool]:
    text = text or ""
    decision_block = re.search(
        r"(?is)\bdecision\s*:\s*(.+?)(?:\n\s*\n|\n\s*main evidence\s*:|\n\s*interpretation\s*:|$)",
        text,
    )
    search_space = decision_block.group(1) if decision_block else text
    if re.search(r"(?i)\bthis interval is anomalous\b", search_space):
        return True
    if re.search(r"(?i)\bthis interval is normal\b", search_space):
        return False
    if re.search(r"(?i)\banomal(?:y|ous)\b", search_space) and not re.search(r"(?i)\bnormal\b", search_space):
        return True
    if re.search(r"(?i)\bnormal\b", search_space) and not re.search(r"(?i)\banomal(?:y|ous)\b", search_space):
        return False
    return None


def _teacher_open_decision(question_row: Dict[str, Any], teacher_row: Dict[str, Any]) -> Optional[bool]:
    teacher_answer = _teacher_answer(teacher_row)
    parsed = _parse_open_decision(teacher_answer)
    if parsed is not None:
        return parsed
    fact = (question_row.get("target_output") or {}).get("fact_check") or {}
    is_anomalous = fact.get("is_anomalous")
    return bool(is_anomalous) if is_anomalous is not None else None


def _target_choice_label(question_row: Dict[str, Any]) -> Optional[str]:
    fact = (question_row.get("target_output") or {}).get("fact_check") or {}
    val = fact.get("choice_answer")
    if val:
        return str(val).strip().upper()
    for candidate in [
        fact.get("answer_label"),
        (question_row.get("target_output") or {}).get("question_answer"),
        (question_row.get("target_output") or {}).get("final_answer"),
    ]:
        if candidate is None:
            continue
        parsed = parse_choice(str(candidate))
        if parsed:
            return parsed
    return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compute_for_run(
    run_dir: Path,
    questions: List[Dict[str, Any]],
    teachers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / "qwen_raw_answers.jsonl")
    choice_n = choice_parsed = choice_correct = 0
    judgment_n = judgment_parsed = judgment_correct = 0
    open_n = open_parsed = open_correct = 0

    for row in rows:
        idx = int(row["index"])
        question_row = questions[idx]
        teacher_row = teachers[idx]
        raw = str(row.get("raw_response") or "")
        answer_type = _question_answer_type(question_row)

        if "choice" in answer_type:
            choice_n += 1
            pred = parse_choice(raw)
            gold = parse_choice(_teacher_answer(teacher_row)) or _target_choice_label(question_row)
            if pred is not None:
                choice_parsed += 1
            if pred is not None and gold is not None and pred == gold:
                choice_correct += 1
        elif "judgment" in answer_type or "true_false" in answer_type or "true-false" in answer_type:
            judgment_n += 1
            pred = parse_bool(raw)
            gold = parse_bool(_teacher_answer(teacher_row))
            if gold is None:
                fact = (question_row.get("target_output") or {}).get("fact_check") or {}
                if fact.get("is_anomalous") is not None:
                    gold = bool(fact["is_anomalous"])
            if pred is not None:
                judgment_parsed += 1
            if pred is not None and gold is not None and pred == gold:
                judgment_correct += 1
        else:
            open_n += 1
            pred = _parse_open_decision(raw)
            gold = _teacher_open_decision(question_row, teacher_row)
            if pred is not None:
                open_parsed += 1
            if pred is not None and gold is not None and pred == gold:
                open_correct += 1

    qa_n = choice_n + judgment_n
    qa_correct = choice_correct + judgment_correct
    all_n = qa_n + open_n
    all_correct = qa_correct + open_correct
    all_parsed = choice_parsed + judgment_parsed + open_parsed
    return {
        "choice_n": choice_n,
        "choice_acc": (choice_correct / choice_n) if choice_n else None,
        "choice_parse_rate": (choice_parsed / choice_n) if choice_n else None,
        "judgment_n": judgment_n,
        "judgment_acc": (judgment_correct / judgment_n) if judgment_n else None,
        "judgment_parse_rate": (judgment_parsed / judgment_n) if judgment_n else None,
        "open_n": open_n,
        "open_decision_match": (open_correct / open_n) if open_n else None,
        "open_decision_parse_rate": (open_parsed / open_n) if open_n else None,
        "overall_qa_n": qa_n,
        "overall_qa_acc": (qa_correct / qa_n) if qa_n else None,
        "overall_with_open_n": all_n,
        "overall_with_open_acc": (all_correct / all_n) if all_n else None,
        "overall_with_open_parse_rate": (all_parsed / all_n) if all_n else None,
    }


def write_outputs(
    root: Path,
    results: Dict[str, Dict[str, Any]],
    run_order: Iterable[str] | None = None,
) -> None:
    by_type_headers = [
        "run",
        "choice_n",
        "choice_acc",
        "choice_parse_rate",
        "judgment_n",
        "judgment_acc",
        "judgment_parse_rate",
        "open_n",
        "open_decision_match",
        "open_decision_parse_rate",
    ]
    with_open_headers = [
        "run",
        "overall_qa_acc",
        "overall_qa_n",
        "overall_with_open_acc",
        "overall_with_open_n",
        "overall_with_open_parse_rate",
    ]

    by_type_lines = ["\t".join(by_type_headers)]
    with_open_lines = ["\t".join(with_open_headers)]

    ordered_runs = list(run_order) if run_order is not None else list(results.keys())
    for run in ordered_runs:
        item = results.get(run)
        if not item:
            continue
        by_type_lines.append(
            "\t".join(_fmt(item.get(header)) for header in by_type_headers)
        )
        with_open_lines.append(
            "\t".join(_fmt(item.get(header)) for header in with_open_headers)
        )

    (root / "metrics_by_answer_type_with_open.tsv").write_text(
        "\n".join(by_type_lines) + "\n",
        encoding="utf-8",
    )
    (root / "metrics_table_with_open_decision.tsv").write_text(
        "\n".join(with_open_lines) + "\n",
        encoding="utf-8",
    )
    (root / "metrics_with_open_decision.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--groups", nargs="*", default=GROUPS)
    args = parser.parse_args()

    root = Path(args.root)
    questions = read_jsonl(Path(args.questions))
    teachers = read_jsonl(Path(args.teacher))
    results: Dict[str, Dict[str, Any]] = {}
    for run in args.groups:
        run_dir = root / run
        jsonl_path = run_dir / "qwen_raw_answers.jsonl"
        if not jsonl_path.exists():
            continue
        item = compute_for_run(run_dir, questions, teachers)
        item["run"] = run
        results[run] = item
    write_outputs(root, results, args.groups)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
