from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from _bootstrap import add_project_root

ROOT = add_project_root()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_bool(text: str) -> Optional[bool]:
    text = (text or "").strip()
    if not text:
        return None
    low = text.lower()
    answer_line = re.search(
        r"(?im)^\s*answer\s*:\s*(?:is\s*)?(yes|no|true|false)\b",
        text,
    )
    if answer_line:
        return answer_line.group(1).lower() in {"yes", "true"}
    fields = list(
        re.finditer(
            r"(?i)\b(?:answer|final answer|judgment|label)\s*[:\-]?\s*(?:is\s*)?(yes|no|true|false)\b",
            text,
        )
    )
    if fields:
        return fields[0].group(1).lower() in {"yes", "true"}
    answer_sentence = list(
        re.finditer(
            r"(?i)\b(?:the\s+)?(?:answer|judgment)\s+(?:should\s+be|is)\s+(yes|no|true|false)\b",
            text,
        )
    )
    if answer_sentence:
        return answer_sentence[-1].group(1).lower() in {"yes", "true"}
    direct = re.match(r"\s*(yes|no|true|false)\b", low)
    if direct:
        return direct.group(1) in {"yes", "true"}
    return None


_CHOICE_PATTERNS = [
    re.compile(r"(?i)\b(?:answer|final answer|option|choice|selected option|correct option)\s*(?:is|:)?\s*\(?\s*([A-E])\s*\)?\b"),
    re.compile(r"(?i)\b(?:the\s+)?(?:answer|correct answer|best answer|best option)\s+(?:should\s+be|is)\s*\(?\s*([A-E])\s*\)?\b"),
    re.compile(r"(?im)^\s*(?:final answer\s*[:\-]\s*)?\(?\s*([A-E])\s*\)?\s*[\.)]\s+"),
    re.compile(r"(?i)\bI\s*(?:would\s*)?(?:choose|select|pick)\s*\(?\s*([A-E])\s*\)?\b"),
]


def parse_choice(text: str) -> Optional[str]:
    head = text or ""
    answer_line = re.search(
        r"(?im)^\s*answer\s*:\s*\(?\s*([A-E])\s*\)?\b",
        head,
    )
    if answer_line:
        return answer_line.group(1).upper()
    for pattern in _CHOICE_PATTERNS:
        matches = list(pattern.finditer(head))
        if matches:
            return matches[0].group(1).upper()
    first = re.match(r"\s*\(?\s*([A-E])\s*\)?\s*(?:[\.)]|$)", head)
    return first.group(1).upper() if first else None


def _teacher_answer(row: Dict[str, Any]) -> str:
    teacher = row.get("teacher_answer_llm") or {}
    if isinstance(teacher, dict):
        return str(teacher.get("answer") or "")
    return str(row.get("answer") or "")


def _question_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    fact = (row.get("target_output") or {}).get("fact_check") or {}
    target = row.get("target_output") or {}
    return {
        "sample_id": row.get("sample_id"),
        "answer_type": row.get("question_answer_type") or fact.get("question_answer_type") or "unknown",
        "difficulty": row.get("question_difficulty") or fact.get("question_difficulty") or "unknown",
        "is_anomalous": fact.get("is_anomalous"),
        "choice_answer": fact.get("choice_answer"),
        "answer_label": fact.get("answer_label") or target.get("question_answer") or target.get("final_answer"),
    }


def _inc(counter: Counter, prefix: str, pred: Any, gold: Any) -> None:
    counter[f"{prefix}_n"] += 1
    if pred is not None:
        counter[f"{prefix}_parsed"] += 1
        if pred == gold:
            counter[f"{prefix}_correct"] += 1


def _inc_explicit(counter: Counter, pred: Any) -> None:
    counter["explicit_n"] += 1
    if pred is not None:
        counter["explicit_parsed"] += 1


def _metric(counter: Counter, prefix: str) -> Dict[str, Any]:
    n = int(counter.get(f"{prefix}_n", 0))
    parsed = int(counter.get(f"{prefix}_parsed", 0))
    correct = int(counter.get(f"{prefix}_correct", 0))
    return {
        "n": n,
        "parsed": parsed,
        "correct": correct,
        "parse_rate": parsed / n if n else None,
        "accuracy": correct / n if n else None,
        "accuracy_on_parsed": correct / parsed if parsed else None,
    }


def _label_metric(counter: Counter) -> Dict[str, Any]:
    metric = _metric(counter, "label")
    explicit_n = int(counter.get("explicit_n", 0))
    explicit_parsed = int(counter.get("explicit_parsed", 0))
    metric["explicit_n"] = explicit_n
    metric["explicit_parsed"] = explicit_parsed
    metric["explicit_answer_rate"] = explicit_parsed / explicit_n if explicit_n else None
    return metric


def _round_tree(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _round_tree(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_round_tree(item) for item in value]
    return value


def evaluate_run(
    run_dir: Path,
    *,
    questions: List[Dict[str, Any]],
    teacher_by_sample: Dict[str, str],
) -> Dict[str, Any]:
    rows = _read_jsonl(run_dir / "qwen_raw_answers.jsonl")
    overall = Counter()
    buckets: Dict[str, Counter] = defaultdict(Counter)
    examples: List[Dict[str, Any]] = []
    for record in rows:
        idx = int(record["index"])
        question = questions[idx]
        meta = _question_meta(question)
        raw = str(record.get("raw_response") or "")
        teacher = teacher_by_sample.get(str(meta["sample_id"]), "")
        answer_type = str(meta["answer_type"])
        difficulty = str(meta["difficulty"])

        pred_label: Any = None
        gold_label: Any = None
        if answer_type == "judgment":
            pred_label = parse_bool(raw)
            gold_label = parse_bool(teacher)
        elif answer_type == "choice":
            pred_label = parse_choice(raw)
            gold_label = parse_choice(teacher)

        if answer_type in {"judgment", "choice"}:
            label_buckets = (
                overall,
                buckets[f"answer_type={answer_type}"],
                buckets[f"difficulty={difficulty}"],
                buckets[f"difficulty={difficulty}|answer_type={answer_type}"],
            )
            for bucket in label_buckets:
                _inc_explicit(bucket, pred_label)
            if gold_label is not None:
                for bucket in label_buckets:
                    _inc(bucket, "label", pred_label, gold_label)
            if pred_label != gold_label and len(examples) < 30:
                examples.append(
                    {
                        "index": idx,
                        "answer_type": answer_type,
                        "difficulty": difficulty,
                        "pred_label": pred_label,
                        "gold_label": gold_label,
                        "student": raw[:500],
                        "teacher": teacher[:500],
                        "question": str(question.get("question") or "")[:300],
                    }
                )

    summary = {
        "overall": {"label": _label_metric(overall)},
        "by_category": {
            key: {"label": _label_metric(counter)}
            for key, counter in sorted(buckets.items())
        },
        "error_examples": examples,
    }
    report_path = run_dir / "qwen_raw_report.json"
    if report_path.exists():
        summary["script_report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return _round_tree(summary)


def _format_table(summary: Dict[str, Any], runs: Iterable[str]) -> str:
    lines = []
    lines.append("Teacher-aligned student answer evaluation")
    lines.append("label: full-text explicit judgment yes/no/true/false + choice A-E aligned to GPT teacher")
    lines.append("explicit_answer_rate: fraction of judgment/choice questions where an explicit final answer can be parsed from the full response")
    lines.append("")
    lines.append(
        f"{'run':<30} {'label_acc':>9} {'label_acc_on_explicit':>21} {'explicit_answer_rate':>21} {'explicit_n':>10} {'label_n':>7}"
    )
    for run in runs:
        item = summary["runs"][run]["overall"]
        label = item["label"]
        lines.append(
            f"{run:<30} {str(label['accuracy']):>9} {str(label['accuracy_on_parsed']):>21} "
            f"{str(label['explicit_answer_rate']):>21} {str(label.get('explicit_n')):>10} {str(label['n']):>7}"
        )
    lines.append("")
    for run in runs:
        lines.append(f"[{run}] by answer_type")
        cats = summary["runs"][run]["by_category"]
        for answer_type in ["judgment", "choice", "open"]:
            cat = cats.get(f"answer_type={answer_type}", {})
            label = cat.get("label", {})
            lines.append(
                f"  {answer_type:<8} label_acc={label.get('accuracy')} "
                f"label_acc_on_explicit={label.get('accuracy_on_parsed')} "
                f"explicit_answer_rate={label.get('explicit_answer_rate')} "
                f"explicit_n={label.get('explicit_n')} label_n={label.get('n')}"
            )
        lines.append(f"[{run}] by difficulty")
        for difficulty in ["simple", "medium", "complex"]:
            cat = cats.get(f"difficulty={difficulty}", {})
            label = cat.get("label", {})
            lines.append(
                f"  {difficulty:<8} label_acc={label.get('accuracy')} "
                f"label_acc_on_explicit={label.get('accuracy_on_parsed')} "
                f"explicit_answer_rate={label.get('explicit_answer_rate')} "
                f"explicit_n={label.get('explicit_n')} label_n={label.get('n')}"
            )
        lines.append("")
    return "\n".join(lines)


def _format_metrics_tsv(summary: Dict[str, Any], runs: Iterable[str]) -> str:
    headers = [
        "run",
        "label_acc",
        "label_acc_on_explicit",
        "explicit_answer_rate",
        "explicit_n",
        "label_n",
    ]
    lines = ["\t".join(headers)]
    for run in runs:
        label = summary["runs"][run]["overall"]["label"]
        cells = [
            run,
            str(label.get("accuracy")),
            str(label.get("accuracy_on_parsed")),
            str(label.get("explicit_answer_rate")),
            str(label.get("explicit_n")),
            str(label.get("n")),
        ]
        lines.append("\t".join(cells))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--questions", default="outputs/questiongeneration_final/questions_600.jsonl")
    parser.add_argument("--teacher", default="outputs/quetiongeneration_1_teacher_answer/teacher_gpt55.jsonl")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_root = _resolve(args.run_root)
    questions = _read_jsonl(_resolve(args.questions))
    if args.limit is not None:
        questions = questions[: int(args.limit)]
    teacher_by_sample = {str(row.get("sample_id")): _teacher_answer(row) for row in _read_jsonl(_resolve(args.teacher))}
    summary = {
        "run_root": str(run_root),
        "questions": str(_resolve(args.questions)),
        "teacher": str(_resolve(args.teacher)),
        "runs": {},
    }
    for run in args.runs:
        summary["runs"][run] = evaluate_run(run_root / run, questions=questions, teacher_by_sample=teacher_by_sample)
    text = _format_table(summary, args.runs)
    metrics_tsv = _format_metrics_tsv(summary, args.runs)
    (run_root / "teacher_aligned_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_root / "teacher_aligned_summary.txt").write_text(text, encoding="utf-8")
    (run_root / "metrics_table.tsv").write_text(metrics_tsv, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
