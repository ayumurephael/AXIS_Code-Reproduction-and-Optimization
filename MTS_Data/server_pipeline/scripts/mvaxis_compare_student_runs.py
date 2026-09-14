from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.utils import read_jsonl


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_bool(text: str) -> Optional[bool]:
    text = (text or "").strip()
    if not text:
        return None
    low = text.lower()
    fields = list(
        re.finditer(
            r"(?i)\b(?:answer|final answer|judgment|label)\s*[:\-]?\s*(?:is\s*)?(yes|no|true|false)\b",
            text,
        )
    )
    if fields:
        return fields[-1].group(1).lower() in {"yes", "true"}
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
    for pattern in _CHOICE_PATTERNS:
        matches = list(pattern.finditer(text or ""))
        if matches:
            return matches[-1].group(1).upper()
    first = re.match(r"\s*\(?\s*([A-E])\s*\)?\s*(?:[\.)]|$)", text or "")
    return first.group(1).upper() if first else None


def _teacher_answer(row: Dict[str, Any]) -> str:
    teacher = row.get("teacher_answer_llm") or {}
    if isinstance(teacher, dict):
        answer = teacher.get("answer")
        if answer:
            return str(answer)
    return str(row.get("answer") or row.get("model_answer") or "")


def _gold_label(question: Dict[str, Any], teacher_text: str) -> Any:
    answer_type = str(question.get("question_answer_type") or "open")
    fact = (question.get("target_output") or {}).get("fact_check") or {}
    if answer_type == "judgment":
        parsed = parse_bool(teacher_text)
        if parsed is None:
            parsed = parse_bool(str(fact.get("answer_label") or ""))
        return parsed
    if answer_type == "choice":
        parsed = parse_choice(teacher_text)
        if parsed is None and fact.get("choice_answer"):
            parsed = str(fact["choice_answer"]).strip().upper()
        return parsed
    return None


def _pred_label(answer_type: str, raw_text: str) -> Any:
    if answer_type == "judgment":
        return parse_bool(raw_text)
    if answer_type == "choice":
        return parse_choice(raw_text)
    return None


def _answer_label_display(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return "None" if value is None else str(value)


def _uri(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return Path(path).resolve().as_uri()
    except Exception:
        return None


def _load_group_rows(run_root: Path, group: str) -> List[Dict[str, Any]]:
    return read_jsonl(run_root / group / "qwen_raw_answers.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--focus-run", required=True)
    parser.add_argument("--compare-runs", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-examples", type=int, default=18)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    run_root = _resolve(args.run_root)
    questions = read_jsonl(_resolve(args.questions))[: int(args.limit)]
    teacher_rows = read_jsonl(_resolve(args.teacher))
    teacher_by_sample = {str(row.get("sample_id")): _teacher_answer(row) for row in teacher_rows}
    groups = [str(args.focus_run)] + [str(name) for name in args.compare_runs]
    run_rows = {group: _load_group_rows(run_root, group) for group in groups}

    pair_counters = {group: Counter() for group in args.compare_runs}
    bucket_counter = Counter()
    examples: List[Dict[str, Any]] = []

    for idx, question in enumerate(questions):
        answer_type = str(question.get("question_answer_type") or "open")
        if answer_type not in {"judgment", "choice"}:
            continue
        sample_id = str(question.get("sample_id"))
        teacher_text = teacher_by_sample.get(sample_id, "")
        gold = _gold_label(question, teacher_text)
        if gold is None:
            continue
        preds = {}
        raws = {}
        records = {}
        for group in groups:
            record = run_rows[group][idx]
            records[group] = record
            raw_text = str(record.get("raw_response") or "")
            raws[group] = raw_text
            preds[group] = _pred_label(answer_type, raw_text)
        focus_ok = preds[args.focus_run] == gold
        compare_status = {group: preds[group] == gold for group in args.compare_runs}
        for group, ok in compare_status.items():
            if focus_ok and not ok:
                pair_counters[group]["focus_better"] += 1
            elif (not focus_ok) and ok:
                pair_counters[group]["compare_better"] += 1
            elif focus_ok and ok:
                pair_counters[group]["both_correct"] += 1
            else:
                pair_counters[group]["both_wrong"] += 1
        if focus_ok and any(not ok for ok in compare_status.values()):
            fact = (question.get("target_output") or {}).get("fact_check") or {}
            bucket_counter[(answer_type, str(question.get("question_difficulty")), bool(fact.get("is_anomalous")))] += 1
            if len(examples) < int(args.max_examples):
                artifacts = question.get("question_generation_artifacts") or {}
                focus_record = records[args.focus_run]
                examples.append(
                    {
                        "index": idx,
                        "sample_id": sample_id,
                        "answer_type": answer_type,
                        "difficulty": question.get("question_difficulty"),
                        "is_anomalous": fact.get("is_anomalous"),
                        "gold_label": _answer_label_display(gold),
                        "teacher_answer": teacher_text,
                        "question": question.get("question"),
                        "image_path": artifacts.get("image_path"),
                        "score_image_path": focus_record.get("score_image_path"),
                        "raw_image_path": focus_record.get("raw_image_path"),
                        "groups": {
                            group: {
                                "pred_label": _answer_label_display(preds[group]),
                                "correct": preds[group] == gold,
                                "raw_response": raws[group],
                            }
                            for group in groups
                        },
                    }
                )

    payload = {
        "focus_run": args.focus_run,
        "compare_runs": args.compare_runs,
        "pair_counters": {
            group: dict(counter) for group, counter in pair_counters.items()
        },
        "bucket_counter": {
            f"{answer_type}|{difficulty}|anomalous={is_anomalous}": count
            for (answer_type, difficulty, is_anomalous), count in bucket_counter.items()
        },
        "examples": examples,
    }

    output_json = _resolve(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    compare_blocks = []
    for group in args.compare_runs:
        stats = pair_counters[group]
        compare_blocks.append(
            "<li><b>{group}</b>: focus_better={fb}, compare_better={cb}, both_correct={bc}, both_wrong={bw}</li>".format(
                group=html.escape(group),
                fb=stats.get("focus_better", 0),
                cb=stats.get("compare_better", 0),
                bc=stats.get("both_correct", 0),
                bw=stats.get("both_wrong", 0),
            )
        )

    bucket_lines = []
    for key, count in sorted(bucket_counter.items(), key=lambda item: (-item[1], item[0])):
        answer_type, difficulty, is_anomalous = key
        bucket_lines.append(
            f"<li>{html.escape(answer_type)} / {html.escape(difficulty)} / anomalous={is_anomalous}: {count}</li>"
        )

    cards = []
    for example in examples:
        question_image = _uri(example.get("image_path"))
        score_image = _uri(example.get("score_image_path"))
        parts = [
            "<section class='card'>",
            f"<h2>#{example['index']} · {html.escape(example['sample_id'])}</h2>",
            f"<p><b>type</b>: {html.escape(str(example['answer_type']))} · <b>difficulty</b>: {html.escape(str(example['difficulty']))} · <b>is_anomalous</b>: {html.escape(str(example['is_anomalous']))}</p>",
            f"<p><b>gold</b>: {html.escape(example['gold_label'])}</p>",
            f"<p><b>question</b>: {html.escape(str(example['question']))}</p>",
        ]
        if question_image:
            parts.append(f"<p><a href='{question_image}'>Open question image</a></p>")
            parts.append(f"<img class='preview' src='{question_image}' alt='question image'>")
        if score_image and score_image != question_image:
            parts.append(f"<p><a href='{score_image}'>Open score image</a></p>")
        parts.append("<details><summary>Teacher answer</summary><pre>{}</pre></details>".format(html.escape(example["teacher_answer"])))
        for group, group_payload in example["groups"].items():
            badge = "correct" if group_payload["correct"] else "wrong"
            parts.append(
                "<details open><summary><span class='badge {badge}'>{status}</span> {group} · pred={pred}</summary><pre>{raw}</pre></details>".format(
                    badge=badge,
                    status="correct" if group_payload["correct"] else "wrong",
                    group=html.escape(group),
                    pred=html.escape(str(group_payload["pred_label"])),
                    raw=html.escape(group_payload["raw_response"]),
                )
            )
        parts.append("</section>")
        cards.append("\n".join(parts))

    html_text = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Student Run Comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; background: #f6f8fb; color: #1f2937; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .card {{ background: #fff; border: 1px solid #dbe3ee; border-radius: 10px; padding: 16px; margin: 18px 0; box-shadow: 0 1px 3px rgba(15,23,42,.06); }}
    .preview {{ max-width: 100%%; border: 1px solid #d1d5db; border-radius: 8px; margin: 10px 0; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; margin-right: 8px; }}
    .badge.correct {{ background: #dcfce7; color: #166534; }}
    .badge.wrong {{ background: #fee2e2; color: #991b1b; }}
    details {{ margin: 10px 0; }}
  </style>
</head>
<body>
  <h1>Focus Run Comparison</h1>
  <p><b>focus</b>: {focus}</p>
  <p>This page lists examples where the focus run is correct on the teacher-aligned label while at least one comparator is wrong.</p>
  <h2>Pairwise Summary</h2>
  <ul>{pairwise}</ul>
  <h2>Bucket Summary</h2>
  <ul>{buckets}</ul>
  {cards}
</body>
</html>
""".format(
        focus=html.escape(str(args.focus_run)),
        pairwise="\n".join(compare_blocks),
        buckets="\n".join(bucket_lines),
        cards="\n".join(cards),
    )
    output_html = _resolve(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "focus_run": args.focus_run,
                "compare_runs": args.compare_runs,
                "output_html": str(output_html),
                "output_json": str(output_json),
                "num_examples": len(examples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
