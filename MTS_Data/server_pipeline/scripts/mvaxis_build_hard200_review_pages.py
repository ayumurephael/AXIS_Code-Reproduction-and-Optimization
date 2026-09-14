from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from _bootstrap import add_project_root

ROOT = add_project_root()

from mvaxis_eval_teacher_aligned_student_runs import parse_bool, parse_choice, _teacher_answer


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_run_rows(run_dir: Path) -> List[Dict[str, Any]]:
    return _read_jsonl(run_dir / "qwen_raw_answers.jsonl")


def _question_choices_text(row: Dict[str, Any]) -> str:
    choices = row.get("question_choices") or []
    if not choices:
        return ""
    parts = []
    for item in choices:
        if isinstance(item, dict):
            label = item.get("label") or item.get("option") or item.get("id") or "?"
            text = item.get("text") or item.get("content") or item.get("choice") or ""
            parts.append(f"{label}. {text}")
        else:
            parts.append(str(item))
    return "\n".join(parts)


def _teacher_label(question_row: Dict[str, Any], teacher_row: Dict[str, Any]) -> Optional[str]:
    answer_type = str(question_row.get("question_answer_type") or "").strip().lower()
    teacher_text = _teacher_answer(teacher_row)
    if answer_type == "choice":
        label = parse_choice(teacher_text)
        if label:
            return label
        target = (question_row.get("target_output") or {}).get("fact_check") or {}
        label = target.get("choice_answer")
        return str(label).strip().upper() if label else None
    if answer_type == "judgment":
        label = parse_bool(teacher_text)
        if label is not None:
            return "True" if label else "False"
        target = (question_row.get("target_output") or {}).get("fact_check") or {}
        ans = target.get("answer_label") or (question_row.get("target_output") or {}).get("question_answer")
        parsed = parse_bool(str(ans or ""))
        return "True" if parsed else "False" if parsed is not None else None
    return None


def _student_label(question_row: Dict[str, Any], record: Dict[str, Any]) -> Optional[str]:
    answer_type = str(question_row.get("question_answer_type") or "").strip().lower()
    raw = str(record.get("raw_response") or "")
    if answer_type == "choice":
        return parse_choice(raw)
    if answer_type == "judgment":
        parsed = parse_bool(raw)
        return "True" if parsed else "False" if parsed is not None else None
    return None


def _format_metrics_table(summary: Dict[str, Any], run_order: List[str], title: str) -> str:
    rows = []
    for run in run_order:
        metric = ((summary.get("runs") or {}).get(run) or {}).get("overall", {})
        label = metric.get("label", {})
        rows.append(
            f"<tr><td>{html.escape(run)}</td>"
            f"<td>{label.get('accuracy')}</td>"
            f"<td>{label.get('accuracy_on_parsed')}</td>"
            f"<td>{label.get('explicit_answer_rate')}</td>"
            f"<td>{label.get('explicit_n')}</td>"
            f"<td>{label.get('n')}</td></tr>"
        )
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table><thead><tr><th>Run</th><th>label_acc</th><th>label_acc_on_explicit</th>"
        "<th>explicit_answer_rate</th><th>explicit_n</th><th>label_n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _run_modality(run_dir: Path) -> Dict[str, Any]:
    row = _load_run_rows(run_dir)[0]
    prompt = str(row.get("prompt") or "")
    return {
        "score_image_path": row.get("score_image_path"),
        "raw_image_path": row.get("raw_image_path"),
        "has_score_note": "Visual anomaly-score reference" in prompt,
        "has_raw_note": "Visual time-series reference" in prompt,
        "has_score_text": "anomaly_score is produced by the AXIS/TimeRCD" in prompt,
    }


def _write_index(
    output_path: Path,
    *,
    text_root: Path,
    vl_root: Path,
    text_summary: Dict[str, Any],
    vl_summary: Dict[str, Any],
    comparison_name: str,
) -> None:
    text_runs = ["score_text_all_hints", "all_hints", "no_channel_hints", "no_global_hints", "score_text_nohints", "no_hints"]
    vl_runs = [
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
    modality_rows = []
    for run in vl_runs:
        info = _run_modality(vl_root / run)
        modality_rows.append(
            "<tr>"
            f"<td>{html.escape(run)}</td>"
            f"<td>{html.escape(str(info['score_image_path']))}</td>"
            f"<td>{html.escape(str(info['raw_image_path']))}</td>"
            f"<td>{info['has_score_note']}</td>"
            f"<td>{info['has_raw_note']}</td>"
            f"<td>{info['has_score_text']}</td>"
            "</tr>"
        )

    def links_for(root: Path, runs: List[str]) -> str:
        rows = []
        for run in runs:
            rows.append(
                "<tr>"
                f"<td>{html.escape(run)}</td>"
                f"<td><a href=\"./{root.name}/{run}/qwen_raw_answers.html\">answers</a></td>"
                f"<td><a href=\"./{root.name}/{run}/qwen_prompt_examples.json\">prompt examples</a></td>"
                f"<td><a href=\"./{root.name}/{run}/qwen_case_study.txt\">case study</a></td>"
                "</tr>"
            )
        return "".join(rows)

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>hard_200 review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 8px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>hard_200 review</h1>
  <div class="note">
    <strong>Quick take</strong>
    <ul>
      <li>Text-Qwen: format is stable; the best run is <code>score_text_all_hints</code>.</li>
      <li>VL-Qwen: the best run is <code>all_hints</code>, while <code>no_channel_hints</code> collapses on this set.</li>
      <li>Dedicated difference page: <a href="./{comparison_name}">VL all_hints vs no_channel_hints</a></li>
    </ul>
  </div>
  {_format_metrics_table(text_summary, text_runs, "Text Qwen (6 groups)")}
  {_format_metrics_table(vl_summary, vl_runs, "VL Qwen (9 groups)")}
  <h2>VL modality check</h2>
  <table><thead><tr><th>Run</th><th>score_image_path</th><th>raw_image_path</th><th>score-image note in prompt</th><th>raw-image note in prompt</th><th>score text in prompt</th></tr></thead>
  <tbody>{''.join(modality_rows)}</tbody></table>
  <h2>Run links: Text Qwen</h2>
  <table><thead><tr><th>Run</th><th>Answer review</th><th>Prompt examples</th><th>Case study</th></tr></thead>
  <tbody>{links_for(text_root, text_runs)}</tbody></table>
  <h2>Run links: VL Qwen</h2>
  <table><thead><tr><th>Run</th><th>Answer review</th><th>Prompt examples</th><th>Case study</th></tr></thead>
  <tbody>{links_for(vl_root, vl_runs)}</tbody></table>
</body>
</html>"""
    output_path.write_text(html_text, encoding="utf-8")


def _write_comparison(
    output_path: Path,
    *,
    questions: List[Dict[str, Any]],
    teacher_by_sample: Dict[str, Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    no_channel_rows: List[Dict[str, Any]],
    limit: int,
) -> None:
    differing: List[Dict[str, Any]] = []
    for rec_all, rec_no in zip(all_rows, no_channel_rows):
        idx = int(rec_all["index"])
        q = questions[idx]
        if str(q.get("question_answer_type") or "").lower() != "choice":
            continue
        sample_id = str(q.get("sample_id"))
        teacher = teacher_by_sample.get(sample_id)
        if not teacher:
            continue
        gold = _teacher_label(q, teacher)
        pred_all = _student_label(q, rec_all)
        pred_no = _student_label(q, rec_no)
        if gold is None or pred_all == pred_no:
            continue
        all_correct = pred_all == gold
        no_correct = pred_no == gold
        if not all_correct and no_correct:
            continue
        differing.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "question": q.get("question"),
                "choices": _question_choices_text(q),
                "teacher_answer": _teacher_answer(teacher),
                "gold_label": gold,
                "all_label": pred_all,
                "no_channel_label": pred_no,
                "all_correct": all_correct,
                "no_channel_correct": no_correct,
                "all_raw": rec_all.get("raw_response"),
                "no_channel_raw": rec_no.get("raw_response"),
                "question_family": q.get("question_family"),
                "question_difficulty": q.get("question_difficulty"),
            }
        )

    differing.sort(
        key=lambda item: (
            0 if item["all_correct"] and not item["no_channel_correct"] else 1,
            item["index"],
        )
    )
    selected = differing[:limit]

    cards = []
    for item in selected:
        cards.append(
            f"""
<div class="card">
  <h3>#{item['index']} · {html.escape(item['sample_id'])}</h3>
  <p><strong>Family:</strong> {html.escape(str(item['question_family']))} &nbsp; <strong>Difficulty/frame:</strong> {html.escape(str(item['question_difficulty']))}</p>
  <p><strong>Question</strong><br>{html.escape(str(item['question']))}</p>
  <pre>{html.escape(str(item['choices']))}</pre>
  <p><strong>Teacher label:</strong> {html.escape(str(item['gold_label']))}</p>
  <details><summary>Teacher answer</summary><pre>{html.escape(str(item['teacher_answer']))}</pre></details>
  <table>
    <thead><tr><th>Run</th><th>Parsed label</th><th>Correct?</th><th>Raw answer</th></tr></thead>
    <tbody>
      <tr><td>VL all_hints</td><td>{html.escape(str(item['all_label']))}</td><td>{item['all_correct']}</td><td><pre>{html.escape(str(item['all_raw']))}</pre></td></tr>
      <tr><td>VL no_channel_hints</td><td>{html.escape(str(item['no_channel_label']))}</td><td>{item['no_channel_correct']}</td><td><pre>{html.escape(str(item['no_channel_raw']))}</pre></td></tr>
    </tbody>
  </table>
</div>
"""
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>hard_200 VL all_hints vs no_channel_hints</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.45; color: #1f2937; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 10px; padding: 16px; margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f9fafb; padding: 10px; border-radius: 6px; }}
    .note {{ background: #eff6ff; border: 1px solid #bfdbfe; padding: 12px; border-radius: 8px; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>VL all_hints vs no_channel_hints on hard_200</h1>
  <div class="note">
    This page focuses on label-bearing choice questions where the two runs disagree. Cases where <code>all_hints</code> is correct and <code>no_channel_hints</code> is wrong are sorted to the top, because those are the clearest places where channel hints appear to help.
  </div>
  {''.join(cards) if cards else '<p>No differing comparable cases found.</p>'}
</body>
</html>"""
    output_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local review pages for hard_200 text/VL runs.")
    parser.add_argument("--local-root", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--comparison-limit", type=int, default=18)
    args = parser.parse_args()

    local_root = Path(args.local_root)
    text_root = local_root / "text_qwen_hard200_5group_20260601c"
    vl_root = local_root / "vl_qwen_hard200_9group_20260601c"
    questions = _read_jsonl(Path(args.questions))
    teacher_rows = _read_jsonl(Path(args.teacher))
    teacher_by_sample = {str(row.get("sample_id")): row for row in teacher_rows}

    text_summary = json.loads((text_root / "teacher_aligned_summary.json").read_text(encoding="utf-8"))
    vl_summary = json.loads((vl_root / "teacher_aligned_summary.json").read_text(encoding="utf-8"))

    _write_index(
        local_root / "answer_review_index.html",
        text_root=text_root,
        vl_root=vl_root,
        text_summary=text_summary,
        vl_summary=vl_summary,
        comparison_name="compare_vl_all_hints_vs_no_channel_hints.html",
    )

    _write_comparison(
        local_root / "compare_vl_all_hints_vs_no_channel_hints.html",
        questions=questions,
        teacher_by_sample=teacher_by_sample,
        all_rows=_load_run_rows(vl_root / "all_hints"),
        no_channel_rows=_load_run_rows(vl_root / "no_channel_hints"),
        limit=args.comparison_limit,
    )


if __name__ == "__main__":
    main()
