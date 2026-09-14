from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _question_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return str(row.get("sample_id") or ""), str(row.get("question") or "").strip()


def _teacher_lookup(rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {_question_key(row): row for row in rows}


def _teacher_fact_label(row: Dict[str, Any]) -> Optional[str]:
    fact = (row.get("target_output") or {}).get("fact_check") or {}
    qtype = str(fact.get("question_answer_type") or row.get("question_answer_type") or "").strip().lower()
    if qtype == "choice":
        answer = fact.get("choice_answer")
        return None if answer in (None, "") else str(answer).strip()
    if qtype == "judgment":
        value = fact.get("is_anomalous")
        if value is None:
            return None
        return "yes" if bool(value) else "no"
    return None


def _student_label(row: Dict[str, Any]) -> Optional[str]:
    parsed = row.get("parsed_student_answer") or row.get("parsed_response") or {}
    label = parsed.get("answer_label")
    if label in (None, ""):
        final_answer = parsed.get("final_answer")
        if final_answer in (None, ""):
            return None
        label = final_answer
    text = str(label).strip()
    lowered = text.lower()
    if lowered in {"yes", "answer: yes", "decision: yes"}:
        return "yes"
    if lowered in {"no", "answer: no", "decision: no"}:
        return "no"
    if len(text) == 1 and text.upper() in {"A", "B", "C", "D", "E"}:
        return text.upper()
    return text


def _match_status(teacher_label: Optional[str], student_label: Optional[str]) -> Tuple[str, str]:
    if teacher_label is None:
        return "warn", "teacher label unavailable"
    if student_label is None:
        return "warn", "student unparsed"
    if str(teacher_label).strip().lower() == str(student_label).strip().lower():
        return "good", f"label match: {student_label}"
    return "bad", f"label mismatch: teacher={teacher_label} student={student_label}"


def _safe_json(obj: Any) -> str:
    return html.escape(json.dumps(obj, ensure_ascii=False, indent=2))


def _truth_text(row: Dict[str, Any]) -> str:
    fact = (row.get("target_output") or {}).get("fact_check") or {}
    root = fact.get("root_cause_channel")
    affected = fact.get("affected_channels") or []
    anomaly_type = fact.get("anomaly_type")
    scope = fact.get("anomaly_scope")
    is_anomalous = fact.get("is_anomalous")
    choice_answer = fact.get("choice_answer")
    lines = [
        f"is_anomalous: {is_anomalous}",
        f"root: {root}",
        f"affected: {affected}",
        f"type: {anomaly_type}",
        f"scope: {scope}",
    ]
    if choice_answer is not None:
        lines.append(f"choice_answer: {choice_answer}")
    return "\n".join(lines)


def _window_range(row: Dict[str, Any]) -> str:
    interval = row.get("prompt_interval") or row.get("target_interval") or {}
    if isinstance(interval, dict):
        start = interval.get("start")
        end = interval.get("end")
    elif isinstance(interval, (list, tuple)) and len(interval) >= 2:
        start = interval[0]
        end = interval[1]
    else:
        start = None
        end = None
    if start is None or end is None:
        return "unknown"
    return f"[{start}, {end}]"


def build_html(
    *,
    title: str,
    student_path: Path,
    teacher_path: Path,
    output_path: Path,
) -> None:
    student_rows = read_jsonl(student_path)
    teacher_rows = read_jsonl(teacher_path)
    teacher_by_key = _teacher_lookup(teacher_rows)

    cards: List[str] = []
    matched = 0
    missing_teacher = 0

    for idx, student in enumerate(student_rows, start=1):
        teacher = teacher_by_key.get(_question_key(student))
        if teacher is None:
            missing_teacher += 1
            teacher = {}
        else:
            matched += 1

        parsed = student.get("parsed_student_answer") or student.get("parsed_response") or {}
        teacher_answer_llm = (teacher.get("teacher_answer_llm") or {}).get("answer")
        teacher_structured = ((teacher.get("windows") or [{}])[0] or {}).get("answer")
        teacher_label = _teacher_fact_label(teacher) if teacher else None
        student_label = _student_label(student)
        badge_kind, badge_text = _match_status(teacher_label, student_label)
        image_path = student.get("raw_image_path")
        image_link = ""
        image_panel = ""
        content_class = "content"
        if image_path:
            href = html.escape(str(image_path))
            image_link = f'<a href="file:///{href.replace(chr(92), "/")}">image</a>'
            image_panel = f'<div><img src="file:///{href.replace(chr(92), "/")}" alt="raw image"></div>'
        else:
            content_class = "content no-image"

        question = html.escape(str(student.get("question") or ""))
        raw_response = html.escape(str(student.get("raw_response") or ""))
        teacher_answer_llm_text = html.escape(str(teacher_answer_llm or ""))
        teacher_structured_text = html.escape(str(teacher_structured or ""))
        truth_text = html.escape(_truth_text(teacher) if teacher else "teacher row missing")
        parsed_text = html.escape(json.dumps(parsed, ensure_ascii=False, indent=2))
        prompt_interval = html.escape(_window_range(student))
        sample_id = html.escape(str(student.get("sample_id") or ""))
        question_type = html.escape(str(student.get("question_type") or ""))
        question_id = html.escape(str(teacher.get("question_id") or ""))
        answer_type = html.escape(str((teacher.get("target_output") or {}).get("fact_check", {}).get("question_answer_type") or ""))
        difficulty = html.escape(str((teacher.get("target_output") or {}).get("fact_check", {}).get("question_difficulty") or ""))
        target_output_text = _safe_json(teacher.get("target_output")) if teacher else "null"
        teacher_llm_text_json = _safe_json(teacher.get("teacher_answer_llm")) if teacher else "null"
        parsed_final = html.escape(str(parsed.get("final_answer") or ""))

        cards.append(
            f"""<section class="card" id="case-{idx:04d}">
  <div class="card-header">
    <div class="meta">
      <span class="badge">#{idx:04d}</span>
      <span class="badge">{sample_id}</span>
      <span class="badge">{question_id or question_type}</span>
      <span class="badge">type={answer_type or "unknown"}</span>
      <span class="badge">difficulty={difficulty or "unknown"}</span>
      <span class="badge">interval={prompt_interval}</span>
      <span class="badge {badge_kind}">{html.escape(badge_text)}</span>
    </div>
    <div class="meta">{image_link}</div>
  </div>
  <div class="{content_class}">
    <div>
      <div class="question">{question}</div>
      <div class="answer-box reference">
        <h3>Teacher / Structured Reference</h3>
        <div class="answer-text">{teacher_structured_text}</div>
      </div>
      <div class="answer-box reference">
        <h3>Teacher LLM Answer</h3>
        <div class="answer-text">{teacher_answer_llm_text}</div>
      </div>
      <div class="answer-box model">
        <h3>ChatTS Parsed Final Answer</h3>
        <div class="answer-text">{parsed_final}</div>
      </div>
      <div class="answer-box model">
        <h3>ChatTS Raw Answer</h3>
        <div class="answer-text">{raw_response}</div>
      </div>
      <div class="answer-box">
        <h3>Fact / label summary</h3>
        <div class="answer-text">{truth_text}</div>
      </div>
      <details>
        <summary>Parsed student answer / metadata</summary>
        <pre>{parsed_text}</pre>
      </details>
      <details>
        <summary>Teacher target_output</summary>
        <pre>{target_output_text}</pre>
      </details>
      <details>
        <summary>Teacher answer llm raw metadata</summary>
        <pre>{teacher_llm_text_json}</pre>
      </details>
    </div>
    {image_panel}
  </div>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #667085;
      --line: #d7dde7;
      --accent: #0f766e;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.55 "Segoe UI", Arial, sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(246, 247, 249, 0.95);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 22px;
      letter-spacing: 0;
    }}
    .subtle {{ color: var(--muted); }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 24px 48px;
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    .badge.good {{ color: var(--good); border-color: #99d6cf; background: #ecfdf9; }}
    .badge.bad {{ color: var(--bad); border-color: #f3b7b1; background: #fff3f1; }}
    .badge.warn {{ color: var(--warn); border-color: #f7c56d; background: #fffaeb; }}
    .content {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(360px, 0.95fr);
      gap: 16px;
      padding: 14px;
    }}
    .content.no-image {{ grid-template-columns: minmax(0, 1fr); }}
    h2, h3 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0;
    }}
    .question {{
      font-size: 17px;
      font-weight: 650;
      margin: 0 0 12px;
      white-space: pre-wrap;
    }}
    .answer-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin: 10px 0;
      background: #fcfcfd;
    }}
    .answer-box.reference {{ border-left: 4px solid var(--accent); }}
    .answer-box.model {{ border-left: 4px solid #475467; }}
    .answer-text {{ white-space: pre-wrap; }}
    img {{
      width: 100%;
      max-height: 720px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    details {{
      border-top: 1px dashed var(--line);
      margin-top: 10px;
      padding-top: 8px;
    }}
    summary {{ cursor: pointer; color: var(--muted); }}
    pre {{
      margin: 8px 0 0;
      padding: 10px;
      overflow: auto;
      max-height: 420px;
      border-radius: 8px;
      background: #101828;
      color: #f2f4f7;
      font-size: 12px;
      line-height: 1.45;
    }}
    a {{ color: #0f766e; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    @media (max-width: 980px) {{
      .content {{ grid-template-columns: minmax(0, 1fr); }}
      header, main {{ padding-left: 14px; padding-right: 14px; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>{html.escape(title)}</h1>
  <div class="toolbar subtle">
    <span>student source: {html.escape(str(student_path))}</span>
    <span>|</span>
    <span>teacher source: {html.escape(str(teacher_path))}</span>
    <span>|</span>
    <span>rows: {len(student_rows)}</span>
    <span>|</span>
    <span>teacher matched: {matched}</span>
    <span>|</span>
    <span>teacher missing: {missing_teacher}</span>
  </div>
</header>
<main>
{''.join(cards)}
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()
    build_html(
        title=args.title,
        student_path=Path(args.student),
        teacher_path=Path(args.teacher),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
