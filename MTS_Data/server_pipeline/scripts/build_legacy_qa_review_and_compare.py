from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from _bootstrap import add_project_root

ROOT = add_project_root()


LEGACY_DATASETS = {
    "AXIS_qa_test": ROOT / "legacy" / "AXIS_repo" / "data" / "AXIS_qa_test" / "series",
    "QA_datasets": ROOT / "legacy" / "AXIS_repo" / "data" / "QA_datasets" / "data" / "series",
    "MCQ_datasets": ROOT / "legacy" / "AXIS_repo" / "data" / "MCQ_datasets" / "data" / "series",
}

DEFAULT_TEACHER_DIR = ROOT / "outputs" / "teacheranswer_question_final_530_structured100"
DEFAULT_TEACHER_QUESTIONS = ROOT / "outputs" / "question_final_530" / "questions_600.jsonl"
DEFAULT_TEACHER_ANSWERS = DEFAULT_TEACHER_DIR / "teacher_gpt55.answers.jsonl"
DEFAULT_TEACHER_AUDIT = (
    ROOT / "mvaxis_answer_audit" / "results" / "teacheranswer_question_final_530_structured100_repair"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "legacy_qa_review"


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    return text or "report"


def normalize_signature(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"ch_\d+", "ch_x", lowered)
    lowered = re.sub(r"\d+", "<n>", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def shorten(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def rel_uri(path: Path, html_path: Path) -> str:
    return path.relative_to(html_path.parent).as_posix()


def _series_to_polyline(values: Sequence[float], width: int, height: int, padding: int) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if abs(span) < 1e-9:
        span = 1.0
    usable_w = max(1, width - 2 * padding)
    usable_h = max(1, height - 2 * padding)
    coords: List[str] = []
    last_index = max(1, len(values) - 1)
    for idx, value in enumerate(values):
        x = padding + usable_w * (idx / last_index)
        y = padding + usable_h * (1.0 - ((value - lo) / span))
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def build_univariate_chart(
    current_values: Sequence[float],
    normal_values: Sequence[float],
    start: int,
    end: int,
    *,
    width: int = 720,
    height: int = 180,
    padding: int = 16,
) -> str:
    if not current_values:
        return '<div class="empty-chart">No time-series values</div>'
    length = len(current_values)
    clamped_start = max(0, min(length - 1, int(start)))
    clamped_end = max(clamped_start + 1, min(length, int(end)))
    current_poly = _series_to_polyline(current_values, width, height, padding)
    normal_poly = _series_to_polyline(normal_values or current_values, width, height, padding)
    usable_w = max(1, width - 2 * padding)
    last_index = max(1, length - 1)
    x0 = padding + usable_w * (clamped_start / last_index)
    x1 = padding + usable_w * ((clamped_end - 1) / last_index)
    rect_w = max(2.0, x1 - x0)
    return f"""
<svg class="series-chart" viewBox="0 0 {width} {height}" role="img" aria-label="time-series chart">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"></rect>
  <rect x="{x0:.2f}" y="0" width="{rect_w:.2f}" height="{height}" fill="rgba(245, 158, 11, 0.18)"></rect>
  <polyline points="{normal_poly}" fill="none" stroke="#94a3b8" stroke-width="1.5"></polyline>
  <polyline points="{current_poly}" fill="none" stroke="#0f766e" stroke-width="2.2"></polyline>
</svg>"""


def json_block(value: Any) -> str:
    return esc(json.dumps(value, ensure_ascii=False, indent=2))


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #667085;
      --line: #d7dde7;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b42318;
      --accent: #155eef;
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
      z-index: 3;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 20px 24px 48px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 22px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin-bottom: 8px; }}
    h3 {{ font-size: 15px; margin-bottom: 8px; }}
    p {{ margin: 0; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .subtle {{ color: var(--muted); }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      align-items: center;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 16px 0;
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
    .content {{
      padding: 14px;
      display: grid;
      gap: 14px;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 9px;
      font-size: 12px;
      color: var(--muted);
      background: #fff;
    }}
    .badge.good {{
      color: var(--good);
      border-color: rgba(15, 118, 110, 0.22);
      background: rgba(15, 118, 110, 0.08);
    }}
    .badge.warn {{
      color: var(--warn);
      border-color: rgba(180, 83, 9, 0.24);
      background: rgba(180, 83, 9, 0.09);
    }}
    .badge.bad {{
      color: var(--bad);
      border-color: rgba(180, 35, 24, 0.22);
      background: rgba(180, 35, 24, 0.08);
    }}
    .answer-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
    }}
    .answer-box.question {{
      background: #f9fbff;
    }}
    .answer-text {{
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .chart-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }}
    .series-chart {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 6px;
    }}
    .empty-chart {{
      color: var(--muted);
      padding: 18px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 20px;
      font-weight: 600;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    li + li {{
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      vertical-align: top;
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      background: #fbfcfe;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #101828;
      color: #f8fafc;
      padding: 12px;
      border-radius: 8px;
      overflow: auto;
    }}
    details summary {{
      cursor: pointer;
      color: var(--accent);
      margin-bottom: 8px;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>"""


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def flatten_legacy_dataset(dataset_name: str, series_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for series_path in sorted(series_dir.glob("series_*.json")):
        payload = read_json(series_path)
        current_values = list(payload.get("original_data", {}).get("time_series") or [])
        normal_values = list(payload.get("original_data", {}).get("normal_series") or [])
        for window_index, window in enumerate(payload.get("windows") or []):
            window_range = window.get("window_range") or {}
            start = int(window_range.get("start", 0))
            end = int(window_range.get("end", start))
            rows.append(
                {
                    "dataset": dataset_name,
                    "series_file": str(series_path),
                    "series_name": series_path.name,
                    "sample_id": payload.get("sample_id"),
                    "window_index": window_index,
                    "window_start": start,
                    "window_end": end,
                    "question": str(window.get("question") or ""),
                    "answer": str(window.get("answer") or ""),
                    "question_type": str(window.get("question_type") or ""),
                    "has_anomaly": bool(window.get("has_anomaly")),
                    "anomaly_descriptions": list(window.get("anomaly_descriptions") or []),
                    "options": window.get("options") or {},
                    "correct_options": list(window.get("correct_options") or []),
                    "time_series": current_values,
                    "normal_series": normal_values,
                }
            )
    return rows


def teacher_label_from_answer(text: str) -> str:
    match = re.search(r"(?im)^answer:\s*([A-Z])\b", text or "")
    if match:
        return match.group(1)
    first = (text or "").strip().splitlines()
    return first[0] if first else ""


def collect_teacher_rows(questions_path: Path, answers_path: Path) -> List[Dict[str, Any]]:
    questions = read_jsonl(questions_path)
    answers = read_jsonl(answers_path)
    rows: List[Dict[str, Any]] = []
    for index, (question_row, answer_row) in enumerate(zip(questions, answers), start=1):
        target = question_row.get("target_output") or {}
        fact = target.get("fact_check") or {}
        answer_text = str(answer_row.get("model_answer") or answer_row.get("answer") or "")
        rows.append(
            {
                "index": index,
                "sample_id": question_row.get("sample_id"),
                "question": str(question_row.get("question") or ""),
                "reference_answer": str(
                    target.get("final_answer")
                    or target.get("question_answer")
                    or ""
                ),
                "teacher_answer": answer_text,
                "teacher_choice": teacher_label_from_answer(answer_text),
                "question_type": str(question_row.get("question_type") or ""),
                "question_answer_type": str(question_row.get("question_answer_type") or ""),
                "question_family": str(question_row.get("question_family") or ""),
                "is_anomalous": bool(fact.get("is_anomalous")),
                "root_cause_channels": list(fact.get("root_cause_channels") or []),
                "affected_channels": list(fact.get("affected_channels") or []),
                "anomaly_type": str(fact.get("anomaly_type") or ""),
                "image_path": str((question_row.get("question_generation_artifacts") or {}).get("image_path") or ""),
            }
        )
    return rows


def build_stats(rows: Sequence[Dict[str, Any]], *, answer_key: str = "answer") -> Dict[str, Any]:
    question_lengths = [len(str(row.get("question") or "")) for row in rows]
    answer_lengths = [len(str(row.get(answer_key) or "")) for row in rows]
    question_signatures = Counter(normalize_signature(str(row.get("question") or "")) for row in rows if row.get("question"))
    question_types = Counter(str(row.get("question_type") or row.get("question_answer_type") or "unknown") for row in rows)
    anomaly_ratio = (sum(1 for row in rows if row.get("has_anomaly") or row.get("is_anomalous")) / len(rows)) if rows else 0.0
    option_counts: List[int] = []
    for row in rows:
        options = row.get("options")
        if isinstance(options, dict) and options:
            option_counts.append(len(options))
        elif isinstance(row.get("teacher_choice"), str) and row.get("teacher_choice"):
            option_counts.append(0)
    return {
        "rows": len(rows),
        "anomaly_ratio": anomaly_ratio,
        "avg_question_chars": mean(question_lengths) if question_lengths else 0.0,
        "avg_answer_chars": mean(answer_lengths) if answer_lengths else 0.0,
        "question_type_counts": dict(question_types.most_common()),
        "top_question_signatures": question_signatures.most_common(5),
        "avg_option_count": mean(option_counts) if option_counts else None,
    }


def dataset_specific_findings(name: str, rows: Sequence[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    stats = build_stats(rows)
    if name == "MCQ_datasets":
        findings.append("Questions are easy to parse, but most option sets are shallow label tests with obvious anomaly-name distractors.")
        if stats["avg_option_count"] is not None:
            findings.append(f"Average option count is {stats['avg_option_count']:.1f}, which keeps review simple but limits distractor richness.")
    elif name == "QA_datasets":
        findings.append("Open QA answers are concise and usually faithful, but they often restate the anomaly label directly instead of explaining why the window is visually distinguishable.")
        findings.append("Question framing is repetitive and single-series only, so it does not stress cross-channel reasoning.")
    elif name == "AXIS_qa_test":
        findings.append("Prompts are more varied and explanatory than the older QA/MCQ sets, but many answers rely on stock phrases like 'current_value compared to normal_value' and boundary-aware boilerplate.")
        findings.append("These rows are useful for free-form review, but answer style is verbose relative to the actual evidence contained in the window.")
    return findings


def build_legacy_review_page(dataset_name: str, rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    stats = build_stats(rows)
    cards: List[str] = []
    for idx, row in enumerate(rows, start=1):
        chart = build_univariate_chart(
            row.get("time_series") or [],
            row.get("normal_series") or [],
            int(row.get("window_start") or 0),
            int(row.get("window_end") or 0),
        )
        option_box = ""
        options = row.get("options") or {}
        if options:
            option_lines = "\n".join(f"{key}: {value}" for key, value in options.items())
            option_box = f"""
      <div class="answer-box">
        <h3>Options</h3>
        <div class="answer-text">{esc(option_lines)}</div>
      </div>"""
        anomaly_kind = "bad" if row.get("has_anomaly") else "good"
        cards.append(
            f"""<section class="card">
  <div class="card-header">
    <div class="badges">
      <span class="badge">#{idx:04d}</span>
      <span class="badge">{esc(row.get('series_name'))}</span>
      <span class="badge">{esc(row.get('question_type'))}</span>
      <span class="badge {anomaly_kind}">{'anomalous' if row.get('has_anomaly') else 'normal'}</span>
      <span class="badge">{esc(row.get('window_start'))}:{esc(row.get('window_end'))}</span>
    </div>
    <div class="subtle">sample_id={esc(row.get('sample_id'))} window={esc(row.get('window_index'))}</div>
  </div>
  <div class="content">
    <div class="chart-wrap">{chart}</div>
    <div class="answer-box question">
      <h3>Question</h3>
      <div class="answer-text">{esc(row.get('question'))}</div>
    </div>
    {option_box}
    <div class="answer-box">
      <h3>Reference Answer</h3>
      <div class="answer-text">{esc(row.get('answer'))}</div>
    </div>
    <details>
      <summary>Metadata</summary>
      <pre>{json_block({
        "anomaly_descriptions": row.get("anomaly_descriptions"),
        "correct_options": row.get("correct_options"),
        "series_file": row.get("series_file"),
      })}</pre>
    </details>
  </div>
</section>"""
        )
    body = (
        "<header>"
        f"<h1>{esc(dataset_name)} Review</h1>"
        f'<div class="toolbar subtle">rows: {len(rows)} | anomaly ratio: {stats["anomaly_ratio"]:.1%} | '
        f'avg question chars: {stats["avg_question_chars"]:.1f} | avg answer chars: {stats["avg_answer_chars"]:.1f}</div>'
        "</header><main>"
        + "\n".join(cards)
        + "</main>"
    )
    write_text(out_path, page(f"{dataset_name} Review", body))


def build_index_page(entries: Sequence[Dict[str, Any]], report_path: Path, out_path: Path) -> None:
    cards = []
    for entry in entries:
        cards.append(
            f"""<section class="card">
  <div class="card-header">
    <h2>{esc(entry['name'])}</h2>
    <div class="badges">
      <span class="badge">{entry['rows']} rows</span>
      <span class="badge">{entry['anomaly_ratio']:.1%} anomalous</span>
    </div>
  </div>
  <div class="content">
    <p class="subtle">avg question chars: {entry['avg_question_chars']:.1f} | avg answer chars: {entry['avg_answer_chars']:.1f}</p>
    <p><a href="{esc(rel_uri(entry['path'], out_path))}">Open review page</a></p>
  </div>
</section>"""
        )
    body = (
        "<header>"
        "<h1>Legacy QA Review Index</h1>"
        f'<div class="toolbar"><a href="{esc(rel_uri(report_path, out_path))}">Open quality comparison report</a></div>'
        "</header><main>"
        + "\n".join(cards)
        + "</main>"
    )
    write_text(out_path, page("Legacy QA Review Index", body))


def build_quality_report(
    legacy_rows_by_dataset: Dict[str, List[Dict[str, Any]]],
    teacher_rows: Sequence[Dict[str, Any]],
    audit_summary: Dict[str, Any],
    audit_rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Dict[str, Any]:
    legacy_stats = {name: build_stats(rows) for name, rows in legacy_rows_by_dataset.items()}
    teacher_stats = build_stats(teacher_rows, answer_key="teacher_answer")
    total_legacy_rows = sum(stats["rows"] for stats in legacy_stats.values())
    repeated_legacy_signatures = sum(count for _, count in Counter(
        normalize_signature(str(row.get("question") or ""))
        for rows in legacy_rows_by_dataset.values()
        for row in rows
    ).most_common(10))
    teacher_issue_rows = [
        row for row in audit_rows
        if str(row.get("verdict") or "aligned") != "aligned" or int(row.get("severity") or 0) > 0
    ]

    strongest_points = [
        "Compared with the legacy sets, structured100 questions are much stronger on multichannel reasoning: they explicitly name channels, force anomaly-vs-distractor discrimination, and separate one-channel from broader event interpretations.",
        "Teacher answers are tightly formatted for downstream evaluation, and the audit summary is good: 97 aligned, 2 minor issues, 1 misaligned out of 100 reviewed rows.",
        "Legacy MCQ and QA pairs are easier to skim, but they mostly test anomaly naming on a single series; they do not push relational reasoning or explanation quality very far.",
    ]
    risks = [
        "Structured100 still shows some over-interpretation pressure. The audit found unsupported visual claim, distractor-channel, affected-channel, and anomaly-type mismatches. That usually means the answer writer is slightly too eager to narrate secondary channels.",
        "Some structured100 questions use softer decision frames such as 'safest interpretation' or 'worth flagging for review'. Those are readable for humans, but they can blur the gold label boundary if you want strict anomaly/normal supervision.",
        "Legacy AXIS_qa_test answers are verbose in a different way: they often include generic boilerplate about boundary checks and comparing current versus normal values, which pads answers without adding much discriminative evidence.",
    ]
    improvements = [
        "For structured100 teacher answers, prefer one concrete claim about the root visual pattern and keep secondary channel mentions to evidence that is directly visible and label-supported.",
        "Where possible, align question framing more directly with the intended gold target. If the gold is a discrete choice or binary judgment, avoid language that sounds like tentative triage unless that nuance is intentional.",
        "For future legacy-style review sets, preserve the simplicity of QA_datasets and MCQ_datasets but borrow structured100's stronger distractor design and channel-specific wording.",
    ]

    issue_cards = []
    for row in teacher_issue_rows[:8]:
        issue_cards.append(
            f"""<section class="card">
  <div class="card-header">
    <div class="badges">
      <span class="badge">{esc(row.get('sample_id'))}</span>
      <span class="badge {'bad' if str(row.get('verdict')) == 'misaligned' else 'warn'}">{esc(row.get('verdict'))}</span>
      <span class="badge">{esc(row.get('severity'))}</span>
      {''.join(f'<span class="badge warn">{esc(item)}</span>' for item in (row.get('mismatch_types') or []))}
    </div>
  </div>
  <div class="content">
    <div class="answer-box question">
      <h3>Question</h3>
      <div class="answer-text">{esc(row.get('question'))}</div>
    </div>
    <div class="answer-box">
      <h3>Generated Teacher Answer</h3>
      <div class="answer-text">{esc(row.get('generated_answer'))}</div>
    </div>
    <div class="answer-box">
      <h3>Audit Rationale</h3>
      <div class="answer-text">{esc(row.get('rationale'))}</div>
    </div>
  </div>
</section>"""
        )

    comparison_rows = []
    for name, stats in legacy_stats.items():
        comparison_rows.append(
            f"<tr><td>{esc(name)}</td><td>{stats['rows']}</td><td>{stats['anomaly_ratio']:.1%}</td><td>{stats['avg_question_chars']:.1f}</td><td>{stats['avg_answer_chars']:.1f}</td><td>{esc(json.dumps(stats['question_type_counts'], ensure_ascii=False))}</td></tr>"
        )
    comparison_rows.append(
        f"<tr><td>teacher_structured100</td><td>{teacher_stats['rows']}</td><td>{teacher_stats['anomaly_ratio']:.1%}</td><td>{teacher_stats['avg_question_chars']:.1f}</td><td>{teacher_stats['avg_answer_chars']:.1f}</td><td>{esc(json.dumps(teacher_stats['question_type_counts'], ensure_ascii=False))}</td></tr>"
    )

    legacy_findings_html = "".join(
        f"""<section class="card">
  <div class="card-header"><h2>{esc(name)}</h2></div>
  <div class="content"><ul>{''.join(f'<li>{esc(item)}</li>' for item in dataset_specific_findings(name, rows))}</ul></div>
</section>"""
        for name, rows in legacy_rows_by_dataset.items()
    )

    body = f"""<header>
  <h1>Legacy QA vs structured100 Quality Report</h1>
  <div class="toolbar subtle">
    legacy rows: {total_legacy_rows} |
    structured100 rows: {teacher_stats['rows']} |
    structured100 reviewed: {audit_summary.get('audit_count')} |
    needs repair: {audit_summary.get('needs_repair_count')}
  </div>
</header>
<main>
  <div class="stats">
    <div class="stat"><div class="label">Legacy rows</div><div class="value">{total_legacy_rows}</div></div>
    <div class="stat"><div class="label">Structured100 rows</div><div class="value">{teacher_stats['rows']}</div></div>
    <div class="stat"><div class="label">Structured100 aligned</div><div class="value">{audit_summary.get('verdict_counts', {}).get('aligned', 0)}</div></div>
    <div class="stat"><div class="label">Structured100 non-aligned</div><div class="value">{len(teacher_issue_rows)}</div></div>
  </div>

  <section class="card">
    <div class="card-header"><h2>Quick Comparison</h2></div>
    <div class="content">
      <table>
        <thead>
          <tr><th>Set</th><th>Rows</th><th>Anomaly ratio</th><th>Avg question chars</th><th>Avg answer chars</th><th>Question types</th></tr>
        </thead>
        <tbody>
          {''.join(comparison_rows)}
        </tbody>
      </table>
    </div>
  </section>

  <div class="grid-2">
    <section class="card">
      <div class="card-header"><h2>What Looks Better In structured100</h2></div>
      <div class="content"><ul>{''.join(f'<li>{esc(item)}</li>' for item in strongest_points)}</ul></div>
    </section>
    <section class="card">
      <div class="card-header"><h2>Where structured100 Still Needs Care</h2></div>
      <div class="content"><ul>{''.join(f'<li>{esc(item)}</li>' for item in risks)}</ul></div>
    </section>
  </div>

  <section class="card">
    <div class="card-header"><h2>Legacy Set Notes</h2></div>
    <div class="content">
      <p class="subtle">Top-10 repeated normalized question signatures cover {repeated_legacy_signatures} legacy rows, which is a decent proxy for strong templating.</p>
    </div>
  </section>
  {legacy_findings_html}

  <section class="card">
    <div class="card-header"><h2>Suggested Improvements</h2></div>
    <div class="content"><ul>{''.join(f'<li>{esc(item)}</li>' for item in improvements)}</ul></div>
  </section>

  <section class="card">
    <div class="card-header"><h2>Structured100 Audit Examples</h2></div>
    <div class="content">
      <p class="subtle">These are the reviewed rows with non-zero severity or a non-aligned verdict.</p>
    </div>
  </section>
  {''.join(issue_cards) if issue_cards else '<section class="card"><div class="content"><p>No non-aligned rows found in audit.</p></div></section>'}
</main>"""

    summary = {
        "legacy_stats": legacy_stats,
        "teacher_stats": teacher_stats,
        "audit_summary": audit_summary,
        "teacher_issue_count": len(teacher_issue_rows),
        "teacher_issue_samples": [
            {
                "sample_id": row.get("sample_id"),
                "verdict": row.get("verdict"),
                "severity": row.get("severity"),
                "mismatch_types": row.get("mismatch_types"),
                "question": row.get("question"),
            }
            for row in teacher_issue_rows
        ],
        "strongest_points": strongest_points,
        "risks": risks,
        "improvements": improvements,
    }
    write_text(out_path, page("Legacy QA vs structured100 Quality Report", body))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export legacy QA review HTML and compare it with structured100 teacher answers.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--teacher-questions", default=str(DEFAULT_TEACHER_QUESTIONS))
    parser.add_argument("--teacher-answers", default=str(DEFAULT_TEACHER_ANSWERS))
    parser.add_argument("--teacher-audit-dir", default=str(DEFAULT_TEACHER_AUDIT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    teacher_questions = Path(args.teacher_questions)
    teacher_answers = Path(args.teacher_answers)
    teacher_audit_dir = Path(args.teacher_audit_dir)

    legacy_rows_by_dataset = {
        name: flatten_legacy_dataset(name, series_dir)
        for name, series_dir in LEGACY_DATASETS.items()
    }
    teacher_rows = collect_teacher_rows(teacher_questions, teacher_answers)
    audit_summary = read_json(teacher_audit_dir / "summary.json")
    audit_rows = read_jsonl(teacher_audit_dir / "audit_results.jsonl")

    entries: List[Dict[str, Any]] = []
    for name, rows in legacy_rows_by_dataset.items():
        dataset_path = output_dir / f"{slugify(name)}.html"
        build_legacy_review_page(name, rows, dataset_path)
        stats = build_stats(rows)
        entries.append(
            {
                "name": name,
                "path": dataset_path,
                "rows": stats["rows"],
                "anomaly_ratio": stats["anomaly_ratio"],
                "avg_question_chars": stats["avg_question_chars"],
                "avg_answer_chars": stats["avg_answer_chars"],
            }
        )

    report_path = output_dir / "quality_report.html"
    summary = build_quality_report(
        legacy_rows_by_dataset,
        teacher_rows,
        audit_summary,
        audit_rows,
        report_path,
    )
    summary_path = output_dir / "quality_report.json"
    write_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2))

    index_path = output_dir / "index.html"
    build_index_page(entries, report_path, index_path)
    print(
        json.dumps(
            {
                "index_html": str(index_path),
                "quality_report_html": str(report_path),
                "quality_report_json": str(summary_path),
                "datasets": {name: len(rows) for name, rows in legacy_rows_by_dataset.items()},
                "teacher_rows": len(teacher_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
