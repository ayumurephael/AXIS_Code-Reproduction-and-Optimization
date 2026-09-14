from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .utils import read_jsonl


def _escape(value: Any) -> str:
    """Escape arbitrary values for HTML text nodes."""

    return html.escape("" if value is None else str(value), quote=True)


def _fmt_list(values: Any) -> str:
    """Format scalar/list values into a compact readable string."""

    if values is None:
        return "none"
    if isinstance(values, str):
        return values or "none"
    if isinstance(values, Iterable) and not isinstance(values, (dict, bytes)):
        items = [str(x) for x in values if x is not None]
        return ", ".join(items) if items else "none"
    return str(values)


def _json_block(value: Any) -> str:
    """Render a small JSON block for diagnostic metadata."""

    return _escape(json.dumps(value, ensure_ascii=False, indent=2))


def _rel_uri(path: Any, html_path: Path) -> str:
    """Return a browser-friendly URI for a local artifact path."""

    if not path:
        return ""
    raw = str(path)
    p = Path(raw)
    try:
        if p.is_absolute():
            return Path(os.path.relpath(p, html_path.parent)).as_posix()
        return p.as_posix()
    except ValueError:
        return p.as_uri() if p.exists() else raw.replace("\\", "/")


def _first_window(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the first QA window payload if present."""

    windows = row.get("windows") or []
    if windows and isinstance(windows[0], dict):
        return windows[0]
    return {}


def _question(row: Dict[str, Any]) -> str:
    """Return the best available question text from a question/answer row."""

    window = _first_window(row)
    return str(row.get("question") or window.get("question") or "")


def _reference_answer(row: Dict[str, Any]) -> str:
    """Return the best available teacher/reference answer."""

    window = _first_window(row)
    target = row.get("target_output") or {}
    return str(
        row.get("windows_0_answer")
        or window.get("answer")
        or target.get("final_answer")
        or target.get("question_answer")
        or ""
    )


def _model_answer(row: Dict[str, Any]) -> str:
    """Return the best available generated answer from teacher/student rows."""

    teacher = row.get("teacher_answer_llm") or {}
    parsed = row.get("parsed_student_answer") or row.get("parsed_response") or {}
    return str(
        teacher.get("answer")
        or row.get("model_answer")
        or parsed.get("final_answer")
        or row.get("raw_response")
        or ""
    )


def _answer_join_key(row: Dict[str, Any]) -> tuple[str, str]:
    """Build a stable join key for answer-only rows and their full source rows."""

    return (_question(row).strip(), _reference_answer(row).strip())


def _companion_answer_source(path: Path) -> Path | None:
    """Return the likely full JSONL companion for an answer-only JSONL file."""

    name = path.name
    if not name.endswith(".answers.jsonl"):
        return None
    candidate = path.with_name(name.replace(".answers.jsonl", ".jsonl"))
    return candidate if candidate.exists() else None


def _enrich_answer_rows(rows: List[Dict[str, Any]], source_path: Path) -> List[Dict[str, Any]]:
    """Attach image and metadata from a companion full JSONL when available."""

    if not rows:
        return rows
    if any(_image_path(row) for row in rows):
        return rows
    companion = _companion_answer_source(source_path)
    if companion is None:
        return rows
    source_rows = read_jsonl(companion)
    source_by_key: Dict[tuple[str, str], Dict[str, Any]] = {}
    for row in source_rows:
        key = _answer_join_key(row)
        if key not in source_by_key:
            source_by_key[key] = row
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        key = _answer_join_key(row)
        source = source_by_key.get(key)
        if not source:
            enriched.append(row)
            continue
        merged = dict(source)
        merged.update(row)
        # Preserve richer source-side fields that answer-only rows do not carry.
        for field in (
            "sample_id",
            "question_type",
            "target_interval",
            "target_output",
            "windows",
            "question_generation_artifacts",
            "visual_context",
        ):
            if field in source and not merged.get(field):
                merged[field] = source[field]
        enriched.append(merged)
    return enriched


def _image_path(row: Dict[str, Any]) -> str:
    """Return the most relevant image path attached to the row."""

    artifacts = row.get("question_generation_artifacts") or {}
    visual = row.get("visual_context") or {}
    return str(
        artifacts.get("image_path")
        or row.get("raw_image_path")
        or row.get("score_image_path")
        or visual.get("raw_image_path")
        or visual.get("score_image_path")
        or ""
    )


def _series_path(row: Dict[str, Any]) -> str:
    """Return the saved series JSON path when available."""

    artifacts = row.get("question_generation_artifacts") or {}
    return str(artifacts.get("series_path") or "")


def _fact(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return the standard fact-check object when available."""

    return dict((row.get("target_output") or {}).get("fact_check") or {})


def _badge(text: Any, kind: str = "") -> str:
    """Render a compact metadata badge."""

    css = f"badge {kind}".strip()
    return f'<span class="{css}">{_escape(text)}</span>'


def _page(title: str, body: str) -> str:
    """Wrap report body content in a shared standalone HTML page."""

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
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
      --code: #101828;
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
{body}
</body>
</html>
"""


def write_question_generation_html(
    questions_jsonl: str | Path,
    html_path: str | Path | None = None,
    *,
    title: str | None = None,
) -> Path:
    """Write an HTML review page for generated question JSONL rows."""

    source_path = Path(questions_jsonl)
    out_path = Path(html_path) if html_path else source_path.with_suffix(".html")
    rows = read_jsonl(source_path)
    title = title or f"Generated Questions: {source_path.name}"
    cards: List[str] = []
    for idx, row in enumerate(rows, start=1):
        fact = _fact(row)
        has_anomaly = bool(fact.get("is_anomalous") or _first_window(row).get("has_anomaly"))
        interval = row.get("target_interval") or {}
        image_uri = _rel_uri(_image_path(row), out_path)
        series_uri = _rel_uri(_series_path(row), out_path)
        badges = [
            _badge(f"#{idx:04d}"),
            _badge(row.get("sample_id")),
            _badge(row.get("question_answer_type") or row.get("answer_type")),
            _badge(row.get("question_difficulty")),
            _badge("anomalous" if has_anomaly else "normal", "bad" if has_anomaly else "good"),
            _badge(f"{interval.get('start')}:{interval.get('end')}"),
        ]
        links = []
        if series_uri:
            links.append(f'<a href="{_escape(series_uri)}">series json</a>')
        if image_uri:
            links.append(f'<a href="{_escape(image_uri)}">image</a>')
        side = (
            f'<aside><img src="{_escape(image_uri)}" alt="highlighted time series"></aside>'
            if image_uri
            else ""
        )
        content_class = "content" if image_uri else "content no-image"
        cards.append(
            f"""<section class="card">
  <div class="card-header">
    <div class="meta">{''.join(badges)}</div>
    <div class="meta">{' | '.join(links)}</div>
  </div>
  <div class="{content_class}">
    <div>
      <div class="question">{_escape(_question(row))}</div>
      <div class="answer-box reference">
        <h3>Reference Answer</h3>
        <div class="answer-text">{_escape(_reference_answer(row))}</div>
      </div>
      <details>
        <summary>Fact / metadata</summary>
        <pre>{_json_block({
            "question_type": row.get("question_type"),
            "question_id": row.get("question_id"),
            "question_family": row.get("question_family"),
            "target_interval": row.get("target_interval"),
            "fact_check": fact,
            "window": _first_window(row),
        })}</pre>
      </details>
    </div>
    {side}
  </div>
</section>"""
        )
    body = (
        "<header>"
        f"<h1>{_escape(title)}</h1>"
        f'<div class="toolbar subtle">source: {_escape(source_path)} | rows: {len(rows)}</div>'
        "</header><main>"
        + "\n".join(cards)
        + "</main>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_page(title, body), encoding="utf-8")
    return out_path


def write_answer_generation_html(
    answers_jsonl: str | Path,
    html_path: str | Path | None = None,
    *,
    title: str | None = None,
) -> Path:
    """Write an HTML review page for teacher or student answer JSONL rows."""

    source_path = Path(answers_jsonl)
    out_path = Path(html_path) if html_path else source_path.with_suffix(".html")
    rows = _enrich_answer_rows(read_jsonl(source_path), source_path)
    title = title or f"Generated Answers: {source_path.name}"
    cards: List[str] = []
    for idx, row in enumerate(rows, start=1):
        fact = _fact(row)
        image_uri = _rel_uri(_image_path(row), out_path)
        model_answer = _model_answer(row)
        model_box = (
            f"""<div class="answer-box model">
        <h3>Generated Answer</h3>
        <div class="answer-text">{_escape(model_answer)}</div>
      </div>"""
            if model_answer
            else ""
        )
        badges = [
            _badge(f"#{idx:04d}"),
            _badge(row.get("sample_id") or row.get("index")),
            _badge(row.get("question_type")),
            _badge((fact or {}).get("is_anomalous")),
        ]
        side = (
            f'<aside><img src="{_escape(image_uri)}" alt="highlighted time series"></aside>'
            if image_uri
            else ""
        )
        content_class = "content" if image_uri else "content no-image"
        cards.append(
            f"""<section class="card">
  <div class="card-header">
    <div class="meta">{''.join(badges)}</div>
    <div class="meta">{f'<a href="{_escape(image_uri)}">image</a>' if image_uri else ''}</div>
  </div>
  <div class="{content_class}">
    <div>
      <div class="question">{_escape(_question(row))}</div>
      <div class="answer-box reference">
        <h3>Reference / windows[0].answer</h3>
        <div class="answer-text">{_escape(_reference_answer(row))}</div>
      </div>
      {model_box}
      <details>
        <summary>Fact / metadata</summary>
        <pre>{_json_block({
            "fact_check": fact,
            "target_interval": row.get("target_interval"),
            "parsed_student_answer": row.get("parsed_student_answer"),
            "teacher_answer_llm": row.get("teacher_answer_llm"),
            "error": row.get("teacher_answer_error"),
        })}</pre>
      </details>
    </div>
    {side}
  </div>
</section>"""
        )
    body = (
        "<header>"
        f"<h1>{_escape(title)}</h1>"
        f'<div class="toolbar subtle">source: {_escape(source_path)} | rows: {len(rows)}</div>'
        "</header><main>"
        + "\n".join(cards)
        + "</main>"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_page(title, body), encoding="utf-8")
    return out_path
