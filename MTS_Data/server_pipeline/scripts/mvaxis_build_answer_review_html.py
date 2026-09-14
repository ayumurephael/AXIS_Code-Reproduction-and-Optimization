from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.student_answer_provider import parse_student_answer


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

PROJECT_ROOT = ROOT
REMOTE_PROJECT_PREFIX = "/root/shared-nvme/axis_project/repos/multiaxis/"
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "outputs" / "questiongeneration_final" / "questions_600.jsonl"
DEFAULT_TEACHER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "quetiongeneration_1_teacher_answer"
    / "teacher_gpt55.jsonl"
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_jsonl_if_exists(path: Path | None) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return read_jsonl(path)


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def json_block(value: Any) -> str:
    return esc(json.dumps(value, ensure_ascii=False, indent=2))


def sanitize_answer_for_review(value: Any) -> str:
    """Remove model-generated local image links from the human review view."""

    text = "" if value is None else str(value)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(
        r"(?:/root/shared-nvme|/root/axis_project|D:\\PythonProjects\\codex)[^\s)\]]+",
        "[local image path hidden]",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_parsed_answer_for_review(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    cleaned = dict(value)
    for key in ("answer_text", "final_answer"):
        if key in cleaned:
            cleaned[key] = sanitize_answer_for_review(cleaned.get(key))
    return cleaned


def path_name(value: Any) -> str | None:
    if not value:
        return None
    return Path(str(value)).name


def rel_uri(path_value: Any, html_path: Path) -> str:
    if not path_value:
        return ""
    raw = str(path_value)
    p = Path(raw)
    try:
        if p.is_absolute():
            return Path(os.path.relpath(p, html_path.parent)).as_posix()
        return p.as_posix()
    except ValueError:
        return p.as_uri() if p.exists() else raw.replace("\\", "/")


def map_to_local_path(path_value: Any) -> Path | None:
    if not path_value:
        return None
    raw = str(path_value)
    if raw.startswith(REMOTE_PROJECT_PREFIX):
        return PROJECT_ROOT / raw[len(REMOTE_PROJECT_PREFIX) :]
    return Path(raw)


def candidate_images(row: Dict[str, Any]) -> List[Any]:
    artifacts = row.get("question_generation_artifacts") or {}
    return [
        row.get("score_image_path"),
        row.get("raw_image_path"),
        artifacts.get("image_path"),
    ]


def first_image(row: Dict[str, Any], search_root: Path | None = None) -> str:
    """Choose a browser-openable local image when possible."""

    candidates = [value for value in candidate_images(row) if value]
    for value in candidates:
        local = map_to_local_path(value)
        if local is not None and local.exists():
            return str(local)
    if search_root is not None:
        for value in candidates:
            local = map_to_local_path(value)
            if local is None:
                continue
            filename = local.name
            matches = list(search_root.rglob(filename))
            if matches:
                return str(matches[0])
    for value in candidates:
        local = map_to_local_path(value)
        if local is None:
            continue
        filename = local.name
        fallback_names = [filename]
        if filename.endswith("_score.png"):
            fallback_names.append(filename.replace("_score.png", ".png"))
        for name in fallback_names:
            fallback = PROJECT_ROOT / "outputs" / "questiongeneration_final" / "images" / name
            if fallback.exists():
                return str(fallback)
    return str(candidates[0]) if candidates else ""


def fact(row: Dict[str, Any]) -> Dict[str, Any]:
    return dict((row.get("target_output") or {}).get("fact_check") or {})


def true_fact(question_row: Dict[str, Any] | None, answer_row: Dict[str, Any]) -> Dict[str, Any]:
    source = question_row if question_row else answer_row
    return dict((source.get("target_output") or {}).get("fact_check") or {})


def groundtruth_answer(
    question_row: Dict[str, Any] | None,
    teacher_row: Dict[str, Any] | None,
    answer_row: Dict[str, Any],
) -> str:
    if teacher_row:
        teacher = teacher_row.get("teacher_answer_llm") or {}
        if teacher.get("answer"):
            return str(teacher.get("answer"))
    target = (question_row or answer_row).get("target_output") or {}
    return str(
        target.get("final_answer")
        or target.get("question_answer")
        or target.get("reasoning_summary")
        or ""
    )


def teacher_metadata(teacher_row: Dict[str, Any] | None) -> Dict[str, Any]:
    if not teacher_row:
        return {}
    teacher = teacher_row.get("teacher_answer_llm") or {}
    raw = teacher.get("raw") or {}
    usage = raw.get("usage") or {}
    return {
        "provider": teacher.get("provider"),
        "model": teacher.get("model"),
        "latency_seconds": teacher.get("latency_seconds"),
        "usage": usage,
    }


def stale_groundtruth_answer(row: Dict[str, Any]) -> str:
    target = row.get("target_output") or {}
    return str(
        target.get("final_answer")
        or target.get("question_answer")
        or target.get("reasoning_summary")
        or ""
    )


def student_label(row: Dict[str, Any]) -> Any:
    parsed = recompute_student_parsed(row)
    return parsed.get("answer_label")


def student_final(row: Dict[str, Any]) -> str:
    parsed = recompute_student_parsed(row)
    return sanitize_answer_for_review(parsed.get("final_answer") or row.get("raw_response") or "")


def recompute_student_parsed(
    row: Dict[str, Any],
    question_row: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    source = question_row or row
    return parse_student_answer(
        str(row.get("raw_response") or ""),
        question_type=str(source.get("question_type") or ""),
        question=str(source.get("question") or ""),
    )


def row_by_index(rows: List[Dict[str, Any]], index: Any) -> Dict[str, Any] | None:
    try:
        idx = int(index)
    except (TypeError, ValueError):
        return None
    if 0 <= idx < len(rows):
        return rows[idx]
    return None


def build_teacher_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id:
            lookup[str(sample_id)] = row
    return lookup


def find_teacher_row(
    answer_row: Dict[str, Any],
    question_row: Dict[str, Any] | None,
    teacher_rows: List[Dict[str, Any]],
    teacher_by_sample: Dict[str, Dict[str, Any]],
) -> Dict[str, Any] | None:
    if question_row and question_row.get("sample_id"):
        hit = teacher_by_sample.get(str(question_row.get("sample_id")))
        if hit:
            return hit
    return row_by_index(teacher_rows, answer_row.get("index"))


def badge(text: Any, kind: str = "") -> str:
    css = f"badge {kind}".strip()
    return f'<span class="{css}">{esc(text)}</span>'


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(title)}</title>
  <style>
    :root {{
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d7dde8;
      --accent: #0f766e;
      --bad: #b42318;
      --good: #0f766e;
      --warn: #b45309;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font: 14px/1.55 "Segoe UI", Arial, sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(245, 247, 250, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      backdrop-filter: blur(8px);
    }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 18px 24px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; letter-spacing: 0; }}
    h2 {{ margin: 22px 0 10px; font-size: 17px; }}
    h3 {{ margin: 0 0 8px; font-size: 14px; }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ border-collapse: collapse; width: 100%; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    th {{ background: #eef2f6; font-weight: 650; }}
    .subtle {{ color: var(--muted); }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
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
    .question {{ font-size: 16px; font-weight: 650; white-space: pre-wrap; margin-bottom: 12px; }}
    .box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin: 10px 0;
      background: #fcfcfd;
    }}
    .box.truth {{ border-left: 4px solid var(--accent); }}
    .box.student {{ border-left: 4px solid #475467; }}
    .box.raw {{ border-left: 4px solid #7c3aed; }}
    .text {{ white-space: pre-wrap; }}
    img {{
      width: 100%;
      max-height: 780px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    details {{ border-top: 1px dashed var(--line); margin-top: 10px; padding-top: 8px; }}
    summary {{ cursor: pointer; color: var(--muted); }}
    pre {{
      margin: 8px 0 0;
      padding: 10px;
      overflow: auto;
      max-height: 460px;
      border-radius: 8px;
      background: #101828;
      color: #f2f4f7;
      font-size: 12px;
      line-height: 1.45;
    }}
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


def write_group_html(
    group_dir: Path,
    group_name: str,
    question_rows: List[Dict[str, Any]],
    teacher_rows: List[Dict[str, Any]],
    teacher_by_sample: Dict[str, Dict[str, Any]],
) -> Path:
    answers_path = group_dir / "qwen_raw_answers.jsonl"
    out_path = group_dir / "answer_review.html"
    rows = read_jsonl(answers_path)
    search_root = group_dir.parents[2] if len(group_dir.parents) >= 3 else group_dir.parent
    cards: List[str] = []
    for ordinal, row in enumerate(rows, start=1):
        question_row = row_by_index(question_rows, row.get("index"))
        teacher_row = find_teacher_row(row, question_row, teacher_rows, teacher_by_sample)
        f = true_fact(question_row, row)
        stale_f = fact(row)
        parsed = recompute_student_parsed(row, question_row)
        stored_parsed = row.get("parsed_student_answer") or row.get("parsed_response") or {}
        interval = row.get("prompt_interval") or row.get("target_interval")
        image_path = first_image(row, search_root=search_root)
        image_uri = rel_uri(image_path, out_path)
        answer_label = student_label(row)
        gold_choice = f.get("choice_answer")
        has_anomaly = f.get("is_anomalous")
        question_answer_type = (
            (question_row or {}).get("question_answer_type")
            or parsed.get("question_answer_type")
            or f.get("question_answer_type")
        )
        question_difficulty = (question_row or {}).get("question_difficulty") or f.get("question_difficulty")
        badges = [
            badge(f"#{ordinal:04d}"),
            badge(f"idx={row.get('index')}"),
            badge((question_row or row).get("question_type")),
            badge(f"type={question_answer_type}"),
            badge(f"difficulty={question_difficulty}"),
            badge(f"interval={interval}"),
            badge(f"truth_anomaly={has_anomaly}", "bad" if has_anomaly else "good"),
            badge(f"student_label={answer_label}", "warn" if answer_label is None else ""),
        ]
        if gold_choice is not None:
            badges.append(badge(f"truth_choice={gold_choice}"))
        image_side = f'<aside><img src="{esc(image_uri)}" alt="time-series image"></aside>' if image_uri else ""
        content_class = "content" if image_uri else "content no-image"
        cards.append(
            f"""<section class="card" id="case-{ordinal:04d}">
  <div class="card-header">
    <div class="meta">{''.join(badges)}</div>
    <div class="meta">{f'<a href="{esc(image_uri)}">image</a>' if image_uri else ''}</div>
  </div>
  <div class="{content_class}">
    <div>
      <div class="question">{esc((question_row or row).get("question"))}</div>
      <div class="box truth">
        <h3>Ground Truth Fact</h3>
        <div class="text">is_anomalous: {esc(f.get("is_anomalous"))}
root: {esc(f.get("root_cause_channel") or f.get("root_cause_channels"))}
affected: {esc(f.get("affected_channels"))}
type: {esc(f.get("anomaly_type"))}
scope: {esc(f.get("anomaly_scope"))}
choice_answer: {esc(f.get("choice_answer"))}</div>
      </div>
      <div class="box truth">
        <h3>Teacher / Ground Truth Answer</h3>
        <div class="text">{esc(groundtruth_answer(question_row, teacher_row, row))}</div>
      </div>
      <div class="box raw">
        <h3>Student Raw Response / Analysis</h3>
        <div class="text">{esc(sanitize_answer_for_review(row.get("raw_response")))}</div>
      </div>
      <div class="box student">
        <h3>Student Parsed Result</h3>
        <div class="text">answer_label: {esc(parsed.get("answer_label"))}
answer_type: {esc(parsed.get("question_answer_type"))}
final_answer:
{esc(student_final(row))}</div>
      </div>
      <details>
        <summary>Metadata / Stored Answer Row</summary>
        <pre>{json_block({
            "sample_id": row.get("sample_id"),
            "question_sample_id": (question_row or {}).get("sample_id"),
            "teacher_metadata": teacher_metadata(teacher_row),
            "true_fact_from_questions": f,
            "stale_fact_from_answer_row": stale_f,
            "stale_target_answer_from_answer_row": stale_groundtruth_answer(row),
            "recomputed_student_parsed": sanitize_parsed_answer_for_review(parsed),
            "stored_student_parsed": sanitize_parsed_answer_for_review(stored_parsed),
            "prompt_interval": row.get("prompt_interval"),
            "proposal": row.get("proposal"),
            "anomaly_score_stats": row.get("anomaly_score_stats"),
            "axis_embedding_hints_used": row.get("axis_embedding_hints_used"),
            "global_embedding_shape": row.get("global_embedding_shape"),
            "channel_embedding_shape": row.get("channel_embedding_shape"),
            "score_image_file": path_name(row.get("score_image_path")),
            "raw_image_file": path_name(row.get("raw_image_path")),
        })}</pre>
      </details>
    </div>
    {image_side}
  </div>
</section>"""
        )
    body = (
        "<header>"
        f"<h1>{esc(group_name)} Answer Review</h1>"
        f'<div class="subtle">source: {esc(answers_path)} | rows: {len(rows)}</div>'
        "</header><main>"
        + "\n".join(cards)
        + "</main>"
    )
    out_path.write_text(page(f"{group_name} Answer Review", body), encoding="utf-8")
    return out_path


def load_metrics(root: Path) -> List[Dict[str, str]]:
    tsv = root / "metrics_table.tsv"
    if not tsv.exists():
        return []
    lines = tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        rows.append({header[i]: cells[i] if i < len(cells) else "" for i in range(len(header))})
    return rows


def load_metrics_with_open_decision(root: Path) -> List[Dict[str, str]]:
    tsv = root / "metrics_table_with_open_decision.tsv"
    if not tsv.exists():
        return []
    lines = tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        rows.append({header[i]: cells[i] if i < len(cells) else "" for i in range(len(header))})
    return rows


def load_metrics_by_answer_type(root: Path) -> List[Dict[str, str]]:
    tsv = root / "metrics_by_answer_type_with_open.tsv"
    if not tsv.exists():
        return []
    lines = tsv.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        rows.append({header[i]: cells[i] if i < len(cells) else "" for i in range(len(header))})
    return rows


def _build_metric_table(
    title: str,
    rows: List[Dict[str, str]],
    out_path: Path,
    group_pages: Dict[str, Path],
) -> str:
    if not rows:
        return ""
    headers = list(rows[0].keys())

    def cell(row: Dict[str, str], header: str) -> str:
        value = row.get(header, "")
        if header == "run" and value in group_pages:
            uri = rel_uri(group_pages[value], out_path)
            return f'<td><a href="{esc(uri)}">{esc(value)}</a></td>'
        return f"<td>{esc(value)}</td>"

    return (
        f"<h2>{esc(title)}</h2><table><thead><tr>"
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(cell(row, h) for h in headers) + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def write_index(root: Path, group_pages: Dict[str, Path]) -> Path:
    out_path = root / "answer_review_index.html"
    metric_rows = load_metrics(root)
    metric_rows_with_open = load_metrics_with_open_decision(root)
    metric_rows_by_type = load_metrics_by_answer_type(root)
    metric_table = _build_metric_table("Overall Metrics", metric_rows, out_path, group_pages)
    metric_table_with_open = _build_metric_table(
        "Overall Metrics (choice + judgment + open decision)",
        metric_rows_with_open,
        out_path,
        group_pages,
    )
    metric_table_by_type = _build_metric_table(
        "By Answer Type (choice / judgment / open decision)",
        metric_rows_by_type,
        out_path,
        group_pages,
    )
    links = [
        f'<li><a href="{esc(rel_uri(path, out_path))}">{esc(group)}</a></li>'
        for group, path in group_pages.items()
    ]
    body = (
        "<header>"
        "<h1>Run Answer Review</h1>"
        f'<div class="subtle">root: {esc(root)}</div>'
        "</header><main>"
        + metric_table
        + metric_table_with_open
        + metric_table_by_type
        + "<h2>Groups</h2><ul>"
        + "\n".join(links)
        + "</ul>"
        + "</main>"
    )
    out_path.write_text(page("Eight-Group Answer Review", body), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--groups", nargs="*", default=GROUPS)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS_PATH))
    parser.add_argument("--teacher", default=str(DEFAULT_TEACHER_PATH))
    args = parser.parse_args()

    root = Path(args.root)
    question_rows = read_jsonl_if_exists(Path(args.questions))
    teacher_rows = read_jsonl_if_exists(Path(args.teacher))
    teacher_by_sample = build_teacher_lookup(teacher_rows)
    group_pages: Dict[str, Path] = {}
    for group in args.groups:
        group_dir = root / group
        answers_path = group_dir / "qwen_raw_answers.jsonl"
        if not answers_path.exists():
            continue
        group_pages[group] = write_group_html(
            group_dir,
            group,
            question_rows,
            teacher_rows,
            teacher_by_sample,
        )
    index_path = write_index(root, group_pages)
    print(json.dumps({"index": str(index_path), "groups": {k: str(v) for k, v in group_pages.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
