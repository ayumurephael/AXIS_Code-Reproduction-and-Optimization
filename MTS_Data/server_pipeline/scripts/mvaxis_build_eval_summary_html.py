from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


RUN_ORDER = [
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


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_runs(summary: Dict[str, Any], runs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        item = summary["runs"].get(run)
        if not item:
            continue
        label = item.get("overall", {}).get("label", {})
        rows.append(
            {
                "run": run,
                "label_acc": label.get("accuracy"),
                "label_acc_on_explicit": label.get("accuracy_on_parsed"),
                "explicit_answer_rate": label.get("explicit_answer_rate"),
                "explicit_n": label.get("explicit_n"),
                "label_n": label.get("n"),
            }
        )
    return rows


def answer_type_rows(summary: Dict[str, Any], runs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        item = summary["runs"].get(run)
        if not item:
            continue
        cats = item.get("by_category", {})
        row: Dict[str, Any] = {"run": run}
        for answer_type in ("choice", "judgment"):
            label = (cats.get(f"answer_type={answer_type}") or {}).get("label", {})
            row[f"{answer_type}_acc"] = label.get("accuracy")
            row[f"{answer_type}_parse_rate"] = label.get("explicit_answer_rate")
            row[f"{answer_type}_n"] = label.get("n")
        rows.append(row)
    return rows


def difficulty_rows(summary: Dict[str, Any], runs: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        item = summary["runs"].get(run)
        if not item:
            continue
        cats = item.get("by_category", {})
        row: Dict[str, Any] = {"run": run}
        found = False
        for difficulty in ("anomaly_frame", "normal_frame", "simple", "medium", "complex"):
            label = (cats.get(f"difficulty={difficulty}") or {}).get("label", {})
            if label:
                found = True
            row[f"{difficulty}_acc"] = label.get("accuracy")
            row[f"{difficulty}_n"] = label.get("n")
        if found:
            rows.append(row)
    return rows


def render_table(title: str, columns: List[str], rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    thead = "".join(f"<th>{esc(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                text = fmt(value)
            else:
                text = esc(value)
            cells.append(f"<td>{text}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = "\n".join(body_rows)
    return f"""
    <section>
      <h2>{esc(title)}</h2>
      <table>
        <thead><tr>{thead}</tr></thead>
        <tbody>
          {tbody}
        </tbody>
      </table>
    </section>
    """


def build_page(title: str, summary: Dict[str, Any]) -> str:
    summary_runs = list((summary.get("runs") or {}).keys())
    runs = [run for run in RUN_ORDER if run in summary.get("runs", {})]
    runs.extend(run for run in summary_runs if run not in runs)
    overall = rows_from_runs(summary, runs)
    answer_types = answer_type_rows(summary, runs)
    difficulties = difficulty_rows(summary, runs)
    teacher = summary.get("teacher")
    questions = summary.get("questions")
    run_root = summary.get("run_root")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{esc(title)}</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 24px;
      color: #1f2937;
      background: #f8fafc;
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    .meta {{
      margin: 0 0 20px 0;
      padding: 12px 14px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    section {{
      margin: 20px 0;
      padding: 16px;
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border: 1px solid #d1d5db;
      padding: 8px 10px;
      text-align: left;
    }}
    th {{
      background: #f3f4f6;
    }}
    code {{
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <div class="meta">
    <div><strong>Run root:</strong> <code>{esc(run_root)}</code></div>
    <div><strong>Questions:</strong> <code>{esc(questions)}</code></div>
    <div><strong>Teacher:</strong> <code>{esc(teacher)}</code></div>
  </div>
  {render_table('Overall Metrics', ['run', 'label_acc', 'label_acc_on_explicit', 'explicit_answer_rate', 'explicit_n', 'label_n'], overall)}
  {render_table('By Answer Type', ['run', 'choice_acc', 'choice_parse_rate', 'choice_n', 'judgment_acc', 'judgment_parse_rate', 'judgment_n'], answer_types)}
  {render_table('By Difficulty / Frame', ['run', 'anomaly_frame_acc', 'anomaly_frame_n', 'normal_frame_acc', 'normal_frame_n', 'simple_acc', 'simple_n', 'medium_acc', 'medium_n', 'complex_acc', 'complex_n'], difficulties)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    summary = load_summary(summary_path)
    html_text = build_page(args.title, summary)
    output_path = Path(args.output_html)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
