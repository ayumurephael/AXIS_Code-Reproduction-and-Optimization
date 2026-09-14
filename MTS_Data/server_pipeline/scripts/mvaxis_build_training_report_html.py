from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_lines(log_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _svg_polyline(values: List[float], *, width: int = 520, height: int = 180, color: str = "#2563eb") -> str:
    if not values:
        return '<svg width="520" height="180"></svg>'
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        max_v = min_v + 1.0
    pts: List[str] = []
    for idx, value in enumerate(values):
        x = 12 + idx * (width - 24) / max(len(values) - 1, 1)
        y = 12 + (height - 24) * (1.0 - ((value - min_v) / (max_v - min_v)))
        pts.append(f"{x:.1f},{y:.1f}")
    grid = "".join(
        f'<line x1="12" y1="{12 + i * (height - 24) / 4:.1f}" x2="{width - 12}" y2="{12 + i * (height - 24) / 4:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        for i in range(5)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#d1d5db"/>'
        f"{grid}"
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(pts)}"/>'
        "</svg>"
    )


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_html(report: Dict[str, Any], log_rows: List[Dict[str, Any]]) -> str:
    epoch_progress = [row for row in log_rows if row.get("stage") == "epoch_progress"]
    epoch_summary = [row for row in log_rows if row.get("stage") == "epoch_summary"]
    epoch_summary = sorted(epoch_summary, key=lambda row: int(row.get("epoch", 0)))

    total_curve = [float(row["mean_total_loss"]) for row in epoch_summary]
    ntp_curve = [float(row["mean_ntp_loss"]) for row in epoch_summary]
    answer_curve = [float(row["mean_answer_loss"]) for row in epoch_summary]

    progress_tail = epoch_progress[-30:]
    tail_rows = "\n".join(
        "<tr>"
        f"<td>{row.get('epoch')}</td>"
        f"<td>{row.get('progress_percent')}%</td>"
        f"<td>{_fmt(row.get('total_loss'))}</td>"
        f"<td>{_fmt(row.get('ntp_loss'))}</td>"
        f"<td>{_fmt(row.get('answer_loss'))}</td>"
        f"<td>{row.get('answer_type')}</td>"
        f"<td>{row.get('answer_label')}</td>"
        f"<td>{row.get('answer_pred')}</td>"
        "</tr>"
        for row in progress_tail
    )

    epoch_rows = "\n".join(
        "<tr>"
        f"<td>{row.get('epoch')}</td>"
        f"<td>{_fmt(row.get('mean_total_loss'))}</td>"
        f"<td>{_fmt(row.get('mean_ntp_loss'))}</td>"
        f"<td>{_fmt(row.get('mean_answer_loss'))}</td>"
        f"<td>{row.get('answer_supervised_items')}</td>"
        f"<td>{row.get('global_steps')}</td>"
        "</tr>"
        for row in epoch_summary
    )

    history_tail = report.get("history_tail") or []
    history_rows = "\n".join(
        "<tr>"
        f"<td>{row.get('epoch')}</td>"
        f"<td>{row.get('sample_id')}</td>"
        f"<td>{_fmt(row.get('total_loss'))}</td>"
        f"<td>{_fmt(row.get('ntp_loss'))}</td>"
        f"<td>{_fmt(row.get('answer_loss'))}</td>"
        f"<td>{row.get('answer_type')}</td>"
        f"<td>{row.get('answer_label')}</td>"
        f"<td>{row.get('answer_pred')}</td>"
        "</tr>"
        for row in history_tail
    )

    meta_items = {
        "run_note": report.get("run_note"),
        "num_samples": report.get("num_samples"),
        "epochs": report.get("epochs"),
        "global_steps": report.get("global_steps"),
        "world_size": report.get("world_size"),
        "mean_total_loss": _fmt(report.get("mean_total_loss")),
        "mean_ntp_loss": _fmt(report.get("mean_ntp_loss")),
        "mean_answer_loss": _fmt(report.get("mean_answer_loss")),
        "elapsed_seconds": _fmt(report.get("elapsed_seconds")),
        "checkpoint_path": report.get("checkpoint_path"),
        "encoder_checkpoint_path": report.get("encoder_checkpoint_path"),
    }
    meta_html = "\n".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>" for key, value in meta_items.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Training Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
    h1, h2 {{ margin: 0 0 12px; }}
    h2 {{ margin-top: 28px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; font-size: 14px; vertical-align: top; }}
    th {{ background: #f3f4f6; text-align: left; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(280px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; background: #fafafa; }}
    .mono {{ font-family: Consolas, monospace; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Question 1000 Training Report</h1>
  <p>This page summarizes the completed next-token training run and its logged loss trajectory.</p>

  <h2>Run Summary</h2>
  <table>{meta_html}</table>

  <h2>Loss Curves</h2>
  <div class="grid">
    <div class="card"><h3>Total Loss</h3>{_svg_polyline(total_curve, color="#2563eb")}</div>
    <div class="card"><h3>NTP Loss</h3>{_svg_polyline(ntp_curve, color="#059669")}</div>
    <div class="card"><h3>Answer Loss</h3>{_svg_polyline(answer_curve, color="#dc2626")}</div>
  </div>

  <h2>Epoch Summary</h2>
  <table>
    <tr><th>epoch</th><th>mean_total_loss</th><th>mean_ntp_loss</th><th>mean_answer_loss</th><th>answer_supervised_items</th><th>global_steps</th></tr>
    {epoch_rows}
  </table>

  <h2>Recent Progress Samples</h2>
  <table>
    <tr><th>epoch</th><th>progress</th><th>total_loss</th><th>ntp_loss</th><th>answer_loss</th><th>answer_type</th><th>answer_label</th><th>answer_pred</th></tr>
    {tail_rows}
  </table>

  <h2>History Tail From Report</h2>
  <table>
    <tr><th>epoch</th><th>sample_id</th><th>total_loss</th><th>ntp_loss</th><th>answer_loss</th><th>answer_type</th><th>answer_label</th><th>answer_pred</th></tr>
    {history_rows}
  </table>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a simple HTML training report from report.json and worker log.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--worker-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report_path = Path(args.report)
    log_path = Path(args.worker_log)
    output_path = Path(args.output)

    report = _load_json(report_path)
    log_rows = _load_json_lines(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html(report, log_rows), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()
