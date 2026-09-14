from __future__ import annotations

import argparse
import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from _bootstrap import add_project_root

ROOT = add_project_root()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _rel(path: str | Path, html_path: Path) -> str:
    return Path(os.path.relpath(Path(path), html_path.parent)).as_posix()


def _label_summary(row: Dict[str, Any]) -> str:
    label = row.get("synthetic_label") or {}
    affected = ", ".join(label.get("affected_channels") or []) or "-"
    roots = ", ".join(label.get("root_cause_channels") or []) or "-"
    scope = label.get("anomaly_scope") or "-"
    atype = label.get("anomaly_type") or "normal"
    return f"type={atype} | scope={scope} | root={roots} | affected={affected}"


def _window_summary(row: Dict[str, Any]) -> str:
    affected = ", ".join(row.get("affected_channels") or []) or "-"
    root = row.get("root_cause_channel") or "-"
    scope = row.get("anomaly_scope") or "-"
    label = "anomalous" if row.get("has_anomaly") else "normal"
    interval = row.get("target_interval") or {}
    return f"{label} | interval=[{interval.get('start')}, {interval.get('end')}) | root={root} | affected={affected} | scope={scope}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local HTML review page for SWaT Axis series and highlighted windows.")
    parser.add_argument("--data-dir", default="data/swat_axis_v1")
    parser.add_argument("--windows-dir", default="data/swat_axis_v1_windows")
    parser.add_argument("--output", default="data/swat_axis_v1_windows/review.html")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    windows_dir = ROOT / args.windows_dir
    output_path = ROOT / args.output

    summary = _read_json(data_dir / "dataset_summary.json")
    rows = list(_iter_jsonl(data_dir / "raw_series.jsonl"))
    windows_rows = list(_iter_jsonl(next(windows_dir.glob("windows_*.jsonl"))))

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in windows_rows:
        by_source[str(row.get("source_sample_id"))].append(row)

    parts: List[str] = []
    parts.append(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SWaT Axis Review</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f7f7f9; color: #111; }
    h1, h2, h3 { margin: 0 0 10px; }
    .top { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 14px; }
    .meta { font-size: 13px; line-height: 1.45; color: #333; margin-bottom: 10px; }
    .mono { font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    .window { border-top: 1px solid #eee; padding-top: 10px; margin-top: 10px; }
    img { width: 100%; border: 1px solid #ddd; border-radius: 6px; background: #fff; }
    .pill { display: inline-block; border: 1px solid #ccc; border-radius: 999px; padding: 2px 8px; margin-right: 6px; font-size: 12px; }
  </style>
</head>
<body>"""
    )
    parts.append('<div class="top">')
    parts.append(f"<h1>SWaT Axis v1 Review</h1>")
    parts.append(
        f"<div class='meta'>series={summary['num_total_series']} | anomaly={summary['num_anomaly_series']} | "
        f"normal={summary['num_normal_series']} | features={summary['num_features']} | "
        f"seq_len={summary['sequence_length']} | downsample={summary['downsample_factor']}</div>"
    )
    parts.append(
        f"<div class='meta'>data_dir: <span class='mono'>{_esc(data_dir)}</span><br>"
        f"windows_dir: <span class='mono'>{_esc(windows_dir)}</span></div>"
    )
    parts.append("</div>")
    parts.append('<div class="grid">')

    for row in rows:
        sample_id = str(row.get("sample_id"))
        src = row.get("source") or {}
        gt = _label_summary(row)
        windows = sorted(by_source.get(sample_id, []), key=lambda item: int((item.get("target_interval") or {}).get("start", 0)))
        parts.append('<div class="card">')
        parts.append(f"<h2>{_esc(sample_id)}</h2>")
        parts.append(
            "<div class='meta'>"
            f"<span class='pill'>{_esc(src.get('split'))}</span>"
            f"<span class='pill'>{_esc(src.get('source_file'))}</span>"
            f"<span class='pill'>{_esc(src.get('event_role') or src.get('normal_source') or '-')}</span>"
            "</div>"
        )
        parts.append(
            f"<div class='meta'><strong>Ground truth:</strong> {_esc(gt)}<br>"
            f"<strong>Time:</strong> {_esc((row.get('original_data') or {}).get('timestamp_start'))} -> {_esc((row.get('original_data') or {}).get('timestamp_end'))}<br>"
            f"<strong>Attack points:</strong> {_esc(', '.join((row.get('original_data') or {}).get('attack_points') or [] ) or '-')}</div>"
        )
        for win in windows:
            image_path = win.get("image_path")
            parts.append('<div class="window">')
            parts.append(f"<div class='meta'><strong>Window:</strong> {_esc(_window_summary(win))}</div>")
            if image_path and Path(image_path).exists():
                parts.append(f"<img src='{_esc(_rel(image_path, output_path))}' alt='{_esc(sample_id)}'>")
            else:
                parts.append(f"<div class='meta'>missing image: <span class='mono'>{_esc(image_path)}</span></div>")
            parts.append("</div>")
        parts.append("</div>")

    parts.append("</div></body></html>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(json.dumps({"output_html": str(output_path), "series": len(rows), "windows": len(windows_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
