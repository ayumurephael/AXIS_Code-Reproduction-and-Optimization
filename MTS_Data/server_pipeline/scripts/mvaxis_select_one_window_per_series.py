from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.utils import read_jsonl, save_json, write_jsonl


def _choose_best_window(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def key(row: Dict[str, Any]) -> tuple:
        interval = row.get("target_interval") or {}
        start = int(interval.get("start", 0))
        end = int(interval.get("end", 0))
        span = max(0, end - start)
        return (
            1 if bool(row.get("has_anomaly")) else 0,
            float(row.get("variance", 0.0)),
            span,
            -int(row.get("window_index", 0)),
        )

    best = max(rows, key=key)
    return copy.deepcopy(best)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one representative window per source series.")
    parser.add_argument(
        "--windows",
        default="data/swat_axis_v1_windows_rich/windows_320.jsonl",
        help="Input windows JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/swat_axis_v1_windows_rich/windows_160_one_per_series.jsonl",
        help="Output JSONL with one selected window per source series.",
    )
    parser.add_argument(
        "--output-summary",
        default="data/swat_axis_v1_windows_rich/windows_160_one_per_series.summary.json",
        help="Output summary JSON.",
    )
    args = parser.parse_args()

    windows_path = Path(args.windows)
    if not windows_path.is_absolute():
        windows_path = ROOT / windows_path
    output_jsonl = Path(args.output_jsonl)
    if not output_jsonl.is_absolute():
        output_jsonl = ROOT / output_jsonl
    output_summary = Path(args.output_summary)
    if not output_summary.is_absolute():
        output_summary = ROOT / output_summary

    rows = read_jsonl(windows_path)
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row.get("source_index", -1)), []).append(row)

    selected: List[Dict[str, Any]] = []
    anomalous_selected = 0
    anomaly_source_with_positive = 0
    for source_index in sorted(grouped):
        group = grouped[source_index]
        chosen = _choose_best_window(group)
        selected.append(chosen)
        if any(bool(row.get("has_anomaly")) for row in group):
            anomaly_source_with_positive += 1
        if bool(chosen.get("has_anomaly")):
            anomalous_selected += 1

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(selected, output_jsonl)
    summary = {
        "input_windows": str(windows_path),
        "output_windows": str(output_jsonl),
        "input_count": len(rows),
        "num_source_series": len(grouped),
        "selected_count": len(selected),
        "selected_anomalous_windows": anomalous_selected,
        "source_series_with_anomalous_candidate": anomaly_source_with_positive,
        "examples": selected[:10],
    }
    save_json(summary, output_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
