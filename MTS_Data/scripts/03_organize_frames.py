from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.frame_builder import QUESTION_FRAMES, build_question_frames
from src.mvaxis.utils import read_jsonl, save_json, write_jsonl


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild selected windows and expand each physical window into explicit question frames."
    )
    parser.add_argument("--windows", required=True, help="Window metadata JSONL from stage 02.")
    parser.add_argument("--raw-series", required=True, help="Raw-series JSONL from stage 01.")
    parser.add_argument("--output", default="outputs/frames/frames.jsonl")
    parser.add_argument("--num-windows", type=int, default=None, help="Default: use all available windows.")
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--source-start-index", type=int, default=0)
    parser.add_argument("--source-end-index", type=int, default=None)
    parser.add_argument(
        "--frames",
        nargs="+",
        choices=list(QUESTION_FRAMES),
        default=list(QUESTION_FRAMES),
        help="Question perspectives created for every selected physical window.",
    )
    args = parser.parse_args()

    windows_path = _resolve(args.windows)
    raw_series_path = _resolve(args.raw_series)
    output_path = _resolve(args.output)
    frame_rows, selected_windows = build_question_frames(
        read_jsonl(windows_path),
        raw_series_path,
        num_windows=args.num_windows,
        seed=int(args.seed),
        source_start_index=int(args.source_start_index),
        source_end_index=args.source_end_index,
        frames=args.frames,
    )
    write_jsonl(frame_rows, output_path)

    frame_counts = Counter(str(row.get("question_frame_override")) for row in frame_rows)
    truth_counts = Counter(
        "anomalous" if bool(((row.get("target_output") or {}).get("fact_check") or {}).get("is_anomalous")) else "normal"
        for row in frame_rows
    )
    summary = {
        "windows_source": str(windows_path),
        "raw_series_source": str(raw_series_path),
        "output": str(output_path),
        "selected_physical_windows": len(selected_windows),
        "generated_frame_rows": len(frame_rows),
        "frames_per_window": list(args.frames),
        "question_frame_counts": dict(frame_counts),
        "underlying_truth_counts": dict(truth_counts),
        "seed": int(args.seed),
    }
    summary_path = output_path.with_suffix(".summary.json")
    save_json(summary, summary_path)
    print(json.dumps({**summary, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
