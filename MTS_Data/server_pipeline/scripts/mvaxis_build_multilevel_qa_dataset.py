from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.question_provider import (
    ANOMALY_RATIO,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    QUESTION_SPECS,
    SAMPLES_PER_SERIES,
    build_qa_samples,
    grouped_question_counts,
)
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


def _choose_source_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = [copy.deepcopy(row) for row in rows]
    rng.shuffle(shuffled)
    if n <= len(shuffled):
        return shuffled[:n]
    selected: List[Dict[str, Any]] = []
    while len(selected) < n:
        batch = [copy.deepcopy(row) for row in shuffled]
        rng.shuffle(batch)
        selected.extend(batch)
    return selected[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale_nocf_schema.json")
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--output-dir", default="outputs/runs/multilevel_qa_v2/data")
    parser.add_argument("--output-config", default="outputs/runs/multilevel_qa_v2/config.json")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--question-provider", choices=["bank", "llm"], default="bank")
    parser.add_argument("--llm-config", default=None)
    parser.add_argument("--samples-per-series", type=int, default=SAMPLES_PER_SERIES)
    parser.add_argument("--min-window-size", type=int, default=MIN_WINDOW_SIZE)
    parser.add_argument("--max-window-size", type=int, default=MAX_WINDOW_SIZE)
    parser.add_argument("--anomaly-ratio", type=float, default=ANOMALY_RATIO)
    parser.add_argument("--no-sample-windows", action="store_true")
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    source_path = Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{args.source_split}.jsonl"
    rows = read_jsonl(source_path)
    per_series = 1 if args.no_sample_windows else max(1, int(args.samples_per_series))
    source_needed = max(1, math.ceil(int(args.num_samples) / per_series))
    selected_rows = _choose_source_rows(rows, source_needed, int(args.seed))
    llm_client = None
    llm_config_summary = None
    if args.question_provider == "llm":
        if not args.llm_config:
            raise ValueError("--llm-config is required for --question-provider llm")
        llm_config = load_json(ROOT / args.llm_config)
        llm_client = create_llm_client(llm_config)
        llm_config_summary = {
            "provider": llm_config.get("provider"),
            "model": llm_config.get("model"),
            "base_url": llm_config.get("base_url"),
        }

    out_rows = build_qa_samples(
        selected_rows,
        seed=int(args.seed),
        question_provider=args.question_provider,
        llm_client=llm_client,
        sample_windows=not bool(args.no_sample_windows),
        samples_per_series=int(args.samples_per_series),
        min_window_size=int(args.min_window_size),
        max_window_size=int(args.max_window_size),
        anomaly_ratio=float(args.anomaly_ratio),
    )[: int(args.num_samples)]

    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    write_jsonl(out_rows, out_dir / "test.jsonl")
    summary = {
        "source_config": args.config,
        "source_split": args.source_split,
        "source_path": str(source_path),
        "num_samples": len(out_rows),
        "source_rows_used": len(selected_rows),
        "question_provider": args.question_provider,
        "llm_config": llm_config_summary,
        "sample_windows": not bool(args.no_sample_windows),
        "samples_per_series": int(args.samples_per_series),
        "min_window_size": int(args.min_window_size),
        "max_window_size": int(args.max_window_size),
        "anomaly_ratio": float(args.anomaly_ratio),
        "fixed_question_specs_size": len(QUESTION_SPECS),
        "question_counts": grouped_question_counts(),
        "output_path": str(out_dir / "test.jsonl"),
        "examples": [
            {
                "sample_id": row.get("sample_id"),
                "interval": row.get("target_interval"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "target_fact": (row.get("target_output") or {}).get("fact_check"),
            }
            for row in out_rows[:5]
        ],
    }
    save_json(summary, out_dir / "question_generation_summary.json")

    derived_config = copy.deepcopy(config)
    derived_config["data"]["output_dir"] = str(Path(args.output_dir).as_posix())
    save_json(derived_config, ROOT / args.output_config)
    print(json.dumps({**summary, "output_config": str(ROOT / args.output_config)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
