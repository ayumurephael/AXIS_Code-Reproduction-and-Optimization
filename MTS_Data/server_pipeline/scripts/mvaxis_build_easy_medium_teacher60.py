from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.question_provider import (
    assign_question,
    fixed_question_specs,
    grouped_question_counts,
    retarget_sample_to_window,
    sample_analysis_windows,
)
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


def _choose_window(sample: Dict[str, Any], *, seed: int, prefer_anomaly: bool) -> Dict[str, Any]:
    windows = sample_analysis_windows(
        sample,
        seed=seed,
        samples_per_series=2,
        min_window_size=15,
        max_window_size=60,
        anomaly_ratio=0.5,
    )
    preferred = [w for w in windows if bool(w.get("has_anomaly")) == bool(prefer_anomaly)]
    return dict(preferred[0] if preferred else windows[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_legacy_axis50_original.json")
    parser.add_argument("--source-split", default="test")
    parser.add_argument("--output-dir", default="outputs/runs/easy_medium_teacher60/data")
    parser.add_argument("--output-config", default="outputs/runs/easy_medium_teacher60/config.json")
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--difficulties", default="simple,medium")
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    source_path = Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{args.source_split}.jsonl"
    source_rows = read_jsonl(source_path)
    if not source_rows:
        raise ValueError(f"No source rows found at {source_path}")

    difficulties = {x.strip() for x in args.difficulties.split(",") if x.strip()}
    specs = [spec for spec in fixed_question_specs() if spec.difficulty in difficulties]
    rng = random.Random(int(args.seed))
    rows_pool = [copy.deepcopy(row) for row in source_rows]
    rng.shuffle(rows_pool)

    out_rows: List[Dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        source = copy.deepcopy(rows_pool[(idx - 1) % len(rows_pool)])
        prefer_anomaly = idx % 2 == 1
        window = _choose_window(source, seed=int(args.seed) + idx * 7919, prefer_anomaly=prefer_anomaly)
        retargeted = retarget_sample_to_window(
            source,
            int(window["start"]),
            int(window["end"]),
            window_index=idx,
            copy_sample=True,
        )
        out_rows.append(
            assign_question(
                retargeted,
                spec,
                index=idx,
                copy_sample=False,
                preserve_source_question=True,
            )
        )

    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    write_jsonl(out_rows, out_dir / "test.jsonl")
    derived_config = copy.deepcopy(config)
    derived_config["data"]["output_dir"] = str(Path(args.output_dir).as_posix())
    save_json(derived_config, ROOT / args.output_config)

    by_difficulty: Dict[str, int] = {}
    by_answer_type: Dict[str, int] = {}
    anomalous = 0
    for row in out_rows:
        by_difficulty[row["question_difficulty"]] = by_difficulty.get(row["question_difficulty"], 0) + 1
        by_answer_type[row["question_answer_type"]] = by_answer_type.get(row["question_answer_type"], 0) + 1
        anomalous += int(bool((row.get("target_output") or {}).get("fact_check", {}).get("is_anomalous")))

    summary = {
        "source_config": args.config,
        "source_path": str(source_path),
        "output_path": str(out_dir / "test.jsonl"),
        "output_config": str(ROOT / args.output_config),
        "num_samples": len(out_rows),
        "requested_difficulties": sorted(difficulties),
                "question_template_counts": grouped_question_counts(),
        "selected_counts_by_difficulty": by_difficulty,
        "selected_counts_by_answer_type": by_answer_type,
        "anomalous_count": anomalous,
        "normal_count": len(out_rows) - anomalous,
        "examples": [
            {
                "sample_id": row.get("sample_id"),
                "interval": row.get("target_interval"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "teacher_reasoning_process": (row.get("target_output") or {}).get("reasoning_process"),
                "teacher_answer": (row.get("target_output") or {}).get("final_answer"),
                "target_fact": (row.get("target_output") or {}).get("fact_check"),
            }
            for row in out_rows[:5]
        ],
    }
    save_json(summary, out_dir / "teacher_generation_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
