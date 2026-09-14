from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from mvaxis_generate_anomaly_score_images import _load_model, _plot_one
from src.mvaxis.utils import read_jsonl, save_json, write_jsonl


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _attach_artifacts(
    row: Dict[str, Any],
    *,
    raw_path: str | None,
    score_path: str | None,
    aggregation: str,
) -> Dict[str, Any]:
    copied = dict(row)
    artifacts = dict(copied.get("question_generation_artifacts") or {})
    if raw_path:
        artifacts["raw_image_path"] = raw_path
        artifacts.setdefault("image_path", raw_path)
    if score_path:
        artifacts["anomaly_score_image_path"] = score_path
        artifacts["anomaly_score_image_aggregation"] = aggregation
    copied["question_generation_artifacts"] = artifacts
    return copied


def _row_num_channels(row: Dict[str, Any]) -> int:
    series = row.get("series") or {}
    shape = series.get("shape") or []
    if len(shape) >= 2:
        return int(shape[1])
    values = np.asarray(series.get("values") or [], dtype=float)
    if values.ndim == 1 and values.size > 0:
        return 1
    if values.ndim == 2:
        return int(values.shape[1])
    channels = row.get("channels") or []
    if channels:
        return int(len(channels))
    raise ValueError("Unable to infer num_channels for score-visual generation.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-generate visualization assets for a multiaxis dataset. "
            "This prepares raw window images for teacher-answer generation and "
            "anomaly-score images/manifests for student-answer eval/training."
        )
    )
    parser.add_argument("--data", required=True, help="Source dataset JSONL.")
    parser.add_argument("--config", default="configs/torch_fixed60_qa600_timercd.json")
    parser.add_argument("--checkpoint", default=None, help="Required when score images are enabled.")
    parser.add_argument("--output-root", required=True, help="Directory to store generated image assets.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["raw", "score"],
        default=["raw", "score"],
        help="Which image families to generate. Default: raw score",
    )
    parser.add_argument("--raw-dir-name", default="raw_images")
    parser.add_argument("--score-dir-name", default="score_images")
    parser.add_argument(
        "--write-updated-jsonl",
        default=None,
        help="Optional path for a JSONL copy with question_generation_artifacts image paths attached.",
    )
    args = parser.parse_args()

    data_path = _resolve(args.data)
    config_path = _resolve(args.config)
    output_root = _resolve(args.output_root)
    updated_jsonl_path = _resolve(args.write_updated_jsonl)

    assert data_path is not None
    assert config_path is not None
    assert output_root is not None

    rows = read_jsonl(data_path)
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    output_root.mkdir(parents=True, exist_ok=True)

    score_model_cache: Dict[int, Any] = {}
    if "score" in args.modes and not args.checkpoint:
        raise ValueError("--checkpoint is required when generating score images")

    manifests: Dict[str, Dict[str, Any]] = {}
    raw_records = {}
    score_records = {}

    if "raw" in args.modes:
        raw_dir = output_root / str(args.raw_dir_name)
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_manifest_records = []
        for idx, row in enumerate(rows):
            record = _plot_one(
                row,
                index=idx,
                output_dir=raw_dir,
                model=None,
                aggregation=str(args.aggregation),
                mode="raw",
            )
            raw_manifest_records.append(record)
            raw_records[idx] = record
        manifests["raw"] = {
            "records": raw_manifest_records,
            "mode": "raw",
            "aggregation": None,
            "data": str(data_path),
        }
        save_json(manifests["raw"], raw_dir / "manifest.json")

    if "score" in args.modes:
        score_dir = output_root / str(args.score_dir_name)
        score_dir.mkdir(parents=True, exist_ok=True)
        score_manifest_records = []
        for idx, row in enumerate(rows):
            row_num_channels = _row_num_channels(row)
            row_model = score_model_cache.get(row_num_channels)
            if row_model is None:
                row_model = _load_model(
                    config_path,
                    str(args.checkpoint),
                    num_channels_override=row_num_channels,
                    skip_global_hint_head=True,
                )
                row_model.eval()
                score_model_cache[row_num_channels] = row_model
            record = _plot_one(
                row,
                index=idx,
                output_dir=score_dir,
                model=row_model,
                aggregation=str(args.aggregation),
                mode="score",
            )
            score_manifest_records.append(record)
            score_records[idx] = record
        manifests["score"] = {
            "records": score_manifest_records,
            "mode": "score",
            "aggregation": str(args.aggregation),
            "data": str(data_path),
        }
        save_json(manifests["score"], score_dir / "manifest.json")

    if updated_jsonl_path is not None:
        updated_rows = []
        for idx, row in enumerate(rows):
            raw_path = (raw_records.get(idx) or {}).get("path")
            score_path = (score_records.get(idx) or {}).get("path")
            updated_rows.append(
                _attach_artifacts(
                    row,
                    raw_path=raw_path,
                    score_path=score_path,
                    aggregation=str(args.aggregation),
                )
            )
        updated_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(updated_rows, updated_jsonl_path)

    summary = {
        "data": str(data_path),
        "num_rows": len(rows),
        "output_root": str(output_root),
        "modes": list(args.modes),
        "raw_manifest": str(output_root / str(args.raw_dir_name) / "manifest.json") if "raw" in args.modes else None,
        "score_manifest": (
            str(output_root / str(args.score_dir_name) / "manifest.json") if "score" in args.modes else None
        ),
        "updated_jsonl": str(updated_jsonl_path) if updated_jsonl_path else None,
        "aggregation": str(args.aggregation),
    }
    summary_path = output_root / "visual_prep_summary.json"
    save_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
