from __future__ import annotations

import argparse
import copy
from datetime import datetime
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.legacy_adapter import convert_legacy_sample
from src.mvaxis.question_provider import (
    QuestionSpec,
    assign_question,
    fixed_question_specs,
    grouped_question_counts,
    retarget_sample_to_window,
    sample_analysis_windows,
)
from src.mvaxis.utils import load_json, save_json, write_jsonl


def _dated_output_dir(prefix: str, count: int) -> str:
    """Return the standard dated output directory name."""

    return f"outputs/{prefix}_{int(count)}_{datetime.now().strftime('%m%d')}"


def _load_legacy_generator(legacy_src: Path):
    src = legacy_src / "src"
    if not src.exists():
        raise FileNotFoundError(f"Legacy AXIS generator src directory not found: {src}")
    if importlib.util.find_spec("statsmodels") is None:
        statsmodels = types.ModuleType("statsmodels")
        tsa = types.ModuleType("statsmodels.tsa")
        arima_process = types.ModuleType("statsmodels.tsa.arima_process")

        class _MissingArmaProcess:
            def __init__(self, *_args, **_kwargs):
                raise ImportError("statsmodels is required for ARIMA trends; use_attribute_set disables ARIMA.")

        arima_process.ArmaProcess = _MissingArmaProcess
        statsmodels.tsa = tsa
        tsa.arima_process = arima_process
        sys.modules.setdefault("statsmodels", statsmodels)
        sys.modules.setdefault("statsmodels.tsa", tsa)
        sys.modules.setdefault("statsmodels.tsa.arima_process", arima_process)
    if importlib.util.find_spec("pywt") is None:
        pywt = types.ModuleType("pywt")

        class _MissingWavelet:
            def __init__(self, *_args, **_kwargs):
                raise ImportError("pywt is required for wavelet seasonality; use_attribute_set disables wavelets.")

        pywt.Wavelet = _MissingWavelet
        sys.modules.setdefault("pywt", pywt)

    old_cwd = Path.cwd()
    os.chdir(legacy_src)
    sys.path.insert(0, str(src))
    try:
        from generate_dataset import generate_dataset  # type: ignore

        try:
            import config as legacy_config  # type: ignore
            import ts_multi_generator  # type: ignore

            for attr_set in (legacy_config.ALL_ATTRIBUTE_SET, ts_multi_generator.ALL_ATTRIBUTE_SET):
                overall = attr_set.get("overall_attribute", {})
                trend = overall.get("trend", {})
                if isinstance(trend, dict):
                    trend.pop("arima", None)
                seasonal = overall.get("seasonal", {})
                if isinstance(seasonal, dict):
                    seasonal.pop("wavelet periodic fluctuation", None)
                seasonal_anomalies = attr_set.get("seasonal_anomalies", {})
                if isinstance(seasonal_anomalies, dict):
                    seasonal_anomalies.pop("wavelet", None)
        except Exception:
            pass
        return generate_dataset
    finally:
        os.chdir(old_cwd)


def _frame_specs(seed: int) -> List[QuestionSpec]:
    specs = fixed_question_specs()
    specs.sort(key=lambda spec: (spec.difficulty, spec.answer_type, spec.question_id))
    if not specs:
        raise RuntimeError("Expected at least one fixed frame-based question.")
    rng = random.Random(seed)
    specs = list(specs)
    rng.shuffle(specs)
    return specs


def _answer_label(sample: Dict[str, Any]) -> Optional[str]:
    answer_type = sample.get("question_answer_type")
    target = sample.get("target_output") or {}
    fact = target.get("fact_check") or {}
    if answer_type == "choice":
        return fact.get("choice_answer") or "choice_unknown"
    if answer_type == "judgment":
        text = str(target.get("question_answer") or target.get("final_answer") or "").strip().lower()
        if text.startswith("yes"):
            return "yes"
        if text.startswith("no"):
            return "no"
        return "judgment_unknown"
    return None


def _label_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "question_id": sample.get("question_id"),
        "question_difficulty": sample.get("question_difficulty"),
        "question_answer_type": sample.get("question_answer_type"),
        "question_family": sample.get("question_family"),
        "answer_label": _answer_label(sample),
        "is_anomalous": bool(fact.get("is_anomalous", False)),
        "anomaly_type": fact.get("anomaly_type") if fact.get("is_anomalous") else "normal",
        "anomaly_scope": fact.get("anomaly_scope") if fact.get("is_anomalous") else "normal",
        "target_interval": sample.get("target_interval"),
    }


def _choose_window(
    sample: Dict[str, Any],
    *,
    seed: int,
    prefer_anomaly: bool,
    min_window_size: int,
    max_window_size: int,
) -> Dict[str, Any]:
    windows = sample_analysis_windows(
        sample,
        seed=seed,
        samples_per_series=2,
        min_window_size=min_window_size,
        max_window_size=max_window_size,
        anomaly_ratio=0.5,
    )
    preferred = [w for w in windows if bool(w.get("has_anomaly")) == bool(prefer_anomaly)]
    return dict(preferred[0] if preferred else windows[0])


def _write_experiment_config(base_config: Dict[str, Any], args: argparse.Namespace, specs: List[QuestionSpec]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("data", {})
    cfg["data"].update(
        {
            "profile": "legacy_axis_fixed60_qa600",
            "generator_backend": "legacy_tsad",
            "question_provider": "fixed_frame_bank",
            "question_seed": int(args.seed),
            "output_dir": str(Path(args.output_dir, "data").as_posix()),
            "legacy_num_samples": int(args.num_samples),
            "length": int(args.seq_len),
            "num_channels": int(args.num_features),
            "legacy_anomaly_sample_ratio": float(args.anomaly_ratio),
            "legacy_activate_function": bool(args.activate_function),
            "legacy_use_attribute_set": bool(args.use_attribute_set),
            "train_ratio": 1.0,
            "val_ratio": 0.0,
            "test_ratio": 0.0,
            "fixed_question_count": len(specs),
            "qa_min_window_size": int(args.min_window_size),
            "qa_max_window_size": int(args.max_window_size),
        }
    )
    cfg.setdefault("model", {})
    cfg["model"]["feature_dim"] = int(args.num_features)
    cfg["model"]["metadata_dim"] = int(args.num_features)
    cfg.setdefault("train_interval_proposal", {})
    cfg["train_interval_proposal"]["checkpoint_path"] = args.interval_proposer_checkpoint
    cfg.setdefault("outputs", {})
    cfg["outputs"]["report_path"] = str(Path(args.output_dir, "train_report.json").as_posix())
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-src", default=str(ROOT / "legacy" / "TSAD_dataset_gen-axis"))
    parser.add_argument("--base-config", default="configs/torch_legacy_axis50_original.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=600)
    parser.add_argument("--seq-len", type=int, default=180)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    parser.add_argument("--activate-function", action="store_true")
    parser.add_argument("--use-attribute-set", action="store_true", default=True)
    parser.add_argument("--min-window-size", type=int, default=15)
    parser.add_argument("--max-window-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument(
        "--interval-proposer-checkpoint",
        default="external_checkpoints/timercd/best_model/pretrain_checkpoint_best_multi.pth",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = _dated_output_dir("data", int(args.num_samples))

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    out_dir = ROOT / args.output_dir
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    specs = _frame_specs(int(args.seed))
    generate_dataset = _load_legacy_generator(Path(args.legacy_src))
    legacy_rows = generate_dataset(
        num_samples=int(args.num_samples),
        seq_len=int(args.seq_len),
        anomaly_sample_ratio=float(args.anomaly_ratio),
        is_multivariate=True,
        num_features=int(args.num_features),
        activate_function=bool(args.activate_function),
        use_attribute_set=bool(args.use_attribute_set),
    )
    raw_rows = [
        convert_legacy_sample(row, sample_id=f"fixed60_qa600_raw_{idx:05d}", base_sample_id=f"fixed60_qa600_{idx:05d}")
        for idx, row in enumerate(legacy_rows)
    ]
    qa_rows: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_rows):
        spec = specs[idx % len(specs)]
        prefer_anomaly = idx % 2 == 0
        window = _choose_window(
            raw,
            seed=int(args.seed) + idx * 15485863,
            prefer_anomaly=prefer_anomaly,
            min_window_size=int(args.min_window_size),
            max_window_size=int(args.max_window_size),
        )
        sample = retarget_sample_to_window(
            raw,
            int(window["start"]),
            int(window["end"]),
            window_index=idx,
            copy_sample=True,
        )
        sample["sample_id"] = f"fixed60_qa600_{idx:05d}_{spec.question_id}"
        sample["experiment"] = {
            "name": Path(args.output_dir).name,
            "raw_series_id": raw.get("sample_id"),
            "fixed_question_index": idx % len(specs),
            "prefer_anomaly_window": prefer_anomaly,
        }
        qa_rows.append(assign_question(sample, spec, index=idx, copy_sample=False, preserve_source_question=False))

    labels = [_label_row(row) for row in qa_rows]
    write_jsonl(raw_rows, data_dir / "raw_series.jsonl")
    write_jsonl(qa_rows, data_dir / "train.jsonl")
    write_jsonl(qa_rows[: min(120, len(qa_rows))], data_dir / "test.jsonl")
    write_jsonl(labels, data_dir / "labels.jsonl")

    base_config = load_json(ROOT / args.base_config)
    config = _write_experiment_config(base_config, args, specs)
    save_json(config, out_dir / "config.json")
    question_manifest = [
        {
            "question_id": spec.question_id,
            "difficulty": spec.difficulty,
            "answer_type": spec.answer_type,
            "family": spec.family,
            "text": spec.text,
            "choices": list(spec.choices),
        }
        for spec in specs
    ]
    save_json({"questions": question_manifest, "counts": grouped_question_counts()}, out_dir / "fixed_question_specs_frame_bank.json")

    by_difficulty: Dict[str, int] = {}
    by_answer_type: Dict[str, int] = {}
    by_anomaly_type: Dict[str, int] = {}
    by_answer_label: Dict[str, int] = {}
    anomalous = 0
    for row, label in zip(qa_rows, labels):
        by_difficulty[row["question_difficulty"]] = by_difficulty.get(row["question_difficulty"], 0) + 1
        by_answer_type[row["question_answer_type"]] = by_answer_type.get(row["question_answer_type"], 0) + 1
        by_anomaly_type[str(label["anomaly_type"])] = by_anomaly_type.get(str(label["anomaly_type"]), 0) + 1
        if label["answer_label"] is not None:
            by_answer_label[str(label["answer_label"])] = by_answer_label.get(str(label["answer_label"]), 0) + 1
        anomalous += int(bool(label["is_anomalous"]))
    summary = {
        "experiment_dir": str(out_dir),
        "data_dir": str(data_dir),
        "config": str(out_dir / "config.json"),
        "raw_series_path": str(data_dir / "raw_series.jsonl"),
        "train_path": str(data_dir / "train.jsonl"),
        "test_preview_path": str(data_dir / "test.jsonl"),
        "labels_path": str(data_dir / "labels.jsonl"),
        "num_samples": len(qa_rows),
        "num_raw_series": len(raw_rows),
        "fixed_question_count": len(specs),
        "counts_by_difficulty": by_difficulty,
        "counts_by_answer_type": by_answer_type,
        "counts_by_anomaly_type": by_anomaly_type,
        "counts_by_answer_label": by_answer_label,
        "anomalous_count": anomalous,
        "normal_count": len(qa_rows) - anomalous,
        "examples": [
            {
                "sample_id": row.get("sample_id"),
                "interval": row.get("target_interval"),
                "question": row.get("question"),
                "label": label,
                "teacher_answer": (row.get("target_output") or {}).get("final_answer"),
            }
            for row, label in list(zip(qa_rows, labels))[:5]
        ],
    }
    save_json(summary, out_dir / "dataset_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
