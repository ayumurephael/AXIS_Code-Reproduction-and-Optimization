from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.legacy_adapter import convert_legacy_sample
from src.mvaxis.utils import load_json, save_json, write_jsonl


def _load_legacy_generator(legacy_src: Path):
    src = legacy_src / "src"
    if not src.exists():
        raise FileNotFoundError(f"Legacy AXIS generator src directory not found: {src}")
    has_statsmodels = importlib.util.find_spec("statsmodels") is not None
    if not has_statsmodels:
        statsmodels = types.ModuleType("statsmodels")
        tsa = types.ModuleType("statsmodels.tsa")
        arima_process = types.ModuleType("statsmodels.tsa.arima_process")

        class _MissingArmaProcess:
            def __init__(self, *_args, **_kwargs):
                raise ImportError("statsmodels is required for ARIMA trends; use_attribute_set generation disables ARIMA.")

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
                raise ImportError("pywt is required for wavelet seasonality; use_attribute_set generation disables wavelets.")

        pywt.Wavelet = _MissingWavelet
        sys.modules.setdefault("pywt", pywt)
    os.chdir(legacy_src)
    sys.path.insert(0, str(src))
    from generate_dataset import generate_dataset  # type: ignore

    # The local environment may not have statsmodels. In that case keep the
    # original multivariate DAG/anomaly code path, but remove the optional ARIMA
    # trend branch from the attribute sampler so the shim above is never used.
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


def _split_rows(rows: List[Dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = int(len(order) * train_ratio)
    n_val = int(len(order) * val_ratio)
    split_map = {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }
    return {split: [rows[i] for i in idxs] for split, idxs in split_map.items()}


def _write_config(base_config: Path, output_config: Path, output_dir: str, num_channels: int) -> None:
    cfg = load_json(base_config)
    cfg["data"]["output_dir"] = output_dir
    cfg["data"]["num_channels"] = num_channels
    cfg["model"]["feature_dim"] = num_channels
    cfg["train_interval_proposal"]["checkpoint_path"] = f"outputs/checkpoints/axis_interval_proposer_{Path(output_dir).name}.pt"
    output_config.parent.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-src",
        default=str(ROOT / "legacy" / "TSAD_dataset_gen-axis"),
        help="Path to extracted TSAD_dataset_gen-axis directory",
    )
    parser.add_argument("--output-dir", default="data/legacy_axis_smoke")
    parser.add_argument("--config-output", default=None)
    parser.add_argument("--base-config", default="configs/torch_semantic_scale.json")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seq-len", type=int, default=160)
    parser.add_argument("--anomaly-ratio", type=float, default=0.75)
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--activate-function", action="store_true")
    parser.add_argument("--use-attribute-set", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
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
    rows = [
        convert_legacy_sample(row, sample_id=f"legacy_axis_{idx:05d}", base_sample_id=f"legacy_axis_{idx:05d}")
        for idx, row in enumerate(legacy_rows)
    ]
    splits = _split_rows(rows, float(args.train_ratio), float(args.val_ratio), int(args.seed))
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, split_rows in splits.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(split_rows, path)
        paths[split] = str(path)
    if args.config_output:
        _write_config(ROOT / args.base_config, ROOT / args.config_output, args.output_dir, int(args.num_features))
    summary = {
        "generated": paths,
        "counts": {split: len(split_rows) for split, split_rows in splits.items()},
        "num_features": int(args.num_features),
        "first_target_output": rows[0]["target_output"] if rows else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
