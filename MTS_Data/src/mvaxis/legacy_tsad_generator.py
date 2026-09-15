from __future__ import annotations

import importlib.util
import os
import random
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List

import numpy as np

from .legacy_adapter import convert_legacy_sample
from .question_provider import (
    ANOMALY_RATIO,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    SAMPLES_PER_SERIES,
    build_qa_samples,
)
from .utils import write_jsonl


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _install_optional_dependency_shims() -> None:
    """Let the legacy generator run when optional ARIMA/wavelet deps are absent."""
    if importlib.util.find_spec("statsmodels") is None:
        statsmodels = types.ModuleType("statsmodels")
        tsa = types.ModuleType("statsmodels.tsa")
        arima_process = types.ModuleType("statsmodels.tsa.arima_process")

        class _MissingArmaProcess:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise ImportError("statsmodels is required for ARIMA trends.")

        arima_process.ArmaProcess = _MissingArmaProcess
        statsmodels.tsa = tsa
        tsa.arima_process = arima_process
        sys.modules.setdefault("statsmodels", statsmodels)
        sys.modules.setdefault("statsmodels.tsa", tsa)
        sys.modules.setdefault("statsmodels.tsa.arima_process", arima_process)
    if importlib.util.find_spec("pywt") is None:
        pywt = types.ModuleType("pywt")

        class _MissingWavelet:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise ImportError("pywt is required for wavelet seasonality.")

        pywt.Wavelet = _MissingWavelet
        sys.modules.setdefault("pywt", pywt)


def _disable_optional_attribute_branches() -> None:
    """Avoid legacy branches that require optional packages in lean envs."""
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


@contextmanager
def _legacy_import_context(legacy_src: Path) -> Iterator[None]:
    src = legacy_src / "src"
    if not src.exists():
        raise FileNotFoundError(f"Legacy TSAD generator src directory not found: {src}")
    old_cwd = Path.cwd()
    sys.path.insert(0, str(src))
    os.chdir(legacy_src)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(src))
        except ValueError:
            pass


def load_legacy_tsad_generate_dataset(legacy_src: Path | None = None) -> Callable[..., List[Dict[str, Any]]]:
    """Load TSAD_dataset_gen-axis with its original generation behavior."""
    legacy_src = legacy_src or (_repo_root() / "third_party" / "datasets_rcd")
    _install_optional_dependency_shims()
    with _legacy_import_context(legacy_src):
        module_path = legacy_src / "src" / "generate_dataset.py"
        spec = importlib.util.spec_from_file_location("mvaxis_legacy_tsad_generate_dataset", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load legacy generator from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _disable_optional_attribute_branches()
        return module.generate_dataset  # type: ignore[attr-defined]


def _split_rows(rows: List[Dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    n_train = int(len(order) * train_ratio)
    n_val = int(len(order) * val_ratio)
    return {
        "train": [rows[i] for i in order[:n_train]],
        "val": [rows[i] for i in order[n_train : n_train + n_val]],
        "test": [rows[i] for i in order[n_train + n_val :]],
    }


def _legacy_sample_count(data_cfg: Dict[str, Any]) -> int:
    if "num_samples" in data_cfg:
        return int(data_cfg["num_samples"])
    if "legacy_num_samples" in data_cfg:
        return int(data_cfg["legacy_num_samples"])
    return int(data_cfg["num_base_systems"]) * max(1, len(data_cfg.get("alpha_values") or [1.0]))


def _legacy_anomaly_ratio(data_cfg: Dict[str, Any]) -> float:
    if "legacy_anomaly_sample_ratio" in data_cfg:
        return float(data_cfg["legacy_anomaly_sample_ratio"])
    alpha_values = list(data_cfg.get("alpha_values") or [1.0])
    if not alpha_values:
        return 1.0
    return float(sum(float(alpha) > 0 for alpha in alpha_values) / len(alpha_values))


def generate_legacy_tsad_dataset(config: Dict[str, Any]) -> Dict[str, str]:
    seed = int(config.get("seed", 72))
    random.seed(seed)
    np.random.seed(seed)
    data_cfg = config["data"]
    out_dir = Path(data_cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_src = Path(data_cfg.get("legacy_src") or (_repo_root() / "third_party" / "datasets_rcd"))
    generate_dataset = load_legacy_tsad_generate_dataset(legacy_src)
    legacy_rows = generate_dataset(
        num_samples=_legacy_sample_count(data_cfg),
        seq_len=int(data_cfg["length"]),
        anomaly_sample_ratio=_legacy_anomaly_ratio(data_cfg),
        is_multivariate=True,
        num_features=int(data_cfg["num_channels"]),
        activate_function=bool(data_cfg.get("activate_function", data_cfg.get("legacy_activate_function", False))),
        use_attribute_set=bool(data_cfg.get("use_attribute_set", data_cfg.get("legacy_use_attribute_set", True))),
    )
    rows = [
        convert_legacy_sample(row, sample_id=f"legacy_tsad_{idx:05d}", base_sample_id=f"legacy_tsad_{idx:05d}")
        for idx, row in enumerate(legacy_rows)
    ]
    rows = build_qa_samples(
        rows,
        seed=int(data_cfg.get("question_seed", seed)),
        question_provider=str(data_cfg.get("question_provider", "bank")),
        sample_windows=bool(data_cfg.get("qa_sample_windows", True)),
        samples_per_series=int(data_cfg.get("qa_samples_per_series", SAMPLES_PER_SERIES)),
        min_window_size=int(data_cfg.get("qa_min_window_size", MIN_WINDOW_SIZE)),
        max_window_size=int(data_cfg.get("qa_max_window_size", MAX_WINDOW_SIZE)),
        anomaly_ratio=float(data_cfg.get("qa_anomaly_ratio", ANOMALY_RATIO)),
        annotate_sample_id=False,
    )
    splits = _split_rows(rows, float(data_cfg["train_ratio"]), float(data_cfg["val_ratio"]), seed)
    paths: Dict[str, str] = {}
    for split, split_rows in splits.items():
        path = out_dir / f"{split}.jsonl"
        write_jsonl(split_rows, path)
        paths[split] = str(path)
    return paths
