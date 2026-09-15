from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.question_provider import (
    ANOMALY_RATIO,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    SAMPLES_PER_SERIES,
    _affected_channels,
    _generate_multichannel_time_series_image,
    sample_analysis_windows,
)
from src.mvaxis.utils import save_json


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _normalize_source_range(total: int, start: int, end: int | None) -> tuple[int, int]:
    range_start = max(0, int(start))
    range_end = total if end is None else int(end)
    range_end = min(total, max(range_start, range_end))
    if range_start >= total:
        raise ValueError(f"source-start-index {range_start} is outside the available range 0..{max(total - 1, 0)}")
    return range_start, range_end


def _window_key(source_index: int, window_index: int, start: int, end: int) -> Tuple[int, int, int, int]:
    return (int(source_index), int(window_index), int(start), int(end))


def _image_name(source_index: int, window_index: int, start: int, end: int) -> str:
    return f"{source_index + 1:05d}_w{window_index:03d}_{start:04d}_{end:04d}.png"


def _parse_image_key(name: str) -> Tuple[int, int, int, int] | None:
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 4 or not parts[1].startswith("w"):
        return None
    try:
        source_index = int(parts[0]) - 1
        window_index = int(parts[1][1:])
        start = int(parts[2])
        end = int(parts[3])
    except ValueError:
        return None
    return _window_key(source_index, window_index, start, end)


def _load_existing_window_keys(windows_path: Path) -> Set[Tuple[int, int, int, int]]:
    if not windows_path.exists():
        return set()
    done: Set[Tuple[int, int, int, int]] = set()
    for row in _iter_jsonl(windows_path):
        interval = row.get("target_interval") or {}
        try:
            done.add(
                _window_key(
                    int(row.get("source_index")),
                    int(row.get("window_index")),
                    int(interval.get("start")),
                    int(interval.get("end")),
                )
            )
        except (TypeError, ValueError):
            continue
    return done


def _dedupe_windows_jsonl(windows_path: Path) -> int:
    if not windows_path.exists():
        return 0
    unique_rows: Dict[Tuple[int, int, int, int], Dict[str, Any]] = {}
    for row in _iter_jsonl(windows_path):
        interval = row.get("target_interval") or {}
        try:
            key = _window_key(
                int(row.get("source_index")),
                int(row.get("window_index")),
                int(interval.get("start")),
                int(interval.get("end")),
            )
        except (TypeError, ValueError):
            continue
        unique_rows[key] = row
    with windows_path.open("w", encoding="utf-8") as handle:
        for row in unique_rows.values():
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(unique_rows)


def _load_existing_image_keys(images_dir: Path) -> Set[Tuple[int, int, int, int]]:
    if not images_dir.exists():
        return set()
    done: Set[Tuple[int, int, int, int]] = set()
    for image_path in images_dir.glob("*.png"):
        key = _parse_image_key(image_path.name)
        if key is not None:
            done.add(key)
    return done


def _scope_from_affected(affected: List[str]) -> str | None:
    if not affected:
        return None
    if len(affected) == 1:
        return "node"
    if len(affected) <= 3:
        return "edge"
    return "subgraph"


def _window_root_channel(sample: Dict[str, Any], affected: List[str]) -> str | None:
    if not affected:
        return None
    root = str(((sample.get("root_cause") or {}).get("channel_id") or "")).strip()
    if root and root in affected:
        return root
    roots = list((sample.get("synthetic_label") or {}).get("root_cause_channels") or [])
    for candidate in roots:
        candidate = str(candidate)
        if candidate in affected:
            return candidate
    return affected[0]


def _window_metadata(
    sample: Dict[str, Any],
    *,
    source_index: int,
    window_index: int,
    start: int,
    end: int,
    has_anomaly: bool,
    variance: float,
    image_path: Path,
) -> Dict[str, Any]:
    affected = _affected_channels(sample, start, end)
    root_channel = _window_root_channel(sample, affected)
    source_id = str(sample.get("sample_id") or f"series_{source_index:05d}")
    synthetic = sample.get("synthetic_label") or {}
    return {
        "sample_id": f"{source_id}_w{window_index:03d}_{start:04d}_{end:04d}",
        "source_sample_id": source_id,
        "source_index": int(source_index),
        "window_index": int(window_index),
        "target_interval": {"start": int(start), "end": int(end)},
        "has_anomaly": bool(has_anomaly),
        "variance": float(variance),
        "image_path": str(image_path),
        "root_cause_channel": root_channel,
        "affected_channels": affected,
        "anomaly_type": synthetic.get("anomaly_type") if has_anomaly else None,
        "anomaly_scope": _scope_from_affected(affected) if has_anomaly else None,
        "channels": [str(ch.get("channel_id")) for ch in (sample.get("channels") or [])],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample analysis windows and render highlighted images.")
    parser.add_argument("--source", default="outputs/ts_data/ts_531/raw_series.jsonl")
    parser.add_argument("--output-dir", default="outputs/ts_data/ts_531_windows")
    parser.add_argument("--samples-per-series", type=int, default=SAMPLES_PER_SERIES)
    parser.add_argument("--min-window-size", type=int, default=MIN_WINDOW_SIZE)
    parser.add_argument("--max-window-size", type=int, default=MAX_WINDOW_SIZE)
    parser.add_argument("--anomaly-ratio", type=float, default=ANOMALY_RATIO)
    parser.add_argument("--seed", type=int, default=531)
    parser.add_argument(
        "--source-start-index",
        type=int,
        default=0,
        help="Inclusive raw tsdata row index to start from before sampling windows.",
    )
    parser.add_argument(
        "--source-end-index",
        type=int,
        default=None,
        help="Exclusive raw tsdata row index to stop at before sampling windows.",
    )
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    args = parser.parse_args()

    source_path = ROOT / args.source if not Path(args.source).is_absolute() else Path(args.source)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    total_series = _count_jsonl(source_path)
    range_start, range_end = _normalize_source_range(total_series, int(args.source_start_index), args.source_end_index)
    selected_series = max(0, range_end - range_start)
    total_windows = selected_series * int(args.samples_per_series)
    windows_path = output_dir / f"windows_{total_windows}.jsonl"
    summary_path = output_dir / "summary.json"
    deduped_rows = _dedupe_windows_jsonl(windows_path)
    progress = ProgressPrinter(
        total_windows,
        label="window-render",
        step_percent=float(args.progress_step_percent),
    )
    existing_metadata_keys = _load_existing_window_keys(windows_path)
    existing_image_keys = _load_existing_image_keys(images_dir)
    anomalous_windows = 0
    normal_windows = 0
    if windows_path.exists():
        for row in _iter_jsonl(windows_path):
            if bool(row.get("has_anomaly")):
                anomalous_windows += 1
            else:
                normal_windows += 1
    written = len(existing_metadata_keys)
    progress.start(
        extra=(
            f"resume_metadata={written} resume_images={len(existing_image_keys)} deduped={deduped_rows} "
            f"anomalous={anomalous_windows} normal={normal_windows}"
        )
    )

    with windows_path.open("a", encoding="utf-8") as handle:
        for sample_idx, sample in enumerate(_iter_jsonl(source_path)):
            if sample_idx < range_start:
                continue
            if sample_idx >= range_end:
                break
            windows = sample_analysis_windows(
                sample,
                seed=int(args.seed) + sample_idx * 7919,
                samples_per_series=int(args.samples_per_series),
                min_window_size=int(args.min_window_size),
                max_window_size=int(args.max_window_size),
                anomaly_ratio=float(args.anomaly_ratio),
            )
            for window_idx, window in enumerate(windows, start=1):
                start = int(window["start"])
                end = int(window["end"])
                key = _window_key(sample_idx, window_idx, start, end)
                image_name = _image_name(sample_idx, window_idx, start, end)
                image_path = images_dir / image_name
                needs_image = not image_path.exists()
                needs_metadata = key not in existing_metadata_keys
                if not needs_image and not needs_metadata:
                    progress.update(
                        written,
                        extra=f"written={written} anomalous={anomalous_windows} normal={normal_windows}",
                    )
                    continue
                if needs_image:
                    image_base64 = _generate_multichannel_time_series_image(
                        sample,
                        window_start=start,
                        window_end=end,
                    )
                    if image_base64:
                        image_path.write_bytes(base64.b64decode(image_base64))
                metadata = _window_metadata(
                    sample,
                    source_index=sample_idx,
                    window_index=window_idx,
                    start=start,
                    end=end,
                    has_anomaly=bool(window.get("has_anomaly")),
                    variance=float(window.get("variance") or 0.0),
                    image_path=image_path,
                )
                if needs_metadata:
                    handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    handle.flush()
                    existing_metadata_keys.add(key)
                    if metadata["has_anomaly"]:
                        anomalous_windows += 1
                    else:
                        normal_windows += 1
                    written += 1
                progress.update(
                    written,
                    extra=f"written={written} anomalous={anomalous_windows} normal={normal_windows}",
                )

    summary = {
        "source": str(source_path),
        "source_start_index": int(range_start),
        "source_end_index": int(range_end),
        "output_dir": str(output_dir),
        "images_dir": str(images_dir),
        "windows_jsonl": str(windows_path),
        "num_series": selected_series,
        "samples_per_series": int(args.samples_per_series),
        "num_windows": written,
        "anomalous_windows": anomalous_windows,
        "normal_windows": normal_windows,
        "min_window_size": int(args.min_window_size),
        "max_window_size": int(args.max_window_size),
        "anomaly_ratio": float(args.anomaly_ratio),
        "seed": int(args.seed),
    }
    save_json(summary, summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

