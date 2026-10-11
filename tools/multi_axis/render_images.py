from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.models.MultiAXIS.config import MultiAxisConfig
from src.models.MultiAXIS.data import (
    ManifestDataset,
    visual_image_id,
    visual_image_relpath,
)


CONTEXT_COLOR = "#000000"
TARGET_COLOR = "#173f7a"
TARGET_BACKGROUND = "#fff2b2"
RENDER_FORMAT = "multi-axis-vl-image-manifest-v1"
TIMES_FONT_FILES = ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf")


_PARALLEL_DATASETS: Dict[str, ManifestDataset] = {}
_PARALLEL_DATA_ROOT: Path | None = None
_PARALLEL_OUTPUT_ROOT: Path | None = None
_PARALLEL_WINDOW_EPSILON = 1e-5
_PARALLEL_WINDOW_SCALE = 100
_PARALLEL_RENDERER_VERSION = ""
_PARALLEL_DPI = 600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_series_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def require_times_new_roman() -> Dict[str, Any]:
    from matplotlib import font_manager

    explicit_root = os.environ.get("MULTI_AXIS_TIMES_FONT_DIR")
    registered: List[Path] = []
    if explicit_root:
        font_root = Path(explicit_root).resolve()
        if not font_root.is_dir():
            raise RuntimeError(
                f"MULTI_AXIS_TIMES_FONT_DIR is not a directory: {font_root}"
            )
        missing = [name for name in TIMES_FONT_FILES if not (font_root / name).is_file()]
        if missing:
            raise RuntimeError(
                f"The explicit Times New Roman bundle is incomplete: {missing}"
            )
        for name in TIMES_FONT_FILES:
            path = font_root / name
            font_manager.fontManager.addfont(str(path))
            registered.append(path)
    try:
        resolved = Path(
            font_manager.findfont("Times New Roman", fallback_to_default=False)
        ).resolve()
    except ValueError as exc:
        raise RuntimeError(
            "Times New Roman is required by the VLM rendering specification but "
            "is not installed on this compute node."
        ) from exc
    if font_manager.FontProperties(fname=str(resolved)).get_name() != "Times New Roman":
        raise RuntimeError("Resolved renderer font is not Times New Roman")
    audited = registered or [resolved]
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in audited
    ]
    bundle_digest = hashlib.sha256()
    for item in files:
        bundle_digest.update(item["name"].encode("utf-8"))
        bundle_digest.update(b"\0")
        bundle_digest.update(item["sha256"].encode("ascii"))
        bundle_digest.update(b"\n")
    return {
        "family": "Times New Roman",
        "registration_source": "explicit_bundle" if registered else "system_font",
        "bundle_sha256": bundle_digest.hexdigest(),
        "files": files,
    }


def render_normalized_series(
    normalized_series: np.ndarray,
    interval: Tuple[int, int],
    channel_ids: Sequence[str],
    output_path: Path,
    *,
    dpi: int = 600,
) -> Dict[str, Any]:
    """Render exactly the normalized model input; no normalization occurs here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(normalized_series, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("normalized_series must be finite [T,C]")
    steps, channels = values.shape
    start, end = map(int, interval)
    if not 0 <= start < end <= steps:
        raise ValueError(f"Invalid half-open interval [{start}, {end}) for T={steps}")
    if len(channel_ids) != channels:
        raise ValueError("Channel label count does not match normalized series")
    if dpi != 600:
        raise ValueError("The VLM architecture fixes renderer dpi at 600")

    font_audit = require_times_new_roman()
    figure_height = max(2.4, min(16.0, 1.0 + 0.55 * channels))
    font_size = max(4.0, min(7.0, 8.0 - 0.06 * channels))
    x = np.arange(steps, dtype=np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".tmp.png")
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": font_size,
        "axes.unicode_minus": False,
        "savefig.dpi": dpi,
        "figure.dpi": dpi,
    }
    with plt.rc_context(rc):
        figure, axes = plt.subplots(
            channels,
            1,
            figsize=(6.4, figure_height),
            sharex=True,
            squeeze=False,
        )
        axes = axes[:, 0]
        try:
            for channel, axis in enumerate(axes):
                axis.axvspan(
                    start - 0.5,
                    end - 0.5,
                    facecolor=TARGET_BACKGROUND,
                    edgecolor="none",
                    zorder=0,
                )
                axis.plot(
                    x,
                    values[:, channel],
                    color=CONTEXT_COLOR,
                    linewidth=0.55,
                    zorder=1,
                )
                axis.plot(
                    x[start:end],
                    values[start:end, channel],
                    color=TARGET_COLOR,
                    linewidth=1.05,
                    zorder=2,
                )
                axis.axhline(0.0, color="#4b5563", linewidth=0.25, zorder=0)
                axis.set_ylabel(str(channel_ids[channel]), rotation=0, ha="right", va="center")
                axis.set_xlim(-0.5, steps - 0.5)
                axis.tick_params(axis="x", labelbottom=True, length=1.5, pad=1)
                axis.tick_params(axis="y", length=1.5, pad=1)
                axis.set_xlabel("Global time index", labelpad=1)
                axis.grid(False)
            figure.subplots_adjust(left=0.12, right=0.995, top=0.995, bottom=0.035, hspace=0.56)
            figure.savefig(
                temporary,
                format="png",
                dpi=dpi,
                facecolor="white",
                metadata={"Software": "Multi-AXIS deterministic renderer"},
            )
        finally:
            plt.close(figure)
    os.replace(temporary, output_path)

    from PIL import Image

    with Image.open(output_path) as rendered:
        rendered.verify()
    with Image.open(output_path) as rendered:
        width, height = map(int, rendered.size)
    return {
        "width": width,
        "height": height,
        "pixels": width * height,
        "sha256": sha256_file(output_path),
        "font_family": font_audit["family"],
        "font_bundle_sha256": font_audit["bundle_sha256"],
    }


def _parallel_worker_init(
    data_root: str,
    output_root: str,
    window_epsilon: float,
    window_scale: int,
    renderer_version: str,
    dpi: int,
    font_dir: str,
) -> None:
    global _PARALLEL_DATASETS
    global _PARALLEL_DATA_ROOT
    global _PARALLEL_OUTPUT_ROOT
    global _PARALLEL_WINDOW_EPSILON
    global _PARALLEL_WINDOW_SCALE
    global _PARALLEL_RENDERER_VERSION
    global _PARALLEL_DPI

    if font_dir:
        os.environ["MULTI_AXIS_TIMES_FONT_DIR"] = font_dir
    _PARALLEL_DATASETS = {}
    _PARALLEL_DATA_ROOT = Path(data_root)
    _PARALLEL_OUTPUT_ROOT = Path(output_root)
    _PARALLEL_WINDOW_EPSILON = float(window_epsilon)
    _PARALLEL_WINDOW_SCALE = int(window_scale)
    _PARALLEL_RENDERER_VERSION = str(renderer_version)
    _PARALLEL_DPI = int(dpi)
    require_times_new_roman()


def _parallel_render_task(task: Dict[str, Any]) -> Dict[str, Any]:
    if _PARALLEL_DATA_ROOT is None or _PARALLEL_OUTPUT_ROOT is None:
        raise RuntimeError("Parallel image worker was not initialized")
    manifest_path = str(task["manifest_path"])
    dataset = _PARALLEL_DATASETS.get(manifest_path)
    if dataset is None:
        dataset = ManifestDataset(
            manifest_path,
            _PARALLEL_DATA_ROOT,
            window_epsilon=_PARALLEL_WINDOW_EPSILON,
            window_scale=_PARALLEL_WINDOW_SCALE,
            renderer_version=_PARALLEL_RENDERER_VERSION,
        )
        _PARALLEL_DATASETS[manifest_path] = dataset
    sample = dataset[int(task["index"])]
    interval = tuple(map(int, sample["interval"]))
    expected_id = visual_image_id(
        sample["base_sample_id"], interval, _PARALLEL_RENDERER_VERSION
    )
    if task["image_id"] != expected_id:
        raise RuntimeError("Manifest image id does not match its series/window identity")
    relative = visual_image_relpath(expected_id)
    output_path = _PARALLEL_OUTPUT_ROOT / relative
    render_audit = render_normalized_series(
        sample["normalized_series"],
        interval,
        sample["channel_ids"],
        output_path,
        dpi=_PARALLEL_DPI,
    )
    return {
        "format": RENDER_FORMAT,
        "image_id": expected_id,
        "image_relpath": relative,
        "base_sample_id": sample["base_sample_id"],
        "interval": list(interval),
        "time_count": sample["time_count"],
        "channel_count": sample["channel_count"],
        "channel_ids": sample["channel_ids"],
        "normalized_series_sha256": normalized_series_sha256(sample["normalized_series"]),
        "renderer_version": _PARALLEL_RENDERER_VERSION,
        "dpi": _PARALLEL_DPI,
        "half_open_boundaries": [interval[0] - 0.5, interval[1] - 0.5],
        "contains_anomaly_score": False,
        "contains_label_annotation": False,
        "colors": {
            "context": CONTEXT_COLOR,
            "target": TARGET_COLOR,
            "target_background": TARGET_BACKGROUND,
        },
        **render_audit,
    }


def iter_manifest_paths(manifest_dir: Path, requested: Sequence[str]) -> Iterable[Path]:
    if requested:
        for name in requested:
            path = manifest_dir / name
            if not path.is_file():
                raise FileNotFoundError(path)
            yield path
        return
    for path in sorted(manifest_dir.glob("*.jsonl")):
        if path.name in {"train.jsonl", "validation.jsonl"} or path.name.startswith("eval_"):
            yield path


def render_manifests(
    config: MultiAxisConfig,
    data_root: Path,
    manifest_dir: Path,
    output_root: Path,
    manifest_names: Sequence[str],
) -> Dict[str, Any]:
    if not config.vision.enabled:
        raise ValueError("Image rendering requires vision.enabled=true")
    font_audit = require_times_new_roman()
    output_root.mkdir(parents=True, exist_ok=True)
    existing_records: Dict[str, Dict[str, Any]] = {}
    existing_manifest = output_root / "image_manifest.jsonl"
    progress_manifest = output_root / "image_manifest.progress.jsonl"
    for audit_path in (existing_manifest, progress_manifest):
        if not audit_path.is_file():
            continue
        with audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    image_id = str(item["image_id"])
                    if image_id in existing_records and existing_records[image_id] != item:
                        raise RuntimeError(f"Conflicting existing image manifest id: {image_id}")
                    existing_records[image_id] = item
    records: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    manifest_counts: Dict[str, int] = {}
    for manifest_path in iter_manifest_paths(manifest_dir, manifest_names):
        dataset = ManifestDataset(
            manifest_path,
            data_root,
            window_epsilon=config.hints.window_epsilon,
            window_scale=config.hints.window_scale,
            renderer_version=config.vision.renderer_version,
        )
        manifest_counts[manifest_path.name] = len(dataset)
        for index in range(len(dataset)):
            manifest_record = dataset._manifest_record(index)
            interval = tuple(map(int, manifest_record["interval"]))
            image_id = str(
                manifest_record.get("image_id")
                or visual_image_id(
                    str(manifest_record["base_sample_id"]),
                    interval,
                    config.vision.renderer_version,
                )
            )
            identity = {
                "base_sample_id": str(manifest_record["base_sample_id"]),
                "interval": list(interval),
            }
            if image_id in seen:
                if seen[image_id] != identity:
                    raise RuntimeError(f"Image-id collision for {image_id}")
                continue
            seen[image_id] = identity
            sample = dataset[index]
            expected_id = visual_image_id(
                sample["base_sample_id"], sample["interval"], config.vision.renderer_version
            )
            if image_id != expected_id:
                raise RuntimeError("Manifest image id does not match its series/window identity")
            relative = visual_image_relpath(image_id)
            output_path = output_root / relative
            source_sha = normalized_series_sha256(sample["normalized_series"])
            previous = existing_records.get(image_id)
            if output_path.exists():
                if previous is None:
                    raise RuntimeError(
                        f"Existing image has no auditable manifest record: {output_path}"
                    )
                expected_fields = {
                    "normalized_series_sha256": source_sha,
                    "renderer_version": config.vision.renderer_version,
                    "dpi": config.vision.renderer_dpi,
                    "image_relpath": relative,
                    "font_family": font_audit["family"],
                    "font_bundle_sha256": font_audit["bundle_sha256"],
                }
                mismatched = {
                    key: (previous.get(key), value)
                    for key, value in expected_fields.items()
                    if previous.get(key) != value
                }
                actual_sha = sha256_file(output_path)
                if previous.get("sha256") != actual_sha:
                    mismatched["sha256"] = (previous.get("sha256"), actual_sha)
                if mismatched:
                    raise RuntimeError(
                        f"Existing render audit mismatch for {image_id}: {mismatched}. "
                        "Use a fresh output root; this command never overwrites audited images."
                    )
                records.append(previous)
                continue
            render_audit = render_normalized_series(
                sample["normalized_series"],
                sample["interval"],
                sample["channel_ids"],
                output_path,
                dpi=config.vision.renderer_dpi,
            )
            records.append(
                {
                    "format": RENDER_FORMAT,
                    "image_id": image_id,
                    "image_relpath": relative,
                    **identity,
                    "time_count": sample["time_count"],
                    "channel_count": sample["channel_count"],
                    "channel_ids": sample["channel_ids"],
                    "normalized_series_sha256": source_sha,
                    "renderer_version": config.vision.renderer_version,
                    "dpi": config.vision.renderer_dpi,
                    "half_open_boundaries": [interval[0] - 0.5, interval[1] - 0.5],
                    "contains_anomaly_score": False,
                    "contains_label_annotation": False,
                    "colors": {
                        "context": CONTEXT_COLOR,
                        "target": TARGET_COLOR,
                        "target_background": TARGET_BACKGROUND,
                    },
                    **render_audit,
                }
            )
            with progress_manifest.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(records[-1], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()

    manifest_output = output_root / "image_manifest.jsonl"
    temporary = manifest_output.with_name(manifest_output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, manifest_output)
    summary = {
        "format": RENDER_FORMAT,
        "renderer_version": config.vision.renderer_version,
        "dpi": config.vision.renderer_dpi,
        "unique_images": len(records),
        "manifest_rows": manifest_counts,
        "normalization": "consumed ManifestDataset.normalized_series without re-normalization",
        "font": font_audit,
        "image_manifest_sha256": sha256_file(manifest_output),
    }
    (output_root / "image_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def render_manifests_parallel(
    config: MultiAxisConfig,
    data_root: Path,
    manifest_dir: Path,
    output_root: Path,
    manifest_names: Sequence[str],
    workers: int,
) -> Dict[str, Any]:
    """Render unique images in parallel while keeping a deterministic final audit order."""
    if not config.vision.enabled:
        raise ValueError("Image rendering requires vision.enabled=true")
    if workers < 2:
        raise ValueError("Parallel rendering requires workers >= 2")
    font_audit = require_times_new_roman()
    font_dir = os.environ.get("MULTI_AXIS_TIMES_FONT_DIR", "")
    if not font_dir:
        raise RuntimeError(
            "Parallel rendering requires MULTI_AXIS_TIMES_FONT_DIR so every worker "
            "uses the same auditable font bundle"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    existing_records: Dict[str, Dict[str, Any]] = {}
    existing_manifest = output_root / "image_manifest.jsonl"
    progress_manifest = output_root / "image_manifest.progress.jsonl"
    for audit_path in (existing_manifest, progress_manifest):
        if not audit_path.is_file():
            continue
        with audit_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                image_id = str(item["image_id"])
                if image_id in existing_records and existing_records[image_id] != item:
                    raise RuntimeError(f"Conflicting existing image manifest id: {image_id}")
                existing_records[image_id] = item

    tasks: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    manifest_counts: Dict[str, int] = {}
    for manifest_path in iter_manifest_paths(manifest_dir, manifest_names):
        dataset = ManifestDataset(
            manifest_path,
            data_root,
            window_epsilon=config.hints.window_epsilon,
            window_scale=config.hints.window_scale,
            renderer_version=config.vision.renderer_version,
        )
        manifest_counts[manifest_path.name] = len(dataset)
        for index in range(len(dataset)):
            row = dataset._manifest_record(index)
            interval = tuple(map(int, row["interval"]))
            image_id = str(
                row.get("image_id")
                or visual_image_id(
                    str(row["base_sample_id"]),
                    interval,
                    config.vision.renderer_version,
                )
            )
            identity = {
                "base_sample_id": str(row["base_sample_id"]),
                "interval": list(interval),
            }
            if image_id in seen:
                if seen[image_id] != identity:
                    raise RuntimeError(f"Image-id collision for {image_id}")
                continue
            seen[image_id] = identity
            tasks.append(
                {
                    "manifest_path": str(manifest_path.resolve()),
                    "index": index,
                    "image_id": image_id,
                }
            )

    records_by_id: Dict[str, Dict[str, Any]] = {}
    pending: List[Dict[str, Any]] = []
    resume_datasets: Dict[str, ManifestDataset] = {}
    for task in tasks:
        image_id = task["image_id"]
        previous = existing_records.get(image_id)
        output_path = output_root / visual_image_relpath(image_id)
        if previous is None or not output_path.is_file():
            # A hard stop may occur after a worker atomically publishes a PNG but
            # before the parent appends its audit row. Such an unaudited orphan is
            # safe to regenerate; audited files remain strictly immutable below.
            pending.append(task)
            continue
        manifest_path = str(task["manifest_path"])
        dataset = resume_datasets.get(manifest_path)
        if dataset is None:
            dataset = ManifestDataset(
                manifest_path,
                data_root,
                window_epsilon=config.hints.window_epsilon,
                window_scale=config.hints.window_scale,
                renderer_version=config.vision.renderer_version,
            )
            resume_datasets[manifest_path] = dataset
        sample = dataset[int(task["index"])]
        expected_fields = {
            "image_relpath": visual_image_relpath(image_id),
            "base_sample_id": sample["base_sample_id"],
            "interval": list(sample["interval"]),
            "normalized_series_sha256": normalized_series_sha256(
                sample["normalized_series"]
            ),
            "renderer_version": config.vision.renderer_version,
            "dpi": config.vision.renderer_dpi,
            "font_family": font_audit["family"],
            "font_bundle_sha256": font_audit["bundle_sha256"],
        }
        mismatched = {
            key: (previous.get(key), value)
            for key, value in expected_fields.items()
            if previous.get(key) != value
        }
        actual_sha = sha256_file(output_path)
        if previous.get("sha256") != actual_sha:
            mismatched["sha256"] = (previous.get("sha256"), actual_sha)
        if mismatched:
            raise RuntimeError(
                f"Existing render audit mismatch for {image_id}: {mismatched}. "
                "Use a fresh output root; this command never overwrites audited images."
            )
        records_by_id[image_id] = previous

    if pending:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_parallel_worker_init,
            initargs=(
                str(data_root),
                str(output_root),
                config.hints.window_epsilon,
                config.hints.window_scale,
                config.vision.renderer_version,
                config.vision.renderer_dpi,
                str(Path(font_dir).resolve()),
            ),
        ) as executor:
            task_iter = iter(pending)
            future_to_task: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
            for _ in range(min(len(pending), workers * 2)):
                task = next(task_iter)
                future_to_task[executor.submit(_parallel_render_task, task)] = task
            with progress_manifest.open("a", encoding="utf-8") as handle:
                while future_to_task:
                    completed, _ = concurrent.futures.wait(
                        future_to_task,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in completed:
                        task = future_to_task.pop(future)
                        try:
                            record = future.result()
                        except Exception as exc:
                            for outstanding in future_to_task:
                                outstanding.cancel()
                            raise RuntimeError(
                                f"Parallel render failed for {task['image_id']}"
                            ) from exc
                        records_by_id[record["image_id"]] = record
                        handle.write(
                            json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                            + "\n"
                        )
                        handle.flush()
                        next_task = next(task_iter, None)
                        if next_task is not None:
                            future_to_task[
                                executor.submit(_parallel_render_task, next_task)
                            ] = next_task

    records = [records_by_id[task["image_id"]] for task in tasks]
    manifest_output = output_root / "image_manifest.jsonl"
    temporary = manifest_output.with_name(manifest_output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, manifest_output)
    summary = {
        "format": RENDER_FORMAT,
        "renderer_version": config.vision.renderer_version,
        "dpi": config.vision.renderer_dpi,
        "unique_images": len(records),
        "manifest_rows": manifest_counts,
        "normalization": "consumed ManifestDataset.normalized_series without re-normalization",
        "font": font_audit,
        "workers": workers,
        "image_manifest_sha256": sha256_file(manifest_output),
    }
    (output_root / "image_manifest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline deterministic Multi-AXIS VLM renderer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifests", nargs="*", default=[])
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Renderer processes. Use 1 for the original serial implementation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultiAxisConfig.load_json(args.config)
    common = (
        config,
        Path(args.data_root).resolve(),
        Path(args.manifest_dir).resolve(),
        Path(args.output_root).resolve(),
        args.manifests,
    )
    summary = (
        render_manifests(*common)
        if args.workers == 1
        else render_manifests_parallel(*common, workers=args.workers)
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
