from __future__ import annotations

import argparse
import base64
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.anomaly_score_context import aggregate_anomaly_scores, score_stats
from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:160] or "sample"


def _channel_names(sample: Dict[str, Any], width: int) -> List[str]:
    channels = sample.get("channels") or []
    names = [
        str(ch.get("channel_id") or ch.get("name") or f"ch_{idx}")
        for idx, ch in enumerate(channels[:width])
    ]
    while len(names) < width:
        names.append(f"ch_{len(names)}")
    return names


def _load_model(
    config_path: Path,
    checkpoint_path: str,
    *,
    num_channels_override: int | None = None,
    skip_global_hint_head: bool = False,
) -> AXISMultivariateIntervalProposer:
    config = load_json(config_path)
    checkpoint = Path(relative_to_root(ROOT, checkpoint_path))
    if num_channels_override is not None:
        model, _ = AXISMultivariateIntervalProposer.load_shape_compatible_checkpoint(
            str(checkpoint),
            config,
            num_channels_override=int(num_channels_override),
            skip_global_hint_head=bool(skip_global_hint_head),
        )
        return model
    try:
        return AXISMultivariateIntervalProposer.load(str(checkpoint), config)
    except Exception:
        model, _ = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
            str(checkpoint),
            config,
            threshold=float((config.get("interval_proposal") or {}).get("threshold", 0.5)),
        )
        return model


def _plot_one(
    sample: Dict[str, Any],
    *,
    index: int,
    output_dir: Path,
    model: AXISMultivariateIntervalProposer | None,
    aggregation: str,
    mode: str,
) -> Dict[str, Any]:
    values = np.asarray(sample["series"]["values"], dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    length, num_channels = int(values.shape[0]), int(values.shape[1])
    interval = sample.get("target_interval") or {}
    start = max(0, min(int(interval.get("start", 0)), max(0, length - 1)))
    end = max(start + 1, min(int(interval.get("end", length)), length))
    channel_ids = _channel_names(sample, num_channels)
    x = np.arange(length)
    include_score_axis = mode == "score"
    scores = None
    if include_score_axis:
        if model is None:
            raise ValueError("score mode requires a loaded interval proposer")
        probs = model.anomaly_probabilities(sample)
        scores = aggregate_anomaly_scores(probs, sample.get("channels") or [], aggregation=aggregation)

    n_axes = num_channels + (1 if include_score_axis else 0)
    fig_height = max(4.5, min(1.6 + 0.9 * n_axes, 26.0))
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(12, fig_height),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": [1.0] * num_channels + ([0.75] if include_score_axis else [])},
    )
    axes = axes[:, 0]

    try:
        for ch_idx in range(num_channels):
            ax = axes[ch_idx]
            raw = np.asarray(values[:, ch_idx], dtype=float)
            if np.isfinite(raw).any():
                mean = float(np.nanmean(raw))
                std = float(np.nanstd(raw))
                plotted = (raw - mean) / std if std > 1e-8 else raw - mean
                scale_note = "z" if std > 1e-8 else "centered"
            else:
                plotted = raw
                scale_note = "value"
            ax.plot(x, plotted, color="#2563eb", linewidth=1.0, alpha=0.75)
            ax.plot(x[start:end], plotted[start:end], color="#dc2626", linewidth=1.8, alpha=0.95)
            ax.axvspan(start, end - 1, color="#facc15", alpha=0.22)
            ax.axvline(start, color="#92400e", linewidth=0.8, alpha=0.65)
            ax.axvline(end - 1, color="#92400e", linewidth=0.8, alpha=0.65)
            ax.set_ylabel(f"{channel_ids[ch_idx]}\n{scale_note}", rotation=0, ha="right", va="center", fontsize=8)
            ax.grid(True, alpha=0.22, linestyle="--")

        if include_score_axis and scores is not None:
            ax = axes[-1]
            ax.plot(x, scores, color="#111111", linewidth=1.0, alpha=0.85)
            ax.plot(x[start:end], scores[start:end], color="#dc2626", linewidth=1.8, alpha=0.95)
            ax.axvspan(start, end - 1, color="#facc15", alpha=0.22)
            ax.axvline(start, color="#92400e", linewidth=0.8, alpha=0.65)
            ax.axvline(end - 1, color="#92400e", linewidth=0.8, alpha=0.65)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel(
                f"anomaly score\n{aggregation}; higher = suspicious",
                rotation=0,
                ha="right",
                va="center",
                fontsize=8,
            )
            ax.grid(True, alpha=0.22, linestyle="--")
            ax.set_xlabel("Time step")
        else:
            axes[-1].set_xlabel("Time step")

        title = f"Multivariate Time Series with Highlighted Analysis Window ({start}-{end})"
        if include_score_axis:
            title += " and detector anomaly score (higher = more suspicious)"
        axes[0].set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout(h_pad=0.25)
        safe_id = _safe_name(str(sample.get("sample_id", f"sample_{index:04d}")))
        suffix = "_score.png" if include_score_axis else ".png"
        path = output_dir / f"{index + 1:04d}_{safe_id}{suffix}"
        fig.savefig(path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)

    record = {
        "index": int(index),
        "sample_id": sample.get("sample_id"),
        "path": str(path),
        "target_interval": [start, end],
        "mode": mode,
        "aggregation": aggregation if include_score_axis else None,
    }
    if include_score_axis and scores is not None:
        record["score_stats"] = score_stats(scores, start, end)
    return record


def _image_to_base64(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _base64_to_image(payload: str, path: Path) -> None:
    buffer = io.BytesIO(base64.b64decode(payload))
    path.write_bytes(buffer.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/torch_fixed60_qa600_timercd.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--aggregation", choices=["max", "mean"], default="max")
    parser.add_argument("--mode", choices=["raw", "score"], default="score")
    parser.add_argument("--write-updated-jsonl", default=None)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_path)[: int(args.limit)]
    model = None
    if args.mode == "score":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --mode score")
        num_channels_override = None
        if rows:
            first_values = np.asarray(rows[0]["series"]["values"], dtype=float)
            if first_values.ndim == 1:
                first_values = first_values[:, None]
            num_channels_override = int(first_values.shape[1])
        model = _load_model(
            config_path,
            args.checkpoint,
            num_channels_override=num_channels_override,
            skip_global_hint_head=True,
        )
        model.eval()
    records: List[Dict[str, Any]] = []
    updated_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        record = _plot_one(
            row,
            index=idx,
            output_dir=output_dir,
            model=model,
            aggregation=args.aggregation,
            mode=str(args.mode),
        )
        records.append(record)
        if args.write_updated_jsonl:
            copied = dict(row)
            artifacts = dict(copied.get("question_generation_artifacts") or {})
            if args.mode == "score":
                artifacts["anomaly_score_image_path"] = record["path"]
                artifacts["anomaly_score_image_aggregation"] = args.aggregation
            else:
                artifacts["raw_image_path"] = record["path"]
            copied["question_generation_artifacts"] = artifacts
            updated_rows.append(copied)
    manifest = {"records": records, "mode": args.mode, "aggregation": args.aggregation, "data": str(data_path)}
    save_json(manifest, output_dir / "manifest.json")
    if args.write_updated_jsonl:
        updated_path = Path(args.write_updated_jsonl)
        if not updated_path.is_absolute():
            updated_path = ROOT / updated_path
        write_jsonl(updated_rows, updated_path)
    print(json.dumps({"output_dir": str(output_dir), "num_images": len(records), "manifest": str(output_dir / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
