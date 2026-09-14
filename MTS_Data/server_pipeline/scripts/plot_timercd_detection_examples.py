from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _as_interval(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    return int(value[0]), int(value[1])


def _span(ax: plt.Axes, interval: Optional[Tuple[int, int]], color: str, label: str, alpha: float) -> None:
    if interval is not None:
        ax.axvspan(interval[0], interval[1], color=color, alpha=alpha, label=label)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-6
    return (values - mean) / std


def _plot_one(
    sample: Dict[str, Any],
    detail: Dict[str, Any],
    out_dir: Path,
    index: int,
    *,
    figure_size: Tuple[float, float] = (15.0, 8.2),
) -> Path:
    values = np.asarray(sample["series"]["values"], dtype=float)
    labels = np.asarray(sample["series"]["labels"], dtype=int)
    channel_ids = [str(ch.get("channel_id", f"ch_{i}")) for i, ch in enumerate(sample["channels"])]
    truth = _as_interval(detail.get("truth_interval"))
    proposal = _as_interval(detail.get("proposal_interval"))
    truth_affected = set(detail.get("truth_affected_channels") or [])
    pred_channels = {
        str(item.get("channel_id"))
        for item in detail.get("predicted_channel_hints", [])
        if item.get("channel_id") is not None
    }
    z_values = _zscore(values)
    offsets = np.arange(values.shape[1], dtype=float) * 3.0
    x = np.arange(values.shape[0])

    fig, (ax, ax_labels) = plt.subplots(
        2,
        1,
        figsize=figure_size,
        sharex=True,
        gridspec_kw={"height_ratios": [3.7, 0.8], "hspace": 0.07},
    )
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, values.shape[1])))
    for ch_idx, ch_id in enumerate(channel_ids):
        focus = ch_id in truth_affected or ch_id in pred_channels
        ax.plot(
            x,
            z_values[:, ch_idx] + offsets[ch_idx],
            color=colors[ch_idx % len(colors)],
            linewidth=2.0 if focus else 0.9,
            alpha=0.95 if focus else 0.38,
        )
        anomalous = np.flatnonzero(labels[:, ch_idx] > 0)
        if anomalous.size:
            ax.scatter(
                anomalous,
                z_values[anomalous, ch_idx] + offsets[ch_idx],
                s=16 if focus else 10,
                color="#D62728",
                alpha=0.9,
                zorder=4,
            )

    _span(ax, truth, "#F58518", "truth interval", 0.26)
    _span(ax, proposal, "#4C78A8", "Time-RCD proposal", 0.20)
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_ids)
    ax.set_ylabel("channels, z-normalized")
    ax.grid(True, axis="x", alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)

    label_heat = (labels.T > 0).astype(float)
    ax_labels.imshow(
        label_heat,
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#F7F7F7", "#D62728"]),
        vmin=0,
        vmax=1,
        extent=[0, values.shape[0], -0.5, values.shape[1] - 0.5],
    )
    _span(ax_labels, truth, "#F58518", "truth interval", 0.30)
    _span(ax_labels, proposal, "#4C78A8", "Time-RCD proposal", 0.23)
    ax_labels.set_yticks(np.arange(values.shape[1]))
    ax_labels.set_yticklabels(channel_ids)
    ax_labels.set_xlabel("time step")
    ax_labels.set_ylabel("labels")

    title = (
        f"{index:02d} {sample['sample_id']} | "
        f"truth={detail.get('truth_interval')} proposal={detail.get('proposal_interval')} "
        f"IoU={float(detail.get('temporal_iou', 0.0)):.3f} "
        f"coverage={float(detail.get('truth_coverage', 0.0)):.3f} "
        f"score={float(detail.get('proposal_score', 0.0)):.3f}\n"
        f"root truth={detail.get('truth_root_cause_channel')} pred={detail.get('predicted_root_cause_channel')} "
        f"truth affected={','.join(detail.get('truth_affected_channels') or [])}"
    )
    fig.suptitle(title, fontsize=10.5)
    fig.subplots_adjust(top=0.86, left=0.08, right=0.98, bottom=0.09)
    out_path = out_dir / f"timercd_detection_{index:02d}_{sample['sample_id']}.png"
    fig.savefig(out_path, dpi=155)
    plt.close(fig)
    return out_path


def _make_contact_sheet(image_paths: Iterable[Path], out_path: Path) -> None:
    images = [plt.imread(str(path)) for path in image_paths]
    if not images:
        return
    cols = 2
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 5.3))
    flat_axes = np.asarray(axes).reshape(-1)
    for ax, image, path in zip(flat_axes, images, image_paths):
        ax.imshow(image)
        ax.set_title(path.stem, fontsize=8)
        ax.axis("off")
    for ax in flat_axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/runs/fixed60_qa600_legacy_dag/data/raw_series.jsonl")
    parser.add_argument("--details", default="outputs/runs/timercd_anomaly_detection_600_v1_details.remote.jsonl")
    parser.add_argument("--output-dir", default="outputs/runs/timercd_anomaly_detection_600_v1/figures_first10")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    details_path = Path(args.details)
    if not details_path.is_absolute():
        details_path = ROOT / details_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    offset = max(0, int(args.offset))
    limit = max(0, int(args.limit))
    samples = _read_jsonl(data_path)[offset : offset + limit]
    details = _read_jsonl(details_path)[offset : offset + limit]
    by_id = {str(item.get("sample_id")): item for item in details}

    image_paths: List[Path] = []
    manifest = []
    for rel_idx, sample in enumerate(samples):
        idx = offset + rel_idx
        detail = by_id.get(str(sample.get("sample_id")), details[rel_idx])
        path = _plot_one(sample, detail, out_dir, idx)
        image_paths.append(path)
        manifest.append(
            {
                "index": idx,
                "sample_id": sample.get("sample_id"),
                "image_path": str(path),
                "truth_interval": detail.get("truth_interval"),
                "proposal_interval": detail.get("proposal_interval"),
                "temporal_iou": detail.get("temporal_iou"),
                "truth_coverage": detail.get("truth_coverage"),
                "proposal_score": detail.get("proposal_score"),
            }
        )

    contact_sheet = out_dir / "timercd_detection_first10_contact_sheet.png"
    _make_contact_sheet(image_paths, contact_sheet)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"contact_sheet": str(contact_sheet), "records": manifest}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "contact_sheet": str(contact_sheet), "num_figures": len(image_paths)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
