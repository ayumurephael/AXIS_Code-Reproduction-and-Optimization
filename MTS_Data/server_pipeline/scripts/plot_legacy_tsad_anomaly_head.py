from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.proposal import interval_coverage, interval_iou, true_anomaly_interval
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def _as_interval(start: Any, end: Any) -> Optional[Tuple[int, int]]:
    if start is None or end is None:
        return None
    return int(start), int(end)


def _load_rows(data_dir: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            rows.extend(read_jsonl(path))
    return rows[:limit]


def _plot_one(
    sample: Dict[str, Any],
    model: AXISMultivariateIntervalProposer,
    config: Dict[str, Any],
    out_dir: Path,
    index: int,
) -> Dict[str, Any]:
    values = np.asarray(sample["series"]["values"], dtype=float)
    labels = np.asarray(sample["series"]["labels"], dtype=int)
    probs = model.anomaly_probabilities(sample)
    proposal = model.propose_interval(
        sample,
        int(config["interval_proposal"]["window_size"]),
        int(config["interval_proposal"]["stride"]),
        int(config["model"]["top_k_channels"]),
    )
    truth = true_anomaly_interval(sample)
    pred = _as_interval(proposal.get("start"), proposal.get("end"))
    time_score = probs.max(axis=1)
    active = [i for i, ch in enumerate(sample["channels"]) if ch.get("active", True)] or list(range(values.shape[1]))
    channel_ids = [sample["channels"][i]["channel_id"] for i in active]

    fig, (ax, ax_score) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
    )
    x = np.arange(values.shape[0])
    offsets = np.arange(len(active), dtype=float) * 3.0
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(active))))
    truth_affected = set(sample["target_output"]["fact_check"].get("affected_channels") or [])
    pred_channels = {item["channel_id"] for item in proposal.get("channel_hints", [])}
    for row_idx, (ch_idx, ch_id) in enumerate(zip(active, channel_ids)):
        series = values[:, ch_idx]
        z = (series - series.mean()) / (series.std() + 1e-6)
        focus = ch_id in truth_affected or ch_id in pred_channels
        ax.plot(x, z + offsets[row_idx], color=colors[row_idx % len(colors)], linewidth=2.0 if focus else 1.0, alpha=0.95 if focus else 0.45)
        anomalous = np.flatnonzero(labels[:, ch_idx] > 0)
        if anomalous.size:
            ax.scatter(anomalous, z[anomalous] + offsets[row_idx], s=10, color="crimson", alpha=0.85, zorder=4)

    if truth is not None:
        ax.axvspan(truth[0], truth[1], color="#F58518", alpha=0.25, label="truth interval")
        ax_score.axvspan(truth[0], truth[1], color="#F58518", alpha=0.25)
    if pred is not None:
        ax.axvspan(pred[0], pred[1], color="#4C78A8", alpha=0.18, label="predicted interval")
        ax_score.axvspan(pred[0], pred[1], color="#4C78A8", alpha=0.18)

    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_ids)
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_ylabel("channels, z-normalized")
    ax.legend(loc="upper right", fontsize=9)

    ax_score.plot(x, time_score, color="#222222", linewidth=1.35, label="max channel anomaly prob")
    ax_score.axhline(float(proposal.get("threshold", model.threshold)), color="crimson", linestyle="--", linewidth=1.0, label="threshold")
    ax_score.set_ylim(-0.02, 1.02)
    ax_score.set_xlabel("time step")
    ax_score.set_ylabel("head score")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="upper right", fontsize=9)

    fact = sample["target_output"]["fact_check"]
    iou = interval_iou(pred, truth)
    coverage = interval_coverage(pred, truth)
    title = (
        f"{index:02d} {sample['sample_id']} | truth={truth} pred={pred} "
        f"IoU={iou:.2f} coverage={coverage:.2f} score={float(proposal.get('proposal_score', 0.0)):.2f}\n"
        f"truth root={fact.get('root_cause_channel')} pred root={proposal.get('predicted_root_cause_channel')} "
        f"type={fact.get('anomaly_type')} affected={','.join(fact.get('affected_channels') or [])}"
    )
    fig.suptitle(title, fontsize=11)
    fig.subplots_adjust(top=0.86, left=0.08, right=0.98, bottom=0.09)
    out_path = out_dir / f"legacy_tsad_{index:02d}_{sample['sample_id']}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {
        "index": index,
        "sample_id": sample["sample_id"],
        "path": str(out_path),
        "truth_interval": list(truth) if truth else None,
        "predicted_interval": list(pred) if pred else None,
        "iou": iou,
        "coverage": coverage,
        "proposal_score": float(proposal.get("proposal_score", 0.0)),
        "threshold": float(proposal.get("threshold", model.threshold)),
        "truth_root": fact.get("root_cause_channel"),
        "predicted_root": proposal.get("predicted_root_cause_channel"),
        "truth_affected": fact.get("affected_channels"),
        "predicted_channels": proposal.get("channel_hints"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_smoke.json")
    parser.add_argument("--data-dir", default="outputs/runs/legacy_tsad_10/data")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/axis_interval_proposer.pt")
    parser.add_argument("--output-dir", default="outputs/runs/legacy_tsad_10/anomaly_head_figures")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    data_dir = Path(relative_to_root(ROOT, args.data_dir))
    rows = _load_rows(data_dir, int(args.limit))
    model = AXISMultivariateIntervalProposer.load(relative_to_root(ROOT, args.checkpoint), config)
    model.eval()
    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [_plot_one(row, model, config, out_dir, idx) for idx, row in enumerate(rows, start=1)]
    save_json({"records": manifest}, out_dir / "manifest.json")
    print(json.dumps({"output_dir": str(out_dir), "num_figures": len(manifest), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
