from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer
from src.mvaxis.proposal import interval_coverage, interval_iou, true_anomaly_interval
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def _fmt_list(values: Iterable[Any] | None) -> str:
    items = list(values or [])
    return ",".join(str(x) for x in items) if items else "none"


def _as_interval(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _window_corr(values: np.ndarray, start: int, end: int) -> Tuple[np.ndarray, float]:
    window = values[start:end]
    if window.shape[0] < 3:
        corr = np.eye(values.shape[1])
    else:
        corr = np.corrcoef(window.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~np.eye(corr.shape[0], dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(corr[mask]))) if mask.any() else 1.0
    return corr, mean_abs_corr


def _complexity(sample: Dict[str, Any]) -> float:
    truth = true_anomaly_interval(sample)
    if truth is None:
        return -1.0
    values = np.asarray(sample["series"]["values"], dtype=float)
    _, mean_abs_corr = _window_corr(values, truth[0], truth[1])
    fact = sample["target_output"]["fact_check"]
    graph = sample["causal_graph"]
    score = 1.0 - mean_abs_corr
    score += 0.45 if graph.get("graph_type") in {"diamond", "fc_layer"} else 0.25
    score += 0.35 if fact.get("anomaly_scope") in {"edge", "subgraph"} else 0.0
    affected = fact.get("affected_channels") or []
    score += 0.15 if 1 <= len(affected) <= 6 else 0.0
    return score


def _select_samples(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    abnormal = [row for row in rows if true_anomaly_interval(row) is not None]
    abnormal.sort(key=_complexity, reverse=True)
    selected: List[Dict[str, Any]] = []
    seen_types = set()
    seen_graphs = set()
    for row in abnormal:
        fact = row["target_output"]["fact_check"]
        key = (row["causal_graph"].get("graph_type"), fact.get("anomaly_type"))
        if key in seen_types and len(selected) < max(2, limit // 2):
            continue
        selected.append(row)
        seen_types.add(key)
        seen_graphs.add(row["causal_graph"].get("graph_type"))
        if len(selected) >= limit:
            return selected
    chosen = {row["sample_id"] for row in selected}
    for row in abnormal:
        if row["sample_id"] in chosen:
            continue
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _plot_sample(
    sample: Dict[str, Any],
    model: AXISMultivariateIntervalProposer,
    config: Dict[str, Any],
    rank: int,
    out_dir: Path,
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
    proposal_interval = _as_interval([proposal.get("start"), proposal.get("end")])
    iou = interval_iou(proposal_interval, truth)
    coverage = interval_coverage(proposal_interval, truth)
    time_score = probs.max(axis=1)
    active = [i for i, ch in enumerate(sample["channels"]) if ch.get("active", True)] or list(range(values.shape[1]))
    channel_ids = [sample["channels"][i]["channel_id"] for i in active]
    fact = sample["target_output"]["fact_check"]
    truth_root = fact.get("root_cause_channel")
    pred_root = proposal.get("predicted_root_cause_channel")
    truth_affected = set(fact.get("affected_channels") or [])
    pred_affected = {item.get("channel_id") for item in proposal.get("channel_hints") or []}
    graph = sample["causal_graph"]
    interval = truth or proposal_interval or (0, values.shape[0])
    corr, mean_abs_corr = _window_corr(values[:, active], int(interval[0]), int(interval[1]))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[2.3, 1.0], hspace=0.16, wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_score = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_hm = fig.add_subplot(gs[:, 1])

    x = np.arange(values.shape[0])
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(active))))
    offsets = np.arange(len(active), dtype=float) * 3.05
    for row_idx, (ch_idx, ch_id) in enumerate(zip(active, channel_ids)):
        series = values[:, ch_idx]
        std = float(np.std(series)) or 1.0
        z = (series - float(np.mean(series))) / std
        offset = offsets[row_idx]
        focus = ch_id == truth_root or ch_id in truth_affected or ch_id == pred_root or ch_id in pred_affected
        ax.plot(x, z + offset, color=colors[row_idx % len(colors)], linewidth=2.3 if focus else 1.0, alpha=0.95 if focus else 0.42)
        abnormal_idx = np.flatnonzero(labels[:, ch_idx] > 0)
        if len(abnormal_idx):
            ax.scatter(abnormal_idx, z[abnormal_idx] + offset, s=11, color="crimson", alpha=0.9, zorder=5)
    if proposal_interval:
        ax.axvspan(proposal_interval[0], proposal_interval[1], color="#4C78A8", alpha=0.16, label="anomaly-head proposal")
        ax_score.axvspan(proposal_interval[0], proposal_interval[1], color="#4C78A8", alpha=0.16)
    if truth:
        ax.axvspan(truth[0], truth[1], color="#F58518", alpha=0.24, label="truth anomaly")
        ax_score.axvspan(truth[0], truth[1], color="#F58518", alpha=0.24)
        ax.axvline(truth[0], color="#B35C00", linestyle="--", linewidth=1.2)
        ax.axvline(truth[1], color="#B35C00", linestyle="--", linewidth=1.2)
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_ids)
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_ylabel("channels, z-normalized")
    ax.legend(loc="upper right", fontsize=8)

    ax_score.plot(x, time_score, color="#222222", linewidth=1.4, label="max channel anomaly prob")
    ax_score.axhline(float(proposal.get("threshold", model.threshold)), color="crimson", linestyle="--", linewidth=1.0, label="threshold")
    ax_score.set_ylim(-0.02, 1.02)
    ax_score.set_xlabel("time step")
    ax_score.set_ylabel("head score")
    ax_score.grid(True, alpha=0.25)
    ax_score.legend(loc="upper right", fontsize=8)

    im = ax_hm.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax_hm.set_title(f"Truth-window correlation\nmean |r|={mean_abs_corr:.2f}")
    ax_hm.set_xticks(range(len(channel_ids)))
    ax_hm.set_yticks(range(len(channel_ids)))
    ax_hm.set_xticklabels(channel_ids, rotation=90, fontsize=7)
    ax_hm.set_yticklabels(channel_ids, fontsize=7)
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    title = (
        f"Relation-stress #{rank} | {sample['sample_id']} | graph={graph.get('graph_type')}/{graph.get('mechanism')} | "
        f"driver={(graph.get('structural_equation') or {}).get('root_driver', {}).get('type')} | "
        f"type={fact.get('anomaly_scope')}/{fact.get('anomaly_type')}\n"
        f"truth interval={truth}, proposal=({proposal.get('start')},{proposal.get('end')}), "
        f"IoU={iou:.2f}, coverage={coverage:.2f}, score={float(proposal.get('proposal_score', 0.0)):.2f}; "
        f"truth root={truth_root}, head root={pred_root}"
    )
    fig.suptitle(title, fontsize=12)
    footer = (
        f"truth affected={_fmt_list(fact.get('affected_channels'))}; "
        f"head top channels={_fmt_list([item.get('channel_id') for item in proposal.get('channel_hints') or []])}; "
        f"effective_alpha={sample.get('synthetic_label', {}).get('effective_anomaly_strength_alpha')}"
    )
    fig.subplots_adjust(bottom=0.08, top=0.86)
    fig.text(0.02, 0.02, "\n".join(wrap(footer, width=170)), fontsize=9, family="monospace")
    name = f"stress_{rank:02d}_{sample['sample_id']}_{graph.get('graph_type')}_{fact.get('anomaly_type')}.png".replace("/", "_")
    path = out_dir / name
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "rank": rank,
        "sample_id": sample["sample_id"],
        "path": str(path),
        "graph_type": graph.get("graph_type"),
        "mechanism": graph.get("mechanism"),
        "driver": (graph.get("structural_equation") or {}).get("root_driver", {}).get("type"),
        "truth_interval": list(truth) if truth else None,
        "proposal_interval": [proposal.get("start"), proposal.get("end")],
        "proposal_score": proposal.get("proposal_score"),
        "threshold": proposal.get("threshold"),
        "iou": iou,
        "coverage": coverage,
        "truth_root": truth_root,
        "head_root": pred_root,
        "truth_type": fact.get("anomaly_type"),
        "truth_scope": fact.get("anomaly_scope"),
        "truth_affected": fact.get("affected_channels"),
        "head_channels": proposal.get("channel_hints"),
        "mean_abs_corr": mean_abs_corr,
        "effective_alpha": sample.get("synthetic_label", {}).get("effective_anomaly_strength_alpha"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_relation_stress_v1.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/axis_interval_proposer_semantic_scale.pt")
    parser.add_argument("--timercd-checkpoint", default="")
    parser.add_argument("--output-dir", default="outputs/runs/relation_stress_v1/anomaly_head_figures")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    rows = read_jsonl(Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{args.split}.jsonl")
    selected = _select_samples(rows, int(args.limit))
    if args.timercd_checkpoint:
        model, _ = AXISMultivariateIntervalProposer.load_timercd_checkpoint(
            relative_to_root(ROOT, args.timercd_checkpoint),
            config,
        )
    else:
        model = AXISMultivariateIntervalProposer.load(relative_to_root(ROOT, args.checkpoint), config)
    model.eval()
    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [_plot_sample(sample, model, config, idx, out_dir) for idx, sample in enumerate(selected, start=1)]
    save_json({"records": manifest}, out_dir / "manifest.json")
    print(json.dumps({"output_dir": str(out_dir), "num_figures": len(manifest), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
