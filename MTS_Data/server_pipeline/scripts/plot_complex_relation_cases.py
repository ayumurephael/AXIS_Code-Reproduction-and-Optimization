from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import shorten, wrap
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.proposal import true_anomaly_interval
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


COMPLEX_TYPES = {
    "parent_child_decoupling",
    "propagation_attenuation",
    "propagation_amplification",
    "lag_shift",
    "sign_flip",
    "root_phase_intervention",
    "trend_reversal",
    "saturation",
}


def _sample_map(config: Dict[str, Any], split: str) -> Dict[str, Dict[str, Any]]:
    path = Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{split}.jsonl"
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def _as_interval(value: Any) -> Optional[Tuple[int, int]]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _fmt_list(values: Iterable[Any] | None) -> str:
    items = list(values or [])
    return ",".join(str(x) for x in items) if items else "none"


def _quality(record: Dict[str, Any]) -> float:
    c = record.get("comparison") or {}
    return (
        float(c.get("llm_anomaly_match", 0.0))
        + float(c.get("llm_root_match", 0.0))
        + float(c.get("llm_type_match", 0.0))
        + float(c.get("llm_affected_f1", 0.0))
    )


def _window_corr(values: np.ndarray, start: int, end: int) -> np.ndarray:
    window = values[start:end]
    if window.shape[0] < 3:
        return np.eye(values.shape[1])
    corr = np.corrcoef(window.T)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def _complexity_score(record: Dict[str, Any], sample: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    values = np.asarray(sample["series"]["values"], dtype=float)
    truth = true_anomaly_interval(sample)
    proposal = _as_interval((record.get("comparison") or {}).get("proposal_interval"))
    interval = truth or proposal or (0, values.shape[0])
    corr = _window_corr(values, int(interval[0]), int(interval[1]))
    mask = ~np.eye(corr.shape[0], dtype=bool)
    mean_abs_corr = float(np.mean(np.abs(corr[mask]))) if mask.any() else 1.0
    diversity = 1.0 - mean_abs_corr
    graph = sample.get("causal_graph") or {}
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    graph_type = graph.get("graph_type")
    anomaly_type = fact.get("anomaly_type")
    scope = fact.get("anomaly_scope")
    affected = fact.get("affected_channels") or []
    active_nodes = int(graph.get("active_nodes") or len(sample.get("channels") or []))
    relation_bonus = 0.0
    relation_bonus += 0.8 if graph_type in {"tree", "diamond", "fc_layer"} else 0.0
    relation_bonus += 0.7 if scope in {"edge", "subgraph"} else 0.0
    relation_bonus += 0.6 if anomaly_type in COMPLEX_TYPES else 0.0
    relation_bonus += 0.25 if 1 <= len(affected) <= max(2, active_nodes - 2) else 0.0
    relation_bonus += 0.15 if active_nodes >= 6 else 0.0
    score = diversity + relation_bonus
    return score, {
        "mean_abs_corr": mean_abs_corr,
        "diversity": diversity,
        "graph_type": graph_type,
        "mechanism": graph.get("mechanism"),
        "active_nodes": active_nodes,
        "scope": scope,
        "anomaly_type": anomaly_type,
        "affected_count": len(affected),
    }


def _choose_cases(
    records: List[Dict[str, Any]],
    samples: Dict[str, Dict[str, Any]],
    limit: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    scored: List[Tuple[float, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    for record in records:
        sample = samples.get(str(record.get("sample_id")))
        if not sample or true_anomaly_interval(sample) is None:
            continue
        score, info = _complexity_score(record, sample)
        scored.append((score, int(record.get("index", 0)), record, sample, info))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected = []
    seen_types = set()
    seen_graphs = set()
    for _, _, record, sample, info in scored:
        key = (info.get("graph_type"), info.get("anomaly_type"))
        if key in seen_types and len(selected) < max(2, limit // 2):
            continue
        selected.append((record, sample, info))
        seen_types.add(key)
        seen_graphs.add(info.get("graph_type"))
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        chosen = {record.get("sample_id") for record, _, _ in selected}
        for _, _, record, sample, info in scored:
            if record.get("sample_id") in chosen:
                continue
            selected.append((record, sample, info))
            if len(selected) >= limit:
                break
    return selected


def _plot_case(record: Dict[str, Any], sample: Dict[str, Any], info: Dict[str, Any], rank: int, output_dir: Path) -> Dict[str, Any]:
    values = np.asarray(sample["series"]["values"], dtype=float)
    labels = np.asarray(sample["series"]["labels"], dtype=float)
    channels = sample["channels"]
    active = [i for i, ch in enumerate(channels) if ch.get("active", True)] or list(range(values.shape[1]))
    channel_ids = [channels[i]["channel_id"] for i in active]
    truth_interval = true_anomaly_interval(sample)
    proposal_interval = _as_interval((record.get("comparison") or {}).get("proposal_interval"))
    interval = truth_interval or proposal_interval or (0, values.shape[0])

    truth_fact = (record.get("target_output") or {}).get("fact_check") or {}
    pred_fact = (record.get("parsed_response") or {}).get("fact_check") or {}
    truth_root = truth_fact.get("root_cause_channel")
    pred_root = pred_fact.get("root_cause_channel")
    truth_affected = set(truth_fact.get("affected_channels") or [])
    pred_affected = set(pred_fact.get("affected_channels") or [])
    comparison = record.get("comparison") or {}

    fig = plt.figure(figsize=(16, 8.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1.0], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[0, 1])

    x = np.arange(values.shape[0])
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(active))))
    offsets = np.arange(len(active), dtype=float) * 3.1
    ytick_positions = []
    for row_idx, (ch_idx, ch_id) in enumerate(zip(active, channel_ids)):
        series = values[:, ch_idx]
        std = float(np.std(series)) or 1.0
        z = (series - float(np.mean(series))) / std
        offset = offsets[row_idx]
        truth_focus = ch_id == truth_root or ch_id in truth_affected
        pred_focus = ch_id == pred_root or ch_id in pred_affected
        linestyle = "-" if truth_focus else ("--" if pred_focus else "-")
        linewidth = 2.5 if truth_focus or pred_focus else 1.0
        alpha = 0.96 if truth_focus or pred_focus else 0.42
        ax.plot(x, z + offset, color=colors[row_idx % len(colors)], linewidth=linewidth, linestyle=linestyle, alpha=alpha)
        abnormal_idx = np.flatnonzero(labels[:, ch_idx] > 0)
        if len(abnormal_idx):
            ax.scatter(abnormal_idx, z[abnormal_idx] + offset, s=12, color="crimson", alpha=0.9, zorder=5)
        ytick_positions.append(offset)

    if proposal_interval:
        ax.axvspan(proposal_interval[0], proposal_interval[1], color="#4C78A8", alpha=0.16, label="target/proposal interval")
    if truth_interval:
        ax.axvspan(truth_interval[0], truth_interval[1], color="#F58518", alpha=0.24, label="truth anomaly interval")
        ax.axvline(truth_interval[0], color="#B35C00", linestyle="--", linewidth=1.4)
        ax.axvline(truth_interval[1], color="#B35C00", linestyle="--", linewidth=1.4)
    ax.set_xlim(0, values.shape[0] - 1)
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(channel_ids)
    ax.set_xlabel("time step")
    ax.set_ylabel("channels, z-normalized with vertical offsets")
    ax.grid(True, axis="x", alpha=0.24)
    ax.legend(loc="upper right", fontsize=8)

    corr = _window_corr(values[:, active], int(interval[0]), int(interval[1]))
    im = ax_hm.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="equal")
    ax_hm.set_title(f"Window correlation\nmean |r|={info['mean_abs_corr']:.2f}", fontsize=10)
    ax_hm.set_xticks(range(len(channel_ids)))
    ax_hm.set_yticks(range(len(channel_ids)))
    ax_hm.set_xticklabels(channel_ids, rotation=90, fontsize=7)
    ax_hm.set_yticklabels(channel_ids, fontsize=7)
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    qscore = _quality(record)
    title = (
        f"Complex relation #{rank} | {record.get('sample_id')} | graph={info.get('graph_type')}/{info.get('mechanism')} | "
        f"scope/type={truth_fact.get('anomaly_scope')}/{truth_fact.get('anomaly_type')} | quality={qscore:.2f}/4\n"
        f"truth root={truth_root}, pred root={pred_root}; truth affected={_fmt_list(truth_fact.get('affected_channels'))}; "
        f"pred affected={_fmt_list(pred_fact.get('affected_channels'))}"
    )
    fig.suptitle(title, fontsize=12)

    question = str(sample.get("question") or "").replace("\n", " ")
    answer = str((record.get("parsed_response") or {}).get("question_answer") or record.get("raw_response") or "")
    metrics = (
        f"metrics: anomaly={comparison.get('llm_anomaly_match')} root={comparison.get('llm_root_match')} "
        f"type={comparison.get('llm_type_match')} affected_f1={float(comparison.get('llm_affected_f1', 0.0)):.2f} "
        f"IoU={float(comparison.get('proposal_temporal_iou', 0.0)):.2f}; "
        f"complexity: diversity={info['diversity']:.2f}, mean_abs_corr={info['mean_abs_corr']:.2f}"
    )
    footer = "\n".join(
        [
            metrics,
            "Q: " + shorten(question, width=185, placeholder="..."),
            "Qwen: " + shorten(" ".join(answer.split()), width=185, placeholder="..."),
        ]
    )
    fig.subplots_adjust(bottom=0.18, top=0.86)
    fig.text(0.02, 0.03, "\n".join(wrap(footer, width=185)), fontsize=9, va="bottom", family="monospace")

    name = f"complex_{rank:02d}_{record.get('sample_id')}_{info.get('graph_type')}_{truth_fact.get('anomaly_type')}.png".replace("/", "_")
    path = output_dir / name
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "rank": rank,
        "sample_id": record.get("sample_id"),
        "path": str(path),
        "complexity": info,
        "quality": qscore,
        "truth_interval": list(truth_interval) if truth_interval else None,
        "proposal_interval": list(proposal_interval) if proposal_interval else None,
        "truth_root": truth_root,
        "pred_root": pred_root,
        "truth_type": truth_fact.get("anomaly_type"),
        "pred_type": pred_fact.get("anomaly_type"),
        "truth_affected": list(truth_fact.get("affected_channels") or []),
        "pred_affected": list(pred_fact.get("affected_channels") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale_nocf_schema.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    samples = _sample_map(config, args.split)
    records = read_jsonl(Path(relative_to_root(ROOT, args.records)))
    selected = _choose_cases(records, samples, int(args.limit))
    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = [
        _plot_case(record, sample, info, idx, out_dir)
        for idx, (record, sample, info) in enumerate(selected, start=1)
    ]
    save_json({"records": manifest}, out_dir / "manifest.json")
    print(json.dumps({"output_dir": str(out_dir), "num_figures": len(manifest), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
