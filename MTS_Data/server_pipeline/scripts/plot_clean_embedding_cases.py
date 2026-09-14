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


def _sample_map(config: Dict[str, Any], split: str) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(Path(relative_to_root(ROOT, config["data"]["output_dir"])) / f"{split}.jsonl")
    return {str(row["sample_id"]): row for row in rows}


def _quality(record: Dict[str, Any]) -> float:
    c = record.get("comparison") or {}
    return (
        float(c.get("llm_anomaly_match", 0.0))
        + float(c.get("llm_root_match", 0.0))
        + float(c.get("llm_type_match", 0.0))
        + float(c.get("llm_affected_f1", 0.0))
    )


def _as_interval(value: Any) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _fmt_list(values: Iterable[Any] | None) -> str:
    values = list(values or [])
    return ",".join(str(x) for x in values) if values else "none"


def _choose_cases(
    records: List[Dict[str, Any]],
    samples: Dict[str, Dict[str, Any]],
    num_good: int,
    num_bad: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    scored = []
    for record in records:
        sample = samples.get(str(record.get("sample_id")))
        if not sample:
            continue
        truth = true_anomaly_interval(sample)
        if truth is None:
            continue
        scored.append((_quality(record), int(record.get("index", 0)), record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    good = [("good", record) for _, _, record in scored[:num_good]]
    bad = [("bad", record) for _, _, record in sorted(scored, key=lambda item: (item[0], item[1]))[:num_bad]]
    return good + bad


def _plot_case(
    record: Dict[str, Any],
    sample: Dict[str, Any],
    category: str,
    rank: int,
    output_dir: Path,
) -> Dict[str, Any]:
    values = np.asarray(sample["series"]["values"], dtype=float)
    labels = np.asarray(sample["series"]["labels"], dtype=float)
    channels = sample["channels"]
    active_indices = [i for i, ch in enumerate(channels) if ch.get("active", True)]
    active_indices = active_indices or list(range(values.shape[1]))
    channel_ids = [channels[i]["channel_id"] for i in active_indices]

    comparison = record.get("comparison") or {}
    truth_interval = _as_interval(comparison.get("truth_interval")) or true_anomaly_interval(sample)
    proposal_interval = _as_interval(comparison.get("proposal_interval"))
    target_interval = (
        int(sample.get("target_interval", {}).get("start", 0)),
        int(sample.get("target_interval", {}).get("end", values.shape[0])),
    )

    truth_fact = (record.get("target_output") or {}).get("fact_check") or {}
    pred_fact = (record.get("parsed_response") or {}).get("fact_check") or {}
    truth_root = truth_fact.get("root_cause_channel")
    pred_root = pred_fact.get("root_cause_channel")
    truth_affected = set(truth_fact.get("affected_channels") or [])
    pred_affected = set(pred_fact.get("affected_channels") or [])

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=False)
    x = np.arange(values.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(active_indices))))
    offsets = np.arange(len(active_indices), dtype=float) * 3.2
    ytick_positions = []

    for row_idx, (ch_idx, ch_id) in enumerate(zip(active_indices, channel_ids)):
        series = values[:, ch_idx]
        std = float(np.std(series)) or 1.0
        z = (series - float(np.mean(series))) / std
        offset = offsets[row_idx]
        is_truth_focus = ch_id == truth_root or ch_id in truth_affected
        is_pred_focus = ch_id == pred_root or ch_id in pred_affected
        linewidth = 2.4 if is_truth_focus or is_pred_focus else 1.0
        alpha = 0.95 if is_truth_focus or is_pred_focus else 0.45
        ax.plot(x, z + offset, color=colors[row_idx % len(colors)], linewidth=linewidth, alpha=alpha)
        if labels[:, ch_idx].max() > 0:
            label_idx = np.flatnonzero(labels[:, ch_idx] > 0)
            ax.scatter(label_idx, z[label_idx] + offset, s=11, color="crimson", alpha=0.9, zorder=4)
        ytick_positions.append(offset)

    if proposal_interval:
        ax.axvspan(proposal_interval[0], proposal_interval[1], color="#4C78A8", alpha=0.16, label="target/proposal interval")
        ax.axvline(proposal_interval[0], color="#4C78A8", linewidth=1.3)
        ax.axvline(proposal_interval[1], color="#4C78A8", linewidth=1.3)
    if truth_interval:
        ax.axvspan(truth_interval[0], truth_interval[1], color="#F58518", alpha=0.22, label="truth anomaly interval")
        ax.axvline(truth_interval[0], color="#B35C00", linestyle="--", linewidth=1.4)
        ax.axvline(truth_interval[1], color="#B35C00", linestyle="--", linewidth=1.4)
    else:
        ax.axvspan(target_interval[0], target_interval[1], color="#4C78A8", alpha=0.12, label="target interval")

    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(channel_ids)
    ax.set_xlabel("time step")
    ax.set_ylabel("channels, z-normalized with vertical offsets")
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_xlim(0, values.shape[0] - 1)

    score = _quality(record)
    title = (
        f"{category.upper()} #{rank} | {record.get('sample_id')} | quality={score:.2f}/4 | "
        f"truth root/type={truth_root}/{truth_fact.get('anomaly_type')} | "
        f"pred root/type={pred_root}/{pred_fact.get('anomaly_type')}"
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9)

    question = str(sample.get("question") or "").replace("\n", " ")
    metrics = (
        f"anomaly_match={comparison.get('llm_anomaly_match')}, "
        f"root_match={comparison.get('llm_root_match')}, "
        f"type_match={comparison.get('llm_type_match')}, "
        f"affected_f1={float(comparison.get('llm_affected_f1', 0.0)):.2f}, "
        f"IoU={float(comparison.get('proposal_temporal_iou', 0.0)):.2f}"
    )
    labels_text = (
        f"truth affected: {_fmt_list(truth_fact.get('affected_channels'))} | "
        f"pred affected: {_fmt_list(pred_fact.get('affected_channels'))}"
    )
    answer = str((record.get("parsed_response") or {}).get("question_answer") or record.get("raw_response") or "")
    footer = "\n".join(
        [
            metrics,
            labels_text,
            "Q: " + shorten(question, width=170, placeholder="..."),
            "Qwen: " + shorten(" ".join(answer.split()), width=170, placeholder="..."),
        ]
    )
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.02, 0.03, "\n".join(wrap(footer, width=170)), fontsize=9, va="bottom", family="monospace")

    filename = f"{category}_{rank:02d}_{record.get('sample_id')}.png".replace("/", "_")
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return {
        "category": category,
        "rank": rank,
        "sample_id": record.get("sample_id"),
        "quality": score,
        "truth_interval": list(truth_interval) if truth_interval else None,
        "proposal_interval": list(proposal_interval) if proposal_interval else None,
        "truth_root": truth_root,
        "pred_root": pred_root,
        "truth_type": truth_fact.get("anomaly_type"),
        "pred_type": pred_fact.get("anomaly_type"),
        "metrics": comparison,
        "path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale_nocf_schema.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-good", type=int, default=3)
    parser.add_argument("--num-bad", type=int, default=3)
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    records = read_jsonl(Path(relative_to_root(ROOT, args.records)))
    samples = _sample_map(config, args.split)
    out_dir = Path(relative_to_root(ROOT, args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _choose_cases(records, samples, int(args.num_good), int(args.num_bad))
    manifest = []
    category_counts = {"good": 0, "bad": 0}
    for category, record in selected:
        category_counts[category] += 1
        sample = samples[str(record["sample_id"])]
        manifest.append(_plot_case(record, sample, category, category_counts[category], out_dir))
    save_json({"records": manifest}, out_dir / "manifest.json")
    print(json.dumps({"output_dir": str(out_dir), "num_figures": len(manifest), "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
