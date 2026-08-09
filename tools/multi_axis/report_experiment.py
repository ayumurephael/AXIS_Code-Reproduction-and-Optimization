from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

from tools.multi_axis.aggregate_geval import COLUMNS, DATASETS, JUDGES


SECTION_NAMES = {
    "478new": "Teacher-Eval 478new",
    "SMD": "SMD（",
    "SWaT": "SWaT（",
    "LEMMA-RCA": "LEMMA-RCA（",
    "VTA": "VTA Articulary（",
}


def clean_cell(value: str) -> str:
    value = html.unescape(re.sub(r"<[^>]+>", "", value))
    return value.replace("**", "").strip()


def parse_baselines(path: Path):
    text = path.read_text(encoding="utf-8")
    output = {}
    for dataset, heading in SECTION_NAMES.items():
        start = text.index("## " + heading)
        end = text.find("\n## ", start + 4)
        section = text[start:] if end < 0 else text[start:end]
        table_lines = [line for line in section.splitlines() if line.startswith("|")]
        if len(table_lines) < 3:
            raise RuntimeError(f"Could not locate baseline table for {dataset}")
        header = [clean_cell(cell) for cell in table_lines[0].strip("|").split("|")]
        expected_header = [
            "MC-Final",
            "MC-Corr",
            "MC-Rsn",
            "OE-Final",
            "OE-Acc",
            "OE-Comp",
            "OE-Rel",
            "TF-Final",
            "TF-Corr",
            "TF-Justif",
        ]
        if header[1:11] != expected_header:
            raise RuntimeError(
                f"Unexpected baseline metric columns for {dataset}: {header[1:11]}"
            )
        rows = {}
        for line in table_lines[2:]:
            cells = [clean_cell(cell) for cell in line.strip("|").split("|")]
            if len(cells) < 11:
                continue
            model = cells[0]
            values = {}
            for metric_index, column in enumerate(COLUMNS, 1):
                match = re.search(r"-?\d+(?:\.\d+)?", cells[metric_index])
                if not match:
                    raise RuntimeError(f"Missing {column} for {dataset}/{model}")
                values[column] = float(match.group())
            rows[model] = values
        output[dataset] = rows
    return output


def fmt(value):
    return "?" if value is None else f"{value:.4f}"


def ten_metric_table(rows: Sequence[tuple[str, dict]]):
    lines = [
        "| Scope | " + " | ".join(COLUMNS) + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    for name, row in rows:
        lines.append(
            "| "
            + name
            + " | "
            + " | ".join(fmt(row[column]) for column in COLUMNS)
            + " |"
        )
    return "\n".join(lines)


def baseline_outputs(baselines, new_tables, output_dir: Path):
    records = []
    md = [
        "# GPT-5.4 baseline comparison",
        "",
        "Relative change is `(Multi-AXIS - baseline) / baseline × 100%`; all scores use the raw 1–5 scale.",
    ]
    for dataset in DATASETS:
        md.extend(
            [
                "",
                f"## {dataset}",
                "",
                "| Baseline | Metric | Baseline | Multi-AXIS | Absolute Δ | Relative Δ |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        new = new_tables[dataset]
        for model, values in baselines[dataset].items():
            for column in COLUMNS:
                baseline = values[column]
                delta = new[column] - baseline
                relative = delta / baseline * 100.0 if baseline else None
                record = {
                    "dataset": dataset,
                    "baseline_model": model,
                    "metric": column,
                    "baseline": baseline,
                    "multi_axis": new[column],
                    "absolute_delta": delta,
                    "relative_change_percent": relative,
                }
                records.append(record)
                md.append(
                    f"| {model} | {column} | {baseline:.4f} | {new[column]:.4f} | {delta:+.4f} | {relative:+.2f}% |"
                )
    with (output_dir / "baseline_deltas.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "baseline_deltas.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    return records


def load_optional(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-tables", required=True)
    parser.add_argument("--geval-summary", required=True)
    parser.add_argument("--label-summary", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--geval-results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geval = json.loads(Path(args.geval_summary).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.label_summary).read_text(encoding="utf-8"))
    baselines = parse_baselines(Path(args.baseline_tables))
    gpt_tables = geval["by_judge_dataset"]["gpt-5.4"]
    delta_records = baseline_outputs(baselines, gpt_tables, output_dir)
    run_dir, manifest_dir, inference_dir, judge_root = map(
        Path,
        (args.run_dir, args.manifest_dir, args.inference_dir, args.geval_results_root),
    )
    run_manifest = load_optional(run_dir / "run_manifest.json") or {}
    split = load_optional(manifest_dir / "train_split_summary.json") or {}
    best = load_optional(run_dir / "best_checkpoint.json") or {}
    epochs = read_jsonl(run_dir / "epochs.jsonl")

    configured_epochs = (
        run_manifest.get("config", {}).get("training", {}).get("epochs", "unknown")
    )
    lines = [
        "# Multi-AXIS-VL formal experiment analysis",
        "",
        "## Outcome",
        "",
        f"The formal student is Qwen3-VL-8B-Instruct with one native time-series image, the exact numeric Window, and Step/Joint/Fixed AXIS hints. TimeRCD and the complete VLM remain frozen in eval mode; only the documented Hint Tuner parameters are optimized for all {configured_epochs} configured epochs.",
        "",
        "### GPT-5.4 results used for baseline comparison",
        "",
        ten_metric_table(
            [(dataset, gpt_tables[dataset]) for dataset in DATASETS]
            + [
                ("Micro", geval["micro"]["gpt-5.4"]),
                ("Macro", geval["macro"]["gpt-5.4"]),
            ]
        ),
        "",
        "### Five-Judge micro summary",
        "",
        ten_metric_table([(judge, geval["micro"][judge]) for judge in JUDGES]),
        "",
        "### Five-Judge macro summary",
        "",
        ten_metric_table([(judge, geval["macro"][judge]) for judge in JUDGES]),
    ]
    label_overall = labels["overall"]
    lines.extend(
        [
            "",
            "## Exact-label and parseability metrics",
            "",
            f"- MC exact-match accuracy: {label_overall['mc_exact_match_accuracy']:.4%}",
            f"- TF exact-match accuracy: {label_overall['tf_exact_match_accuracy']:.4%}",
            f"- OE parse rate: {label_overall['oe_parse_rate']:.4%}",
            f"- Combined task success rate: {label_overall['combined_task_success_rate']:.4%}",
            "",
            "Dataset-level and question-type-level values are preserved in `label_metrics_summary.json`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Training and selection audit",
            "",
            f"- Branch/commit: `{run_manifest.get('git', {}).get('branch', 'unknown')}` / `{run_manifest.get('git', {}).get('commit', 'unknown')}`; dirty at launch: `{run_manifest.get('git', {}).get('dirty', 'unknown')}`.",
            f"- Training examples / validation examples: {split.get('train_examples')} / {split.get('validation_examples')}; filtered empty teacher answers: {split.get('empty_teacher_answers_filtered')}.",
            f"- Split: grouped by `base_sample_id`, seed={split.get('seed')}, validation fraction={split.get('validation_fraction')}, overlap={split.get('group_overlap')}.",
            f"- Completed epochs: {best.get('completed_epochs')}; selected epoch: {best.get('best_epoch')}; best validation answer NLL: {best.get('best_value')}.",
            f"- Actual effective batch size: {run_manifest.get('actual_effective_batch_size')}; world size: {run_manifest.get('world_size')}; precision: BF16.",
            f"- TimeRCD SHA-256: `{run_manifest.get('timercd_sha256')}`.",
            f"- Trainable parameters: {run_manifest.get('trainable_parameter_count')}; VLM/TimeRCD parameters remained frozen and the VLM stayed in eval mode while autograd propagated to the Hint Tuner.",
            f"- Modality: {run_manifest.get('modality')}; official Qwen3-VL pixel budget: {json.dumps(run_manifest.get('qwen3_vl_pixel_budget', {}), ensure_ascii=False)}.",
            f"- FlashAttention runtime audit: {json.dumps(run_manifest.get('flash_attention_audit', {}), ensure_ascii=False)}.",
        ]
    )
    if epochs:
        lines.extend(
            [
                f"- First/final train answer NLL: {epochs[0]['train_answer_nll']:.6f} / {epochs[-1]['train_answer_nll']:.6f}.",
                f"- First/final validation answer NLL: {epochs[0]['validation_answer_nll']:.6f} / {epochs[-1]['validation_answer_nll']:.6f}.",
            ]
        )
    lines.extend(["", "## Judge audit", ""])
    for judge in JUDGES:
        methods = {}
        task_count = 0
        for dataset in DATASETS:
            rows = read_jsonl(judge_root / judge / f"{dataset}.jsonl")
            task_count += len(rows)
            for row in rows:
                methods[row["method"]] = methods.get(row["method"], 0) + 1
        lines.append(
            f"- {judge}: {task_count} dimension calls, methods={json.dumps(methods, ensure_ascii=False)}."
        )
    lines.extend(
        [
            "",
            "## Baseline comparison",
            "",
            "The complete per-dataset × baseline-model × metric absolute and relative changes are in `baseline_deltas.md` and `baseline_deltas.csv`. Only GPT-5.4 is compared with baseline scores; the other four Judges are used solely for cross-Judge robustness, as pre-registered.",
        ]
    )
    best_comparisons = []
    for dataset in DATASETS:
        for column in COLUMNS:
            model, value = max(
                (
                    (model, metrics[column])
                    for model, metrics in baselines[dataset].items()
                ),
                key=lambda item: item[1],
            )
            new = gpt_tables[dataset][column]
            best_comparisons.append(
                (
                    dataset,
                    column,
                    model,
                    value,
                    new,
                    new - value,
                    (new - value) / value * 100.0,
                )
            )
    lines.extend(
        [
            "",
            "### Versus the strongest listed baseline per metric",
            "",
            "| Dataset | Metric | Strongest baseline | Baseline | Multi-AXIS | Absolute Δ | Relative Δ |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset, column, model, value, new, delta, relative in best_comparisons:
        lines.append(
            f"| {dataset} | {column} | {model} | {value:.4f} | {new:.4f} | {delta:+.4f} | {relative:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Cross-Judge robustness and future work",
            "",
            "Per-metric mean, population standard deviation, minimum, maximum, and range across the five Judges are stored in `geval_10metrics.json`. Large Judge ranges should be treated as evaluator sensitivity rather than model gain. Future changes should target the weakest dataset/type cells, rerun the registered A/B/C/D ablations, and retain grouped validation NLL for checkpoint selection; the test/Judge scores must not be used to select checkpoints.",
            "",
            "## Artifact inventory",
            "",
            "- `run_manifest.json`: code, hardware, configuration, trainable parameter list, and TimeRCD hash.",
            "- `train_split_summary.json`: input hashes, pairing/filter counts, group split, and type counts.",
            "- `epochs.jsonl` and `train_steps.jsonl`: intermediate loss/LR/gradient records.",
            "- `inference_manifest.json` and per-dataset prediction files: generation protocol and raw outputs.",
            "- Per-Judge JSONL: raw judge responses, score method/distribution, usage, fingerprint, latency, and prompt hash.",
            "- `label_metrics_summary.json`: overall, type, and dataset exact-label/parseability results.",
            "- `geval_10metrics.json`: all dataset × Judge tables plus micro/macro and robustness statistics.",
        ]
    )
    (output_dir / "EXPERIMENT_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
