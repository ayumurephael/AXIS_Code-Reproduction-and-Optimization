from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.MultiAXIS.ablation import (
    ABLATION_VARIANTS,
    DISPLAY_NAMES,
    FULL_VARIANT,
)
from tools.multi_axis.aggregate_geval import COLUMNS


DEFAULT_JUDGE = "qwen3-30b-a3b-instruct-2507"
DEFAULT_DATASET = "478new"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _format(value: float | None) -> str:
    return "?" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate the fixed Epoch-17 inference-time ablation table"
    )
    parser.add_argument("--study-root", required=True)
    parser.add_argument(
        "--baseline",
        default="experiments/multi_axis/ablation_epoch17_qwen30b_baseline.json",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    root = Path(args.study_root)
    baseline = _load(Path(args.baseline))
    if baseline.get("variant") != FULL_VARIANT:
        raise RuntimeError("Baseline provenance is not the full Multi-AXIS variant")
    if baseline.get("provenance", {}).get("judge") != args.judge:
        raise RuntimeError("Baseline Judge identity differs from the requested Judge")
    if baseline.get("provenance", {}).get("dataset") != args.dataset:
        raise RuntimeError("Baseline dataset identity differs from the requested dataset")

    rows = {FULL_VARIANT: dict(baseline["metrics"])}
    labels = {}
    for variant in ABLATION_VARIANTS:
        if variant == FULL_VARIANT:
            continue
        aggregate = _load(root / variant / "aggregate" / "geval_10metrics.json")
        if aggregate.get("judges") != [args.judge] or aggregate.get("datasets") != [
            args.dataset
        ]:
            raise RuntimeError(f"{variant}: aggregate Judge/dataset mismatch")
        table = aggregate["by_judge_dataset"][args.judge][args.dataset]
        if table.get("counts") != {"MC": 176, "OE": 139, "TF": 163}:
            raise RuntimeError(f"{variant}: unexpected question-type counts")
        rows[variant] = {column: float(table[column]) for column in COLUMNS}
        label_path = root / variant / "label_metrics" / "label_metrics_summary.json"
        if label_path.is_file():
            labels[variant] = _load(label_path)

    if tuple(rows) != ABLATION_VARIANTS:
        raise RuntimeError("Ablation variants are missing or out of predeclared order")
    baseline_metrics = rows[FULL_VARIANT]
    deltas = {}
    for variant, metrics in rows.items():
        deltas[variant] = {}
        for column in COLUMNS:
            absolute = metrics[column] - baseline_metrics[column]
            relative = 100.0 * absolute / baseline_metrics[column]
            deltas[variant][column] = {
                "absolute": absolute,
                "relative_percent": relative,
            }

    output = {
        "judge": args.judge,
        "dataset": args.dataset,
        "baseline_provenance": baseline["provenance"],
        "variant_order": list(ABLATION_VARIANTS),
        "metrics": rows,
        "delta_vs_multi_axis": deltas,
        "label_metrics": labels,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_10metrics.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Multi-AXIS Epoch-17 inference-time ablation",
        "",
        f"Judge: `{args.judge}`; dataset: `{args.dataset}`.",
        "",
        "| Variant | " + " | ".join(COLUMNS) + " |",
        "|---|" + "---:|" * len(COLUMNS),
    ]
    for variant in ABLATION_VARIANTS:
        lines.append(
            "| "
            + DISPLAY_NAMES[variant]
            + " | "
            + " | ".join(_format(rows[variant][column]) for column in COLUMNS)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Absolute change versus Multi-AXIS",
            "",
            "| Variant | " + " | ".join(COLUMNS) + " |",
            "|---|" + "---:|" * len(COLUMNS),
        ]
    )
    for variant in ABLATION_VARIANTS[1:]:
        lines.append(
            "| "
            + DISPLAY_NAMES[variant]
            + " | "
            + " | ".join(
                f"{deltas[variant][column]['absolute']:+.4f}" for column in COLUMNS
            )
            + " |"
        )
    (output_dir / "ablation_10metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"variants": len(rows), "complete": True}))


if __name__ == "__main__":
    main()
