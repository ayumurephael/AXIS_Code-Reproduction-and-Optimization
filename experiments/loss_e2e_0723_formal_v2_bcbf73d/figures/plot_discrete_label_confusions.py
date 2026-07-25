"""Render publication-quality AXIS discrete-label confusion matrices.

The figure contains TF and MC rows and Released/Control/Treatment columns.
Each standard confusion-matrix cell reports the count and its fraction of the
full reference-label support.  Unparseable predictions are reported in a side
annotation so the displayed matrix remains the conventional 2x2 or 4x4 form.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


MODEL_ORDER = ("released", "control", "treatment")
MODEL_TITLES = {
    "released": "Released",
    "control": "Control",
    "treatment": "Treatment",
}
QUESTION_TYPES = (
    ("true_false", "TF"),
    ("multiple_choice", "MC"),
)


def _matrix(metrics: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = list(metrics["labels"])
    counts = np.array(
        [
            [
                metrics["confusion_counts_parseable"][truth][prediction]
                for prediction in labels
            ]
            for truth in labels
        ],
        dtype=float,
    )
    fractions = np.array(
        [
            [
                metrics["confusion_row_fraction_full_support"][truth][prediction]
                for prediction in labels
            ]
            for truth in labels
        ],
        dtype=float,
    )
    return labels, counts, fractions


def _annotate_cells(
    ax: mpl.axes.Axes,
    counts: np.ndarray,
    fractions: np.ndarray,
    maximum: float,
) -> None:
    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            count = int(counts[row, column])
            fraction = 100.0 * float(fractions[row, column])
            color = "white" if counts[row, column] > maximum * 0.55 else "#1A1A1A"
            ax.text(
                column,
                row,
                f"{count}\n{fraction:.1f}%",
                ha="center",
                va="center",
                fontsize=8.0,
                color=color,
                linespacing=1.15,
            )


def _unparseable_text(metrics: dict) -> str:
    total = int(metrics["unparseable"])
    rate = 100.0 * float(metrics["unparseable_rate"])
    by_label = metrics["unparseable_by_reference_label"]
    detail = ", ".join(f"{label}={int(by_label[label])}" for label in metrics["labels"])
    return f"U: {total} ({rate:.1f}%)\n{detail}"


def render(summary: dict, output_dir: Path) -> tuple[Path, Path]:
    missing = [name for name in MODEL_ORDER if name not in summary["models"]]
    if missing:
        raise ValueError(f"summary is missing models: {missing}")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    # Sequential ramp anchored on the Okabe-Ito blue.
    cmap = LinearSegmentedColormap.from_list(
        "axis_blue",
        ["#FFFFFF", "#D7EBF4", "#56B4E9", "#0072B2", "#003B5C"],
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(11.4, 7.6),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.82, 1.18]},
    )
    row_images: list[mpl.image.AxesImage] = []
    for row_index, (question_type, short_title) in enumerate(QUESTION_TYPES):
        row_maximum = max(
            max(
                max(values.values())
                for values in summary["models"][model]["by_question_type"][
                    question_type
                ]["confusion_counts_parseable"].values()
            )
            for model in MODEL_ORDER
        )
        for column_index, model in enumerate(MODEL_ORDER):
            ax = axes[row_index, column_index]
            metrics = summary["models"][model]["by_question_type"][question_type]
            labels, counts, fractions = _matrix(metrics)
            image = ax.imshow(
                counts,
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=max(row_maximum, 1),
                aspect="equal",
            )
            if column_index == 2:
                row_images.append(image)
            _annotate_cells(ax, counts, fractions, max(row_maximum, 1))
            ax.set_xticks(np.arange(len(labels)), labels=labels)
            ax.set_yticks(np.arange(len(labels)), labels=labels)
            ax.set_xlabel("Predicted label")
            if column_index == 0:
                ax.set_ylabel(f"{short_title}: reference label")
            if row_index == 0:
                ax.set_title(MODEL_TITLES[model], fontweight="semibold")
            ax.set_xticks(
                np.arange(-0.5, len(labels), 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, len(labels), 1),
                minor=True,
            )
            ax.grid(which="minor", color="#D0D0D0", linewidth=0.55)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.text(
                1.02,
                0.02,
                _unparseable_text(metrics),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7.3,
                color="#3A3A3A",
                linespacing=1.25,
            )

    for row_index, image in enumerate(row_images):
        colorbar = fig.colorbar(
            image,
            ax=axes[row_index, :],
            location="right",
            shrink=0.78,
            pad=0.08,
        )
        colorbar.set_label("Number of parseable predictions")

    fig.suptitle(
        "Discrete-label confusion matrices on the full 284-QA test set",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        -0.012,
        "Cells show count and percentage of the full reference-label support; "
        "U denotes an unparseable generated label and is reported separately.",
        ha="center",
        va="top",
        fontsize=8.3,
        color="#3A3A3A",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "discrete_label_confusion_matrices_600dpi.png"
    pdf_path = output_dir / "discrete_label_confusion_matrices.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    png, pdf = render(summary, Path(args.output_dir))
    print(json.dumps({"png": str(png), "pdf": str(pdf)}, indent=2))


if __name__ == "__main__":
    main()
