"""Build separate Table-I and delta tables for the three locked Judges."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.reproduction.prompt_round9_paper140_multi_judge_20260728.src.paper140_multi_judge import (
    MODES,
)
from tools.axis_repro.common import read_jsonl
from tools.axis_repro.table_runner import aggregate


JUDGE_ORDER = ("deepseek-v4-pro", "qwen3.5-397b-a17b", "qwen3-30b-a3b-instruct-2507")
METRICS = (
    ("MC Final", "multiple_choice/final"),
    ("MC Corr.", "multiple_choice/correctness"),
    ("MC Rsn.", "multiple_choice/reasoning_quality"),
    ("OE Final", "open_ended/final"),
    ("OE Acc.", "open_ended/accuracy"),
    ("OE Comp.", "open_ended/completeness"),
    ("OE Rel.", "open_ended/relevance"),
    ("TF Final", "true_false/final"),
    ("TF Corr.", "true_false/correctness"),
    ("TF Justif.", "true_false/justification_quality"),
)
DISPLAY = {
    "base": "AXIS Baseline",
    "prompt_final_round9_mc_oe": "Round 9 MC+OE",
    "prompt_final_round9_tf_oe": "Round 9 TF+OE",
    "prompt_final_round9_joint": "Round 9 Joint",
}


def parse_score_map(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected JUDGE=PATH, got {value!r}")
        result[label] = Path(path)
    if set(result) != set(JUDGE_ORDER):
        raise ValueError(f"Judge map mismatch: {sorted(result)}")
    return result


def table_header(extra: tuple[str, ...] = ()) -> list[str]:
    return ["Prompt", *(label for label, _ in METRICS), *extra]


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def analyze_judge(path: Path) -> dict:
    rows = read_jsonl(path)
    models, paired = aggregate(rows)
    if set(models) != set(MODES):
        raise RuntimeError(f"Mode mismatch in {path}: {sorted(models)}")
    baseline = models["base"]
    modes = {}
    for mode in MODES:
        values = models[mode]
        metrics = {}
        for label, key in METRICS:
            value = values[key]
            base_value = baseline[key]
            delta = value - base_value
            metrics[label] = {
                "value": value,
                "baseline": base_value,
                "absolute_delta": delta,
                "relative_percent": 100.0 * delta / base_value,
                "nonlower_4dp": round(value, 4) >= round(base_value, 4),
                "strict_improvement_4dp": round(value, 4) > round(base_value, 4),
            }
        modes[mode] = {
            "metrics": metrics,
            "all_ten_nonlower_4dp": all(
                metric["nonlower_4dp"] for metric in metrics.values()
            ),
            "strict_improvements_4dp": sum(
                metric["strict_improvement_4dp"] for metric in metrics.values()
            ),
        }
    return {
        "score_path": str(path),
        "score_rows": len(rows),
        "methods": dict(Counter(str(row.get("method")) for row in rows)),
        "models": dict(Counter(str(row.get("model")) for row in rows)),
        "providers": dict(Counter(str(row.get("provider")) for row in rows)),
        "enable_thinking": dict(
            Counter(
                "auto"
                if row.get("enable_thinking") is None
                else str(row.get("enable_thinking")).lower()
                for row in rows
            )
        ),
        "logprobs_returned": sum(bool(row.get("logprobs_returned")) for row in rows),
        "valid_prompt_hashes": sum(
            len(str(row.get("prompt_sha256", ""))) == 64 for row in rows
        ),
        "modes": modes,
        "paired_vs_axis": paired,
    }


def render(payload: dict) -> str:
    lines = []
    for judge in JUDGE_ORDER:
        result = payload["judges"][judge]
        lines.extend((f"## {judge}", "", "### Table-I scores", ""))
        score_rows = []
        for mode in MODES:
            mode_result = result["modes"][mode]
            score_rows.append(
                [
                    DISPLAY[mode],
                    *(
                        f"{mode_result['metrics'][label]['value']:.4f}"
                        for label, _ in METRICS
                    ),
                    (
                        "—"
                        if mode == "base"
                        else str(mode_result["all_ten_nonlower_4dp"])
                    ),
                    (
                        "—"
                        if mode == "base"
                        else str(mode_result["strict_improvements_4dp"])
                    ),
                ]
            )
        lines.extend(
            markdown_table(
                table_header(("10/10 non-lower", "Strict +")),
                score_rows,
            )
        )
        lines.extend(("", "### Absolute delta from this Judge's Baseline", ""))
        delta_rows = []
        for mode in MODES[1:]:
            mode_result = result["modes"][mode]
            delta_rows.append(
                [
                    DISPLAY[mode],
                    *(
                        f"{mode_result['metrics'][label]['absolute_delta']:+.4f}"
                        for label, _ in METRICS
                    ),
                ]
            )
        lines.extend(markdown_table(table_header(), delta_rows))
        lines.extend(("", "### Relative change from this Judge's Baseline", ""))
        relative_rows = []
        for mode in MODES[1:]:
            mode_result = result["modes"][mode]
            relative_rows.append(
                [
                    DISPLAY[mode],
                    *(
                        f"{mode_result['metrics'][label]['relative_percent']:+.4f}%"
                        for label, _ in METRICS
                    ),
                ]
            )
        lines.extend(markdown_table(table_header(), relative_rows))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    score_map = parse_score_map(args.scores)
    payload = {
        "dataset": "paper140",
        "records_per_mode": 140,
        "score_dimensions_per_mode": 335,
        "modes": list(MODES),
        "judges": {
            judge: analyze_judge(score_map[judge]) for judge in JUDGE_ORDER
        },
    }
    Path(args.output_json).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    Path(args.output_md).write_text(render(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                judge: {
                    mode: {
                        "all_ten_nonlower_4dp": result["modes"][mode][
                            "all_ten_nonlower_4dp"
                        ],
                        "strict_improvements_4dp": result["modes"][mode][
                            "strict_improvements_4dp"
                        ],
                    }
                    for mode in MODES[1:]
                }
                for judge, result in payload["judges"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
