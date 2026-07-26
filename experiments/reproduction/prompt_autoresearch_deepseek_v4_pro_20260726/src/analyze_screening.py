"""Aggregate screening scores, rank Pareto candidates, and extract cases."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from tools.axis_repro.common import extract_choice, extract_true_false, read_jsonl
from tools.axis_repro.build_tables import DIMS, ORDER

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


def aggregate_scores(rows):
    dimensions = collections.defaultdict(dict)
    weights = collections.defaultdict(dict)
    qtypes = {}
    notes = collections.defaultdict(dict)
    for row in rows:
        key = (row["record_id"], row["mode"])
        dimensions[key][row["dimension"]] = float(row["score"])
        weights[key][row["dimension"]] = float(row["weight"])
        qtypes[key] = row["question_type"]
        notes[key][row["dimension"]] = row.get("judge_content", "")

    per_metric = collections.defaultdict(lambda: collections.defaultdict(list))
    record_final = {}
    complete_dimensions = {}
    for key, values in dimensions.items():
        qtype = qtypes[key]
        expected = DIMS[qtype]
        if any(dimension not in values for dimension in expected):
            continue
        final = sum(values[d] * weights[key][d] for d in expected)
        record_final[key] = final
        complete_dimensions[key] = values
        per_metric[key[1]][f"{qtype}/final"].append(final)
        for dimension in expected:
            per_metric[key[1]][f"{qtype}/{dimension}"].append(values[dimension])

    models = {
        mode: {metric: sum(values) / len(values) for metric, values in metrics.items()}
        for mode, metrics in per_metric.items()
    }
    return models, record_final, complete_dimensions, notes


def decision(qtype, text):
    if qtype == "multiple_choice":
        return extract_choice(text)
    if qtype == "true_false":
        return extract_true_false(text)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-mode", default="base")
    parser.add_argument("--title", default="Screening round 1 results")
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    scores = read_jsonl(args.scores)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    pred_by_key = {
        (row["record_id"], row["mode"]): row for row in predictions
    }
    models, record_final, dimensions, notes = aggregate_scores(scores)
    baseline = models[args.baseline_mode]
    expected_keys = [key for _, key in METRICS]
    missing = {
        mode: [key for key in expected_keys if key not in metrics]
        for mode, metrics in models.items()
        if any(key not in metrics for key in expected_keys)
    }
    if missing:
        raise RuntimeError(f"Incomplete Table-I metrics: {missing}")

    rankings = []
    case_rows = []
    pair_summary = {}
    for mode, metrics in models.items():
        if mode == args.baseline_mode:
            continue
        deltas = {key: metrics[key] - baseline[key] for key in expected_keys}
        nonnegative = sum(value >= -1e-12 for value in deltas.values())
        worst = min(deltas.values())
        mean_delta = sum(deltas.values()) / len(deltas)
        rankings.append(
            {
                "mode": mode,
                "nonnegative_dimensions": nonnegative,
                "worst_delta": worst,
                "mean_delta": mean_delta,
                "all_nonnegative": nonnegative == len(expected_keys),
                "deltas": deltas,
                "metrics": metrics,
            }
        )

        paired = []
        correct_to_wrong = 0
        wrong_to_correct = 0
        for (record_id, candidate_mode), final in record_final.items():
            if candidate_mode != mode:
                continue
            base_key = (record_id, args.baseline_mode)
            candidate_key = (record_id, mode)
            if base_key not in record_final:
                continue
            base_prediction = pred_by_key[base_key]
            candidate_prediction = pred_by_key[candidate_key]
            delta = final - record_final[base_key]
            qtype = candidate_prediction["question_type"]
            base_decision = decision(qtype, base_prediction["response"])
            candidate_decision = decision(qtype, candidate_prediction["response"])
            gold = decision(qtype, candidate_prediction["answer"])
            if gold is not None:
                base_correct = base_decision == gold
                candidate_correct = candidate_decision == gold
                correct_to_wrong += int(base_correct and not candidate_correct)
                wrong_to_correct += int(not base_correct and candidate_correct)
            row = {
                "record_id": record_id,
                "mode": mode,
                "question_type": qtype,
                "has_anomaly": candidate_prediction.get("has_anomaly"),
                "start_index": candidate_prediction["start_index"],
                "end_index": candidate_prediction["end_index"],
                "question": candidate_prediction["question"],
                "gold_answer": candidate_prediction["answer"],
                "base_response": base_prediction["response"],
                "candidate_response": candidate_prediction["response"],
                "base_final": record_final[base_key],
                "candidate_final": final,
                "delta": delta,
                "base_dimensions": dimensions[base_key],
                "candidate_dimensions": dimensions[candidate_key],
                "base_judge_notes": notes[base_key],
                "candidate_judge_notes": notes[candidate_key],
                "gold_decision": gold,
                "base_decision": base_decision,
                "candidate_decision": candidate_decision,
            }
            paired.append(row)
            case_rows.append(row)
        pair_summary[mode] = {
            "wins": sum(row["delta"] > 1e-12 for row in paired),
            "ties": sum(abs(row["delta"]) <= 1e-12 for row in paired),
            "losses": sum(row["delta"] < -1e-12 for row in paired),
            "correct_to_wrong": correct_to_wrong,
            "wrong_to_correct": wrong_to_correct,
            "worst_cases": [row["record_id"] for row in sorted(paired, key=lambda x: x["delta"])[:5]],
            "best_cases": [row["record_id"] for row in sorted(paired, key=lambda x: -x["delta"])[:3]],
        }

    rankings.sort(
        key=lambda row: (
            row["nonnegative_dimensions"],
            row["worst_delta"],
            row["mean_delta"],
        ),
        reverse=True,
    )
    summary = {
        "baseline_mode": args.baseline_mode,
        "prediction_rows": len(predictions),
        "score_rows": len(scores),
        "models": models,
        "rankings": rankings,
        "pair_summary": pair_summary,
    }
    (output / "screening_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output / "paired_cases.jsonl").open("w", encoding="utf-8") as handle:
        for row in case_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    headers = ["Rank", "Mode", "Nonneg.", "Worst", "Mean"] + [name for name, _ in METRICS]
    lines = [
        f"# {args.title}",
        "",
        "## Delta versus split Baseline",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for index, row in enumerate(rankings, start=1):
        cells = [
            str(index),
            row["mode"],
            f"{row['nonnegative_dimensions']}/10",
            f"{row['worst_delta']:+.3f}",
            f"{row['mean_delta']:+.3f}",
        ] + [f"{row['deltas'][key]:+.3f}" for _, key in METRICS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Absolute scores", ""])
    abs_headers = ["Mode"] + [name for name, _ in METRICS]
    lines.extend([
        "| " + " | ".join(abs_headers) + " |",
        "|" + "---|" * len(abs_headers),
    ])
    ordered_modes = [args.baseline_mode] + [row["mode"] for row in rankings]
    for mode in ordered_modes:
        lines.append(
            "| "
            + " | ".join([mode] + [f"{models[mode][key]:.3f}" for _, key in METRICS])
            + " |"
        )
    lines.extend(["", "## Paired outcome counts", ""])
    for mode in ordered_modes[1:]:
        pair = pair_summary[mode]
        lines.append(
            f"- `{mode}`: wins={pair['wins']}, ties={pair['ties']}, "
            f"losses={pair['losses']}, correct→wrong={pair['correct_to_wrong']}, "
            f"wrong→correct={pair['wrong_to_correct']}."
        )
    (output / "screening_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rankings": rankings, "pair_summary": pair_summary}))


if __name__ == "__main__":
    main()