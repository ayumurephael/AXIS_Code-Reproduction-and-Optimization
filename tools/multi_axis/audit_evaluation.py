from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.models.MultiAXIS.ablation import ABLATION_VARIANTS
from tools.multi_axis.aggregate_geval import JUDGES, TYPE_INFO
from tools.multi_axis.datasets import (
    EXPECTED_DATASET_COUNTS,
    EXPECTED_TYPE_COUNTS,
    SUPPORTED_EVALUATION_DATASETS,
    validate_datasets,
)


def read_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Modality-neutral Multi-AXIS inference and G-Eval audit"
    )
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--judge-input-dir", required=True)
    parser.add_argument("--geval-results-root", required=True)
    parser.add_argument("--label-metrics-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_EVALUATION_DATASETS,
        required=True,
    )
    parser.add_argument("--judges", nargs="+", choices=JUDGES, required=True)
    parser.add_argument("--expected-checkpoint-epoch", type=int)
    parser.add_argument("--expected-mode", choices=ABLATION_VARIANTS)
    args = parser.parse_args()

    datasets = validate_datasets(args.datasets)
    judges = tuple(args.judges)
    if len(judges) != len(set(judges)):
        raise ValueError("Judge names must be unique")
    manifest_dir = Path(args.manifest_dir)
    predictions_dir = Path(args.predictions_dir)
    judge_input_dir = Path(args.judge_input_dir)
    judge_root = Path(args.geval_results_root)
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    inference_manifest = json.loads(
        (predictions_dir / "inference_manifest.json").read_text(encoding="utf-8")
    )
    check(
        "inference_complete",
        (predictions_dir / "INFERENCE_COMPLETE").is_file(),
        str(predictions_dir / "INFERENCE_COMPLETE"),
    )
    check(
        "inference_datasets",
        tuple(inference_manifest.get("datasets", [])) == datasets,
        inference_manifest.get("datasets"),
    )
    check(
        "checkpoint_epoch",
        args.expected_checkpoint_epoch is None
        or inference_manifest.get("hint_checkpoint_epoch")
        == args.expected_checkpoint_epoch,
        inference_manifest.get("hint_checkpoint_epoch"),
    )
    check(
        "checkpoint_hash",
        len(str(inference_manifest.get("hint_checkpoint_sha256") or "")) == 64,
        inference_manifest.get("hint_checkpoint_sha256"),
    )
    check(
        "ablation_mode",
        args.expected_mode is None
        or inference_manifest.get("ablation_variant") == args.expected_mode,
        inference_manifest.get("ablation_variant"),
    )
    check(
        "timercd_hash",
        len(str(inference_manifest.get("timercd_sha256") or "")) == 64,
        inference_manifest.get("timercd_sha256"),
    )
    attention = inference_manifest.get("attention_backend_audit") or {}
    check(
        "flash_attention",
        attention.get("requested") == "flash_attention_2"
        and attention.get("dense_kernel_importable") is True
        and attention.get("varlen_kernel_importable") is True
        and attention.get("fallback_allowed") is False,
        attention,
    )

    for dataset in datasets:
        references = read_jsonl(manifest_dir / f"eval_{dataset}.jsonl")
        counts = dict(collections.Counter(row["question_group"] for row in references))
        check(
            f"{dataset}_manifest_count",
            len(references) == EXPECTED_DATASET_COUNTS[dataset],
            len(references),
        )
        check(
            f"{dataset}_type_counts",
            counts == EXPECTED_TYPE_COUNTS[dataset],
            counts,
        )
        predictions = read_jsonl(predictions_dir / f"{dataset}.predictions.jsonl")
        prediction_ids = [row.get("sample_id") for row in predictions]
        check(
            f"{dataset}_predictions",
            len(predictions) == len(references)
            and [row.get("index") for row in predictions] == list(range(len(references)))
            and prediction_ids == [row.get("sample_id") for row in references],
            {"rows": len(predictions), "unique_sample_ids": len(set(prediction_ids))},
        )
        check(
            f"{dataset}_prediction_modes",
            args.expected_mode is None
            or all(row.get("ablation_variant") == args.expected_mode for row in predictions),
            sorted({row.get("ablation_variant") for row in predictions}, key=str),
        )
        judge_inputs = read_jsonl(judge_input_dir / f"{dataset}.judge_input.jsonl")
        expected_tasks = sum(
            len(TYPE_INFO[row["question_type"]][1]) for row in judge_inputs
        )
        expected_keys = {
            (row["record_id"], dimension)
            for row in judge_inputs
            for dimension in TYPE_INFO[row["question_type"]][1]
        }
        check(
            f"{dataset}_judge_inputs",
            len(judge_inputs) == len(references)
            and [row.get("sample_id") for row in judge_inputs]
            == [row.get("sample_id") for row in references],
            len(judge_inputs),
        )
        check(
            f"{dataset}_judge_input_modes",
            args.expected_mode is None
            or all(row.get("mode") == args.expected_mode for row in judge_inputs),
            sorted({row.get("mode") for row in judge_inputs}, key=str),
        )
        for judge in judges:
            scores = read_jsonl(judge_root / judge / f"{dataset}.jsonl")
            keys = {(row.get("record_id"), row.get("dimension")) for row in scores}
            expected_method = (
                "integer_score" if judge in {"gpt-5.4", "gemini-2.5-pro"} else None
            )
            check(
                f"{dataset}_{judge}_scores",
                len(scores) == expected_tasks
                and len(keys) == expected_tasks
                and keys == expected_keys
                and all(row.get("dataset") == dataset for row in scores)
                and all(1.0 <= float(row["score"]) <= 5.0 for row in scores)
                and all(row.get("prompt_sha256") for row in scores)
                and all(row.get("provider_profile") == judge for row in scores)
                and (
                    args.expected_mode is None
                    or all(row.get("mode") == args.expected_mode for row in scores)
                )
                and all(row.get("requested_model") for row in scores)
                and all(
                    len(str(row.get("endpoint_host_sha256") or "")) == 64
                    for row in scores
                )
                and (
                    judge != "gpt-5.4"
                    or all(
                        len(str(row.get("tls_ca_bundle_sha256") or "")) == 64
                        for row in scores
                    )
                )
                and (
                    expected_method is None
                    or all(row.get("method") == expected_method for row in scores)
                ),
                {"expected": expected_tasks, "rows": len(scores), "unique": len(keys)},
            )
            if judge in {"qwen3.5-397b-a17b", "qwen3-30b-a3b-instruct-2507"}:
                qwen_methods = {
                    "final_score_top_logprobs_bounded",
                    "a_e_json_readout_top_logprobs_bounded",
                }
                check(
                    f"{dataset}_{judge}_bounded_logprobs",
                    all(row.get("method") in qwen_methods for row in scores)
                    and all(row.get("logprobs_requested") is True for row in scores)
                    and all(row.get("top_logprobs") == 5 for row in scores)
                    and all(float((row.get("distribution_metadata") or {}).get("missing_mass_upper_bound", 1.0)) <= 1e-6 for row in scores),
                    {"methods": sorted({row.get("method") for row in scores})},
                )
            completion_path = (
                judge_root / judge / f"{dataset}.jsonl.complete.json"
            )
            completion = (
                json.loads(completion_path.read_text(encoding="utf-8"))
                if completion_path.is_file()
                else {}
            )
            check(
                f"{dataset}_{judge}_complete_marker",
                completion.get("judge") == judge
                and completion.get("tasks") == expected_tasks
                and completion.get("examples") == len(references),
                completion,
            )

    label_summary = json.loads(
        (Path(args.label_metrics_dir) / "label_metrics_summary.json").read_text(
            encoding="utf-8"
        )
    )
    check(
        "label_metric_datasets",
        tuple(label_summary.get("by_dataset", {})) == datasets,
        list(label_summary.get("by_dataset", {})),
    )
    label_rows = read_jsonl(Path(args.label_metrics_dir) / "label_metrics.jsonl")
    label_expected = [
        (dataset, index, row["sample_id"])
        for dataset in datasets
        for index, row in enumerate(
            read_jsonl(manifest_dir / f"eval_{dataset}.jsonl")
        )
    ]
    check(
        "label_metric_rows",
        [
            (row.get("dataset"), row.get("index"), row.get("sample_id"))
            for row in label_rows
        ]
        == label_expected,
        len(label_rows),
    )
    passed = all(item["passed"] for item in checks)
    payload = {
        "passed": passed,
        "mode": args.expected_mode,
        "datasets": list(datasets),
        "judges": list(judges),
        "checks": checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": len(checks)}))
    if not passed:
        raise SystemExit(
            "Evaluation audit failed: "
            + ", ".join(item["name"] for item in checks if not item["passed"])
        )


if __name__ == "__main__":
    main()
