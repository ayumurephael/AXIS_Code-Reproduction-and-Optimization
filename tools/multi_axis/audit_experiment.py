from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.models.MultiAXIS.config import MultiAxisConfig, QWEN3_VL_MODEL_ID
from src.models.MultiAXIS.data import visual_image_id, visual_image_relpath
from src.models.MultiAXIS.timercd import sha256_file
from tools.multi_axis.aggregate_geval import DATASETS, JUDGES, TYPE_INFO


EXPECTED_DATASET_COUNTS = {
    "478new": 478,
    "SMD": 200,
    "SWaT": 184,
    "LEMMA-RCA": 12,
    "VTA": 200,
}
EXPECTED_TYPE_COUNTS = {
    "478new": {"MC": 176, "OE": 139, "TF": 163},
    "SMD": {"MC": 66, "OE": 67, "TF": 67},
    "SWaT": {"MC": 61, "OE": 62, "TF": 61},
    "LEMMA-RCA": {"MC": 4, "OE": 4, "TF": 4},
    "VTA": {"MC": 67, "OE": 67, "TF": 66},
}


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--judge-input-dir", required=True)
    parser.add_argument("--geval-results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checks = []

    def check(name, condition, detail):
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    config = MultiAxisConfig.load_json(args.config)
    check(
        "formal_model",
        config.llm.model_name == QWEN3_VL_MODEL_ID and config.vision.enabled,
        {"model": config.llm.model_name, "vision_enabled": config.vision.enabled},
    )
    check(
        "configured_epochs",
        config.training.epochs == 25
        and config.training.expected_world_size == 8
        and config.training.expected_nodes == 2
        and config.training.effective_batch_size in {32, 48}
        and not config.training.early_stopping,
        config.training.__dict__,
    )
    check(
        "full_architecture",
        config.hints.ablation_phase == "D",
        config.hints.ablation_phase,
    )
    check(
        "flash_attention",
        config.hints.require_flash_attention
        and config.llm.attention_implementation == "flash_attention_2",
        {
            "cross_attention": config.hints.require_flash_attention,
            "llm": config.llm.attention_implementation,
        },
    )
    split = json.loads(
        (Path(args.manifest_dir) / "train_split_summary.json").read_text(
            encoding="utf-8"
        )
    )
    check(
        "empty_teacher_filter",
        split["empty_teacher_answers_filtered"] == 17,
        split["empty_teacher_answers_filtered"],
    )
    check(
        "group_split",
        split["seed"] == 42
        and split["validation_fraction"] == 0.10
        and split["group_overlap"] == 0,
        {key: split[key] for key in ("seed", "validation_fraction", "group_overlap")},
    )
    check(
        "aligned_training_rows",
        split["paired_before_empty_filter"] == 67820
        and split["train_examples"] + split["validation_examples"] == 67803,
        {
            "paired": split["paired_before_empty_filter"],
            "retained": split["train_examples"] + split["validation_examples"],
        },
    )
    check(
        "audited_training_alignment",
        split.get("alignment_protocol")
        == "authoritative-teacher-summary/text-one-to-one/same-index-audited-recovery-v1"
        and split.get("teacher_target_field") == "model_answer"
        and split.get("question_shards") == 62
        and split.get("direct_question_matches") == 67773
        and split.get("recovered_question_matches") == 47
        and split.get("structured_reference_matches") == 67820
        and split.get("unused_teacher_rows") == 0,
        {
            key: split.get(key)
            for key in (
                "alignment_protocol",
                "teacher_target_field",
                "question_shards",
                "direct_question_matches",
                "recovered_question_matches",
                "structured_reference_matches",
                "unused_teacher_rows",
            )
        },
    )
    run_dir = Path(args.run_dir)
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    topology = run_manifest.get("distributed_topology", {})
    check(
        "native_vlm_runtime",
        run_manifest.get("modality") == "image+numeric-window+soft-hints"
        and run_manifest.get("actual_effective_batch_size") in {32, 48}
        and run_manifest.get("world_size") == 8
        and topology.get("node_count") == 2
        and topology.get("gpus_per_node") == 4
        and len(topology.get("nodes", [])) == 2
        and all(
            "H800" in gpu_name
            for node in topology.get("nodes", [])
            for gpu_name in node.get("gpu_names", [])
        )
        and run_manifest.get("qwen3_vl_pixel_budget", {}).get("official_min_pixels") == 65_536
        and run_manifest.get("qwen3_vl_pixel_budget", {}).get("official_max_pixels") == 16_777_216
        and run_manifest.get("qwen3_vl_pixel_budget", {}).get("runtime_min_pixels") == 65_536
        and 65_536
        <= run_manifest.get("qwen3_vl_pixel_budget", {}).get("runtime_max_pixels", 0)
        <= 16_777_216
        and run_manifest.get("llm_gradient_checkpointing") is True
        and run_manifest.get("eval_safe_checkpointed_decoder_layers") == 36,
        {
            "modality": run_manifest.get("modality"),
            "world_size": run_manifest.get("world_size"),
            "effective_batch": run_manifest.get("actual_effective_batch_size"),
            "distributed_topology": topology,
            "pixel_budget": run_manifest.get("qwen3_vl_pixel_budget"),
        },
    )
    flash_audit = run_manifest.get("flash_attention_audit", {})
    resolved_configs = flash_audit.get("llm_resolved_configs", {})
    check(
        "flash_attention_runtime",
        flash_audit.get("requested") == "flash_attention_2"
        and flash_audit.get("fallback_allowed") is False
        and flash_audit.get("llm_attention_module_count", 0) > 0
        and flash_audit.get("hint_cross_attention_module_count", 0) > 0
        and resolved_configs
        and all(
            (value.get("resolved") or value.get("internal")) == "flash_attention_2"
            for value in resolved_configs.values()
        ),
        flash_audit,
    )
    complete = run_dir / "TRAINING_COMPLETE"
    training_payload = (
        json.loads(complete.read_text(encoding="utf-8")) if complete.exists() else {}
    )
    check(
        "training_complete",
        training_payload.get("epochs") == config.training.epochs,
        training_payload,
    )
    best = json.loads((run_dir / "best_checkpoint.json").read_text(encoding="utf-8"))
    check(
        "checkpoint_selected_by_validation",
        best.get("selection_metric") == "validation_answer_nll"
        and best.get("completed_epochs") == config.training.epochs
        and not best.get("early_stopping"),
        best,
    )
    manifest_dir, predictions_dir, judge_input_dir, judge_root = map(
        Path,
        (
            args.manifest_dir,
            args.predictions_dir,
            args.judge_input_dir,
            args.geval_results_root,
        ),
    )
    image_root = Path(args.image_root)
    image_manifest_path = image_root / "image_manifest.jsonl"
    image_records = {
        row["image_id"]: row for row in read_jsonl(image_manifest_path)
    }
    image_audit_ok = bool(image_records)
    for image_id, row in image_records.items():
        image_path = image_root / row["image_relpath"]
        image_audit_ok = image_audit_ok and (
            row.get("renderer_version") == config.vision.renderer_version
            and row.get("dpi") == 600
            and row.get("contains_anomaly_score") is False
            and row.get("contains_label_annotation") is False
            and row.get("half_open_boundaries")
            == [row["interval"][0] - 0.5, row["interval"][1] - 0.5]
            and image_path.is_file()
            and sha256_file(image_path) == row.get("sha256")
        )
        if not image_audit_ok:
            break
    check(
        "vlm_image_manifest",
        image_audit_ok,
        {"unique_images": len(image_records), "path": str(image_manifest_path)},
    )
    for dataset in DATASETS:
        references = read_jsonl(manifest_dir / f"eval_{dataset}.jsonl")
        reference_image_ids = [
            str(
                row.get("image_id")
                or visual_image_id(
                    str(row["base_sample_id"]), tuple(row["interval"]), config.vision.renderer_version
                )
            )
            for row in references
        ]
        check(
            f"{dataset}_images",
            all(
                image_id in image_records
                and (image_root / visual_image_relpath(image_id)).is_file()
                for image_id in reference_image_ids
            ),
            {"rows": len(reference_image_ids), "unique": len(set(reference_image_ids))},
        )
        counts = collections.Counter(row["question_group"] for row in references)
        check(
            f"{dataset}_manifest_count",
            len(references) == EXPECTED_DATASET_COUNTS[dataset],
            len(references),
        )
        check(
            f"{dataset}_type_counts",
            dict(counts) == EXPECTED_TYPE_COUNTS[dataset],
            dict(counts),
        )
        predictions = read_jsonl(predictions_dir / f"{dataset}.predictions.jsonl")
        prediction_ids = [row["sample_id"] for row in predictions]
        check(
            f"{dataset}_predictions",
            len(predictions) == len(references)
            and len(set(prediction_ids)) == len(references)
            and prediction_ids == [row["sample_id"] for row in references]
            and [row.get("image_id") for row in predictions] == reference_image_ids,
            {"rows": len(predictions), "unique": len(set(prediction_ids))},
        )
        judge_inputs = read_jsonl(judge_input_dir / f"{dataset}.judge_input.jsonl")
        expected_tasks = sum(
            len(TYPE_INFO[row["question_type"]][1]) for row in judge_inputs
        )
        for judge in JUDGES:
            rows = read_jsonl(judge_root / judge / f"{dataset}.jsonl")
            keys = {(row["record_id"], row["dimension"]) for row in rows}
            valid_scores = all(1.0 <= float(row["score"]) <= 5.0 for row in rows)
            method_ok = all(
                (row["method"] == "integer_score")
                == (judge in {"gemini-2.5-pro", "gpt-5.4"})
                for row in rows
            )
            qwen_bound_ok = (
                all(
                    (row.get("distribution_metadata") or {}).get(
                        "missing_mass_upper_bound", 0.0
                    )
                    <= 1e-6
                    for row in rows
                )
                if judge.startswith("qwen")
                else True
            )
            check(
                f"{dataset}_{judge}_judge_tasks",
                len(keys) == expected_tasks and len(rows) == expected_tasks,
                {"expected": expected_tasks, "rows": len(rows), "unique": len(keys)},
            )
            check(
                f"{dataset}_{judge}_score_contract",
                valid_scores and method_ok and qwen_bound_ok,
                {
                    "valid_scores": valid_scores,
                    "method_ok": method_ok,
                    "qwen_bound_ok": qwen_bound_ok,
                },
            )
    passed = all(item["passed"] for item in checks)
    payload = {"passed": passed, "checks": checks}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"passed": passed, "checks": len(checks)}))
    if not passed:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise SystemExit("Audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
