from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from src.models.MultiAXIS.config import DEEPSEEK_MODEL_ID, MultiAxisConfig
from tools.multi_axis.aggregate_geval import DATASETS, JUDGES, TYPE_INFO


EXPECTED_DATASET_COUNTS = {"478new": 478, "SMD": 200, "SWaT": 184, "LEMMA-RCA": 12, "VTA": 200}
EXPECTED_TYPE_COUNTS = {"478new": {"MC": 176, "OE": 139, "TF": 163}, "SMD": {"MC": 66, "OE": 67, "TF": 67}, "SWaT": {"MC": 61, "OE": 62, "TF": 61}, "LEMMA-RCA": {"MC": 4, "OE": 4, "TF": 4}, "VTA": {"MC": 67, "OE": 67, "TF": 66}}


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", required=True)
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
    check("formal_model", config.llm.model_name == DEEPSEEK_MODEL_ID, config.llm.model_name)
    check("forty_epochs", config.training.epochs == 40 and not config.training.early_stopping, config.training.__dict__)
    check("full_architecture", config.hints.ablation_phase == "D", config.hints.ablation_phase)
    check("flash_attention", config.hints.require_flash_attention and config.llm.attention_implementation == "flash_attention_2", {"cross_attention": config.hints.require_flash_attention, "llm": config.llm.attention_implementation})
    split = json.loads((Path(args.manifest_dir) / "train_split_summary.json").read_text(encoding="utf-8"))
    check("empty_teacher_filter", split["empty_teacher_answers_filtered"] == 17, split["empty_teacher_answers_filtered"])
    check("group_split", split["seed"] == 42 and split["validation_fraction"] == 0.10 and split["group_overlap"] == 0, {key: split[key] for key in ("seed", "validation_fraction", "group_overlap")})
    check("aligned_training_rows", split["paired_before_empty_filter"] == 67820 and split["train_examples"] + split["validation_examples"] == 67803, {"paired": split["paired_before_empty_filter"], "retained": split["train_examples"] + split["validation_examples"]})
    check("audited_training_alignment", split.get("alignment_protocol") == "authoritative-teacher-summary/text-one-to-one/same-index-audited-recovery-v1" and split.get("teacher_target_field") == "model_answer" and split.get("question_shards") == 62 and split.get("direct_question_matches") == 67773 and split.get("recovered_question_matches") == 47 and split.get("structured_reference_matches") == 67820 and split.get("unused_teacher_rows") == 0, {key: split.get(key) for key in ("alignment_protocol", "teacher_target_field", "question_shards", "direct_question_matches", "recovered_question_matches", "structured_reference_matches", "unused_teacher_rows")})
    complete = Path(args.run_dir) / "TRAINING_COMPLETE"
    training_payload = json.loads(complete.read_text(encoding="utf-8")) if complete.exists() else {}
    check("training_complete", training_payload.get("epochs") == 40, training_payload)
    best = json.loads((Path(args.run_dir) / "best_checkpoint.json").read_text(encoding="utf-8"))
    check("checkpoint_selected_by_validation", best.get("selection_metric") == "validation_answer_nll" and best.get("completed_epochs") == 40 and not best.get("early_stopping"), best)

    manifest_dir, predictions_dir, judge_input_dir, judge_root = map(Path, (args.manifest_dir, args.predictions_dir, args.judge_input_dir, args.geval_results_root))
    for dataset in DATASETS:
        references = read_jsonl(manifest_dir / f"eval_{dataset}.jsonl")
        counts = collections.Counter(row["question_group"] for row in references)
        check(f"{dataset}_manifest_count", len(references) == EXPECTED_DATASET_COUNTS[dataset], len(references))
        check(f"{dataset}_type_counts", dict(counts) == EXPECTED_TYPE_COUNTS[dataset], dict(counts))
        predictions = read_jsonl(predictions_dir / f"{dataset}.predictions.jsonl")
        prediction_ids = [row["sample_id"] for row in predictions]
        check(f"{dataset}_predictions", len(predictions) == len(references) and len(set(prediction_ids)) == len(references) and prediction_ids == [row["sample_id"] for row in references], {"rows": len(predictions), "unique": len(set(prediction_ids))})
        judge_inputs = read_jsonl(judge_input_dir / f"{dataset}.judge_input.jsonl")
        expected_tasks = sum(len(TYPE_INFO[row["question_type"]][1]) for row in judge_inputs)
        for judge in JUDGES:
            rows = read_jsonl(judge_root / judge / f"{dataset}.jsonl")
            keys = {(row["record_id"], row["dimension"]) for row in rows}
            valid_scores = all(1.0 <= float(row["score"]) <= 5.0 for row in rows)
            method_ok = all((row["method"] == "integer_score") == (judge in {"gemini-2.5-pro", "gpt-5.4"}) for row in rows)
            qwen_bound_ok = all((row.get("distribution_metadata") or {}).get("missing_mass_upper_bound", 0.0) <= 1e-6 for row in rows) if judge.startswith("qwen") else True
            check(f"{dataset}_{judge}_judge_tasks", len(keys) == expected_tasks and len(rows) == expected_tasks, {"expected": expected_tasks, "rows": len(rows), "unique": len(keys)})
            check(f"{dataset}_{judge}_score_contract", valid_scores and method_ok and qwen_bound_ok, {"valid_scores": valid_scores, "method_ok": method_ok, "qwen_bound_ok": qwen_bound_ok})
    passed = all(item["passed"] for item in checks)
    payload = {"passed": passed, "checks": checks}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": len(checks)}))
    if not passed:
        failed = [item["name"] for item in checks if not item["passed"]]
        raise SystemExit("Audit failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
