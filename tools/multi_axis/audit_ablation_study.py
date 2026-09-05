from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.models.MultiAXIS.ablation import ABLATION_VARIANTS, FULL_VARIANT, ablation_spec
from tools.multi_axis.aggregate_geval import COLUMNS, TYPE_INFO, example_scores, ten_metrics


JUDGE = "qwen3-30b-a3b-instruct-2507"
DATASET = "478new"
TRAINED_SOURCE_COMMIT = "303ca1d49303eff4ab161fafd6117c3f1c010b3a"
CHECKPOINT_SHA256 = "3a1c6e912371635f5f8b2f736224d42b0f7d023eab044fd078ffae745d2937be"
MANIFEST_SHA256 = "a5f919294545cc96b0d3a834d9dd4b223c440a26f32edd40d8eb663571b75a26"


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of the Epoch-17 Multi-AXIS ablation study"
    )
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--baseline",
        default="experiments/multi_axis/ablation_epoch17_qwen30b_baseline.json",
    )
    parser.add_argument("--expected-inference-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checks: list[dict] = []

    def check(name: str, condition: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(condition), "detail": detail})

    root = Path(args.study_root)
    manifest_path = Path(args.manifest)
    references = _jsonl(manifest_path)
    reference_ids = [row.get("sample_id") for row in references]
    check("manifest_sha256", _sha256(manifest_path) == MANIFEST_SHA256, _sha256(manifest_path))
    check("manifest_rows", len(references) == 478, len(references))
    check("manifest_unique_ids", len(set(reference_ids)) == 478, len(set(reference_ids)))

    baseline = _json(Path(args.baseline))
    provenance = baseline.get("provenance", {})
    check("baseline_status", baseline.get("status") == "reused_existing_result", baseline.get("status"))
    check("baseline_variant", baseline.get("variant") == FULL_VARIANT, baseline.get("variant"))
    check("baseline_source_commit", provenance.get("trained_source_commit") == TRAINED_SOURCE_COMMIT, provenance.get("trained_source_commit"))
    check("baseline_checkpoint", provenance.get("checkpoint_epoch") == 17 and provenance.get("checkpoint_sha256") == CHECKPOINT_SHA256, {"epoch": provenance.get("checkpoint_epoch"), "sha256": provenance.get("checkpoint_sha256")})
    check("baseline_dataset", provenance.get("dataset") == DATASET and provenance.get("dataset_sha256") == MANIFEST_SHA256, provenance.get("dataset"))
    check("baseline_judge", provenance.get("judge") == JUDGE, provenance.get("judge"))
    check("baseline_generation", provenance.get("generation") == {"num_beams": 5, "do_sample": False, "max_new_tokens": 1000}, provenance.get("generation"))
    check("baseline_metrics", set(baseline.get("metrics", {})) == set(COLUMNS), sorted(baseline.get("metrics", {})))

    for variant in ABLATION_VARIANTS[1:]:
        variant_root = root / variant
        predictions_dir = variant_root / "predictions"
        inference = _json(predictions_dir / "inference_manifest.json")
        predictions = _jsonl(predictions_dir / f"{DATASET}.predictions.jsonl")
        judge_inputs = _jsonl(variant_root / "judge_inputs" / f"{DATASET}.judge_input.jsonl")
        scores = _jsonl(variant_root / "geval" / JUDGE / f"{DATASET}.jsonl")
        aggregate = _json(variant_root / "aggregate" / "geval_10metrics.json")
        evaluation_audit = _json(variant_root / "evaluation_audit.json")

        prefix = variant
        check(f"{prefix}_inference_complete", (predictions_dir / "INFERENCE_COMPLETE").is_file(), str(predictions_dir))
        check(f"{prefix}_mode", inference.get("ablation_variant") == variant and inference.get("ablation_spec") == ablation_spec(variant).to_dict(), {"mode": inference.get("ablation_variant")})
        check(f"{prefix}_inference_commit", inference.get("inference_source_commit") == args.expected_inference_commit.lower(), inference.get("inference_source_commit"))
        check(f"{prefix}_checkpoint", inference.get("hint_checkpoint_epoch") == 17 and inference.get("hint_checkpoint_sha256") == CHECKPOINT_SHA256, {"epoch": inference.get("hint_checkpoint_epoch"), "sha256": inference.get("hint_checkpoint_sha256")})
        generation = inference.get("generation", {})
        check(f"{prefix}_generation", generation.get("num_beams") == 5 and generation.get("do_sample") is False and generation.get("max_new_tokens") == 1000, generation)
        check(f"{prefix}_dataset", inference.get("datasets") == [DATASET], inference.get("datasets"))
        check(f"{prefix}_predictions", len(predictions) == 478 and [row.get("index") for row in predictions] == list(range(478)) and [row.get("sample_id") for row in predictions] == reference_ids and all(row.get("ablation_variant") == variant for row in predictions) and all(str(row.get("raw_response", "")).strip() for row in predictions), {"rows": len(predictions), "unique": len({row.get('sample_id') for row in predictions})})
        check(f"{prefix}_judge_inputs", len(judge_inputs) == 478 and [row.get("sample_id") for row in judge_inputs] == reference_ids and all(row.get("mode") == variant for row in judge_inputs) and all(row.get("question") == reference.get("question") and row.get("answer") == reference.get("teacher_answer") and row.get("response") == prediction.get("raw_response") for row, reference, prediction in zip(judge_inputs, references, predictions)), len(judge_inputs))
        expected_keys = {
            (row["record_id"], dimension)
            for row in judge_inputs
            for dimension in TYPE_INFO[row["question_type"]][1]
        }
        score_keys = {(row.get("record_id"), row.get("dimension")) for row in scores}
        qwen_methods = {"final_score_top_logprobs_bounded", "a_e_json_readout_top_logprobs_bounded"}
        check(f"{prefix}_judge_scores", len(scores) == 1095 and score_keys == expected_keys and all(row.get("mode") == variant and row.get("provider_profile") == JUDGE and row.get("requested_model") == JUDGE and row.get("temperature") == 0.0 and row.get("logprobs_requested") is True and row.get("top_logprobs") == 5 and row.get("method") in qwen_methods and 1.0 <= float(row.get("score")) <= 5.0 and float((row.get("distribution_metadata") or {}).get("missing_mass_upper_bound", 1.0)) <= 1e-6 for row in scores), {"rows": len(scores), "unique": len(score_keys)})
        table = ten_metrics(example_scores(scores))
        stored = aggregate.get("by_judge_dataset", {}).get(JUDGE, {}).get(DATASET, {})
        check(f"{prefix}_aggregate", all(abs(float(stored.get(column, -999)) - float(table[column])) <= 1e-12 for column in COLUMNS), stored)
        check(f"{prefix}_evaluation_audit", evaluation_audit.get("passed") is True and evaluation_audit.get("mode") == variant, {"passed": evaluation_audit.get("passed"), "mode": evaluation_audit.get("mode")})

    passed = all(item["passed"] for item in checks)
    payload = {
        "passed": passed,
        "dataset": DATASET,
        "judge": JUDGE,
        "trained_source_commit": TRAINED_SOURCE_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "variants": list(ABLATION_VARIANTS),
        "checks": checks,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": passed, "checks": len(checks)}))
    if not passed:
        raise SystemExit(
            "Ablation audit failed: "
            + ", ".join(item["name"] for item in checks if not item["passed"])
        )


if __name__ == "__main__":
    main()
