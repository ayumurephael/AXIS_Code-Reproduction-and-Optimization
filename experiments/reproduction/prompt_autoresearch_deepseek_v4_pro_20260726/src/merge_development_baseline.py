"""Merge previously audited Baseline rows into the exposed 96-QA dev pool."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screening", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_ids = set(manifest["record_ids"])
    prediction_rows: list[dict] = []
    score_rows: list[dict] = []
    sources = []
    for name, root in (("screening", args.screening), ("validation", args.validation)):
        predictions_path = root / "predictions.jsonl"
        scores_path = root / "scores.jsonl"
        predictions = [row for row in read_jsonl(predictions_path) if row.get("mode") == "base"]
        scores = [row for row in read_jsonl(scores_path) if row.get("mode") == "base"]
        prediction_rows.extend(predictions)
        score_rows.extend(scores)
        sources.append(
            {
                "name": name,
                "predictions": str(predictions_path),
                "predictions_sha256": sha256(predictions_path),
                "scores": str(scores_path),
                "scores_sha256": sha256(scores_path),
                "baseline_predictions": len(predictions),
                "baseline_scores": len(scores),
            }
        )

    errors = []
    prediction_keys = [(row.get("record_id"), row.get("mode")) for row in prediction_rows]
    score_keys = [(row.get("record_id"), row.get("mode"), row.get("dimension")) for row in score_rows]
    if len(prediction_rows) != 96:
        errors.append(f"prediction_rows={len(prediction_rows)}")
    if len(set(prediction_keys)) != len(prediction_keys):
        errors.append("duplicate prediction key")
    if {row.get("record_id") for row in prediction_rows} != expected_ids:
        errors.append("prediction record coverage mismatch")
    if len(score_rows) != 221:
        errors.append(f"score_rows={len(score_rows)}")
    if len(set(score_keys)) != len(score_keys):
        errors.append("duplicate score key")
    if {row.get("record_id") for row in score_rows} != expected_ids:
        errors.append("score record coverage mismatch")
    if {row.get("model") for row in score_rows} != {"deepseek-v4-pro"}:
        errors.append("Judge model mismatch")
    if {row.get("provider") for row in score_rows} != {"deepseek"}:
        errors.append("Judge provider mismatch")
    allowed_methods = {"final_score_top_logprobs", "exact_sample_mean_20"}
    if not {row.get("method") for row in score_rows}.issubset(allowed_methods):
        errors.append("unapproved scoring method")
    if any(not row.get("prompt_sha256") for row in score_rows):
        errors.append("missing Judge prompt hash")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    predictions_out = output / "baseline_cached_predictions.jsonl"
    scores_out = output / "baseline_cached_scores.jsonl"
    prediction_rows.sort(key=lambda row: row["record_id"])
    score_rows.sort(key=lambda row: (row["record_id"], row["dimension"]))
    write_jsonl(predictions_out, prediction_rows)
    write_jsonl(scores_out, score_rows)
    audit = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "prediction_rows": len(prediction_rows),
        "score_rows": len(score_rows),
        "score_methods": dict(collections.Counter(row.get("method") for row in score_rows)),
        "score_models": dict(collections.Counter(row.get("model") for row in score_rows)),
        "sources": sources,
        "output_hashes": {
            predictions_out.name: sha256(predictions_out),
            scores_out.name: sha256(scores_out),
        },
        "errors": errors,
        "ok": not errors,
    }
    (output / "baseline_cache_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()