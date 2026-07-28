"""Fail-closed integrity audit for the locked paper140 multi-Judge experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from experiments.reproduction.prompt_round9_paper140_multi_judge_20260728.src.paper140_multi_judge import (
    MODES,
)
from tools.axis_repro.common import read_jsonl


EXPECTED = {
    "deepseek": {
        "model": "deepseek-v4-pro",
        "provider": "deepseek",
        "enable_thinking": None,
        "raw_fallback": 6,
        "allowed_methods": {
            "final_score_top_logprobs",
            "exact_sample_mean_20",
        },
    },
    "qwen35": {
        "model": "qwen3.5-397b-a17b",
        "provider": "qwen",
        "enable_thinking": False,
        "raw_fallback": 1,
        "allowed_methods": {
            "final_score_top_logprobs",
            "final_score_top_logprobs_bounded",
            "final_score_top_logprobs_readout",
            "exact_sample_mean_20",
        },
    },
    "qwen30b": {
        "model": "qwen3-30b-a3b-instruct-2507",
        "provider": "qwen",
        "enable_thinking": None,
        "raw_fallback": 0,
        "allowed_methods": {
            "final_score_top_logprobs",
            "final_score_top_logprobs_bounded",
            "final_score_top_logprobs_readout_bounded",
        },
    },
}
QWEN35_AUTHORIZED_FALLBACK_KEY = (
    "series_000096:0",
    "base",
    "accuracy",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_RE = re.compile(rb"sk-[A-Za-z0-9_-]{12,}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_equal(actual, expected, label: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{label}: got={actual!r}, expected={expected!r}")


def audit_judge(root: Path, label: str, spec: dict, errors: list[str]) -> dict:
    judge_dir = root / "artifacts" / label
    raw = read_jsonl(judge_dir / "raw_scores.jsonl")
    scores = read_jsonl(judge_dir / "scores.jsonl")
    audit = json.loads((judge_dir / "audit.json").read_text(encoding="utf-8"))

    raw_keys = [
        (row["record_id"], row["mode"], row["dimension"]) for row in raw
    ]
    expanded_keys = [
        (row["record_id"], row["mode"], row["dimension"]) for row in scores
    ]
    assert_equal(len(raw), 457, f"{label}.raw_rows", errors)
    assert_equal(len(set(raw_keys)), len(raw_keys), f"{label}.raw_unique", errors)
    assert_equal(len(scores), 1340, f"{label}.expanded_rows", errors)
    assert_equal(
        len(set(expanded_keys)),
        len(expanded_keys),
        f"{label}.expanded_unique",
        errors,
    )
    assert_equal(audit.get("ok"), True, f"{label}.audit_ok", errors)

    methods = Counter(str(row.get("method")) for row in raw)
    if not set(methods).issubset(spec["allowed_methods"]):
        errors.append(f"{label}.unexpected_methods={dict(methods)}")
    assert_equal(
        methods.get("exact_sample_mean_20", 0),
        spec["raw_fallback"],
        f"{label}.raw_fallback",
        errors,
    )

    for row in raw:
        key = (row["record_id"], row["mode"], row["dimension"])
        assert_equal(row.get("model"), spec["model"], f"{label}.model@{key}", errors)
        assert_equal(
            row.get("provider"),
            spec["provider"],
            f"{label}.provider@{key}",
            errors,
        )
        assert_equal(
            row.get("enable_thinking"),
            spec["enable_thinking"],
            f"{label}.enable_thinking@{key}",
            errors,
        )
        if not SHA256_RE.fullmatch(str(row.get("prompt_sha256", ""))):
            errors.append(f"{label}.invalid_prompt_sha256@{key}")
        score = row.get("score")
        if (
            not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 1.0 <= float(score) <= 5.0
        ):
            errors.append(f"{label}.invalid_score@{key}")

        method = row.get("method")
        if method == "exact_sample_mean_20":
            samples = row.get("fallback_scores")
            if (
                not isinstance(samples, list)
                or len(samples) != 20
                or any(value not in {1, 2, 3, 4, 5} for value in samples)
            ):
                errors.append(f"{label}.invalid_exact20@{key}")
        elif "bounded" in str(method):
            bound = row.get("missing_score_mass_upper_bound")
            if (
                not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                or not 0.0 <= float(bound) <= 1e-6
            ):
                errors.append(f"{label}.invalid_missing_mass_bound@{key}")

    if label == "qwen35":
        fallback_keys = {
            (row["record_id"], row["mode"], row["dimension"])
            for row in raw
            if row.get("method") == "exact_sample_mean_20"
        }
        assert_equal(
            fallback_keys,
            {QWEN35_AUTHORIZED_FALLBACK_KEY},
            "qwen35.authorized_fallback_key",
            errors,
        )

    return {
        "raw_rows": len(raw),
        "expanded_rows": len(scores),
        "raw_methods": dict(methods),
        "expanded_methods": dict(
            Counter(str(row.get("method")) for row in scores)
        ),
        "model": spec["model"],
        "provider": spec["provider"],
        "enable_thinking": (
            "auto" if spec["enable_thinking"] is None
            else spec["enable_thinking"]
        ),
        "audit_ok": audit.get("ok"),
    }


def audit_experiment(root: Path) -> dict:
    errors: list[str] = []
    predictions = read_jsonl(
        root / "artifacts" / "predictions" / "paper140_all_modes.jsonl"
    )
    unique_predictions = read_jsonl(
        root / "artifacts" / "predictions" / "paper140_unique_responses.jsonl"
    )
    dedup = json.loads(
        (
            root
            / "artifacts"
            / "predictions"
            / "deduplication_manifest.json"
        ).read_text(encoding="utf-8")
    )
    prediction_keys = [
        (row["record_id"], row["mode"]) for row in predictions
    ]
    assert_equal(len(predictions), 560, "prediction_rows", errors)
    assert_equal(
        len(set(prediction_keys)),
        len(prediction_keys),
        "prediction_keys_unique",
        errors,
    )
    assert_equal(len(unique_predictions), 201, "unique_prediction_rows", errors)
    assert_equal(
        dedup.get("unique_expected_score_dimensions"),
        457,
        "unique_score_dimensions",
        errors,
    )
    assert_equal(
        dedup.get("expanded_expected_score_dimensions"),
        1340,
        "expanded_score_dimensions",
        errors,
    )
    assert_equal(
        set(row["mode"] for row in predictions),
        set(MODES),
        "prediction_modes",
        errors,
    )

    judges = {
        label: audit_judge(root, label, spec, errors)
        for label, spec in EXPECTED.items()
    }

    secret_hits: list[str] = []
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "integrity_manifest.json":
            continue
        data = path.read_bytes()
        if SECRET_RE.search(data):
            secret_hits.append(relative)
        hashes[relative] = file_sha256(path)
    if secret_hits:
        errors.append(f"credential-like token found in: {secret_hits}")

    return {
        "experiment": "prompt_round9_paper140_multi_judge_20260728",
        "dataset": "paper140",
        "records_per_mode": 140,
        "modes": list(MODES),
        "prediction_rows": len(predictions),
        "unique_prediction_rows": len(unique_predictions),
        "unique_score_dimensions_per_judge": 457,
        "expanded_score_dimensions_per_judge": 1340,
        "judges": judges,
        "credential_like_token_hits": secret_hits,
        "artifact_sha256": hashes,
        "errors": errors,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = audit_experiment(Path(args.experiment_root))
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "errors": payload["errors"],
                "hashed_files": len(payload["artifact_sha256"]),
            },
            ensure_ascii=False,
        )
    )
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
