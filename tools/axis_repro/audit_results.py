"""Fail-closed integrity audit for AXIS prediction and G-Eval JSONL files."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .build_tables import DIMS
from .common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--scores")
    parser.add_argument("--manifest")
    parser.add_argument(
        "--modes", nargs="+",
        default=["base", "wo_fixed_hint", "wo_local_hint", "wo_windows"],
    )
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-provider")
    parser.add_argument("--allowed-methods", nargs="+")
    parser.add_argument(
        "--require-logprobs",
        action="store_true",
        help="Require a normalized complete 1--5 distribution on every row.",
    )
    parser.add_argument(
        "--require-prompt-hashes",
        action="store_true",
        help="Require a valid SHA-256 prompt identity on every score row.",
    )
    parser.add_argument(
        "--max-missing-score-mass-upper-bound",
        type=float,
        help="Reject score rows whose recorded missing-mass bound exceeds this.",
    )
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    errors = []
    prediction_keys = [(x["record_id"], x["mode"]) for x in predictions]
    if len(prediction_keys) != len(set(prediction_keys)):
        errors.append("duplicate prediction (record_id, mode) keys")
    if any(not str(x.get("response", "")).strip() for x in predictions):
        errors.append("empty generated response")

    record_ids = sorted({x["record_id"] for x in predictions})
    if args.manifest:
        expected_ids = set(json.loads(Path(args.manifest).read_text(encoding="utf-8"))["record_ids"])
        if set(record_ids) != expected_ids:
            errors.append(
                f"record-id mismatch: got={len(record_ids)} expected={len(expected_ids)}"
            )
    expected_prediction_keys = {
        (record_id, mode) for record_id in record_ids for mode in args.modes
    }
    missing_predictions = expected_prediction_keys - set(prediction_keys)
    unexpected_predictions = set(prediction_keys) - expected_prediction_keys
    if missing_predictions:
        errors.append(f"missing predictions: {len(missing_predictions)}")
    if unexpected_predictions:
        errors.append(f"unexpected predictions: {len(unexpected_predictions)}")

    summary = {
        "prediction_rows": len(predictions),
        "records": len(record_ids),
        "prediction_modes": dict(Counter(x["mode"] for x in predictions)),
    }

    if args.scores:
        scores = read_jsonl(args.scores)
        score_keys = [(x["record_id"], x["mode"], x["dimension"]) for x in scores]
        if len(score_keys) != len(set(score_keys)):
            errors.append("duplicate score (record_id, mode, dimension) keys")
        qtypes = {(x["record_id"], x["mode"]): x["question_type"] for x in predictions}
        expected_score_keys = {
            (record_id, mode, dimension)
            for record_id, mode in expected_prediction_keys
            for dimension in DIMS[qtypes[(record_id, mode)]]
        }
        missing_scores = expected_score_keys - set(score_keys)
        unexpected_scores = set(score_keys) - expected_score_keys
        if missing_scores:
            errors.append(f"missing scores: {len(missing_scores)}")
        if unexpected_scores:
            errors.append(f"unexpected scores: {len(unexpected_scores)}")
        methods = Counter(str(row.get("method", "")) for row in scores)
        models = Counter(str(row.get("model", "")) for row in scores)
        providers = Counter(str(row.get("provider", "")) for row in scores)
        logprobs_returned = sum(
            bool(row.get("logprobs_returned")) for row in scores
        )
        prompt_hashes = [
            str(row.get("prompt_sha256", "")) for row in scores
        ]
        for row in scores:
            method = str(row.get("method", ""))
            try:
                score = float(row.get("score"))
            except (TypeError, ValueError):
                errors.append(
                    f"non-numeric score at {row.get('record_id')} "
                    f"{row.get('mode')} {row.get('dimension')}"
                )
                break
            if not math.isfinite(score) or not 1.0 <= score <= 5.0:
                errors.append(
                    f"score outside [1,5] at {row.get('record_id')} "
                    f"{row.get('mode')} {row.get('dimension')}"
                )
                break
            fallback_match = re.fullmatch(
                r"exact_sample_mean_(\d+)", method
            )
            if fallback_match:
                requested_samples = int(fallback_match.group(1))
                if (
                    requested_samples != 20
                    or len(row.get("fallback_scores", [])) != 20
                ):
                    errors.append(
                        f"fallback must contain exactly 20 samples at "
                        f"{row['record_id']} {row['mode']} "
                        f"{row['dimension']}"
                    )
                    break
            if args.require_logprobs:
                distribution = row.get("distribution")
                if (
                    not row.get("logprobs_returned")
                    or not isinstance(distribution, dict)
                    or set(distribution) != set("12345")
                ):
                    errors.append(
                        f"incomplete logprobs at {row['record_id']} "
                        f"{row['mode']} {row['dimension']}"
                    )
                    break
                try:
                    probabilities = [
                        float(distribution[key]) for key in "12345"
                    ]
                except (TypeError, ValueError):
                    errors.append(
                        f"non-numeric logprob distribution at "
                        f"{row['record_id']} {row['mode']} "
                        f"{row['dimension']}"
                    )
                    break
                if (
                    any(not math.isfinite(value) or value < 0.0
                        for value in probabilities)
                    or not math.isclose(
                        sum(probabilities), 1.0, rel_tol=1e-8, abs_tol=1e-8
                    )
                ):
                    errors.append(
                        f"invalid logprob distribution at {row['record_id']} "
                        f"{row['mode']} {row['dimension']}"
                    )
                    break
            if args.max_missing_score_mass_upper_bound is not None:
                try:
                    missing_mass_bound = float(
                        row.get("missing_score_mass_upper_bound")
                    )
                except (TypeError, ValueError):
                    errors.append(
                        f"missing score-mass bound at {row['record_id']} "
                        f"{row['mode']} {row['dimension']}"
                    )
                    break
                if (
                    not math.isfinite(missing_mass_bound)
                    or missing_mass_bound < 0.0
                    or missing_mass_bound
                    > args.max_missing_score_mass_upper_bound
                ):
                    errors.append(
                        f"score-mass bound exceeds limit at "
                        f"{row['record_id']} {row['mode']} "
                        f"{row['dimension']}"
                    )
                    break
        if args.expected_model and set(models) != {args.expected_model}:
            errors.append(f"model mismatch: {dict(models)}")
        if args.expected_provider and set(providers) != {args.expected_provider}:
            errors.append(f"provider mismatch: {dict(providers)}")
        if args.allowed_methods and not set(methods).issubset(
            set(args.allowed_methods)
        ):
            errors.append(f"unexpected score methods: {dict(methods)}")
        if args.require_prompt_hashes and any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in prompt_hashes
        ):
            errors.append("missing or invalid prompt SHA-256")
        summary.update({
            "score_rows": len(scores),
            "score_methods": dict(methods),
            "score_models": dict(models),
            "score_providers": dict(providers),
            "logprobs_returned": logprobs_returned,
            "valid_prompt_hashes": sum(
                re.fullmatch(r"[0-9a-f]{64}", value) is not None
                for value in prompt_hashes
            ),
        })

    summary["errors"] = errors
    summary["ok"] = not errors
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
