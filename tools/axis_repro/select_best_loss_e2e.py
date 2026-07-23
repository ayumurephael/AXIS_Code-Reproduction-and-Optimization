"""Fail-closed selection of the best loss_e2e validation checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


DECLARED_STEPS = (4750, 9500, 14250, 19000)
EFFECTIVE_TOKEN_DEFINITION = (
    "shifted AXIS labels != -100 on non-error rows under the common global "
    "continuation objective"
)
SELECTION_METRIC = "effective-answer-token-weighted teacher-forced global token NLL"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def select_best_candidate(candidates: Sequence[Mapping], arm: str) -> dict:
    """Validate the four summaries and select minimum token-weighted NLL."""
    if arm not in {"control", "treatment"}:
        raise ValueError(f"unsupported arm: {arm}")
    if len(candidates) != len(DECLARED_STEPS):
        raise ValueError("selection requires exactly four candidate summaries")
    by_step = {}
    shared_fields = (
        "author_checkpoint_sha256",
        "training_data_audit_sha256",
        "split_manifest_sha256",
        "validation_data_manifest_sha256",
        "effective_answer_token_count",
        "effective_token_definition",
        "validation_series",
        "total_rows",
        "valid_rows",
        "excluded_rows",
        "validation_seed",
        "train_ratio",
        "world_size",
        "frozen_llm_storage_dtype",
        "autocast_dtype",
        "loss_accumulation_dtype",
        "loss_chunk_size",
    )
    shared = None
    checkpoint_hashes = set()
    summary_hashes = set()
    normalized = []
    for candidate in candidates:
        step = candidate.get("candidate_step")
        if step not in DECLARED_STEPS:
            raise ValueError(f"undeclared candidate step: {step!r}")
        if step in by_step:
            raise ValueError(f"duplicate candidate step: {step}")
        if candidate.get("experiment") != "loss_e2e_0723":
            raise ValueError(f"candidate step {step} has the wrong experiment")
        if candidate.get("arm") != arm:
            raise ValueError(f"candidate step {step} has the wrong arm")
        if candidate.get("selection_metric") != SELECTION_METRIC:
            raise ValueError(f"candidate step {step} has the wrong metric")
        if candidate.get("metric_direction") != "minimize":
            raise ValueError(f"candidate step {step} has the wrong metric direction")
        if candidate.get("evaluator_source_dirty") is not False:
            raise ValueError(f"candidate step {step} used a dirty evaluator source")
        summary_file = candidate.get("validation_summary_file")
        summary_hash = candidate.get("validation_summary_sha256")
        if not isinstance(summary_file, str) or not summary_file:
            raise ValueError(f"candidate step {step} has no summary filename")
        if not isinstance(summary_hash, str) or len(summary_hash) != 64:
            raise ValueError(f"candidate step {step} has an invalid summary hash")
        if Path(summary_file).name != summary_file:
            raise ValueError(f"candidate step {step} has a non-local summary filename")
        if summary_hash in summary_hashes:
            raise ValueError("candidate validation summary hashes are not unique")
        summary_hashes.add(summary_hash)

        token_sum = float(candidate.get("token_nll_sum", math.nan))
        token_count = int(candidate.get("effective_answer_token_count", 0))
        reported_nll = float(candidate.get("global_token_nll", math.nan))
        if not math.isfinite(token_sum) or token_sum <= 0:
            raise ValueError(f"candidate step {step} has an invalid token NLL sum")
        if token_count <= 0:
            raise ValueError(f"candidate step {step} has no effective tokens")
        recomputed_nll = token_sum / token_count
        if not math.isfinite(reported_nll) or not math.isclose(
            reported_nll,
            recomputed_nll,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"candidate step {step} has an inconsistent token NLL")
        checkpoint_hash = candidate.get("checkpoint_sha256")
        if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
            raise ValueError(f"candidate step {step} has an invalid checkpoint hash")
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError("candidate checkpoint hashes are not unique")
        checkpoint_hashes.add(checkpoint_hash)
        identity = {field: candidate.get(field) for field in shared_fields}
        if shared is None:
            shared = identity
        elif identity != shared:
            mismatches = [
                field for field in shared_fields if identity[field] != shared[field]
            ]
            raise ValueError(
                "candidate validation identities differ: " + ", ".join(mismatches)
            )
        row = dict(candidate)
        row["global_token_nll_recomputed"] = recomputed_nll
        normalized.append(row)
        by_step[step] = row
    if tuple(sorted(by_step)) != DECLARED_STEPS:
        raise ValueError("candidate steps must be exactly 4750, 9500, 14250, and 19000")
    normalized.sort(key=lambda row: row["candidate_step"])
    # Exact numerical ties prefer the earlier checkpoint as the lower-budget
    # deterministic choice; test/judge metrics are never consulted.
    selected = min(
        normalized,
        key=lambda row: (
            row["global_token_nll_recomputed"],
            row["candidate_step"],
        ),
    )
    return {
        "schema_version": 1,
        "experiment": "loss_e2e_0723",
        "arm": arm,
        "selection_metric": SELECTION_METRIC,
        "effective_token_definition": EFFECTIVE_TOKEN_DEFINITION,
        "metric_direction": "minimize",
        "tie_break": "lower candidate step",
        "declared_candidate_steps": list(DECLARED_STEPS),
        "validation_seed": 72,
        "train_ratio": 0.95,
        "test_or_judge_metrics_consulted": False,
        "candidates": normalized,
        "selected": {
            "candidate_step": selected["candidate_step"],
            "candidate_epoch": selected["candidate_epoch"],
            "checkpoint_file": selected["checkpoint_file"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "validation_summary_file": selected["validation_summary_file"],
            "validation_summary_sha256": selected["validation_summary_sha256"],
            "global_token_nll": selected["global_token_nll_recomputed"],
            "token_nll_sum": selected["token_nll_sum"],
            "effective_answer_token_count": selected["effective_answer_token_count"],
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("summaries", nargs="+")
    result.add_argument("--arm", choices=["control", "treatment"], required=True)
    result.add_argument("--output", required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    candidates = []
    for value in args.summaries:
        path = Path(value)
        candidate = _read_json(path)
        candidate["validation_summary_file"] = path.name
        candidate["validation_summary_sha256"] = _sha256_file(path)
        candidates.append(candidate)
    payload = select_best_candidate(candidates, args.arm)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, payload)
    print(json.dumps(payload["selected"]))


if __name__ == "__main__":
    main()
