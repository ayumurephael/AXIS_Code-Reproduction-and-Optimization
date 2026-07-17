"""Fail-closed audit for coherent counterfactual index v2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .loss_redesign import COUNTERFACTUAL_INDEX_VERSION, CounterfactualAXISDataset
from .model_utils import sha256_file


def audit_index(
    data: str,
    index_path: str,
    *,
    seed: int = 72,
    train_ratio: float = 0.95,
) -> dict:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != COUNTERFACTUAL_INDEX_VERSION:
        raise ValueError("counterfactual index version is not v2")
    policy = payload.get("policy", {})
    required_policy = {
        "name": "coherent_full_sequence_patch_with_explicit_state_targets",
        "phase_fallback": False,
        "post_patch_dual_source_validation": True,
        "same_length_required": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"counterfactual policy mismatch for {key}")
    if not float(policy.get("gamma_window", 0.0)) > 0.0:
        raise ValueError("Window gamma must be data-derived and positive")
    if not float(policy.get("gamma_local", 0.0)) > 0.0:
        raise ValueError("Local gamma must be data-derived and positive")

    dataset = CounterfactualAXISDataset(
        data,
        index_path,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
    )
    expected_keys = set()
    valid_normal = 0
    valid_anomalous = 0
    invalid = 0
    for dataset_index, path in enumerate(dataset.series_files):
        item = dataset[dataset_index]
        for window_index, (window, counterfactual) in enumerate(
            zip(item["analysis_data"], item["counterfactuals"])
        ):
            key = f"{path.name}:{window_index}"
            expected_keys.add(key)
            reference = payload["records"].get(key)
            if reference is None:
                raise ValueError(f"counterfactual index is missing {key}")
            state = int(bool(window["has_anomaly"]))
            if bool(reference.get("has_anomaly")) != bool(state):
                raise ValueError(f"anchor label mismatch for {key}")
            if not counterfactual["valid"]:
                invalid += 1
                if reference.get("kind") != "invalid":
                    raise ValueError(f"invalid pair has non-invalid kind for {key}")
                continue
            if int(reference.get("target_state", -1)) != 1 - state:
                raise ValueError(f"explicit counterfactual target mismatch for {key}")
            expected_kind = "self_normal_patch" if state else "residual_transplant"
            if reference.get("kind") != expected_kind:
                raise ValueError(f"counterfactual kind mismatch for {key}")
            start = int(window["window_range"]["start"])
            end = int(window["window_range"]["end"])
            anchor = item["time_series"]
            patched = counterfactual["series"]
            if patched.shape != anchor.shape:
                raise ValueError(f"full-sequence shape changed for {key}")
            outside = torch.ones(anchor.shape[0], dtype=torch.bool)
            outside[start:end] = False
            if not torch.equal(patched[outside], anchor[outside]):
                raise ValueError(f"counterfactual changed values outside target window for {key}")
            if not torch.equal(patched[start:end], counterfactual["values"]):
                raise ValueError(f"Window and full-sequence patch disagree for {key}")
            if not bool(torch.isfinite(patched).all()):
                raise ValueError(f"counterfactual contains non-finite values for {key}")
            if state:
                valid_anomalous += 1
            else:
                valid_normal += 1

    record_keys = set(payload["records"])
    if record_keys != expected_keys:
        missing = len(expected_keys - record_keys)
        extra = len(record_keys - expected_keys)
        raise ValueError(f"counterfactual key set mismatch: missing={missing}, extra={extra}")
    valid_pairs = valid_normal + valid_anomalous
    stats = payload.get("stats", {})
    if int(stats.get("valid_pairs", -1)) != valid_pairs:
        raise ValueError("counterfactual stats valid-pair count is stale")
    if int(stats.get("valid_residual_transplants", -1)) != valid_normal:
        raise ValueError("counterfactual stats normal count is stale")
    if int(stats.get("valid_anomaly_deletions", -1)) != valid_anomalous:
        raise ValueError("counterfactual stats anomalous count is stale")
    return {
        "ok": True,
        "version": COUNTERFACTUAL_INDEX_VERSION,
        "series": len(dataset),
        "records": len(expected_keys),
        "valid_pairs": valid_pairs,
        "valid_normal": valid_normal,
        "valid_anomalous": valid_anomalous,
        "invalid": invalid,
        "valid_pair_rate": valid_pairs / max(1, len(expected_keys)),
        "gamma_window": policy["gamma_window"],
        "gamma_local": policy["gamma_local"],
        "tau": policy["tau"],
        "index_sha256": sha256_file(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit_index(
        args.data,
        args.index,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
