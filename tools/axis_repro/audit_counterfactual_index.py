"""Fail-closed audit for coherent counterfactual index v3."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .loss_redesign import COUNTERFACTUAL_INDEX_VERSION, CounterfactualAXISDataset
from .model_utils import sha256_file


def _reuse_statistics(usage: Counter[str]) -> dict:
    counts = np.asarray(list(usage.values()), dtype=np.float64)
    if not counts.size:
        return {
            "unique_residual_donors": 0,
            "max_residual_donor_reuse": 0,
            "p95_residual_donor_reuse": 0.0,
            "top_residual_donor_share": 0.0,
            "effective_residual_donors": 0.0,
        }
    probabilities = counts / counts.sum()
    return {
        "unique_residual_donors": len(usage),
        "max_residual_donor_reuse": int(counts.max()),
        "p95_residual_donor_reuse": float(np.percentile(counts, 95, method="linear")),
        "top_residual_donor_share": float(counts.max() / counts.sum()),
        "effective_residual_donors": float(np.exp(-(probabilities * np.log(probabilities)).sum())),
    }


def audit_index(
    data: str,
    index_path: str,
    *,
    seed: int = 72,
    train_ratio: float = 0.95,
    expected_donor_top_m: int = 8,
) -> dict:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if int(payload.get("version", 0)) != COUNTERFACTUAL_INDEX_VERSION:
        raise ValueError(f"counterfactual index version is not v{COUNTERFACTUAL_INDEX_VERSION}")
    policy = payload.get("policy", {})
    required_policy = {
        "name": "coherent_full_sequence_patch_with_explicit_state_targets",
        "phase_fallback": False,
        "post_patch_dual_source_validation": True,
        "same_length_required": True,
        "donor_selection": "uniform_top_m_by_window_distance",
        "donor_sampling_key": "sha256(seed:anchor_key)",
        "replacement_across_anchors": True,
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"counterfactual policy mismatch for {key}")
    donor_top_m = int(policy.get("donor_top_m", 0))
    if donor_top_m != expected_donor_top_m or donor_top_m <= 0:
        raise ValueError("counterfactual donor Top-M does not match the preregistered value")
    if int(policy.get("donor_sampling_seed", -1)) != seed or int(payload.get("seed", -1)) != seed:
        raise ValueError("counterfactual donor sampling seed mismatch")
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
    donor_usage: Counter[str] = Counter()
    sampled_ranks: list[int] = []
    pool_sizes: list[int] = []
    eligible_counts: list[int] = []
    match_ratios: list[float] = []
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
                continue

            valid_normal += 1
            donor_key = str(reference.get("donor_key", ""))
            eligible = int(reference.get("eligible_donor_count", 0))
            pool = int(reference.get("top_m_pool_size", 0))
            rank = int(reference.get("donor_sample_rank", -1))
            ratio = float(reference.get("match_ratio", float("inf")))
            if not donor_key or eligible <= 0 or pool != min(donor_top_m, eligible):
                raise ValueError(f"invalid Top-M donor metadata for {key}")
            if not 0 <= rank < pool:
                raise ValueError(f"sampled donor rank lies outside Top-M pool for {key}")
            if not math.isfinite(ratio) or not 0.0 <= ratio <= float(policy["tau"]) + 1e-6:
                raise ValueError(f"sampled donor violates background compatibility for {key}")
            donor_usage[donor_key] += 1
            sampled_ranks.append(rank)
            pool_sizes.append(pool)
            eligible_counts.append(eligible)
            match_ratios.append(ratio)

    record_keys = set(payload["records"])
    if record_keys != expected_keys:
        missing = len(expected_keys - record_keys)
        extra = len(record_keys - expected_keys)
        raise ValueError(f"counterfactual key set mismatch: missing={missing}, extra={extra}")
    valid_pairs = valid_normal + valid_anomalous
    stats = payload.get("stats", {})
    exact_counts = {
        "valid_pairs": valid_pairs,
        "valid_residual_transplants": valid_normal,
        "valid_anomaly_deletions": valid_anomalous,
        "donor_top_m": donor_top_m,
    }
    for name, expected in exact_counts.items():
        if int(stats.get(name, -1)) != expected:
            raise ValueError(f"counterfactual stats {name} is stale")
    reuse = _reuse_statistics(donor_usage)
    for name, expected in reuse.items():
        actual = stats.get(name)
        if isinstance(expected, int):
            matches = int(actual if actual is not None else -1) == expected
        else:
            matches = actual is not None and math.isclose(float(actual), expected, rel_tol=1e-9, abs_tol=1e-12)
        if not matches:
            raise ValueError(f"counterfactual donor reuse statistic {name} is stale")

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
        "donor_top_m": donor_top_m,
        **reuse,
        "mean_sampled_donor_rank": float(np.mean(sampled_ranks)) if sampled_ranks else 0.0,
        "mean_top_m_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "mean_eligible_donor_count": float(np.mean(eligible_counts)) if eligible_counts else 0.0,
        "mean_match_ratio": float(np.mean(match_ratios)) if match_ratios else 0.0,
        "p95_match_ratio": float(np.percentile(match_ratios, 95, method="linear")) if match_ratios else 0.0,
        "index_sha256": sha256_file(index_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--expected-donor-top-m", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit_index(
        args.data,
        args.index,
        seed=args.seed,
        train_ratio=args.train_ratio,
        expected_donor_top_m=args.expected_donor_top_m,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
