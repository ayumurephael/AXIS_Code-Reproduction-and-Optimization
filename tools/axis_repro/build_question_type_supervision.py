"""Build a versioned, deterministic auxiliary-supervision cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from .question_type_objectives import SUPERVISION_CACHE_VERSION, build_supervision_record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_order(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/anomaly_llava_training_dataset")
    parser.add_argument("--counterfactual-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--expected-donor-top-m", type=int, default=1)
    parser.add_argument("--audit-per-type", type=int, default=200)
    args = parser.parse_args()

    data_root = Path(args.data)
    index_path = Path(args.counterfactual_index)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    actual_top_m = int(index.get("policy", {}).get("donor_top_m", -1))
    if actual_top_m != args.expected_donor_top_m:
        raise ValueError(f"counterfactual donor Top-M {actual_top_m} != {args.expected_donor_top_m}")
    if int(index.get("seed", -1)) != args.seed:
        raise ValueError("counterfactual index seed mismatch")

    records: dict[str, dict] = {}
    stats = Counter()
    valid_by_type: dict[str, list[dict]] = defaultdict(list)
    for series_name in index["train_series"]:
        payload = json.loads((data_root / "series" / series_name).read_text(encoding="utf-8"))
        for window_index, window in enumerate(payload["windows"]):
            key = f"{series_name}:{window_index}"
            counterfactual = index["records"].get(key, {"valid": False, "kind": "invalid"})
            record = build_supervision_record(window, counterfactual)
            records[key] = record
            stats[f"rows/{record['question_type']}"] += 1
            for field in ("mc_pair_valid", "tf_pair_valid", "oe_pair_valid"):
                if record[field]:
                    stats[f"valid/{field}"] += 1
                    valid_by_type[field].append({
                        "record_id": key,
                        "question": window.get("question", ""),
                        "answer": window.get("answer", ""),
                        **record,
                    })

    cache = {
        "version": SUPERVISION_CACHE_VERSION,
        "seed": args.seed,
        "train_ratio": float(index["train_ratio"]),
        "train_series": index["train_series"],
        "counterfactual_index_sha256": sha256_file(index_path),
        "policy": {
            "name": "final_question_type_auxiliary_supervision_v1",
            "question_type_routing": True,
            "tf_high_precision_state_predicates": True,
            "tf_requires_metadata_consistency": True,
            "oe_high_precision_neutral_questions": True,
            "oe_explicit_answer_state_first_two_sentences": True,
            "oe_deletion_only": True,
            "answer_loss_keeps_all_rows": True,
            "expected_donor_top_m": args.expected_donor_top_m,
        },
        "stats": dict(sorted(stats.items())),
        "records": records,
    }
    atomic_json(Path(args.output), cache)

    audit_rows = []
    for field in ("tf_pair_valid", "oe_pair_valid"):
        ordered = sorted(valid_by_type[field], key=lambda row: stable_order(args.seed, row["record_id"]))
        audit_rows.extend(ordered[: args.audit_per_type])
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, audit_path)
    print(json.dumps({
        "cache": str(Path(args.output)),
        "cache_sha256": sha256_file(Path(args.output)),
        "stats": cache["stats"],
        "audit_rows": len(audit_rows),
    }))


if __name__ == "__main__":
    main()
