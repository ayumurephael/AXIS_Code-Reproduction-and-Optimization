"""Mechanistic diagnostics for question-type-routed prompt experiments."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.axis_repro.common import extract_choice, extract_true_false


EQUIVALENT_GROUPS = {
    "mc_f0": (
        "multiple_choice",
        ("route_r2_01_minimal", "route_r2_03_tf_boundary",
         "route_r2_04_oe_old_contract", "route_r2_05_oe_contract_coverage"),
    ),
    "mc_f1": (
        "multiple_choice",
        ("route_r2_02_mc_stable", "route_r2_06_full_routed"),
    ),
    "tf_p0": (
        "true_false",
        ("route_r2_01_minimal", "route_r2_02_mc_stable",
         "route_r2_04_oe_old_contract", "route_r2_05_oe_contract_coverage"),
    ),
    "tf_p1": (
        "true_false",
        ("route_r2_03_tf_boundary", "route_r2_06_full_routed"),
    ),
    "oe_s0": (
        "open_ended",
        ("route_r2_01_minimal", "route_r2_02_mc_stable",
         "route_r2_03_tf_boundary"),
    ),
    "oe_c1": (
        "open_ended",
        ("route_r2_05_oe_contract_coverage", "route_r2_06_full_routed"),
    ),
}


def decision(qtype: str, text: str):
    if qtype == "multiple_choice":
        return extract_choice(text)
    if qtype == "true_false":
        return extract_true_false(text)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_key = {(row["record_id"], row["mode"]): row for row in rows}
    equivalent = {}
    for name, (qtype, modes) in EQUIVALENT_GROUPS.items():
        ids = sorted({row["record_id"] for row in rows if row["question_type"] == qtype})
        available = [
            record_id for record_id in ids
            if all((record_id, mode) in by_key for mode in modes)
        ]
        equivalent[name] = {
            "question_type": qtype,
            "modes": list(modes),
            "records": len(available),
            "identical_responses": sum(
                len({by_key[(record_id, mode)]["response"] for mode in modes}) == 1
                for record_id in available
            ),
            "identical_decisions": (
                None if qtype == "open_ended" else sum(
                    len({decision(qtype, by_key[(record_id, mode)]["response"])
                         for mode in modes}) == 1
                    for record_id in available
                )
            ),
        }

    closed_errors = {}
    for mode in (
        "route_r2_01_minimal",
        "route_r2_02_mc_stable",
        "route_r2_03_tf_boundary",
    ):
        errors = []
        for row in rows:
            if row["mode"] != mode or row["question_type"] == "open_ended":
                continue
            gold = decision(row["question_type"], row["answer"])
            predicted = decision(row["question_type"], row["response"])
            if gold != predicted:
                errors.append(
                    {
                        "record_id": row["record_id"],
                        "question_type": row["question_type"],
                        "gold": gold,
                        "predicted": predicted,
                        "has_anomaly": row.get("has_anomaly"),
                        "question": row["question"],
                    }
                )
        closed_errors[mode] = errors

    phrase_patterns = {
        "no_further_needed": re.compile(
            r"no (?:further|additional) (?:evidence|analysis).*needed|"
            r"no need for (?:further|additional)",
            re.I,
        ),
        "categorical_no_anomaly": re.compile(
            r"no (?:clear )?evidence of (?:an|any )?anomal",
            re.I,
        ),
    }
    oe_behavior = {}
    for mode in sorted({row["mode"] for row in rows}):
        responses = [
            row["response"] for row in rows
            if row["mode"] == mode and row["question_type"] == "open_ended"
        ]
        if not responses:
            continue
        oe_behavior[mode] = {
            "records": len(responses),
            **{
                name: sum(bool(pattern.search(response)) for response in responses)
                for name, pattern in phrase_patterns.items()
            },
            "mentions_support_and_challenge": sum(
                "support" in response.lower() and "challeng" in response.lower()
                for response in responses
            ),
        }

    output = {
        "prediction_rows": len(rows),
        "equivalent_route_replicates": equivalent,
        "closed_decision_errors": closed_errors,
        "open_ended_phrase_behavior": oe_behavior,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"prediction_rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()