"""Build the frozen inventory of every implemented AXIS prompt mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.AXIS.prompt_stage_a import (
    MODE_SPECS,
    build_question_prompt,
)


METRICS = (
    "multiple_choice/final",
    "multiple_choice/correctness",
    "multiple_choice/reasoning_quality",
    "open_ended/final",
    "open_ended/accuracy",
    "open_ended/completeness",
    "open_ended/relevance",
    "true_false/final",
    "true_false/correctness",
    "true_false/justification_quality",
)
OE_METRICS = tuple(metric for metric in METRICS if metric.startswith("open_ended/"))
TOLERANCE = 1e-6


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sample_prompt(mode: str, question_type: str) -> str:
    questions = {
        "multiple_choice": (
            "Which option best describes the window?\n\n"
            "A) Normal.\n\nB) Anomalous."
        ),
        "open_ended": (
            "What evidence supports or challenges an anomaly assessment?"
        ),
        "true_false": (
            "True or False: There is no evidence of an anomaly in the window."
        ),
    }
    return build_question_prompt(
        question=questions[question_type],
        question_type=question_type,
        start=10,
        end=13,
        serialized_values="100, 110, 90",
        local_hint_tokens="<|local_hint|>" * 3,
        fixed_hint_tokens="<|fixed_hint|>" * 30,
        mode=mode,
        aligned_rows=(
            "Step 0010 | value=+00100 | context=<|local_hint|>\n"
            "Step 0011 | value=+00110 | context=<|local_hint|>\n"
            "Step 0012 | value=+00090 | context=<|local_hint|>"
        ),
    )


def source_size(path: Path) -> int:
    value = str(path).replace("\\", "/").lower()
    if "prompt_stage_a_" in value or "prompt_followup_" in value:
        return 140
    if "prompt_revised_contract_" in value:
        return 140
    if "holdout" in value:
        return 48
    if "pooled96" in value or "development-round-2" in value:
        return 96
    if "development-round-3" in value:
        return 96
    if "validation72" in value:
        return 72
    if "/validation/" in value or "\\validation\\" in str(path).lower():
        return 72
    if "screening24" in value or "screening-round-1" in value:
        return 24
    if "/screening/" in value or "\\screening\\" in str(path).lower():
        return 24
    if "literature-round-1" in value:
        return 24
    if "literature-round-2" in value:
        return 72
    return 0


def summarize_deltas(deltas: dict[str, float]) -> dict[str, Any]:
    ordered = {metric: float(deltas[metric]) for metric in METRICS}
    values = list(ordered.values())
    return {
        "deltas": ordered,
        "nonnegative_dimensions": sum(
            value >= -TOLERANCE for value in values
        ),
        "worst_delta": min(values),
        "mean_delta": sum(values) / len(values),
        "positive_dimensions": sum(value > TOLERANCE for value in values),
        "positive_oe_dimensions": sum(
            ordered[metric] > TOLERANCE for metric in OE_METRICS
        ),
    }


def load_table_evidence(repo: Path) -> list[dict[str, Any]]:
    relative_paths = (
        "experiments/reproduction/prompt_stage_a_deepseek_v4_pro_20260725/all_table.json",
        "experiments/reproduction/prompt_followup_deepseek_v4_pro_20260726/table_followup.json",
        "experiments/reproduction/prompt_revised_contract_deepseek_v4_pro_20260726/table_revised_contract.json",
    )
    evidence: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = repo / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data["models"]
        baseline = models["base"]
        for mode, metrics in models.items():
            if mode == "base":
                continue
            deltas = {
                metric: float(metrics[metric]) - float(baseline[metric])
                for metric in METRICS
            }
            evidence.append(
                {
                    "mode": mode,
                    "source": relative,
                    "records": source_size(Path(relative)),
                    **summarize_deltas(deltas),
                }
            )
    return evidence


def load_screening_evidence(repo: Path) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    root = repo / "experiments/reproduction"
    for path in sorted(root.rglob("screening_summary.json")):
        if "prompt_final_full284" in str(path):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data.get("rankings", []):
            deltas = {
                metric: float(row["deltas"][metric]) for metric in METRICS
            }
            evidence.append(
                {
                    "mode": row["mode"],
                    "source": path.relative_to(repo).as_posix(),
                    "records": source_size(path),
                    **summarize_deltas(deltas),
                }
            )
    return evidence


def category(mode: str) -> str:
    if mode == "base":
        return "baseline"
    if mode.startswith("wo_"):
        return "released_ablation_control"
    if mode.startswith("lit_"):
        return "literature_prompt"
    if mode.startswith("route_"):
        return "routed_prompt"
    if mode.startswith("pareto_"):
        return "pareto_prompt"
    if mode.startswith("expert_") or "evidence_contract_revised" in mode:
        return "followup_contract_prompt"
    return "stage_a_prompt"


def select_status(
    mode: str,
    changes_oe: bool,
    rows: list[dict[str, Any]],
) -> tuple[str, str, dict[str, Any] | None]:
    if mode == "base":
        return "baseline", "Canonical comparison mode.", None
    if mode.startswith("wo_"):
        return (
            "excluded",
            "Released ablation control; paper and reproduction evidence show "
            "hint/window removal is harmful, not a prompt-improvement candidate.",
            None,
        )
    if not changes_oe:
        return (
            "excluded",
            "OE prompt branch is exact Baseline and cannot satisfy the frozen "
            "OE-change requirement.",
            None,
        )
    if not rows:
        return "excluded", "No audited positive historical evidence.", None

    qualifying = [
        row
        for row in rows
        if row["nonnegative_dimensions"] >= 9
        and row["worst_delta"] >= -0.15
        and row["mean_delta"] > 0.0
        and row["positive_oe_dimensions"] >= 1
    ]
    if not qualifying:
        best = max(
            rows,
            key=lambda row: (
                row["nonnegative_dimensions"],
                row["worst_delta"],
                row["mean_delta"],
            ),
        )
        return (
            "excluded",
            "Did not meet the wide 9/10, worst>=-0.15, positive-mean, "
            "OE-improvement screen.",
            best,
        )

    seed = max(
        qualifying,
        key=lambda row: (
            row["records"],
            row["nonnegative_dimensions"],
            row["worst_delta"],
            row["mean_delta"],
        ),
    )
    contradictions = [
        row
        for row in rows
        if row["records"] > seed["records"]
        and (
            row["nonnegative_dimensions"] < 9
            or row["worst_delta"] < -0.15
        )
    ]
    if contradictions:
        strongest = max(
            contradictions,
            key=lambda row: (
                row["records"],
                -row["nonnegative_dimensions"],
                -row["worst_delta"],
            ),
        )
        return (
            "excluded",
            "A later broader audited evaluation contradicts the initial "
            "wide-screen signal.",
            strongest,
        )
    return (
        "advanced_full284",
        "Passes the frozen wide screen and has no later broader contradiction.",
        seed,
    )


def build_inventory(repo: Path) -> dict[str, Any]:
    evidence_by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_table_evidence(repo) + load_screening_evidence(repo):
        if row["mode"] in MODE_SPECS:
            evidence_by_mode[row["mode"]].append(row)

    base_oe = sample_prompt("base", "open_ended")
    modes = []
    for mode, condition in MODE_SPECS.items():
        prompt_hashes = {
            question_type: sha256_text(sample_prompt(mode, question_type))
            for question_type in (
                "multiple_choice",
                "open_ended",
                "true_false",
            )
        }
        changes_oe = (
            prompt_hashes["open_ended"] != sha256_text(base_oe)
            or condition.answer_boundary
            or condition.remove_fixed_hint
            or condition.remove_local_hint
            or condition.remove_window_values
        )
        rows = evidence_by_mode.get(mode, [])
        status, reason, decisive = select_status(mode, changes_oe, rows)
        modes.append(
            {
                "mode": mode,
                "category": category(mode),
                "description": condition.description,
                "changes_open_ended_prompt": changes_oe,
                "sample_prompt_sha256": prompt_hashes,
                "status": status,
                "reason": reason,
                "decisive_evidence": decisive,
                "all_evidence": sorted(
                    rows,
                    key=lambda row: (
                        row["records"],
                        row["source"],
                    ),
                ),
            }
        )

    counts = defaultdict(int)
    for row in modes:
        counts[row["status"]] += 1
    advanced = [
        row["mode"] for row in modes if row["status"] == "advanced_full284"
    ]
    return {
        "registered_mode_count": len(MODE_SPECS),
        "tolerance": TOLERANCE,
        "status_counts": dict(sorted(counts.items())),
        "advanced_modes": advanced,
        "modes": modes,
    }


def render_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Frozen Prompt Screening Inventory",
        "",
        f"- Registered modes: {inventory['registered_mode_count']}",
        f"- Status counts: `{inventory['status_counts']}`",
        f"- Full-284 candidates: `{inventory['advanced_modes']}`",
        "",
        "| Mode | Category | OE changed? | Decision | Decisive evidence | Reason |",
        "|---|---|:---:|---|---|---|",
    ]
    for row in inventory["modes"]:
        evidence = row["decisive_evidence"]
        if evidence is None:
            evidence_text = "—"
        else:
            evidence_text = (
                f"{evidence['records']} QA; "
                f"{evidence['nonnegative_dimensions']}/10; "
                f"worst {evidence['worst_delta']:+.4f}; "
                f"mean {evidence['mean_delta']:+.4f}"
            )
        reason = row["reason"].replace("|", "\\|")
        lines.append(
            f"| `{row['mode']}` | {row['category']} | "
            f"{'yes' if row['changes_open_ended_prompt'] else 'no'} | "
            f"`{row['status']}` | {evidence_text} | {reason} |"
        )
    lines.extend(
        [
            "",
            "The machine-readable JSON retains every historical evidence row, "
            "all ten deltas, and representative prompt hashes.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = repo / output
    output.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(repo)
    (output / "screening_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "screening_inventory.md").write_text(
        render_markdown(inventory),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "registered_modes": inventory["registered_mode_count"],
                "status_counts": inventory["status_counts"],
                "advanced_modes": inventory["advanced_modes"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

