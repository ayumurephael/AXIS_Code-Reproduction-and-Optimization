"""Audit, assemble, summarize, and render the frozen full-284 evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_PREDICTIONS = PROJECT / "artifacts/inference_raw/predictions.jsonl"
RAW_SCORES = PROJECT / "artifacts/judge_raw/scores.jsonl"
ASSEMBLED = PROJECT / "artifacts/assembled"
MANIFEST = REPO_ROOT / "experiments/reproduction/manifests/full.json"
ASSEMBLER = PROJECT / "src/assemble_final_components.py"
RENDERER = PROJECT / "src/render_final_report.py"
RAW_AUDIT = PROJECT / "artifacts/raw_audit.json"
ASSEMBLED_AUDIT = PROJECT / "artifacts/assembled_audit.json"


def environment() -> dict[str, str]:
    output = os.environ.copy()
    output["PYTHONDONTWRITEBYTECODE"] = "1"
    return output


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="")
    if result.stdout.strip():
        print(result.stdout, end="")
    return result.stdout


def audit(
    predictions: Path,
    scores: Path,
    modes: tuple[str, ...],
    output: Path,
) -> None:
    stdout = run(
        [
            sys.executable,
            "-m",
            "tools.axis_repro.audit_results",
            "--predictions",
            str(predictions),
            "--scores",
            str(scores),
            "--manifest",
            str(MANIFEST),
            "--modes",
            *modes,
            "--expected-model",
            "deepseek-v4-pro",
            "--expected-provider",
            "deepseek",
            "--expected-enable-thinking",
            "auto",
            "--allowed-methods",
            "final_score_top_logprobs",
            "exact_sample_mean_20",
            "--require-prompt-hashes",
        ]
    )
    summary = json.loads(stdout)
    if not summary["ok"]:
        raise RuntimeError(f"Audit failed: {summary['errors']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for required in (RAW_PREDICTIONS, RAW_SCORES, MANIFEST):
        if not required.is_file():
            raise FileNotFoundError(required)

    audit(
        RAW_PREDICTIONS,
        RAW_SCORES,
        ("base", "lit_r1_10_triplet_re2"),
        RAW_AUDIT,
    )
    ASSEMBLED.mkdir(parents=True, exist_ok=True)
    assembled_predictions = ASSEMBLED / "predictions.jsonl"
    assembled_scores = ASSEMBLED / "scores.jsonl"
    run(
        [
            sys.executable,
            str(ASSEMBLER),
            "predictions",
            "--raw-predictions",
            str(RAW_PREDICTIONS),
            "--output",
            str(assembled_predictions),
            "--provenance",
            str(ASSEMBLED / "prediction_provenance.json"),
            "--expected-records",
            "284",
        ]
    )
    run(
        [
            sys.executable,
            str(ASSEMBLER),
            "scores",
            "--assembled-predictions",
            str(assembled_predictions),
            "--raw-scores",
            str(RAW_SCORES),
            "--output",
            str(assembled_scores),
            "--provenance",
            str(ASSEMBLED / "score_provenance.json"),
        ]
    )
    audit(
        assembled_predictions,
        assembled_scores,
        ("base", "lit_r1_08_oe_re2", "lit_r1_10_triplet_re2"),
        ASSEMBLED_AUDIT,
    )
    run(
        [
            sys.executable,
            "-m",
            "tools.axis_repro.table_runner",
            "--scores",
            str(assembled_scores),
            "--output-prefix",
            str(PROJECT / "tables/full284"),
        ]
    )
    run([sys.executable, str(RENDERER)])


if __name__ == "__main__":
    main()
