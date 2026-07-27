"""Render every preregistered literature prompt and lexical-router branch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.models.AXIS.prompt_stage_a import (
    LITERATURE_R1_MODES,
    LITERATURE_R2_MODES,
    LITERATURE_R3_MODES,
    LITERATURE_R4_MODES,
    LITERATURE_R5_MODES,
    build_question_prompt,
)


MODES = tuple(dict.fromkeys(
    LITERATURE_R1_MODES
    + LITERATURE_R2_MODES
    + LITERATURE_R3_MODES
    + LITERATURE_R4_MODES
    + LITERATURE_R5_MODES
))
MC_QUESTION = (
    "Which description best matches the window?\n"
    "A) Stable\nB) Spike\nC) Trend\nD) Cyclic"
)
OE_QUESTION = "What evidence supports or refutes an anomaly in this window?"
TF_QUESTIONS = {
    "default": "True or False: The window contains an anomalous deviation.",
    "explicit_negative": "True or False: There is no evidence of an anomaly.",
    "normality_word": "True or False: The window shows stable normal behavior.",
}


def render(mode: str, question_type: str, question: str) -> str:
    return build_question_prompt(
        question=question,
        question_type=question_type,
        start=7,
        end=10,
        serialized_values="10, 11, 42",
        local_hint_tokens="<|local_hint|>" * 3,
        fixed_hint_tokens="<|fixed_hint|>" * 30,
        mode=mode,
        aligned_rows=None,
    )


def variants(mode: str) -> list[tuple[str, str, str]]:
    result = [
        ("multiple_choice", "default", MC_QUESTION),
        ("open_ended", "default", OE_QUESTION),
        ("true_false", "default", TF_QUESTIONS["default"]),
    ]
    if mode in LITERATURE_R4_MODES + LITERATURE_R5_MODES:
        result.append((
            "true_false", "explicit_negative", TF_QUESTIONS["explicit_negative"]
        ))
    if mode in {
        "lit_r4_03_tf_nonanomaly_re2",
        "lit_r4_04_joint_semantic_tf_nonanomaly_re2",
    }:
        result.append((
            "true_false", "normality_word", TF_QUESTIONS["normality_word"]
        ))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    rows = []
    lines = [
        "# All literature-guided AXIS prompt templates",
        "",
        "Rendered from the frozen implementation with representative evidence. "
        "Router modes include every distinct lexical branch used in experiments.",
        "",
    ]
    for mode in MODES:
        lines.extend([f"## `{mode}`", ""])
        for question_type, variant, question in variants(mode):
            prompt = render(mode, question_type, question)
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            rows.append({
                "mode": mode,
                "question_type": question_type,
                "variant": variant,
                "prompt_sha256": digest,
                "prompt": prompt,
            })
            lines.extend([
                f"### {question_type} / {variant}",
                "",
                f"SHA-256: `{digest}`",
                "",
                "```text",
                prompt.rstrip(),
                "```",
                "",
            ])
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    Path(args.json).write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"modes": len(MODES), "prompts": len(rows)}))


if __name__ == "__main__":
    main()
