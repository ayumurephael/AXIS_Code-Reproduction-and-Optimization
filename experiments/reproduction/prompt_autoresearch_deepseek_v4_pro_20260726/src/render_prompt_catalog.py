"""Render and hash every screening prompt before model inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.models.AXIS.prompt_stage_a import (
    LITERATURE_R1_MODES,
    LITERATURE_R2_MODES,
    PARETO_SCREEN_MODES,
    ROUTED_R2_MODES,
    ROUTED_R3_MODES,
    build_question_prompt,
)

QUESTIONS = {
    "multiple_choice": "Which description best matches the window?\nA) Stable\nB) Spike\nC) Trend\nD) Cyclic",
    "open_ended": "What evidence supports or refutes an anomaly in this window?",
    "true_false": "True or False: The window contains an anomalous deviation.",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode-set",
        choices=(
            "screening",
            "routed_r2",
            "routed_r3",
            "literature_r1",
            "literature_r2",
        ),
        default="screening",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode_set == "screening":
        modes = PARETO_SCREEN_MODES
    elif args.mode_set == "routed_r2":
        modes = ROUTED_R2_MODES
    elif args.mode_set == "routed_r3":
        modes = ROUTED_R3_MODES
    elif args.mode_set == "literature_r1":
        modes = LITERATURE_R1_MODES
    else:
        modes = LITERATURE_R2_MODES
    titles = {
        "screening": "Screening round 1 prompt catalog",
        "routed_r2": "Development round 2 routed prompt catalog",
        "routed_r3": "Development round 3 routed prompt catalog",
        "literature_r1": "Literature-guided round 1 prompt catalog",
        "literature_r2": "Literature-guided round 2 prompt catalog",
    }
    title = titles[args.mode_set]

    rows = []
    markdown = [
        f"# {title}",
        "",
        "Rendered before inference with a fixed illustrative window. Runtime",
        "values, steps, latent placeholders, and questions are substituted per sample.",
        "",
    ]
    for mode in modes:
        for question_type, question in QUESTIONS.items():
            prompt = build_question_prompt(
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
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            rows.append(
                {
                    "mode": mode,
                    "question_type": question_type,
                    "prompt_sha256": digest,
                    "prompt": prompt,
                }
            )
            markdown.extend(
                [
                    f"## {mode} / {question_type}",
                    "",
                    f"SHA-256: `{digest}`",
                    "",
                    "```text",
                    prompt.rstrip(),
                    "```",
                    "",
                ]
            )
    output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps({"rows": len(rows), "modes": len(modes), "mode_set": args.mode_set}))


if __name__ == "__main__":
    main()