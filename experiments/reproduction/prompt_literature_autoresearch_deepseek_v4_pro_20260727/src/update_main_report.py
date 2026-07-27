"""Append the final literature study and rendered prompts to the main report."""
from __future__ import annotations

from pathlib import Path


PROJECT = Path(
    "experiments/reproduction/"
    "prompt_literature_autoresearch_deepseek_v4_pro_20260727"
)
REPORT = Path(
    "experiments/reproduction/"
    "prompt_stage_a_deepseek_v4_pro_20260725/"
    "Prompt_Stage_A_消融实验结果分析.md"
)
MARKER = "# 文献驱动 Prompt Autoresearch 最终补充（2026-07-27）"


def main() -> None:
    report = REPORT.read_text(encoding="utf-8")
    if MARKER in report:
        raise RuntimeError("Final literature appendix already exists")
    narrative = (PROJECT / "to_human/literature_final_append.md").read_text(
        encoding="utf-8"
    ).rstrip()
    prompts = (PROJECT / "all-prompts.md").read_text(encoding="utf-8").rstrip()
    REPORT.write_text(
        report.rstrip()
        + "\n\n---\n\n"
        + narrative
        + "\n\n---\n\n"
        + prompts
        + "\n",
        encoding="utf-8",
    )
    print(
        {
            "report": str(REPORT),
            "narrative_chars": len(narrative),
            "prompt_chars": len(prompts),
        }
    )


if __name__ == "__main__":
    main()
