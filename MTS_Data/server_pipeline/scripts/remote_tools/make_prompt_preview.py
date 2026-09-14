from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    base = Path("outputs/runs/doflow_hint_ablation_preview_v2")
    variants = [
        "all_text_hints",
        "no_channel_hints",
        "no_global_hints",
        "no_evidence_card",
    ]
    lines = [
        "# Qwen Prompt Preview: Hint Ablation",
        "",
        "Same split/test order and same anomaly-head proposal mode.",
        "Generated with --mock, so Qwen was not called.",
        "",
    ]
    for variant in variants:
        data = json.loads((base / f"{variant}_proposal_limit4_prompt_examples.json").read_text(encoding="utf-8"))
        chosen = next(
            (
                example
                for example in data["examples"]
                if (example.get("target_fact_check") or {}).get("is_anomalous")
            ),
            data["examples"][0],
        )
        proposal = chosen["proposal"]
        lines.extend(
            [
                f"## {variant}",
                "",
                f"sample_id: {chosen['sample_id']}",
                f"proposal: [{proposal.get('start')}, {proposal.get('end')}], score={proposal.get('proposal_score')}",
                "target_fact_check:",
                "```json",
                json.dumps(chosen["target_fact_check"], ensure_ascii=False, indent=2),
                "```",
                "prompt:",
                "```text",
                chosen["prompt"],
                "```",
                "",
            ]
        )
    out = base / "prompt_preview_anomalous.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
