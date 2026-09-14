from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()


VARIANTS: Dict[str, List[str]] = {
    "all_text_hints": [
        "--disable-axis-embedding-hints",
        "--force-soft-hints",
    ],
    "no_channel_hints": [
        "--disable-axis-embedding-hints",
        "--force-soft-hints",
        "--hide-channel-hints",
    ],
    "no_global_hints": [
        "--disable-axis-embedding-hints",
        "--force-soft-hints",
        "--hide-global-hints",
    ],
    "no_evidence_card": [
        "--disable-axis-embedding-hints",
        "--force-soft-hints",
        "--hide-evidence-card",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_semantic_scale.json")
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--interval-source", choices=["truth", "proposal"], default="truth")
    parser.add_argument("--interval-proposer-checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/runs/doflow_hint_ablation")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--raw-log-examples", type=int, default=2)
    parser.add_argument("--dump-prompt-examples", type=int, default=1)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for name, flags in VARIANTS.items():
        output = out_dir / f"{name}_{args.interval_source}_limit{args.limit}.jsonl"
        report = out_dir / f"{name}_{args.interval_source}_limit{args.limit}_report.json"
        raw = out_dir / f"{name}_{args.interval_source}_limit{args.limit}_raw_examples.json"
        prompts = out_dir / f"{name}_{args.interval_source}_limit{args.limit}_prompt_examples.json"
        cmd = [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "run_llm_on_proposals.py"),
            "--config",
            args.config,
            "--llm-config",
            args.llm_config,
            "--split",
            args.split,
            "--limit",
            str(args.limit),
            "--interval-source",
            args.interval_source,
            "--window-size",
            str(args.window_size),
            "--stride",
            str(args.stride),
            "--output",
            str(output.relative_to(ROOT)),
            "--report",
            str(report.relative_to(ROOT)),
            "--raw-log-examples",
            str(args.raw_log_examples),
            "--raw-log-path",
            str(raw.relative_to(ROOT)),
            "--dump-prompt-examples",
            str(args.dump_prompt_examples),
            "--prompt-log-path",
            str(prompts.relative_to(ROOT)),
            *flags,
        ]
        if args.interval_proposer_checkpoint:
            cmd.extend(["--interval-proposer-checkpoint", args.interval_proposer_checkpoint])
        if args.mock:
            cmd.append("--mock")
        print("== run", name, "==")
        subprocess.run(cmd, cwd=ROOT, check=True)
        reports[name] = json.loads(report.read_text())
    summary = {
        "config": args.config,
        "llm_config": args.llm_config,
        "split": args.split,
        "limit": args.limit,
        "interval_source": args.interval_source,
        "variants": reports,
    }
    summary_path = out_dir / f"summary_{args.interval_source}_limit{args.limit}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
