"""Write/print the deterministic seed-72 loss experiment data audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.AXIS.dataset import AXISAnomalyQADataset

from .train_loss_e2e_ddp import build_data_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--output")
    args = parser.parse_args()
    dataset = AXISAnomalyQADataset(
        args.data,
        split="train",
        train_ratio=0.95,
        seed=args.seed,
    )
    audit = build_data_audit(dataset, args.seed)
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
