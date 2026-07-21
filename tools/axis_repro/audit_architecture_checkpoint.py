"""Fail-closed audit for AXIS architecture and ablation checkpoints."""
from __future__ import annotations

import argparse
import json

import torch

from .architecture_redesign import ARCHITECTURE_VARIANTS, audit_architecture_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-variant", choices=sorted(ARCHITECTURE_VARIANTS))
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = audit_architecture_checkpoint(
        payload,
        expected_variant=args.expected_variant,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
