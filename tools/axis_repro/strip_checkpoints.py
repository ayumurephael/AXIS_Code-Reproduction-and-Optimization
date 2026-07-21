"""Remove optimizer state from Phase-II checkpoints used for inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def strip_checkpoint_payload(payload: dict) -> dict:
    """Keep inference weights and all provenance needed to rebuild the architecture."""
    return {
        "model_state_dict": payload["model_state_dict"],
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "meta": payload.get("meta", {}),
        "reproduction_meta": payload.get("reproduction_meta", {}),
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    for name in args.checkpoints:
        source = Path(name)
        payload = torch.load(source, map_location="cpu", weights_only=False)
        target = source.with_name(f"{source.stem}_inference{source.suffix}")
        torch.save(strip_checkpoint_payload(payload), target)
        print(f"{target} {target.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
