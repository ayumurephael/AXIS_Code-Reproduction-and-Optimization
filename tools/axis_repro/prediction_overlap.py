"""Measure exact response reuse between prediction JSONL files."""
from __future__ import annotations

import argparse
import json

from .common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    args = parser.parse_args()
    left = {(x["record_id"], x["mode"]): x["response"] for x in read_jsonl(args.left)}
    right = {(x["record_id"], x["mode"]): x["response"] for x in read_jsonl(args.right)}
    shared = sorted(set(left) & set(right))
    exact = sum(left[key] == right[key] for key in shared)
    print(json.dumps({
        "shared": len(shared),
        "exact_response_matches": exact,
        "exact_fraction": exact / len(shared) if shared else None,
    }))


if __name__ == "__main__":
    main()
