from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.generator import generate_dataset
from src.mvaxis.utils import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/doflow.json")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    paths = generate_dataset(config)
    print(json.dumps({"generated": paths}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
