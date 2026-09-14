from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import train_interval_proposer
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_smoke.json")
    parser.add_argument("--report", default="outputs/axis_interval_train_metrics.json")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    config["train_interval_proposal"]["checkpoint_path"] = relative_to_root(
        ROOT, config["train_interval_proposal"]["checkpoint_path"]
    )
    data_dir = ROOT / config["data"]["output_dir"]
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    result = train_interval_proposer(train_rows, val_rows, config)
    output_path = ROOT / args.report
    save_json({"history": result["history"], "val_metrics": result["val_metrics"]}, output_path)
    print(json.dumps(result["val_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
