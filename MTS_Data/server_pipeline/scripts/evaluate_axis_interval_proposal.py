from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.axis_interval import AXISMultivariateIntervalProposer, evaluate_interval_proposer
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_smoke.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--report", default="outputs/axis_interval_eval_report.json")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    checkpoint_path = relative_to_root(ROOT, config["train_interval_proposal"]["checkpoint_path"])
    model = AXISMultivariateIntervalProposer.load(checkpoint_path, config)
    rows = read_jsonl(ROOT / config["data"]["output_dir"] / f"{args.split}.jsonl")
    metrics = evaluate_interval_proposer(model, rows, config)
    payload = metrics.__dict__
    payload["split"] = args.split
    payload["checkpoint_path"] = checkpoint_path
    save_json(payload, ROOT / args.report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
