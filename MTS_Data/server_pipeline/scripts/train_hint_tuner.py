from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.training import load_encoder, train_hint_tuner
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    data_dir = config["data"]["output_dir"]
    train_rows = read_jsonl(ROOT / data_dir / "train.jsonl")
    val_rows = read_jsonl(ROOT / data_dir / "val.jsonl")
    encoder_path = relative_to_root(ROOT, config["train_encoder"]["checkpoint_path"])
    config["train_encoder"]["checkpoint_path"] = encoder_path
    config["train_hint_tuner"]["checkpoint_path"] = relative_to_root(ROOT, config["train_hint_tuner"]["checkpoint_path"])
    encoder = load_encoder(config)
    result = train_hint_tuner(
        encoder,
        [convert_to_model_input(x) for x in train_rows],
        [convert_to_model_input(x) for x in val_rows],
        config,
    )
    metrics_path = ROOT / "outputs" / "hint_tuner_metrics.json"
    save_json({"history": result["history"], "val_metrics": result["val_metrics"]}, metrics_path)
    print(json.dumps(result["val_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
