from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.eval import evaluate_closed_loop
from src.mvaxis.prompts import build_prompt
from src.mvaxis.training import load_encoder, load_hint_tuner
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    data_dir = config["data"]["output_dir"]
    rows = read_jsonl(ROOT / data_dir / f"{args.split}.jsonl")
    inputs = [convert_to_model_input(x) for x in rows]
    config["train_encoder"]["checkpoint_path"] = relative_to_root(ROOT, config["train_encoder"]["checkpoint_path"])
    config["train_hint_tuner"]["checkpoint_path"] = relative_to_root(ROOT, config["train_hint_tuner"]["checkpoint_path"])
    encoder = load_encoder(config)
    hint_tuner = load_hint_tuner(config)
    metrics = evaluate_closed_loop(encoder, hint_tuner, inputs)
    if inputs:
        hints = hint_tuner.make_soft_hint_summary(encoder, inputs[0], int(config["model"]["top_k_channels"]))
        prompt = build_prompt(inputs[0], hints)
        sample_prompt_path = ROOT / config["outputs"]["sample_prompt_path"]
        sample_prompt_path.parent.mkdir(parents=True, exist_ok=True)
        sample_prompt_path.write_text(prompt, encoding="utf-8")
        metrics["sample_prompt_path"] = str(sample_prompt_path)
    report_path = ROOT / config["outputs"]["report_path"]
    save_json(metrics, report_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
