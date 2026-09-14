from __future__ import annotations

import argparse
import json

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.data_schema import convert_to_model_input
from src.mvaxis.eval import evaluate_closed_loop
from src.mvaxis.generator import generate_dataset
from src.mvaxis.prompts import build_prompt
from src.mvaxis.training import train_encoder, train_hint_tuner
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    config = load_json(ROOT / args.config)
    config["train_encoder"]["checkpoint_path"] = relative_to_root(ROOT, config["train_encoder"]["checkpoint_path"])
    config["train_hint_tuner"]["checkpoint_path"] = relative_to_root(ROOT, config["train_hint_tuner"]["checkpoint_path"])

    paths = generate_dataset(config)
    train_inputs = [convert_to_model_input(x) for x in read_jsonl(paths["train"])]
    val_inputs = [convert_to_model_input(x) for x in read_jsonl(paths["val"])]
    test_inputs = [convert_to_model_input(x) for x in read_jsonl(paths["test"])]

    encoder_result = train_encoder(train_inputs, val_inputs, config)
    hint_result = train_hint_tuner(encoder_result["model"], train_inputs, val_inputs, config)
    test_metrics = evaluate_closed_loop(encoder_result["model"], hint_result["model"], test_inputs)

    if test_inputs:
        hints = hint_result["model"].make_soft_hint_summary(
            encoder_result["model"],
            test_inputs[0],
            int(config["model"]["top_k_channels"]),
        )
        prompt = build_prompt(test_inputs[0], hints)
        prompt_path = ROOT / config["outputs"]["sample_prompt_path"]
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        test_metrics["sample_prompt_path"] = str(prompt_path)

    report = {
        "data_paths": paths,
        "encoder_val_metrics": encoder_result["val_metrics"],
        "hint_tuner_val_metrics": hint_result["val_metrics"],
        "test_metrics": test_metrics,
    }
    report_path = ROOT / config["outputs"]["report_path"]
    save_json(report, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

