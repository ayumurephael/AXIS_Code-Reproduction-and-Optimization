from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from src.models.MultiAXIS.config import MultiAxisConfig
from src.models.MultiAXIS.data import ManifestDataset
from src.models.MultiAXIS.prompting import HINT_TOKENS, MultiAxisPromptBuilder
from tools.multi_axis.datasets import (
    SUPPORTED_EVALUATION_DATASETS,
    validate_datasets,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit full, untruncated Multi-AXIS prompt lengths by dataset group"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_EVALUATION_DATASETS,
        required=True,
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MultiAxisConfig.load_json(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(HINT_TOKENS)})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    builder = MultiAxisPromptBuilder(
        tokenizer,
        fixed_tokens=config.hints.fixed_tokens,
        max_context_tokens=config.llm.max_context_tokens,
        include_joint=True,
        use_images=False,
    )
    result = {
        "max_context_tokens": config.llm.max_context_tokens,
        "datasets": {},
    }
    for dataset_name in validate_datasets(args.datasets):
        dataset = ManifestDataset(
            Path(args.manifest_dir) / f"eval_{dataset_name}.jsonl",
            Path(args.data_root),
            window_epsilon=config.hints.window_epsilon,
            window_scale=config.hints.window_scale,
            require_image=False,
        )
        group_by_record = {
            index: group_number
            for group_number, group in enumerate(dataset.groups)
            for index in group
        }
        rows = []
        for index in range(len(dataset)):
            sample = dataset[index]
            tokenized = builder.tokenize(
                [sample["question"]],
                [sample["interval"]],
                [sample["window_values"]],
                [sample["channel_ids"]],
                [sample["question_group"]],
                [None],
                answers=None,
            )
            rows.append(
                {
                    "index": index,
                    "sample_id": sample["sample_id"],
                    "base_sample_id": sample["base_sample_id"],
                    "group_number": group_by_record[index],
                    "prompt_tokens": tokenized.prompt_lengths[0],
                    "channels": sample["channel_count"],
                    "window_length": sample["interval"][1]
                    - sample["interval"][0],
                }
            )
        lengths = [row["prompt_tokens"] for row in rows]
        grouped_rows = collections.defaultdict(list)
        for row in rows:
            grouped_rows[row["group_number"]].append(row)
        groups = [
            {
                "group_number": group_number,
                "base_sample_id": values[0]["base_sample_id"],
                "records": len(values),
                "max_prompt_tokens": max(
                    row["prompt_tokens"] for row in values
                ),
                "sum_prompt_tokens": sum(
                    row["prompt_tokens"] for row in values
                ),
            }
            for group_number, values in sorted(grouped_rows.items())
        ]
        result["datasets"][dataset_name] = {
            "examples": len(rows),
            "minimum": min(lengths),
            "median": statistics.median(lengths),
            "mean": statistics.fmean(lengths),
            "maximum": max(lengths),
            "over_32768": sum(value > 32768 for value in lengths),
            "over_40960": sum(value > 40960 for value in lengths),
            "over_65536": sum(value > 65536 for value in lengths),
            "max_record": max(rows, key=lambda row: row["prompt_tokens"]),
            "mod8_over_32768": dict(
                collections.Counter(
                    str(row["group_number"] % 8)
                    for row in rows
                    if row["prompt_tokens"] > 32768
                )
            ),
            "groups": groups,
            "records_over_32768": [
                row for row in rows if row["prompt_tokens"] > 32768
            ],
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                dataset: {
                    key: value
                    for key, value in profile.items()
                    if key not in {"records_over_32768", "max_record", "groups"}
                }
                for dataset, profile in result["datasets"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
