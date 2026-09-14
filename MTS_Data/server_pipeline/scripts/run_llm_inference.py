from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.data_schema import ModelInput, convert_to_model_input
from src.mvaxis.llm_client import MockLLMClient, create_llm_client
from src.mvaxis.llm_eval import score_llm_outputs
from src.mvaxis.prompts import build_prompt, parse_json_output
from src.mvaxis.question_provider import ensure_questions
from src.mvaxis.training import load_encoder, load_hint_tuner
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json, write_jsonl


SYSTEM_PROMPT = (
    "Return exactly one valid json object following the requested schema. "
    "Do not include markdown fences. Use only the evidence supplied by the user. "
    "The top-level keys must be fact_check, reasoning_summary, final_answer, "
    "abnormality_score, and answer_confidence."
)


def _messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _parse_or_empty(text: str) -> Dict[str, Any]:
    try:
        return parse_json_output(text)
    except Exception:
        return {}


def _load_inputs(
    config: Dict[str, Any],
    split: str,
    limit: int | None,
    *,
    question_provider: str,
    question_seed: int,
    overwrite_questions: bool,
) -> List[ModelInput]:
    data_dir = config["data"]["output_dir"]
    rows = read_jsonl(ROOT / data_dir / f"{split}.jsonl")
    if limit is not None:
        rows = rows[:limit]
    rows = ensure_questions(
        rows,
        seed=question_seed,
        provider=question_provider,
        overwrite=overwrite_questions,
        overwrite_fixed_placeholder=True,
        copy_samples=False,
    )
    return [convert_to_model_input(x) for x in rows]


def _raw_examples(records: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    examples = []
    for record in records[: max(0, limit)]:
        target_fact = (record.get("target_output") or {}).get("fact_check") or {}
        parsed_fact = (record.get("parsed_response") or {}).get("fact_check") or {}
        examples.append(
            {
                "sample_id": record.get("sample_id"),
                "raw_response": record.get("raw_response"),
                "parsed_fact_check": parsed_fact,
                "target_fact_check": target_fact,
            }
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/torch_smoke.json")
    parser.add_argument("--llm-config", default="configs/deepseek_llm.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--mock-mode", default="target")
    parser.add_argument("--output", default="outputs/llm_inference.jsonl")
    parser.add_argument("--report", default="outputs/llm_inference_report.json")
    parser.add_argument("--raw-log-examples", type=int, default=2)
    parser.add_argument("--raw-log-path", default=None)
    parser.add_argument("--question-provider", default="bank", choices=["bank", "none"])
    parser.add_argument("--question-seed", type=int, default=520)
    parser.add_argument("--overwrite-questions", action="store_true")
    parser.add_argument(
        "--include-normal-counterpart",
        action="store_true",
        help="Oracle ablation only: expose the synthetic normal counterpart window to the LLM.",
    )
    args = parser.parse_args()

    config = load_json(ROOT / args.config)
    config["train_encoder"]["checkpoint_path"] = relative_to_root(ROOT, config["train_encoder"]["checkpoint_path"])
    config["train_hint_tuner"]["checkpoint_path"] = relative_to_root(ROOT, config["train_hint_tuner"]["checkpoint_path"])
    inputs = _load_inputs(
        config,
        args.split,
        args.limit,
        question_provider=args.question_provider,
        question_seed=int(args.question_seed),
        overwrite_questions=bool(args.overwrite_questions),
    )
    encoder = load_encoder(config)
    hint_tuner = load_hint_tuner(config)

    llm_config = load_json(ROOT / args.llm_config)
    real_client = None if args.mock else create_llm_client(llm_config)
    mock_client = MockLLMClient(args.mock_mode)

    records = []
    parsed_outputs = []
    for idx, mi in enumerate(inputs):
        hints = hint_tuner.make_soft_hint_summary(encoder, mi, int(config["model"]["top_k_channels"]))
        prompt = build_prompt(mi, hints, include_normal_counterpart=bool(args.include_normal_counterpart))
        if args.mock:
            response = mock_client.complete_with_target(mi.target_output)
        else:
            response = real_client.complete(_messages(prompt))
        parsed = _parse_or_empty(response.content)
        parsed_outputs.append(parsed)
        records.append(
            {
                "index": idx,
                "sample_id": f"{args.split}:{idx}",
                "prompt": prompt,
                "raw_response": response.content,
                "parsed_response": parsed,
                "target_output": mi.target_output,
                "latency_seconds": response.latency_seconds,
                "llm_raw_metadata": response.raw if args.mock else {"model": llm_config.get("model"), **response.raw},
            }
        )

    output_path = ROOT / args.output
    write_jsonl(records, output_path)
    metrics = score_llm_outputs(inputs, parsed_outputs)
    metrics["json_parse_rate"] = sum(float(isinstance((x or {}).get("fact_check"), dict)) for x in parsed_outputs) / max(1, len(parsed_outputs))
    metrics["output_path"] = str(output_path)
    metrics["mock"] = bool(args.mock)
    metrics["question_provider"] = args.question_provider
    metrics["question_seed"] = int(args.question_seed)
    metrics["overwrite_questions"] = bool(args.overwrite_questions)
    metrics["llm_config"] = {
        "provider": llm_config.get("provider"),
        "base_url": llm_config.get("base_url"),
        "model": llm_config.get("model"),
    }
    save_json(metrics, ROOT / args.report)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    examples = _raw_examples(records, int(args.raw_log_examples))
    if examples:
        raw_path = Path(args.raw_log_path) if args.raw_log_path else output_path.with_name(output_path.stem + "_raw_examples.json")
        if not raw_path.is_absolute():
            raw_path = ROOT / raw_path
        save_json({"examples": examples}, raw_path)
        print("== raw_response_examples ==")
        print(json.dumps({"raw_log_path": str(raw_path), "examples": examples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
