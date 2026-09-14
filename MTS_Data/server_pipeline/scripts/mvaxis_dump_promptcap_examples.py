from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoProcessor

from src.mvaxis.student_answer_provider import (
    AXIS_CHANNEL_HINT_TOKEN_PLACEHOLDER,
    AXIS_GLOBAL_HINT_TOKEN_PLACEHOLDER,
    AXIS_HINT_TOKEN_PLACEHOLDER,
    build_student_answer_messages,
)
from src.mvaxis.utils import load_json, read_jsonl, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--sample-ids-json", required=True, help="JSON file containing a list of sample_ids to export.")
    parser.add_argument("--max-window-rows", type=int, default=48)
    parser.add_argument("--prompt-cap", type=int, default=2048)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
    rows_by = {str(row.get("sample_id")): row for row in rows}
    llm_config = load_json(args.llm_config)
    processor = AutoProcessor.from_pretrained(
        llm_config["model"],
        trust_remote_code=bool(llm_config.get("trust_remote_code", True)),
    )
    axis_cfg = dict(llm_config.get("axis_hints") or {})
    global_tokens = int(axis_cfg.get("max_global_tokens", axis_cfg.get("num_global_tokens", 8)))
    channel_tokens = int(axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens", 16)))
    global_text = "<|global_hint|>" * global_tokens
    channel_text = "<|channel_hint|>" * channel_tokens

    sample_ids = json.loads(Path(args.sample_ids_json).read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = []
    for sample_id in sample_ids:
        row = rows_by[str(sample_id)]
        messages = build_student_answer_messages(
            row,
            use_axis_hints=True,
            include_global_hints=True,
            include_channel_hints=True,
            include_anomaly_score_text=True,
            include_anomaly_score_image_note=False,
            include_raw_image_note=False,
            max_window_rows=int(args.max_window_rows),
        )
        rendered: List[Dict[str, str]] = []
        for msg in messages:
            content = str(msg.get("content", ""))
            content = content.replace(AXIS_GLOBAL_HINT_TOKEN_PLACEHOLDER, global_text)
            content = content.replace(AXIS_CHANNEL_HINT_TOKEN_PLACEHOLDER, channel_text)
            content = content.replace(AXIS_HINT_TOKEN_PLACEHOLDER, global_text + channel_text)
            rendered.append({"role": str(msg.get("role", "user")), "content": content})
        vl_messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in rendered]
        prompt_text = processor.apply_chat_template(vl_messages, tokenize=False, add_generation_prompt=True)
        encoded = processor(text=[prompt_text], padding=True, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        original_prompt_tokens = int(input_ids.shape[0])
        kept_ids = input_ids[-int(args.prompt_cap) :] if int(args.prompt_cap) > 0 and original_prompt_tokens > int(args.prompt_cap) else input_ids
        truncated_prompt_text = processor.tokenizer.decode(
            kept_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        records.append(
            {
                "sample_id": str(sample_id),
                "original_prompt_tokens": original_prompt_tokens,
                "kept_prompt_tokens": int(kept_ids.shape[0]),
                "question": str(row.get("question") or ""),
                "truncated_prompt_text": truncated_prompt_text,
            }
        )

    save_json({"records": records}, args.output_json)
    lines: List[str] = []
    for rec in records:
        lines.append(f"## {rec['sample_id']}")
        lines.append(f"- original_prompt_tokens: {rec['original_prompt_tokens']}")
        lines.append(f"- kept_prompt_tokens: {rec['kept_prompt_tokens']}")
        lines.append(f"- question: {rec['question']}")
        lines.append("")
        lines.append("```text")
        lines.append(rec["truncated_prompt_text"])
        lines.append("```")
        lines.append("")
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"num_records": len(records), "output_json": args.output_json, "output_md": args.output_md}, ensure_ascii=False))


if __name__ == "__main__":
    main()
