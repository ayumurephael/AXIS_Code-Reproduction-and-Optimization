from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from transformers import AutoProcessor

from src.mvaxis.llm_client import LocalHFVLChatClient
from src.mvaxis.semantic_alignment import extract_teacher_answer
from src.mvaxis.student_answer_provider import (
    AXIS_CHANNEL_HINT_TOKEN_PLACEHOLDER,
    AXIS_GLOBAL_HINT_TOKEN_PLACEHOLDER,
    AXIS_HINT_TOKEN_PLACEHOLDER,
    build_student_answer_messages,
)
from src.mvaxis.utils import load_json, read_jsonl, save_json


def _sample_keys(row: Dict[str, Any]) -> List[str]:
    keys: List[str] = []
    for key in ("sample_id", "id", "image_id", "question_id"):
        value = row.get(key)
        if value is not None:
            keys.append(str(value))
    return keys


def _teacher_lookup(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        for key in _sample_keys(row):
            lookup[key] = row
    return lookup


def _teacher_record(sample: Dict[str, Any], teacher_rows: List[Dict[str, Any]], teacher_lookup: Dict[str, Dict[str, Any]], index: int) -> Dict[str, Any]:
    for key in _sample_keys(sample):
        candidate = teacher_lookup.get(key)
        if candidate is not None:
            return candidate
    if 0 <= index < len(teacher_rows):
        return teacher_rows[index]
    raise KeyError(f"No teacher row found for sample index {index}")


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _summarize(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": float(max(values)),
    }


def _render_hint_messages(messages: List[Dict[str, str]], *, global_tokens: int, channel_tokens: int) -> List[Dict[str, str]]:
    rendered: List[Dict[str, str]] = []
    global_text = LocalHFVLChatClient.GLOBAL_HINT_TOKEN * int(global_tokens)
    channel_text = LocalHFVLChatClient.CHANNEL_HINT_TOKEN * int(channel_tokens)
    for msg in messages:
        content = str(msg.get("content", ""))
        content = content.replace(AXIS_GLOBAL_HINT_TOKEN_PLACEHOLDER, global_text)
        content = content.replace(AXIS_CHANNEL_HINT_TOKEN_PLACEHOLDER, channel_text)
        content = content.replace(AXIS_HINT_TOKEN_PLACEHOLDER, global_text + channel_text)
        rendered.append({"role": str(msg.get("role", "user")), "content": content})
    return rendered


def _vl_messages(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    vl_messages: List[Dict[str, Any]] = []
    for msg in messages:
        vl_messages.append({"role": msg["role"], "content": [{"type": "text", "text": msg["content"]}]})
    return vl_messages


def _question_type(sample: Dict[str, Any]) -> str:
    fact = (sample.get("target_output") or {}).get("fact_check") or {}
    return str(
        sample.get("question_answer_type")
        or fact.get("question_answer_type")
        or sample.get("question_type")
        or "open"
    ).lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-window-rows", type=int, default=48)
    parser.add_argument("--max-target-tokens", type=int, default=192)
    parser.add_argument("--assume-global-tokens", type=int, default=-1)
    parser.add_argument("--assume-channel-tokens", type=int, default=-1)
    parser.add_argument("--prompt-cap", type=int, action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    llm_config = load_json(args.llm_config)
    processor = AutoProcessor.from_pretrained(
        llm_config["model"],
        trust_remote_code=bool(llm_config.get("trust_remote_code", True)),
    )
    axis_cfg = dict(llm_config.get("axis_hints") or {})
    global_tokens = int(args.assume_global_tokens if args.assume_global_tokens >= 0 else axis_cfg.get("max_global_tokens", axis_cfg.get("num_global_tokens", 8)))
    channel_tokens = int(args.assume_channel_tokens if args.assume_channel_tokens >= 0 else axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens", 16)))

    rows = read_jsonl(args.data)
    teacher_rows = read_jsonl(args.teacher)
    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]
    teacher_lookup = _teacher_lookup(teacher_rows)

    prompt_tokens: List[int] = []
    target_tokens_raw: List[int] = []
    target_tokens_trimmed: List[int] = []
    full_tokens: List[int] = []
    by_type: Dict[str, Dict[str, List[int]]] = {}
    longest: List[Dict[str, Any]] = []

    caps = [int(v) for v in args.prompt_cap if int(v) > 0]
    capped_full: Dict[int, List[int]] = {cap: [] for cap in caps}
    capped_prompt: Dict[int, List[int]] = {cap: [] for cap in caps}

    tokenizer = processor.tokenizer

    for idx, row in enumerate(rows):
        teacher_row = _teacher_record(row, teacher_rows, teacher_lookup, idx)
        teacher_answer = extract_teacher_answer(teacher_row)
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
        rendered = _render_hint_messages(messages, global_tokens=global_tokens, channel_tokens=channel_tokens)
        prompt_vl_messages = _vl_messages(rendered)
        prompt_text = processor.apply_chat_template(prompt_vl_messages, tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_text], padding=True, return_tensors="pt")
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])

        teacher_raw_ids = tokenizer(teacher_answer.strip(), return_tensors="pt", add_special_tokens=False)["input_ids"]
        target_raw_len = int(teacher_raw_ids.shape[-1])
        teacher_trimmed_ids = teacher_raw_ids[:, : int(args.max_target_tokens)]
        target_trimmed_len = int(teacher_trimmed_ids.shape[-1])

        prompt_tokens.append(prompt_len)
        target_tokens_raw.append(target_raw_len)
        target_tokens_trimmed.append(target_trimmed_len)
        full_tokens.append(prompt_len + target_trimmed_len)

        qtype = _question_type(row)
        bucket = by_type.setdefault(qtype, {"prompt_tokens": [], "target_tokens_trimmed": [], "full_tokens": []})
        bucket["prompt_tokens"].append(prompt_len)
        bucket["target_tokens_trimmed"].append(target_trimmed_len)
        bucket["full_tokens"].append(prompt_len + target_trimmed_len)

        for cap in caps:
            clipped_prompt = min(prompt_len, cap)
            capped_prompt[cap].append(clipped_prompt)
            capped_full[cap].append(clipped_prompt + target_trimmed_len)

        longest.append(
            {
                "index": idx,
                "sample_id": row.get("sample_id"),
                "question_type": qtype,
                "prompt_tokens": prompt_len,
                "target_tokens_raw": target_raw_len,
                "target_tokens_trimmed": target_trimmed_len,
                "full_tokens": prompt_len + target_trimmed_len,
                "question": str(row.get("question") or "")[:240],
            }
        )

    longest.sort(key=lambda item: item["prompt_tokens"], reverse=True)
    by_type_summary = {
        key: {
            metric: _summarize(values)
            for metric, values in metrics.items()
        }
        for key, metrics in by_type.items()
    }

    hypothetical_caps: Dict[str, Any] = {}
    for cap in caps:
        over = sum(1 for value in prompt_tokens if value > cap)
        hypothetical_caps[str(cap)] = {
            "prompt_tokens": _summarize(capped_prompt[cap]),
            "full_tokens": _summarize(capped_full[cap]),
            "clipped_rate": float(over / len(prompt_tokens)) if prompt_tokens else 0.0,
        }

    summary = {
        "data": str(Path(args.data)),
        "teacher": str(Path(args.teacher)),
        "llm_config": str(Path(args.llm_config)),
        "num_samples": len(rows),
        "assumed_hint_tokens": {
            "global": global_tokens,
            "channel": channel_tokens,
        },
        "training_shape_under_test": {
            "include_anomaly_score_text": True,
            "include_anomaly_score_image_note": False,
            "use_axis_hints": True,
            "include_global_hints": True,
            "include_channel_hints": True,
            "max_window_rows": int(args.max_window_rows),
            "max_target_tokens": int(args.max_target_tokens),
        },
        "prompt_tokens": _summarize(prompt_tokens),
        "target_tokens_raw": _summarize(target_tokens_raw),
        "target_tokens_trimmed": _summarize(target_tokens_trimmed),
        "full_tokens": _summarize(full_tokens),
        "by_question_type": by_type_summary,
        "hypothetical_prompt_caps": hypothetical_caps,
        "top_longest_prompt_samples": longest[:20],
    }
    save_json(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
