"""Crash-resilient two-stage DeepSeek G-Eval.

Primary responses that cannot yield all five score-token probabilities are
journaled before fallback sampling. A restart therefore never repeats a
successful primary call merely because the fallback phase was interrupted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from pathlib import Path

from . import common
from .io_utils import append_jsonl

common.append_jsonl = append_jsonl

from . import geval_deepseek as g  # noqa: E402
from .common import read_jsonl  # noqa: E402
from .geval_correct import exact_samples, final_score_distribution  # noqa: E402
from .runtime_fixes import score_from_text  # noqa: E402

g.score_from_text = score_from_text


def task_key(row: dict, dimension: str) -> tuple[str, str, str]:
    return row["record_id"], row["mode"], dimension


def compact_primary(primary: dict) -> dict:
    """Keep only fields needed to build the final auditable score row."""
    message = primary.get("choices", [{}])[0].get("message", {})
    return {
        "choices": [{"message": {"content": message.get("content", "") or ""}}],
        "model": primary.get("model"),
        "system_fingerprint": primary.get("system_fingerprint"),
        "usage": primary.get("usage"),
    }

def result_row(row, dimension, spec, prompt, primary, score, method, probs, samples):
    message = primary.get("choices", [{}])[0].get("message", {})
    raw = message.get("content", "") or ""
    return {
        "record_id": row["record_id"],
        "mode": row["mode"],
        "question_type": g.normalize_type(row["question_type"]),
        "dimension": dimension,
        "weight": spec[0],
        "score": score,
        "method": method,
        "raw_score": g.score_from_text(raw),
        "distribution": probs,
        "fallback_scores": samples,
        "judge_content": raw,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "model": primary.get("model"),
        "system_fingerprint": primary.get("system_fingerprint"),
        "usage": primary.get("usage"),
        "judge_max_tokens": g.MAX_TOKENS,
    }


def run_retrying(items, worker, workers, phase):
    """Yield successful results; retry failed items without losing progress."""
    queue = list(items)
    round_index = 0
    while queue:
        round_index += 1
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(worker, item): item for item in queue}
            for future in concurrent.futures.as_completed(futures):
                # Release completed Future results (large token-logprob payloads).
                item = futures.pop(future)
                try:
                    yield future.result()
                except Exception as exc:  # network/API errors are retriable
                    failed.append(item)
                    print(json.dumps({
                        "phase": phase,
                        "state": "retry",
                        "error_type": type(exc).__name__,
                        "round": round_index,
                    }), flush=True)
        queue = failed
        if queue:
            time.sleep(min(60, 2 ** min(round_index, 5)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pending")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--fallback-samples", type=int, default=20)
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="Judge completion budget. 4096 avoids truncating the final score "
             "under high-thinking mode.",
    )
    parser.add_argument("--primary-workers", type=int, default=12)
    parser.add_argument("--fallback-workers", type=int, default=8)
    args = parser.parse_args()
    g.MAX_TOKENS = args.max_tokens

    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending_path = Path(args.pending) if args.pending else output.with_name(
        output.stem + ".fallback_pending.jsonl"
    )
    completed = {
        (x["record_id"], x["mode"], x["dimension"])
        for x in read_jsonl(output)
    } if output.exists() else set()
    pending_rows = read_jsonl(pending_path) if pending_path.exists() else []
    pending_latest = {
        (x["row"]["record_id"], x["row"]["mode"], x["dimension"]): x
        for x in pending_rows
    }
    # A missing final score normally means the high-thinking response was cut
    # off by the old 1,200-token budget. Re-run one probability call with the
    # configured larger budget instead of multiplying the truncation into 20
    # fallback calls. Rows that did emit a score but lack all five top-logprobs
    # remain genuine fallback candidates.
    pending = {
        k: x for k, x in pending_latest.items()
        if g.score_from_text(
            x.get("primary", {}).get("choices", [{}])[0]
             .get("message", {}).get("content", "") or ""
        ) is not None
    }

    primary_tasks = []
    for row in read_jsonl(args.predictions):
        qtype = g.normalize_type(row["question_type"])
        for dimension, spec in g.RUBRICS[qtype].items():
            k = task_key(row, dimension)
            if k not in completed and k not in pending:
                primary_tasks.append((row, dimension, spec))

    def primary_work(item):
        row, dimension, spec = item
        prompt = g.prompt_for(row, dimension, spec[1], spec[2])
        primary = g.api_call(key, args.model, prompt, 0, True)
        return item, prompt, primary, final_score_distribution(primary)

    processed = 0
    for item, prompt, primary, distribution in run_retrying(
        primary_tasks, primary_work, args.primary_workers, "primary"
    ):
        row, dimension, spec = item
        processed += 1
        if distribution is not None:
            score, probs = distribution
            append_jsonl(output, result_row(
                row, dimension, spec, prompt, primary, score,
                "final_score_top_logprobs", probs, [],
            ))
            state = "completed"
        else:
            journal = {
                "row": row,
                "dimension": dimension,
                "spec": spec,
                "prompt": prompt,
                "primary": compact_primary(primary),
            }
            append_jsonl(pending_path, journal)
            pending[task_key(row, dimension)] = journal
            state = "fallback_pending"
        print(json.dumps({
            "phase": "primary", "processed": processed,
            "total": len(primary_tasks), "state": state,
        }), flush=True)

    completed = {
        (x["record_id"], x["mode"], x["dimension"])
        for x in read_jsonl(output)
    }
    fallback_items = [x for k, x in pending.items() if k not in completed]

    def fallback_work(item):
        samples = exact_samples(
            key, args.model, item["prompt"], args.fallback_samples
        )
        return result_row(
            item["row"], item["dimension"], item["spec"], item["prompt"],
            item["primary"], sum(samples) / len(samples),
            f"exact_sample_mean_{args.fallback_samples}", None, samples,
        )

    finished = 0
    for result in run_retrying(
        fallback_items, fallback_work, args.fallback_workers, "fallback"
    ):
        append_jsonl(output, result)
        finished += 1
        print(json.dumps({
            "phase": "fallback", "completed": finished,
            "total": len(fallback_items),
        }), flush=True)


if __name__ == "__main__":
    main()

