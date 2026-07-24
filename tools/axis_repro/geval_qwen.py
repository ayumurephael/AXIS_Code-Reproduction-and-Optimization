"""Crash-resilient Qwen G-Eval using native score-token log probabilities.

Alibaba Cloud Model Studio exposes Qwen models through an OpenAI-compatible
chat-completions endpoint.  Qwen3 open-source models return token log
probabilities, but the service limits ``top_logprobs`` to five.  This runner
therefore locates the final ``**Score:**`` token and requires all five score
digits before computing the normalized expectation.

The primary path is deterministic and directly logprob-weighted.  An exact
20-sample fallback is available for protocol completeness, but is disabled
unless ``--fallback-samples`` is positive.  API keys are accepted only through
an environment variable and are never written to outputs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .io_utils import append_jsonl
from .common import read_jsonl
from .geval_correct import SCORE_RE
from .geval_deepseek import normalize_type
from .geval_gemini import AUTHOR_RUBRICS, author_prompt_for
from .geval_resilient import run_retrying
from .runtime_fixes import score_from_text


DEFAULT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/"
    "compatible-mode/v1/chat/completions"
)
DEFAULT_MODEL = "qwen3-30b-a3b-instruct-2507"
PROVIDER = "qwen"
MAX_QWEN_TOP_LOGPROBS = 5


class MissingScoreLogprobsError(RuntimeError):
    """Raised when score-token logprobs fail the configured error bound."""


def bounded_final_score_distribution(
    response: dict,
    requested_top_logprobs: int,
    max_missing_mass_upper_bound: float,
) -> tuple[float, dict[str, float], dict] | None:
    """Return a rigorously bounded 1--5 distribution from Qwen top-k.

    Model Studio caps ``top_logprobs`` at five, so an irrelevant token can
    displace one or more score digits. Because the returned alternatives are
    the top-k tokens, every omitted exact ASCII digit has logprob no greater
    than the returned cutoff. We accept truncated score support only when the
    resulting worst-case missing categorical mass is below the configured
    bound. Missing digits remain explicit zeroes in the reported conditional
    distribution and the approximation error bound is persisted for audit.
    """
    if not 0.0 <= max_missing_mass_upper_bound < 1.0:
        raise ValueError(
            "max_missing_mass_upper_bound must be in [0,1)"
        )
    choice = response.get("choices", [{}])[0]
    content = (
        choice.get("message", {}).get("content", "") or ""
    )
    matches = list(SCORE_RE.finditer(content))
    if not matches:
        return None
    score_offset = matches[-1].start(1)
    entries = (
        choice.get("logprobs", {}).get("content", []) or []
    )
    cursor = 0
    target = None
    for entry in entries:
        token = str(entry.get("token", ""))
        end = cursor + len(token)
        if cursor <= score_offset < end:
            target = entry
            break
        cursor = end
    if target is None:
        return None
    alternatives = target.get("top_logprobs", []) or []
    if len(alternatives) != requested_top_logprobs:
        return None
    ranked = []
    for item in alternatives:
        try:
            logprob = float(item["logprob"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(logprob):
            return None
        ranked.append((str(item.get("token", "")), logprob))
    digit_logs: dict[str, float] = {}
    for token, logprob in ranked:
        if len(token) == 1 and token in "12345":
            digit_logs[token] = max(
                digit_logs.get(token, -math.inf), logprob
            )
    if not digit_logs:
        return None
    missing = [digit for digit in "12345" if digit not in digit_logs]
    peak = max(digit_logs.values())
    weights = {
        digit: math.exp(logprob - peak)
        for digit, logprob in digit_logs.items()
    }
    observed_weight = sum(weights.values())
    cutoff = min(logprob for _token, logprob in ranked)
    missing_weight_upper = (
        len(missing) * math.exp(cutoff - peak)
    )
    missing_mass_upper = missing_weight_upper / (
        observed_weight + missing_weight_upper
    )
    if missing_mass_upper > max_missing_mass_upper_bound:
        return None
    probabilities = {
        digit: weights.get(digit, 0.0) / observed_weight
        for digit in "12345"
    }
    score = sum(
        int(digit) * probability
        for digit, probability in probabilities.items()
    )
    metadata = {
        "method": (
            "final_score_top_logprobs_bounded"
            if missing else "final_score_top_logprobs"
        ),
        "observed_score_tokens": sorted(digit_logs),
        "missing_score_tokens": missing,
        "top_logprobs_cutoff": cutoff,
        "missing_score_mass_upper_bound": missing_mass_upper,
        "score_error_upper_bound": 4.0 * missing_mass_upper,
        "distribution_is_truncated": bool(missing),
    }
    return score, probabilities, metadata


def request_body(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    request_logprobs: bool,
    top_logprobs: int,
    seed: int,
    enable_thinking: bool | None,
) -> dict:
    """Build a validated Model Studio OpenAI-compatible request."""
    if not 0 <= top_logprobs <= MAX_QWEN_TOP_LOGPROBS:
        raise ValueError(
            "Qwen top_logprobs must be between 0 and 5 inclusive"
        )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if request_logprobs:
        if top_logprobs == 0:
            raise ValueError(
                "top_logprobs must be positive when logprobs are requested"
            )
        body.update({
            "logprobs": True,
            "top_logprobs": top_logprobs,
        })
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


def api_call(
    key: str,
    endpoint: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    request_logprobs: bool,
    top_logprobs: int,
    seed: int,
    enable_thinking: bool | None,
) -> dict:
    """Make one request; retry classification is handled by ``run_retrying``."""
    body = request_body(
        model, prompt, temperature, max_tokens, request_logprobs,
        top_logprobs, seed, enable_thinking,
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def task_key(row: dict, dimension: str) -> tuple[str, str, str]:
    return row["record_id"], row["mode"], dimension


def compact_primary(primary: dict) -> dict:
    """Keep only safe fields required by a later fallback or audit."""
    choice = primary.get("choices", [{}])[0]
    return {
        "choices": [{
            "message": {
                "content": choice.get("message", {}).get("content", "") or "",
            },
        }],
        "model": primary.get("model"),
        "system_fingerprint": primary.get("system_fingerprint"),
        "usage": primary.get("usage"),
    }


def result_row(
    row: dict,
    dimension: str,
    spec: tuple,
    prompt: str,
    primary: dict,
    requested_model: str,
    endpoint: str,
    max_tokens: int,
    top_logprobs: int,
    seed: int,
    score: float,
    method: str,
    distribution: dict[str, float] | None,
    samples: list[int],
    logprob_metadata: dict | None = None,
    enable_thinking: bool | None = None,
) -> dict:
    choice = primary.get("choices", [{}])[0]
    raw = choice.get("message", {}).get("content", "") or ""
    metadata = logprob_metadata or {}
    return {
        "observed_score_tokens": metadata.get("observed_score_tokens"),
        "missing_score_tokens": metadata.get("missing_score_tokens"),
        "top_logprobs_cutoff": metadata.get("top_logprobs_cutoff"),
        "missing_score_mass_upper_bound": metadata.get("missing_score_mass_upper_bound"),
        "score_error_upper_bound": metadata.get("score_error_upper_bound"),
        "distribution_is_truncated": metadata.get("distribution_is_truncated"),
        "record_id": row["record_id"],
        "mode": row["mode"],
        "question_type": normalize_type(row["question_type"]),
        "dimension": dimension,
        "weight": spec[0],
        "score": score,
        "method": method,
        "raw_score": score_from_text(raw),
        "distribution": distribution,
        "fallback_scores": samples,
        "judge_content": raw,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "prompt_template": "author",
        "scoring_mode": "score_token_logprobs",
        "provider": PROVIDER,
        "model": primary.get("model", requested_model),
        "requested_model": requested_model,
        "system_fingerprint": primary.get("system_fingerprint"),
        "usage": primary.get("usage"),
        "judge_max_tokens": max_tokens,
        "seed": seed,
        "top_logprobs": top_logprobs,
        "logprobs_requested": True,
        "logprobs_returned": method.startswith("final_score_top_logprobs"),
        "enable_thinking": enable_thinking,
        "endpoint_host": urlparse(endpoint).netloc,
    }


def exact_samples(
    key: str,
    endpoint: str,
    model: str,
    prompt: str,
    count: int,
    max_tokens: int,
    workers: int,
    seed: int,
    enable_thinking: bool | None,
) -> list[int]:
    """Collect exactly ``count`` valid integer scores using distinct seeds."""
    if count <= 0:
        return []
    scores: list[int] = []
    attempts = 0

    def one(sample_seed: int) -> int | None:
        response = api_call(
            key, endpoint, model, prompt, 1.0, max_tokens,
            False, 0, sample_seed, enable_thinking,
        )
        content = (
            response.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return score_from_text(content)

    max_attempts = max(100, count * 5)
    while len(scores) < count and attempts < max_attempts:
        batch = min(workers, count - len(scores))
        sample_seeds = [
            seed + attempts + index + 1 for index in range(batch)
        ]
        attempts += batch
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=batch
        ) as executor:
            scores.extend(
                value for value in executor.map(one, sample_seeds)
                if value is not None
            )
    if len(scores) < count:
        raise RuntimeError(
            f"Only {len(scores)}/{count} valid Qwen fallback scores"
        )
    return scores[:count]


def parse_enable_thinking(value: str) -> bool | None:
    normalized = value.lower()
    if normalized == "auto":
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected auto, true, or false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pending")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("QWEN_BASE_URL", DEFAULT_ENDPOINT),
        help="Region-matched OpenAI-compatible chat/completions endpoint.",
    )
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--top-logprobs", type=int, default=MAX_QWEN_TOP_LOGPROBS
    )
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument(
        "--enable-thinking",
        type=parse_enable_thinking,
        default=None,
        metavar="{auto,true,false}",
        help=(
            "Send Qwen's enable_thinking extension. The default 'auto' "
            "omits it; instruct-2507 is non-thinking only."
        ),
    )
    parser.add_argument(
        "--max-missing-score-mass-upper-bound",
        type=float,
        default=1e-6,
        help=(
            "Fail closed unless omitted score digits carry at most this "
            "worst-case conditional probability mass."
        ),
    )
    parser.add_argument(
        "--fallback-samples",
        type=int,
        default=0,
        help=(
            "Exact sampling fallback count when a complete final-score "
            "distribution is unavailable. Use 20 for the AXIS fallback "
            "protocol; 0 fails closed."
        ),
    )
    parser.add_argument("--primary-workers", type=int, default=6)
    parser.add_argument("--fallback-workers", type=int, default=2)
    parser.add_argument("--max-retry-rounds", type=int, default=12)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()

    if args.fallback_samples not in {0, 20}:
        raise ValueError(
            "AXIS Qwen G-Eval permits only no fallback or exactly 20 samples"
        )
    if not 0.0 <= args.max_missing_score_mass_upper_bound < 1.0:
        raise ValueError(
            "--max-missing-score-mass-upper-bound must be in [0,1)"
        )
    # Validate before spending any API calls.
    request_body(
        args.model, "validation", 0.0, args.max_tokens, True,
        args.top_logprobs, args.seed, args.enable_thinking,
    )
    key = os.getenv(args.api_key_env)
    if not key:
        raise RuntimeError(
            f"{args.api_key_env} is required in the process environment"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending_path = Path(args.pending) if args.pending else output.with_name(
        output.stem + ".fallback_pending.jsonl"
    )
    completed = {
        task_key(row, row["dimension"]) for row in read_jsonl(output)
    } if output.exists() else set()
    pending_rows = read_jsonl(
        pending_path
    ) if pending_path.exists() else []
    pending_latest = {
        task_key(item["row"], item["dimension"]): item
        for item in pending_rows
    }
    pending = {
        key_: item for key_, item in pending_latest.items()
        if score_from_text(
            item.get("primary", {}).get("choices", [{}])[0]
            .get("message", {}).get("content", "") or ""
        ) is not None
    }

    rows = read_jsonl(args.predictions)
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    primary_tasks = []
    for row in rows:
        question_type = normalize_type(row["question_type"])
        if question_type not in AUTHOR_RUBRICS:
            raise ValueError(
                f"Unknown question_type {row['question_type']!r}"
            )
        for dimension, spec in AUTHOR_RUBRICS[question_type].items():
            key_ = task_key(row, dimension)
            if key_ not in completed and key_ not in pending:
                primary_tasks.append((row, dimension, spec))
    if args.max_tasks is not None:
        primary_tasks = primary_tasks[:args.max_tasks]

    def primary_work(item):
        row, dimension, spec = item
        prompt = author_prompt_for(
            row, dimension, spec[1], spec[2]
        )
        primary = api_call(
            key, args.endpoint, args.model, prompt, 0.0,
            args.max_tokens, True, args.top_logprobs, args.seed,
            args.enable_thinking,
        )
        distribution = bounded_final_score_distribution(
            primary, args.top_logprobs,
            args.max_missing_score_mass_upper_bound,
        )
        return item, prompt, primary, distribution

    processed = 0
    for item, prompt, primary, distribution in run_retrying(
        primary_tasks,
        primary_work,
        args.primary_workers,
        "qwen_primary",
        max_rounds=args.max_retry_rounds,
    ):
        row, dimension, spec = item
        processed += 1
        if distribution is not None:
            score, probabilities, metadata = distribution
            append_jsonl(output, result_row(
                row, dimension, spec, prompt, primary, args.model,
                args.endpoint, args.max_tokens, args.top_logprobs,
                args.seed, score, metadata["method"],
                probabilities, [], metadata,
                args.enable_thinking,
            ))
            state = "completed"
        else:
            journal = {
                "row": row,
                "dimension": dimension,
                "spec": spec,
                "prompt": prompt,
                "primary": compact_primary(primary),
                "max_missing_score_mass_upper_bound": (
                    args.max_missing_score_mass_upper_bound
                ),
                "enable_thinking": args.enable_thinking,
            }
            append_jsonl(pending_path, journal)
            pending[task_key(row, dimension)] = journal
            state = "fallback_pending"
        print(json.dumps({
            "phase": "qwen_primary",
            "processed": processed,
            "total": len(primary_tasks),
            "state": state,
        }), flush=True)

    completed_rows = read_jsonl(output) if output.exists() else []
    completed = {
        task_key(row, row["dimension"]) for row in completed_rows
    }
    fallback_items = [
        item for key_, item in pending.items() if key_ not in completed
    ]
    if fallback_items and args.fallback_samples == 0:
        raise MissingScoreLogprobsError(
            f"{len(fallback_items)} Qwen task(s) lack an admissible "
            "score-token distribution under the configured missing-mass "
            "bound; sampling fallback remains disabled"
        )

    def fallback_work(item):
        samples = exact_samples(
            key, args.endpoint, args.model, item["prompt"],
            args.fallback_samples, args.max_tokens, args.fallback_workers,
            args.seed, args.enable_thinking,
        )
        return result_row(
            item["row"], item["dimension"], tuple(item["spec"]),
            item["prompt"], item["primary"], args.model, args.endpoint,
            args.max_tokens, args.top_logprobs, args.seed,
            sum(samples) / len(samples),
            f"exact_sample_mean_{args.fallback_samples}",
            None, samples, None,
            args.enable_thinking,
        )

    finished = 0
    for result in run_retrying(
        fallback_items,
        fallback_work,
        args.fallback_workers,
        "qwen_fallback",
        max_rounds=args.max_retry_rounds,
    ):
        append_jsonl(output, result)
        finished += 1
        print(json.dumps({
            "phase": "qwen_fallback",
            "completed": finished,
            "total": len(fallback_items),
        }), flush=True)


if __name__ == "__main__":
    main()
