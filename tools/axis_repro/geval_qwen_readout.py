"""Direct-logprob Qwen readout for rare top-5 score-token truncation.

The primary Qwen author response is preserved in the pending journal. This
runner asks the same model to encode that already-written integer score as one
JSON label (A=1 through E=5). JSON context prevents whitespace and Markdown
token variants from consuming the entire top-5. The returned label must agree
with the primary integer, and any omitted label probability is accepted only
under the same explicit missing-mass bound as the primary runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .common import read_jsonl
from .geval_qwen import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    result_row,
    task_key,
)
from .geval_resilient import run_retrying
from .io_utils import append_jsonl
from .runtime_fixes import score_from_text


LABEL_TO_SCORE = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
}
READOUT_INSTRUCTION = (
    "Encode the final evaluation score using A=1, B=2, C=3, D=4, E=5. "
    'Return exactly one JSON object with this schema: {"score":"D"}. '
    "Replace D with the correct one-letter code. Output valid JSON only."
)
READOUT_SCORE_RE = re.compile(r'"score"\s*:\s*"([A-E])"')


class ReadoutScoreLogprobsError(RuntimeError):
    """Raised when deterministic JSON readout cannot satisfy the bound."""


def request_body(
    model: str,
    author_prompt: str,
    primary_content: str,
    max_tokens: int,
    top_logprobs: int,
    seed: int,
) -> dict:
    if not 1 <= top_logprobs <= 5:
        raise ValueError("Qwen readout top_logprobs must be in [1,5]")
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": author_prompt},
            {"role": "assistant", "content": primary_content},
            {"role": "user", "content": READOUT_INSTRUCTION},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "response_format": {"type": "json_object"},
    }


def api_call(
    key: str,
    endpoint: str,
    body: dict,
) -> dict:
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


def bounded_label_distribution(
    response: dict,
    requested_top_logprobs: int,
    max_missing_mass_upper_bound: float,
) -> tuple[float, dict[str, float], dict] | None:
    if not 0.0 <= max_missing_mass_upper_bound < 1.0:
        raise ValueError(
            "max_missing_mass_upper_bound must be in [0,1)"
        )
    choice = response.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "") or ""
    matches = list(READOUT_SCORE_RE.finditer(content))
    if not matches:
        return None
    chosen_label = matches[-1].group(1)
    offset = matches[-1].start(1)
    entries = choice.get("logprobs", {}).get("content", []) or []
    cursor = 0
    target = None
    for entry in entries:
        token = str(entry.get("token", ""))
        end = cursor + len(token)
        if cursor <= offset < end:
            target = entry
            break
        cursor = end
    if target is None or str(target.get("token", "")) != chosen_label:
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
    label_logs: dict[str, float] = {}
    for token, logprob in ranked:
        if token in LABEL_TO_SCORE:
            label_logs[token] = max(
                label_logs.get(token, -math.inf), logprob
            )
    if chosen_label not in label_logs:
        return None
    missing = [
        label for label in LABEL_TO_SCORE if label not in label_logs
    ]
    peak = max(label_logs.values())
    weights = {
        label: math.exp(logprob - peak)
        for label, logprob in label_logs.items()
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
        str(score): weights.get(label, 0.0) / observed_weight
        for label, score in LABEL_TO_SCORE.items()
    }
    score = sum(
        int(value) * probability
        for value, probability in probabilities.items()
    )
    metadata = {
        "method": (
            "final_score_top_logprobs_readout_bounded"
            if missing else "final_score_top_logprobs_readout"
        ),
        "chosen_label": chosen_label,
        "observed_score_tokens": sorted(label_logs),
        "missing_score_tokens": missing,
        "top_logprobs_cutoff": cutoff,
        "missing_score_mass_upper_bound": missing_mass_upper,
        "score_error_upper_bound": 4.0 * missing_mass_upper,
        "distribution_is_truncated": bool(missing),
    }
    return score, probabilities, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pending", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--endpoint",
        default=os.getenv("QWEN_BASE_URL", DEFAULT_ENDPOINT),
    )
    parser.add_argument("--api-key-env", default="QWEN_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument(
        "--max-missing-score-mass-upper-bound",
        type=float,
        default=1e-6,
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-retry-rounds", type=int, default=12)
    args = parser.parse_args()

    request_body(
        args.model, "prompt", "**Score:** 3", args.max_tokens,
        args.top_logprobs, args.seed,
    )
    if not 0.0 <= args.max_missing_score_mass_upper_bound < 1.0:
        raise ValueError(
            "--max-missing-score-mass-upper-bound must be in [0,1)"
        )
    key = os.getenv(args.api_key_env)
    if not key:
        raise RuntimeError(
            f"{args.api_key_env} is required in the process environment"
        )

    pending_rows = read_jsonl(args.pending)
    latest = {
        task_key(item["row"], item["dimension"]): item
        for item in pending_rows
    }
    output = Path(args.output)
    completed = {
        task_key(row, row["dimension"])
        for row in read_jsonl(output)
    } if output.exists() else set()
    items = [
        item for key_, item in latest.items() if key_ not in completed
    ]

    def work(item):
        primary = item["primary"]
        primary_content = (
            primary.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            or ""
        )
        primary_score = score_from_text(primary_content)
        if primary_score is None:
            raise ReadoutScoreLogprobsError(
                "pending primary response has no integer score"
            )
        body = request_body(
            args.model, item["prompt"], primary_content,
            args.max_tokens, args.top_logprobs, args.seed,
        )
        response = api_call(key, args.endpoint, body)
        distribution = bounded_label_distribution(
            response, args.top_logprobs,
            args.max_missing_score_mass_upper_bound,
        )
        if distribution is None:
            raise ReadoutScoreLogprobsError(
                "JSON readout does not satisfy the missing-mass bound"
            )
        score, probabilities, metadata = distribution
        encoded_score = LABEL_TO_SCORE[metadata["chosen_label"]]
        if encoded_score != primary_score:
            raise ReadoutScoreLogprobsError(
                "JSON readout label disagrees with primary integer score"
            )
        result = result_row(
            item["row"], item["dimension"], tuple(item["spec"]),
            item["prompt"], primary, args.model, args.endpoint,
            args.max_tokens, args.top_logprobs, args.seed, score,
            metadata["method"], probabilities, [], metadata,
        )
        serialized_messages = json.dumps(
            body["messages"],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        choice = response.get("choices", [{}])[0]
        result.update({
            "score_readout_used": True,
            "score_readout_content": (
                choice.get("message", {}).get("content", "") or ""
            ),
            "score_readout_prompt_sha256": hashlib.sha256(
                serialized_messages.encode("utf-8")
            ).hexdigest(),
            "score_readout_model": response.get("model"),
            "score_readout_usage": response.get("usage"),
            "score_readout_endpoint_host": urlparse(
                args.endpoint
            ).netloc,
            "score_readout_target_label": metadata["chosen_label"],
        })
        return result

    done = 0
    for result in run_retrying(
        items,
        work,
        args.workers,
        "qwen_json_readout",
        max_rounds=args.max_retry_rounds,
    ):
        append_jsonl(output, result)
        done += 1
        print(json.dumps({
            "phase": "qwen_json_readout",
            "completed": done,
            "total": len(items),
        }), flush=True)


if __name__ == "__main__":
    main()
