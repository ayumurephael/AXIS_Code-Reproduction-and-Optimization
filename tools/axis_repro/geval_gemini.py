"""Crash-resilient Gemini 2.5 Pro G-Eval for compatible gateways.

PackyAPI currently accepts OpenAI ``logprobs`` request fields but does not
return ``choices[].logprobs`` for gemini-2.5-pro. Its native endpoint reports
that logprobs are not enabled for this model. This runner exposes both useful
semantics instead of silently pretending a probability distribution exists:

``author`` reproduces the supplied AdvancedGEvaluator prompt and falls back to
one temperature-zero integer. ``strict`` uses a final-score-token distribution
when present and otherwise estimates the expectation with repeated samples.

The API key is read only from an environment variable and is never persisted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from . import common
from .io_utils import append_jsonl
common.append_jsonl = append_jsonl

from .common import read_jsonl
from .geval_correct import final_score_distribution
from .geval_deepseek import RUBRICS, normalize_type, prompt_for as baseline_prompt_for
from .runtime_fixes import score_from_text


DEFAULT_ENDPOINT = "https://www.packyapi.com/v1/chat/completions"

# Exact descriptions and guidelines from the supplied GEvaluator. Keeping
# these separate from the reproduction rubric makes prompt drift measurable.
AUTHOR_RUBRICS = {
    "multiple_choice": {
        "correctness": (0.70, "How accurate is the generated response compared to the expected answer?", [
            "5: Perfect match with the expected answer, completely correct",
            "4: Mostly correct with minor deviations that don't affect the core meaning",
            "3: Partially correct, captures some key aspects but misses important details",
            "2: Somewhat relevant but contains significant errors or omissions",
            "1: Incorrect or completely irrelevant to the expected answer",
        ]),
        "reasoning_quality": (0.30, "How well does the response demonstrate logical reasoning and explanation?", [
            "5: Clear, logical, and comprehensive reasoning that fully explains the choice",
            "4: Good reasoning with minor gaps in explanation",
            "3: Adequate reasoning but lacks depth or contains some logical inconsistencies",
            "2: Weak reasoning with significant gaps or flawed logic",
            "1: No clear reasoning or completely flawed logic",
        ]),
    },
    "open_ended": {
        "relevance": (0.30, "How relevant and on-topic is the generated response to the question?", [
            "5: Completely relevant, directly addresses all aspects of the question",
            "4: Highly relevant, addresses most aspects with minor omissions",
            "3: Moderately relevant, addresses core aspects but misses some details",
            "2: Somewhat relevant but contains off-topic information or misses key points",
            "1: Irrelevant or completely off-topic",
        ]),
        "completeness": (0.35, "How complete and comprehensive is the generated response?", [
            "5: Fully comprehensive, covers all necessary aspects and details",
            "4: Mostly complete with minor gaps",
            "3: Adequately complete but missing some important details",
            "2: Incomplete, missing significant information",
            "1: Very incomplete, lacks most necessary information",
        ]),
        "accuracy": (0.35, "How factually accurate is the information in the generated response?", [
            "5: Completely accurate with no factual errors",
            "4: Mostly accurate with very minor inaccuracies",
            "3: Generally accurate but contains some notable errors",
            "2: Contains several factual errors that affect reliability",
            "1: Contains major factual errors or is mostly inaccurate",
        ]),
    },
    "true_false": {
        "correctness": (0.60, "How correct is the true/false judgment and supporting explanation?", [
            "5: Perfect judgment with excellent supporting explanation",
            "4: Correct judgment with good supporting explanation",
            "3: Correct judgment with adequate explanation",
            "2: Incorrect judgment but reasonable attempt at explanation",
            "1: Incorrect judgment with poor or no explanation",
        ]),
        "justification_quality": (0.40, "How well does the response justify the true/false decision?", [
            "5: Excellent justification with clear evidence and reasoning",
            "4: Good justification with solid supporting evidence",
            "3: Adequate justification with some supporting evidence",
            "2: Weak justification with little supporting evidence",
            "1: No meaningful justification provided",
        ]),
    },
}


def author_prompt_for(row: dict, dimension: str, description: str, guides: list[str]) -> str:
    """Return the exact prompt layout from AdvancedGEvaluator."""
    guidelines = "\n".join(f"Score {guide}" for guide in guides)
    return f"""You are an expert evaluator for natural language generation systems. Your task is to evaluate the quality of generated responses using the G-Eval methodology with chain-of-thought reasoning.

Control the Maximum Length to 500 words.

**Evaluation Criterion: {dimension}**
{description}

**Scoring Guidelines:**
{guidelines}

**Question:** {row['question']}

**Expected Answer:** {row['answer']}

**Generated Response:** {row['response']}

**Instructions:**
1. Analyze the generated response step by step using chain-of-thought reasoning
2. Compare it against the expected answer for the specified criterion
3. Consider both content quality and alignment with the expected answer
4. Provide detailed reasoning for your evaluation
5. Conclude with a single score from 1-5
6. Ignore error in index mismatch, just focus on the content

Please follow this exact format for your response:

**Step-by-step Analysis:**
[Provide detailed chain-of-thought analysis]

**Comparison with Expected Answer:**
[Compare generated response with expected answer]

**Final Assessment:**
[Summarize your evaluation]

**Score:** [Single integer: 1, 2, 3, 4, or 5]"""


def api_call(key: str, endpoint: str, model: str, prompt: str,
             temperature: float, max_tokens: int,
             request_logprobs: bool, top_logprobs: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if request_logprobs:
        body.update({"logprobs": True, "top_logprobs": top_logprobs})
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")[:1000]
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(
                    f"Gemini HTTP {exc.code}: {response_text}"
                ) from exc
            if attempt == 5:
                raise RuntimeError(
                    f"Gemini HTTP {exc.code}: {response_text}"
                ) from exc
        except (urllib.error.URLError, TimeoutError):
            if attempt == 5:
                raise
        time.sleep(min(60, 2 ** attempt))
    raise AssertionError("unreachable")


def author_distribution(response: dict) -> tuple[float, dict[str, float] | None, str]:
    """Match AdvancedGEvaluator's cross-position weighting and fallback."""
    choice = response.get("choices", [{}])[0]
    content = choice.get("logprobs", {}).get("content", []) or []
    score_logs: dict[int, float] = {}
    for index, token_info in enumerate(content):
        token = str(token_info.get("token", "")).strip()
        if token in "12345" and len(token) == 1:
            score = int(token)
            previous = "".join(
                str(item.get("token", ""))
                for item in content[max(0, index - 5):index]
            ).lower()
            bonus = 0.5 if any(
                marker in previous for marker in ("score", "rating", "final")
            ) else 0.0
            value = float(token_info.get("logprob", -math.inf)) + bonus
            score_logs[score] = max(score_logs.get(score, -math.inf), value)
        for alternative in token_info.get("top_logprobs", []) or []:
            alt_token = str(alternative.get("token", "")).strip()
            if alt_token in "12345" and len(alt_token) == 1:
                score = int(alt_token)
                value = float(alternative.get("logprob", -math.inf))
                score_logs[score] = max(score_logs.get(score, -math.inf), value)
    if score_logs:
        peak = max(score_logs.values())
        normalizer = sum(math.exp(value - peak) for value in score_logs.values())
        probabilities = {
            str(score): math.exp(value - peak) / normalizer
            for score, value in score_logs.items()
        }
        expected = sum(
            int(score) * probability
            for score, probability in probabilities.items()
        )
        return expected, probabilities, "author_cross_position_logprobs"
    text = choice.get("message", {}).get("content", "") or ""
    raw_score = score_from_text(text)
    if raw_score is None:
        return 3.0, None, "author_default_3"
    return float(raw_score), None, "author_temperature0_integer"


def sample_scores(key: str, endpoint: str, model: str, prompt: str,
                  count: int, max_tokens: int, workers: int) -> list[int]:
    scores: list[int] = []
    attempts = 0

    def one(_: int) -> int | None:
        response = api_call(
            key, endpoint, model, prompt, 1.0, max_tokens, False, 0
        )
        text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        return score_from_text(text)

    while len(scores) < count and attempts < max(100, count * 5):
        batch = min(workers, count - len(scores))
        attempts += batch
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as executor:
            scores.extend(
                value for value in executor.map(one, range(batch))
                if value is not None
            )
    if len(scores) < count:
        raise RuntimeError(f"Only {len(scores)}/{count} valid Gemini fallback scores")
    return scores[:count]


def task_key(row: dict, dimension: str) -> tuple[str, str, str]:
    return row["record_id"], row["mode"], dimension


def judge_task(task: tuple) -> dict:
    (key, endpoint, model, max_tokens, top_logprobs, scoring_mode,
     prompt_template, fallback_samples, fallback_workers,
     row, dimension, spec) = task
    weight, description, guides = spec
    prompt = (
        author_prompt_for(row, dimension, description, guides)
        if prompt_template == "author"
        else baseline_prompt_for(row, dimension, description, guides)
    )
    primary = api_call(
        key, endpoint, model, prompt, 0.0, max_tokens, True, top_logprobs
    )
    choice = primary.get("choices", [{}])[0]
    raw_content = choice.get("message", {}).get("content", "") or ""
    has_logprobs = bool(choice.get("logprobs", {}).get("content", []) or [])
    fallback: list[int] = []
    distribution = None
    if scoring_mode == "author":
        score, distribution, method = author_distribution(primary)
    else:
        strict = final_score_distribution(primary)
        if strict is not None:
            score, distribution = strict
            method = "final_score_top_logprobs"
        else:
            fallback = sample_scores(
                key, endpoint, model, prompt, fallback_samples,
                max_tokens, fallback_workers,
            )
            score = sum(fallback) / len(fallback)
            method = f"exact_sample_mean_{fallback_samples}"
    return {
        "record_id": row["record_id"],
        "mode": row["mode"],
        "question_type": normalize_type(row["question_type"]),
        "dimension": dimension,
        "weight": weight,
        "score": score,
        "method": method,
        "raw_score": score_from_text(raw_content),
        "distribution": distribution,
        "fallback_scores": fallback,
        "judge_content": raw_content,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_template": prompt_template,
        "scoring_mode": scoring_mode,
        "model": primary.get("model", model),
        "system_fingerprint": primary.get("system_fingerprint"),
        "usage": primary.get("usage"),
        "judge_max_tokens": max_tokens,
        "logprobs_requested": True,
        "logprobs_returned": has_logprobs,
        "endpoint_host": urlparse(endpoint).netloc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--endpoint", default=os.getenv("GEMINI_BASE_URL", DEFAULT_ENDPOINT)
    )
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    parser.add_argument(
        "--scoring-mode", choices=["author", "strict"], default="author"
    )
    parser.add_argument(
        "--prompt-template", choices=["author", "baseline"], default="author"
    )
    parser.add_argument("--max-tokens", type=int, default=5000)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--fallback-samples", type=int, default=20)
    parser.add_argument("--primary-workers", type=int, default=3)
    parser.add_argument("--fallback-workers", type=int, default=3)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-task-rounds", type=int, default=6)
    args = parser.parse_args()

    key = os.getenv(args.api_key_env)
    if not key:
        raise RuntimeError(f"{args.api_key_env} is required")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        task_key(row, row["dimension"]) for row in read_jsonl(output)
    } if output.exists() else set()
    rows = read_jsonl(args.predictions)
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    rubric = AUTHOR_RUBRICS if args.prompt_template == "author" else RUBRICS
    tasks = []
    for row in rows:
        question_type = normalize_type(row["question_type"])
        if question_type not in rubric:
            raise ValueError(f"Unknown question_type {row['question_type']!r}")
        for dimension, spec in rubric[question_type].items():
            if task_key(row, dimension) not in completed:
                tasks.append((
                    key, args.endpoint, args.model, args.max_tokens,
                    args.top_logprobs, args.scoring_mode,
                    args.prompt_template, args.fallback_samples,
                    args.fallback_workers, row, dimension, spec,
                ))
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]

    pending = tasks
    retry_round = 0
    processed = 0
    while pending and retry_round < args.max_task_rounds:
        retry_round += 1
        failed = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.primary_workers
        ) as executor:
            futures = {executor.submit(judge_task, task): task for task in pending}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed.append(task)
                    print(json.dumps({
                        "state": "retry",
                        "round": retry_round,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:300],
                    }), flush=True)
                    continue
                append_jsonl(output, result)
                processed += 1
                print(json.dumps({
                    "state": "completed",
                    "processed": processed,
                    "total": len(tasks),
                    "record_id": result["record_id"],
                    "dimension": result["dimension"],
                    "score": result["score"],
                    "method": result["method"],
                }), flush=True)
        pending = failed
        if pending:
            time.sleep(min(60, 2 ** min(retry_round, 5)))
    if pending:
        raise RuntimeError(
            f"{len(pending)} Gemini tasks still failed after "
            f"{args.max_task_rounds} rounds"
        )


if __name__ == "__main__":
    main()
