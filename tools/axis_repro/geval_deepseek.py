from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from .common import append_jsonl, read_jsonl

# The high-thinking judge can consume more than 1,200 reasoning tokens before
# emitting the visible ``**Score:**`` line. Formal crash-resilient evaluation
# overrides this value explicitly; the module-level default keeps legacy CLIs
# backwards compatible.
MAX_TOKENS = 1200
API_ENDPOINT = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")

RUBRICS = {
 "multiple_choice": {
  "correctness": (0.70, "How accurate is the generated response compared to the expected answer?", ["5: Perfect match, completely correct", "4: Mostly correct; minor deviations not affecting core meaning", "3: Partially correct; captures some key aspects but misses important details", "2: Somewhat relevant but with significant errors or omissions", "1: Incorrect or completely irrelevant"]),
  "reasoning_quality": (0.30, "How well does the response demonstrate logical reasoning and explanation?", ["5: Clear, logical, comprehensive reasoning fully explains the choice", "4: Good reasoning with minor gaps", "3: Adequate reasoning; lacks depth or has inconsistencies", "2: Weak reasoning; significant gaps or flawed logic", "1: No clear reasoning or completely flawed logic"])},
 "open_ended": {
  "relevance": (0.30, "How relevant and on-topic is the generated response?", ["5: Completely relevant; directly addresses all aspects", "4: Highly relevant; minor omissions", "3: Moderately relevant; addresses core aspects but misses details", "2: Somewhat relevant; off-topic content or key omissions", "1: Irrelevant or off-topic"]),
  "completeness": (0.35, "How complete and comprehensive is the response?", ["5: Fully comprehensive; covers all necessary aspects", "4: Mostly complete; minor gaps", "3: Adequately complete; missing some important details", "2: Incomplete; significant information missing", "1: Very incomplete"]),
  "accuracy": (0.35, "How factually accurate is the response?", ["5: Completely accurate; no factual errors", "4: Mostly accurate; very minor inaccuracies", "3: Generally accurate; some notable errors", "2: Several factual errors affecting reliability", "1: Major factual errors or mostly inaccurate"])},
 "true_false": {
  "correctness": (0.60, "How correct is the T/F judgment and supporting explanation?", ["5: Perfect judgment; excellent supporting explanation", "4: Correct judgment; good explanation", "3: Correct judgment; adequate explanation", "2: Incorrect judgment; reasonable attempt at explanation", "1: Incorrect judgment; poor or no explanation"]),
  "justification_quality": (0.40, "How well does the response justify the decision?", ["5: Excellent justification with clear evidence and reasoning", "4: Good justification with solid evidence", "3: Adequate justification with some evidence", "2: Weak justification with little evidence", "1: No meaningful justification"])}}


def normalize_type(x: str) -> str:
    x = re.sub(r"[^a-z]+", "_", x.lower()).strip("_")
    aliases = {"multiplechoice": "multiple_choice", "openended": "open_ended", "truefalse": "true_false"}
    return aliases.get(x.replace("_", ""), x)


def prompt_for(row, dim, desc, guides):
    return f'''You are an expert evaluator for natural language generation systems. Your task is to evaluate the quality of generated responses using the G-Eval methodology with chain-of-thought reasoning.
Control the Maximum Length to 500 words.

**Evaluation Criterion: {dim.replace('_', ' ')}**
{desc}

**Scoring Guidelines:**
{chr(10).join(guides)}

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

**Score:** [Single integer: 1, 2, 3, 4, or 5]'''


def api_call(key, model, prompt, temperature, logprobs):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": MAX_TOKENS,
            "thinking": {"type": "enabled", "level": "high"}}
    if logprobs: body.update({"logprobs": True, "top_logprobs": 20})
    req = urllib.request.Request(API_ENDPOINT,
        data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp: return json.load(resp)
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == 5: raise
            time.sleep(min(30, 2**attempt))


def score_from_text(text):
    hits = re.findall(r"(?:\*\*)?Score(?:\*\*)?\s*:\s*(?:\*\*)?\s*\[?([1-5])", text or "", flags=re.I)
    return int(hits[-1]) if hits else None


def logprob_distribution(response):
    content = response.get("choices", [{}])[0].get("logprobs", {}).get("content", []) or []
    # Use the score-position token whose alternatives contain most score digits.
    candidates = {}
    for token in content:
        local = {}
        for alt in token.get("top_logprobs", []) or []:
            t = str(alt.get("token", "")).strip().strip("[]")
            if t in "12345" and len(t) == 1: local[int(t)] = max(local.get(int(t), -math.inf), float(alt["logprob"]))
        if len(local) > len(candidates): candidates = local
    if len(candidates) != 5: return None
    m = max(candidates.values()); z = sum(math.exp(v-m) for v in candidates.values())
    probs = {str(k): math.exp(v-m)/z for k, v in candidates.items()}
    return sum(int(k)*v for k, v in probs.items()), probs


def judge_one(key, model, row, dim, spec, fallback_samples):
    weight, desc, guides = spec; prompt = prompt_for(row, dim, desc, guides)
    primary = api_call(key, model, prompt, 0, True)
    dist = logprob_distribution(primary)
    method = "top_logprobs"
    if dist:
        score, probs = dist; samples = []
    else:
        method = f"sample_mean_{fallback_samples}"
        def sample(_):
            response = api_call(key, model, prompt, 1, False)
            return score_from_text(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, fallback_samples)) as ex:
            samples = [x for x in ex.map(sample, range(fallback_samples)) if x is not None]
        if not samples: raise RuntimeError("No valid 1-5 score in fallback samples")
        score, probs = sum(samples)/len(samples), None
    return {"record_id": row["record_id"], "mode": row["mode"], "question_type": normalize_type(row["question_type"]),
            "dimension": dim, "weight": weight, "score": score, "method": method,
            "distribution": probs, "fallback_scores": samples,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "model": model,
            "system_fingerprint": primary.get("system_fingerprint"), "usage": primary.get("usage"),
            "endpoint_host": urlparse(API_ENDPOINT).netloc, "logprobs_requested": True,
            "logprobs_returned": bool(primary.get("choices", [{}])[0].get("logprobs", {}).get("content"))}


def main():
    global API_ENDPOINT
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True); p.add_argument("--output", required=True)
    p.add_argument("--model", default="deepseek-v4-pro"); p.add_argument("--fallback-samples", type=int, default=20)
    p.add_argument("--endpoint", default=os.getenv("DEEPSEEK_BASE_URL", API_ENDPOINT))
    p.add_argument("--max-rows", type=int); a = p.parse_args()
    API_ENDPOINT = a.endpoint
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("Set DEEPSEEK_API_KEY in the process environment; never place it in arguments or files")
    rows = read_jsonl(a.predictions)[:a.max_rows]; out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    done = {(x["record_id"], x["mode"], x["dimension"]) for x in read_jsonl(out)} if out.exists() else set()
    for row in rows:
        qtype = normalize_type(row["question_type"])
        if qtype not in RUBRICS: raise ValueError(f"Unknown question_type {row['question_type']!r}")
        for dim, spec in RUBRICS[qtype].items():
            if (row["record_id"], row["mode"], dim) in done: continue
            result = judge_one(key, a.model, row, dim, spec, a.fallback_samples)
            append_jsonl(out, result); done.add((row["record_id"], row["mode"], dim))
            print(json.dumps({k: result[k] for k in ("record_id", "mode", "dimension", "score", "method")}), flush=True)


if __name__ == "__main__": main()


