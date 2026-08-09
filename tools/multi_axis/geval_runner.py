from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


SCORE_RE = re.compile(r"(?i)(?:\*\*)?Score(?:\*\*)?\s*:\s*(?:\*\*)?\s*\[?([1-5])")
READOUT_RE = re.compile(r'"score"\s*:\s*"([A-E])"')
LABEL_TO_SCORE = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
READOUT_INSTRUCTION = ('Encode the final evaluation score using A=1, B=2, C=3, D=4, E=5. Return exactly one JSON object with this schema: {"score":"D"}. Replace D with the correct one-letter code. Output valid JSON only.')
APPEND_LOCK = threading.Lock()


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, record: dict):
    with APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def endpoint_url(value: str) -> str:
    value = value.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def content_text(response: dict) -> str:
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def parse_integer_score(text: str):
    matches = list(SCORE_RE.finditer(text or ""))
    return int(matches[-1].group(1)) if matches else None


def build_prompt(row: dict, dimension: str, spec: dict) -> str:
    guidelines = "\n".join(f"Score {item}" for item in spec["guidelines"])
    return f"""You are an expert evaluator for natural language generation systems. Your task is to evaluate the quality of generated responses using the G-Eval methodology with chain-of-thought reasoning.

Control the Maximum Length to 500 words.

**Evaluation Criterion: {dimension.replace('_', ' ')}**
{spec['description']}

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


class JudgeClient:
    def __init__(self, profile_name: str, profile: dict):
        self.profile_name = profile_name
        self.profile = profile
        endpoint_value = os.environ.get(profile["endpoint_env"])
        self.key = os.environ.get(profile["api_key_env"])
        if not endpoint_value:
            raise RuntimeError(f"Missing endpoint environment variable {profile['endpoint_env']}")
        if not self.key:
            raise RuntimeError(f"Missing API-key environment variable {profile['api_key_env']}")
        self.endpoint = endpoint_url(endpoint_value)
        self.endpoint_fingerprint = hashlib.sha256(urlparse(self.endpoint).netloc.encode()).hexdigest()
        self.ssl_context = None
        self.tls_ca_bundle_sha256 = None
        ca_bundle_env = profile.get("ca_bundle_env")
        if ca_bundle_env:
            ca_bundle_value = os.environ.get(ca_bundle_env)
            if not ca_bundle_value:
                raise RuntimeError(f"Missing CA-bundle environment variable {ca_bundle_env}")
            ca_bundle = Path(ca_bundle_value).resolve()
            if not ca_bundle.is_file():
                raise RuntimeError(f"Configured CA bundle is not a file: {ca_bundle}")
            self.ssl_context = ssl.create_default_context(cafile=str(ca_bundle))
            self.tls_ca_bundle_sha256 = hashlib.sha256(ca_bundle.read_bytes()).hexdigest()

    def body(self, messages, *, temperature=None, logprobs=False, readout=False):
        profile = self.profile
        body = {"model": profile["model"], "messages": messages, "temperature": profile["temperature"] if temperature is None else temperature, "stream": False}
        body[profile.get("max_tokens_field", "max_tokens")] = 20 if readout else int(profile["max_tokens"])
        if logprobs:
            body.update({"logprobs": True, "top_logprobs": int(profile["top_logprobs"])})
        if "reasoning_effort" in profile:
            body["reasoning_effort"] = profile["reasoning_effort"]
        if "thinking" in profile:
            body["thinking"] = profile["thinking"]
        if "enable_thinking" in profile:
            body["enable_thinking"] = bool(profile["enable_thinking"])
        if readout:
            body["response_format"] = {"type": "json_object"}
        return body

    def request(self, body: dict):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last = None
        for attempt in range(6):
            request = urllib.request.Request(self.endpoint, data=encoded, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(
                    request, timeout=300, context=self.ssl_context
                ) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"Permanent judge HTTP status {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionResetError) as exc:
                last = exc
            if attempt < 5:
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"Judge request exhausted retries ({type(last).__name__})") from last


def token_entry_at(response: dict, character_offset: int):
    entries = response.get("choices", [{}])[0].get("logprobs", {}).get("content", []) or []
    cursor = 0
    for entry in entries:
        token = str(entry.get("token", ""))
        end = cursor + len(token)
        if cursor <= character_offset < end:
            return entry
        cursor = end
    return None


def bounded_distribution(entry: dict, symbols: dict, requested_topk: int, bound: float):
    alternatives = entry.get("top_logprobs", []) or []
    if len(alternatives) != requested_topk:
        return None
    ranked = []
    for item in alternatives:
        try:
            logprob = float(item["logprob"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(logprob):
            return None
        ranked.append((str(item.get("token", "")).strip(), logprob))
    logs = {}
    for token, logprob in ranked:
        if token in symbols:
            logs[token] = max(logs.get(token, -math.inf), logprob)
    if not logs:
        return None
    peak = max(logs.values())
    weights = {token: math.exp(value - peak) for token, value in logs.items()}
    observed_weight = sum(weights.values())
    missing = [token for token in symbols if token not in logs]
    cutoff = min(value for _, value in ranked)
    missing_weight_upper = len(missing) * math.exp(cutoff - peak)
    missing_mass_upper = missing_weight_upper / (observed_weight + missing_weight_upper)
    if missing_mass_upper > bound:
        return None
    probabilities = {str(symbols[token]): weights.get(token, 0.0) / observed_weight for token in symbols}
    score = sum(float(value) * probability for value, probability in probabilities.items())
    return score, probabilities, {"observed_symbols": sorted(logs), "missing_symbols": missing, "missing_mass_upper_bound": missing_mass_upper, "score_error_upper_bound": 4.0 * missing_mass_upper, "distribution_is_truncated": bool(missing)}


def primary_score_distribution(response: dict, topk: int, bound: float):
    text = content_text(response)
    matches = list(SCORE_RE.finditer(text))
    if not matches:
        return None
    entry = token_entry_at(response, matches[-1].start(1))
    return bounded_distribution(entry, {str(i): i for i in range(1, 6)}, topk, bound) if entry else None


def qwen_readout(client: JudgeClient, prompt: str, primary: dict, primary_integer: int, bound: float):
    primary_content = content_text(primary)
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": primary_content}, {"role": "user", "content": READOUT_INSTRUCTION}]
    response = client.request(client.body(messages, temperature=0.0, logprobs=True, readout=True))
    text = content_text(response)
    matches = list(READOUT_RE.finditer(text))
    if not matches:
        return None
    label = matches[-1].group(1)
    if LABEL_TO_SCORE[label] != primary_integer:
        return None
    entry = token_entry_at(response, matches[-1].start(1))
    distribution = bounded_distribution(entry, LABEL_TO_SCORE, int(client.profile["top_logprobs"]), bound) if entry else None
    if distribution is None:
        return None
    return distribution, response


def deepseek_fallback(client: JudgeClient, prompt: str, count: int = 20):
    def sample(_):
        response = client.request(client.body([{"role": "user", "content": prompt}], temperature=1.0, logprobs=False))
        return parse_integer_score(content_text(response))
    scores = []
    attempts = 0
    while len(scores) < count and attempts < 4:
        attempts += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            scores.extend(score for score in executor.map(sample, range(count - len(scores))) if score is not None)
    if len(scores) < count:
        raise RuntimeError(f"DeepSeek fallback returned only {len(scores)}/{count} valid scores")
    return sum(scores[:count]) / count, scores[:count]


def judge_task(client: JudgeClient, task: dict, missing_mass_bound: float):
    row, dimension, weight, dimension_spec = task["row"], task["dimension"], task["weight"], task["spec"]
    prompt = build_prompt(row, dimension, dimension_spec)
    scoring = client.profile["scoring"]
    request_logprobs = scoring in {"deepseek_logprobs", "qwen_logprobs_readout"}
    started = time.monotonic()
    primary = client.request(client.body([{"role": "user", "content": prompt}], logprobs=request_logprobs))
    raw = content_text(primary)
    integer = parse_integer_score(raw)
    if integer is None:
        raise RuntimeError("Judge response omitted the final 1-5 score")
    distribution = None
    metadata = {}
    fallback_scores = []
    if scoring == "integer":
        score, method = float(integer), "integer_score"
    else:
        direct = primary_score_distribution(primary, int(client.profile["top_logprobs"]), missing_mass_bound)
        if direct is not None:
            score, distribution, metadata = direct
            method = "final_score_top_logprobs_bounded"
        elif scoring == "qwen_logprobs_readout":
            readout = qwen_readout(client, prompt, primary, integer, missing_mass_bound)
            if readout is None:
                raise RuntimeError("Qwen score-token and A-E readout both failed the probability-mass bound")
            (score, distribution, metadata), readout_response = readout
            method = "a_e_json_readout_top_logprobs_bounded"
            metadata["readout_usage"] = readout_response.get("usage")
        else:
            score, fallback_scores = deepseek_fallback(client, prompt, count=20)
            method = "exact_sample_mean_20"
    return {"record_id": row["record_id"], "sample_id": row["sample_id"], "dataset": row["dataset"], "mode": row["mode"], "question_type": row["question_type"], "dimension": dimension, "weight": weight, "score": score, "integer_score": integer, "method": method, "distribution": distribution, "distribution_metadata": metadata, "fallback_scores": fallback_scores, "judge_content": raw, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "provider_profile": client.profile_name, "requested_model": client.profile["model"], "returned_model": primary.get("model"), "system_fingerprint": primary.get("system_fingerprint"), "usage": primary.get("usage"), "latency_seconds": time.monotonic() - started, "temperature": client.profile["temperature"], "logprobs_requested": request_logprobs, "top_logprobs": client.profile.get("top_logprobs"), "endpoint_host_sha256": client.endpoint_fingerprint, "tls_ca_bundle_sha256": client.tls_ca_bundle_sha256}


def main():
    parser = argparse.ArgumentParser(description="Crash-resilient five-model Multi-AXIS G-Eval")
    parser.add_argument("--input", required=True)
    parser.add_argument("--rubrics", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-missing-score-mass-upper-bound", type=float, default=1e-6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    if args.judge not in profiles:
        raise KeyError(f"Unknown judge profile {args.judge}")
    profile = profiles[args.judge]
    client = JudgeClient(args.judge, profile)
    rubrics = json.loads(Path(args.rubrics).read_text(encoding="utf-8"))["rubrics"]
    rows = read_jsonl(Path(args.input))
    if args.limit is not None:
        rows = rows[:args.limit]
    tasks = []
    for row in rows:
        rubric = rubrics[row["question_type"]]
        for dimension, spec in rubric["dimensions"].items():
            tasks.append({"row": row, "dimension": dimension, "weight": rubric["weights"][dimension], "spec": spec})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {(row["record_id"], row["dimension"]) for row in read_jsonl(output)}
    pending = [task for task in tasks if (task["row"]["record_id"], task["dimension"]) not in completed]
    if args.probe_only:
        if not pending:
            raise RuntimeError("No task available for endpoint probe")
        result = judge_task(client, pending[0], args.max_missing_score_mass_upper_bound)
        print(json.dumps({"probe": "ok", "judge": args.judge, "score": result["score"], "method": result["method"]}))
        return
    workers = args.workers or int(profile["workers"])
    round_index = 0
    while pending:
        round_index += 1
        failed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(judge_task, client, task, args.max_missing_score_mass_upper_bound): task for task in pending}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed.append(task)
                    print(json.dumps({"state": "retry", "round": round_index, "error_type": type(exc).__name__}), flush=True)
                    continue
                append_jsonl(output, result)
                completed.add((result["record_id"], result["dimension"]))
                print(json.dumps({"state": "completed", "record_id": result["record_id"], "dimension": result["dimension"], "score": result["score"], "method": result["method"]}), flush=True)
        pending = failed
        if pending:
            if round_index >= 12:
                raise RuntimeError(f"Judge scoring exhausted retry rounds with {len(pending)} tasks pending")
            time.sleep(min(60, 2 ** min(round_index, 5)))
    expected = len(tasks)
    actual = len({(row["record_id"], row["dimension"]) for row in read_jsonl(output)})
    if actual != expected:
        raise RuntimeError(f"Incomplete judge output: {actual}/{expected}")
    complete = output.with_suffix(output.suffix + ".complete.json")
    complete.write_text(json.dumps({"judge": args.judge, "tasks": actual, "examples": len(rows), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
