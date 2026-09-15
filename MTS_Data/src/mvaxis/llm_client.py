from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any]
    latency_seconds: float


class OpenAICompatibleChatClient:
    """Minimal chat-completions client used by the GPT question and answer stages."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.base_url = str(config["base_url"]).rstrip("/")
        self.model = str(config["model"])
        self.api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
        self.api_key = os.environ.get(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        self.timeout = float(config.get("timeout_seconds", 180))
        self.temperature = float(config.get("temperature", 0.5))
        self.max_tokens = int(config.get("max_tokens", 900))
        self.reasoning_effort = config.get("reasoning_effort")
        self.response_format = config.get("response_format")

    def complete(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.response_format:
            payload["response_format"] = self.response_format

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        raw = json.loads(body)
        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected chat-completions response: {raw}") from exc
        return LLMResponse(
            content=str(content or ""),
            raw=raw,
            latency_seconds=time.perf_counter() - started,
        )


def create_llm_client(config: Dict[str, Any]) -> OpenAICompatibleChatClient:
    provider = str(config.get("provider", "openai_compatible")).lower()
    if provider not in {"openai", "openai_compatible", "api"}:
        raise ValueError(
            f"Unsupported LLM provider {provider!r}; this generation repository keeps only the GPT API path."
        )
    return OpenAICompatibleChatClient(config)
