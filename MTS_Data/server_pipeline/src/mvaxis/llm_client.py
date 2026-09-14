from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    content: str
    raw: Dict[str, Any]
    latency_seconds: float


def _client_debug(event: str, **payload: Any) -> None:
    if str(os.environ.get("MVAXIS_CLIENT_DEBUG", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        rank = 0
        world_size = 1
    print(json.dumps({"stage": "client_debug", "event": event, "rank": rank, "world_size": world_size, **payload}, ensure_ascii=False), flush=True)


class OpenAICompatibleChatClient:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.base_url = str(config["base_url"]).rstrip("/")
        self.model = str(config["model"])
        api_key_env = str(config.get("api_key_env", "DEEPSEEK_API_KEY"))
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
        self.timeout = float(config.get("timeout_seconds", 120))
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = int(config.get("max_tokens", 2048))
        self.reasoning_effort = config.get("reasoning_effort")
        self.thinking = config.get("thinking")
        self.response_format = config.get("response_format")

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.thinking:
            payload["thinking"] = self.thinking
        if self.response_format:
            payload["response_format"] = self.response_format

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
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
        raw = json.loads(body)
        content = raw["choices"][0]["message"]["content"]
        return LLMResponse(content=content, raw=raw, latency_seconds=time.perf_counter() - started)


class LocalHFChatClient:
    """Local Hugging Face chat client with optional AXIS-style embedding hints.

    The normal `complete` method keeps the same JSON prompt interface used by the
    API clients. When `axis_hints.enabled=true`, callers may use
    `complete_with_axis_hints` to append AXIS special hint tokens and replace
    their input embeddings with projected time-series embeddings, following the
    original AXIS implementation pattern.
    """

    GLOBAL_HINT_TOKEN = "<|global_hint|>"
    CHANNEL_HINT_TOKEN = "<|channel_hint|>"
    LOCAL_HINT_TOKEN = "<|local_hint|>"
    FIXED_HINT_TOKEN = "<|fixed_hint|>"
    HINT_TOKEN_PLACEHOLDER = "[[AXIS_HINT_TOKENS]]"
    GLOBAL_HINT_TOKEN_PLACEHOLDER = "[[AXIS_GLOBAL_HINT_TOKENS]]"
    CHANNEL_HINT_TOKEN_PLACEHOLDER = "[[AXIS_CHANNEL_HINT_TOKENS]]"

    def __init__(self, config: Dict[str, Any]) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "LocalHFChatClient requires torch and transformers in the selected Python environment."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.config = config
        self.model_name = str(config["model"])
        self.max_new_tokens = int(config.get("max_tokens", config.get("max_new_tokens", 1024)))
        self.min_new_tokens = max(0, int(config.get("min_new_tokens", 0)))
        self.temperature = float(config.get("temperature", 0.0))
        self.do_sample = bool(config.get("do_sample", self.temperature > 0))
        self.use_cache = bool(config.get("use_cache", True))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.device = self._resolve_device(str(config.get("device", "auto")))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self._token(config),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = self._resolve_dtype(str(config.get("torch_dtype", "auto")))
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "token": self._token(config),
        }
        device_map = config.get("device_map")
        if device_map:
            model_kwargs["device_map"] = device_map
            if config.get("max_memory"):
                model_kwargs["max_memory"] = {
                    int(k) if isinstance(k, str) and k.isdigit() else k: v
                    for k, v in dict(config["max_memory"]).items()
                }
            if config.get("offload_folder"):
                model_kwargs["offload_folder"] = str(config["offload_folder"])
            model_kwargs["low_cpu_mem_usage"] = bool(config.get("low_cpu_mem_usage", True))
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if not device_map:
            self.model.to(self.device)
        self.model.eval()

        self.axis_hints_config = dict(config.get("axis_hints") or {})
        self.axis_hints_enabled = bool(self.axis_hints_config.get("enabled", False))
        self.perceiver = None
        if self.axis_hints_enabled:
            self._init_axis_hints()

    def _token(self, config: Dict[str, Any]) -> Optional[str]:
        token_env = config.get("token_env")
        if token_env:
            return os.environ.get(str(token_env))
        return config.get("token")

    def _resolve_device(self, requested: str):
        if requested == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return self.torch.device(requested)

    def _resolve_dtype(self, requested: str):
        requested = requested.lower()
        if requested == "auto":
            return self.torch.float16 if self.device.type == "cuda" else None
        if requested in {"float16", "fp16"}:
            return self.torch.float16
        if requested in {"bfloat16", "bf16"}:
            return self.torch.bfloat16
        if requested in {"float32", "fp32"}:
            return self.torch.float32
        if requested in {"none", "default"}:
            return None
        raise ValueError(f"Unsupported torch_dtype: {requested}")

    def _init_axis_hints(self) -> None:
        special_tokens = [
            self.GLOBAL_HINT_TOKEN,
            self.CHANNEL_HINT_TOKEN,
            self.LOCAL_HINT_TOKEN,
            self.FIXED_HINT_TOKEN,
        ]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
        hidden_size = int(self.model.config.hidden_size)
        d_proj = int(self.axis_hints_config.get("d_proj", 32))
        num_prototype = int(self.axis_hints_config.get("num_prototype", 256))
        num_fixed_tokens = int(self.axis_hints_config.get("num_fixed_tokens", 8))
        num_heads = int(self.axis_hints_config.get("num_heads", 4))
        self.perceiver = _AxisStylePerceiver(
            vocab_size=int(self.model.get_input_embeddings().weight.shape[0]),
            hidden_size=hidden_size,
            d_proj=d_proj,
            num_prototype=num_prototype,
            num_fixed_tokens=num_fixed_tokens,
            num_heads=num_heads,
        ).to(self.device)
        checkpoint_path = self.axis_hints_config.get("checkpoint_path")
        if checkpoint_path:
            state = self.torch.load(str(checkpoint_path), map_location=self.device)
            raw_state = state.get("perceiver", state)
            current_state = self.perceiver.state_dict()
            compatible_state = {
                key: value
                for key, value in raw_state.items()
                if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
            }
            self.perceiver.load_state_dict(compatible_state, strict=False)
        self.perceiver.eval()
        self.global_hint_token_id = self.tokenizer.convert_tokens_to_ids(self.GLOBAL_HINT_TOKEN)
        self.channel_hint_token_id = self.tokenizer.convert_tokens_to_ids(self.CHANNEL_HINT_TOKEN)
        self.local_hint_token_id = self.tokenizer.convert_tokens_to_ids(self.LOCAL_HINT_TOKEN)
        self.fixed_hint_token_id = self.tokenizer.convert_tokens_to_ids(self.FIXED_HINT_TOKEN)

    def _render_messages(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        rendered = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            rendered.append(f"{role}: {msg.get('content', '')}")
        rendered.append("ASSISTANT:")
        return "\n\n".join(rendered)

    def _generate_from_prompt(self, prompt: str, raw_extra: Optional[Dict[str, Any]] = None) -> LLMResponse:
        started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            outputs = self.model.generate(**encoded, **generate_kwargs)
        new_tokens = outputs[0, encoded["input_ids"].shape[-1] :]
        content = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        metadata = self._generation_metadata(
            prompt_token_count=int(encoded["input_ids"].shape[-1]),
            generated_tokens=new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        raw = {"provider": "local_hf", "model": self.model_name, **metadata}
        if raw_extra:
            raw.update(raw_extra)
        return LLMResponse(content=content, raw=raw, latency_seconds=time.perf_counter() - started)

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        return self._generate_from_prompt(self._render_messages(messages))

    def get_input_token_embeddings(
        self,
        texts: List[str],
        *,
        max_length: int = 128,
    ):
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        with self.torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings.detach(), attention_mask.bool()

    def _generation_metadata(self, *, prompt_token_count: int, generated_tokens: Any, eos_token_id: Any) -> Dict[str, Any]:
        generated_token_count = int(generated_tokens.shape[-1]) if hasattr(generated_tokens, "shape") else int(len(generated_tokens))
        eos_ids: List[int] = []
        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = [int(x) for x in eos_token_id]
        else:
            eos_ids = [int(eos_token_id)]
        last_token_id = int(generated_tokens[-1].item()) if generated_token_count > 0 else None
        hit_max_new_tokens = bool(generated_token_count >= int(self.max_new_tokens))
        if hit_max_new_tokens:
            finish_reason = "length"
        elif last_token_id is not None and last_token_id in eos_ids:
            finish_reason = "stop"
        elif generated_token_count == 0:
            finish_reason = "empty"
        else:
            finish_reason = "unknown"
        return {
            "prompt_token_count": int(prompt_token_count),
            "generated_token_count": generated_token_count,
            "total_sequence_token_count": int(prompt_token_count) + generated_token_count,
            "last_generated_token_id": last_token_id,
            "finish_reason": finish_reason,
            "hit_max_new_tokens": hit_max_new_tokens,
        }

    def _downsample_hint_tensor(self, tensor: Any, limit: int) -> Any:
        hint_tensor = self.torch.as_tensor(tensor, dtype=self.torch.float32, device=self.device)
        if hint_tensor.ndim != 2:
            raise ValueError("AXIS hint embeddings must have shape [num_tokens, d_proj]")
        if hint_tensor.shape[0] > int(limit):
            idx = self.torch.linspace(0, hint_tensor.shape[0] - 1, int(limit), device=self.device).long()
            hint_tensor = hint_tensor.index_select(0, idx)
        return hint_tensor

    def _prepare_axis_hint_tensors(self, axis_embeddings: Any) -> Dict[str, Any]:
        if isinstance(axis_embeddings, dict):
            tensors: Dict[str, Any] = {}
            global_embeddings = axis_embeddings.get("global")
            channel_embeddings = axis_embeddings.get("channel", axis_embeddings.get("local"))
            if global_embeddings is not None:
                max_global = int(
                    self.axis_hints_config.get(
                        "max_global_tokens",
                        self.axis_hints_config.get("num_global_tokens", self.axis_hints_config.get("num_fixed_tokens", 4)),
                    )
                )
                tensors["global"] = self._downsample_hint_tensor(global_embeddings, max_global)
            if channel_embeddings is not None:
                max_channel = int(self.axis_hints_config.get("max_channel_tokens", self.axis_hints_config.get("max_local_tokens", 64)))
                tensors["channel"] = self._downsample_hint_tensor(channel_embeddings, max_channel)
            if not tensors:
                raise ValueError("axis_embeddings dict must contain at least one of: global, channel")
            return tensors
        return {"legacy_local": self._downsample_hint_tensor(axis_embeddings, int(self.axis_hints_config.get("max_local_tokens", 64)))}

    def complete_with_axis_hints(self, messages: List[Dict[str, str]], axis_embeddings: Any) -> LLMResponse:
        if not self.axis_hints_enabled or self.perceiver is None:
            return self.complete(messages)
        hint_tensors = self._prepare_axis_hint_tensors(axis_embeddings)
        if "legacy_local" in hint_tensors:
            local_tensor = hint_tensors["legacy_local"]
            global_tokens = ""
            channel_tokens = ""
            axis_hint_block = (
                f"{self.LOCAL_HINT_TOKEN * int(local_tensor.shape[0])}"
                f"{self.FIXED_HINT_TOKEN * int(self.perceiver.num_fixed_tokens)}"
            )
        else:
            global_tensor = hint_tensors.get("global")
            channel_tensor = hint_tensors.get("channel")
            global_tokens = self.GLOBAL_HINT_TOKEN * int(0 if global_tensor is None else global_tensor.shape[0])
            channel_tokens = self.CHANNEL_HINT_TOKEN * int(0 if channel_tensor is None else channel_tensor.shape[0])
            axis_hint_block = f"{global_tokens}{channel_tokens}"
        axis_messages = [dict(m) for m in messages]
        content = axis_messages[-1].get("content", "")
        if self.GLOBAL_HINT_TOKEN_PLACEHOLDER in content or self.CHANNEL_HINT_TOKEN_PLACEHOLDER in content:
            content = content.replace(self.GLOBAL_HINT_TOKEN_PLACEHOLDER, global_tokens, 1)
            content = content.replace(self.CHANNEL_HINT_TOKEN_PLACEHOLDER, channel_tokens, 1)
            axis_messages[-1]["content"] = content
        elif self.HINT_TOKEN_PLACEHOLDER in content:
            axis_messages[-1]["content"] = content.replace(self.HINT_TOKEN_PLACEHOLDER, axis_hint_block, 1)
        else:
            axis_messages[-1]["content"] = content + "\n\n" + axis_hint_block + "\n"
        prompt = self._render_messages(axis_messages)
        started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        embedding_device = input_embeddings.device
        # Keep the small trainable Perceiver in fp32, matching training. The
        # resulting prompt embeddings are cast only at the injection boundary.
        if next(self.perceiver.parameters()).device != embedding_device or next(self.perceiver.parameters()).dtype != self.torch.float32:
            self.perceiver.to(device=embedding_device, dtype=self.torch.float32)
        word_embeddings = self.model.get_input_embeddings().weight.detach().to(device=embedding_device, dtype=self.torch.float32)
        source_embeddings = self.perceiver.get_source_embeddings(word_embeddings)
        projected_global = None
        projected_channel = None
        projected_local = None
        if "legacy_local" in hint_tensors:
            local_tensor = hint_tensors["legacy_local"].to(device=embedding_device, dtype=self.torch.float32)
            projected_local = self.perceiver.process_local_embeddings(local_tensor, source_embeddings)
        else:
            global_tensor = hint_tensors.get("global")
            channel_tensor = hint_tensors.get("channel")
            if global_tensor is not None:
                global_tensor = global_tensor.to(device=embedding_device, dtype=self.torch.float32)
                projected_global = self.perceiver.process_local_embeddings(global_tensor, source_embeddings)
            if channel_tensor is not None:
                channel_tensor = channel_tensor.to(device=embedding_device, dtype=self.torch.float32)
                projected_channel = self.perceiver.process_local_embeddings(channel_tensor, source_embeddings)
        global_positions = (input_ids[0] == self.global_hint_token_id).nonzero(as_tuple=True)[0]
        channel_positions = (input_ids[0] == self.channel_hint_token_id).nonzero(as_tuple=True)[0]
        local_positions = (input_ids[0] == self.local_hint_token_id).nonzero(as_tuple=True)[0]
        fixed_positions = (input_ids[0] == self.fixed_hint_token_id).nonzero(as_tuple=True)[0]
        neutral_ids = self.tokenizer("\n", add_special_tokens=False)["input_ids"]
        neutral_id = int(neutral_ids[0]) if neutral_ids else int(self.tokenizer.eos_token_id)
        neutral_embedding = self.model.get_input_embeddings().weight[neutral_id].detach().to(
            device=embedding_device,
            dtype=input_embeddings.dtype,
        )
        if len(global_positions) > 0:
            scale = float(self.axis_hints_config.get("global_injection_scale", self.axis_hints_config.get("injection_scale", 1.0)))
            scale = max(0.0, min(1.0, scale))
            projected = projected_global[: len(global_positions)].to(input_embeddings.dtype) if projected_global is not None else None
            if projected is not None:
                input_embeddings[0, global_positions] = (1.0 - scale) * neutral_embedding + scale * projected
        if len(channel_positions) > 0:
            scale = float(self.axis_hints_config.get("channel_injection_scale", self.axis_hints_config.get("injection_scale", 1.0)))
            scale = max(0.0, min(1.0, scale))
            projected = projected_channel[: len(channel_positions)].to(input_embeddings.dtype) if projected_channel is not None else None
            if projected is not None:
                input_embeddings[0, channel_positions] = (1.0 - scale) * neutral_embedding + scale * projected
        if len(local_positions) > 0 and projected_local is not None:
            scale = float(self.axis_hints_config.get("injection_scale", 1.0))
            scale = max(0.0, min(1.0, scale))
            projected = projected_local[: len(local_positions)].to(input_embeddings.dtype)
            input_embeddings[0, local_positions] = (1.0 - scale) * neutral_embedding + scale * projected
        if len(fixed_positions) > 0:
            fixed_embeddings = self.perceiver.process_fixed_embeddings(source_embeddings, len(fixed_positions))
            projected = fixed_embeddings.to(input_embeddings.dtype)
            scale = float(self.axis_hints_config.get("fixed_injection_scale", self.axis_hints_config.get("injection_scale", 1.0)))
            scale = max(0.0, min(1.0, scale))
            input_embeddings[0, fixed_positions] = (
                (1.0 - scale) * neutral_embedding
                + scale * projected
            )
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": self.use_cache,
        }
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        if outputs.shape[-1] > input_ids.shape[-1]:
            decoded_tokens = outputs[0, input_ids.shape[-1] :]
        else:
            decoded_tokens = outputs[0]
        content = self.tokenizer.decode(decoded_tokens, skip_special_tokens=True).strip()
        metadata = self._generation_metadata(
            prompt_token_count=int(input_ids.shape[-1]),
            generated_tokens=decoded_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return LLMResponse(
            content=content,
            raw={
                "provider": "local_hf",
                "model": self.model_name,
                **metadata,
                "axis_embedding_hints": True,
                "global_hint_tokens": int(len(global_positions)),
                "channel_hint_tokens": int(len(channel_positions)),
                "local_hint_tokens": int(len(local_positions)),
                "fixed_hint_tokens": int(len(fixed_positions)),
                "perceiver_checkpoint": self.axis_hints_config.get("checkpoint_path"),
            },
            latency_seconds=time.perf_counter() - started,
        )


class LocalHFVLChatClient:
    """Local Hugging Face vision-language chat client for Qwen-VL style models."""

    GLOBAL_HINT_TOKEN = "<|global_hint|>"
    CHANNEL_HINT_TOKEN = "<|channel_hint|>"
    GLOBAL_HINT_TOKEN_PLACEHOLDER = "[[AXIS_GLOBAL_HINT_TOKENS]]"
    CHANNEL_HINT_TOKEN_PLACEHOLDER = "[[AXIS_CHANNEL_HINT_TOKENS]]"

    def __init__(self, config: Dict[str, Any]) -> None:
        try:
            import torch
            import torch.nn as nn
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover - depends on local environment
            raise RuntimeError(
                "LocalHFVLChatClient requires torch, transformers, and qwen-vl-utils."
            ) from exc

        self.torch = torch
        self.nn = nn
        self.process_vision_info = process_vision_info
        self.config = config
        self.model_name = str(config["model"])
        self.max_new_tokens = int(config.get("max_tokens", config.get("max_new_tokens", 1024)))
        self.min_new_tokens = max(0, int(config.get("min_new_tokens", 0)))
        self.suppress_tokens = [int(token) for token in (config.get("suppress_tokens") or [])]
        self.temperature = float(config.get("temperature", 0.0))
        self.do_sample = bool(config.get("do_sample", self.temperature > 0))
        self.use_cache = bool(config.get("use_cache", True))
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.device = self._resolve_device(str(config.get("device", "auto")))
        dtype = self._resolve_dtype(str(config.get("torch_dtype", "auto")))
        processor_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if config.get("min_pixels") is not None:
            processor_kwargs["min_pixels"] = int(config["min_pixels"])
        if config.get("max_pixels") is not None:
            processor_kwargs["max_pixels"] = int(config["max_pixels"])
        _client_debug("processor_from_pretrained_start", model=self.model_name)
        self.processor = AutoProcessor.from_pretrained(self.model_name, **processor_kwargs)
        _client_debug("processor_from_pretrained_done", model=self.model_name)
        if getattr(self.processor, "tokenizer", None) is not None and self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        model_kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if config.get("device_map"):
            model_kwargs["device_map"] = config.get("device_map")
            if config.get("max_memory"):
                model_kwargs["max_memory"] = {
                    int(k) if isinstance(k, str) and k.isdigit() else k: v
                    for k, v in dict(config["max_memory"]).items()
                }
            if config.get("offload_folder"):
                model_kwargs["offload_folder"] = str(config["offload_folder"])
            model_kwargs["low_cpu_mem_usage"] = bool(config.get("low_cpu_mem_usage", True))
        _client_debug("vl_model_from_pretrained_start", model=self.model_name, device=str(self.device))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)
        _client_debug("vl_model_from_pretrained_done", model=self.model_name, device=str(self.device))
        if not config.get("device_map"):
            self.model.to(self.device)
        self.model.eval()

        self.axis_hints_config = dict(config.get("axis_hints") or {})
        self.axis_hints_enabled = bool(self.axis_hints_config.get("enabled", False))
        self.perceiver = None
        if self.axis_hints_enabled:
            _client_debug("axis_hints_init_start", checkpoint=str(self.axis_hints_config.get("checkpoint_path") or ""))
            self._init_axis_hints()
            _client_debug("axis_hints_init_done", checkpoint=str(self.axis_hints_config.get("checkpoint_path") or ""))

    def _resolve_device(self, requested: str):
        if requested == "auto":
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return self.torch.device(requested)

    def _resolve_dtype(self, requested: str):
        requested = requested.lower()
        if requested == "auto":
            return self.torch.float16 if self.device.type == "cuda" else None
        if requested in {"float16", "fp16"}:
            return self.torch.float16
        if requested in {"bfloat16", "bf16"}:
            return self.torch.bfloat16
        if requested in {"float32", "fp32"}:
            return self.torch.float32
        if requested in {"none", "default"}:
            return None
        raise ValueError(f"Unsupported torch_dtype: {requested}")

    def _to_vl_messages(self, messages: List[Dict[str, str]], image_paths: List[str]) -> List[Dict[str, Any]]:
        vl_messages: List[Dict[str, Any]] = []
        first_user_done = False
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "user" and not first_user_done and image_paths:
                content: List[Dict[str, str]] = []
                for image_path in image_paths:
                    content.append({"type": "image", "image": str(image_path)})
                content.append({"type": "text", "text": text})
                vl_messages.append({"role": role, "content": content})
                first_user_done = True
            else:
                vl_messages.append({"role": role, "content": [{"type": "text", "text": text}]})
        return vl_messages

    def _init_axis_hints(self) -> None:
        tokenizer = self.processor.tokenizer
        special_tokens = [self.GLOBAL_HINT_TOKEN, self.CHANNEL_HINT_TOKEN]
        added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        embedding_rows = int(self.model.get_input_embeddings().weight.shape[0])
        # Qwen checkpoints pad the embedding table beyond the tokenizer size.
        # Shrinking a dispatched/offloaded table invalidates Accelerate's saved
        # tensor shape; resize only when the new token ids exceed existing rows.
        if added and len(tokenizer) > embedding_rows:
            self.model.resize_token_embeddings(len(tokenizer))
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or self.model.config.text_config.hidden_size)
        d_proj = int(self.axis_hints_config.get("d_proj", 256))
        num_prototype = int(self.axis_hints_config.get("num_prototype", 256))
        num_fixed_tokens = int(self.axis_hints_config.get("num_fixed_tokens", 8))
        num_heads = int(self.axis_hints_config.get("num_heads", 4))
        self.perceiver = _AxisStylePerceiver(
            vocab_size=int(self.model.get_input_embeddings().weight.shape[0]),
            hidden_size=hidden_size,
            d_proj=d_proj,
            num_prototype=num_prototype,
            num_fixed_tokens=num_fixed_tokens,
            num_heads=num_heads,
        ).to(self.device)
        checkpoint_path = self.axis_hints_config.get("checkpoint_path")
        if checkpoint_path:
            state = self.torch.load(str(checkpoint_path), map_location=self.device)
            raw_state = state.get("perceiver", state)
            current_state = self.perceiver.state_dict()
            compatible_state = {
                key: value
                for key, value in raw_state.items()
                if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
            }
            self.perceiver.load_state_dict(compatible_state, strict=False)
        self.perceiver.eval()
        self.global_hint_token_id = tokenizer.convert_tokens_to_ids(self.GLOBAL_HINT_TOKEN)
        self.channel_hint_token_id = tokenizer.convert_tokens_to_ids(self.CHANNEL_HINT_TOKEN)

    def _downsample_hint_tensor(self, tensor: Any, limit: int) -> Any:
        hint_tensor = self.torch.as_tensor(tensor, dtype=self.torch.float32, device=self.device)
        if hint_tensor.ndim != 2:
            raise ValueError("AXIS hint embeddings must have shape [num_tokens, d_proj]")
        if hint_tensor.shape[0] > int(limit):
            idx = self.torch.linspace(0, hint_tensor.shape[0] - 1, int(limit), device=self.device).long()
            hint_tensor = hint_tensor.index_select(0, idx)
        return hint_tensor

    def _prepare_axis_hint_tensors(self, axis_embeddings: Any) -> Dict[str, Any]:
        if not isinstance(axis_embeddings, dict):
            raise ValueError("Vision-language AXIS hints require a dict with global/channel embeddings.")
        tensors: Dict[str, Any] = {}
        global_embeddings = axis_embeddings.get("global")
        channel_embeddings = axis_embeddings.get("channel", axis_embeddings.get("local"))
        if global_embeddings is not None:
            max_global = int(
                self.axis_hints_config.get(
                    "max_global_tokens",
                    self.axis_hints_config.get("num_global_tokens", self.axis_hints_config.get("num_fixed_tokens", 8)),
                )
            )
            tensors["global"] = self._downsample_hint_tensor(global_embeddings, max_global)
        if channel_embeddings is not None:
            max_channel = int(
                self.axis_hints_config.get(
                    "max_channel_tokens",
                    self.axis_hints_config.get("max_local_tokens", 48),
                )
            )
            tensors["channel"] = self._downsample_hint_tensor(channel_embeddings, max_channel)
        if not tensors:
            raise ValueError("No AXIS hint embeddings were provided.")
        return tensors

    def _inject_axis_hint_tokens(
        self,
        messages: List[Dict[str, str]],
        hint_tensors: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        global_tensor = hint_tensors.get("global")
        channel_tensor = hint_tensors.get("channel")
        global_tokens = self.GLOBAL_HINT_TOKEN * int(0 if global_tensor is None else global_tensor.shape[0])
        channel_tokens = self.CHANNEL_HINT_TOKEN * int(0 if channel_tensor is None else channel_tensor.shape[0])
        axis_messages = [dict(m) for m in messages]
        content = axis_messages[-1].get("content", "")
        if self.GLOBAL_HINT_TOKEN_PLACEHOLDER in content or self.CHANNEL_HINT_TOKEN_PLACEHOLDER in content:
            content = content.replace(self.GLOBAL_HINT_TOKEN_PLACEHOLDER, global_tokens, 1)
            content = content.replace(self.CHANNEL_HINT_TOKEN_PLACEHOLDER, channel_tokens, 1)
            axis_messages[-1]["content"] = content
        else:
            axis_messages[-1]["content"] = content + "\n\n" + global_tokens + channel_tokens + "\n"
        return axis_messages

    def complete_with_images(self, messages: List[Dict[str, str]], image_paths: List[str]) -> LLMResponse:
        started = time.perf_counter()
        vl_messages = self._to_vl_messages(messages, image_paths)
        text = self.processor.apply_chat_template(vl_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(vl_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self.model.device if not isinstance(getattr(self.model, "hf_device_map", None), dict) else self.device
        inputs = inputs.to(device)
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }
        if self.min_new_tokens:
            generate_kwargs["min_new_tokens"] = min(self.min_new_tokens, self.max_new_tokens)
        if self.suppress_tokens:
            generate_kwargs["suppress_tokens"] = self.suppress_tokens
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_tokens = trimmed[0]
        content = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        metadata = self._generation_metadata(
            prompt_token_count=int(inputs.input_ids.shape[-1]),
            generated_tokens=generated_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        return LLMResponse(
            content=content,
            raw={
                "provider": "local_hf_vl",
                "model": self.model_name,
                **metadata,
                "image_paths": image_paths,
                "num_images": len(image_paths),
            },
            latency_seconds=time.perf_counter() - started,
        )

    def complete_with_images_and_axis_hints(
        self,
        messages: List[Dict[str, str]],
        image_paths: List[str],
        axis_embeddings: Any,
    ) -> LLMResponse:
        if not self.axis_hints_enabled or self.perceiver is None:
            return self.complete_with_images(messages, image_paths)
        started = time.perf_counter()
        hint_tensors = self._prepare_axis_hint_tensors(axis_embeddings)
        axis_messages = self._inject_axis_hint_tokens(messages, hint_tensors)
        vl_messages = self._to_vl_messages(axis_messages, image_paths)
        text = self.processor.apply_chat_template(vl_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(vl_messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = self.model.device if not isinstance(getattr(self.model, "hf_device_map", None), dict) else self.device
        inputs = inputs.to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        input_embeddings = self.model.get_input_embeddings()(input_ids)
        embedding_device = input_embeddings.device
        if next(self.perceiver.parameters()).device != embedding_device or next(self.perceiver.parameters()).dtype != self.torch.float32:
            self.perceiver.to(device=embedding_device, dtype=self.torch.float32)
        word_embeddings = self.model.get_input_embeddings().weight.detach().to(device=embedding_device, dtype=self.torch.float32)
        source_embeddings = self.perceiver.get_source_embeddings(word_embeddings)

        projected_global = None
        projected_channel = None
        global_tensor = hint_tensors.get("global")
        channel_tensor = hint_tensors.get("channel")
        if global_tensor is not None:
            global_tensor = global_tensor.to(device=embedding_device, dtype=self.torch.float32)
            projected_global = self.perceiver.process_local_embeddings(global_tensor, source_embeddings)
        if channel_tensor is not None:
            channel_tensor = channel_tensor.to(device=embedding_device, dtype=self.torch.float32)
            projected_channel = self.perceiver.process_local_embeddings(channel_tensor, source_embeddings)

        global_positions = (input_ids[0] == self.global_hint_token_id).nonzero(as_tuple=True)[0]
        channel_positions = (input_ids[0] == self.channel_hint_token_id).nonzero(as_tuple=True)[0]
        neutral_ids = self.processor.tokenizer("\n", add_special_tokens=False)["input_ids"]
        neutral_id = int(neutral_ids[0]) if neutral_ids else int(self.processor.tokenizer.eos_token_id)
        neutral_embedding = self.model.get_input_embeddings().weight[neutral_id].detach().to(
            device=embedding_device,
            dtype=input_embeddings.dtype,
        )
        if len(global_positions) > 0 and projected_global is not None:
            scale = float(self.axis_hints_config.get("global_injection_scale", self.axis_hints_config.get("injection_scale", 1.0)))
            scale = max(0.0, min(1.0, scale))
            projected = projected_global[: len(global_positions)].to(input_embeddings.dtype)
            input_embeddings[0, global_positions] = (1.0 - scale) * neutral_embedding + scale * projected
        if len(channel_positions) > 0 and projected_channel is not None:
            scale = float(self.axis_hints_config.get("channel_injection_scale", self.axis_hints_config.get("injection_scale", 1.0)))
            scale = max(0.0, min(1.0, scale))
            projected = projected_channel[: len(channel_positions)].to(input_embeddings.dtype)
            input_embeddings[0, channel_positions] = (1.0 - scale) * neutral_embedding + scale * projected

        generate_kwargs = {
            "input_ids": input_ids,
            "inputs_embeds": input_embeddings,
            "attention_mask": attention_mask,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "use_cache": self.use_cache,
        }
        if self.min_new_tokens:
            generate_kwargs["min_new_tokens"] = min(self.min_new_tokens, self.max_new_tokens)
        if self.suppress_tokens:
            generate_kwargs["suppress_tokens"] = self.suppress_tokens
        for optional_key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "second_per_grid_ts"):
            if optional_key in inputs:
                generate_kwargs[optional_key] = inputs[optional_key]
        if self.do_sample:
            generate_kwargs["temperature"] = self.temperature
        with self.torch.no_grad():
            generated_ids = self.model.generate(**generate_kwargs)
        trimmed = [
            output_ids[len(prompt_ids) :]
            for prompt_ids, output_ids in zip(input_ids, generated_ids)
        ]
        generated_tokens = trimmed[0]
        content = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        metadata = self._generation_metadata(
            prompt_token_count=int(input_ids.shape[-1]),
            generated_tokens=generated_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        return LLMResponse(
            content=content,
            raw={
                "provider": "local_hf_vl",
                "model": self.model_name,
                **metadata,
                "image_paths": image_paths,
                "num_images": len(image_paths),
                "axis_embedding_hints": True,
                "global_hint_tokens": int(len(global_positions)),
                "channel_hint_tokens": int(len(channel_positions)),
                "perceiver_checkpoint": self.axis_hints_config.get("checkpoint_path"),
            },
            latency_seconds=time.perf_counter() - started,
        )

    def complete_with_axis_hints(self, messages: List[Dict[str, str]], axis_embeddings: Any) -> LLMResponse:
        return self.complete_with_images_and_axis_hints(messages, [], axis_embeddings)

    def complete(self, messages: List[Dict[str, str]]) -> LLMResponse:
        return self.complete_with_images(messages, [])

    def get_input_token_embeddings(
        self,
        texts: List[str],
        *,
        max_length: int = 128,
    ):
        tokenizer = self.processor.tokenizer
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        device = self.model.device if not isinstance(getattr(self.model, "hf_device_map", None), dict) else self.device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with self.torch.no_grad():
            embeddings = self.model.get_input_embeddings()(input_ids)
        return embeddings.detach(), attention_mask.bool()

    def _generation_metadata(self, *, prompt_token_count: int, generated_tokens: Any, eos_token_id: Any) -> Dict[str, Any]:
        generated_token_count = int(generated_tokens.shape[-1]) if hasattr(generated_tokens, "shape") else int(len(generated_tokens))
        eos_ids: List[int] = []
        if eos_token_id is None:
            eos_ids = []
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids = [int(x) for x in eos_token_id]
        else:
            eos_ids = [int(eos_token_id)]
        last_token_id = int(generated_tokens[-1].item()) if generated_token_count > 0 else None
        hit_max_new_tokens = bool(generated_token_count >= int(self.max_new_tokens))
        if hit_max_new_tokens:
            finish_reason = "length"
        elif last_token_id is not None and last_token_id in eos_ids:
            finish_reason = "stop"
        elif generated_token_count == 0:
            finish_reason = "empty"
        else:
            finish_reason = "unknown"
        return {
            "prompt_token_count": int(prompt_token_count),
            "generated_token_count": generated_token_count,
            "total_sequence_token_count": int(prompt_token_count) + generated_token_count,
            "last_generated_token_id": last_token_id,
            "finish_reason": finish_reason,
            "hit_max_new_tokens": hit_max_new_tokens,
        }


class _AxisStyleMultiheadAttention:
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self, embed_dim: int, num_heads: int) -> None:
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                if self.head_dim * num_heads != embed_dim:
                    raise ValueError("embed_dim must be divisible by num_heads")
                self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
                self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
                self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
                self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

            def forward(self, query, key, value):
                import torch.nn.functional as F

                q = self.q_proj(query)
                k = self.k_proj(key)
                v = self.v_proj(value)
                q = q.view(query.shape[0], query.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(key.shape[0], key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(value.shape[0], value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                y = y.transpose(1, 2).contiguous().view(query.shape[0], query.shape[1], -1)
                return self.out_proj(y)

        return Module(*args, **kwargs)


class _AxisStylePerceiver:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(
                self,
                vocab_size: int,
                hidden_size: int,
                d_proj: int,
                num_prototype: int,
                num_fixed_tokens: int,
                num_heads: int,
            ) -> None:
                super().__init__()
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.d_proj = d_proj
                self.num_prototype = num_prototype
                self.num_fixed_tokens = num_fixed_tokens
                self.mapping_layer = nn.Linear(vocab_size, num_prototype)
                self.fix_prompt_embeddings = nn.Parameter(torch.randn(1, num_fixed_tokens, hidden_size))
                self.local_word_proj = nn.Linear(d_proj, hidden_size)
                self.local_attention = _AxisStyleMultiheadAttention(hidden_size, num_heads)
                self._init_parameters()

            def _init_parameters(self) -> None:
                nn.init.xavier_uniform_(self.mapping_layer.weight)
                nn.init.zeros_(self.mapping_layer.bias)
                nn.init.xavier_uniform_(self.local_word_proj.weight)
                nn.init.zeros_(self.local_word_proj.bias)
                for layer in [
                    self.local_attention.q_proj,
                    self.local_attention.k_proj,
                    self.local_attention.v_proj,
                    self.local_attention.out_proj,
                ]:
                    nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(self.fix_prompt_embeddings, mean=0.0, std=0.02)

            def get_source_embeddings(self, word_embeddings):
                return self.mapping_layer(word_embeddings.permute(1, 0)).permute(1, 0).unsqueeze(0)

            def process_local_embeddings(self, local_embeddings, source_embeddings):
                local_ts_embeddings = self.local_word_proj(local_embeddings.unsqueeze(0))
                return self.local_attention(local_ts_embeddings, source_embeddings, source_embeddings).squeeze(0)

            def process_fixed_embeddings(self, source_embeddings, num_tokens: int):
                fixed_embeddings = self.fix_prompt_embeddings.squeeze(0)[:num_tokens].unsqueeze(0)
                return self.local_attention(fixed_embeddings, source_embeddings, source_embeddings).squeeze(0)

        return Module(*args, **kwargs)


class MockLLMClient:
    def __init__(self, mode: str = "target") -> None:
        self.mode = mode

    def complete_with_target(self, target_output: Dict[str, Any]) -> LLMResponse:
        started = time.perf_counter()
        if self.mode == "target":
            content = json.dumps(target_output, ensure_ascii=False)
        else:
            content = json.dumps(
                {
                    "fact_check": {
                        "is_anomalous": False,
                        "root_cause_channel": None,
                        "affected_channels": [],
                        "anomaly_type": None,
                        "has_relation_break": None,
                        "left_right_gap_level": "unknown",
                    },
                    "reasoning_summary": "Mock response.",
                    "final_answer": "Mock response.",
                    "abnormality_score": 0.0,
                    "answer_confidence": 0.0,
                },
                ensure_ascii=False,
            )
        return LLMResponse(content=content, raw={"mock": True}, latency_seconds=time.perf_counter() - started)


def create_llm_client(config: Dict[str, Any]):
    provider = str(config.get("provider", "openai_compatible")).lower()
    if provider in {"deepseek", "openai", "openai_compatible", "api"}:
        return OpenAICompatibleChatClient(config)
    if provider in {"local", "local_hf", "huggingface", "hf"}:
        return LocalHFChatClient(config)
    if provider in {"local_hf_vl", "qwen_vl", "vl", "vision_language"}:
        return LocalHFVLChatClient(config)
    raise ValueError(f"Unsupported LLM provider: {provider}")
