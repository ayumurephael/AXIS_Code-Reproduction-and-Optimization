from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FlashCrossAttention
from .config import MultiAxisConfig
from .prompting import HINT_TOKENS, MultiAxisPromptBuilder, TokenizedPrompts
from .question_semantics import QuestionSemanticDiskCache, question_semantic_text
from .timercd import FrozenTimeRCD


def rms_unit(value: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    mean_square = value.float().square().mean(dim=-1, keepdim=True)
    return (value * torch.rsqrt(mean_square + epsilon)).to(value.dtype)


@dataclass
class AnswerOnlyCausalLMOutput:
    loss: torch.Tensor
    supervised_token_count: int


def answer_only_causal_nll(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    output_embeddings: nn.Module,
) -> AnswerOnlyCausalLMOutput:
    """Exact causal-LM NLL without materializing prompt-position vocabulary logits."""
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "hidden_states and labels must have shapes [B, L, D] and [B, L]"
        )
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError(
            "hidden_states and labels must share batch/sequence dimensions"
        )
    shifted_labels = labels[:, 1:].contiguous()
    supervised = shifted_labels.ne(-100)
    supervised_token_count = int(supervised.sum().item())
    if supervised_token_count == 0:
        raise ValueError("Answer-only supervision contains no causal target tokens")
    selected_hidden = hidden_states[:, :-1, :][supervised]
    selected_labels = shifted_labels[supervised]
    answer_logits = output_embeddings(selected_hidden)
    loss = F.cross_entropy(answer_logits.float(), selected_labels, reduction="mean")
    return AnswerOnlyCausalLMOutput(
        loss=loss,
        supervised_token_count=supervised_token_count,
    )


class MultiAxisHintTuner(nn.Module):
    """The complete trainable part of multivariate AXIS."""

    def __init__(
        self,
        vocab_size: int,
        llm_hidden_size: int,
        d_proj: int,
        config,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.llm_hidden_size = llm_hidden_size
        self.d_proj = d_proj
        self.config = config
        phase = getattr(config, "ablation_phase", "D")
        self.use_anomaly_evidence = phase in {"B", "D"}
        self.use_joint_hint = phase in {"C", "D"}
        self.prototype_mapping = nn.Parameter(
            torch.empty(config.num_prototypes, vocab_size)
        )
        self.anomaly_direction = nn.Parameter(torch.zeros(d_proj))
        self.step_projection = nn.Linear(d_proj, llm_hidden_size)
        self.prototype_attention = FlashCrossAttention(
            llm_hidden_size,
            config.prototype_heads,
            require_flash=config.require_flash_attention,
        )
        self.channel_query = nn.Parameter(torch.empty(1, 1, d_proj))
        self.channel_pool = FlashCrossAttention(
            d_proj,
            config.channel_heads,
            require_flash=config.require_flash_attention,
        )
        self.joint_query = nn.Parameter(torch.empty(1, 1, d_proj))
        self.joint_pool = FlashCrossAttention(
            d_proj,
            config.joint_heads,
            require_flash=config.require_flash_attention,
        )
        self.question_projection = nn.Linear(llm_hidden_size, d_proj, bias=False)
        self.fixed_queries = nn.Parameter(
            torch.empty(1, config.fixed_tokens, llm_hidden_size)
        )
        self.reset_parameters()
        if not self.use_anomaly_evidence:
            self.anomaly_direction.requires_grad_(False)
        if not self.use_joint_hint:
            self.joint_query.requires_grad_(False)
            for parameter in self.joint_pool.parameters():
                parameter.requires_grad_(False)
            for parameter in self.question_projection.parameters():
                parameter.requires_grad_(False)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.prototype_mapping)
        nn.init.zeros_(self.anomaly_direction)
        nn.init.xavier_uniform_(self.step_projection.weight)
        nn.init.zeros_(self.step_projection.bias)
        nn.init.normal_(self.channel_query, mean=0.0, std=0.02)
        nn.init.normal_(self.joint_query, mean=0.0, std=0.02)
        nn.init.zeros_(self.question_projection.weight)
        nn.init.normal_(self.fixed_queries, mean=0.0, std=0.02)

    def build_prototype_bank(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        if word_embeddings.shape != (self.vocab_size, self.llm_hidden_size):
            raise ValueError(
                f"Vocabulary embedding mismatch: {tuple(word_embeddings.shape)} != "
                f"{(self.vocab_size, self.llm_hidden_size)}"
            )
        return torch.matmul(self.prototype_mapping, word_embeddings)

    def forward(
        self,
        local_embeddings: torch.Tensor,
        anomaly_logits: torch.Tensor,
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        question_semantics: Optional[torch.Tensor],
        word_embeddings: torch.Tensor,
        prototype_override: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        batch, total_steps, max_channels, d_proj = local_embeddings.shape
        if d_proj != self.d_proj or anomaly_logits.shape != (
            batch,
            total_steps,
            max_channels,
            2,
        ):
            raise ValueError("TimeRCD embeddings/logits have incompatible shapes")
        if len(intervals) != batch or len(channel_counts) != batch:
            raise ValueError(
                "Interval/channel-count batches do not match TimeRCD output"
            )

        signed_evidence = torch.tanh(
            (anomaly_logits[..., 1] - anomaly_logits[..., 0]) / 2.0
        )
        enhanced_samples: List[torch.Tensor] = []
        enhanced_channels: List[torch.Tensor] = []
        for index, ((start, end), channels) in enumerate(
            zip(intervals, channel_counts)
        ):
            if not (0 <= start < end <= total_steps):
                raise ValueError(
                    f"Invalid interval [{start}, {end}) for T={total_steps}"
                )
            if not (1 <= channels <= max_channels):
                raise ValueError(
                    f"Invalid channel count {channels} for C={max_channels}"
                )
            # [L,C,D] -> [C,L,D] -> [C*L,D], exactly channel-major.
            step = (
                local_embeddings[index, start:end, :channels]
                .transpose(0, 1)
                .reshape(-1, d_proj)
            )
            evidence = (
                (
                    signed_evidence[index, start:end, :channels]
                    .transpose(0, 1)
                    .reshape(-1)
                )
                if self.use_anomaly_evidence
                else signed_evidence.new_zeros(step.shape[0])
            )
            enhanced = rms_unit(
                rms_unit(step, self.config.representation_epsilon)
                + evidence[:, None] * self.anomaly_direction[None, :],
                self.config.representation_epsilon,
            )
            enhanced_samples.append(enhanced)
            enhanced_channels.append(enhanced.view(channels, end - start, d_proj))

        max_step_hints = max(sample.shape[0] for sample in enhanced_samples)
        enhanced_padded = local_embeddings.new_zeros(batch, max_step_hints, self.d_proj)
        step_mask = torch.zeros(
            batch, max_step_hints, dtype=torch.bool, device=local_embeddings.device
        )
        for index, enhanced in enumerate(enhanced_samples):
            count = enhanced.shape[0]
            enhanced_padded[index, :count] = enhanced
            step_mask[index, :count] = True

        prototype_bank = (
            prototype_override
            if prototype_override is not None
            else self.build_prototype_bank(word_embeddings)
        )
        expected = (self.config.num_prototypes, self.llm_hidden_size)
        if prototype_bank.shape != expected:
            raise ValueError(
                f"Prototype bank shape {tuple(prototype_bank.shape)} != {expected}"
            )
        prototypes = prototype_bank.unsqueeze(0).expand(batch, -1, -1)

        step_queries = self.step_projection(enhanced_padded)
        step_padded = self.prototype_attention(step_queries, prototypes, prototypes)
        step_hints = [step_padded[i, step_mask[i]] for i in range(batch)]

        # The same learned query pools each channel independently over its full
        # target-window timeline. Keys and values remain channel-specific.
        total_channels = sum(channel_counts)
        max_window_length = max(memory.shape[1] for memory in enhanced_channels)
        channel_memory = local_embeddings.new_zeros(
            total_channels, max_window_length, self.d_proj
        )
        channel_mask = torch.zeros(
            total_channels,
            max_window_length,
            dtype=torch.bool,
            device=local_embeddings.device,
        )
        channel_slices: List[Tuple[int, int]] = []
        cursor = 0
        for memory in enhanced_channels:
            channels, length, _ = memory.shape
            channel_memory[cursor : cursor + channels, :length] = memory
            channel_mask[cursor : cursor + channels, :length] = True
            channel_slices.append((cursor, cursor + channels))
            cursor += channels
        channel_query = self.channel_query.expand(total_channels, -1, -1)
        # Deliberately no residual connection from the shared channel query.
        channel_repr = self.channel_pool(
            channel_query,
            channel_memory,
            channel_memory,
            key_padding_mask=channel_mask,
        )
        channel_queries = self.step_projection(
            rms_unit(channel_repr, self.config.representation_epsilon)
        )
        channel_prototypes = prototype_bank.unsqueeze(0).expand(
            total_channels, -1, -1
        )
        channel_pooled = self.prototype_attention(
            channel_queries, channel_prototypes, channel_prototypes
        )
        channel_hints = [
            channel_pooled[start:end, 0] for start, end in channel_slices
        ]

        joint_hint = None
        if self.use_joint_hint:
            if question_semantics is None:
                raise ValueError(
                    "Question-conditioned Joint-Local pooling requires question semantics"
                )
            expected_question_shape = (batch, self.llm_hidden_size)
            if tuple(question_semantics.shape) != expected_question_shape:
                raise ValueError(
                    f"Question semantics shape {tuple(question_semantics.shape)} != "
                    f"{expected_question_shape}"
                )
            semantic_delta = self.question_projection(
                rms_unit(question_semantics, self.config.representation_epsilon)
            )
            joint_query = self.joint_query.expand(batch, -1, -1) + semantic_delta[:, None]
            # Deliberately no residual connection from the sample-independent query.
            joint_repr = self.joint_pool(
                joint_query,
                enhanced_padded,
                enhanced_padded,
                key_padding_mask=step_mask,
            )
            joint_queries = self.step_projection(
                rms_unit(joint_repr, self.config.representation_epsilon)
            )
            joint_hint = self.prototype_attention(joint_queries, prototypes, prototypes)

        fixed_queries = self.fixed_queries.expand(batch, -1, -1)
        fixed_hints = self.prototype_attention(fixed_queries, prototypes, prototypes)
        return step_hints, channel_hints, joint_hint, fixed_hints


class MultiAxisForConditionalGeneration(nn.Module):
    def __init__(self, llm: nn.Module, tokenizer, config: MultiAxisConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.llm = llm
        self.tokenizer = tokenizer
        self._prepare_tokenizer()
        self._freeze_llm()
        hidden_size = self._hidden_size()
        vocab_size = self.llm.get_input_embeddings().weight.shape[0]
        self.timercd = FrozenTimeRCD(config.timercd)
        self.hint_tuner = MultiAxisHintTuner(
            vocab_size=vocab_size,
            llm_hidden_size=hidden_size,
            d_proj=config.timercd.d_proj,
            config=config.hints,
        )
        self.prompt_builder = MultiAxisPromptBuilder(
            tokenizer,
            fixed_tokens=config.hints.fixed_tokens,
            max_context_tokens=config.llm.max_context_tokens,
            include_joint=config.hints.ablation_phase in {"C", "D"},
            window_epsilon=config.hints.window_epsilon,
        )
        self.question_semantic_cache: Optional[QuestionSemanticDiskCache] = None

    @classmethod
    def from_pretrained(
        cls,
        config: MultiAxisConfig,
        token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, config.llm.torch_dtype)
        source = os.environ.get("MULTI_AXIS_MODEL_PATH") or config.llm.model_name
        if source != config.llm.model_name and not Path(source).is_dir():
            raise FileNotFoundError(
                f"MULTI_AXIS_MODEL_PATH is not a directory: {source}"
            )
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            token=token,
            trust_remote_code=config.llm.trust_remote_code,
            use_fast=True,
        )
        kwargs = dict(
            token=token,
            trust_remote_code=config.llm.trust_remote_code,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=config.llm.attention_implementation,
        )
        if device is not None:
            kwargs["device_map"] = {"": str(device)}
        try:
            llm = AutoModelForCausalLM.from_pretrained(source, **kwargs)
        except ValueError:
            # Qwen3.5 releases may register through the unified image-text auto class
            # even when used text-only. DeepSeek uses the causal-LM path above.
            from transformers import AutoModelForImageTextToText

            llm = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
        instance = cls(llm=llm, tokenizer=tokenizer, config=config)
        instance.pretrained_source = str(source)
        return instance

    def _hidden_size(self) -> int:
        if hasattr(self.llm.config, "hidden_size"):
            return int(self.llm.config.hidden_size)
        text_config = getattr(self.llm.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "hidden_size"):
            return int(text_config.hidden_size)
        raise AttributeError("Could not determine the LLM text hidden size")

    def _prepare_tokenizer(self) -> None:
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(HINT_TOKENS)}
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer has neither pad nor EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if added:
            self.llm.resize_token_embeddings(len(self.tokenizer))
        for token in HINT_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise AssertionError(
                    f"Registered hint token {token} is not atomic: {ids}"
                )

    def _freeze_llm(self) -> None:
        for parameter in self.llm.parameters():
            parameter.requires_grad_(False)
        self.llm.eval()
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = self.config.llm.use_cache_during_training
        if self.config.llm.gradient_checkpointing and hasattr(
            self.llm, "gradient_checkpointing_enable"
        ):
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.llm.gradient_checkpointing_enable()

    def _set_llm_runtime_mode(self, training: bool) -> None:
        if training and self.config.llm.gradient_checkpointing:
            # Transformers activates decoder checkpointing only in train mode.
            # Parameters stay frozen and dropout stays disabled for determinism.
            self.llm.train(True)
            for module in self.llm.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        else:
            self.llm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._set_llm_runtime_mode(mode)
        self.timercd.eval()
        self.hint_tuner.train(mode)
        return self

    def _llm_backbone(self) -> nn.Module:
        backbone = getattr(self.llm, "model", None)
        if backbone is None:
            prefix = getattr(self.llm, "base_model_prefix", "")
            backbone = getattr(self.llm, prefix, None) if prefix else None
        if backbone is None or backbone is self.llm:
            raise AttributeError(
                "Could not locate the causal LLM backbone for sparse answer NLL"
            )
        return backbone

    def attention_backend_audit(self) -> Dict[str, Any]:
        """Fail closed unless every formal attention path resolves to FlashAttention 2."""
        requested = self.config.llm.attention_implementation
        if not self.config.hints.require_flash_attention:
            raise RuntimeError("Formal execution requires require_flash_attention=true")
        if requested != "flash_attention_2":
            raise RuntimeError(
                f"Formal execution requested attention backend {requested!r}, not flash_attention_2"
            )

        config_objects = [("llm.config", self.llm.config)]
        text_config = getattr(self.llm.config, "text_config", None)
        if text_config is not None and text_config is not self.llm.config:
            config_objects.append(("llm.config.text_config", text_config))
        resolved_configs = {}
        for name, value in config_objects:
            resolved = getattr(value, "_attn_implementation", None)
            internal = getattr(value, "_attn_implementation_internal", None)
            resolved_configs[name] = {
                "resolved": resolved,
                "internal": internal,
            }
        effective = [
            item["resolved"] or item["internal"] for item in resolved_configs.values()
        ]
        if not effective or any(item != "flash_attention_2" for item in effective):
            raise RuntimeError(
                "LLM attention backend did not resolve to flash_attention_2: "
                f"{resolved_configs}"
            )

        attention_modules = []
        for name, module in self.llm.named_modules():
            class_name = type(module).__name__
            if name.endswith("self_attn") or class_name.lower().endswith("attention"):
                module_config = getattr(module, "config", None)
                if module_config is None:
                    continue
                resolved = getattr(module_config, "_attn_implementation", None)
                internal = getattr(module_config, "_attn_implementation_internal", None)
                implementation = resolved or internal
                attention_modules.append(
                    {
                        "name": name,
                        "class": class_name,
                        "implementation": implementation,
                    }
                )
        if not attention_modules:
            raise RuntimeError(
                "Could not identify any LLM attention modules for backend audit"
            )
        invalid_modules = [
            item
            for item in attention_modules
            if item["implementation"] != "flash_attention_2"
        ]
        if invalid_modules:
            raise RuntimeError(
                "One or more LLM attention modules resolved away from flash_attention_2: "
                f"{invalid_modules[:5]}"
            )

        try:
            import flash_attn
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except Exception as error:
            raise RuntimeError("FlashAttention kernels are not importable") from error
        if not callable(flash_attn_func) or not callable(flash_attn_varlen_func):
            raise RuntimeError(
                "FlashAttention dense/varlen kernel entry points are unavailable"
            )

        named_cross_attention_modules = {
            name: module
            for name, module in self.hint_tuner.named_modules()
            if isinstance(module, FlashCrossAttention)
        }
        expected_cross_attention_modules = {
            "prototype_attention",
            "channel_pool",
            "joint_pool",
        }
        if set(named_cross_attention_modules) != expected_cross_attention_modules:
            raise RuntimeError(
                "Hint Tuner cross-attention inventory is incomplete or unexpected: "
                f"{sorted(named_cross_attention_modules)}"
            )
        if any(
            not module.require_flash
            for module in named_cross_attention_modules.values()
        ):
            raise RuntimeError(
                "Hint Tuner cross-attention is not fail-closed on FlashAttention"
            )
        eager_cross_attention = [
            name
            for name, module in self.hint_tuner.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        ]
        if eager_cross_attention:
            raise RuntimeError(
                "Ordinary MultiheadAttention remains in the Hint Tuner: "
                f"{eager_cross_attention}"
            )

        return {
            "requested": requested,
            "llm_resolved_configs": resolved_configs,
            "llm_attention_module_count": len(attention_modules),
            "llm_attention_module_classes": sorted(
                {item["class"] for item in attention_modules}
            ),
            "hint_cross_attention_module_count": len(named_cross_attention_modules),
            "hint_cross_attention_modules": sorted(named_cross_attention_modules),
            "hint_cross_attention_backend": "flash-attn-v2-package",
            "flash_attn_version": getattr(flash_attn, "__version__", "unknown"),
            "dense_kernel_importable": True,
            "varlen_kernel_importable": True,
            "fallback_allowed": False,
        }

    def trainable_parameter_names(self) -> List[str]:
        return [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]

    def assert_freeze_contract(self) -> None:
        leaked = [name for name, p in self.llm.named_parameters() if p.requires_grad]
        leaked += [
            f"timercd.{name}"
            for name, p in self.timercd.named_parameters()
            if p.requires_grad
        ]
        if leaked:
            raise AssertionError(f"Frozen parameter contract violated: {leaked[:10]}")
        expected_prefix = "hint_tuner."
        unexpected = [
            name
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith(expected_prefix)
        ]
        if unexpected:
            raise AssertionError(f"Unexpected trainable parameters: {unexpected}")

    def load_timercd_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        return self.timercd.load_checkpoint(checkpoint_path, strict=True)

    def build_prototype_bank(self) -> torch.Tensor:
        return self.hint_tuner.build_prototype_bank(
            self.llm.get_input_embeddings().weight
        )

    def configure_question_semantic_cache(self, cache_dir: str | Path) -> None:
        self.question_semantic_cache = QuestionSemanticDiskCache(
            cache_dir,
            model_id=self.config.llm.model_name,
            tokenizer=self.tokenizer,
            hidden_size=self._hidden_size(),
        )

    def question_semantic_cache_manifest(self) -> Dict[str, object]:
        if self.question_semantic_cache is None:
            return {"configured": False}
        return {"configured": True, **self.question_semantic_cache.manifest()}

    def _compute_question_semantics(
        self, questions: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        texts = [question_semantic_text(question) for question in questions]
        previous_padding = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(
                texts,
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = previous_padding
        input_ids = encoded["input_ids"].to(device, non_blocking=True)
        attention_mask = encoded["attention_mask"].to(device, non_blocking=True)
        last_indices = attention_mask.sum(dim=-1) - 1
        if bool((last_indices < 0).any()):
            raise ValueError("Question tokenization produced an empty sequence")
        self.llm.eval()
        autocast = (
            torch.autocast("cuda", dtype=getattr(torch, self.config.llm.torch_dtype))
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        try:
            with torch.no_grad(), autocast:
                outputs = self._llm_backbone()(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                result = outputs.last_hidden_state[
                    torch.arange(len(texts), device=device), last_indices
                ]
        finally:
            self._set_llm_runtime_mode(self.training)
        return result.detach()

    def _question_semantics(
        self, questions: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        if self.question_semantic_cache is None:
            raise RuntimeError(
                "Question-conditioned Joint-Local requires a configured disk cache"
            )
        values: List[Optional[torch.Tensor]] = [
            self.question_semantic_cache.load(question) for question in questions
        ]
        missing = [index for index, value in enumerate(values) if value is None]
        if missing:
            computed = self._compute_question_semantics(
                [questions[index] for index in missing], device
            )
            for row, index in enumerate(missing):
                values[index] = self.question_semantic_cache.store(
                    questions[index], computed[row]
                )
        if any(value is None for value in values):
            raise AssertionError("Question semantic cache fill was incomplete")
        stacked = torch.stack([value for value in values if value is not None])
        return stacked.to(device=device, non_blocking=True).detach()

    def _hint_embeddings(
        self,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        questions: Sequence[str],
        prototype_override: Optional[torch.Tensor],
        timercd_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], torch.Tensor
    ]:
        if timercd_override is None:
            local, anomaly_logits = self.timercd(
                normalized_series, time_mask, channel_mask
            )
        else:
            local, anomaly_logits = timercd_override
        question_semantics = (
            self._question_semantics(questions, normalized_series.device)
            if self.hint_tuner.use_joint_hint
            else None
        )
        return self.hint_tuner(
            local,
            anomaly_logits,
            intervals,
            channel_counts,
            question_semantics,
            self.llm.get_input_embeddings().weight,
            prototype_override=prototype_override,
        )

    def _inject_hints(
        self,
        tokenized: TokenizedPrompts,
        step_hints: Sequence[torch.Tensor],
        channel_hints: Sequence[torch.Tensor],
        joint_hints: Optional[torch.Tensor],
        fixed_hints: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        input_ids = tokenized.input_ids.to(device)
        attention_mask = tokenized.attention_mask.to(device)
        labels = tokenized.labels.to(device) if tokenized.labels is not None else None
        embeddings = self.llm.get_input_embeddings()(input_ids).clone()
        for index in range(input_ids.shape[0]):
            step_pos = tokenized.step_positions[index].to(device)
            channel_pos = tokenized.channel_positions[index].to(device)
            joint_pos = tokenized.joint_positions[index].to(device)
            fixed_pos = tokenized.fixed_positions[index].to(device)
            if step_hints[index].shape[0] != step_pos.numel():
                raise AssertionError(
                    "Step hint count changed between prompt and encoder"
                )
            embeddings[index, step_pos] = step_hints[index].to(embeddings.dtype)
            if channel_hints[index].shape[0] != channel_pos.numel():
                raise AssertionError(
                    "Channel hint count changed between prompt and encoder"
                )
            embeddings[index, channel_pos] = channel_hints[index].to(
                embeddings.dtype
            )
            if joint_hints is not None and joint_pos.numel():
                embeddings[index, joint_pos] = joint_hints[index].to(embeddings.dtype)
            embeddings[index, fixed_pos] = fixed_hints[index].to(embeddings.dtype)
        return input_ids, attention_mask, labels, embeddings

    def forward(
        self,
        *,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        questions: Sequence[str],
        answers: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_means: Sequence[Sequence[float]],
        channel_stds: Sequence[Sequence[float]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        prototype_override: Optional[torch.Tensor] = None,
        timercd_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.assert_freeze_contract()
        tokenized = self.prompt_builder.tokenize(
            questions,
            intervals,
            window_values,
            channel_means,
            channel_stds,
            channel_ids,
            question_groups,
            answers=answers,
        )
        step, channel, joint, fixed = self._hint_embeddings(
            normalized_series,
            time_mask,
            channel_mask,
            intervals,
            channel_counts,
            questions,
            prototype_override,
            timercd_override,
        )
        device = normalized_series.device
        _, attention_mask, labels, embeddings = self._inject_hints(
            tokenized, step, channel, joint, fixed, device
        )
        if labels is None:
            raise ValueError("Training forward requires answer labels")
        backbone_outputs = self._llm_backbone()(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return answer_only_causal_nll(
            backbone_outputs.last_hidden_state,
            labels,
            self.llm.get_output_embeddings(),
        )

    @torch.no_grad()
    def generate_answers(
        self,
        *,
        normalized_series: torch.Tensor,
        time_mask: torch.Tensor,
        channel_mask: torch.Tensor,
        questions: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        channel_counts: Sequence[int],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_means: Sequence[Sequence[float]],
        channel_stds: Sequence[Sequence[float]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        prototype_override: Optional[torch.Tensor] = None,
        timercd_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        generation_overrides: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        self.eval()
        tokenized = self.prompt_builder.tokenize(
            questions,
            intervals,
            window_values,
            channel_means,
            channel_stds,
            channel_ids,
            question_groups,
            answers=None,
        )
        step, channel, joint, fixed = self._hint_embeddings(
            normalized_series,
            time_mask,
            channel_mask,
            intervals,
            channel_counts,
            questions,
            prototype_override,
            timercd_override,
        )
        device = normalized_series.device
        input_ids, attention_mask, _, embeddings = self._inject_hints(
            tokenized, step, channel, joint, fixed, device
        )
        kwargs = asdict(self.config.generation)
        if generation_overrides:
            kwargs.update(dict(generation_overrides))
        forbidden = self.prompt_builder.forbidden_reasoning_token_sequences()
        supplied_bad_words = kwargs.pop("bad_words_ids", None)
        if supplied_bad_words:
            forbidden.extend(supplied_bad_words)
        kwargs["bad_words_ids"] = forbidden
        old_cache = getattr(self.llm.config, "use_cache", None)
        if old_cache is not None:
            self.llm.config.use_cache = True
        try:
            sequences = self.llm.generate(
                input_ids=input_ids,
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        finally:
            if old_cache is not None:
                self.llm.config.use_cache = old_cache
        prompt_width = input_ids.shape[1]
        decoded = []
        for sequence in sequences:
            generated = (
                sequence[prompt_width:] if sequence.numel() > prompt_width else sequence
            )
            decoded.append(
                self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            )
        return decoded

    def hint_checkpoint_payload(
        self, epoch: int, global_step: int, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "format": "multi-axis-hints-v2-channel-question",
            "architecture": "step-channel-question-conditioned-joint-v1",
            "epoch": epoch,
            "global_step": global_step,
            "config": self.config.to_dict(),
            "timercd_sha256": self.timercd.checkpoint_sha256,
            "hint_tuner_state_dict": self.hint_tuner.state_dict(),
            "metadata": metadata,
        }

    def load_hint_checkpoint(
        self, path: str | Path, strict: bool = True
    ) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("format") != "multi-axis-hints-v2-channel-question":
            raise ValueError("Not a Multi-AXIS hint checkpoint")
        expected_hash = payload.get("timercd_sha256")
        if expected_hash and self.timercd.checkpoint_sha256 != expected_hash:
            raise ValueError(
                "TimeRCD checkpoint hash does not match the hint checkpoint"
            )
        self.hint_tuner.load_state_dict(payload["hint_tuner_state_dict"], strict=strict)
        return payload
