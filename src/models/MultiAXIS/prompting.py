from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .channel_ids import canonical_channel_ids
from .response_contracts import OUTPUT_CONTRACTS


STEP_TOKEN = "<STEP_HINT>"
CHANNEL_TOKEN = "<CHANNEL_HINT>"
JOINT_TOKEN = "<JOINT_HINT>"
FIXED_TOKEN = "<FIXED_HINT>"
HINT_TOKENS = (STEP_TOKEN, CHANNEL_TOKEN, JOINT_TOKEN, FIXED_TOKEN)

SYSTEM_TEXT = """You are an expert in multivariate time-series analysis. Answer the exact
question using the numeric and learned contextual evidence supplied in the
user message.

The numeric Window is a compact, approximately reversible representation of
observed values from the target interval. Treat
learned contextual hints as supporting model evidence, not as ground-truth
labels or calibrated anomaly probabilities. Treat task-prior hints as general
task information rather than sample-specific facts.

Return only the visible final answer and its evidence-based explanation.
Do not expose private chain-of-thought, scratch work, hidden-reasoning tags,
placeholder tokens, or internal representations. Follow the Output Contract
in the user message exactly."""

USER_PREAMBLE = """You are solving a multivariate time-series question.

Return only the visible final response required by the Output Contract.
Do not expose private chain-of-thought, scratch work, or hidden-reasoning tags."""

EVIDENCE_RULES = """### Evidence-use rules
For every channel, mean and std are computed from the full valid series. Each
listed integer is approximately 100 * (raw_value - mean) / (std + epsilon),
where epsilon is stated below. Use normalized integers to assess deviations
relative to each channel's own historical scale.

When the question requires raw levels, absolute changes, or cross-channel
magnitude comparisons, interpret the integers together with that channel's
mean, std, and epsilon using the reconstruction relation stated below.

Do not compare normalized integer magnitudes across different channels as
if they were raw-scale amplitudes.

Each numeric value is immediately followed by one Step-Local hint
for the same channel and the same time step.

Step-Local hints provide fine-grained contextual evidence.
Each Channel-Local hint is a question-independent, channel-anchored summary
of the complete target window. It may reflect temporal context and information
from other channels. The Joint-Local hint is a question-conditioned summary
derived from all channels and all target-window steps.
Fixed hints contain task-level priors, not sample-specific facts.

Use all channels jointly when making the overall judgment.
Use Channel-Local hints for channel-level behavior and comparisons.
Use Window values and Step-Local hints for precise numeric, channel-aligned,
and time-local evidence.
Use the question-conditioned Joint-Local hint for the overall multivariate
interpretation relevant to the current question.
Use Fixed hints only as task-level guidance.
Do not treat the largest deviation as automatically being the root cause.
Do not infer a causal direction solely from anomaly magnitude."""


@dataclass
class TokenizedPrompts:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    prompt_lengths: List[int]
    step_positions: List[torch.Tensor]
    channel_positions: List[torch.Tensor]
    joint_positions: List[torch.Tensor]
    fixed_positions: List[torch.Tensor]
    padding_side: str


class MultiAxisPromptBuilder:
    """Render native text chat conversations and mask every non-answer token."""

    def __init__(
        self,
        tokenizer,
        fixed_tokens: int = 30,
        max_context_tokens: int = 32768,
        include_joint: bool = True,
        window_epsilon: float = 1e-5,
    ):
        self.tokenizer = tokenizer
        self.fixed_tokens = int(fixed_tokens)
        self.include_joint = bool(include_joint)
        self.window_epsilon = float(window_epsilon)
        if self.window_epsilon <= 0:
            raise ValueError("window_epsilon must be positive")
        self.max_context_tokens = int(max_context_tokens)
        self.token_ids: Dict[str, int] = {}
        for token in HINT_TOKENS:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{token} must tokenize to exactly one token, got {ids}")
            self.token_ids[token] = int(ids[0])
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before prompt construction")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("Formal prompt construction requires a native chat template")

    @contextmanager
    def _padding_side(self, side: str):
        if side not in {"left", "right"}:
            raise ValueError(f"Unsupported padding side: {side}")
        previous = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = side
        try:
            yield
        finally:
            self.tokenizer.padding_side = previous

    def _chat_template_kwargs(self) -> Dict[str, Any]:
        template = getattr(self.tokenizer, "chat_template", "")
        if isinstance(template, dict):
            template = "\n".join(map(str, template.values()))
        return {"enable_thinking": False} if "enable_thinking" in str(template) else {}

    def _render_chat(
        self, messages: List[Dict[str, str]], *, add_generation_prompt: bool
    ) -> str:
        return str(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **self._chat_template_kwargs(),
            )
        )

    @staticmethod
    def _messages(user_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": user_text},
        ]

    def build_text(
        self,
        question: str,
        start: int,
        end: int,
        window_values: Sequence[Sequence[int]],
        channel_means: Sequence[float],
        channel_stds: Sequence[float],
        question_group: str,
        channel_ids: Optional[Sequence[str]] = None,
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question is empty")
        length = int(end) - int(start)
        if length <= 0:
            raise ValueError(f"Invalid target interval [{start}, {end})")
        if not window_values:
            raise ValueError("At least one channel is required")
        if any(len(channel) != length for channel in window_values):
            raise ValueError("Every channel must contain exactly end-start target-window values")
        if question_group not in OUTPUT_CONTRACTS:
            raise ValueError(f"Unknown question group: {question_group!r}")
        identifiers = canonical_channel_ids(channel_ids, len(window_values))
        if len(channel_means) != len(window_values) or len(channel_stds) != len(
            window_values
        ):
            raise ValueError("Channel statistic count does not match the numeric Window")

        fixed = " ".join([FIXED_TOKEN] * self.fixed_tokens)
        overview_lines = [
            f"{identifier}: {CHANNEL_TOKEN}" for identifier in identifiers
        ]
        channel_lines = []
        for identifier, mean, std, values in zip(
            identifiers, channel_means, channel_stds, window_values
        ):
            evidence = " ".join(
                f"{int(value)} {STEP_TOKEN}" for value in values
            )
            channel_lines.append(
                f"{identifier} [mean={self._format_stat(mean)}, "
                f"std={self._format_stat(std)}]: {evidence}"
            )
        sections = [
            USER_PREAMBLE,
            f"### Question\n{question.strip()}",
            EVIDENCE_RULES,
            f"### Task-prior hints\n{fixed}",
            (
                "### All-channel overview\n"
                "Each channel appears once at overview resolution. Channel-Local "
                "hints are question-independent summaries of the complete "
                "target window.\n"
                + "\n".join(overview_lines)
            ),
            (
                "### Target window\n"
                f"Global interval: [{start}, {end})\n"
                f"Window length: {length}\n"
                f"Normalization epsilon: {self._format_stat(self.window_epsilon)}\n"
                "Each integer is rounded from 100 * (raw_value - mean) / "
                "(std + epsilon). Therefore raw_value is approximately mean + "
                "(std + epsilon) * integer / 100."
            ),
            (
                "### Channel-aligned Window and Step-Local evidence\n"
                f"Within every channel, value/hint pairs are ordered by global time "
                f"from t={start} through t={end - 1}; pair offset k corresponds exactly "
                f"to t={start}+k.\n"
                + "\n".join(channel_lines)
            ),
        ]
        if self.include_joint:
            sections.append(
                "### Question-conditioned full-window Joint-Local evidence\n"
                f"{JOINT_TOKEN}"
            )
        # Each contract already owns its single heading. Never append an answer prefill.
        sections.append(OUTPUT_CONTRACTS[question_group])
        return "\n\n".join(sections)

    @staticmethod
    def _format_stat(value: float) -> str:
        number = float(value)
        if not torch.isfinite(torch.tensor(number)):
            raise ValueError("Channel statistics must be finite")
        return format(number, ".10g")

    def forbidden_reasoning_token_sequences(self) -> List[List[int]]:
        sequences: List[List[int]] = []
        for marker in ("<think>", "</think>"):
            ids = list(self.tokenizer.encode(marker, add_special_tokens=False))
            if ids and ids not in sequences:
                sequences.append(ids)
        return sequences

    def tokenize(
        self,
        questions: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_means: Sequence[Sequence[float]],
        channel_stds: Sequence[Sequence[float]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        answers: Optional[Sequence[str]] = None,
    ) -> TokenizedPrompts:
        batch = len(questions)
        related = (
            intervals,
            window_values,
            channel_means,
            channel_stds,
            channel_ids,
            question_groups,
        )
        if any(len(values) != batch for values in related):
            raise ValueError("Prompt batch fields must have identical lengths")
        if answers is not None and len(answers) != batch:
            raise ValueError("Answer batch length must match questions")

        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for index, question in enumerate(questions):
            start, end = intervals[index]
            user_text = self.build_text(
                question,
                start,
                end,
                window_values[index],
                channel_means[index],
                channel_stds[index],
                question_groups[index],
                channel_ids[index],
            )
            messages = self._messages(user_text)
            prompt_text = self._render_chat(messages, add_generation_prompt=True)
            prompt_texts.append(prompt_text)
            if answers is None:
                full_texts.append(prompt_text)
            else:
                answer = answers[index]
                if not answer or not answer.strip():
                    raise ValueError("Empty teacher answer reached model forward")
                full_texts.append(
                    self._render_chat(
                        messages + [{"role": "assistant", "content": answer.strip()}],
                        add_generation_prompt=False,
                    )
                )

        encoded = [
            list(self.tokenizer.encode(text, add_special_tokens=False)) for text in full_texts
        ]
        prompt_ids = [
            list(self.tokenizer.encode(text, add_special_tokens=False)) for text in prompt_texts
        ]
        for index, (full, prompt) in enumerate(zip(encoded, prompt_ids)):
            if full[: len(prompt)] != prompt:
                raise RuntimeError(
                    "Native chat-template prompt is not an exact prefix of the supervised sequence "
                    f"for sample {index}"
                )
            if len(full) > self.max_context_tokens:
                raise ValueError(
                    f"Prompt length {len(full)} exceeds {self.max_context_tokens}; "
                    "Multi-AXIS never truncates prompts."
                )

        padding_side = "right" if answers is not None else "left"
        max_len = max(map(len, encoded))
        input_ids = torch.full(
            (batch, max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros(batch, max_len, dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long) if answers is not None else None
        step_positions: List[torch.Tensor] = []
        channel_positions: List[torch.Tensor] = []
        joint_positions: List[torch.Tensor] = []
        fixed_positions: List[torch.Tensor] = []
        prompt_lengths: List[int] = []
        with self._padding_side(padding_side):
            for index, ids in enumerate(encoded):
                valid_length = len(ids)
                row_start = 0 if padding_side == "right" else max_len - valid_length
                row_end = row_start + valid_length
                input_ids[index, row_start:row_end] = torch.tensor(ids, dtype=torch.long)
                attention_mask[index, row_start:row_end] = 1
                prompt_length = len(prompt_ids[index])
                prompt_lengths.append(prompt_length)
                if labels is not None:
                    labels[index, row_start + prompt_length:row_end] = input_ids[
                        index, row_start + prompt_length:row_end
                    ]
                valid = attention_mask[index].bool()
                row_ids = input_ids[index]
                step = torch.where((row_ids == self.token_ids[STEP_TOKEN]) & valid)[0]
                channel = torch.where(
                    (row_ids == self.token_ids[CHANNEL_TOKEN]) & valid
                )[0]
                joint = torch.where((row_ids == self.token_ids[JOINT_TOKEN]) & valid)[0]
                fixed = torch.where((row_ids == self.token_ids[FIXED_TOKEN]) & valid)[0]
                expected_steps = len(window_values[index]) * (
                    intervals[index][1] - intervals[index][0]
                )
                if step.numel() != expected_steps:
                    raise AssertionError(
                        f"STEP placeholder mismatch: {step.numel()} != {expected_steps}"
                    )
                expected_channels = len(window_values[index])
                if channel.numel() != expected_channels:
                    raise AssertionError(
                        f"CHANNEL placeholder mismatch: {channel.numel()} != "
                        f"{expected_channels}"
                    )
                expected_joint = 1 if self.include_joint else 0
                if joint.numel() != expected_joint:
                    raise AssertionError(
                        f"JOINT placeholder mismatch: {joint.numel()} != {expected_joint}"
                    )
                if fixed.numel() != self.fixed_tokens:
                    raise AssertionError(
                        f"FIXED placeholder mismatch: {fixed.numel()} != {self.fixed_tokens}"
                    )
                step_positions.append(step)
                channel_positions.append(channel)
                joint_positions.append(joint)
                fixed_positions.append(fixed)
        if labels is not None and int(labels.ne(-100).sum().item()) == 0:
            raise ValueError("No answer tokens remain after chat-template masking")
        return TokenizedPrompts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            prompt_lengths=prompt_lengths,
            step_positions=step_positions,
            channel_positions=channel_positions,
            joint_positions=joint_positions,
            fixed_positions=fixed_positions,
            padding_side=padding_side,
        )
