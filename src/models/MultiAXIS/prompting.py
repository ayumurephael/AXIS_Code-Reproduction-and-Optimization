from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .response_contracts import OUTPUT_CONTRACTS


STEP_TOKEN = "<STEP_HINT>"
JOINT_TOKEN = "<JOINT_HINT>"
FIXED_TOKEN = "<FIXED_HINT>"
HINT_TOKENS = (STEP_TOKEN, JOINT_TOKEN, FIXED_TOKEN)

SYSTEM_TEXT = """You are an expert in multivariate time-series analysis. Answer the exact
question using the numeric and learned contextual evidence supplied in the
user message.

The numeric Window contains observed values from the target interval. Treat
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
The numeric Window contains observed values from the target interval.
Each numeric value is immediately followed by one Step-Local hint
for the same channel and the same time step.

Step-Local hints provide fine-grained contextual evidence.
The Joint-Local hint summarizes evidence across all channels and
all steps in the target window.
Fixed hints contain task-level priors, not sample-specific facts.

Use all channels jointly when making the overall judgment.
Use the per-channel Step-Local evidence when identifying channels
or time positions.
Do not treat the largest deviation as automatically being the root cause.
Do not infer a causal direction solely from anomaly magnitude."""


@dataclass
class TokenizedPrompts:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    prompt_lengths: List[int]
    step_positions: List[torch.Tensor]
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
    ):
        self.tokenizer = tokenizer
        self.fixed_tokens = int(fixed_tokens)
        self.include_joint = bool(include_joint)
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
        identifiers = list(
            channel_ids or [f"ch_{index}" for index in range(len(window_values))]
        )
        if len(identifiers) != len(window_values):
            raise ValueError("Channel id count does not match the numeric Window")

        fixed = " ".join([FIXED_TOKEN] * self.fixed_tokens)
        channel_blocks = []
        for identifier, values in zip(identifiers, window_values):
            lines = [f"Channel {identifier}:"]
            lines.extend(
                f"t={start + relative}, value={int(value)} {STEP_TOKEN}"
                for relative, value in enumerate(values)
            )
            channel_blocks.append("\n".join(lines))
        sections = [
            USER_PREAMBLE,
            f"### Question\n{question.strip()}",
            EVIDENCE_RULES,
            f"### Task-prior hints\n{fixed}",
            (
                "### Target window\n"
                f"Global interval: [{start}, {end})\n"
                f"Window length: {length}\n"
                "Each value is normalized within its own channel, multiplied by 100, "
                "and rounded to an integer."
            ),
            "### Channel-aligned Window and Step-Local evidence\n"
            + "\n\n".join(channel_blocks),
        ]
        if self.include_joint:
            sections.append(f"### Joint-Local multivariate evidence\n{JOINT_TOKEN}")
        # Each contract already owns its single heading. Never append an answer prefill.
        sections.append(OUTPUT_CONTRACTS[question_group])
        return "\n\n".join(sections)

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
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        answers: Optional[Sequence[str]] = None,
    ) -> TokenizedPrompts:
        batch = len(questions)
        related = (intervals, window_values, channel_ids, question_groups)
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
                joint = torch.where((row_ids == self.token_ids[JOINT_TOKEN]) & valid)[0]
                fixed = torch.where((row_ids == self.token_ids[FIXED_TOKEN]) & valid)[0]
                expected_steps = len(window_values[index]) * (
                    intervals[index][1] - intervals[index][0]
                )
                if step.numel() != expected_steps:
                    raise AssertionError(
                        f"STEP placeholder mismatch: {step.numel()} != {expected_steps}"
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
            joint_positions=joint_positions,
            fixed_positions=fixed_positions,
            padding_side=padding_side,
        )
