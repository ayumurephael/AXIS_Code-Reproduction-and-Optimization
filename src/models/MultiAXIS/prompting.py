from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


STEP_TOKEN = "<STEP_HINT>"
JOINT_TOKEN = "<JOINT_HINT>"
FIXED_TOKEN = "<FIXED_HINT>"
HINT_TOKENS = (STEP_TOKEN, JOINT_TOKEN, FIXED_TOKEN)

SYSTEM_TEXT = """<System>
You are an expert in multivariate time-series analysis.

The Window contains observed numeric values.
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
Do not infer a causal direction solely from anomaly magnitude.
"""


@dataclass
class TokenizedPrompts:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    prompt_lengths: List[int]
    step_positions: List[torch.Tensor]
    joint_positions: List[torch.Tensor]
    fixed_positions: List[torch.Tensor]


class MultiAxisPromptBuilder:
    def __init__(self, tokenizer, fixed_tokens: int = 30, max_context_tokens: int = 32768, include_joint: bool = True):
        self.tokenizer = tokenizer
        self.fixed_tokens = fixed_tokens
        self.include_joint = include_joint
        self.max_context_tokens = max_context_tokens
        self.token_ids = {}
        for token in HINT_TOKENS:
            ids = tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{token} must tokenize to exactly one token, got {ids}")
            self.token_ids[token] = ids[0]
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before prompt construction")
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")

    def build_text(
        self,
        question: str,
        start: int,
        end: int,
        window_values: Sequence[Sequence[int]],
    ) -> str:
        if not question or not question.strip():
            raise ValueError("Question is empty")
        length = end - start
        if length <= 0:
            raise ValueError(f"Invalid target interval [{start}, {end})")
        if not window_values:
            raise ValueError("At least one channel is required")
        if any(len(channel) != length for channel in window_values):
            raise ValueError("Every channel must contain exactly end-start target-window values")
        fixed = " ".join([FIXED_TOKEN] * self.fixed_tokens)
        channel_lines = []
        for channel_index, values in enumerate(window_values):
            evidence = " ".join(f"{int(value)} {STEP_TOKEN}" for value in values)
            channel_lines.append(f"ch_{channel_index}: {evidence}")
        channels = "\n".join(channel_lines)
        joint_section = f"### Joint-Local multivariate evidence\n{JOINT_TOKEN}\n\n" if self.include_joint else ""
        return (
            f"{SYSTEM_TEXT}\n<User>\n### Question\n{question.strip()}\n\n"
            f"### Task-prior hints\n{fixed}\n\n"
            f"### Target window\nGlobal interval: [{start}, {end})\n"
            f"Window length: {length}\n"
            "Each value is normalized within its own channel,\n"
            "multiplied by 100, and rounded to an integer.\n\n"
            "### Channel-aligned Window and Step-Local evidence\n"
            f"{channels}\n\n"
            f"{joint_section}### Answer\n"
        )

    def tokenize(
        self,
        questions: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        window_values: Sequence[Sequence[Sequence[int]]],
        answers: Optional[Sequence[str]] = None,
    ) -> TokenizedPrompts:
        batch = len(questions)
        if len(intervals) != batch or len(window_values) != batch:
            raise ValueError("Question, interval, and window-value batch lengths must match")
        if answers is not None and len(answers) != batch:
            raise ValueError("Answer batch length must match questions")

        encoded: List[List[int]] = []
        encoded_labels: List[List[int]] = []
        prompt_lengths: List[int] = []
        expected_steps: List[int] = []
        for i, question in enumerate(questions):
            start, end = intervals[i]
            prompt = self.build_text(question, start, end, window_values[i])
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            full_ids = list(prompt_ids)
            labels = [-100] * len(prompt_ids)
            if answers is not None:
                answer = answers[i]
                if not answer or not answer.strip():
                    raise ValueError("Empty teacher answer reached model forward")
                answer_ids = self.tokenizer.encode(answer.strip(), add_special_tokens=False)
                answer_ids.append(self.tokenizer.eos_token_id)
                full_ids.extend(answer_ids)
                labels.extend(answer_ids)
            if len(full_ids) > self.max_context_tokens:
                raise ValueError(
                    f"Prompt length {len(full_ids)} exceeds {self.max_context_tokens}; "
                    "Multi-AXIS never truncates prompts."
                )
            encoded.append(full_ids)
            encoded_labels.append(labels)
            prompt_lengths.append(len(prompt_ids))
            expected_steps.append(len(window_values[i]) * (end - start))

        max_len = max(map(len, encoded))
        input_ids = torch.full(
            (batch, max_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros(batch, max_len, dtype=torch.long)
        label_tensor = torch.full((batch, max_len), -100, dtype=torch.long) if answers is not None else None
        step_positions: List[torch.Tensor] = []
        joint_positions: List[torch.Tensor] = []
        fixed_positions: List[torch.Tensor] = []
        for i, ids in enumerate(encoded):
            length = len(ids)
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1
            if label_tensor is not None:
                label_tensor[i, :length] = torch.tensor(encoded_labels[i], dtype=torch.long)
            step = torch.nonzero(
                input_ids[i, :prompt_lengths[i]] == self.token_ids[STEP_TOKEN], as_tuple=False
            ).flatten()
            joint = torch.nonzero(
                input_ids[i, :prompt_lengths[i]] == self.token_ids[JOINT_TOKEN], as_tuple=False
            ).flatten()
            fixed = torch.nonzero(
                input_ids[i, :prompt_lengths[i]] == self.token_ids[FIXED_TOKEN], as_tuple=False
            ).flatten()
            if step.numel() != expected_steps[i]:
                raise AssertionError(f"STEP placeholder mismatch: {step.numel()} != {expected_steps[i]}")
            expected_joint = 1 if self.include_joint else 0
            if joint.numel() != expected_joint:
                raise AssertionError(f"JOINT placeholder mismatch: {joint.numel()} != {expected_joint}")
            if fixed.numel() != self.fixed_tokens:
                raise AssertionError(
                    f"FIXED placeholder mismatch: {fixed.numel()} != {self.fixed_tokens}"
                )
            step_positions.append(step)
            joint_positions.append(joint)
            fixed_positions.append(fixed)
        return TokenizedPrompts(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_tensor,
            prompt_lengths=prompt_lengths,
            step_positions=step_positions,
            joint_positions=joint_positions,
            fixed_positions=fixed_positions,
        )
