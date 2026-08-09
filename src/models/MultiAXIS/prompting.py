from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


STEP_TOKEN = "<STEP_HINT>"
JOINT_TOKEN = "<JOINT_HINT>"
FIXED_TOKEN = "<FIXED_HINT>"
HINT_TOKENS = (FIXED_TOKEN, STEP_TOKEN, JOINT_TOKEN)

SYSTEM_TEXT = """You are a multivariate time-series analyst.

The plotted series and the numeric Window are observable evidence.
Step-Local and Joint-Local are learned model hints, not ground-truth labels.
Fixed Hint provides task-level prior information.

Use the figure for global shape and cross-channel comparison.
Use the numeric Window for exact values and exact time positions.
If a visual impression conflicts with an exact numeric statement,
use the numeric Window for the numeric claim.

Do not mention image files, prompt tokens, hidden embeddings,
or the internal names of the hints in the answer.
Follow the Output Contract exactly."""

SYSTEM_TEXT_IMAGE_OFF = """You are a multivariate time-series analyst.

The numeric Window is observable evidence.
Step-Local and Joint-Local are learned model hints, not ground-truth labels.
Fixed Hint provides task-level prior information.

Use all channels jointly for the overall judgment.
Use the numeric Window for exact values and exact time positions.

Do not mention prompt tokens, hidden embeddings,
or the internal names of the hints in the answer.
Follow the Output Contract exactly."""

VISUAL_EVIDENCE = """### Visual Evidence
The attached figure shows the same per-channel normalized multivariate
series used by TimeRCD and by the numeric Window below.

Each row represents one channel. The row order is identical to the
channel order in the Window. The horizontal axis is the global time index.
The black curve is the surrounding sequence. The dark-blue segment
and pale-yellow background mark exactly the target interval [{start}, {end}),
that is, positions {start} through {last}.

The colors indicate location only; they do not indicate whether the
interval is anomalous. The figure contains no labels, anomaly scores,
root-cause annotations, or detector decisions.

Use the figure to compare global patterns and multiple channels jointly.
Use the Window below for exact values and exact locations."""

OUTPUT_CONTRACTS = {
    "MC": (
        "Begin with `Answer: X`, where X is exactly one option letter from the "
        "question. Then give a concise evidence-based explanation."
    ),
    "TF": (
        "Begin with exactly `Answer: True` or `Answer: False`. Then give a "
        "concise evidence-based justification."
    ),
    "OE": "Give a direct natural-language answer and a concise evidence-based explanation.",
}


@dataclass
class TokenizedPrompts:
    model_inputs: Dict[str, torch.Tensor]
    labels: Optional[torch.Tensor]
    prompt_lengths: List[int]
    step_positions: List[torch.Tensor]
    joint_positions: List[torch.Tensor]
    fixed_positions: List[torch.Tensor]
    visual_token_counts: List[int]
    original_image_sizes: List[Optional[Tuple[int, int]]]
    padding_side: str

    @property
    def input_ids(self) -> torch.Tensor:
        return self.model_inputs["input_ids"]

    @property
    def attention_mask(self) -> torch.Tensor:
        return self.model_inputs["attention_mask"]


class MultiAxisPromptBuilder:
    """Build native Qwen3-VL chat inputs without truncating numeric evidence."""

    def __init__(
        self,
        processor,
        fixed_tokens: int = 30,
        max_context_tokens: int = 131072,
        include_joint: bool = True,
        use_images: bool = True,
        min_pixels: int = 65_536,
        max_pixels: int = 4_194_304,
    ):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.fixed_tokens = int(fixed_tokens)
        self.include_joint = bool(include_joint)
        self.use_images = bool(use_images)
        self.max_context_tokens = int(max_context_tokens)
        self.min_pixels = int(min_pixels)
        self.max_pixels = int(max_pixels)
        if self.use_images and not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("Invalid Qwen3-VL runtime pixel budget")
        self.token_ids: Dict[str, int] = {}
        for token in HINT_TOKENS:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{token} must tokenize to exactly one token, got {ids}")
            self.token_ids[token] = int(ids[0])
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad token before prompt construction")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")

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

    def build_text(
        self,
        question: str,
        start: int,
        end: int,
        window_values: Sequence[Sequence[int]],
        channel_ids: Optional[Sequence[str]] = None,
        question_group: str = "OE",
        include_visual: Optional[bool] = None,
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
        identifiers = list(channel_ids or [f"ch_{index}" for index in range(len(window_values))])
        if len(identifiers) != len(window_values):
            raise ValueError("Channel id count does not match the numeric Window")
        if question_group not in OUTPUT_CONTRACTS:
            raise ValueError(f"Unknown question group: {question_group}")

        fixed = " ".join([FIXED_TOKEN] * self.fixed_tokens)
        channel_blocks = []
        for identifier, values in zip(identifiers, window_values):
            lines = [f"Channel {identifier}:"]
            lines.extend(
                f"t={start + relative}, value={int(value)} {STEP_TOKEN}"
                for relative, value in enumerate(values)
            )
            channel_blocks.append("\n".join(lines))
        sections = [f"### Question\n{question.strip()}"]
        visual = self.use_images if include_visual is None else bool(include_visual)
        if visual:
            sections.append(
                VISUAL_EVIDENCE.format(start=start, end=end, last=end - 1)
            )
        sections.extend(
            [
                f"### Fixed Hint\n{fixed}",
                (
                    "### Target Window\n"
                    f"Target interval: [{start}, {end})\n"
                    "Values are per-channel normalized values multiplied by 100 and rounded "
                    "to integers.\n\n"
                    + "\n\n".join(channel_blocks)
                ),
            ]
        )
        if self.include_joint:
            sections.append(f"### Joint-Local Hint\n{JOINT_TOKEN}")
        sections.append(
            f"### Output Contract\n{OUTPUT_CONTRACTS[question_group]}\n\nAnswer:"
        )
        return "\n\n".join(sections)

    def _messages(self, user_text: str, image: Any = None) -> List[Dict[str, Any]]:
        if image is None:
            # Text-only tokenizers such as DeepSeek/Qwen3 use chat templates that
            # concatenate message content directly and therefore require strings.
            user_content: Any = user_text
        else:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]
        return [
            {"role": "system", "content": SYSTEM_TEXT if image is not None else SYSTEM_TEXT_IMAGE_OFF},
            {"role": "user", "content": user_content},
        ]

    def _render_chat(self, messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
        renderer = getattr(self.processor, "apply_chat_template", None)
        if renderer is None:
            renderer = getattr(self.tokenizer, "apply_chat_template", None)
        if renderer is None:
            raise TypeError("Formal prompt construction requires a native chat template")
        return str(
            renderer(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        )

    def _open_images(
        self, image_paths: Sequence[Optional[str]]
    ) -> Tuple[List[Any], List[Optional[Tuple[int, int]]]]:
        if not self.use_images:
            if any(path is not None for path in image_paths):
                raise ValueError("Image-off control received image paths")
            return [], [None] * len(image_paths)
        if any(not path for path in image_paths):
            raise FileNotFoundError("Every image-enabled sample requires a pre-rendered image")
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for Qwen3-VL image inputs") from exc
        images: List[Any] = []
        sizes: List[Optional[Tuple[int, int]]] = []
        for raw_path in image_paths:
            path = Path(str(raw_path))
            if not path.is_file():
                raise FileNotFoundError(f"Pre-rendered VLM image does not exist: {path}")
            with Image.open(path) as source:
                image = source.convert("RGB").copy()
            images.append(image)
            sizes.append(tuple(map(int, image.size)))
        return images, sizes

    def _processor_call(
        self, texts: Sequence[str], images: Sequence[Any], padding: bool
    ) -> Dict[str, torch.Tensor]:
        if not callable(self.processor):
            raise TypeError("Formal VLM prompt construction requires a callable processor")
        kwargs: Dict[str, Any] = {
            "text": list(texts),
            "padding": padding,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = list(images)
            kwargs["return_mm_token_type_ids"] = True
            # Pass the budget on every call. Transformers 4.57 may otherwise
            # recover the AutoProcessor construction-time value from tokenizer
            # init kwargs instead of the current image-processor size.
            kwargs["min_pixels"] = self.min_pixels
            kwargs["max_pixels"] = self.max_pixels
        features = self.processor(**kwargs)
        native_keys = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        }
        tensors = {
            key: value
            for key, value in dict(features).items()
            if key in native_keys and torch.is_tensor(value)
        }
        if "input_ids" not in tensors or "attention_mask" not in tensors:
            raise RuntimeError("Processor omitted input_ids or attention_mask")
        if images:
            required = {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
            missing = sorted(required - set(tensors))
            if missing:
                raise RuntimeError(f"Qwen3-VL processor omitted required tensors: {missing}")
        return tensors

    def _prompt_ids(
        self,
        prompt_text: str,
        image_grid_thw: Optional[torch.Tensor],
        image: Any,
    ) -> List[int]:
        expanded = prompt_text
        if image_grid_thw is not None:
            replace = getattr(self.processor, "replace_image_token", None)
            if callable(replace):
                replacement = replace({"image_grid_thw": image_grid_thw}, 0)
                image_token = str(getattr(self.processor, "image_token", "<|image_pad|>"))
                if image_token not in prompt_text:
                    raise RuntimeError("Native chat template omitted the Qwen3-VL image token")
                expanded = prompt_text.replace(image_token, replacement, 1)
            else:
                single = self._processor_call([prompt_text], [image], padding=False)
                return single["input_ids"][0].tolist()
        return list(self.tokenizer.encode(expanded, add_special_tokens=False))

    def tokenize(
        self,
        questions: Sequence[str],
        intervals: Sequence[Tuple[int, int]],
        window_values: Sequence[Sequence[Sequence[int]]],
        channel_ids: Sequence[Sequence[str]],
        question_groups: Sequence[str],
        image_paths: Sequence[Optional[str]],
        answers: Optional[Sequence[str]] = None,
    ) -> TokenizedPrompts:
        batch = len(questions)
        related = (intervals, window_values, channel_ids, question_groups, image_paths)
        if any(len(values) != batch for values in related):
            raise ValueError("Prompt batch fields must have identical lengths")
        if answers is not None and len(answers) != batch:
            raise ValueError("Answer batch length must match questions")
        images, original_sizes = self._open_images(image_paths)

        prompt_texts: List[str] = []
        full_texts: List[str] = []
        for index, question in enumerate(questions):
            start, end = intervals[index]
            user_text = self.build_text(
                question,
                start,
                end,
                window_values[index],
                channel_ids[index],
                question_groups[index],
                include_visual=self.use_images,
            )
            image = images[index] if images else None
            messages = self._messages(user_text, image)
            prompt_text = self._render_chat(messages, add_generation_prompt=True)
            if image_paths[index] and str(image_paths[index]) in prompt_text:
                raise RuntimeError("A local image path leaked into the textual prompt")
            prompt_texts.append(prompt_text)
            if answers is None:
                full_texts.append(prompt_text)
            else:
                answer = answers[index]
                if not answer or not answer.strip():
                    raise ValueError("Empty teacher answer reached model forward")
                assistant_content: Any = (
                    [{"type": "text", "text": answer.strip()}]
                    if self.use_images
                    else answer.strip()
                )
                full_messages = messages + [
                    {"role": "assistant", "content": assistant_content}
                ]
                full_texts.append(
                    self._render_chat(full_messages, add_generation_prompt=False)
                )

        padding_side = "right" if answers is not None else "left"
        with self._padding_side(padding_side):
            model_inputs = self._processor_call(full_texts, images, padding=True)

        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        labels = torch.full_like(input_ids, -100) if answers is not None else None
        prompt_lengths: List[int] = []
        grids = model_inputs.get("image_grid_thw")
        if images and (grids is None or grids.shape[0] != batch):
            raise RuntimeError("Exactly one image grid is required per VLM sample")

        step_positions: List[torch.Tensor] = []
        joint_positions: List[torch.Tensor] = []
        fixed_positions: List[torch.Tensor] = []
        visual_token_counts: List[int] = []
        merge_size = int(getattr(getattr(self.processor, "image_processor", None), "merge_size", 2))
        for index in range(batch):
            valid_length = int(attention_mask[index].sum().item())
            if valid_length > self.max_context_tokens:
                raise ValueError(
                    f"Prompt length {valid_length} exceeds {self.max_context_tokens}; "
                    "Multi-AXIS never truncates prompts."
                )
            row_start = 0 if padding_side == "right" else input_ids.shape[1] - valid_length
            full_ids = input_ids[index, row_start : row_start + valid_length].tolist()
            grid = grids[index : index + 1] if grids is not None else None
            prompt_ids = self._prompt_ids(
                prompt_texts[index], grid, images[index] if images else None
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(
                    "Native chat-template prompt is not an exact prefix of the supervised sequence"
                )
            prompt_lengths.append(len(prompt_ids))
            if labels is not None:
                labels[index, row_start : row_start + valid_length] = input_ids[
                    index, row_start : row_start + valid_length
                ]
                labels[index, row_start : row_start + len(prompt_ids)] = -100

            row_ids = input_ids[index]
            valid = attention_mask[index].bool()
            step = torch.where((row_ids == self.token_ids[STEP_TOKEN]) & valid)[0]
            joint = torch.where((row_ids == self.token_ids[JOINT_TOKEN]) & valid)[0]
            fixed = torch.where((row_ids == self.token_ids[FIXED_TOKEN]) & valid)[0]
            expected_steps = len(window_values[index]) * (intervals[index][1] - intervals[index][0])
            if step.numel() != expected_steps:
                raise AssertionError(f"STEP placeholder mismatch: {step.numel()} != {expected_steps}")
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
            visual_token_counts.append(
                int(grid.prod().item() // (merge_size**2)) if grid is not None else 0
            )

        if labels is not None and int(labels.ne(-100).sum().item()) == 0:
            raise ValueError("No answer tokens remain after chat-template masking")
        return TokenizedPrompts(
            model_inputs=model_inputs,
            labels=labels,
            prompt_lengths=prompt_lengths,
            step_positions=step_positions,
            joint_positions=joint_positions,
            fixed_positions=fixed_positions,
            visual_token_counts=visual_token_counts,
            original_image_sizes=original_sizes,
            padding_side=padding_side,
        )
