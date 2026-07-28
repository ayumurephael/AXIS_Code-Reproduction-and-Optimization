from __future__ import annotations

import unittest

import torch

from .loss_e2e import (
    SEGMENT_CONCLUSION,
    SEGMENT_EXPLANATION,
    build_full_segment_ids,
)
from .loss_e2e_40epoch_runtime import _offsets_for_existing_segment_mapper
from .prompt_boundary import (
    SIMPLIFIED_FINAL_ANSWER_V1,
    answer_continuations,
    build_simplified_prompt,
    install_simplified_prompt_runtime,
)


class CharacterTokenizer:
    is_fast = True
    eos_token_id = 0
    pad_token_id = 0
    model_max_length = 10_000

    def __call__(
        self,
        values,
        *,
        return_tensors=None,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        del truncation, add_special_tokens
        scalar = isinstance(values, str)
        rows = [values] if scalar else list(values)
        encoded = [[ord(char) + 1 for char in value] for value in rows]
        offsets = [
            [(index, index + 1) for index in range(len(value))]
            for value in rows
        ]
        masks = [[1] * len(row) for row in encoded]
        if padding:
            width = max(len(row) for row in encoded)
            for row, mask, row_offsets in zip(encoded, masks, offsets):
                missing = width - len(row)
                row.extend([self.pad_token_id] * missing)
                mask.extend([0] * missing)
                row_offsets.extend([(0, 0)] * missing)
        result = {
            "input_ids": encoded[0] if scalar else encoded,
            "attention_mask": masks[0] if scalar else masks,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets[0] if scalar else offsets
        if return_tensors == "pt":
            result = {
                key: torch.tensor(value, dtype=torch.long)
                for key, value in result.items()
            }
        return result


class DummyAxis:
    def __init__(self):
        self.tokenizer = CharacterTokenizer()
        self.num_fixed_tokens = 2

    def generate_input_ids_and_labels(self, *args, **kwargs):
        raise AssertionError("original prompt path must not be called")


class PromptBoundaryTests(unittest.TestCase):
    def test_exact_approved_prompt_suffix_and_headers(self):
        prompt = build_simplified_prompt(
            question="Is the final point anomalous?",
            time_series=torch.tensor([0.1, 0.2, 0.3]),
            start_index=0,
            end_index=2,
            num_local_hint_tokens=2,
            num_fixed_hint_tokens=2,
        )
        self.assertTrue(
            prompt.startswith(
                "You are an expert time-series anomaly analyst. "
                "Analyze the provided data and produce one precise, "
                "evidence-grounded answer to the question."
            )
        )
        self.assertIn(
            "### Learned Task Guidance/Shared Task-Control Tokens",
            prompt,
        )
        self.assertTrue(prompt.endswith("### Final Answer\nAnswer:"))
        self.assertNotIn("Overall Summary Hints", prompt)

    def test_training_has_shared_prefix_no_middle_eos_and_terminal_eos(self):
        axis = DummyAxis()
        install_simplified_prompt_runtime(axis)
        full, attention, labels, prefix_width = (
            axis.generate_input_ids_and_labels(
                ["Question one?", "Q2?"],
                ["True. Stable.", "B. Shift."],
                torch.tensor(
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    dtype=torch.float32,
                ),
                [0, 0],
                [2, 2],
            )
        )
        layout = axis._loss_e2e_prompt_layout
        self.assertEqual(layout["protocol"], SIMPLIFIED_FINAL_ANSWER_V1)
        self.assertEqual(layout["answer_block_start"], prefix_width)
        self.assertTrue(
            all(prompt.endswith("### Final Answer\nAnswer:") for prompt in layout["prefix_rows"])
        )
        self.assertTrue(torch.all(labels[:, :prefix_width] == -100))
        # The first target token is the natural separator space, not EOS.
        self.assertEqual(
            int(full[0, prefix_width]),
            ord(" ") + 1,
        )
        self.assertEqual(int(labels[0, prefix_width]), ord(" ") + 1)
        self.assertEqual(int(full[0, -1]), axis.tokenizer.eos_token_id)
        self.assertEqual(int(labels[0, -1]), axis.tokenizer.eos_token_id)
        self.assertTrue(torch.all(attention[:, -1] == 1))
        self.assertEqual(
            full.shape[1],
            prefix_width + layout["answer_width"] + 1,
        )
        # Padding EOS IDs inside the answer rectangle remain unsupervised.
        answer_mask = labels[:, prefix_width:-1] != -100
        self.assertEqual(
            int(answer_mask[0].sum()),
            len(answer_continuations(["True. Stable."])[0]),
        )
        self.assertEqual(
            int(answer_mask[1].sum()),
            len(answer_continuations(["B. Shift."])[0]),
        )

    def test_terminal_eos_receives_last_treatment_segment(self):
        answer = "False. First fact. More detail."
        tokenizer = CharacterTokenizer()
        continuation = answer_continuations([answer])[0]
        encoding = tokenizer(
            [continuation],
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        prefix = torch.tensor([[91, 92]])
        terminal_eos = torch.tensor([[tokenizer.eos_token_id]])
        full = torch.cat(
            [prefix, encoding["input_ids"], terminal_eos],
            dim=1,
        )
        segments, _ = build_full_segment_ids(
            full_input_ids=full,
            answer_input_ids=encoding["input_ids"],
            answer_offsets=_offsets_for_existing_segment_mapper(
                encoding["offset_mapping"]
            ),
            answer_block_start=prefix.size(1),
            answers=[answer],
            question_types=["true_false"],
            terminal_eos_index=full.size(1) - 1,
            terminal_eos_token_id=tokenizer.eos_token_id,
        )
        self.assertEqual(
            int(segments[0, prefix.size(1) + 1]),
            SEGMENT_CONCLUSION,
        )
        self.assertEqual(int(segments[0, -1]), SEGMENT_EXPLANATION)


if __name__ == "__main__":
    unittest.main()
