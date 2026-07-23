from __future__ import annotations

import unittest

import torch

from .loss_e2e import (
    SEGMENT_CONCLUSION,
    SEGMENT_EXPLANATION,
    build_full_segment_ids,
    objective_from_hidden,
    segment_answer,
)
from .loss_e2e_runtime import _answer_encoding


class TestDeterministicSegmentation(unittest.TestCase):
    def test_multiple_choice_marker_excludes_marker(self):
        answer = "B. A sustained upward shift.\nExplanation: The level remains high."
        result = segment_answer(answer, "Multiple Choice")
        self.assertEqual(
            answer[result.conclusion.start:result.conclusion.end],
            "B. A sustained upward shift.",
        )
        self.assertEqual(
            answer[result.explanation.start:result.explanation.end],
            "The level remains high.",
        )
        self.assertEqual(result.rule, "mc_marker")

    def test_tf_conclusion_includes_label_and_first_statement(self):
        answer = (
            "False. The point is consistent with its neighbors. "
            "There is no isolated deviation."
        )
        result = segment_answer(answer, "True/False")
        self.assertEqual(
            answer[result.conclusion.start:result.conclusion.end],
            "False. The point is consistent with its neighbors.",
        )
        self.assertEqual(
            answer[result.explanation.start:result.explanation.end],
            "There is no isolated deviation.",
        )
        self.assertEqual(result.rule, "tf_label_plus_first_statement")

    def test_open_ended_short_first_sentence_merges_second(self):
        answer = "No anomaly. Values stay within the local range. The trend is stable."
        result = segment_answer(answer, "Open-Ended")
        self.assertEqual(
            answer[result.conclusion.start:result.conclusion.end],
            "No anomaly. Values stay within the local range.",
        )
        self.assertEqual(
            answer[result.explanation.start:result.explanation.end],
            "The trend is stable.",
        )
        self.assertEqual(result.rule, "oe_short_merge_second")

    def test_sentence_split_protects_common_abbreviations(self):
        answer = (
            "The level changes, e.g. it remains elevated for five steps. "
            "This supports the anomaly decision."
        )
        result = segment_answer(answer, "Open-Ended")
        self.assertEqual(
            answer[result.conclusion.start:result.conclusion.end],
            "The level changes, e.g. it remains elevated for five steps.",
        )
        self.assertEqual(
            answer[result.explanation.start:result.explanation.end],
            "This supports the anomaly decision.",
        )

    def test_sentence_split_keeps_abbreviation_at_real_sentence_end(self):
        answer = "The behavior differs from the expected baseline, etc. It is anomalous."
        result = segment_answer(answer, "Open-Ended")
        self.assertEqual(result.rule, "oe_first_sentence")
        self.assertEqual(answer[result.conclusion.start:result.conclusion.end], "The behavior differs from the expected baseline, etc.")

    def test_leading_mc_label_is_not_treated_as_a_sentence(self):
        answer = "A. The values remain stable."
        result = segment_answer(answer, "Multiple Choice")
        self.assertEqual(result.rule, "mc_all")
        self.assertEqual(
            answer[result.conclusion.start:result.conclusion.end],
            answer,
        )

    def test_offsets_exclude_answer_prefix_and_special_tokens(self):
        answer = "False. First fact. More detail."
        answer_ids = torch.tensor([[90, 10, 11, 12, 13, 14, 15, 16, 91]])
        offsets = torch.tensor(
            [[
                [0, 0],   # BOS
                [0, 6],   # Answer
                [6, 7],   # :
                [7, 13],  # leading space + False
                [13, 14], # .
                [14, 20], # leading space + First
                [20, 26], # leading space + fact.
                [26, 39], # leading space + More detail.
                [0, 0],   # EOS
            ]]
        )
        full = torch.cat(
            [torch.tensor([[1, 2]]), answer_ids, torch.tensor([[3]])],
            dim=1,
        )
        segments, _ = build_full_segment_ids(
            full_input_ids=full,
            answer_input_ids=answer_ids,
            answer_offsets=offsets,
            answer_block_start=2,
            answers=[answer],
            question_types=["true_false"],
            terminal_eos_index=full.size(1) - 1,
            terminal_eos_token_id=3,
        )
        answer_segments = segments[0, 2:2 + answer_ids.size(1)].tolist()
        self.assertEqual(answer_segments[:3], [0, 0, 0])
        self.assertEqual(
            answer_segments[3:7],
            [SEGMENT_CONCLUSION] * 4,
        )
        self.assertEqual(answer_segments[7], SEGMENT_EXPLANATION)
        self.assertEqual(answer_segments[8], 0)
        self.assertEqual(segments[0, -1], SEGMENT_EXPLANATION)

    def test_terminal_eos_uses_only_available_conclusion_segment(self):
        answer = "A) Stable"
        answer_ids = torch.tensor([[90, 10, 11, 12, 91]])
        offsets = torch.tensor(
            [[[0, 0], [0, 6], [6, 7], [7, 17], [0, 0]]]
        )
        full = torch.cat(
            [torch.tensor([[1, 2]]), answer_ids, torch.tensor([[3]])],
            dim=1,
        )
        segments, _ = build_full_segment_ids(
            full_input_ids=full,
            answer_input_ids=answer_ids,
            answer_offsets=offsets,
            answer_block_start=2,
            answers=[answer],
            question_types=["multiple_choice"],
            terminal_eos_index=full.size(1) - 1,
            terminal_eos_token_id=3,
        )
        self.assertEqual(segments[0, -1], SEGMENT_CONCLUSION)

    def test_answer_encoding_explicitly_disables_truncation(self):
        class RecordingTokenizer:
            is_fast = True

            def __init__(self):
                self.kwargs = None

            def __call__(self, _values, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": torch.tensor([[1]])}

        class Axis:
            tokenizer = RecordingTokenizer()

        axis = Axis()
        _answer_encoding(axis, ["short answer"])
        self.assertIs(axis.tokenizer.kwargs["truncation"], False)


class TestObjective(unittest.TestCase):
    def test_row_balanced_segmented_loss(self):
        hidden = torch.tensor(
            [
                [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [0, 0, 0]],
                [[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0, 0, 0]],
            ],
            requires_grad=True,
        )
        labels = torch.tensor(
            [
                [-100, 0, 1, 2],
                [-100, 1, 0, -100],
            ]
        )
        segments = torch.tensor(
            [
                [0, 1, 2, 2],
                [0, 1, 1, 0],
            ]
        )
        result = objective_from_hidden(
            hidden=hidden,
            labels=labels,
            lm_head=torch.nn.Identity(),
            mode="treatment",
            valid_rows=torch.tensor([True, True]),
            segment_ids=segments,
            question_types=["multiple_choice", "true_false"],
            alpha=0.40,
            chunk_size=2,
        )
        token_loss = torch.nn.functional.cross_entropy(
            hidden[:, :-1].reshape(-1, 3),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(2, 3)
        expected_row0 = 0.4 * token_loss[0, 0] + 0.6 * token_loss[0, 1:3].mean()
        expected_row1 = token_loss[1, :2].mean()
        self.assertTrue(
            torch.allclose(
                result["objective_sum"],
                expected_row0 + expected_row1,
            )
        )
        self.assertEqual(result["objective_count"].item(), 2)
        self.assertEqual(result["both_segment_row_count"].item(), 1)
        self.assertEqual(result["conclusion_only_row_count"].item(), 1)
        result["objective_sum"].backward()
        self.assertIsNotNone(hidden.grad)


if __name__ == "__main__":
    unittest.main()
