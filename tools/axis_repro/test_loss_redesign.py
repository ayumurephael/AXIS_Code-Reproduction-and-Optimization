from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from .loss_redesign import (
    CounterfactualAXISDataset,
    build_answer_head_mask_from_offsets,
    compute_source_likelihood_ratio_loss,
    counterfactual_collate_fn,
    evidence_mode_spec,
    extract_answer_head,
    format_axis_question_prompt,
    memory_safe_token_nll_sums,
    retrieve_exact,
)


class AnswerHeadTests(unittest.TestCase):
    def test_true_false_uses_only_leading_label(self):
        self.assertEqual(
            extract_answer_head("False. The window is anomalous.", "true_false"),
            "False",
        )

    def test_first_sentence_does_not_split_decimal(self):
        answer = "A) The value rises from 1.25 to 2.50. This supports the choice."
        self.assertEqual(
            extract_answer_head(answer, "multiple_choice"),
            "A) The value rises from 1.25 to 2.50.",
        )

    def test_open_ended_requires_decision_in_first_sentence(self):
        answer = "The values rise smoothly. There is no anomaly in the window."
        self.assertIsNone(extract_answer_head(answer, "open_ended"))

    def test_open_ended_keeps_complete_decision_sentence(self):
        answer = "There is no anomaly near value 1.25. The pattern is smooth."
        self.assertEqual(
            extract_answer_head(answer, "open_ended"),
            "There is no anomaly near value 1.25.",
        )


class PromptTests(unittest.TestCase):
    def test_local_only_removes_window_without_inventing_values(self):
        prompt = format_axis_question_prompt(
            question="Is it anomalous?",
            start_index=10,
            end_index=12,
            window_values=torch.tensor([9.0, 8.0]),
            num_fixed_tokens=2,
            include_local=True,
            include_window=False,
        )
        self.assertIn("Steps 10 to 12", prompt)
        self.assertIn("(not provided)", prompt)
        self.assertEqual(prompt.count("<|local_hint|>"), 2)

    def test_window_only_uses_override_values_and_removes_local_tokens(self):
        prompt = format_axis_question_prompt(
            question="Is it anomalous?",
            start_index=10,
            end_index=12,
            window_values=torch.tensor([1.11, -2.22]),
            num_fixed_tokens=2,
            include_local=False,
            include_window=True,
        )
        self.assertIn("111, -222", prompt)
        self.assertNotIn("<|local_hint|>", prompt)
        self.assertIn("Steps 10 to 12", prompt)


class LossTests(unittest.TestCase):
    def test_slr_matches_documented_formula_and_skips_invalid_rows(self):
        answer_loss = torch.tensor(2.0)
        e_pos = torch.tensor([4.0, 6.0, 100.0])
        e_neg = torch.tensor([5.0, 5.0, 0.0])
        counts = torch.tensor([2, 2, 1])
        valid = torch.tensor([True, True, False])
        total, stats = compute_source_likelihood_ratio_loss(
            answer_loss,
            e_pos,
            e_neg,
            counts,
            valid,
            beta=0.5,
            margin=math.log(2.0),
        )
        expected_row0 = 4.0 / 2.0
        expected_row1 = 6.0 / 2.0 + (math.log(2.0) + 6.0 - 5.0) / 2.0
        expected = 2.0 + 0.5 * ((expected_row0 + expected_row1) / 2.0)
        self.assertAlmostEqual(float(total), expected, places=6)
        self.assertEqual(stats["valid_rows"], 2)
        self.assertEqual(stats["active_hinges"], 1)


class RetrievalTests(unittest.TestCase):
    def test_hard_constraints_reject_contaminated_donor(self):
        anchor = torch.tensor([[0.0, 0.0]])
        donor_normal = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
        donor_abnormal = torch.tensor([[4.0, 0.0], [4.1, 0.1]])
        control_gap = torch.tensor([2.0, 0.0])
        best, found = retrieve_exact(
            anchor,
            donor_normal,
            donor_abnormal,
            control_gap,
            gamma=0.0,
            tau=0.25,
        )
        self.assertTrue(bool(found[0]))
        self.assertEqual(int(best[0]), 1)

    def test_selection_minimizes_distance_not_pollution_ratio(self):
        anchor = torch.tensor([[0.0, 0.0]])
        donor_normal = torch.tensor([[0.10, 0.10], [0.20, 0.20]])
        donor_abnormal = torch.tensor([[1.10, 0.10], [10.20, 0.20]])
        control_gap = torch.zeros(2)
        best, found = retrieve_exact(
            anchor,
            donor_normal,
            donor_abnormal,
            control_gap,
            gamma=0.0,
            tau=0.25,
        )
        self.assertTrue(bool(found[0]))
        self.assertEqual(int(best[0]), 0)


class CounterfactualDatasetTests(unittest.TestCase):
    def _write_series(self, root: Path, name: str, time_series, normal_series, windows):
        payload = {"sample_id": int(name[7:13]), "original_data": {"time_series": time_series, "normal_series": normal_series}, "windows": windows}
        (root / "series" / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_collate_builds_self_normal_and_donor_negatives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "series").mkdir()
            windows0 = [
                {"window_range": {"start": 0, "end": 2}, "question": "q0", "answer": "True. anomaly", "question_type": "true_false", "has_anomaly": True},
                {"window_range": {"start": 2, "end": 4}, "question": "q1", "answer": "There is no anomaly.", "question_type": "open_ended", "has_anomaly": False},
            ]
            windows1 = [{"window_range": {"start": 1, "end": 3}, "question": "donor", "answer": "True.", "question_type": "true_false", "has_anomaly": True}]
            self._write_series(root, "series_000000.json", [0, 1, 2, 3], [0, 0, 0, 0], windows0)
            self._write_series(root, "series_000001.json", [9, 8, 7, 6], [0, 0, 0, 0], windows1)
            index = {
                "seed": 0,
                "train_ratio": 1.0,
                "records": {
                    "series_000000.json:0": {"window": {"valid": True, "kind": "self_normal"}, "local": {"valid": True, "kind": "self_normal"}},
                    "series_000000.json:1": {
                        "window": {"valid": True, "kind": "donor", "series_file": "series_000001.json", "window_index": 0},
                        "local": {"valid": True, "kind": "donor", "series_file": "series_000001.json", "window_index": 0},
                    },
                },
            }
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            dataset = CounterfactualAXISDataset(str(root), str(index_path), split="train", train_ratio=1.0, seed=0)
            batch = counterfactual_collate_fn([dataset[0]])
            self.assertTrue(torch.equal(batch["negative_window_values"][0], torch.tensor([0.0, 0.0])))
            self.assertTrue(torch.equal(batch["negative_window_values"][1], torch.tensor([8.0, 7.0])))
            self.assertTrue(torch.equal(batch["negative_local_sequences"][0, :4], torch.zeros(4)))
            self.assertTrue(torch.equal(batch["negative_local_sequences"][1, :4], torch.tensor([9.0, 8.0, 7.0, 6.0], dtype=torch.bfloat16)))
            self.assertEqual(batch["negative_local_start_indices"], [0, 1])
            self.assertEqual(batch["negative_local_end_indices"], [2, 3])
            self.assertTrue(bool(batch["local_source_valid"].all()))
            self.assertTrue(bool(batch["window_source_valid"].all()))


class TokenMaskAndModeTests(unittest.TestCase):
    def test_answer_head_mask_uses_character_offsets(self):
        answers = ["False. explanation", "Values rise. There is no anomaly."]
        question_types = ["true_false", "open_ended"]
        offsets = torch.tensor([
            [[0, 0], [0, 7], [8, 13], [13, 14], [15, 26]],
            [[0, 0], [0, 7], [8, 14], [15, 21], [21, 22]],
        ])
        attention = torch.ones(offsets.shape[:2], dtype=torch.long)
        mask, heads = build_answer_head_mask_from_offsets(
            answers, question_types, offsets, attention
        )
        self.assertEqual(heads, ["False", None])
        self.assertTrue(torch.equal(mask[0], torch.tensor([False, False, True, False, False])))
        self.assertFalse(bool(mask[1].any()))

    def test_mode_mapping_changes_exactly_one_source(self):
        self.assertEqual(evidence_mode_spec("local_only"), (True, False, "local"))
        self.assertEqual(evidence_mode_spec("window_only"), (False, True, "window"))
        self.assertEqual(evidence_mode_spec("full_local"), (True, True, "local"))
        self.assertEqual(evidence_mode_spec("full_window"), (True, True, "window"))


class MemorySafeNLLTests(unittest.TestCase):
    def test_chunked_token_nll_matches_direct_cross_entropy(self):
        lm_head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            lm_head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
        hidden = torch.tensor([
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0], [0.5, 0.5]],
        ])
        targets = torch.tensor([[0, 1, -100], [2, 1, 0]])
        head_mask = torch.tensor([[True, False, False], [False, True, True]])
        full_sums, full_counts, head_sums, head_counts = memory_safe_token_nll_sums(
            lm_head, hidden, targets, head_mask, chunk_size=1, checkpoint_chunks=False
        )
        direct = torch.nn.functional.cross_entropy(
            lm_head(hidden).reshape(-1, 3), targets.reshape(-1),
            ignore_index=-100, reduction="none"
        ).reshape(2, 3)
        valid = targets.ne(-100)
        self.assertTrue(torch.allclose(full_sums, (direct * valid).sum(1)))
        self.assertTrue(torch.equal(full_counts, valid.sum(1)))
        self.assertTrue(torch.allclose(head_sums, (direct * (valid & head_mask)).sum(1)))
        self.assertTrue(torch.equal(head_counts, (valid & head_mask).sum(1)))


if __name__ == "__main__":
    unittest.main()
