from __future__ import annotations

import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import torch

from .loss_redesign import (
    COUNTERFACTUAL_INDEX_VERSION,
    CounterfactualAXISDataset,
    build_state_question,
    compute_state_loss,
    counterfactual_collate_fn,
    format_axis_question_prompt,
    memory_safe_token_nll_sums,
    retrieve_consistent_donors,
    scheduled_beta,
    stable_uniform_from_key,
    select_state_verbalizers,
)
from .audit_counterfactual_index import audit_index
from .audit_loss_objective_checkpoint import audit_loss_objective_checkpoint
from .train_phase2_loss_redesign import phase2_parameter_groups


class FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        self.last_add_special_tokens = add_special_tokens
        return self.mapping[text]


class StateVerbalizerTests(unittest.TestCase):
    def test_uses_first_distinct_one_token_pair(self):
        tokenizer = FakeTokenizer({
            " normal": [10],
            " anomalous": [11],
            " N": [12],
            " A": [13],
            " 0": [14],
            " 1": [15],
        })
        verbalizers = select_state_verbalizers(tokenizer)
        self.assertEqual((verbalizers.normal_id, verbalizers.anomalous_id), (10, 11))
        self.assertIn("NORMAL", verbalizers.question)
        self.assertIn("ANOMALOUS", verbalizers.question)
        self.assertFalse(tokenizer.last_add_special_tokens)

    def test_falls_back_and_makes_mapping_explicit(self):
        tokenizer = FakeTokenizer({
            " normal": [1, 2],
            " anomalous": [3, 4],
            " N": [5],
            " A": [6],
            " 0": [7],
            " 1": [8],
        })
        verbalizers = select_state_verbalizers(tokenizer)
        self.assertEqual((verbalizers.normal_text, verbalizers.anomalous_text), (" N", " A"))
        self.assertIn("N = NORMAL", verbalizers.question)
        self.assertIn("A = ANOMALOUS", verbalizers.question)

    def test_state_question_ends_at_classification_position(self):
        self.assertTrue(build_state_question(" 0", " 1").endswith("State:"))


class PromptTests(unittest.TestCase):
    def test_state_prompt_contains_both_consistent_sources(self):
        prompt = format_axis_question_prompt(
            question=build_state_question(" 0", " 1"),
            start_index=10,
            end_index=12,
            window_values=torch.tensor([1.11, -2.22]),
            num_fixed_tokens=2,
            include_local=True,
            include_window=True,
        )
        self.assertIn("111, -222", prompt)
        self.assertEqual(prompt.count("<|local_hint|>"), 2)
        self.assertEqual(prompt.count("<|fixed_hint|>"), 2)
        self.assertIn("0 = NORMAL", prompt)
        self.assertIn("1 = ANOMALOUS", prompt)

    def test_state_forward_can_detach_fixed_hint_only(self):
        source = Path("src/models/AXIS/AXIS.py").read_text(encoding="utf-8")
        self.assertIn("detach_fixed_hint: bool = False", source)
        self.assertIn("processed_fixed = processed_fixed.detach()", source)
        self.assertIn("previous_padding_side = self.tokenizer.padding_side", source)
        self.assertIn("self.tokenizer.padding_side = previous_padding_side", source)
        self.assertIn("state tokenization did not produce right-side padding", source)


class LossAndScheduleTests(unittest.TestCase):
    def test_binary_state_ce_uses_only_two_logits(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        targets = torch.tensor([0, 1])
        expected = torch.nn.functional.cross_entropy(logits, targets)
        self.assertAlmostEqual(float(compute_state_loss(logits, targets)), float(expected), places=7)

    def test_beta_ramps_zero_to_point_two_in_first_ten_percent(self):
        self.assertEqual(scheduled_beta(0, 100, target_beta=0.2, warmup_ratio=0.1), 0.0)
        self.assertAlmostEqual(scheduled_beta(9, 100, target_beta=0.2, warmup_ratio=0.1), 0.2)
        self.assertAlmostEqual(scheduled_beta(99, 100, target_beta=0.2, warmup_ratio=0.1), 0.2)


class RetrievalTests(unittest.TestCase):
    def test_retrieval_respects_donor_validity_and_ratio(self):
        anchors = torch.tensor([[0.0, 0.0]])
        donor_normal = torch.tensor([[0.0, 0.0], [0.1, 0.1]])
        donor_abnormal = torch.tensor([[4.0, 0.0], [4.1, 0.1]])
        best, found, ratio, rank, eligible = retrieve_consistent_donors(
            anchors,
            donor_normal,
            donor_abnormal,
            torch.tensor([False, True]),
            tau=0.25,
            top_m=8,
            random_values=torch.tensor([0.75]),
        )
        self.assertTrue(bool(found[0]))
        self.assertEqual(int(best[0]), 1)
        self.assertEqual(int(rank[0]), 0)
        self.assertEqual(int(eligible[0]), 1)
        self.assertLessEqual(float(ratio[0]), 0.25)

    def test_uniform_sampling_selects_only_from_nearest_top_m(self):
        anchors = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        donor_normal = torch.tensor([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        donor_abnormal = donor_normal + torch.tensor([[4.0, 0.0]])
        best, found, _ratio, rank, eligible = retrieve_consistent_donors(
            anchors,
            donor_normal,
            donor_abnormal,
            torch.tensor([True, True, True]),
            tau=0.25,
            top_m=2,
            random_values=torch.tensor([0.0, 0.999]),
        )
        self.assertTrue(bool(found.all()))
        self.assertTrue(torch.equal(best, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(rank, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(eligible, torch.tensor([3, 3])))
        self.assertNotIn(2, best.tolist())

    def test_keyed_uniform_is_deterministic_and_bounded(self):
        first = stable_uniform_from_key(72, "series_000001.json:0")
        second = stable_uniform_from_key(72, "series_000001.json:0")
        other = stable_uniform_from_key(72, "series_000001.json:1")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)

    def test_retrieval_rejects_baseline_mismatch(self):
        _best, found, ratio, rank, eligible = retrieve_consistent_donors(
            torch.tensor([[0.0, 0.0]]),
            torch.tensor([[10.0, 10.0]]),
            torch.tensor([[11.0, 10.0]]),
            torch.tensor([True]),
            tau=0.25,
            top_m=8,
            random_values=torch.tensor([0.5]),
        )
        self.assertFalse(bool(found[0]))
        self.assertTrue(math.isinf(float(ratio[0])))
        self.assertEqual(int(rank[0]), -1)
        self.assertEqual(int(eligible[0]), 0)


class CounterfactualDatasetTests(unittest.TestCase):
    @staticmethod
    def _write_series(root: Path, name: str, time_series, normal_series, windows):
        payload = {
            "sample_id": int(name[7:13]),
            "original_data": {"time_series": time_series, "normal_series": normal_series},
            "windows": windows,
        }
        (root / "series" / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_collate_uses_anchor_coordinate_patches_for_both_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "series").mkdir()
            windows0 = [
                {"window_range": {"start": 0, "end": 2}, "question": "q0", "answer": "a0", "question_type": "true_false", "has_anomaly": True},
                {"window_range": {"start": 2, "end": 4}, "question": "q1", "answer": "a1", "question_type": "open_ended", "has_anomaly": False},
            ]
            windows1 = [
                {"window_range": {"start": 1, "end": 3}, "question": "donor", "answer": "a", "question_type": "true_false", "has_anomaly": True}
            ]
            self._write_series(root, "series_000000.json", [0, 1, 2, 3], [0, 0, 0, 0], windows0)
            self._write_series(root, "series_000001.json", [1, 9, 7, 4], [1, 2, 3, 4], windows1)
            train_series = ["series_000000.json", "series_000001.json"]
            random.Random(0).shuffle(train_series)
            index = {
                "version": COUNTERFACTUAL_INDEX_VERSION,
                "seed": 0,
                "train_ratio": 1.0,
                "train_series": train_series,
                "policy": {
                    "name": "coherent_full_sequence_patch_with_explicit_state_targets",
                    "phase_fallback": False,
                    "post_patch_dual_source_validation": True,
                    "same_length_required": True,
                    "gamma_window": 0.1,
                    "gamma_local": 0.1,
                    "tau": 0.25,
                    "donor_selection": "uniform_top_m_by_window_distance",
                    "donor_top_m": 8,
                    "donor_sampling_seed": 0,
                    "donor_sampling_key": "sha256(seed:anchor_key)",
                    "replacement_across_anchors": True,
                },
                "stats": {
                    "valid_pairs": 2,
                    "valid_residual_transplants": 1,
                    "valid_anomaly_deletions": 1,
                    "donor_top_m": 8,
                    "unique_residual_donors": 1,
                    "max_residual_donor_reuse": 1,
                    "p95_residual_donor_reuse": 1.0,
                    "top_residual_donor_share": 1.0,
                    "effective_residual_donors": 1.0,
                },
                "records": {
                    "series_000000.json:0": {"has_anomaly": True, "valid": True, "kind": "self_normal_patch", "target_state": 0},
                    "series_000000.json:1": {
                        "has_anomaly": False,
                        "valid": True,
                        "kind": "residual_transplant",
                        "target_state": 1,
                        "donor_series_file": "series_000001.json",
                        "donor_window_index": 0,
                        "donor_key": "series_000001.json:0",
                        "donor_sample_rank": 0,
                        "eligible_donor_count": 1,
                        "top_m_pool_size": 1,
                        "match_ratio": 0.0,
                    },
                    "series_000001.json:0": {
                        "has_anomaly": True,
                        "valid": False,
                        "kind": "invalid",
                        "reason": "fixture",
                    },
                },
            }
            index_path = root / "index.json"
            index_path.write_text(json.dumps(index), encoding="utf-8")
            dataset = CounterfactualAXISDataset(
                str(root), str(index_path), split="train", train_ratio=1.0, seed=0
            )
            item_index = next(index for index, path in enumerate(dataset.series_files) if path.name == "series_000000.json")
            batch = counterfactual_collate_fn([dataset[item_index]])
            self.assertTrue(torch.equal(
                batch["counterfactual_sequences"][0, :4].float(),
                torch.tensor([0.0, 0.0, 2.0, 3.0]),
            ))
            self.assertTrue(torch.equal(
                batch["counterfactual_sequences"][1, :4].float(),
                torch.tensor([0.0, 1.0, 9.0, 7.0]),
            ))
            self.assertTrue(torch.equal(batch["state_targets"], torch.tensor([1, 0])))
            self.assertTrue(bool(batch["counterfactual_valid"].all()))
            self.assertEqual(batch["start_indices"], [0, 2])
            self.assertEqual(batch["end_indices"], [2, 4])
            audit = audit_index(str(root), str(index_path), seed=0, train_ratio=1.0)
            self.assertTrue(audit["ok"])
            self.assertEqual(audit["valid_pairs"], 2)

    def test_invalid_pair_is_copied_but_never_marked_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "series").mkdir()
            windows = [{
                "window_range": {"start": 0, "end": 2},
                "question": "q",
                "answer": "a",
                "question_type": "true_false",
                "has_anomaly": False,
            }]
            self._write_series(root, "series_000000.json", [1, 2], [1, 2], windows)
            index_path = root / "index.json"
            index_path.write_text(json.dumps({
                "version": COUNTERFACTUAL_INDEX_VERSION,
                "seed": 0,
                "train_ratio": 1.0,
                "train_series": ["series_000000.json"],
                "records": {},
            }), encoding="utf-8")
            dataset = CounterfactualAXISDataset(str(root), str(index_path), split="train", train_ratio=1.0, seed=0)
            batch = counterfactual_collate_fn([dataset[0]])
            self.assertFalse(bool(batch["counterfactual_valid"][0]))
            self.assertTrue(torch.equal(batch["padded_sequences"], batch["counterfactual_sequences"]))


class MemorySafeNLLTests(unittest.TestCase):
    def test_chunked_token_nll_matches_direct_cross_entropy(self):
        lm_head = torch.nn.Linear(2, 3, bias=False)
        hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        targets = torch.tensor([[0, 1, -100]])
        sums, counts = memory_safe_token_nll_sums(
            lm_head, hidden, targets, chunk_size=1, checkpoint_chunks=False
        )
        direct = torch.nn.functional.cross_entropy(
            lm_head(hidden).reshape(-1, 3), targets.reshape(-1),
            ignore_index=-100, reduction="none",
        ).reshape(1, 3)
        valid = targets.ne(-100)
        self.assertTrue(torch.allclose(sums, (direct * valid).sum(1)))
        self.assertTrue(torch.equal(counts, valid.sum(1)))

    def test_checkpoint_backward_uses_each_chunks_own_targets(self):
        lm_head = torch.nn.Linear(2, 3, bias=False)
        hidden = torch.randn(1, 3, 2, requires_grad=True)
        sums, _ = memory_safe_token_nll_sums(
            lm_head,
            hidden,
            torch.tensor([[0, 1, 2]]),
            chunk_size=2,
            checkpoint_chunks=True,
        )
        sums.sum().backward()
        self.assertIsNotNone(hidden.grad)


class OptimizerGroupingTests(unittest.TestCase):
    class TinyPerceiver(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping_layer = torch.nn.Linear(2, 2)
            self.local_attention = torch.nn.Linear(2, 2)
            self.local_word_proj = torch.nn.Linear(2, 2)
            self.continuous_proj = torch.nn.Linear(2, 2)
            self.gate_proj = torch.nn.Linear(4, 2)
            self.fusion_norm = torch.nn.LayerNorm(2)
            self.task_prompt_embeddings = torch.nn.Parameter(torch.randn(1, 3, 2))

    def test_recommended_learning_rates_cover_every_parameter_once(self):
        module = self.TinyPerceiver()
        groups, metadata = phase2_parameter_groups(module)
        self.assertEqual(metadata["prototype_attention"]["lr"], 2e-5)
        self.assertEqual(metadata["local_continuous"]["lr"], 5e-5)
        self.assertEqual(metadata["task_prompt"]["lr"], 5e-5)
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))
        self.assertEqual(
            sum(parameter.numel() for parameter in module.parameters()),
            sum(item["parameters"] for item in metadata.values()),
        )


class ObjectiveCheckpointAuditTests(unittest.TestCase):
    def _payload(self):
        return {
            "epoch": 1,
            "global_step": 10,
            "reproduction_meta": {
                "objective": "answer_nll_plus_coherent_binary_state_ce_v2",
                "objective_version": 2,
                "seed": 72,
                "counterfactual_index_version": COUNTERFACTUAL_INDEX_VERSION,
                "beta_target": 0.2,
                "beta_warmup_ratio": 0.1,
                "gradient_clip": 1.0,
                "fixed_hint_state_gradient": False,
                "fixed_hint_answer_gradient": True,
                "state_verbalizers": {
                    "normal_text": " N",
                    "anomalous_text": " A",
                    "normal_id": 10,
                    "anomalous_id": 11,
                    "question": "Classify.\nState:",
                },
                "optimizer_groups": {
                    "local_continuous": {"lr": 5e-5, "parameters": 1},
                    "prototype_attention": {"lr": 2e-5, "parameters": 1},
                    "task_prompt": {"lr": 5e-5, "parameters": 1},
                },
                "counterfactual_policy": {
                    "phase_fallback": False,
                    "post_patch_dual_source_validation": True,
                    "donor_selection": "uniform_top_m_by_window_distance",
                    "donor_top_m": 8,
                    "donor_sampling_key": "sha256(seed:anchor_key)",
                    "donor_sampling_seed": 72,
                    "replacement_across_anchors": True,
                    "gamma_window": 0.1,
                    "gamma_local": 0.1,
                },
            },
        }

    def test_accepts_only_v2_objective_metadata(self):
        result = audit_loss_objective_checkpoint(self._payload(), expected_donor_top_m=8)
        self.assertTrue(result["ok"])
        with self.assertRaises(ValueError):
            audit_loss_objective_checkpoint(self._payload(), expected_donor_top_m=1)
        broken = self._payload()
        broken["reproduction_meta"]["margin"] = math.log(2.0)
        with self.assertRaises(ValueError):
            audit_loss_objective_checkpoint(broken)
if __name__ == "__main__":
    unittest.main()
