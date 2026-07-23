from __future__ import annotations

import unittest

import torch

from .loss_e2e import _chunk_token_nll
from .select_best_loss_e2e import (
    EFFECTIVE_TOKEN_DEFINITION,
    DECLARED_STEPS,
    SELECTION_METRIC,
    select_best_candidate,
)
from .validate_loss_e2e import (
    validate_checkpoint_identity,
    validate_split_manifest,
)


class ValidationSplitTests(unittest.TestCase):
    def setUp(self):
        self.train = [f"series_{index}.json" for index in range(28500)]
        self.val = [f"series_{index}.json" for index in range(28500, 30000)]
        self.manifest = {
            "seed": 72,
            "train_ratio": 0.95,
            "train_series": self.train,
            "val_series": self.val,
        }

    def test_accepts_exact_seed72_split_and_order(self):
        result = validate_split_manifest(self.manifest, self.val)
        self.assertEqual(result["validation_series"], 1500)
        self.assertEqual(len(result["ordered_validation_series_sha256"]), 64)

    def test_rejects_validation_order_drift(self):
        changed = self.val[:]
        changed[0], changed[1] = changed[1], changed[0]
        with self.assertRaisesRegex(ValueError, "series/order"):
            validate_split_manifest(self.manifest, changed)

    def test_rejects_train_validation_overlap(self):
        manifest = dict(self.manifest)
        manifest["val_series"] = [self.train[0], *self.val[1:]]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_manifest(manifest, manifest["val_series"])


class CheckpointIdentityTests(unittest.TestCase):
    @staticmethod
    def payload(step: int = 9500, arm: str = "treatment"):
        state = {
            "ts_pretrain_model": {},
            "moirai_trainable": {},
        }
        if arm == "treatment":
            state["fixed_hint_reference"] = torch.zeros(1)
        return {
            "epoch": 1 if step <= 9500 else 2,
            "global_step": step,
            "model_state_dict": state,
            "reproduction_meta": {
                "experiment": "loss_e2e_0723",
                "arm": arm,
                "run_purpose": "formal",
                "world_size": 3,
                "epochs": 2,
                "planned_steps": 19000,
                "seed": 72,
                "segment_alpha": 0.40,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "source_dirty": False,
                "source_commit": "a" * 40,
                "data_audit_sha256": "b" * 64,
                "author_checkpoint_sha256": "c" * 64,
                "data_counts": {
                    "train_series": 28500,
                    "qa_rows_total": 57000,
                    "qa_rows_included": 56987,
                    "qa_rows_excluded": 13,
                },
            },
        }

    def test_accepts_all_declared_steps(self):
        for step in DECLARED_STEPS:
            with self.subTest(step=step):
                result = validate_checkpoint_identity(self.payload(step), "treatment")
                self.assertEqual(result["step"], step)

    def test_rejects_diagnostic_mid_step(self):
        with self.assertRaisesRegex(ValueError, "not a declared milestone"):
            validate_checkpoint_identity(self.payload(11650), "treatment")

    def test_rejects_control_with_cached_hint(self):
        payload = self.payload(9500, "control")
        payload["model_state_dict"]["fixed_hint_reference"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "control candidate"):
            validate_checkpoint_identity(payload, "control")


class TokenLossValidationTests(unittest.TestCase):
    def test_no_grad_path_matches_gradient_enabled_path(self):
        generator = torch.Generator().manual_seed(72)
        hidden = torch.randn(2, 7, 5, generator=generator, requires_grad=True)
        targets = torch.tensor([[1, 2, -100, 3, 4, 0, 2], [0, 1, 2, 3, -100, 4, 1]])
        head = torch.nn.Linear(5, 5, bias=False)
        with torch.enable_grad():
            expected = _chunk_token_nll(hidden, targets, head, chunk_size=3)
        with torch.inference_mode():
            actual = _chunk_token_nll(
                hidden.detach(),
                targets,
                head,
                chunk_size=3,
            )
        torch.testing.assert_close(actual, expected.detach())


class BestCandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def candidate(step: int, nll: float) -> dict:
        count = 1000
        return {
            "schema_version": 1,
            "experiment": "loss_e2e_0723",
            "arm": "treatment",
            "candidate_step": step,
            "candidate_epoch": 1 if step <= 9500 else 2,
            "checkpoint_file": f"step_{step}.pth",
            "checkpoint_sha256": f"{step:064x}",
            "checkpoint_source_commit": "a" * 40,
            "author_checkpoint_sha256": "b" * 64,
            "training_data_audit_sha256": "c" * 64,
            "selection_metric": SELECTION_METRIC,
            "metric_direction": "minimize",
            "token_nll_sum": nll * count,
            "effective_answer_token_count": count,
            "effective_token_definition": EFFECTIVE_TOKEN_DEFINITION,
            "global_token_nll": nll,
            "validation_series": 1500,
            "total_rows": 3000,
            "valid_rows": 2999,
            "excluded_rows": 1,
            "validation_seed": 72,
            "train_ratio": 0.95,
            "split_manifest_sha256": "d" * 64,
            "validation_data_manifest_sha256": "e" * 64,
            "world_size": 3,
            "frozen_llm_storage_dtype": "torch.float16",
            "autocast_dtype": "torch.float16",
            "loss_accumulation_dtype": "torch.float64",
            "loss_chunk_size": 64,
            "evaluator_source_dirty": False,
            "validation_summary_file": f"step_{step}.json",
            "validation_summary_sha256": f"{step + 1:064x}",
        }

    def test_selects_lowest_token_weighted_nll(self):
        candidates = [
            self.candidate(4750, 0.90),
            self.candidate(9500, 0.80),
            self.candidate(14250, 0.70),
            self.candidate(19000, 0.75),
        ]
        result = select_best_candidate(candidates, "treatment")
        self.assertEqual(result["selected"]["candidate_step"], 14250)
        self.assertFalse(result["test_or_judge_metrics_consulted"])

    def test_exact_tie_prefers_earlier_step(self):
        candidates = [self.candidate(step, 0.70) for step in DECLARED_STEPS]
        result = select_best_candidate(candidates, "treatment")
        self.assertEqual(result["selected"]["candidate_step"], 4750)

    def test_rejects_denominator_drift(self):
        candidates = [self.candidate(step, 0.70) for step in DECLARED_STEPS]
        candidates[-1]["effective_answer_token_count"] = 999
        candidates[-1]["token_nll_sum"] = 0.70 * 999
        with self.assertRaisesRegex(ValueError, "identities differ"):
            select_best_candidate(candidates, "treatment")

    def test_rejects_effective_token_definition_drift(self):
        candidates = [self.candidate(step, 0.70) for step in DECLARED_STEPS]
        candidates[-1]["effective_token_definition"] = "different mask"
        with self.assertRaisesRegex(ValueError, "identities differ"):
            select_best_candidate(candidates, "treatment")

    def test_rejects_dirty_evaluator(self):
        candidates = [self.candidate(step, 0.70) for step in DECLARED_STEPS]
        candidates[0]["evaluator_source_dirty"] = True
        with self.assertRaisesRegex(ValueError, "dirty evaluator"):
            select_best_candidate(candidates, "treatment")


if __name__ == "__main__":
    unittest.main()
