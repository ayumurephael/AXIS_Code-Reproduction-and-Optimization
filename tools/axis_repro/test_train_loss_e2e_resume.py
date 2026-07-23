import sys
import types
import unittest

axis_test_stub = types.ModuleType("src.models.AXIS.AXIS_test")
axis_test_stub.collate_fn = lambda batch: batch
dataset_stub = types.ModuleType("src.models.AXIS.dataset")
dataset_stub.AXISAnomalyQADataset = object
sys.modules.setdefault("src.models.AXIS.AXIS_test", axis_test_stub)
sys.modules.setdefault("src.models.AXIS.dataset", dataset_stub)

from tools.axis_repro.train_loss_e2e_ddp import (
    LEGACY_EPOCH1_FP16_SOURCE_COMMITS,
    METRIC_KEYS,
    _ensure_frozen_llm_storage_dtype,
    _epoch_frozen_llm_storage_dtype,
    _epoch_learning_rate,
    _optimization_state_from_summary,
    _validate_epoch_boundary_resume,
    _validate_formal_optimizer_schedule,
)
import torch


class TestEpochBoundaryResume(unittest.TestCase):
    def payload(self):
        optimization = {
            "gradient_norm_count": 9500,
            "gradient_l2_norm_mean": 0.05,
            "gradient_l2_norm_max": 2.0,
            "gradient_l2_norm_last": 0.01,
            "nonfinite_gradient_steps": 0,
            "clipped_steps": 0,
            "max_grad_norm": None,
            "parameter_relative_change_by_step": {
                "9500": {"all_perceiver_parameters": 0.4}
            },
        }
        return {
            "epoch": 1,
            "global_step": 9500,
            "model_state_dict": {
                "ts_pretrain_model": {},
                "moirai_trainable": {},
                "fixed_hint_reference": object(),
            },
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "reproduction_meta": {
                "experiment": "loss_e2e_0723",
                "arm": "treatment",
                "run_purpose": "formal",
                "source_commit": next(iter(LEGACY_EPOCH1_FP16_SOURCE_COMMITS)),
                "source_dirty": False,
                "world_size": 3,
                "epochs": 2,
                "planned_steps": 19000,
                "seed": 72,
                "segment_alpha": 0.4,
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "data_audit_sha256": "data-audit",
                "author_checkpoint_sha256": "author-checkpoint",
                "optimization_diagnostics": optimization,
                "data_counts": {"qa_rows_included": 1},
            },
            "cumulative_training_metrics": {key: 1.0 for key in METRIC_KEYS},
        }

    def validate(self, payload):
        return _validate_epoch_boundary_resume(
            payload,
            arm="treatment",
            steps_per_epoch=9500,
            seed=72,
            alpha=0.4,
            initial_lr=1e-4,
            weight_decay=1e-5,
            data_audit_sha256="data-audit",
            author_checkpoint_sha256="author-checkpoint",
        )

    def test_accepts_exact_epoch_one_boundary(self):
        meta = self.validate(self.payload())
        self.assertEqual(meta["arm"], "treatment")

    def test_rejects_mid_epoch_checkpoint(self):
        payload = self.payload()
        payload["global_step"] = 9986
        with self.assertRaisesRegex(ValueError, "epoch-1 boundary"):
            self.validate(payload)

    def test_rejects_changed_data_identity(self):
        payload = self.payload()
        payload["reproduction_meta"]["data_audit_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "data_audit_sha256"):
            self.validate(payload)

    def test_rejects_dirty_source_checkpoint(self):
        payload = self.payload()
        payload["reproduction_meta"]["source_dirty"] = True
        with self.assertRaisesRegex(ValueError, "dirty source"):
            self.validate(payload)

    def test_rejects_missing_fixed_hint(self):
        payload = self.payload()
        del payload["model_state_dict"]["fixed_hint_reference"]
        with self.assertRaisesRegex(ValueError, "cached F0"):
            self.validate(payload)

    def test_restores_raw_gradient_sum(self):
        summary = self.payload()["reproduction_meta"]["optimization_diagnostics"]
        state = _optimization_state_from_summary(
            summary,
            expected_max_grad_norm=None,
        )
        self.assertEqual(state["gradient_norm_count"], 9500)
        self.assertAlmostEqual(state["gradient_norm_sum"], 475.0)
        self.assertEqual(
            state["parameter_relative_change_by_step"],
            summary["parameter_relative_change_by_step"],
        )

    def test_rejects_missing_epoch_one_dtype_evidence(self):
        payload = self.payload()
        payload["reproduction_meta"]["source_commit"] = "unknown-source"
        with self.assertRaisesRegex(ValueError, "dtype evidence"):
            self.validate(payload)

    def test_accepts_manifested_epoch_one_fp16(self):
        payload = self.payload()
        meta = payload["reproduction_meta"]
        meta["source_commit"] = "new-source"
        meta["frozen_llm_storage_dtype_observed_by_epoch"] = {"1": "torch.float16"}
        self.assertEqual(self.validate(payload)["source_commit"], "new-source")

    def test_rejects_manifested_epoch_one_bfloat16(self):
        payload = self.payload()
        payload["reproduction_meta"]["frozen_llm_storage_dtype_observed_by_epoch"] = {
            "1": "torch.bfloat16"
        }
        with self.assertRaisesRegex(ValueError, "not torch.float16"):
            self.validate(payload)

    def test_epoch_learning_rate_schedule(self):
        self.assertEqual(
            _epoch_learning_rate(1, initial_lr=1e-4, epoch2_lr=3e-5),
            1e-4,
        )
        self.assertEqual(
            _epoch_learning_rate(2, initial_lr=1e-4, epoch2_lr=3e-5),
            3e-5,
        )
        self.assertEqual(
            _epoch_learning_rate(2, initial_lr=1e-4, epoch2_lr=None),
            1e-4,
        )

    def test_formal_optimizer_schedule_is_locked(self):
        _validate_formal_optimizer_schedule(
            initial_lr=1e-4,
            epoch2_lr=3e-5,
            weight_decay=1e-5,
            max_grad_norm=None,
        )
        invalid = (
            {
                "initial_lr": 3e-5,
                "epoch2_lr": 3e-5,
                "weight_decay": 1e-5,
                "max_grad_norm": None,
            },
            {
                "initial_lr": 1e-4,
                "epoch2_lr": None,
                "weight_decay": 1e-5,
                "max_grad_norm": None,
            },
            {
                "initial_lr": 1e-4,
                "epoch2_lr": 3e-5,
                "weight_decay": 0.0,
                "max_grad_norm": None,
            },
            {
                "initial_lr": 1e-4,
                "epoch2_lr": 3e-5,
                "weight_decay": 1e-5,
                "max_grad_norm": 1.0,
            },
        )
        for kwargs in invalid:
            with self.assertRaisesRegex(ValueError, "schedule mismatch"):
                _validate_formal_optimizer_schedule(**kwargs)

    def test_frozen_llm_storage_dtype_schedule(self):
        self.assertEqual(_epoch_frozen_llm_storage_dtype(1), torch.float16)
        self.assertEqual(_epoch_frozen_llm_storage_dtype(2), torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "only epochs 1 and 2"):
            _epoch_frozen_llm_storage_dtype(3)

    def test_frozen_llm_dtype_transition_is_audited(self):
        module = torch.nn.Module()
        module.input_embeddings = torch.nn.Embedding(4, 3)
        module.output_embeddings = torch.nn.Linear(3, 4, bias=False)
        module.get_input_embeddings = lambda: module.input_embeddings
        module.get_output_embeddings = lambda: module.output_embeddings
        module.requires_grad_(False)
        first = _ensure_frozen_llm_storage_dtype(module, torch.float16)
        self.assertTrue(first["transition_applied"])
        self.assertEqual(first["after_dtype"], "torch.float16")
        second = _ensure_frozen_llm_storage_dtype(module, torch.bfloat16)
        self.assertTrue(second["transition_applied"])
        self.assertEqual(second["input_embedding_dtype"], "torch.bfloat16")
        third = _ensure_frozen_llm_storage_dtype(module, torch.bfloat16)
        self.assertFalse(third["transition_applied"])

    def test_frozen_llm_dtype_transition_rejects_trainable_parameters(self):
        module = torch.nn.Module()
        module.input_embeddings = torch.nn.Embedding(4, 3)
        module.output_embeddings = torch.nn.Linear(3, 4, bias=False)
        module.get_input_embeddings = lambda: module.input_embeddings
        module.get_output_embeddings = lambda: module.output_embeddings
        with self.assertRaisesRegex(RuntimeError, "trainable parameters"):
            _ensure_frozen_llm_storage_dtype(module, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
