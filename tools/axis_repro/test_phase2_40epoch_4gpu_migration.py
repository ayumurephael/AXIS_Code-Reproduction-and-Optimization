from __future__ import annotations

import unittest
from types import SimpleNamespace

from .prompt_boundary import SIMPLIFIED_FINAL_ANSWER_V1
from .train_loss_e2e_ddp import METRIC_KEYS
from .train_phase2_treatment_40epoch_ddp import (
    EXPERIMENT,
    FORMAL_ALPHA,
    FORMAL_EPOCHS,
    FORMAL_LR,
    FORMAL_MAX_GRAD_NORM,
    FORMAL_SEED,
    FORMAL_STEPS_PER_EPOCH,
    FORMAL_TOTAL_STEPS,
    FORMAL_WEIGHT_DECAY,
    FORMAL_WORLD_SIZE,
    LEGACY_STEPS_PER_EPOCH,
    LEGACY_WORLD_SIZE,
    _validate_args,
    expected_global_step_for_epoch,
    validate_resume_checkpoint,
)
from .validate_select_phase2_40epoch import validate_checkpoint_identity


SPLIT_HASH = "split"
DATA_AUDIT_HASH = "data"
PHASE1_HASH = "phase1"
F0_HASH = "f0"


def _meta(epoch: int) -> dict:
    legacy = epoch == 1
    meta = {
        "experiment": EXPERIMENT,
        "run_purpose": "formal",
        "objective": "treatment",
        "world_size": LEGACY_WORLD_SIZE if legacy else FORMAL_WORLD_SIZE,
        "epochs": FORMAL_EPOCHS,
        "steps_per_rank_epoch": (
            LEGACY_STEPS_PER_EPOCH if legacy else FORMAL_STEPS_PER_EPOCH
        ),
        "planned_steps": (
            FORMAL_EPOCHS * LEGACY_STEPS_PER_EPOCH
            if legacy
            else FORMAL_TOTAL_STEPS
        ),
        "seed": FORMAL_SEED,
        "segment_alpha": FORMAL_ALPHA,
        "lr": FORMAL_LR,
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "max_grad_norm": FORMAL_MAX_GRAD_NORM,
        "prompt_protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "split_manifest_sha256": SPLIT_HASH,
        "data_audit_sha256": DATA_AUDIT_HASH,
        "phase1_checkpoint_sha256": PHASE1_HASH,
        "source_dirty": False,
        "source_commit": "legacy" if legacy else "migration",
        "fixed_hint": {"sha256": F0_HASH},
        "optimization_diagnostics": {
            "gradient_norm_count": 0,
            "gradient_l2_norm_mean": None,
            "gradient_l2_norm_max": 0.0,
            "gradient_l2_norm_last": None,
            "nonfinite_gradient_steps": 0,
            "clipped_steps": 0,
            "max_grad_norm": FORMAL_MAX_GRAD_NORM,
            "parameter_relative_change_by_step": {},
        },
    }
    if not legacy:
        meta["optimizer_step_schedule"] = {
            "epoch_1": LEGACY_STEPS_PER_EPOCH,
            "epochs_2_40": FORMAL_STEPS_PER_EPOCH,
            "total": FORMAL_TOTAL_STEPS,
        }
    return meta


def _payload(epoch: int) -> dict:
    return {
        "epoch": epoch,
        "global_step": expected_global_step_for_epoch(epoch),
        "model_state_dict": {"fixed_hint_reference": object()},
        "optimizer_state_dict": {},
        "reproduction_meta": _meta(epoch),
        "cumulative_training_metrics": {key: 0.0 for key in METRIC_KEYS},
    }


class FourGpuMigrationTests(unittest.TestCase):
    def test_exact_mixed_step_schedule(self):
        self.assertEqual(expected_global_step_for_epoch(0), 0)
        self.assertEqual(expected_global_step_for_epoch(1), 5_700)
        self.assertEqual(expected_global_step_for_epoch(2), 12_825)
        self.assertEqual(expected_global_step_for_epoch(40), 283_575)
        self.assertEqual(FORMAL_TOTAL_STEPS, 283_575)

    def test_epoch_one_legacy_checkpoint_is_valid_resume(self):
        result = validate_resume_checkpoint(
            _payload(1),
            split_manifest_sha256=SPLIT_HASH,
            data_audit_sha256=DATA_AUDIT_HASH,
            phase1_sha256=PHASE1_HASH,
        )
        self.assertEqual(result["world_size"], 5)

    def test_four_gpu_checkpoint_is_valid_resume(self):
        result = validate_resume_checkpoint(
            _payload(2),
            split_manifest_sha256=SPLIT_HASH,
            data_audit_sha256=DATA_AUDIT_HASH,
            phase1_sha256=PHASE1_HASH,
        )
        self.assertEqual(result["world_size"], 4)

    def test_linear_epoch_times_7125_boundary_is_rejected(self):
        payload = _payload(2)
        payload["global_step"] = 2 * FORMAL_STEPS_PER_EPOCH
        with self.assertRaisesRegex(ValueError, "complete epoch boundary"):
            validate_resume_checkpoint(
                payload,
                split_manifest_sha256=SPLIT_HASH,
                data_audit_sha256=DATA_AUDIT_HASH,
                phase1_sha256=PHASE1_HASH,
            )

    def test_selection_accepts_both_lineage_segments(self):
        first = validate_checkpoint_identity(
            _payload(1),
            expected_epoch=1,
            split_manifest_sha256=SPLIT_HASH,
        )
        second = validate_checkpoint_identity(
            _payload(2),
            expected_epoch=2,
            split_manifest_sha256=SPLIT_HASH,
        )
        self.assertEqual(first["global_step"], 5_700)
        self.assertEqual(second["global_step"], 12_825)

    def test_formal_protocol_requires_four_ranks_and_resume(self):
        args = SimpleNamespace(
            run_purpose="formal",
            epochs=FORMAL_EPOCHS,
            seed=FORMAL_SEED,
            lr=FORMAL_LR,
            weight_decay=FORMAL_WEIGHT_DECAY,
            alpha=FORMAL_ALPHA,
            max_grad_norm=FORMAL_MAX_GRAD_NORM,
            max_steps=None,
            resume_from="epoch_01.pth",
        )
        _validate_args(args, FORMAL_WORLD_SIZE)
        with self.assertRaisesRegex(ValueError, "world_size=5"):
            _validate_args(args, LEGACY_WORLD_SIZE)
        args.resume_from = None
        with self.assertRaisesRegex(ValueError, "requires.*resume"):
            _validate_args(args, FORMAL_WORLD_SIZE)


if __name__ == "__main__":
    unittest.main()
