from __future__ import annotations

import unittest
from types import SimpleNamespace

from .prompt_boundary import SIMPLIFIED_FINAL_ANSWER_V1
from .train_loss_e2e_ddp import METRIC_KEYS
from .train_phase2_treatment_40epoch_ddp import (
    EXPERIMENT,
    FORMAL_ALPHA,
    FORMAL_EPOCHS,
    FORMAL_GRADIENT_SKIP_THRESHOLD,
    FORMAL_EPOCH5_LR,
    FORMAL_LR_SWITCH_EPOCH,
    FORMAL_LR,
    FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
    FORMAL_MAX_GRAD_NORM,
    FORMAL_SEED,
    FORMAL_STEPS_PER_EPOCH,
    FORMAL_TOTAL_STEPS,
    FORMAL_WEIGHT_DECAY,
    FOUR_GPU_WORLD_SIZE,
    LEGACY_STEPS_PER_EPOCH,
    LEGACY_WORLD_SIZE,
    RESUMED_STEPS_PER_EPOCH,
    RESUMED_WORLD_SIZE,
    _validate_args,
    expected_global_step_for_epoch,
    learning_rate_for_epoch,
    planned_steps_for_completed_epoch,
    steps_per_epoch_for_completed_epoch,
    validate_resume_checkpoint,
    world_size_for_completed_epoch,
)
from .validate_select_phase2_40epoch import validate_checkpoint_identity


SPLIT_HASH = "split"
DATA_AUDIT_HASH = "data"
PHASE1_HASH = "phase1"
F0_HASH = "f0"


def _meta(epoch: int) -> dict:
    meta = {
        "experiment": EXPERIMENT,
        "run_purpose": "formal",
        "objective": "treatment",
        "world_size": world_size_for_completed_epoch(epoch),
        "epochs": FORMAL_EPOCHS,
        "steps_per_rank_epoch": steps_per_epoch_for_completed_epoch(epoch),
        "planned_steps": planned_steps_for_completed_epoch(epoch),
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
        "source_commit": f"epoch-{epoch}",
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
    if 2 <= epoch <= 3:
        meta["optimizer_step_schedule"] = {
            "epoch_1": LEGACY_STEPS_PER_EPOCH,
            "epochs_2_40": FORMAL_STEPS_PER_EPOCH,
            "total": planned_steps_for_completed_epoch(epoch),
        }
    elif epoch >= 4:
        meta["optimizer_step_schedule"] = {
            "epoch_1": LEGACY_STEPS_PER_EPOCH,
            "epochs_2_3": FORMAL_STEPS_PER_EPOCH,
            "epochs_4_40": RESUMED_STEPS_PER_EPOCH,
            "total": FORMAL_TOTAL_STEPS,
        }
        meta["numerical_stability"] = {
            "perceiver_wide_operations": "FP32 under BF16 outer autocast",
            "gradient_guard_scope": "synchronized across all DDP ranks",
            "gradient_skip_threshold": FORMAL_GRADIENT_SKIP_THRESHOLD,
            "max_consecutive_gradient_skips": (
                FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS
            ),
            "unsafe_updates_applied": False,
        }
        meta["optimization_diagnostics"]["gradient_guard"] = {
            "threshold": FORMAL_GRADIENT_SKIP_THRESHOLD,
            "skipped_steps": 0,
            "finite_spike_skipped_steps": 0,
            "nonfinite_skipped_steps": 0,
            "consecutive_skips": 0,
            "max_consecutive_skips": 0,
            "consecutive_skip_limit": FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
            "events": [],
        }
    if epoch >= FORMAL_LR_SWITCH_EPOCH:
        meta["epoch5_lr"] = FORMAL_EPOCH5_LR
        meta["lr_schedule"] = {
            "epochs_1_4": FORMAL_LR,
            "epochs_5_40": FORMAL_EPOCH5_LR,
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
        self.assertEqual(expected_global_step_for_epoch(3), 19_950)
        self.assertEqual(expected_global_step_for_epoch(4), 25_650)
        self.assertEqual(expected_global_step_for_epoch(40), 230_850)
        self.assertEqual(FORMAL_TOTAL_STEPS, 230_850)

    def test_two_stage_learning_rate_schedule(self):
        for epoch in range(1, FORMAL_LR_SWITCH_EPOCH):
            with self.subTest(epoch=epoch):
                self.assertEqual(
                    learning_rate_for_epoch(
                        epoch,
                        initial_lr=FORMAL_LR,
                        epoch5_lr=FORMAL_EPOCH5_LR,
                    ),
                    FORMAL_LR,
                )
        self.assertEqual(
            learning_rate_for_epoch(
                5,
                initial_lr=FORMAL_LR,
                epoch5_lr=FORMAL_EPOCH5_LR,
            ),
            FORMAL_EPOCH5_LR,
        )

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

    def test_resumed_five_gpu_checkpoint_is_valid_resume(self):
        result = validate_resume_checkpoint(
            _payload(4),
            split_manifest_sha256=SPLIT_HASH,
            data_audit_sha256=DATA_AUDIT_HASH,
            phase1_sha256=PHASE1_HASH,
        )
        self.assertEqual(result["world_size"], 5)

    def test_resumed_checkpoint_without_guard_audit_is_rejected(self):
        payload = _payload(4)
        del payload["reproduction_meta"]["numerical_stability"]
        del payload["reproduction_meta"]["optimization_diagnostics"][
            "gradient_guard"
        ]
        with self.assertRaisesRegex(ValueError, "gradient-guard"):
            validate_resume_checkpoint(
                payload,
                split_manifest_sha256=SPLIT_HASH,
                data_audit_sha256=DATA_AUDIT_HASH,
                phase1_sha256=PHASE1_HASH,
            )

    def test_epoch_five_without_lr_schedule_is_rejected(self):
        payload = _payload(5)
        del payload["reproduction_meta"]["epoch5_lr"]
        del payload["reproduction_meta"]["lr_schedule"]
        with self.assertRaisesRegex(ValueError, "LR schedule"):
            validate_resume_checkpoint(
                payload,
                split_manifest_sha256=SPLIT_HASH,
                data_audit_sha256=DATA_AUDIT_HASH,
                phase1_sha256=PHASE1_HASH,
            )

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

    def test_selection_accepts_all_three_lineage_segments(self):
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
        third = validate_checkpoint_identity(
            _payload(4),
            expected_epoch=4,
            split_manifest_sha256=SPLIT_HASH,
        )
        self.assertEqual(first["global_step"], 5_700)
        self.assertEqual(second["global_step"], 12_825)
        self.assertEqual(third["global_step"], 25_650)
        self.assertEqual(
            third["gradient_guard"]["threshold"],
            FORMAL_GRADIENT_SKIP_THRESHOLD,
        )

    def test_formal_protocol_requires_resumed_five_ranks_and_resume(self):
        args = SimpleNamespace(
            run_purpose="formal",
            epochs=FORMAL_EPOCHS,
            seed=FORMAL_SEED,
            lr=FORMAL_LR,
            epoch5_lr=FORMAL_EPOCH5_LR,
            weight_decay=FORMAL_WEIGHT_DECAY,
            alpha=FORMAL_ALPHA,
            max_grad_norm=FORMAL_MAX_GRAD_NORM,
            gradient_skip_threshold=FORMAL_GRADIENT_SKIP_THRESHOLD,
            max_consecutive_gradient_skips=(
                FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS
            ),
            max_steps=None,
            resume_from="epoch_03.pth",
        )
        _validate_args(args, RESUMED_WORLD_SIZE)
        with self.assertRaisesRegex(ValueError, "world_size=4"):
            _validate_args(args, FOUR_GPU_WORLD_SIZE)
        args.resume_from = None
        with self.assertRaisesRegex(ValueError, "requires.*resume"):
            _validate_args(args, RESUMED_WORLD_SIZE)


if __name__ == "__main__":
    unittest.main()
