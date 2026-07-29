from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

from .loss_e2e_40epoch_runtime import (
    LOCAL_INPUT_MAX_METRIC_KEYS,
    LOCAL_INPUT_SUM_METRIC_KEYS,
)
from .phase2_accumulation import (
    AccumulationPlan,
    add_numeric_totals,
    ddp_window_loss_multiplier,
    new_window_numerical_totals,
    validate_accumulated_objective_denominator,
)
from .prompt_boundary import SIMPLIFIED_FINAL_ANSWER_V1
from .train_loss_e2e_ddp import METRIC_KEYS
from .train_phase2_treatment_40epoch_ddp import (
    EXPERIMENT,
    FORMAL_ACCUMULATION_PLAN,
    FORMAL_ALPHA,
    FORMAL_EPOCHS,
    FORMAL_GRADIENT_ACCUMULATION_STEPS,
    FORMAL_GRADIENT_SKIP_THRESHOLD,
    FORMAL_LR,
    FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
    FORMAL_MAX_EPOCH_GRADIENT_SKIP_RATE,
    FORMAL_MAX_GRAD_NORM,
    FORMAL_MICRO_STEPS_PER_EPOCH,
    FORMAL_PROTOCOL_VERSION,
    FORMAL_SEED,
    FORMAL_STEPS_PER_EPOCH,
    FORMAL_TOTAL_MICRO_STEPS,
    FORMAL_TOTAL_STEPS,
    FORMAL_WEIGHT_DECAY,
    FORMAL_WORLD_SIZE,
    _phase2_summary,
    _validate_args,
    expected_global_step_for_epoch,
    expected_micro_step_for_epoch,
    formal_gradient_guard_policy,
    learning_rate_for_epoch,
    local_input_normalization_policy,
    validate_checkpoint_step_counters,
    validate_resume_checkpoint,
)
from .validate_select_phase2_40epoch import validate_checkpoint_identity


SPLIT_HASH = "split"
DATA_AUDIT_HASH = "data"
PHASE1_HASH = "phase1"
F0_HASH = "f0"


def _guard(epoch: int) -> dict:
    return {
        "threshold": FORMAL_GRADIENT_SKIP_THRESHOLD,
        "skipped_steps": 0,
        "finite_spike_skipped_steps": 0,
        "nonfinite_skipped_steps": 0,
        "consecutive_skips": 0,
        "max_consecutive_skips": 0,
        "consecutive_skip_limit": FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
        "finite_spike_abort_count": 2,
        "last_completed_epoch": {
            "epoch": epoch,
            "epoch_attempted_steps": FORMAL_STEPS_PER_EPOCH,
            "epoch_skipped_steps": 0,
            "epoch_nonfinite_steps": 0,
            "epoch_skip_rate": 0.0,
            "fail_fast_triggered": False,
            "checkpoint_eligible": True,
        },
        "events": [],
    }


def _meta(epoch: int, mode: str = "none") -> dict:
    return {
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "experiment": EXPERIMENT,
        "experiment_arm": (
            "accumulation_only" if mode == "none" else "accumulation_plus_rmsnorm"
        ),
        "run_purpose": "formal",
        "objective": "treatment",
        "world_size": FORMAL_WORLD_SIZE,
        "epochs": FORMAL_EPOCHS,
        "micro_steps_per_rank_epoch": FORMAL_MICRO_STEPS_PER_EPOCH,
        "optimizer_attempts_per_epoch": FORMAL_STEPS_PER_EPOCH,
        "planned_steps": FORMAL_TOTAL_STEPS,
        "planned_micro_steps": FORMAL_TOTAL_MICRO_STEPS,
        "seed": FORMAL_SEED,
        "segment_alpha": FORMAL_ALPHA,
        "lr": FORMAL_LR,
        "lr_schedule": {"epochs_1_40": FORMAL_LR},
        "weight_decay": FORMAL_WEIGHT_DECAY,
        "max_grad_norm": FORMAL_MAX_GRAD_NORM,
        "gradient_accumulation_steps": FORMAL_GRADIENT_ACCUMULATION_STEPS,
        "global_batch_series": FORMAL_ACCUMULATION_PLAN.full_global_batch_series,
        "gradient_accumulation": {"plan": FORMAL_ACCUMULATION_PLAN.to_dict()},
        "prompt_protocol": SIMPLIFIED_FINAL_ANSWER_V1,
        "split_manifest_sha256": SPLIT_HASH,
        "data_audit_sha256": DATA_AUDIT_HASH,
        "phase1_checkpoint_sha256": PHASE1_HASH,
        "source_dirty": False,
        "source_commit": "test",
        "fixed_hint": {"sha256": F0_HASH},
        "local_input_normalization": local_input_normalization_policy(mode, 1e-6),
        "numerical_stability": {
            "gradient_skip_threshold": FORMAL_GRADIENT_SKIP_THRESHOLD,
            "unsafe_updates_applied": False,
            "fail_fast_policy": formal_gradient_guard_policy(),
        },
        "optimization_diagnostics": {
            "optimizer_attempts": expected_global_step_for_epoch(epoch),
            "optimizer_steps_applied": expected_global_step_for_epoch(epoch),
            "micro_steps": expected_micro_step_for_epoch(epoch),
            "full_accumulation_windows": epoch * 219,
            "partial_accumulation_windows": epoch,
            "gradient_norm_count": expected_global_step_for_epoch(epoch),
            "clip_coefficient_count": expected_global_step_for_epoch(epoch),
            "gradient_l2_norm_mean": 1.0,
            "gradient_l2_norm_max": 1.0,
            "gradient_l2_norm_last": 1.0,
            "local_word_proj_gradient_l2_norm_mean": 1.0,
            "local_word_proj_gradient_l2_norm_max": 1.0,
            "local_word_proj_gradient_l2_norm_last": 1.0,
            "clipped_windows": 0,
            "clip_rate": 0.0,
            "clip_coefficient_mean": 1.0,
            "clip_coefficient_min": 1.0,
            "clip_coefficient_last": 1.0,
            "skipped_windows": 0,
            "skip_rate": 0.0,
            "finite_spike_windows": 0,
            "nonfinite_windows": 0,
            "max_grad_norm": FORMAL_MAX_GRAD_NORM,
            "parameter_relative_change_by_step": {},
            "gradient_guard": _guard(epoch),
        },
    }


def _payload(epoch: int, mode: str = "none") -> dict:
    zero_metrics = {key: 0.0 for key in METRIC_KEYS}
    zero_scale = {
        key: 0.0 for key in LOCAL_INPUT_SUM_METRIC_KEYS + LOCAL_INPUT_MAX_METRIC_KEYS
    }
    return {
        "schema_version": FORMAL_PROTOCOL_VERSION,
        "epoch": epoch,
        "global_step": expected_global_step_for_epoch(epoch),
        "micro_step": expected_micro_step_for_epoch(epoch),
        "optimizer_step": expected_global_step_for_epoch(epoch),
        "model_state_dict": {"fixed_hint_reference": object()},
        "optimizer_state_dict": {},
        "rng_state_by_rank": [{} for _ in range(FORMAL_WORLD_SIZE)],
        "reproduction_meta": _meta(epoch, mode),
        "cumulative_training_metrics": zero_metrics,
        "successful_training_metrics": dict(zero_metrics),
        "cumulative_local_input_metrics": zero_scale,
        "last_completed_epoch_training_metrics": {},
    }


def _formal_args(mode: str = "none") -> SimpleNamespace:
    return SimpleNamespace(
        run_purpose="formal",
        epochs=FORMAL_EPOCHS,
        seed=FORMAL_SEED,
        lr=FORMAL_LR,
        weight_decay=FORMAL_WEIGHT_DECAY,
        alpha=FORMAL_ALPHA,
        max_grad_norm=FORMAL_MAX_GRAD_NORM,
        gradient_accumulation_steps=FORMAL_GRADIENT_ACCUMULATION_STEPS,
        local_input_norm=mode,
        local_input_rmsnorm_eps=1e-6,
        gradient_skip_threshold=FORMAL_GRADIENT_SKIP_THRESHOLD,
        max_consecutive_gradient_skips=FORMAL_MAX_CONSECUTIVE_GRADIENT_SKIPS,
        max_steps=None,
        resume_from=None,
    )


class AccumulationV2ProtocolTests(unittest.TestCase):
    def test_frozen_five_gpu_plan_and_partial_window(self):
        self.assertEqual(FORMAL_ACCUMULATION_PLAN.full_global_batch_series, 130)
        self.assertEqual(FORMAL_ACCUMULATION_PLAN.optimizer_attempts_per_epoch, 220)
        self.assertEqual(FORMAL_ACCUMULATION_PLAN.window_sizes.count(26), 219)
        self.assertEqual(FORMAL_ACCUMULATION_PLAN.final_window_micro_steps, 6)
        self.assertEqual(FORMAL_ACCUMULATION_PLAN.final_global_batch_series, 30)

    def test_exact_step_semantics(self):
        self.assertEqual(expected_global_step_for_epoch(1), 220)
        self.assertEqual(expected_global_step_for_epoch(40), 8_800)
        self.assertEqual(expected_micro_step_for_epoch(1), 5_700)
        self.assertEqual(expected_micro_step_for_epoch(40), 228_000)
        self.assertEqual(FORMAL_TOTAL_STEPS, 8_800)
        self.assertEqual(FORMAL_TOTAL_MICRO_STEPS, 228_000)

    def test_learning_rate_is_constant_and_not_divided(self):
        for epoch in (1, 5, 40):
            self.assertEqual(
                learning_rate_for_epoch(epoch, initial_lr=FORMAL_LR),
                FORMAL_LR,
            )

    def test_ddp_multiplier_produces_global_row_mean(self):
        multiplier = ddp_window_loss_multiplier(
            world_size=5,
            global_valid_rows=257,
        )
        rank_objective_sums = [12.0, 20.0, 18.0, 11.0, 16.0]
        ddp_rank_mean = sum(x * multiplier for x in rank_objective_sums) / 5
        self.assertAlmostEqual(ddp_rank_mean, sum(rank_objective_sums) / 257)

    def test_denominator_drift_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_accumulated_objective_denominator(
                declared_global_valid_rows=257,
                observed_global_objective_count=256,
            )

    def test_numeric_merge_ignores_target_only_metadata(self):
        target = new_window_numerical_totals()
        target["metadata"] = 7
        source = new_window_numerical_totals()
        source["optimizer_attempts"] = 1
        add_numeric_totals(target, source)
        self.assertEqual(target["optimizer_attempts"], 1)
        self.assertEqual(target["metadata"], 7)

    def test_formal_args_accept_both_declared_arms(self):
        _validate_args(_formal_args("none"), FORMAL_WORLD_SIZE)
        _validate_args(_formal_args("rmsnorm"), FORMAL_WORLD_SIZE)

    def test_formal_args_reject_old_accumulation_or_world_size(self):
        args = _formal_args()
        args.gradient_accumulation_steps = 1
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
            _validate_args(args, FORMAL_WORLD_SIZE)
        args = _formal_args()
        with self.assertRaisesRegex(ValueError, "world_size"):
            _validate_args(args, 4)

    def test_phase2_summary_exposes_evidence_alias(self):
        values = {key: 0.0 for key in METRIC_KEYS}
        values["explanation_mean_sum"] = 6.0
        values["explanation_row_count"] = 3.0
        summary = _phase2_summary(values)
        self.assertEqual(summary["explanation_nll"], 2.0)
        self.assertEqual(summary["evidence_nll"], 2.0)

    def test_persisted_step_counter_drift_is_rejected(self):
        payload = _payload(1)
        payload["reproduction_meta"]["optimization_diagnostics"][
            "optimizer_attempts"
        ] -= 1
        with self.assertRaisesRegex(ValueError, "optimizer_attempts"):
            validate_checkpoint_step_counters(payload)

    def test_resume_and_selection_accept_v2_none_checkpoint(self):
        payload = _payload(1, "none")
        meta = validate_resume_checkpoint(
            payload,
            split_manifest_sha256=SPLIT_HASH,
            data_audit_sha256=DATA_AUDIT_HASH,
            phase1_sha256=PHASE1_HASH,
            local_input_norm_mode="none",
            local_input_rmsnorm_eps=1e-6,
        )
        self.assertEqual(meta["experiment_arm"], "accumulation_only")
        identity = validate_checkpoint_identity(
            payload,
            expected_epoch=1,
            split_manifest_sha256=SPLIT_HASH,
        )
        self.assertEqual(identity["micro_step"], 5_700)

    def test_resume_rejects_cross_arm_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "local_input_normalization"):
            validate_resume_checkpoint(
                _payload(1, "none"),
                split_manifest_sha256=SPLIT_HASH,
                data_audit_sha256=DATA_AUDIT_HASH,
                phase1_sha256=PHASE1_HASH,
                local_input_norm_mode="rmsnorm",
                local_input_rmsnorm_eps=1e-6,
            )

    def test_old_or_unsafe_checkpoint_is_rejected(self):
        old = _payload(1)
        old["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "predates"):
            validate_checkpoint_identity(
                old,
                expected_epoch=1,
                split_manifest_sha256=SPLIT_HASH,
            )
        unsafe = _payload(1)
        audit = unsafe["reproduction_meta"]["optimization_diagnostics"][
            "gradient_guard"
        ]["last_completed_epoch"]
        audit.update(
            {
                "epoch_skipped_steps": 3,
                "epoch_skip_rate": 3 / FORMAL_STEPS_PER_EPOCH,
                "checkpoint_eligible": False,
            }
        )
        with self.assertRaises(ValueError):
            validate_checkpoint_identity(
                unsafe,
                expected_epoch=1,
                split_manifest_sha256=SPLIT_HASH,
            )


if __name__ == "__main__":
    unittest.main()
