from __future__ import annotations

import collections
import unittest

import torch
from .audit_author_posttrain_checkpoint import AUTHOR_KEYS, audit_author_posttrain_checkpoint

from .author_posttrain import (
    calibrated_beta,
    cosine_warmup_multiplier,
    distributed_cyclic_indices,
    gradient_l2_norm,
    stratified_component_schedule,
)


class StratifiedScheduleTests(unittest.TestCase):
    def test_exact_four_three_one_counts(self):
        schedule = stratified_component_schedule(1600, seed=72)
        self.assertEqual(collections.Counter(schedule), {"mc": 800, "tf": 600, "oe": 200})
        self.assertEqual(schedule, stratified_component_schedule(1600, seed=72))
        self.assertNotEqual(schedule, stratified_component_schedule(1600, seed=73))

    def test_schedule_rejects_inexact_step_budget(self):
        with self.assertRaises(ValueError):
            stratified_component_schedule(1599, seed=72)

    def test_distributed_cycle_has_no_early_cross_rank_duplicates(self):
        pool = list(range(20))
        shards = [
            distributed_cyclic_indices(
                pool,
                local_count=5,
                world_size=3,
                rank=rank,
                seed=72,
                component="mc",
            )
            for rank in range(3)
        ]
        flattened = [shards[rank][step] for step in range(5) for rank in range(3)]
        self.assertEqual(len(flattened), 15)
        self.assertEqual(len(set(flattened)), 15)

    def test_distributed_cycle_repeats_only_after_pool_exhaustion(self):
        pool = list(range(5))
        shards = [
            distributed_cyclic_indices(
                pool,
                local_count=4,
                world_size=3,
                rank=rank,
                seed=72,
                component="oe",
            )
            for rank in range(3)
        ]
        flattened = [shards[rank][step] for step in range(4) for rank in range(3)]
        self.assertEqual(set(flattened[:5]), set(pool))
        self.assertEqual(set(flattened[5:10]), set(pool))


class CalibrationAndScheduleTests(unittest.TestCase):
    def test_cosine_schedule_warms_up_and_finishes_at_target(self):
        values = [cosine_warmup_multiplier(step, 1600) for step in range(1600)]
        self.assertAlmostEqual(values[0], 1 / 80)
        self.assertAlmostEqual(values[79], 1.0)
        self.assertAlmostEqual(values[-1], 0.1)
        self.assertTrue(all(0.1 <= value <= 1.0 for value in values[79:]))

    def test_gradient_norm_and_beta_calibration(self):
        first = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        first.grad = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(float(gradient_l2_norm([first])), 5.0)
        beta = calibrated_beta(
            [2.0, 4.0, 6.0],
            [1.0, 2.0, 3.0],
            target_ratio=0.1,
            lower=0.01,
            upper=1.0,
        )
        self.assertAlmostEqual(beta, 0.2)

    def test_beta_calibration_obeys_cap(self):
        beta = calibrated_beta(
            [100.0], [0.01], target_ratio=0.1, lower=0.01, upper=0.2
        )
        self.assertEqual(beta, 0.2)


class PosttrainCheckpointAuditTests(unittest.TestCase):
    @staticmethod
    def payload(arm: str) -> dict:
        treatment = arm == "treatment"
        return {
            "epoch": 1,
            "global_step": 1600,
            "model_state_dict": {
                "moirai_trainable": {key: torch.tensor(1.0) for key in AUTHOR_KEYS}
            },
            "optimizer_state_dict": {},
            "reproduction_meta": {
                "objective_version": 6,
                "objective": (
                    "author_posttrain_question_type_joint_v1"
                    if treatment else "author_posttrain_answer_control_v1"
                ),
                "stage": "author_checkpoint_posttrain",
                "arm": arm,
                "initializer_kind": "released_author_phase2_best",
                "init_phase2_sha256": "abc",
                "optimizer_reset": True,
                "resume_optimizer": False,
                "world_size": 3,
                "epochs": 1,
                "steps_per_epoch": 1600,
                "seed": 72,
                "fixed_hint_answer_gradient": True,
                "fixed_hint_auxiliary_gradient": False,
                "architecture": {
                    "variant": "loss_only",
                    "qk_norm": False,
                    "continuous_bypass": False,
                    "direct_task_prompt": False,
                    "fixed_query_tokens": 30,
                },
                "calibrated_betas": (
                    {"mc": 0.2, "tf": 0.1, "oe": 0.05}
                    if treatment else {"mc": 0.0, "tf": 0.0, "oe": 0.0}
                ),
                "beta_calibration": {"mc": {}} if treatment else None,
                "beta_calibration_steps": 200 if treatment else 0,
                "counterfactual_index_version": 3 if treatment else None,
                "supervision_cache_version": 1 if treatment else None,
                "auxiliary_sampling": {
                    "ratio_mc_tf_oe": [4, 3, 1],
                    "global_exposure_per_epoch": {"mc": 2400, "tf": 1800, "oe": 600},
                    "valid_pool_counts": {"mc": 2822, "tf": 1048, "oe": 198},
                } if treatment else None,
            },
        }

    def test_audit_accepts_matched_control_and_treatment(self):
        self.assertTrue(audit_author_posttrain_checkpoint(
            self.payload("control"), expected_arm="control"
        )["ok"])
        self.assertTrue(audit_author_posttrain_checkpoint(
            self.payload("treatment"), expected_arm="treatment"
        )["ok"])

    def test_audit_rejects_wrong_exposure_and_architecture(self):
        exposure = self.payload("treatment")
        exposure["reproduction_meta"]["auxiliary_sampling"][
            "global_exposure_per_epoch"
        ]["oe"] = 599
        with self.assertRaises(ValueError):
            audit_author_posttrain_checkpoint(exposure, expected_arm="treatment")
        architecture = self.payload("control")
        architecture["model_state_dict"]["moirai_trainable"]["task_prompt_embeddings"] = torch.tensor(1.0)
        with self.assertRaises(ValueError):
            audit_author_posttrain_checkpoint(architecture, expected_arm="control")


if __name__ == "__main__":
    unittest.main()
