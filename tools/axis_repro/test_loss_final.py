from __future__ import annotations

import argparse
import sys
import types
import unittest

import torch


class _TensorAnnotation:
    def __class_getitem__(cls, _item):
        return torch.Tensor


if "jaxtyping" not in sys.modules:
    jaxtyping = types.ModuleType("jaxtyping")
    jaxtyping.Float = _TensorAnnotation
    jaxtyping.Int = _TensorAnnotation
    sys.modules["jaxtyping"] = jaxtyping
if "einops" not in sys.modules:
    einops = types.ModuleType("einops")
    einops.rearrange = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("einops is unavailable in CPU unit tests")
    )
    sys.modules["einops"] = einops


from src.models.AXIS.AXIS import AXIS
from .audit_loss_final_checkpoint import audit_loss_final_checkpoint
from .loss_redesign import answer_collate_fn
from .question_type_objectives import (
    build_supervision_record,
    canonical_question_type,
    ddp_global_mean,
    oe_question_counterfactual_valid,
    parse_oe_answer_state,
    parse_tf_state_predicate,
    parse_tf_target,
    select_tf_verbalizers,
)
from .train_phase2_loss_final import build_parser


class SemanticRuleTests(unittest.TestCase):
    def test_question_types_are_canonicalized(self):
        self.assertEqual(canonical_question_type("Multiple Choice"), "multiple_choice")
        self.assertEqual(canonical_question_type("true-false"), "true_false")
        self.assertEqual(canonical_question_type("OE"), "open_ended")
        self.assertEqual(canonical_question_type("unsupported"), "unknown")

    def test_tf_parser_requires_pure_state_and_metadata_consistency(self):
        self.assertEqual(parse_tf_target("True. Explanation"), 1)
        self.assertEqual(parse_tf_target("Answer: False because..."), 0)
        self.assertEqual(parse_tf_state_predicate("True or False: The window contains an anomaly."), 1)
        self.assertEqual(parse_tf_state_predicate("The window is free from anomalies."), 0)
        self.assertIsNone(parse_tf_state_predicate("The anomaly is a downward spike near the center."))
        valid = build_supervision_record(
            {
                "question_type": "true_false",
                "question": "True or False: The window contains an anomaly.",
                "answer": "True. It is anomalous.",
                "has_anomaly": True,
            },
            {"valid": True, "kind": "self_normal_patch"},
        )
        self.assertTrue(valid["tf_pair_valid"])
        inconsistent = build_supervision_record(
            {
                "question_type": "true_false",
                "question": "True or False: The window contains an anomaly.",
                "answer": "False. It is anomalous.",
                "has_anomaly": True,
            },
            {"valid": True, "kind": "self_normal_patch"},
        )
        self.assertFalse(inconsistent["tf_pair_valid"])

    def test_oe_is_neutral_explicit_consistent_and_deletion_only(self):
        self.assertEqual(parse_oe_answer_state("There is an anomaly. A peak is visible."), 1)
        self.assertEqual(parse_oe_answer_state("No anomaly is present. Values are smooth."), 0)
        self.assertTrue(oe_question_counterfactual_valid("Assess whether anomalies are present in this window."))
        self.assertFalse(oe_question_counterfactual_valid("Considering the presence of a clear anomaly, explain it."))
        valid = build_supervision_record(
            {
                "question_type": "open_ended",
                "question": "Assess whether anomalies are present in this window.",
                "answer": "There is an anomaly. A sharp change is visible.",
                "has_anomaly": True,
            },
            {"valid": True, "kind": "self_normal_patch"},
        )
        self.assertTrue(valid["oe_pair_valid"])
        injection = build_supervision_record(
            {
                "question_type": "open_ended",
                "question": "Assess whether anomalies are present in this window.",
                "answer": "No anomaly is present. Values are smooth.",
                "has_anomaly": False,
            },
            {"valid": True, "kind": "residual_transplant"},
        )
        self.assertFalse(injection["oe_pair_valid"])


class VerbalizerAndMaskTests(unittest.TestCase):
    class Tokenizer:
        is_fast = True
        pad_token_id = 0
        eos_token_id = 2
        padding_side = "left"

        def encode(self, text, add_special_tokens=False):
            mapping = {" False": [10], " True": [11], "False": [12], "True": [13]}
            return mapping[text]

        def __call__(self, text, **kwargs):
            if kwargs.get("return_offsets_mapping"):
                return {
                    "input_ids": [1, 20, 21],
                    "offset_mapping": [(0, 0), (0, 7), (8, len(text))],
                }
            if isinstance(text, list):
                width = max(len(item) for item in text)
                return {
                    "input_ids": torch.ones(len(text), width, dtype=torch.long),
                    "attention_mask": torch.ones(len(text), width, dtype=torch.long),
                }
            return {"input_ids": [1, 3, 4]}

    def test_tf_verbalizers_use_false_true_order(self):
        selected = select_tf_verbalizers(self.Tokenizer())
        self.assertEqual((selected.false_id, selected.true_id), (10, 11))

    def test_oe_mask_excludes_answer_prefix_and_eos(self):
        axis = AXIS.__new__(AXIS)
        torch.nn.Module.__init__(axis)
        axis.tokenizer = self.Tokenizer()
        axis.num_fixed_tokens = 1
        _ids, _mask, labels = axis.generate_oe_answer_score_batch(
            ["Assess whether an anomaly is present."],
            ["There is an anomaly."],
            torch.tensor([[0.1, 0.2]]),
            [0],
            [2],
            [torch.tensor([0.1, 0.2])],
        )
        supervised = labels[labels.ne(-100)].tolist()
        self.assertEqual(supervised, [21])


class TrainingProtocolTests(unittest.TestCase):
    def test_answer_collate_has_no_counterfactual_fields(self):
        batch = answer_collate_fn([{
            "time_series": torch.tensor([1.0, 2.0]),
            "analysis_data": [{
                "question": "q",
                "answer": "a",
                "question_type": "Multiple Choice",
                "window_range": {"start": 0, "end": 2},
            }],
        }])
        self.assertEqual(batch["question_types"], ["multiple_choice"])
        self.assertNotIn("counterfactual_sequences", batch)

    def test_non_distributed_global_mean(self):
        value = torch.tensor(6.0, requires_grad=True)
        loss = ddp_global_mean(value, 3)
        self.assertEqual(float(loss), 2.0)
        loss.backward()
        self.assertAlmostEqual(float(value.grad), 1 / 3)

    def test_formal_parser_defaults(self):
        parsed = build_parser().parse_args([
            "--stage", "joint",
            "--phase1", "phase1.pth",
            "--counterfactual-index", "cf.json",
            "--supervision-cache", "cache.json",
            "--output", "out",
        ])
        self.assertEqual(parsed.epochs, 6)
        self.assertEqual(parsed.beta_mc, 0.10)
        self.assertEqual(parsed.beta_tf, 0.05)
        self.assertEqual(parsed.beta_oe, 0.05)
        self.assertEqual(parsed.expected_donor_top_m, 1)
        self.assertEqual(parsed.architecture_variant, "full")

    @staticmethod
    def _payload(stage):
        joint = stage == "joint"
        return {
            "epoch": 6,
            "global_step": 57000,
            "model_state_dict": {"moirai_trainable": {"x": torch.tensor(1)}},
            "optimizer_state_dict": {},
            "reproduction_meta": {
                "objective_version": 4,
                "objective": "question_type_routed_joint_v1" if joint else "answer_only_recovery_v1",
                "stage": stage,
                "fixed_hint_answer_gradient": True,
                "fixed_hint_auxiliary_gradient": False,
                "fixed_hint_auxiliary_isolation": "direct_task_prompt_output_stop_gradient",
                "component_backward": "sequential_same_optimizer_step",
                "world_size": 3,
                "epochs": 6,
                "seed": 72,
                "architecture": {"variant": "full", "qk_norm_seq_len": 40},
                "init_phase2": "answer.pth" if joint else None,
                "init_phase2_sha256": "abc" if joint else None,
                "resume_optimizer": joint,
                "auxiliary_warmup_restarted": joint,
                "counterfactual_index": "cf.json" if joint else None,
                "counterfactual_index_version": 3 if joint else None,
                "supervision_cache_version": 1 if joint else None,
                "counterfactual_policy": {"donor_top_m": 1} if joint else None,
                "beta_mc_target": 0.10 if joint else 0.0,
                "beta_tf_target": 0.05 if joint else 0.0,
                "beta_oe_target": 0.05 if joint else 0.0,
                "beta_warmup_ratio": 0.10 if joint else 0.0,
                "ddp_auxiliary_denominator": "global_valid_rows_per_component",
                "oe_qcar_target_tokens": "answer_content_only_no_prefix_eos_padding",
                "oe_qcar_counterfactual_policy": "abnormal_self_normal_patch_only",
                "question_type_routing": {
                    "multiple_choice": "answer+state_ce",
                    "true_false": "answer+tf_q_ce",
                    "open_ended": "answer+oe_qcar",
                },
            },
        }

    def test_checkpoint_audit_accepts_both_stages_and_fails_closed(self):
        self.assertTrue(audit_loss_final_checkpoint(self._payload("answer_only"), expected_stage="answer_only")["ok"])
        self.assertTrue(audit_loss_final_checkpoint(self._payload("joint"), expected_stage="joint")["ok"])
        broken = self._payload("joint")
        broken["reproduction_meta"]["beta_oe_target"] = 0.1
        with self.assertRaises(ValueError):
            audit_loss_final_checkpoint(broken, expected_stage="joint")


if __name__ == "__main__":
    unittest.main()
