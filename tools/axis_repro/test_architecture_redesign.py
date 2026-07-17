from __future__ import annotations

import importlib.util
import argparse
import inspect
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path

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
    einops.rearrange = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("einops is unavailable in CPU unit tests"))
    sys.modules["einops"] = einops


from src.models.AXIS.AXIS import MultiheadAttention, Perceiver
from tools.axis_repro import architecture_redesign as architecture
from tools.axis_repro import model_utils, strip_checkpoints


class ArchitectureModuleTests(unittest.TestCase):
    def test_architecture_redesign_module_exists(self) -> None:
        spec = importlib.util.find_spec("tools.axis_repro.architecture_redesign")
        self.assertIsNotNone(spec, "architecture redesign support module is missing")

    def test_all_eight_factorial_variants_are_explicit(self) -> None:
        self.assertTrue(hasattr(architecture, "ARCHITECTURE_VARIANTS"))
        variants = architecture.ARCHITECTURE_VARIANTS
        self.assertEqual(set(variants), {
            "loss_only", "qk_only", "bypass_only", "task_prompt_only",
            "qk_bypass", "qk_task_prompt", "bypass_task_prompt", "full",
        })
        self.assertEqual(len({tuple(sorted(value.items())) for value in variants.values()}), 8)
        self.assertEqual(variants["full"], {
            "qk_norm": True,
            "continuous_bypass": True,
            "direct_task_prompt": True,
        })

    def test_qk_scale_matches_paper_initialization(self) -> None:
        self.assertTrue(hasattr(architecture, "qk_scale_initial_value"))
        actual = architecture.qk_scale_initial_value(79)
        self.assertAlmostEqual(actual, math.log2(79**2 - 79), places=12)
        with self.assertRaises(ValueError):
            architecture.qk_scale_initial_value(1)

    def test_local_length_percentile_uses_training_windows_and_rounds_up(self) -> None:
        self.assertTrue(hasattr(architecture, "local_length_percentile"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "series_a.json"
            second = root / "series_b.json"
            first.write_text(json.dumps({"windows": [
                {"window_range": {"start": 0, "end": 2}},
                {"window_range": {"start": 3, "end": 7}},
            ]}), encoding="utf-8")
            second.write_text(json.dumps({"windows": [
                {"window_range": {"start": 10, "end": 20}},
            ]}), encoding="utf-8")
            self.assertEqual(architecture.local_length_percentile([first, second], 50.0), 4)
            self.assertEqual(architecture.local_length_percentile([first, second], 97.5), 10)

    def test_configuration_applies_selected_variant(self) -> None:
        self.assertTrue(hasattr(architecture, "configure_llm_architecture"))
        config = type("Config", (), {})()
        metadata = architecture.configure_llm_architecture(
            config,
            variant="full",
            qk_norm_seq_len=79,
            gate_bias=-2.0,
        )
        self.assertTrue(config.qk_norm)
        self.assertTrue(config.continuous_bypass)
        self.assertTrue(config.direct_task_prompt)
        self.assertEqual(config.qk_norm_seq_len, 79)
        self.assertEqual(config.gate_bias, -2.0)
        self.assertEqual(metadata["variant"], "full")
        self.assertAlmostEqual(metadata["qk_scale_initial"], math.log2(79**2 - 79))


class QKNormAttentionTests(unittest.TestCase):
    @staticmethod
    def _identity_attention() -> MultiheadAttention:
        signature = inspect.signature(MultiheadAttention)
        if "qk_norm" not in signature.parameters:
            raise AssertionError("MultiheadAttention lacks qk_norm support")
        module = MultiheadAttention(2, 1, qk_norm=True, qk_norm_seq_len=4)
        with torch.no_grad():
            identity = torch.eye(2)
            module.q_proj.weight.copy_(identity)
            module.k_proj.weight.copy_(identity)
            module.v_proj.weight.copy_(identity)
            module.out_proj.weight.copy_(identity)
        return module

    def test_qk_norm_has_one_module_shared_scalar(self) -> None:
        module = self._identity_attention()
        self.assertTrue(hasattr(module, "qk_scale"))
        self.assertEqual(module.qk_scale.numel(), 1)
        self.assertAlmostEqual(float(module.qk_scale.detach()), math.log2(12), places=6)

    def test_qk_norm_is_invariant_to_query_norm(self) -> None:
        module = self._identity_attention()
        keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        values = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
        first = module(torch.tensor([[[1.0, 1.0]]]), keys, values)
        second = module(torch.tensor([[[100.0, 100.0]]]), keys, values)
        torch.testing.assert_close(first, second)

    def test_qk_norm_does_not_normalize_values(self) -> None:
        module = self._identity_attention()
        keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        values = torch.tensor([[[2.0, 0.0], [0.0, 4.0]]])
        output = module(torch.zeros(1, 1, 2), keys, values)
        torch.testing.assert_close(output, torch.tensor([[[1.0, 2.0]]]))


class PerceiverRedesignTests(unittest.TestCase):
    @staticmethod
    def _full_perceiver() -> Perceiver:
        signature = inspect.signature(Perceiver)
        required = {"qk_norm", "qk_norm_seq_len", "continuous_bypass", "direct_task_prompt", "gate_bias"}
        missing = required.difference(signature.parameters)
        if missing:
            raise AssertionError(f"Perceiver redesign arguments missing: {sorted(missing)}")
        return Perceiver(
            vocab_size=6,
            hidden_size=4,
            d_proj=4,
            num_prototype=3,
            num_fixed_tokens=2,
            num_heads=2,
            qk_norm=True,
            qk_norm_seq_len=10,
            continuous_bypass=True,
            direct_task_prompt=True,
            gate_bias=-2.0,
        )

    def test_direct_task_tokens_bypass_prototypes(self) -> None:
        module = self._full_perceiver()
        self.assertTrue(hasattr(module, "task_prompt_embeddings"))
        self.assertFalse(hasattr(module, "fix_prompt_embeddings"))
        first = module.process_fixed_embeddings(torch.randn(1, 3, 4), 2)
        second = module.process_fixed_embeddings(torch.randn(1, 3, 4) * 100, 2)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first, module.task_prompt_embeddings[0])

    def test_continuous_bypass_uses_vector_gate_residual_and_layernorm(self) -> None:
        module = self._full_perceiver()
        self.assertEqual(tuple(module.gate_proj.weight.shape), (4, 8))
        with torch.no_grad():
            module.continuous_proj.weight.copy_(torch.eye(4))
            module.continuous_proj.bias.zero_()
            module.gate_proj.weight.zero_()
            module.gate_proj.bias.fill_(-2.0)
        local = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 3.0]])
        source = torch.randn(1, 3, 4)
        continuous = module.continuous_proj(local.unsqueeze(0))
        semantic = module.local_attention(module.local_word_proj(local.unsqueeze(0)), source, source)
        expected = module.fusion_norm(continuous + torch.sigmoid(torch.full_like(continuous, -2.0)) * semantic)
        actual = module.process_local_embeddings(local, source, 0, 2)
        torch.testing.assert_close(actual, expected.squeeze(0))

    def test_continuous_bypass_accepts_encoder_singleton_feature_axis(self) -> None:
        module = self._full_perceiver()
        local = torch.randn(3, 1, 4)
        source = torch.randn(1, 3, 4)
        actual = module.process_local_embeddings(local, source, 0, 3)
        expected = module.process_local_embeddings(local.squeeze(1), source, 0, 3)
        self.assertEqual(tuple(actual.shape), (3, 4))
        torch.testing.assert_close(actual, expected)

    def test_continuous_bypass_rejects_ambiguous_feature_axis(self) -> None:
        module = self._full_perceiver()
        with self.assertRaisesRegex(ValueError, "singleton feature dimension"):
            module.process_local_embeddings(
                torch.randn(3, 2, 4),
                torch.randn(1, 3, 4),
                0,
                3,
            )

    def test_full_variant_has_no_unused_trainable_parameters(self) -> None:
        module = self._full_perceiver()
        source = module.get_source_embeddings(torch.randn(6, 4))
        local_output = module.process_local_embeddings(torch.randn(3, 4), source, 0, 3)
        task_output = module.process_fixed_embeddings(source, 2)
        (local_output.square().sum() + task_output.square().sum()).backward()
        missing = [name for name, parameter in module.named_parameters() if parameter.requires_grad and parameter.grad is None]
        self.assertEqual(missing, [])

    def test_checkpoint_audit_matches_full_state_and_fails_on_missing_gate(self) -> None:
        self.assertTrue(hasattr(architecture, "audit_architecture_checkpoint"))
        module = self._full_perceiver()
        payload = {
            "model_state_dict": {
                "moirai_trainable": module.state_dict(),
            },
            "reproduction_meta": {
                "architecture": {
                    "variant": "full",
                    "qk_norm": True,
                    "continuous_bypass": True,
                    "direct_task_prompt": True,
                    "qk_norm_seq_len": 10,
                    "gate_bias": -2.0,
                    "task_prompt_tokens": 2,
                },
            },
        }
        report = architecture.audit_architecture_checkpoint(
            payload,
            expected_variant="full",
        )
        self.assertTrue(report["ok"])
        self.assertEqual(report["variant"], "full")
        self.assertEqual(report["task_prompt_tokens"], 2)
        broken = {
            **payload,
            "model_state_dict": {
                "moirai_trainable": dict(module.state_dict()),
            },
        }

        del broken["model_state_dict"]["moirai_trainable"]["gate_proj.weight"]
        with self.assertRaises(ValueError):
            architecture.audit_architecture_checkpoint(broken, expected_variant="full")


class PipelineCompatibilityTests(unittest.TestCase):
    def test_model_builder_accepts_architecture_configuration(self) -> None:
        parameters = inspect.signature(model_utils.build_model).parameters
        self.assertIn("architecture_variant", parameters)
        self.assertIn("qk_norm_seq_len", parameters)
        self.assertIn("gate_bias", parameters)

    def test_shared_cli_arguments_default_to_full_for_architecture_runner(self) -> None:
        self.assertTrue(hasattr(architecture, "add_architecture_arguments"))
        parser = argparse.ArgumentParser()
        architecture.add_architecture_arguments(parser, default_variant="full")
        defaults = parser.parse_args([])
        self.assertEqual(defaults.architecture_variant, "full")
        self.assertIsNone(defaults.qk_norm_seq_len)
        self.assertEqual(defaults.gate_bias, -2.0)
        explicit = parser.parse_args([
            "--architecture-variant", "qk_only",
            "--qk-norm-seq-len", "79",
            "--gate-bias", "-1.5",
        ])
        self.assertEqual(explicit.architecture_variant, "qk_only")
        self.assertEqual(explicit.qk_norm_seq_len, 79)
        self.assertEqual(explicit.gate_bias, -1.5)

    def test_training_builder_accepts_architecture_configuration(self) -> None:
        from tools.axis_repro import train_phase2_loss_redesign

        parameters = inspect.signature(train_phase2_loss_redesign.build_model).parameters
        self.assertIn("architecture_variant", parameters)
        self.assertIn("qk_norm_seq_len", parameters)
        self.assertIn("gate_bias", parameters)

    def test_training_parser_supports_full_architecture_default(self) -> None:
        from tools.axis_repro import train_phase2_loss_redesign

        self.assertTrue(hasattr(train_phase2_loss_redesign, "build_parser"))
        parsed = train_phase2_loss_redesign.build_parser(
            default_architecture_variant="full"
        ).parse_args([
            "--phase1", "phase1.pth",
            "--counterfactual-index", "counterfactual.json",
        ])
        self.assertEqual(parsed.architecture_variant, "full")
        self.assertIsNone(parsed.qk_norm_seq_len)
        self.assertEqual(parsed.gate_bias, -2.0)
        self.assertEqual(parsed.beta, 0.2)
        self.assertEqual(parsed.beta_warmup_ratio, 0.1)
        self.assertEqual(parsed.local_lr, 5e-5)
        self.assertEqual(parsed.attention_lr, 2e-5)
        self.assertEqual(parsed.prompt_lr, 5e-5)
        self.assertFalse(hasattr(parsed, "margin"))

    def test_checkpoint_architecture_validation_is_fail_closed(self) -> None:
        self.assertTrue(hasattr(model_utils, "validate_checkpoint_architecture"))
        config = type("Config", (), {
            "architecture_variant": "full",
            "qk_norm_seq_len": 79,
            "gate_bias": -2.0,
        })()
        payload = {"reproduction_meta": {"architecture": {
            "variant": "full",
            "qk_norm_seq_len": 79,
            "gate_bias": -2.0,
        }}}
        model_utils.validate_checkpoint_architecture(payload, config)
        payload["reproduction_meta"]["architecture"]["qk_norm_seq_len"] = 80
        with self.assertRaises(ValueError):
            model_utils.validate_checkpoint_architecture(payload, config)
        with self.assertRaises(ValueError):
            model_utils.validate_checkpoint_architecture({}, config)

    def test_inference_exposes_architecture_arguments(self) -> None:
        from tools.axis_repro import run_inference

        self.assertTrue(hasattr(run_inference, "build_parser"))
        parsed = run_inference.build_parser().parse_args([
            "--checkpoint", "model.pth", "--output", "output",
            "--architecture-variant", "full", "--qk-norm-seq-len", "79",
        ])
        self.assertEqual(parsed.architecture_variant, "full")
        self.assertEqual(parsed.qk_norm_seq_len, 79)

    def test_stripped_checkpoint_preserves_reproduction_metadata(self) -> None:
        self.assertTrue(hasattr(strip_checkpoints, "strip_checkpoint_payload"))
        payload = {
            "model_state_dict": {"x": torch.tensor([1.0])},
            "epoch": 3,
            "global_step": 10,
            "reproduction_meta": {"architecture": {"variant": "full"}},
        }
        stripped = strip_checkpoints.strip_checkpoint_payload(payload)
        self.assertEqual(stripped["reproduction_meta"], payload["reproduction_meta"])
        self.assertNotIn("optimizer_state_dict", stripped)


if __name__ == "__main__":
    unittest.main()
