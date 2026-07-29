from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from src.models.AXIS.AXIS import (
    LOCAL_INPUT_NORM_NONE,
    LOCAL_INPUT_NORM_RMSNORM,
    NonAffineRMSNorm,
    Perceiver,
)
from .loss_e2e_40epoch_runtime import (
    LOCAL_INPUT_MAX_METRIC_KEYS,
    LOCAL_INPUT_SUM_METRIC_KEYS,
    configure_local_input_normalization_from_checkpoint,
    local_input_statistics,
)


class LocalInputRMSNormTests(unittest.TestCase):
    def test_non_affine_fp32_per_timestep_rms(self):
        norm = NonAffineRMSNorm(3, eps=1e-6)
        values = torch.tensor(
            [[[3.0, 4.0, 12.0], [0.5, -1.5, 2.0]]],
            dtype=torch.bfloat16,
        )
        result = norm(values)
        self.assertEqual(result.dtype, torch.float32)
        rms = result.square().mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)
        self.assertEqual(list(norm.parameters()), [])
        self.assertEqual(norm.state_dict(), {})

    def test_none_arm_is_exact_fp32_identity(self):
        perceiver = Perceiver(8, 4, 3, 2, 2, 1)
        values = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        result = perceiver.normalize_local_inputs(values)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, values.float(), rtol=0, atol=0)
        self.assertEqual(perceiver.local_input_norm_mode, LOCAL_INPUT_NORM_NONE)

    def test_rmsnorm_is_applied_immediately_before_local_projection(self):
        torch.manual_seed(7)
        perceiver = Perceiver(8, 4, 3, 2, 2, 1)
        perceiver.configure_local_input_normalization(
            LOCAL_INPUT_NORM_RMSNORM,
            eps=1e-6,
        )
        observed = []
        handle = perceiver.local_word_proj.register_forward_pre_hook(
            lambda _module, inputs: observed.append(inputs[0].detach().clone())
        )
        local = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [100.0, -200.0, 50.0],
                [0.2, 0.3, -0.4],
                [7.0, 8.0, 9.0],
            ]
        )
        source = torch.randn(1, 2, 4)
        perceiver.process_local_embeddings(local, source, 1, 3)
        handle.remove()
        self.assertEqual(len(observed), 1)
        expected = perceiver.local_input_norm(local[1:3].unsqueeze(0))
        torch.testing.assert_close(observed[0], expected)
        rms = observed[0].square().mean(dim=-1).sqrt()
        torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-5, rtol=1e-5)

    def test_fixed_hint_path_is_not_affected(self):
        torch.manual_seed(9)
        perceiver = Perceiver(8, 4, 3, 2, 2, 1)
        source = torch.randn(1, 2, 4)
        before = perceiver.process_fixed_embeddings(source, 2)
        perceiver.configure_local_input_normalization(LOCAL_INPUT_NORM_RMSNORM)
        after = perceiver.process_fixed_embeddings(source, 2)
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_mode_is_metadata_driven_and_legacy_defaults_to_none(self):
        perceiver = Perceiver(8, 4, 3, 2, 2, 1)
        axis = SimpleNamespace(perceiver=perceiver)
        legacy = configure_local_input_normalization_from_checkpoint(axis, {})
        self.assertEqual(legacy["mode"], LOCAL_INPUT_NORM_NONE)
        payload = {
            "reproduction_meta": {
                "local_input_normalization": {
                    "mode": LOCAL_INPUT_NORM_RMSNORM,
                    "affine": False,
                    "eps": 1e-6,
                }
            }
        }
        configured = configure_local_input_normalization_from_checkpoint(axis, payload)
        self.assertEqual(configured["mode"], LOCAL_INPUT_NORM_RMSNORM)
        self.assertEqual(perceiver.local_input_norm_mode, LOCAL_INPUT_NORM_RMSNORM)

    def test_nonfinite_checkpoint_epsilon_is_rejected(self):
        perceiver = Perceiver(8, 4, 3, 2, 2, 1)
        axis = SimpleNamespace(perceiver=perceiver)
        payload = {
            "reproduction_meta": {
                "local_input_normalization": {
                    "mode": LOCAL_INPUT_NORM_RMSNORM,
                    "affine": False,
                    "eps": float("nan"),
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "epsilon"):
            configure_local_input_normalization_from_checkpoint(axis, payload)

    def test_scale_metrics_are_exact_and_exclude_qk_activations(self):
        perceiver = Perceiver(8, 4, 3, 2, 2, 1)
        perceiver.configure_local_input_normalization(LOCAL_INPUT_NORM_RMSNORM)
        local = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
        metrics = local_input_statistics(perceiver, local, [0], [2])
        self.assertEqual(
            set(metrics),
            set(LOCAL_INPUT_SUM_METRIC_KEYS + LOCAL_INPUT_MAX_METRIC_KEYS),
        )
        self.assertFalse(any("query" in key or "key" in key for key in metrics))
        self.assertEqual(metrics["local_input_raw_element_count"].item(), 6)
        normalized_rms = (
            metrics["local_input_normalized_square_sum"]
            / metrics["local_input_normalized_element_count"]
        ).sqrt()
        self.assertAlmostEqual(normalized_rms.item(), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
