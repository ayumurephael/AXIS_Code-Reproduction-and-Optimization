import unittest

from .posttrain_architecture_pipeline import (
    PipelineError,
    build_parser,
    strict_best_epoch,
    training_log_failure,
)


class PosttrainArchitecturePipelineTest(unittest.TestCase):
    def test_strict_best_epoch(self) -> None:
        payload = {
            "candidates": [
                {"predictions": "validation/epoch_1/predictions.jsonl", "mean_loss": 1.1},
                {"predictions": "validation/epoch_2/predictions.jsonl", "mean_loss": 0.9},
                {"predictions": "validation/epoch_3/predictions.jsonl", "mean_loss": 1.0},
            ]
        }
        self.assertEqual(strict_best_epoch(payload), 2)

    def test_tied_minimum_fails_closed(self) -> None:
        payload = {
            "candidates": [
                {"predictions": "epoch_1/predictions.jsonl", "mean_loss": 0.9},
                {"predictions": "epoch_2/predictions.jsonl", "mean_loss": 0.9},
                {"predictions": "epoch_3/predictions.jsonl", "mean_loss": 1.0},
            ]
        }
        with self.assertRaises(PipelineError):
            strict_best_epoch(payload)

    def test_training_error_signatures(self) -> None:
        self.assertIsNone(training_log_failure('{"loss": 0.7, "step": 20}'))
        self.assertIn("CUDA out of memory", training_log_failure("CUDA out of memory"))
        self.assertIn("Traceback", training_log_failure("Traceback (most recent call last)"))

    def test_stop_before_geval_is_explicit_and_default_off(self) -> None:
        required = [
            "--model-name",
            "/models/axis",
            "--credential-file",
            "/secrets/deepseek.md",
        ]
        default_args = build_parser().parse_args(required)
        self.assertFalse(default_args.stop_before_geval)
        guarded_args = build_parser().parse_args([
            *required,
            "--stop-before-geval",
        ])
        self.assertTrue(guarded_args.stop_before_geval)


if __name__ == "__main__":
    unittest.main()
