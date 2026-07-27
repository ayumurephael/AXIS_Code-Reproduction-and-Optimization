import unittest

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round7_pipeline import (
    ROUND7_MODE,
    round1_pipeline,
    select_repaired_answer,
)


class Round7BoundaryRepairTest(unittest.TestCase):
    def test_mode_is_registered_as_oe(self):
        self.assertEqual(
            round1_pipeline.ACTIVE_FAMILY,
            {ROUND7_MODE: "open_ended"},
        )

    def test_selects_unclosed_think_without_answer(self):
        self.assertTrue(
            select_repaired_answer(
                "<think>\nUseful reasoning only.",
                "Answer: Useful reasoning.",
            )
        )

    def test_preserves_baseline_with_explicit_answer(self):
        self.assertFalse(
            select_repaired_answer(
                "<think>\nReasoning\nAnswer: Complete response.",
                "Answer: Short response.",
            )
        )

    def test_preserves_closed_reasoning(self):
        self.assertFalse(
            select_repaired_answer(
                "<think>Reasoning</think>\nConclusion.",
                "Answer: Conclusion.",
            )
        )

    def test_rejects_candidate_without_clean_answer_boundary(self):
        self.assertFalse(
            select_repaired_answer(
                "<think>\nReasoning only.",
                "Clean prose but no answer marker.",
            )
        )
        self.assertFalse(
            select_repaired_answer(
                "<think>\nReasoning only.",
                "<think>Answer: still leaked reasoning.",
            )
        )


if __name__ == "__main__":
    unittest.main()