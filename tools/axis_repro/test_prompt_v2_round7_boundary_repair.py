import unittest

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round7_pipeline import (
    BOUNDARY_REPAIR_MODES,
    ROUND7_MODE,
    ROUND8_MODE,
    round1_pipeline,
    select_repaired_answer,
    selected_for_mode,
)


class Round7BoundaryRepairTest(unittest.TestCase):
    def test_modes_are_registered_as_oe(self):
        self.assertEqual(
            round1_pipeline.ACTIVE_FAMILY,
            {mode: "open_ended" for mode in BOUNDARY_REPAIR_MODES},
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

    def test_no_anomaly_mode_requires_preserved_negative_conclusion(self):
        baseline = "<think>There is no evidence of subtle anomalies."
        candidate = "Answer: There are no anomalies in this segment."
        self.assertTrue(selected_for_mode(ROUND7_MODE, baseline, candidate))
        self.assertTrue(selected_for_mode(ROUND8_MODE, baseline, candidate))
        self.assertFalse(
            selected_for_mode(
                ROUND8_MODE,
                "<think>Several spikes suggest possible anomalies.",
                "Answer: Several spikes suggest possible anomalies.",
            )
        )

    def test_no_anomaly_mode_rejects_verdict_change(self):
        self.assertFalse(
            selected_for_mode(
                ROUND8_MODE,
                "<think>There is no anomalous behavior.",
                "Answer: A spike is anomalous.",
            )
        )


if __name__ == "__main__":
    unittest.main()