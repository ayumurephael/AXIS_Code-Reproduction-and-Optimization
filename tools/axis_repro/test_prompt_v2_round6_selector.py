import unittest

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.run_round6_selector import (
    ROUND6_MODES,
    SELECTOR_RULES,
    parse_selection,
    selector_question,
)


class Round6SelectorTest(unittest.TestCase):
    def test_mode_matrix_is_locked(self):
        self.assertEqual(len(ROUND6_MODES), 3)
        self.assertEqual(set(ROUND6_MODES), set(SELECTOR_RULES))

    def test_pipeline_registers_every_mode_as_oe(self):
        from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src import (
            round6_pipeline,
        )

        self.assertEqual(
            round6_pipeline.round1_pipeline.ACTIVE_FAMILY,
            {mode: "open_ended" for mode in ROUND6_MODES},
        )

    def test_selector_prompt_contains_both_frozen_candidates(self):
        for mode in ROUND6_MODES:
            prompt = selector_question("Question?", "Draft A", "Draft B", mode)
            self.assertIn("### Original Question\nQuestion?", prompt)
            self.assertIn("### Candidate A — Baseline Draft\nDraft A", prompt)
            self.assertIn(
                "### Candidate B — Conservative Revision\nDraft B",
                prompt,
            )
            self.assertIn(SELECTOR_RULES[mode], prompt)
            self.assertTrue(prompt.endswith("Do not write a new answer."))

    def test_selector_parser_is_conservative(self):
        self.assertEqual(parse_selection("Selection: B"), ("B", True))
        self.assertEqual(parse_selection("Answer: A."), ("A", True))
        self.assertEqual(parse_selection("B) Candidate B is better"), ("B", True))
        self.assertEqual(parse_selection("I am unsure."), ("A", False))
        self.assertEqual(
            parse_selection("Selection: B, but Answer: A"),
            ("A", False),
        )


if __name__ == "__main__":
    unittest.main()
