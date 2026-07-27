import unittest
from types import SimpleNamespace

from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.round5_selection import (
    ACTIVE_FAMILY,
    ROUND5_MODES,
    prompt_changed,
)
from experiments.reproduction.prompt_research_v2_deepseek_v4_pro_20260727.src.run_round5_refinement import (
    REVISION_RULES,
    refinement_question,
)


class Round5RefinementTest(unittest.TestCase):
    def test_mode_matrix_is_locked(self):
        self.assertEqual(len(ROUND5_MODES), 5)
        self.assertEqual(set(ROUND5_MODES), set(REVISION_RULES))
        self.assertTrue(all(ACTIVE_FAMILY[m] == "open_ended" for m in ROUND5_MODES))

    def test_only_oe_is_selected(self):
        for mode in ROUND5_MODES:
            self.assertTrue(
                prompt_changed(
                    SimpleNamespace(question_type="open_ended"),
                    mode,
                )
            )
            self.assertFalse(
                prompt_changed(
                    SimpleNamespace(question_type="multiple_choice"),
                    mode,
                )
            )
            self.assertFalse(
                prompt_changed(
                    SimpleNamespace(question_type="true_false"),
                    mode,
                )
            )

    def test_refinement_prompt_preserves_inputs(self):
        question = "What happened, and why?"
        draft = "\nAnswer: A local spike occurred.\n"
        for mode in ROUND5_MODES:
            prompt = refinement_question(question, draft, mode)
            self.assertIn("### Original Question\n" + question, prompt)
            self.assertIn(
                "### Existing Draft Answer\nAnswer: A local spike occurred.",
                prompt,
            )
            self.assertIn("### Revision Task\n" + REVISION_RULES[mode], prompt)
            self.assertNotIn("scaled by", prompt)

    def test_every_rule_locks_the_verdict(self):
        for rule in REVISION_RULES.values():
            self.assertTrue(
                "anomaly/normal conclusion" in rule
                or "anomaly/normal verdict" in rule
            )
            self.assertTrue(
                "preserv" in rule.lower()
                or "keep" in rule.lower()
                or "unchanged" in rule.lower()
            )


if __name__ == "__main__":
    unittest.main()
