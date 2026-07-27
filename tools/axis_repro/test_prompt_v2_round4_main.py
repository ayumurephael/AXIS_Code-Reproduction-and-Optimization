import unittest

from src.models.AXIS.prompt_stage_a import (
    V2_R4_MAIN_MODES,
    build_question_prompt,
)


LOCAL = "<|local_hint|>" * 3
FIXED = "<|fixed_hint|>" * 30
QUESTION = (
    "How would you assess whether the apparent spikes are anomalous, what "
    "evidence supports or refutes that conclusion, and how do the window "
    "boundaries affect the assessment?"
)


def prompt(mode, question_type="open_ended"):
    return build_question_prompt(
        question=QUESTION,
        question_type=question_type,
        start=7,
        end=10,
        serialized_values="39, -12, 100",
        local_hint_tokens=LOCAL,
        fixed_hint_tokens=FIXED,
        mode=mode,
        aligned_rows="",
    )


class Round4MainPromptTests(unittest.TestCase):
    def test_all_modes_change_open_ended_only(self):
        baseline = prompt("base")
        for mode in V2_R4_MAIN_MODES:
            with self.subTest(mode=mode):
                self.assertNotEqual(prompt(mode), baseline)
                self.assertEqual(
                    prompt(mode, "multiple_choice"),
                    prompt("base", "multiple_choice"),
                )
                self.assertEqual(
                    prompt(mode, "true_false"),
                    prompt("base", "true_false"),
                )

    def test_released_scaffold_and_boundary_are_preserved(self):
        baseline = prompt("base")
        for mode in V2_R4_MAIN_MODES:
            text = prompt(mode)
            with self.subTest(mode=mode):
                self.assertEqual(text.count("<|local_hint|>"), 3)
                self.assertEqual(text.count("<|fixed_hint|>"), 30)
                self.assertIn("- **Overall Summary Hints:**", text)
                self.assertLess(
                    text.index("### Time Series Data"),
                    text.index("### Contextual Hints"),
                )
                self.assertLess(
                    text.index("### Contextual Hints"),
                    text.index("### Answering Rule"),
                )
                self.assertLess(
                    text.index("### Answering Rule"),
                    text.index("### Question"),
                )
                self.assertTrue(text.rstrip().endswith(QUESTION))
                self.assertEqual(
                    baseline.split("### Time Series Data", 1)[0],
                    text.split("### Time Series Data", 1)[0],
                )

    def test_modes_have_distinct_answering_rules(self):
        rules = set()
        for mode in V2_R4_MAIN_MODES:
            text = prompt(mode)
            rule = text.split("### Answering Rule", 1)[1].split(
                "### Question",
                1,
            )[0].strip()
            rules.add(rule)
        self.assertEqual(len(rules), len(V2_R4_MAIN_MODES))

    def test_forbidden_structural_changes_are_absent(self):
        for mode in V2_R4_MAIN_MODES:
            text = prompt(mode)
            with self.subTest(mode=mode):
                self.assertNotIn("Shared Task-Control Tokens", text)
                self.assertNotIn("Aligned context", text)
                self.assertNotIn("Read the question again", text)
                self.assertNotIn("### Final Answer", text)
                self.assertFalse(text.rstrip().endswith("Answer:"))


if __name__ == "__main__":
    unittest.main()
