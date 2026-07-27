import unittest

from src.models.AXIS.prompt_stage_a import (
    NEW_MD_SHORT_EVIDENCE_RULE,
    V2_R2_NEW_MD_MODES,
    build_aligned_rows,
    build_question_prompt,
    get_condition,
)


QUESTION = "True or False: This window contains an anomaly."
LOCAL = "<|local_hint|>" * 3
FIXED = "<|fixed_hint|>" * 30
ROWS = build_aligned_rows(7, [0.394, -0.125, 1.0])


def prompt(mode, question_type="true_false"):
    return build_question_prompt(
        question=QUESTION,
        question_type=question_type,
        start=7,
        end=10,
        serialized_values="39, -12, 100",
        local_hint_tokens=LOCAL,
        fixed_hint_tokens=FIXED,
        mode=mode,
        aligned_rows=ROWS,
    )


class NewMdPromptTests(unittest.TestCase):
    def test_all_modes_preserve_placeholder_counts(self):
        for mode in V2_R2_NEW_MD_MODES:
            with self.subTest(mode=mode):
                text = prompt(mode)
                self.assertEqual(text.count("<|local_hint|>"), 3)
                self.assertEqual(text.count("<|fixed_hint|>"), 30)

    def test_p1_changes_only_by_short_rule_on_released_layout(self):
        text = prompt("v2_r2_01_new_short_rule")
        self.assertIn(NEW_MD_SHORT_EVIDENCE_RULE, text)
        self.assertLess(text.index("### Time Series Data"), text.index("### Contextual Hints"))
        self.assertLess(text.index("### Contextual Hints"), text.index("### Evidence Rule"))
        self.assertLess(text.index("### Evidence Rule"), text.index("### Question"))
        self.assertIn("- **Overall Summary Hints:**", text)
        self.assertNotIn("### Aligned Sample Evidence", text)

    def test_p2_uses_value_before_local_and_question_last(self):
        text = prompt("v2_r2_02_new_step_aligned")
        row = "Step 0007 | Value +00039 | Aligned context <|local_hint|>"
        self.assertIn(row, text)
        self.assertLess(text.index(row), text.index("Overall Summary Hints"))
        self.assertLess(text.index("Overall Summary Hints"), text.index("### Question"))

    def test_p3_conditions_aligned_evidence_on_question(self):
        text = prompt("v2_r2_03_new_question_first")
        row = "Step 0007 | Value +00039 | Aligned context <|local_hint|>"
        self.assertLess(text.index(QUESTION), text.index(row))
        self.assertLess(text.index(row), text.index("Overall Summary Hints"))

    def test_p4_is_fixed_question_evidence_flow_without_prefill(self):
        text = prompt("v2_r2_04_new_evidence_last")
        row = "Step 0007 | Value +00039 | Aligned context <|local_hint|>"
        self.assertLess(text.index("### Shared Task-Control Tokens"), text.index(QUESTION))
        self.assertLess(text.index(QUESTION), text.index(row))
        self.assertTrue(text.rstrip().endswith("Aligned context <|local_hint|>"))
        self.assertFalse(text.rstrip().endswith("Answer:"))

    def test_full_prompt_has_active_short_protocol_and_inline_boundary(self):
        for question_type, expected in {
            "multiple_choice": 'Begin with "<LETTER>) <exact selected option text>".',
            "true_false": 'Begin with exactly "True." or "False."',
            "open_ended": "Begin with a direct diagnostic conclusion.",
        }.items():
            with self.subTest(question_type=question_type):
                text = prompt("v2_r2_05_new_full_prompt", question_type)
                self.assertIn(expected, text)
                self.assertIn("### Evidence-Use Rule", text)
                self.assertIn("### Output Rule", text)
                self.assertIn("### Final Answer", text)
                self.assertTrue(text.endswith("Answer:"))
                self.assertNotIn("Step {start+0:04d}", text)
        condition = get_condition("v2_r2_05_new_full_prompt")
        self.assertTrue(condition.inline_answer_boundary)
        self.assertFalse(condition.answer_boundary)


if __name__ == "__main__":
    unittest.main()
