import unittest

from src.models.AXIS.prompt_stage_a import (
    MODE_SPECS,
    V2_R1_MODES,
    build_question_prompt,
)


LOCAL = "<|local_hint|>" * 3
FIXED = "<|fixed_hint|>" * 30
MC_QUESTION = (
    "Which option best describes the window?\n\n"
    "A) Normal.\n\nB) Anomalous."
)
OE_QUESTION = "Describe the observed pattern in this window."
TF_QUESTION = "True or False: The window contains no anomaly."


def prompt(mode: str, question_type: str, question: str) -> str:
    return build_question_prompt(
        question=question,
        question_type=question_type,
        start=10,
        end=13,
        serialized_values="10, 20, 30",
        local_hint_tokens=LOCAL,
        fixed_hint_tokens=FIXED,
        mode=mode,
        aligned_rows="",
    )


class PromptV2Round1Tests(unittest.TestCase):
    def test_all_round1_modes_are_registered(self):
        self.assertEqual(len(V2_R1_MODES), 9)
        self.assertTrue(set(V2_R1_MODES).issubset(MODE_SPECS))

    def test_trained_scaffold_is_preserved(self):
        examples = {
            "multiple_choice": MC_QUESTION,
            "open_ended": OE_QUESTION,
            "true_false": TF_QUESTION,
        }
        for mode in V2_R1_MODES:
            for question_type, question in examples.items():
                with self.subTest(mode=mode, question_type=question_type):
                    value = prompt(mode, question_type, question)
                    self.assertEqual(value.count("<|fixed_hint|>"), 30)
                    self.assertEqual(value.count("<|local_hint|>"), 3)
                    self.assertIn("- **Overall Summary Hints:**", value)
                    self.assertNotIn(
                        "Learned Task Guidance/Shared Task-Control Tokens",
                        value,
                    )
                    self.assertNotIn("### Evidence Contract", value)
                    self.assertNotIn("Read the question again:", value)
                    self.assertNotIn("### Time-Series Evidence", value)
                    self.assertNotIn("### Final Answer", value)

    def test_single_family_routing(self):
        examples = {
            "multiple_choice": MC_QUESTION,
            "open_ended": OE_QUESTION,
            "true_false": TF_QUESTION,
        }
        changed_family = {
            "v2_r1_01_oe_context_balanced": "open_ended",
            "v2_r1_02_oe_context_contrast": "open_ended",
            "v2_r1_04_oe_context_direct": "open_ended",
            "v2_r1_05_oe_fixed_postnote": "open_ended",
            "v2_r1_06_mc_context_balanced": "multiple_choice",
            "v2_r1_07_mc_content_output": "multiple_choice",
            "v2_r1_08_tf_context_whole": "true_false",
            "v2_r1_09_tf_prefix_context": "true_false",
        }
        for mode, active in changed_family.items():
            for question_type, question in examples.items():
                with self.subTest(mode=mode, question_type=question_type):
                    candidate = prompt(mode, question_type, question)
                    baseline = prompt("base", question_type, question)
                    if question_type == active:
                        self.assertNotEqual(candidate, baseline)
                        self.assertIn("### Answering Rule", candidate)
                    else:
                        self.assertEqual(candidate, baseline)

    def test_oe_evidence_router(self):
        mode = "v2_r1_03_oe_evidence_router"
        matching = (
            "What evidence would you look for to identify an anomaly?",
            "What features support the assessment?",
            "What indicators should be examined?",
            "How would you determine whether the change is anomalous?",
        )
        for question in matching:
            with self.subTest(question=question):
                candidate = prompt(mode, "open_ended", question)
                baseline = prompt("base", "open_ended", question)
                self.assertNotEqual(candidate, baseline)
                self.assertIn("Do not replace", candidate)

        candidate = prompt(mode, "open_ended", OE_QUESTION)
        baseline = prompt("base", "open_ended", OE_QUESTION)
        self.assertEqual(candidate, baseline)


if __name__ == "__main__":
    unittest.main()
