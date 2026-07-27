import unittest

from src.models.AXIS.prompt_stage_a import (
    V2_R3_MAIN_MODES,
    build_question_prompt,
)


LOCAL = "<|local_hint|>" * 3
FIXED = "<|fixed_hint|>" * 30


def prompt(mode, question, question_type="open_ended"):
    return build_question_prompt(
        question=question,
        question_type=question_type,
        start=7,
        end=10,
        serialized_values="39, -12, 100",
        local_hint_tokens=LOCAL,
        fixed_hint_tokens=FIXED,
        mode=mode,
        aligned_rows="",
    )


BALANCED_BOUNDARY = (
    "How would you assess whether the observed spikes near the window "
    "boundaries are anomalous, and what evidence would support or challenge "
    "this interpretation?"
)
BALANCED_ASSESSMENT = (
    "How would you determine whether this pattern is anomalous, and what "
    "evidence would support or refute that conclusion?"
)
MULTIPLE_TYPES = (
    "What evidence would support or refute the presence of multiple types "
    "of anomalies near the window boundaries?"
)
SUPPORT_ONLY = (
    "What evidence would support the conclusion that this window is normal?"
)


class Round3MainPromptTests(unittest.TestCase):
    def test_broad_router_reuses_round1_evidence_prompt(self):
        routed = prompt("v2_r3_01_oe_balanced_support", BALANCED_BOUNDARY)
        source = prompt("v2_r1_03_oe_evidence_router", BALANCED_BOUNDARY)
        self.assertEqual(routed, source)
        self.assertIn("### Answering Rule", routed)

    def test_boundary_router_is_more_selective(self):
        routed = prompt("v2_r3_02_oe_boundary_balanced", BALANCED_BOUNDARY)
        source = prompt("v2_r1_03_oe_evidence_router", BALANCED_BOUNDARY)
        self.assertEqual(routed, source)
        self.assertEqual(
            prompt(
                "v2_r3_02_oe_boundary_balanced",
                BALANCED_ASSESSMENT,
            ),
            prompt("base", BALANCED_ASSESSMENT),
        )

    def test_assessment_router_is_more_selective(self):
        routed = prompt(
            "v2_r3_03_oe_assessment_balanced",
            BALANCED_ASSESSMENT,
        )
        source = prompt(
            "v2_r1_03_oe_evidence_router",
            BALANCED_ASSESSMENT,
        )
        self.assertEqual(routed, source)

    def test_multiplicity_and_one_sided_questions_stay_baseline(self):
        for mode in V2_R3_MAIN_MODES:
            for question in (MULTIPLE_TYPES, SUPPORT_ONLY):
                with self.subTest(mode=mode, question=question):
                    self.assertEqual(
                        prompt(mode, question),
                        prompt("base", question),
                    )

    def test_closed_families_stay_baseline(self):
        for mode in V2_R3_MAIN_MODES:
            for question_type in ("multiple_choice", "true_false"):
                with self.subTest(mode=mode, question_type=question_type):
                    self.assertEqual(
                        prompt(mode, BALANCED_BOUNDARY, question_type),
                        prompt("base", BALANCED_BOUNDARY, question_type),
                    )

    def test_selected_prompts_preserve_released_scaffold(self):
        for mode in V2_R3_MAIN_MODES:
            text = prompt(mode, BALANCED_BOUNDARY)
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
                    text.index("### Question"),
                )
                self.assertTrue(text.rstrip().endswith(BALANCED_BOUNDARY))


if __name__ == "__main__":
    unittest.main()
