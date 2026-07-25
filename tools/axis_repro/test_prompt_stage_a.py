import unittest

from src.models.AXIS.prompt_stage_a import (
    MODE_SPECS,
    OUTPUT_PROTOCOLS,
    STAGE_A_MODES,
    build_question_prompt,
    get_condition,
)


def _prompt(mode, question_type="multiple_choice"):
    condition = get_condition(mode)
    return build_question_prompt(
        question="Which option?\nA) Flat\nB) Spike",
        question_type=question_type,
        start=7,
        end=10,
        serialized_values="10, 11, 42",
        local_hint_tokens="<|local_hint|>" * 3,
        fixed_hint_tokens="" if condition.remove_fixed_hint else "<|fixed_hint|>" * 30,
        mode=mode,
    )


class StageAPromptTest(unittest.TestCase):
    def test_formal_matrix_is_explicit(self):
        self.assertEqual(
            STAGE_A_MODES,
            (
                "base",
                "answer_boundary",
                "answer_boundary_wo_fixed",
                "task_protocol",
                "fixed_role",
                "fixed_role_evidence_contract",
                "combined_234",
                "answer_boundary_combined_234",
            ),
        )
        self.assertTrue(set(STAGE_A_MODES).issubset(MODE_SPECS))

    def test_base_prompt_exactly_matches_released_template(self):
        expected = f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** {'<|local_hint|>' * 3}
            - **Overall Summary Hints:** {'<|fixed_hint|>' * 30}

            ### Question
            Which option?
A) Flat
B) Spike
            """
        self.assertEqual(_prompt("base"), expected)
        self.assertEqual(_prompt("answer_boundary"), expected)

    def test_boundary_without_fixed_changes_only_fixed_placeholders_in_text(self):
        prompt = _prompt("answer_boundary_wo_fixed")
        self.assertIn("- **Overall Summary Hints:** ", prompt)
        self.assertNotIn("<|fixed_hint|>", prompt)
        self.assertNotIn("### Active Output Protocol", prompt)
        self.assertNotIn("### Evidence Contract", prompt)

    def test_task_protocol_adds_only_active_protocol(self):
        prompt = _prompt("task_protocol", "true_false")
        self.assertIn(OUTPUT_PROTOCOLS["true_false"], prompt)
        self.assertNotIn(OUTPUT_PROTOCOLS["multiple_choice"], prompt)
        self.assertNotIn(OUTPUT_PROTOCOLS["open_ended"], prompt)
        self.assertIn("Overall Summary Hints", prompt)
        self.assertNotIn("### Evidence Contract", prompt)

    def test_fixed_role_changes_only_label(self):
        prompt = _prompt("fixed_role")
        self.assertIn("Learned Task Guidance/Shared Task-Control Tokens", prompt)
        self.assertNotIn("Overall Summary Hints", prompt)
        self.assertNotIn("### Active Output Protocol", prompt)
        self.assertNotIn("### Evidence Contract", prompt)

    def test_contract_condition_keeps_specified_wording_and_block_layout(self):
        prompt = _prompt("fixed_role_evidence_contract")
        self.assertIn("interval [7, 10)", prompt)
        self.assertIn("steps 7 through 9", prompt)
        self.assertIn("value on the same row", prompt)
        self.assertLess(prompt.index("### Evidence Contract"), prompt.index("### Time Series Data"))
        self.assertIn("- **Values (scaled by 100):** 10, 11, 42", prompt)
        self.assertIn(
            f"- **Per-Step Analysis:** {'<|local_hint|>' * 3}",
            prompt,
        )

    def test_combined_modes_have_all_declared_factors(self):
        for mode in ("combined_234", "answer_boundary_combined_234"):
            with self.subTest(mode=mode):
                prompt = _prompt(mode)
                condition = get_condition(mode)
                self.assertTrue(condition.add_output_protocol)
                self.assertTrue(condition.rename_fixed_hint)
                self.assertTrue(condition.add_evidence_contract)
                self.assertIn("### Active Output Protocol", prompt)
                self.assertIn("### Evidence Contract", prompt)
                self.assertIn(
                    "Learned Task Guidance/Shared Task-Control Tokens",
                    prompt,
                )
        self.assertFalse(get_condition("combined_234").answer_boundary)
        self.assertTrue(get_condition("answer_boundary_combined_234").answer_boundary)

    def test_protocol_mode_rejects_missing_or_unknown_question_type(self):
        with self.assertRaises(ValueError):
            _prompt("task_protocol", None)
        with self.assertRaises(ValueError):
            _prompt("task_protocol", "unknown")


if __name__ == "__main__":
    unittest.main()
