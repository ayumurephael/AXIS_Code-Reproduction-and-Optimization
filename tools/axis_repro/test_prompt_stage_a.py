import unittest

from src.models.AXIS.prompt_stage_a import (
    MODE_SPECS,
    OUTPUT_PROTOCOLS,
    FOLLOWUP_PROMPT_MODES,
    PARETO_SCREEN_MODES,
    ROUTED_R2_MODES,
    ROUTED_R3_MODES,
    STAGE_A_MODES,
    build_aligned_rows,
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
        aligned_rows=(
            "Step 0007 | value=+00010 | context=<|local_hint|>\n"
            "Step 0008 | value=+00011 | context=<|local_hint|>\n"
            "Step 0009 | value=+00042 | context=<|local_hint|>"
        ),
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
        self.assertEqual(
            FOLLOWUP_PROMPT_MODES,
            (
                "expert_contract3_fixed_section",
                "expert_contract3_interleaved",
                "fixed_role_evidence_contract_revised",
            ),
        )
        self.assertTrue(set(FOLLOWUP_PROMPT_MODES).issubset(MODE_SPECS))

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


    def test_expert_contract3_uses_approved_standalone_fixed_section(self):
        prompt = _prompt("expert_contract3_fixed_section")
        self.assertIn(
            "You are an expert time-series anomaly analyst. Produce one "
            "precise, evidence-grounded answer to the question.",
            prompt,
        )
        self.assertIn("1. Window values are exact", prompt)
        self.assertIn("2. Each Per-Step Context token", prompt)
        self.assertIn("3. Shared Task-Control tokens", prompt)
        self.assertNotIn("4. The wording of the question", prompt)
        self.assertNotIn("5. An anomaly is an unexpected", prompt)
        self.assertIn("### Time Series Data", prompt)
        self.assertIn("### Contextual Hints", prompt)
        self.assertIn(
            f"- **Per-Step Analysis:** {'<|local_hint|>' * 3}",
            prompt,
        )
        self.assertIn(
            f"### Learned Task Guidance/Shared Task-Control Tokens\n\n"
            f"            {'<|fixed_hint|>' * 30}",
            prompt,
        )
        self.assertNotIn(
            "- **Learned Task Guidance/Shared Task-Control Tokens:**",
            prompt,
        )
        self.assertNotIn("### Time-Series Evidence", prompt)
        self.assertNotIn("### Active Output Protocol", prompt)
        self.assertFalse(
            get_condition("expert_contract3_fixed_section").answer_boundary
        )

    def test_interleaved_prompt_contains_only_real_aligned_rows(self):
        prompt = _prompt("expert_contract3_interleaved")
        self.assertIn("### Time-Series Evidence", prompt)
        self.assertIn(
            "step | observed value | aligned per-step latent evidence",
            prompt,
        )
        self.assertEqual(prompt.count("context=<|local_hint|>"), 3)
        self.assertIn(
            "Step 0007 | value=+00010 | context=<|local_hint|>",
            prompt,
        )
        self.assertNotIn("Step {s+0:04d}", prompt)
        self.assertNotIn("Step {e-1:04d}", prompt)
        self.assertNotIn("### Time Series Data", prompt)
        self.assertNotIn("### Contextual Hints", prompt)
        self.assertLess(
            prompt.index("### Time-Series Evidence"),
            prompt.index(
                "### Learned Task Guidance/Shared Task-Control Tokens"
            ),
        )
        self.assertLess(
            prompt.index(
                "### Learned Task Guidance/Shared Task-Control Tokens"
            ),
            prompt.index("### Question"),
        )

    def test_aligned_row_serialization_matches_approved_format(self):
        rows = build_aligned_rows(8, [2.26, -0.07, 0.0])
        self.assertEqual(
            rows,
            "\n".join(
                (
                    "Step 0008 | value=+00226 | context=<|local_hint|>",
                    "Step 0009 | value=-00007 | context=<|local_hint|>",
                    "Step 0010 | value=+00000 | context=<|local_hint|>",
                )
            ),
        )

    def test_interleaved_mode_requires_aligned_rows(self):
        with self.assertRaises(ValueError):
            build_question_prompt(
                question="Question?",
                question_type="multiple_choice",
                start=0,
                end=1,
                serialized_values="1",
                local_hint_tokens="<|local_hint|>",
                fixed_hint_tokens="<|fixed_hint|>" * 30,
                mode="expert_contract3_interleaved",
            )


    def test_revised_contract_prompt_exactly_matches_approved_template(self):
        prompt = _prompt("fixed_role_evidence_contract_revised")
        expected = f"""
            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window Values are rounded integer encodings of sample-specific
   observations for the half-open interval [7, 10), listed in
   chronological order from step 7 through step 9.
   Use the displayed scale consistently when comparing shape, direction,
   relative change, and local deviations. Do not infer physical units.
   If the question explicitly requires an approximate unscaled numerical
   value, divide the displayed integer by 100 exactly once; otherwise,
   do not rescale the values.

2. Per-Step Context tokens are sample-specific and align one-to-one with
   Window Values by sequence position, not by text row. The first token
   corresponds to the first value at step 7; each subsequent token
   corresponds to the next value; and the last token corresponds to the
   last value at step 9. Use each token only with its aligned
   value when judging whether that local behavior is unexpected relative
   to the complete temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** {'<|local_hint|>' * 3}
            - **Learned Task Guidance/Shared Task-Control Tokens:** {'<|fixed_hint|>' * 30}

            ### Question
            Which option?
A) Flat
B) Spike
            """
        self.assertEqual(prompt, expected)
        self.assertNotIn(
            "The wording of the question and the order of answer options",
            prompt,
        )
        self.assertNotIn("same row", prompt)
        self.assertNotIn("### Active Output Protocol", prompt)
        self.assertNotIn("### Time-Series Evidence", prompt)
        self.assertFalse(
            get_condition(
                "fixed_role_evidence_contract_revised"
            ).answer_boundary
        )

    def test_pareto_screen_matrix_is_locked(self):
        self.assertEqual(
            PARETO_SCREEN_MODES,
            (
                "fixed_role",
                "pareto_p01_boundary",
                "pareto_p02_calibration",
                "pareto_p03_task_rule",
                "pareto_p04_numeric_guard",
                "pareto_p05_single_answer",
                "pareto_p06_abc",
                "pareto_p07_abd",
                "pareto_p08_acde",
                "pareto_p09_bce",
                "pareto_p10_abcde",
                "pareto_p11_abce",
            ),
        )
        self.assertTrue(set(PARETO_SCREEN_MODES).issubset(MODE_SPECS))

    def test_full_pareto_prompt_contains_all_five_factors(self):
        prompt = _prompt("pareto_p10_abcde", "true_false")
        self.assertIn("### Evidence Use", prompt)
        self.assertIn("half-open window [7, 10)", prompt)
        self.assertIn("Call a behavior anomalous only", prompt)
        self.assertIn("Do not convert or quote exact numbers", prompt)
        self.assertIn("### Task Rule", prompt)
        self.assertIn("complete proposition, including negation", prompt)
        self.assertIn("### Response Rule", prompt)
        self.assertIn("Give one final answer only", prompt)
        self.assertIn(
            "Learned Task Guidance/Shared Task-Control Tokens",
            prompt,
        )
        self.assertNotIn("### Evidence Contract", prompt)
        self.assertFalse(get_condition("pareto_p10_abcde").answer_boundary)
        self.assertEqual(prompt.count("<|fixed_hint|>"), 30)

    def test_task_factor_is_question_type_conditioned(self):
        mc = _prompt("pareto_p03_task_rule", "multiple_choice")
        tf = _prompt("pareto_p03_task_rule", "true_false")
        oe = _prompt("pareto_p03_task_rule", "open_ended")
        self.assertIn("complete meaning of every option", mc)
        self.assertIn("map it to True or False", tf)
        self.assertIn("Answer every component requested", oe)
        self.assertNotEqual(mc, tf)
        self.assertNotEqual(tf, oe)
        for prompt in (mc, tf, oe):
            self.assertNotIn("### Evidence Use", prompt)
            self.assertNotIn("### Response Rule", prompt)

    def test_pareto_factor_omissions_match_matrix(self):
        p07 = _prompt("pareto_p07_abd", "multiple_choice")
        self.assertIn("half-open window [7, 10)", p07)
        self.assertIn("Call a behavior anomalous only", p07)
        self.assertIn("Do not convert or quote exact numbers", p07)
        self.assertNotIn("### Task Rule", p07)
        self.assertNotIn("### Response Rule", p07)

        p11 = _prompt("pareto_p11_abce", "true_false")
        self.assertIn("### Task Rule", p11)
        self.assertIn("### Response Rule", p11)
        self.assertNotIn("Do not convert or quote exact numbers", p11)

    def test_pareto_task_rule_rejects_unknown_question_type(self):
        with self.assertRaises(ValueError):
            _prompt("pareto_p03_task_rule", "unknown")

    def test_round2_routed_mode_matrix_is_locked(self):
        self.assertEqual(
            ROUTED_R2_MODES,
            (
                "route_r2_01_minimal",
                "route_r2_02_mc_stable",
                "route_r2_03_tf_boundary",
                "route_r2_04_oe_old_contract",
                "route_r2_05_oe_contract_coverage",
                "route_r2_06_full_routed",
            ),
        )
        self.assertTrue(set(ROUTED_R2_MODES).issubset(MODE_SPECS))
        for mode in ROUTED_R2_MODES:
            self.assertEqual(_prompt(mode).count("<|fixed_hint|>"), 30)
            self.assertIn(
                "Learned Task Guidance/Shared Task-Control Tokens",
                _prompt(mode),
            )

    def test_round2_minimal_routes_by_question_type(self):
        mc = _prompt("route_r2_01_minimal", "multiple_choice")
        tf = _prompt("route_r2_01_minimal", "true_false")
        oe = _prompt("route_r2_01_minimal", "open_ended")
        self.assertNotIn("### Task Rule", mc)
        self.assertIn("complete proposition exactly as written", tf)
        self.assertIn("label and the first explanatory sentence", tf)
        self.assertIn("diagnosis, an assessment method", oe)
        self.assertIn("Cover every requested part", oe)
        self.assertNotIn("### Evidence Contract", mc)
        self.assertNotIn("### Evidence Contract", tf)
        self.assertNotIn("### Evidence Contract", oe)

    def test_round2_mc_stable_is_mc_only(self):
        mc = _prompt("route_r2_02_mc_stable", "multiple_choice")
        tf = _prompt("route_r2_02_mc_stable", "true_false")
        oe = _prompt("route_r2_02_mc_stable", "open_ended")
        self.assertIn("Select exactly one best-supported option", mc)
        self.assertNotIn("Select exactly one best-supported option", tf)
        self.assertNotIn("Select exactly one best-supported option", oe)
        self.assertIn("complete proposition exactly as written", tf)
        self.assertIn("Cover every requested part", oe)

    def test_round2_tf_boundary_is_tf_only_and_half_open(self):
        tf = _prompt("route_r2_03_tf_boundary", "true_false")
        mc = _prompt("route_r2_03_tf_boundary", "multiple_choice")
        oe = _prompt("route_r2_03_tf_boundary", "open_ended")
        self.assertIn("half-open window [7, 10)", tf)
        self.assertNotIn("half-open window [7, 10)", mc)
        self.assertNotIn("half-open window [7, 10)", oe)
        self.assertIn("complete proposition exactly as written", tf)

    def test_round2_old_contract_is_oe_only(self):
        for mode in (
            "route_r2_04_oe_old_contract",
            "route_r2_05_oe_contract_coverage",
            "route_r2_06_full_routed",
        ):
            with self.subTest(mode=mode):
                mc = _prompt(mode, "multiple_choice")
                tf = _prompt(mode, "true_false")
                oe = _prompt(mode, "open_ended")
                self.assertNotIn("### Evidence Contract", mc)
                self.assertNotIn("### Evidence Contract", tf)
                self.assertIn("### Evidence Contract", oe)
                self.assertIn("value on the same row", oe)
        self.assertNotIn(
            "Cover every requested part",
            _prompt("route_r2_04_oe_old_contract", "open_ended"),
        )
        self.assertIn(
            "Cover every requested part",
            _prompt("route_r2_05_oe_contract_coverage", "open_ended"),
        )

    def test_round2_rejects_unknown_question_type(self):
        with self.assertRaises(ValueError):
            _prompt("route_r2_01_minimal", "unknown")

    def test_round3_routed_mode_matrix_is_locked(self):
        self.assertEqual(
            ROUTED_R3_MODES,
            (
                "route_r3_01_mc_f0_safe",
                "route_r3_02_mc_f1_safe",
                "route_r3_03_mc_f0_tf_p0",
                "route_r3_04_mc_f1_tf_p0",
                "route_r3_05_oe_q1",
                "route_r3_06_oe_q2",
                "route_r3_07_oe_q3",
                "route_r3_08_historical",
            ),
        )
        self.assertTrue(set(ROUTED_R3_MODES).issubset(MODE_SPECS))
        for mode in ROUTED_R3_MODES:
            for question_type in ("multiple_choice", "open_ended", "true_false"):
                self.assertEqual(_prompt(mode, question_type).count("<|fixed_hint|>"), 30)

    def test_round3_safe_components_are_byte_identical(self):
        base_oe = _prompt("base", "open_ended")
        base_tf = _prompt("base", "true_false")
        for mode in ("route_r3_01_mc_f0_safe", "route_r3_02_mc_f1_safe"):
            self.assertEqual(_prompt(mode, "open_ended"), base_oe)
            self.assertEqual(_prompt(mode, "true_false"), base_tf)
        self.assertEqual(
            _prompt("route_r3_01_mc_f0_safe", "multiple_choice"),
            _prompt("route_r2_01_minimal", "multiple_choice"),
        )
        self.assertEqual(
            _prompt("route_r3_02_mc_f1_safe", "multiple_choice"),
            _prompt("route_r2_02_mc_stable", "multiple_choice"),
        )

    def test_round3_new_oe_rules_keep_released_evidence_layout(self):
        expected = {
            "route_r3_05_oe_q1": "two or three observed qualitative features",
            "route_r3_06_oe_q2": "event or pattern named in the question",
            "route_r3_07_oe_q3": "First answer the assessment or analysis requested",
        }
        for mode, phrase in expected.items():
            with self.subTest(mode=mode):
                oe = _prompt(mode, "open_ended")
                self.assertIn(phrase, oe)
                self.assertIn("Overall Summary Hints", oe)
                self.assertNotIn("Learned Task Guidance", oe)
                self.assertNotIn("### Evidence Contract", oe)
                self.assertIn("### Time Series Data", oe)
                self.assertIn("### Contextual Hints", oe)
                self.assertEqual(_prompt(mode, "multiple_choice"), _prompt("route_r2_01_minimal", "multiple_choice"))
                self.assertEqual(_prompt(mode, "true_false"), _prompt("route_r2_01_minimal", "true_false"))

    def test_round3_historical_components_are_byte_identical(self):
        self.assertEqual(
            _prompt("route_r3_08_historical", "multiple_choice"),
            _prompt("route_r2_01_minimal", "multiple_choice"),
        )
        self.assertEqual(
            _prompt("route_r3_08_historical", "true_false"),
            _prompt("base", "true_false"),
        )
        self.assertEqual(
            _prompt("route_r3_08_historical", "open_ended"),
            _prompt("route_r2_04_oe_old_contract", "open_ended"),
        )
if __name__ == "__main__":
    unittest.main()
