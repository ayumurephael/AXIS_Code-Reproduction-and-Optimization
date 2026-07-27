import unittest

from src.models.AXIS.prompt_stage_a import (
    MODE_SPECS,
    OUTPUT_PROTOCOLS,
    FOLLOWUP_PROMPT_MODES,
    LITERATURE_R1_MODES,
    LITERATURE_R2_MODES,
    LITERATURE_R3_MODES,
    LITERATURE_R4_MODES,
    PARETO_SCREEN_MODES,
    ROUTED_R2_MODES,
    ROUTED_R3_MODES,
    STAGE_A_MODES,
    build_aligned_rows,
    build_question_prompt,
    get_condition,
)


def _prompt(
    mode,
    question_type="multiple_choice",
    question="Which option?\nA) Flat\nB) Spike",
):
    condition = get_condition(mode)
    return build_question_prompt(
        question=question,
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

    def test_literature_round1_matrix_is_locked(self):
        self.assertEqual(
            LITERATURE_R1_MODES,
            (
                "lit_r1_01_mc_semantic_bind",
                "lit_r1_02_mc_pointwise",
                "lit_r1_03_mc_re2",
                "lit_r1_04_tf_minimal",
                "lit_r1_05_tf_clause",
                "lit_r1_06_tf_re2",
                "lit_r1_07_oe_direct",
                "lit_r1_08_oe_re2",
                "lit_r1_09_triplet_minimal",
                "lit_r1_10_triplet_re2",
                "lit_r1_11_decoupled",
                "lit_r1_12_mc_bind_re2",
            ),
        )
        self.assertTrue(set(LITERATURE_R1_MODES).issubset(MODE_SPECS))

    def test_literature_round1_preserves_released_scaffold(self):
        for mode in LITERATURE_R1_MODES:
            for question_type in ("multiple_choice", "open_ended", "true_false"):
                with self.subTest(mode=mode, question_type=question_type):
                    prompt = _prompt(mode, question_type)
                    self.assertIn("### Time Series Data", prompt)
                    self.assertIn("### Contextual Hints", prompt)
                    self.assertIn("Overall Summary Hints", prompt)
                    self.assertNotIn("Learned Task Guidance", prompt)
                    self.assertNotIn("### Evidence Contract", prompt)
                    self.assertNotIn("Answer:", prompt)
                    self.assertEqual(prompt.count("<|fixed_hint|>"), 30)
                    self.assertLess(
                        prompt.index("Per-Step Analysis"),
                        prompt.index("Overall Summary Hints"),
                    )

    def test_literature_round1_routes_only_the_registered_task(self):
        base_mc = _prompt("base", "multiple_choice")
        base_tf = _prompt("base", "true_false")
        base_oe = _prompt("base", "open_ended")
        single_task_modes = {
            "lit_r1_01_mc_semantic_bind": "multiple_choice",
            "lit_r1_02_mc_pointwise": "multiple_choice",
            "lit_r1_03_mc_re2": "multiple_choice",
            "lit_r1_04_tf_minimal": "true_false",
            "lit_r1_05_tf_clause": "true_false",
            "lit_r1_06_tf_re2": "true_false",
            "lit_r1_07_oe_direct": "open_ended",
            "lit_r1_08_oe_re2": "open_ended",
            "lit_r1_12_mc_bind_re2": "multiple_choice",
        }
        baselines = {
            "multiple_choice": base_mc,
            "true_false": base_tf,
            "open_ended": base_oe,
        }
        for mode, changed_type in single_task_modes.items():
            for question_type, baseline in baselines.items():
                if question_type == changed_type:
                    self.assertNotEqual(_prompt(mode, question_type), baseline)
                else:
                    self.assertEqual(_prompt(mode, question_type), baseline)

    def test_literature_round1_re2_repeats_question_exactly(self):
        for mode, question_type in (
            ("lit_r1_03_mc_re2", "multiple_choice"),
            ("lit_r1_06_tf_re2", "true_false"),
            ("lit_r1_08_oe_re2", "open_ended"),
            ("lit_r1_10_triplet_re2", "multiple_choice"),
            ("lit_r1_10_triplet_re2", "true_false"),
            ("lit_r1_10_triplet_re2", "open_ended"),
            ("lit_r1_12_mc_bind_re2", "multiple_choice"),
        ):
            prompt = _prompt(mode, question_type)
            self.assertIn("Read the question again:", prompt)
            self.assertEqual(prompt.count("Which option?"), 2)

    def test_literature_round1_combined_mc_has_rule_and_re2(self):
        prompt = _prompt("lit_r1_12_mc_bind_re2", "multiple_choice")
        self.assertIn("complete text before mapping it", prompt)
        self.assertIn("Read the question again:", prompt)

    def test_literature_round1_rejects_unknown_question_type(self):
        with self.assertRaises(ValueError):
            _prompt("lit_r1_09_triplet_minimal", "unknown")

    def test_literature_round2_matrix_is_locked(self):
        self.assertEqual(
            LITERATURE_R2_MODES,
            ("lit_r2_01_mc_tf_re2_safe",),
        )
        self.assertTrue(set(LITERATURE_R2_MODES).issubset(MODE_SPECS))

    def test_literature_round2_route_reuses_registered_components(self):
        route = "lit_r2_01_mc_tf_re2_safe"
        self.assertEqual(
            _prompt(route, "multiple_choice"),
            _prompt("lit_r1_03_mc_re2", "multiple_choice"),
        )
        self.assertEqual(
            _prompt(route, "true_false"),
            _prompt("lit_r1_06_tf_re2", "true_false"),
        )
        self.assertEqual(
            _prompt(route, "open_ended"),
            _prompt("base", "open_ended"),
        )
        for question_type in ("multiple_choice", "open_ended", "true_false"):
            prompt = _prompt(route, question_type)
            self.assertIn("Overall Summary Hints", prompt)
            self.assertNotIn("Learned Task Guidance", prompt)
            self.assertEqual(prompt.count("<|fixed_hint|>"), 30)

    def test_literature_round3_matrix_is_locked(self):
        self.assertEqual(
            LITERATURE_R3_MODES,
            (
                "lit_r3_01_mc_pointwise_qual",
                "lit_r3_02_mc_semantic_qual",
                "lit_r3_03_mc_salient_aftermath",
                "lit_r3_04_tf_re2_verdict",
                "lit_r3_05_tf_re2_qual_verdict",
                "lit_r3_06_joint_pointwise_tf_qual",
                "lit_r3_07_joint_semantic_tf_qual",
            ),
        )
        self.assertTrue(set(LITERATURE_R3_MODES).issubset(MODE_SPECS))

    def test_literature_round3_single_task_routes_keep_other_tasks_exact(self):
        baselines = {
            "multiple_choice": _prompt("base", "multiple_choice"),
            "true_false": _prompt("base", "true_false"),
            "open_ended": _prompt("base", "open_ended"),
        }
        single_task_modes = {
            "lit_r3_01_mc_pointwise_qual": "multiple_choice",
            "lit_r3_02_mc_semantic_qual": "multiple_choice",
            "lit_r3_03_mc_salient_aftermath": "multiple_choice",
            "lit_r3_04_tf_re2_verdict": "true_false",
            "lit_r3_05_tf_re2_qual_verdict": "true_false",
        }
        for mode, changed_type in single_task_modes.items():
            for question_type, baseline in baselines.items():
                prompt = _prompt(mode, question_type)
                if question_type == changed_type:
                    self.assertNotEqual(prompt, baseline)
                else:
                    self.assertEqual(prompt, baseline)

    def test_literature_round3_rules_target_observed_failures(self):
        pointwise = _prompt(
            "lit_r3_01_mc_pointwise_qual", "multiple_choice"
        )
        self.assertIn("continuation or recovery support it", pointwise)
        self.assertIn("Use qualitative evidence", pointwise)
        self.assertNotIn("brief reason", pointwise)

        salient = _prompt(
            "lit_r3_03_mc_salient_aftermath", "multiple_choice"
        )
        self.assertIn("most salient local change", salient)
        self.assertIn("what happens immediately afterward", salient)

        verdict = _prompt(
            "lit_r3_04_tf_re2_verdict", "true_false"
        )
        self.assertIn('End with exactly "Your answer: True."', verdict)
        self.assertIn("Read the question again:", verdict)

        qualitative_verdict = _prompt(
            "lit_r3_05_tf_re2_qual_verdict", "true_false"
        )
        self.assertIn("Use qualitative shape, direction", qualitative_verdict)
        self.assertIn("do not quote exact values", qualitative_verdict)
        self.assertIn("Read the question again:", qualitative_verdict)

    def test_literature_round3_joint_routes_reuse_exact_components(self):
        self.assertEqual(
            _prompt(
                "lit_r3_06_joint_pointwise_tf_qual",
                "multiple_choice",
            ),
            _prompt("lit_r3_01_mc_pointwise_qual", "multiple_choice"),
        )
        self.assertEqual(
            _prompt("lit_r3_06_joint_pointwise_tf_qual", "true_false"),
            _prompt("lit_r3_05_tf_re2_qual_verdict", "true_false"),
        )
        self.assertEqual(
            _prompt("lit_r3_07_joint_semantic_tf_qual", "multiple_choice"),
            _prompt("lit_r3_02_mc_semantic_qual", "multiple_choice"),
        )
        for mode in LITERATURE_R3_MODES:
            for question_type in (
                "multiple_choice",
                "open_ended",
                "true_false",
            ):
                prompt = _prompt(mode, question_type)
                self.assertIn("Overall Summary Hints", prompt)
                self.assertNotIn("Learned Task Guidance", prompt)
                self.assertNotIn("### Evidence Contract", prompt)
                self.assertEqual(prompt.count("<|fixed_hint|>"), 30)


    def test_literature_round4_matrix_is_locked(self):
        self.assertEqual(
            LITERATURE_R4_MODES,
            (
                "lit_r4_01_tf_neg_re2",
                "lit_r4_02_joint_semantic_tf_neg_re2",
                "lit_r4_03_tf_nonanomaly_re2",
                "lit_r4_04_joint_semantic_tf_nonanomaly_re2",
                "lit_r4_05_tf_neg_prefix",
                "lit_r4_06_joint_semantic_tf_neg_prefix",
            ),
        )
        self.assertTrue(set(LITERATURE_R4_MODES).issubset(MODE_SPECS))

    def test_literature_round4_negative_router_reuses_exact_components(self):
        negative = "True or False: There is no evidence of an anomaly."
        positive = "True or False: An upward spike is anomalous."
        normality = "True or False: The window shows stable normal behavior."

        self.assertEqual(
            _prompt("lit_r4_01_tf_neg_re2", "true_false", negative),
            _prompt("lit_r3_05_tf_re2_qual_verdict", "true_false", negative),
        )
        self.assertEqual(
            _prompt("lit_r4_01_tf_neg_re2", "true_false", positive),
            _prompt("base", "true_false", positive),
        )
        self.assertEqual(
            _prompt("lit_r4_01_tf_neg_re2", "true_false", normality),
            _prompt("base", "true_false", normality),
        )
        self.assertEqual(
            _prompt("lit_r4_03_tf_nonanomaly_re2", "true_false", normality),
            _prompt(
                "lit_r3_05_tf_re2_qual_verdict",
                "true_false",
                normality,
            ),
        )

    def test_literature_round4_prefix_router_does_not_reread(self):
        negative = "True or False: The window does not contain an anomaly."
        positive = "True or False: An upward spike is anomalous."
        prompt = _prompt("lit_r4_05_tf_neg_prefix", "true_false", negative)
        self.assertIn('begin with exactly "Answer: True."', prompt)
        self.assertIn("Preserve the proposition's polarity", prompt)
        self.assertNotIn("Read the question again:", prompt)
        self.assertEqual(prompt.count(negative), 1)
        self.assertEqual(
            _prompt("lit_r4_05_tf_neg_prefix", "true_false", positive),
            _prompt("base", "true_false", positive),
        )

    def test_literature_round4_joint_routes_preserve_scaffold(self):
        negative = "True or False: There is no anomaly."
        self.assertEqual(
            _prompt(
                "lit_r4_02_joint_semantic_tf_neg_re2",
                "multiple_choice",
            ),
            _prompt("lit_r3_02_mc_semantic_qual", "multiple_choice"),
        )
        self.assertEqual(
            _prompt(
                "lit_r4_06_joint_semantic_tf_neg_prefix",
                "multiple_choice",
            ),
            _prompt("lit_r3_02_mc_semantic_qual", "multiple_choice"),
        )
        for mode in LITERATURE_R4_MODES:
            for question_type in (
                "multiple_choice",
                "open_ended",
                "true_false",
            ):
                question = negative if question_type == "true_false" else (
                    "Which option?\nA) Flat\nB) Spike"
                )
                prompt = _prompt(mode, question_type, question)
                self.assertIn("Overall Summary Hints", prompt)
                self.assertNotIn("Learned Task Guidance", prompt)
                self.assertNotIn("### Evidence Contract", prompt)
                self.assertEqual(prompt.count("<|fixed_hint|>"), 30)
if __name__ == "__main__":
    unittest.main()
