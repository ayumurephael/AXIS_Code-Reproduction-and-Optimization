# Frozen Prompt Screening Inventory

- Registered modes: 69
- Status counts: `{'advanced_full284': 2, 'baseline': 1, 'excluded': 66}`
- Full-284 candidates: `['lit_r1_08_oe_re2', 'lit_r1_10_triplet_re2']`

| Mode | Category | OE changed? | Decision | Decisive evidence | Reason |
|---|---|:---:|---|---|---|
| `base` | baseline | no | `baseline` | — | Canonical comparison mode. |
| `answer_boundary` | stage_a_prompt | yes | `excluded` | 140 QA; 2/10; worst -0.1395; mean -0.0582 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `answer_boundary_wo_fixed` | stage_a_prompt | yes | `excluded` | 140 QA; 0/10; worst -2.0930; mean -1.7472 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `task_protocol` | stage_a_prompt | yes | `excluded` | 140 QA; 1/10; worst -0.4000; mean -0.2030 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `fixed_role` | stage_a_prompt | yes | `excluded` | 140 QA; 6/10; worst -0.1673; mean +0.0026 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `fixed_role_evidence_contract` | stage_a_prompt | yes | `excluded` | 140 QA; 4/10; worst -0.4988; mean -0.1190 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `fixed_role_evidence_contract_revised` | followup_contract_prompt | yes | `excluded` | 140 QA; 0/10; worst -0.3636; mean -0.1752 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `combined_234` | stage_a_prompt | yes | `excluded` | 140 QA; 2/10; worst -0.6517; mean -0.2018 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `answer_boundary_combined_234` | stage_a_prompt | yes | `excluded` | 140 QA; 0/10; worst -0.2364; mean -0.1436 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `expert_contract3_fixed_section` | followup_contract_prompt | yes | `excluded` | 140 QA; 0/10; worst -0.3636; mean -0.2033 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `expert_contract3_interleaved` | followup_contract_prompt | yes | `excluded` | 140 QA; 0/10; worst -0.6547; mean -0.3473 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p01_boundary` | pareto_prompt | yes | `excluded` | 24 QA; 9/10; worst -0.2857; mean +0.2526 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p02_calibration` | pareto_prompt | yes | `excluded` | 24 QA; 6/10; worst -0.2500; mean +0.0716 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p03_task_rule` | pareto_prompt | yes | `excluded` | 24 QA; 8/10; worst -0.1428; mean +0.2052 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p04_numeric_guard` | pareto_prompt | yes | `excluded` | 24 QA; 7/10; worst -0.4286; mean +0.0256 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p05_single_answer` | pareto_prompt | yes | `excluded` | 24 QA; 6/10; worst -0.7143; mean +0.1366 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p06_abc` | pareto_prompt | yes | `excluded` | 24 QA; 8/10; worst -0.2857; mean +0.2868 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p07_abd` | pareto_prompt | yes | `excluded` | 24 QA; 7/10; worst -0.1430; mean +0.0383 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p08_acde` | pareto_prompt | yes | `excluded` | 24 QA; 7/10; worst -0.5714; mean +0.1089 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p09_bce` | pareto_prompt | yes | `excluded` | 24 QA; 6/10; worst -0.7143; mean +0.0001 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p10_abcde` | pareto_prompt | yes | `excluded` | 24 QA; 6/10; worst -0.2859; mean +0.0959 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `pareto_p11_abce` | pareto_prompt | yes | `excluded` | 24 QA; 6/10; worst -0.5713; mean +0.0144 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r2_01_minimal` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -0.6207; mean +0.0110 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r2_02_mc_stable` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -0.7293; mean -0.0354 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r2_03_tf_boundary` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -0.5517; mean -0.0317 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r2_04_oe_old_contract` | routed_prompt | yes | `excluded` | 96 QA; 7/10; worst -0.3810; mean +0.2053 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r2_05_oe_contract_coverage` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -1.0690; mean -0.0725 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r2_06_full_routed` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -1.0345; mean -0.0772 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r3_01_mc_f0_safe` | routed_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `route_r3_02_mc_f1_safe` | routed_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `route_r3_03_mc_f0_tf_p0` | routed_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `route_r3_04_mc_f1_tf_p0` | routed_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `route_r3_05_oe_q1` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -0.3777; mean +0.1047 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r3_06_oe_q2` | routed_prompt | yes | `excluded` | 96 QA; 6/10; worst -0.5862; mean +0.0706 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r3_07_oe_q3` | routed_prompt | yes | `excluded` | 96 QA; 7/10; worst -0.5517; mean +0.0979 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `route_r3_08_historical` | routed_prompt | yes | `excluded` | 96 QA; 7/10; worst -0.3810; mean +0.0450 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `lit_r1_01_mc_semantic_bind` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r1_02_mc_pointwise` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r1_03_mc_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r1_04_tf_minimal` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r1_05_tf_clause` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r1_06_tf_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r1_07_oe_direct` | literature_prompt | yes | `excluded` | 24 QA; 6/10; worst -0.2857; mean -0.0565 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `lit_r1_08_oe_re2` | literature_prompt | yes | `advanced_full284` | 24 QA; 9/10; worst -0.1429; mean +0.0200 | Passes the frozen wide screen and has no later broader contradiction. |
| `lit_r1_09_triplet_minimal` | literature_prompt | yes | `excluded` | 24 QA; 3/10; worst -0.3333; mean -0.0396 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `lit_r1_10_triplet_re2` | literature_prompt | yes | `advanced_full284` | 24 QA; 9/10; worst -0.1429; mean +0.2721 | Passes the frozen wide screen and has no later broader contradiction. |
| `lit_r1_11_decoupled` | literature_prompt | yes | `excluded` | 24 QA; 4/10; worst -0.4285; mean +0.0171 | Did not meet the wide 9/10, worst>=-0.15, positive-mean, OE-improvement screen. |
| `lit_r1_12_mc_bind_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r2_01_mc_tf_re2_safe` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_01_mc_pointwise_qual` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_02_mc_semantic_qual` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_03_mc_salient_aftermath` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_04_tf_re2_verdict` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_05_tf_re2_qual_verdict` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_06_joint_pointwise_tf_qual` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r3_07_joint_semantic_tf_qual` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r4_01_tf_neg_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r4_02_joint_semantic_tf_neg_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r4_03_tf_nonanomaly_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r4_04_joint_semantic_tf_nonanomaly_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r4_05_tf_neg_prefix` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r4_06_joint_semantic_tf_neg_prefix` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r5_01_mc_structured_guard` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r5_02_mc_status_then_shape` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r5_03_joint_structured_tf_neg_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `lit_r5_04_joint_status_tf_neg_re2` | literature_prompt | no | `excluded` | — | OE prompt branch is exact Baseline and cannot satisfy the frozen OE-change requirement. |
| `wo_local_hint` | released_ablation_control | yes | `excluded` | — | Released ablation control; paper and reproduction evidence show hint/window removal is harmful, not a prompt-improvement candidate. |
| `wo_fixed_hint` | released_ablation_control | yes | `excluded` | — | Released ablation control; paper and reproduction evidence show hint/window removal is harmful, not a prompt-improvement candidate. |
| `wo_windows` | released_ablation_control | yes | `excluded` | — | Released ablation control; paper and reproduction evidence show hint/window removal is harmful, not a prompt-improvement candidate. |

The machine-readable JSON retains every historical evidence row, all ten deltas, and representative prompt hashes.
