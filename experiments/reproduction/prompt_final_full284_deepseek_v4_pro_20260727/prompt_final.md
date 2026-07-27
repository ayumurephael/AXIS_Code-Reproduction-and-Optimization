# AXIS Prompt 最终 full-284 实验结果

## 结论

两个宽口径候选均未通过最终 full-284 门槛；本轮没有可宣称为全面提升的 Prompt。已确认失败的 Prompt 没有重复运行。

这里已彻底删除 `zero correct→wrong` 门槛；MC/TF 决策迁移只可用于诊断，不参与 PASS/FAIL。

## 正式 Table-I 十项结果

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| AXIS Baseline | 4.0400 | 4.0708 | 3.9681 | 3.1486 | 3.0328 | 2.7818 | 3.7116 | 3.3624 | 3.3769 | 3.3407 |
| OE-only RE2 | 4.0400 | 4.0708 | 3.9681 | 3.0982 | 3.0000 | 2.6670 | 3.7157 | 3.3624 | 3.3769 | 3.3407 |
| MC/OE/TF RE2 | 4.0135 | 4.0745 | 3.8714 | 3.0982 | 3.0000 | 2.6670 | 3.7157 | 3.5664 | 3.5923 | 3.5275 |

| Prompt | 10项非降 | 严格提升项数 | OE严格提升项数 | 最终判定 |
|---|---:|---:|---:|---|
| `lit_r1_08_oe_re2` | 7/10 | 1 | 1 | FAIL |
| `lit_r1_10_triplet_re2` | 5/10 | 5 | 1 | FAIL |

## 相对 Baseline 的绝对差值

| Prompt（绝对差值） | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| OE-only RE2 | +0.0000 | +0.0000 | +0.0000 | -0.0504 | -0.0328 | -0.1148 | +0.0041 | +0.0000 | +0.0000 | +0.0000 |
| MC/OE/TF RE2 | -0.0265 | +0.0036 | -0.0967 | -0.0504 | -0.0328 | -0.1148 | +0.0041 | +0.2040 | +0.2154 | +0.1868 |

## 相对 Baseline 的变化比例

| Prompt（相对变化（%）） | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| OE-only RE2 | +0.0000 | +0.0000 | +0.0000 | -1.6018 | -1.0821 | -4.1259 | +0.1098 | +0.0000 | +0.0000 | +0.0000 |
| MC/OE/TF RE2 | -0.6551 | +0.0896 | -2.4376 | -1.6018 | -1.0821 | -4.1259 | +0.1098 | +6.0660 | +6.3785 | +5.5922 |

相对比例按 `(候选−Baseline)/Baseline×100%` 计算；最终判定使用未四舍五入浮点值和 `1e-6` 容差。

## 评测口径与完整性

- 数据：作者发布 `AXIS_qa_test` 的全部 142 个 series、284 QA；不是论文 Table I 的 `paper140`。
- 推理：发布 checkpoint；远端授权 GPU；series batching；beam size 5；`max_new_tokens=1000`；`skip_loss=true`。
- Judge：仅 `deepseek-v4-pro`；Baseline 与候选采用同一 G-Eval rubric、同一评分读出。
- 发布 checkpoint SHA-256：`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。
- GPU 推理源提交：`c113d3080375f05cec2b73702e8c1d2f5bcffa0f`；后续提交只增加精确复用、审计、报告与 Judge 断线恢复代码，未改变正式 Prompt。
- 组装后预测 `852` 行；评分 `2001` 行；每个 Prompt 284 个回答。
- Baseline 题型分布：`{'open_ended': 99, 'multiple_choice': 94, 'true_false': 91}`。
- 原始 Judge 方法：`final_score_top_logprobs=1322`，`exact_sample_mean_20=12`；精确复用组装后方法分布：`{'exact_sample_mean_20': 19, 'final_score_top_logprobs': 1982}`。
- 预测 SHA-256：`d97197cf775350eca0ace0670d41bc9e685c1704bdc8f5d575e702af2c8cdb5e`。
- 评分 SHA-256：`b15681500f111f74eb9f2cd2ca7e8429af0cdbd078490e9889a037fb8330818b`。

## 最终实验 Prompt 路由

| Prompt | MC | OE | TF |
|---|---|---|---|
| AXIS Baseline | 原论文 Prompt | 原论文 Prompt | 原论文 Prompt |
| `lit_r1_08_oe_re2` | 原论文 Prompt（精确复用） | RE2 Prompt | 原论文 Prompt（精确复用） |
| `lit_r1_10_triplet_re2` | RE2 Prompt | RE2 Prompt | RE2 Prompt |

`lit_r1_08_oe_re2` 的 OE 与 `lit_r1_10_triplet_re2` 的 OE Prompt 完全相同；前者 MC/TF 与 Baseline 完全相同。因此只生成两套不同的原始回答，再按题型精确复用，避免把随机重复生成误报为 Prompt 效果。

## 具体回答质量改善案例

没有候选通过最终十项指标门槛，因此不能把局部高分回答包装成“全面提升 Prompt”的成功案例。

## 全部正式实验 Prompt

下面是去除 Python f-string 固有缩进后的等价完整模板。`{local_hint_tokens}` 与 `{fixed_hint_tokens}` 的数量及注入位置保持作者实现不变。

### 原 AXIS Baseline Prompt

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
{question}
```

### RE2 Prompt

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
{question}

Read the question again:
{question}
```

RE2 唯一新增行为是：在原问题之后加入 `Read the question again:` 并原样重复问题；未加入 Evidence Contract、数值解释、新角色名或额外输出协议。

## 完整 69-mode 筛选清单

以下清单包含 Baseline、65 个 Prompt 候选和 3 个发布消融对照。已确认失败项依据已有审计结果排除，没有重复运行。

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
