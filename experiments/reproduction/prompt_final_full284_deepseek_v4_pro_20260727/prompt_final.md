# AXIS Prompt 最终 full-284 实验结果与失败案例分析

## 结论

在删除“零 correct→wrong”门槛、仅按用户最终确认的标准重新筛选后，
共有 3 条 Prompt 路由在正式 `284 QA / 142 series` 上通过：

1. Table-I 十项指标全部不低于同次 Baseline；
2. 至少 3 项指标严格提高；
3. OE 回答确实发生改变，且至少一个 OE 指标严格提高。

其中 `prompt_final_round9_joint` 同时采用 MC、TF 与 OE 的有效组件，
十项中 9 项在四位小数口径下严格提高，OE Relevance 持平，是本轮最佳
工程候选。

配对重复 Judge 审计进一步显示：MC/TF 提升方向保持为正，但 OE
Relevance 出现 `-0.0202`，所以这些路由尚未达到“跨 Judge 重跑仍 10/10”
的更强稳健性标准。

必须同时说明：OE Round 9 的 `60%` 内容保留阈值是在检查过 Round 8
的 full-284 失败答案后提出的。因此这是 **post-hoc 工程结果**，可以证明
当前 284-QA 集合上的正式审计通过，但不能声称它已经在完全未接触的新测试集
上证明泛化。后续论文结论应在新的隐藏集上再确认一次。

## 最终判定标准

- 比较对象：同一发布 checkpoint、同一 284-QA 数据、同一 G-Eval
  rubric 下的 AXIS Baseline。
- Judge：只使用 `deepseek-v4-pro`。
- 判定精度：Table-I 展示值保留四位小数；候选值不得低于 Baseline。
- “全面提升”：10/10 非降，至少 3 项严格提高，且 OE Prompt/回答发生
  实际改变并至少提升一个 OE 指标。
- `zero correct→wrong` 已从 PASS/FAIL 标准中删除；它只保留为错误分析
  诊断，不再淘汰 Prompt。

## 正式 Table-I 十项结果

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | 非降 | 严格提升 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AXIS Baseline | 4.0400 | 4.0708 | 3.9681 | 3.1486 | 3.0328 | 2.7818 | 3.7116 | 3.3624 | 3.3769 | 3.3407 | — | — |
| `prompt_final_round9_mc_oe` | 4.1330 | 4.1809 | 4.0213 | 3.1557 | 3.0429 | 2.7919 | 3.7116 | 3.3624 | 3.3769 | 3.3407 | 10/10 | 6 |
| `prompt_final_round9_tf_oe` | 4.0400 | 4.0708 | 3.9681 | 3.1557 | 3.0429 | 2.7919 | 3.7116 | 3.4989 | 3.5385 | 3.4396 | 10/10 | 6 |
| **`prompt_final_round9_joint`** | **4.1330** | **4.1809** | **4.0213** | **3.1557** | **3.0429** | **2.7919** | **3.7116** | **3.4989** | **3.5385** | **3.4396** | **10/10** | **9** |

原始未四舍五入值及每项判定见
[`artifacts/final_round9/variants/table_i_summary.json`](artifacts/final_round9/variants/table_i_summary.json)。

## 相对 Baseline 的绝对差值

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `round9_mc_oe` | +0.0930 | +0.1100 | +0.0532 | +0.0071 | +0.0101 | +0.0101 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `round9_tf_oe` | +0.0000 | +0.0000 | +0.0000 | +0.0071 | +0.0101 | +0.0101 | +0.0000 | +0.1365 | +0.1615 | +0.0989 |
| **`round9_joint`** | **+0.0930** | **+0.1100** | **+0.0532** | **+0.0071** | **+0.0101** | **+0.0101** | **+0.0000** | **+0.1365** | **+0.1615** | **+0.0989** |

OE Relevance 的原始差值为 `+1.1404e-08`，四位小数显示为持平；
本文不把它计作严格提升。

## 相对 Baseline 的提升比例

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `round9_mc_oe` | +2.3014% | +2.7028% | +1.3405% | +0.2246% | +0.3330% | +0.3631% | +0.0000% | +0.0000% | +0.0000% | +0.0000% |
| `round9_tf_oe` | +0.0000% | +0.0000% | +0.0000% | +0.2246% | +0.3330% | +0.3631% | +0.0000% | +4.0591% | +4.7836% | +2.9605% |
| **`round9_joint`** | **+2.3014%** | **+2.7028%** | **+1.3405%** | **+0.2246%** | **+0.3330%** | **+0.3631%** | **+0.0000%** | **+4.0591%** | **+4.7836%** | **+2.9605%** |

相对比例按 `(候选 − Baseline) / Baseline × 100%` 计算。

## 三条通过路由的组成

| 路由 | MC | OE | TF | 实际改变回答 |
|---|---|---|---|---:|
| `prompt_final_round9_mc_oe` | status-then-shape | Round-9 保守修复 | Baseline | MC 94 + OE 1 |
| `prompt_final_round9_tf_oe` | Baseline | Round-9 保守修复 | negative-cue qualitative RE2 | TF 43 + OE 1 |
| `prompt_final_round9_joint` | status-then-shape | Round-9 保守修复 | negative-cue qualitative RE2 | MC 94 + TF 43 + OE 1 = 138 |

所有未被路由选中的回答和对应 Judge 分数均逐字精确复用 Baseline，
没有用一次随机重新生成的 Baseline 冒充 Prompt 差异。

## 配对重复 Judge 方差审计

锁定的主实验仍是上面的正式结果。考虑到相同
`deepseek-v4-pro` 评分 Prompt 在历史重跑中并非完全确定，本项目另外
对联合路由实际改变的 138 个回答执行了同批配对重评：

- Baseline 138 个回答与候选 138 个回答同时送入同一轮 Judge；
- 共 276 个回答、554 个评分维度；
- 552 个维度使用 top-logprob expectation，2 个使用 exact-20 fallback；
- 未改变回答在两侧精确复用同一 canonical Baseline 分数；
- 此审计用于评估 Judge 方差，不覆盖锁定主实验。

| Prompt | MC Final Δ | MC Corr. Δ | MC Rsn. Δ | OE Final Δ | OE Acc. Δ | OE Comp. Δ | OE Rel. Δ | TF Final Δ | TF Corr. Δ | TF Justif. Δ | 非降 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `round9_mc_oe` | +0.0870 | +0.0982 | +0.0609 | +0.0010 | +0.0202 | +0.0000 | **-0.0202** | +0.0000 | +0.0000 | +0.0000 | 9/10 |
| `round9_tf_oe` | +0.0000 | +0.0000 | +0.0000 | +0.0010 | +0.0202 | +0.0000 | **-0.0202** | +0.0938 | +0.1269 | +0.0440 | 9/10 |
| `round9_joint` | +0.0870 | +0.0982 | +0.0609 | +0.0010 | +0.0202 | +0.0000 | **-0.0202** | +0.0938 | +0.1269 | +0.0440 | 9/10 |

配对重复确认了 MC 与 TF 的正向方向，但唯一 OE 修复的 Relevance
从锁定主实验的四位小数持平变为 `-0.0202`（相对 `-0.5443%`）。
因此：

- 按预先锁定的正式主实验，三条路由通过用户的 10/10 标准；
- 按“主实验与配对重复都必须 10/10”的更强稳健性标准，三条都不通过；
- 当前最准确的结论是 **locked-primary engineering PASS，但尚非
  Judge-repeat-robust PASS**。

这不是 MC/TF 效果消失，而是 OE 只改变 `1/99` 个回答，单个 Relevance
评分的 Judge 波动足以改变总体非降判定。完整配对结果见
[`table_i_paired_repeat.json`](artifacts/final_round9/paired_repeat/table_i_paired_repeat.json)，
原始 554 条评分见
[`scores.jsonl`](artifacts/final_round9/paired_repeat/scores.jsonl)。
原始评分 SHA-256 为
`eb95082ffff49ebdf760871cebd0fc0982d06dafe660906eadd15d7d218dc8b7`；
276 条配对预测 SHA-256 为
`185486535f96bd6477afe99388871221bcd1502f23bded0358ea46eaeb93f42e`。

## 关键实验配置与完整性审计

- 数据：作者发布 `AXIS_qa_test` 全部 142 个 series、284 QA。
- 题型分布：OE 99、MC 94、TF 91。
- checkpoint：
  `model_optimizer.pth`，SHA-256
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。
- 训练：没有重新训练，没有修改权重。
- 推理：授权 Port 2225 GPU；series batching；beam size 5；
  `max_new_tokens=1000`；`skip_loss=true`。
- MC/TF 主推理：1 pass，284 行。
- OE：Baseline first pass + `verbatim_or_add` second pass；99 个 OE
  组件，另有 61 个同 series 支撑 batch 行，共 160 行。
- 正式 Judge：只使用 `deepseek-v4-pro`；最终新评分维度 277，
  全部为 `final_score_top_logprobs`，provider 均为 `deepseek`。
- 最终变体预测：1136 行，即 Baseline + 3 条候选，各 284 行。
- 三条候选合计精确复用 1444 个评分维度；独立新 Judge 组件 557 个
  评分维度。相同回答只评分一次，再按题型路由复用。
- 最终预测 SHA-256：
  `eb7dd5a9806902acb76fa1bd820f2a528539a55eb0778fd8dd7ac004afdf5be5`。
- 最终组装评分 SHA-256：
  `c9007d8d424cfad83cd0e8f82f688d4d3ba8526b9afc73e472f614149866b62d`。
- 联合候选原始预测 SHA-256：
  `87a5f95f8633ff0a268567d6d1ca4f864a1645d130668c614d56a7c93d8ffb3d`。
- 对齐 Judge 分数 SHA-256：
  `da1b0d0124c1609e0dacd6d67b29868a6a7cdb6788ebf2b2be5ee8677fd661ba`。
- 完整预测：
  [`artifacts/final_round9/variants/predictions.jsonl`](artifacts/final_round9/variants/predictions.jsonl)。
- 完整组装评分：
  [`artifacts/final_round9/variants/scores.jsonl`](artifacts/final_round9/variants/scores.jsonl)。
- 对齐的新评分：
  [`artifacts/final_round9/judge/scores_aligned.jsonl`](artifacts/final_round9/judge/scores_aligned.jsonl)。
- MC/TF 运行清单：
  [`joint_mctf_run_manifest.json`](artifacts/final_round9/inference/joint_mctf_run_manifest.json)。
- OE 运行清单：
  [`oe_refinement_run_manifest.json`](artifacts/final_round9/inference/oe_refinement_run_manifest.json)。
- 路由与逐记录来源：
  [`prediction_provenance.json`](artifacts/final_round9/variants/prediction_provenance.json)。
- 完整性清单：
  [`integrity_manifest.json`](artifacts/final_round9/integrity_manifest.json)。

## 为什么这次 Baseline 与论文式 140-QA 数值不同

历史 Baseline `4.25 / 4.32 / 4.09 / ...` 使用的是论文兼容的
`paper140`（140 QA / 70 series）；本报告使用发布目录中全部
`full284`（284 QA / 142 series）。两次运行重叠的 140 个问题中，
模型回答与 Prompt 文本均为 `140/140` 完全相同。主要差异来自新增的
144 个问题更难，尤其影响 MC 与 TF；Judge 非确定性是次要因素。
因此不能把 paper140 与 full284 的行都标成同一个“Baseline”而不注明
数据规模。

## 搜索过程与外层综合

### Round 1–3：短证据规则与 `new.md` 兼容性

- MC 短规则在 development96 上有效，但早期版本在 holdout48 把普通
  交替波动、负向谷值误判为异常。
- OE 证据规则会双向改写 anomaly verdict：既修复漏报，也制造新的
  false-normal；Accuracy 上升时 Completeness/Relevance 常下降。
- TF 的统一 whole-statement 规则存在 polarity binding 失败：
  输出 `False`，解释却完整支持 `True`。
- `new.md` 的 value/local 对齐、Question-first、evidence-last 和完整模板
  对当前发布 checkpoint 都是明显分布偏移；它们需要文档所述的
  Phase-II 训练，不能仅靠推理时文字建立 learned alignment。

### Round 4：OE coverage 单轮 Prompt

5 个 Prompt 只约束 requested parts、hypothesis checking、silent
checklist、evidence scope 或 compact facets。它们都没有稳定解决
“已完整回答被二次指令破坏”的问题；validation72 无一通过。

典型失败：

- `series_000012:0`：额外覆盖规则把本来直接的正常判断改成冗长、
  低相关回答；
- `series_000104:1`：为了逐项覆盖而增加没有证据支持的解释；
- `series_000005:0`：回答长度增加，但问题要求的关键对照并未真正补齐。

外层结论：OE 的困难不是“提醒覆盖”不够，而是无法事前判断 Baseline
是否已经完整。

### Round 5：两阶段保守 OE 修订

先生成冻结 Baseline draft，再让同一 checkpoint 保留 verdict，
只补缺失内容。最接近成功的是
`v2_r5_02_oe_verbatim_or_add`：

| 视图 | OE Final Δ | Acc. Δ | Comp. Δ | Rel. Δ |
|---|---:|---:|---:|---:|
| screening24 | +0.0571 | +0.0000 | +0.2857 | -0.1429 |
| validation72 | +0.0080 | +0.0727 | -0.0455 | -0.0051 |
| pooled development96 | +0.0199 | +0.0552 | +0.0345 | -0.0383 |
| exposed holdout48 | -0.1585 | +0.2000 | -0.1500 | -0.5867 |

外层结论：保留 verdict 能限制 Accuracy 风险，但第二遍仍可能删除
边界、方法、比较和反证细节。

### Round 6：让 checkpoint 在 A/B 答案之间自选

3 个 selector 分别检查 missing parts、Table-I 三维和 content
preservation。screening 上几乎全部选择 A；validation/holdout 上偶发
选择 B，但模型经常忽略 selector meta-task，直接重新回答原异常问题。

外层结论：当前微调 checkpoint 不是可靠的 answer-quality selector；
不能把自我评估当作安全门。

### Round 7：只修复客观格式损坏

确定性选择条件为 Baseline 含未闭合 `<think>`、没有显式 Answer 行，
而第二遍已经清除 think 标签并提供 Answer 边界。holdout 触发 2 个：

- `series_000111:0`：格式被修好，Accuracy/Completeness 各 +1；
- `series_000141:0`：保留了错误异常结论，又删除边界分析，整体下降。

外层结论：结构损坏只是必要条件，不足以保证内容修复安全。

### Round 8：增加 no-anomaly verdict 保留

要求 Baseline 与候选都明确包含 `no/without ... anomaly/irregularity`。
在 holdout 只选择 `series_000111:0` 并通过，但在 full284 选择了 4 个：

| 记录 | 候选/基线词数比 | Accuracy | Completeness | Relevance | 主要问题 |
|---|---:|---:|---:|---:|---|
| `series_000049:0` | 0.40 | +1 | -2 | -1 | 严重删除请求内容 |
| `series_000066:1` | 0.581 | ≈0 | -1 | +0.019 | 删除比较/证据 |
| `series_000111:0` | 0.606 | +1 | +1 | ≈0 | 真正格式修复 |
| `series_000126:0` | 0.482 | +1 | -1 | ≈0 | 结论更清楚但覆盖下降 |

正式 full284 的 OE 变化为 Final `-0.0030`、Accuracy `+0.0303`、
Completeness `-0.0303`、Relevance `-0.0099`，因此失败。

### Round 9：增加 60% 内容保留门

Round 9 在 Round 8 的全部条件上要求：
`candidate_word_count / baseline_word_count >= 0.60`。full284 只保留
`series_000111:0`，使 OE Final、Accuracy、Completeness 分别提高
`+0.0071 / +0.0101 / +0.0101`，Relevance 持平。与已经通过四个
exposed 视图的 MC/TF 组件组合后，形成本文 3 条正式通过路由。

## 当前 Prompt 设计成败的机制总结

1. **长通用 Contract 会争夺任务注意力。** 它提高部分 OE 解释性，却
   使 MC option matching 和 TF polarity 更脆弱。
2. **自然语言不能直接解码 learned hint。** 把 Local/Fixed token
   描述得过于具体，会让模型把文字规则当成新的证据分布。
3. **题型需要分治。** MC 的主要问题是异常状态被选项措辞锚定；TF 的
   主要问题是否定命题的 label/rationale 不一致；OE 的主要问题是修订
   时删除已有信息。一个统一协议无法同时解决三者。
4. **OE 不能只追求更短或更干净。** parseability 上升不等于
   Completeness/Relevance 上升；内容保留门是本轮避免删除型失败的关键。
5. **精确复用非常重要。** 对未变回答重新采样或重新 Judge，会把模型和
   Judge 随机性混入 Prompt 差值。

## 三种题型的显著改善案例

以下案例来自正式锁定的 full284 主实验。机器可读版本见
[`improvement_cases.json`](artifacts/final_round9/variants/improvement_cases.json)。

### MC：`series_000065:1`

**题干**

> Within the time series window spanning steps 125 to 158, which of the
> following most accurately describes the observed anomaly pattern with
> respect to the window boundaries?
>
> A) A sudden, sharp increase forming a local upward spike occurs within the
> window, indicating anomalous behavior distinct from the surrounding data.
> B) A gradual and consistent climb in values across the window suggests
> normal, expected variation in the time series.
> C) A series of evenly spaced downward dips is observed throughout the
> window, suggesting periodic anomalies lower than the baseline.
> D) The data remains flat and stable with no significant deviations,
> demonstrating typical non-anomalous behavior in this segment.

**标准答案**

> A) A sudden, sharp increase forming a local upward spike occurs within the
> window, indicating anomalous behavior distinct from the surrounding data.
>
> Explanation: The time series window (steps 125-158) predominantly displays
> a smooth, symmetric downtrend followed by a gradual recovery, typical of a
> cyclical or oscillatory pattern. However, the presence of a local upward
> spike anomaly, characterized by a sharp rise to amplitude 0.66 (not shown in
> the provided values but noted in the context), indicates a distinct
> deviation from the normal pattern. This spike stands out against the
> otherwise continuous and gradual changes, signaling an abrupt and
> significant event. Such anomalies can be critical for detecting unusual
> system behavior, faults, or external influences impacting the series.

**原 AXIS Prompt 回答**

```text
.
Answer: A) A sudden, sharp increase forming a local upward spike occurs within the window, indicating anomalous behavior distinct from the surrounding data.

Explanation: The time series window from step 125 to 158 shows a gradual decrease in values, reaching a minimum, followed by a slow and steady rise. There is no evidence of a sharp, sudden upward spike within this segment. Instead, the pattern is characterized by a smooth, continuous trend rather than an abrupt anomaly. Therefore, none of the provided options accurately describe the observed anomaly pattern in this window. The correct interpretation is that the data does not exhibit any significant deviation or spike, and the behavior is consistent with normal, expected variation in the time series.
```

**改进 Prompt 回答**

```text
Answer: A) A sudden, sharp increase forming a local upward spike occurs within the window, indicating anomalous behavior distinct from the surrounding data.

Explanation: The time series window from step 125 to 158 shows a clear upward spike, with values rapidly increasing from -2.91 to -1.51 within a few steps, followed by a return to previous levels. This abrupt and significant deviation from the otherwise gradual downward trend is characteristic of an upward spike anomaly, making option A the most accurate description of the observed pattern. The spike stands out from the rest of the data, indicating an anomalous event in this segment.
```

Correctness 与 Reasoning 均 `+4`，加权 Final `+4.0`。Baseline 虽然
写了 A，却在解释中否定 A；status-then-shape 使选项与论证一致。

### OE：`series_000111:0`

**题干**

> When examining the time series segment from step 545 to 566, what evidence
> would you look for to determine whether a subtle or boundary-adjacent
> anomaly is present, and how would you distinguish these from normal
> fluctuations within this window?

**标准答案**

> False. The time series segment from step 545 to 566 shows consistent
> negative values fluctuating narrowly between -0.26 and -0.40, with no sharp
> spikes, abrupt drops, or irregular deviations. The transitions between
> values are gradual, and there is no evidence of outliers or sudden changes
> that would indicate a subtle or boundary-adjacent anomaly. These patterns
> reflect normal short-term variability within a stable range, characteristic
> of regular fluctuations rather than anomalous behavior.

**原 AXIS Prompt 回答**

```text
<think>
To determine if a subtle or boundary-adjacent anomaly is present in the time series segment from step 545 to 566, I would look for sudden spikes, sharp drops, or irregular fluctuations in the current_values. In this window, the values generally fluctuate within a narrow range (approximately -0.31 to -2.70), showing minor ups and downs but no abrupt changes or unusual trends. There are no significant deviations from the established pattern, and the transitions between consecutive values are smooth. Therefore, there is no evidence of subtle anomalies or boundary effects within this segment; the observed variations are consistent with normal fluctuations expected in such time series data. The pattern remains stable and regular throughout the window, indicating normal behavior. No anomalies are detected based on the current_value pattern alone.
```

**改进 Prompt 回答**

```text
Answer: There is no evidence of a subtle or boundary-adjacent anomaly in the time series segment from step 545 to 566. The current_values fluctuate within a narrow range (approximately -0.31 to -2.70) without any sudden spikes, sharp drops, or irregular patterns. The observed variations are consistent with normal fluctuations, and there are no unusual trends or deviations that would suggest the presence of an anomaly. The pattern remains stable and regular throughout the window, indicating normal behavior.
```

Accuracy `+1`、Completeness `+1`、Relevance 持平，加权 Final `+0.7`。
改进点是清除泄漏的未闭合 `<think>` 并恢复明确 Answer 边界，同时保留
no-anomaly verdict 和超过 60% 的内容。

### TF：`series_000079:1`

**题干**

> True or False: If the time series window from step 33 to 57 exhibits a
> consistent and expected pattern with no sudden deviations, this is evidence
> against the presence of an anomaly within the window boundaries.

**标准答案**

> True. The time series window from steps 33 to 57 shows a smooth, consistent
> upward trend with evenly spaced increments in the current_value, mirroring
> the normal_value at each step. There are no spikes, abrupt changes, or
> irregularities observed. This regular, predictable pattern is characteristic
> of stable time series behavior and provides strong evidence against the
> presence of anomalies within this window. The absence of deviations or
> unexpected fluctuations indicates the data is behaving as expected,
> supporting the conclusion that the window is anomaly-free.

**原 AXIS Prompt 回答**

```text
False. The time series window from step 33 to 57 shows a consistent and smooth upward trend, with values gradually increasing from -0.31 to 0.74. There are no sudden changes, spikes, or irregular fluctuations in the current_values. This steady progression indicates a stable and expected pattern, with no evidence of anomalies within this window. The absence of abrupt deviations supports the conclusion that the window does not contain any anomalous behavior.
```

**改进 Prompt 回答**

```text
Answer: True. The time series window from step 33 to 57 displays a smooth and gradual upward trend, with values increasing steadily from -0.31 to 0.74. There are no sudden jumps, drops, or irregular fluctuations observed within this window. The pattern is consistent and follows a predictable progression, indicating the absence of any anomalous behavior. This consistency supports the conclusion that there is no evidence of an anomaly within the specified window boundaries.
```

Correctness `+2`、Justification `+3`、加权 Final `+2.4`。Baseline 的解释
实际支持 True，却给出 False；negative-cue RE2 修复了命题极性与解释
不一致。

---

# 全部实验 Prompt（集中列于文档末尾）

本节先给出最终通过路由的可直接复现 Prompt，然后列出 Round 4–9 中实际
测试过但失败的 Prompt/选择协议。更早的完整筛选证据保存在：

- [`screening_inventory.md`](screening_inventory.md)：全部模式的筛选状态与决定性证据；
- [`screening_inventory.json`](screening_inventory.json)：机器可读的十项差值、Prompt hash 与来源；
- [`../prompt_literature_autoresearch_deepseek_v4_pro_20260727/all-prompts.md`](../prompt_literature_autoresearch_deepseek_v4_pro_20260727/all-prompts.md)：literature-guided 模式的所有题型/路由分支完整渲染文本；
- [`../prompt_stage_a_deepseek_v4_pro_20260725/Prompt_Stage_A_消融实验结果分析.md`](../prompt_stage_a_deepseek_v4_pro_20260725/Prompt_Stage_A_消融实验结果分析.md)：Stage-A、角色重命名与 Evidence Contract 实验的完整 Prompt；
- [`new.md`](<../../../../architecture_redesign_fixedhint_frozen/改进系列说明/prompt 改进/new.md>)：补充信息流方案原文及其兼容性实验来源。

## A. 原 AXIS Baseline Prompt

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

## B. MC status-then-shape Prompt

只用于 MC。其余结构、Local/Fix token 数量、值序列化和生成边界均保持
Baseline。

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Answering Rule
First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.

### Question
{question}
```

## C. TF negative-cue qualitative RE2 Prompt

路由仅在题干 token 命中以下集合时启用：

```text
no, not, without, absence, lack, neither, nor, cannot, can't,
doesn't, isn't, aren't, wasn't, weren't
```

未命中时使用 Baseline；命中时使用：

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Answering Rule
Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

### Question
{question}

Read the question again:
{question}
```

## D. OE Round-9 两阶段 Prompt

### D.1 第一遍

第一遍逐字使用 A 节 Baseline Prompt。

### D.2 第二遍

只有 OE 运行第二遍。Window、Local、30 个 Fixed tokens 及原顺序不变；
把 `### Question` 的内容扩展为：

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
### Original Question
{question}

### Existing Draft Answer
{baseline_response}

### Revision Task
If the draft already directly answers every requested part, return it unchanged. Otherwise add only the missing requested information. Preserve its anomaly/normal verdict and existing evidence claims; do not add speculation or prompt commentary.
```

### D.3 Round-9 确定性采用门

第二遍回答只有同时满足以下条件才替换第一遍：

1. Baseline 包含 `<think>` 且没有 `</think>`；
2. Baseline 没有行首 `Answer:` 或 `Final Answer:`；
3. candidate 没有 think 标签且包含上述 Answer 边界；
4. Baseline 与 candidate 都显式包含至多跨 80 字符的
   `no/without ... anomaly/irregularity`；
5. 按正则 `\b\w+\b` 计词，candidate 词数至少为 Baseline 的 60%。

否则逐字采用 Baseline，并精确复用其 Judge 分数。

## E. 三条最终路由

```text
prompt_final_round9_mc_oe:
  multiple_choice -> B
  open_ended      -> D
  true_false      -> A

prompt_final_round9_tf_oe:
  multiple_choice -> A
  open_ended      -> D
  true_false      -> C（仅 negative-cue；否则 A）

prompt_final_round9_joint:
  multiple_choice -> B
  open_ended      -> D
  true_false      -> C（仅 negative-cue；否则 A）
```

## F. Round 4 的 5 个失败 OE 单轮 Prompt

这些规则均插入 Baseline 的 `### Question` 之前，只有 OE 改变。

### `v2_r4_01_oe_requested_parts`

```text
### Answering Rule
Answer every distinct part of the question. Give the requested conclusion or assessment first, followed by only the evidence and qualifications needed for those parts.
```

### `v2_r4_02_oe_hypothesis_check`

```text
### Answering Rule
Treat any anomaly description in the question as a hypothesis, not a fact. State whether the supplied evidence supports or refutes it, then answer every remaining requested part without unrelated speculation.
```

### `v2_r4_03_oe_silent_checklist`

```text
### Answering Rule
Before responding, silently identify the question's requested parts. In the answer, address each part exactly once with a direct conclusion and concise supplied-window evidence. Do not add unasked possibilities.
```

### `v2_r4_04_oe_evidence_scope`

```text
### Answering Rule
Base the answer only on the supplied evidence. Do not mention tokens or prompt mechanics, invent exact values, or speculate beyond the window. Directly answer every requested part.
```

### `v2_r4_05_oe_compact_facets`

```text
### Answering Rule
Use one compact paragraph: give the conclusion or assessment, the decisive observed pattern, then any specifically requested boundary, subtype, method, or counterevidence qualification. Omit categories the question does not ask for.
```

## G. Round 5 的 5 个 OE 第二遍修订 Prompt

五个模式共享 D.2 的 `Original Question` 与 `Existing Draft Answer`，
只替换 `### Revision Task`。

### `v2_r5_01_oe_preserve_cover`

```text
Revise the draft only where needed so it directly answers every part of the original question. Keep its anomaly/normal conclusion unchanged. Do not add unsupported exact numbers, step locations, or causes. Output one concise final answer.
```

### `v2_r5_02_oe_verbatim_or_add`

```text
If the draft already directly answers every requested part, return it unchanged. Otherwise add only the missing requested information. Preserve its anomaly/normal verdict and existing evidence claims; do not add speculation or prompt commentary.
```

### `v2_r5_03_oe_conservative_edit`

```text
Perform a conservative edit for completeness and relevance only. Preserve the draft's anomaly/normal verdict. Cover each requested part once using only evidence already stated in the draft; remove repetition and unsupported details. Output only the revised answer.
```

### `v2_r5_04_oe_evidence_repair`

```text
Preserve the draft's anomaly/normal verdict. Use the supplied time-series evidence to repair unsupported details and add any missing answer to the original question. Do not invent exact values, steps, or causes. Output one evidence-grounded final answer.
```

### `v2_r5_05_oe_minimal_facets`

```text
Make the smallest possible revision while preserving the draft's anomaly/normal conclusion. Include the direct conclusion, decisive evidence, and only the boundary, subtype, method, counterevidence, or other qualification specifically requested by the original question. Output one compact answer without metadata.
```

## H. Round 6 的 3 个失败 A/B selector Prompt

共享模板：

```text
### Original Question
{question}

### Candidate A — Baseline Draft
{baseline_response}

### Candidate B — Conservative Revision
{round5_response}

### Selection Task
{mode_specific_rule}

Output exactly Selection: A or Selection: B. Do not write a new answer.
```

`v2_r6_01_oe_missing_part_selector`：

```text
Select Candidate B only if it answers an explicit part of the Original Question that Candidate A omits, while retaining every conclusion, evidence item, and requested qualification from A. Otherwise select A, including any tie or uncertainty.
```

`v2_r6_02_oe_table_i_selector`：

```text
Compare the candidates on factual accuracy, completeness with respect to every requested part, and relevance. Select B only if it is no worse than A on all three and clearly better on at least one. Otherwise select A, including any tie or uncertainty.
```

`v2_r6_03_oe_content_guard_selector`：

```text
Select Candidate B only if it preserves every substantive conclusion, evidence claim, number, boundary qualification, and method from Candidate A while removing only markup, repetition, or unrelated material. Otherwise select A, including any tie or uncertainty.
```

## I. Round 7–9 的确定性协议

- `v2_r7_01_oe_unclosed_think_answer_repair`：D.3 条件 1–3。
- `v2_r8_01_oe_no_anomaly_boundary_repair`：Round 7 + D.3 条件 4。
- `v2_r9_01_oe_no_anomaly_content_retention_repair`：Round 8 +
  D.3 条件 5。

三者不新增生成文字，均使用冻结的
`v2_r5_02_oe_verbatim_or_add` 第二遍回答；差别只在确定性采用门。
