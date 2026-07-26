## 大规模 Prompt Pareto 自研究与确认实验（2026-07-26 至 2026-07-27）

### 最终结论

本轮在已发布 checkpoint 上共完成 35 个 prompt 条件/候选的分阶段研究，
但**没有任何候选通过预注册的 untouched holdout 确认**。因此，没有 prompt
被锁定，也没有对 `paper140` 正式集运行候选。这里的“没有运行”不是资源或
网络原因，而是预注册停止条件：若两个冻结候选均未在 holdout 达到 10/10
指标非下降并通过 correct→wrong 防护，则禁止从 holdout 选较不差者、禁止
继续调参，也禁止触碰最终测试集。

最终确认结果为：

| Holdout 模式 | 非下降指标 | 最差差值 | 十指标平均差值 | MC correct→wrong | 是否确认 |
|---|---:|---:|---:|---:|:---:|
| `route_r3_02_mc_f1_safe` | 7/10 | -0.058823 | -0.010482 | 1 | 否 |
| `route_r3_01_mc_f0_safe` | 7/10 | -0.411745 | -0.113527 | 1 | 否 |

`route_r3_02_mc_f1_safe` 的 MC Final/Correctness/Reasoning 相对 holdout
Baseline 分别下降 0.042289、0.058823、0.003710（相对下降
1.094%、1.515%、0.097%）。`route_r3_01_mc_f0_safe` 分别下降
0.370582、0.352940、0.411745（相对下降 9.589%、9.091%、10.769%）。
两个候选的 OE/TF 七个指标与 Baseline 严格相等，因为它们逐条复用了同一
holdout Baseline 的回答和 Judge 分数。

### 研究漏斗与数据隔离

| 阶段 | 数据 | 候选/模式 | GPU 预测 | 新 Judge 维度 | 结果 |
|---|---|---:|---:|---:|---|
| Screening | 24 QA / 12 series | Baseline + 12 | 312 | 715 | 最好 9/10，未成功 |
| Validation | 72 QA / 36 series | Baseline + 5 | 432 | 996 | 无候选通过 gate |
| Development R2 | 暴露后的 96 QA / 48 series | 6 | 576 | 1,326 | 最好 7/10 |
| Development R3 | 同一 development96 | 8 | 288 个新预测并做组件复用 | 261 个新分数；合成 1,989 | 4 个数值 10/10；2 个通过决策防护 |
| Untouched holdout | 48 QA / 24 series | Baseline + 2 | 144 | 179；合成 333 | 两候选均失败 |
| Final `paper140` | 140 QA | 0 | 0 | 0 | 按停止规则保持未触碰 |

数据拆分按 series 互斥。screening 与 validation 在形成新候选后合并为暴露的
development96；holdout 在 R3-01/R3-02 冻结以前从未被读取或评分。
`paper140` 只允许在 holdout 锁定唯一胜者后运行一次，因此本轮没有使用其
候选结果。论文正式 Baseline 仍为：

| MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.255815 | 4.325584 | 4.093021 | 3.085931 | 2.836509 | 2.770042 | 3.745460 | 3.642142 | 3.641666 | 3.642856 |

这些 paper140 数字只作为研究开始前已经锁定的 Baseline，不曾用于本轮
候选选择。

### 关键失败 case 与失败原因

#### R3-02：一致性规则可以修复矛盾，也会冻结错误的初始判断

在 `series_000022:1` 中，Baseline 先选 D，解释又说没有任何选项正确；
R3-02 改为金标 B 并保持解释一致，单题 Final 从 1.000001 升到 4.700000。
这说明 “select exactly one / do not revise” 确实能修复回答后改口。

但在正常窗口 `series_000141:1` 中，Baseline 正确选择 A，R3-02 将普通的
正负交替误读为“急跌并立即恢复”，错误选择异常选项 D，Final 从 4.700000
降到 1.299998。该规则约束的是**选择后的承诺**，并没有提高证据到选项的
初始映射；初始判断错误时，它反而阻止模型自我纠正。一个 +3.70 的纠错和
一个 -3.40 的退化基本抵消，其余若干正确选项又因解释变短而丢失 reasoning
分，最终三个 MC 聚合指标全部略降。

#### R3-01：对 learned fixed tokens 的角色重命名不是语义中性的

在正常窗口 `series_000139:0` 中，Baseline 正确选择 D。R3-01 却选择描述
异常波动的 C，但解释正文明确说这是 normal variability，并以 “no evidence
of anomalous behavior” 结束。Final 从 4.999999 降到 1.300001。

30 个 learned fixed tokens 是在原 `Overall Summary Hints` 标题下共同训练
出来的。把标题改成 `Learned Task Guidance/Shared Task-Control Tokens`
虽然对人更准确，却改变了 checkpoint 解码这些 latent tokens 的文本条件，
造成选项标签与后续理由脱钩。因此不能把 header/角色重命名视为仅格式修改。

#### OE：新增文字主要改变诊断，而不是补充证据覆盖

R2 的旧 Contract 只在 OE Relevance 上提高，但 Accuracy/Completeness 下降；
Q1/Q2/Q3 三个正向 OE 规则在 R3 的 Final/Accuracy/Completeness 分别全面
下降。典型错误包括：否认输入中清晰的 downward event、编造窗口外 step 或
数值、把正常可逆波动改判为异常、以及对方法/证据类问题强行给出诊断闭环。
“写得更长”或“明确列出必须覆盖的内容”没有解决问题，因为新增规则首先改变
了模型的内容选择与异常判断。

#### TF：直接极性规则的平均收益被真实异常退化否决

TF-P0 在 development96 上修复多条否定命题映射错误，但把
`series_000132:1` 的真实 spike + sustained increase 从正确 True 改成
False。包含 TF-P0 的 R3-03/R3-04 虽然数值达到 10/10，仍因这一条
Baseline-correct→candidate-wrong 被预注册防护淘汰。聚合均值上升不能解释
或抵消一个未解决的真实异常漏检。

### 综合机制结论

1. 发布模型对训练 prompt 具有明显共同适配；latent-token 标题、任务语义和
   生成行为不是可独立替换的模块。
2. 长 Contract 同时激活缩放、对齐、异常定义与回答控制，导致 instruction
   competition；修改其中的事实错误仍不能消除竞争。
3. anomaly feature 列表即使以否定形式出现，也会提高这些特征的显著性，
   在正常高波动窗口上增加 false positive。
4. 输出一致性约束只能防止改口，无法保证初始选项正确；错误时会形成
   premature commitment。
5. 题型路由能够把 OE/TF 保护为 Baseline，但本轮没有找到可在 series-disjoint
   holdout 上同时保住 MC 三指标的文本变化。
6. development 的改善由少数灾难性 case 修复驱动时，必须检查对称的灾难性
   新错误；仅看均值或 10/10 的零差组合会高估泛化能力。

### 正式配置与审计汇总

- checkpoint：作者发布权重，SHA-256
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。
- 所有正式推理均在授权 GPU 上执行，series batching、beam size 5、
  `--skip-loss`；没有在本机 CPU 上运行模型推理或训练。
- Judge 仅使用 `deepseek-v4-pro`，主读数为最终 Score token 的 1–5
  概率期望；缺少完整概率分布时使用严格 20 次独立采样均值。
- Holdout 成功推理为 144/144；选择性 Judge 为 179/179，其中 178 个概率
  读数、1 个 exact-20；合成审计为 144 个预测、333 个维度，全部 prompt
  hash 有效，无缺失键、重复键、模型/供应商混用。
- 首次 holdout 三卡加载期间有无关进程抢占 GPU 0，造成零预测 OOM。失败
  日志和 preflight 均保留；进程退出后使用原三卡协议在新目录成功重跑，
  没有混合失败产物。
- 完整研究状态、时间线与结论分别在
  `prompt_autoresearch_deepseek_v4_pro_20260726/research-state.yaml`、
  `research-log.md`、`findings.md`。
- Holdout 完整分析和逐题配对数据在
  `experiments/holdout-confirmation/analysis.md` 与
  `analysis_output/paired_cases.jsonl`。

### 停止条件与后续研究边界

本轮不存在满足“MC/OE/TF 全部 Table-I 指标不低于 Baseline”的确认 prompt。
若继续使用当前 holdout 迭代，就会把确认集变成开发集；若在没有胜者时直接
测试 `paper140`，又会把最终测试集用于选择。二者都会破坏本轮声明的泛化
证据。因此当前可复现结论是**负确认**：在既定数据预算和 checkpoint 下，
已测试的 prompt-only 改动没有全面优于 Baseline。

未来若继续，应先获得一个新的、series-disjoint、此前未查看的选择集，然后
重新预注册候选数量与唯一 final-run 规则；不能在本轮 holdout failure cases
上修改文字后继续报告同一个 holdout 的“提升”。

## 全部 Autoresearch 指标表

下列五组表保留每个候选的绝对分数与相对同阶段 Baseline 的绝对差值；随后
给出逐指标相对百分比。不同 split 的 Baseline 组成不同，因此只能在同一
阶段内比较，不应跨表直接比较绝对数值。

### Screening24：全部绝对分数与绝对差值

#### Screening round 1 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pareto_p01_boundary | 9/10 | -0.286 | +0.253 | +0.375 | +0.375 | +0.375 | +0.129 | -0.286 | +0.286 | +0.429 | +0.289 | +0.333 | +0.222 |
| 2 | pareto_p03_task_rule | 8/10 | -0.143 | +0.205 | +0.387 | +0.500 | +0.125 | +0.056 | -0.000 | +0.282 | -0.143 | +0.289 | +0.333 | +0.222 |
| 3 | pareto_p06_abc | 8/10 | -0.286 | +0.287 | +0.532 | +0.706 | +0.125 | +0.029 | -0.286 | -0.000 | +0.429 | +0.444 | +0.444 | +0.444 |
| 4 | pareto_p07_abd | 7/10 | -0.143 | +0.038 | +0.125 | +0.125 | +0.125 | +0.043 | -0.143 | +0.143 | +0.143 | -0.067 | -0.111 | +0.000 |
| 5 | pareto_p04_numeric_guard | 7/10 | -0.429 | +0.026 | +0.063 | +0.250 | -0.375 | +0.007 | -0.429 | +0.571 | -0.143 | +0.089 | +0.000 | +0.222 |
| 6 | pareto_p08_acde | 7/10 | -0.571 | +0.109 | +0.513 | +0.625 | +0.250 | -0.229 | -0.571 | +0.286 | -0.429 | +0.200 | +0.111 | +0.333 |
| 7 | pareto_p10_abcde | 6/10 | -0.286 | +0.096 | +0.350 | +0.500 | +0.000 | -0.186 | -0.286 | -0.000 | -0.286 | +0.311 | +0.444 | +0.111 |
| 8 | pareto_p11_abce | 6/10 | -0.571 | +0.014 | +0.362 | +0.625 | -0.250 | -0.200 | -0.571 | -0.000 | +0.000 | +0.067 | +0.111 | +0.000 |
| 9 | pareto_p05_single_answer | 6/10 | -0.714 | +0.137 | +0.775 | +1.000 | +0.250 | -0.144 | -0.229 | +0.428 | -0.714 | +0.000 | +0.000 | -0.000 |
| 10 | pareto_p09_bce | 6/10 | -0.714 | +0.000 | +0.388 | +0.500 | +0.125 | -0.414 | -0.429 | -0.143 | -0.714 | +0.244 | +0.333 | +0.111 |
| 11 | pareto_p02_calibration | 5/10 | -0.250 | +0.072 | -0.250 | -0.250 | -0.250 | +0.057 | -0.000 | +0.286 | -0.143 | +0.433 | +0.500 | +0.333 |
| 12 | fixed_role | 5/10 | -0.429 | +0.061 | +0.562 | +0.750 | +0.125 | -0.007 | -0.429 | +0.285 | +0.143 | -0.267 | -0.222 | -0.333 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| pareto_p01_boundary | 4.075 | 4.000 | 4.250 | 2.664 | 2.286 | 2.143 | 3.714 | 3.622 | 3.667 | 3.556 |
| pareto_p03_task_rule | 4.087 | 4.125 | 4.000 | 2.592 | 2.571 | 2.139 | 3.143 | 3.622 | 3.667 | 3.556 |
| pareto_p06_abc | 4.232 | 4.331 | 4.000 | 2.564 | 2.286 | 1.857 | 3.715 | 3.778 | 3.778 | 3.778 |
| pareto_p07_abd | 3.825 | 3.750 | 4.000 | 2.579 | 2.428 | 2.000 | 3.429 | 3.267 | 3.222 | 3.333 |
| pareto_p04_numeric_guard | 3.763 | 3.875 | 3.500 | 2.543 | 2.143 | 2.429 | 3.143 | 3.422 | 3.333 | 3.556 |
| pareto_p08_acde | 4.213 | 4.250 | 4.125 | 2.307 | 2.000 | 2.143 | 2.857 | 3.533 | 3.444 | 3.667 |
| pareto_p10_abcde | 4.050 | 4.125 | 3.875 | 2.350 | 2.286 | 1.857 | 3.000 | 3.644 | 3.778 | 3.444 |
| pareto_p11_abce | 4.063 | 4.250 | 3.625 | 2.336 | 2.000 | 1.857 | 3.286 | 3.400 | 3.444 | 3.333 |
| pareto_p05_single_answer | 4.475 | 4.625 | 4.125 | 2.391 | 2.343 | 2.286 | 2.571 | 3.333 | 3.333 | 3.333 |
| pareto_p09_bce | 4.088 | 4.125 | 4.000 | 2.121 | 2.143 | 1.714 | 2.571 | 3.578 | 3.667 | 3.444 |
| pareto_p02_calibration | 3.450 | 3.375 | 3.625 | 2.593 | 2.571 | 2.143 | 3.143 | 3.767 | 3.833 | 3.667 |
| fixed_role | 4.262 | 4.375 | 4.000 | 2.529 | 2.143 | 2.143 | 3.429 | 3.067 | 3.111 | 3.000 |

##### Paired outcome counts

- `pareto_p01_boundary`: wins=15, ties=0, losses=9, correct→wrong=0, wrong→correct=1.
- `pareto_p03_task_rule`: wins=13, ties=0, losses=11, correct→wrong=2, wrong→correct=5.
- `pareto_p06_abc`: wins=16, ties=0, losses=8, correct→wrong=1, wrong→correct=5.
- `pareto_p07_abd`: wins=11, ties=0, losses=13, correct→wrong=0, wrong→correct=1.
- `pareto_p04_numeric_guard`: wins=12, ties=0, losses=12, correct→wrong=0, wrong→correct=1.
- `pareto_p08_acde`: wins=9, ties=0, losses=15, correct→wrong=1, wrong→correct=5.
- `pareto_p10_abcde`: wins=12, ties=0, losses=12, correct→wrong=1, wrong→correct=5.
- `pareto_p11_abce`: wins=13, ties=0, losses=11, correct→wrong=1, wrong→correct=5.
- `pareto_p05_single_answer`: wins=12, ties=0, losses=12, correct→wrong=1, wrong→correct=5.
- `pareto_p09_bce`: wins=8, ties=0, losses=16, correct→wrong=1, wrong→correct=5.
- `pareto_p02_calibration`: wins=14, ties=0, losses=10, correct→wrong=1, wrong→correct=0.
- `fixed_role`: wins=12, ties=0, losses=12, correct→wrong=1, wrong→correct=1.

### Validation72：全部绝对分数与绝对差值

#### Screening round 1 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pareto_p07_abd | 7/10 | -0.457 | +0.128 | +0.081 | +0.038 | +0.181 | -0.271 | -0.457 | -0.318 | +0.001 | +0.662 | +0.575 | +0.792 |
| 2 | fixed_role | 6/10 | -0.318 | +0.173 | +0.232 | +0.298 | +0.079 | -0.191 | -0.318 | -0.227 | -0.000 | +0.604 | +0.506 | +0.750 |
| 3 | pareto_p01_boundary | 5/10 | -0.182 | +0.063 | +0.042 | +0.077 | -0.038 | -0.109 | -0.182 | -0.091 | -0.045 | +0.304 | +0.173 | +0.500 |
| 4 | pareto_p03_task_rule | 4/10 | -0.727 | -0.221 | +0.042 | +0.077 | -0.038 | -0.509 | -0.727 | -0.727 | +0.000 | -0.121 | -0.202 | +0.000 |
| 5 | pareto_p06_abc | 3/10 | -0.545 | -0.173 | -0.093 | -0.038 | -0.219 | -0.432 | -0.545 | -0.455 | -0.273 | +0.112 | +0.131 | +0.083 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| pareto_p07_abd | 4.020 | 4.000 | 4.065 | 3.179 | 3.270 | 2.773 | 3.547 | 3.608 | 3.652 | 3.542 |
| fixed_role | 4.171 | 4.260 | 3.963 | 3.259 | 3.409 | 2.864 | 3.545 | 3.550 | 3.583 | 3.500 |
| pareto_p01_boundary | 3.981 | 4.039 | 3.846 | 3.341 | 3.545 | 3.000 | 3.500 | 3.250 | 3.250 | 3.250 |
| pareto_p03_task_rule | 3.981 | 4.038 | 3.846 | 2.941 | 3.000 | 2.364 | 3.545 | 2.825 | 2.875 | 2.750 |
| pareto_p06_abc | 3.846 | 3.923 | 3.665 | 3.018 | 3.182 | 2.636 | 3.273 | 3.058 | 3.208 | 2.833 |

##### Paired outcome counts

- `pareto_p07_abd`: wins=44, ties=0, losses=28, correct→wrong=0, wrong→correct=5.
- `fixed_role`: wins=44, ties=0, losses=28, correct→wrong=1, wrong→correct=7.
- `pareto_p01_boundary`: wins=39, ties=0, losses=33, correct→wrong=0, wrong→correct=3.
- `pareto_p03_task_rule`: wins=33, ties=0, losses=39, correct→wrong=4, wrong→correct=10.
- `pareto_p06_abc`: wins=35, ties=0, losses=37, correct→wrong=4, wrong→correct=10.

### Development96 R2：全部绝对分数与绝对差值

#### Development round 2 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r2_04_oe_old_contract | 7/10 | -0.381 | +0.205 | +0.312 | +0.294 | +0.355 | -0.061 | -0.381 | -0.060 | +0.311 | +0.409 | +0.298 | +0.576 |
| 2 | route_r2_03_tf_boundary | 6/10 | -0.552 | -0.032 | +0.223 | +0.294 | +0.057 | -0.378 | -0.552 | -0.379 | -0.173 | +0.191 | +0.156 | +0.242 |
| 3 | route_r2_01_minimal | 6/10 | -0.621 | +0.011 | +0.229 | +0.265 | +0.147 | -0.460 | -0.621 | -0.517 | -0.207 | +0.421 | +0.398 | +0.455 |
| 4 | route_r2_02_mc_stable | 6/10 | -0.729 | -0.035 | +0.264 | +0.294 | +0.193 | -0.588 | -0.729 | -0.655 | -0.345 | +0.391 | +0.308 | +0.515 |
| 5 | route_r2_06_full_routed | 6/10 | -1.034 | -0.077 | +0.312 | +0.382 | +0.147 | -0.590 | -1.034 | -0.414 | -0.276 | +0.233 | +0.227 | +0.241 |
| 6 | route_r2_05_oe_contract_coverage | 6/10 | -1.069 | -0.073 | +0.256 | +0.265 | +0.235 | -0.643 | -1.069 | -0.517 | -0.291 | +0.337 | +0.282 | +0.420 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r2_04_oe_old_contract | 4.195 | 4.176 | 4.237 | 3.168 | 3.067 | 2.733 | 3.794 | 3.461 | 3.445 | 3.485 |
| route_r2_03_tf_boundary | 4.105 | 4.176 | 3.940 | 2.852 | 2.897 | 2.414 | 3.310 | 3.242 | 3.303 | 3.152 |
| route_r2_01_minimal | 4.112 | 4.147 | 4.029 | 2.769 | 2.828 | 2.276 | 3.276 | 3.473 | 3.545 | 3.364 |
| route_r2_02_mc_stable | 4.146 | 4.176 | 4.075 | 2.641 | 2.719 | 2.138 | 3.137 | 3.442 | 3.455 | 3.424 |
| route_r2_06_full_routed | 4.194 | 4.265 | 4.029 | 2.640 | 2.414 | 2.379 | 3.207 | 3.285 | 3.374 | 3.150 |
| route_r2_05_oe_contract_coverage | 4.138 | 4.147 | 4.118 | 2.587 | 2.379 | 2.276 | 3.191 | 3.389 | 3.429 | 3.329 |

##### Paired outcome counts

- `route_r2_04_oe_old_contract`: wins=59, ties=0, losses=37, correct→wrong=1, wrong→correct=17.
- `route_r2_03_tf_boundary`: wins=46, ties=0, losses=50, correct→wrong=2, wrong→correct=15.
- `route_r2_01_minimal`: wins=53, ties=0, losses=43, correct→wrong=1, wrong→correct=17.
- `route_r2_02_mc_stable`: wins=49, ties=0, losses=47, correct→wrong=1, wrong→correct=18.
- `route_r2_06_full_routed`: wins=53, ties=0, losses=43, correct→wrong=2, wrong→correct=16.
- `route_r2_05_oe_contract_coverage`: wins=52, ties=0, losses=44, correct→wrong=1, wrong→correct=17.

### Development96 R3：全部绝对分数与绝对差值

#### Development round 3 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r3_04_mc_f1_tf_p0 | 10/10 | +0.000 | +0.202 | +0.264 | +0.294 | +0.193 | +0.000 | +0.000 | +0.000 | +0.000 | +0.421 | +0.398 | +0.455 |
| 2 | route_r3_03_mc_f0_tf_p0 | 10/10 | +0.000 | +0.192 | +0.229 | +0.265 | +0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.421 | +0.398 | +0.455 |
| 3 | route_r3_02_mc_f1_safe | 10/10 | +0.000 | +0.075 | +0.264 | +0.294 | +0.193 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | route_r3_01_mc_f0_safe | 10/10 | +0.000 | +0.064 | +0.229 | +0.265 | +0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | route_r3_08_historical | 7/10 | -0.381 | +0.045 | +0.229 | +0.265 | +0.147 | -0.061 | -0.381 | -0.060 | +0.311 | +0.000 | +0.000 | +0.000 |
| 6 | route_r3_07_oe_q3 | 7/10 | -0.552 | +0.098 | +0.229 | +0.265 | +0.147 | -0.247 | -0.552 | -0.241 | +0.103 | +0.421 | +0.398 | +0.455 |
| 7 | route_r3_05_oe_q1 | 6/10 | -0.378 | +0.105 | +0.229 | +0.265 | +0.147 | -0.221 | -0.378 | -0.165 | -0.103 | +0.421 | +0.398 | +0.455 |
| 8 | route_r3_06_oe_q2 | 6/10 | -0.586 | +0.071 | +0.229 | +0.265 | +0.147 | -0.312 | -0.586 | -0.276 | -0.034 | +0.421 | +0.398 | +0.455 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_04_mc_f1_tf_p0 | 4.146 | 4.176 | 4.075 | 3.229 | 3.448 | 2.793 | 3.483 | 3.473 | 3.545 | 3.364 |
| route_r3_03_mc_f0_tf_p0 | 4.112 | 4.147 | 4.029 | 3.229 | 3.448 | 2.793 | 3.483 | 3.473 | 3.545 | 3.364 |
| route_r3_02_mc_f1_safe | 4.146 | 4.176 | 4.075 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_01_mc_f0_safe | 4.112 | 4.147 | 4.029 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_08_historical | 4.112 | 4.147 | 4.029 | 3.168 | 3.067 | 2.733 | 3.794 | 3.052 | 3.147 | 2.909 |
| route_r3_07_oe_q3 | 4.112 | 4.147 | 4.029 | 2.983 | 2.897 | 2.552 | 3.586 | 3.473 | 3.545 | 3.364 |
| route_r3_05_oe_q1 | 4.112 | 4.147 | 4.029 | 3.008 | 3.071 | 2.628 | 3.379 | 3.473 | 3.545 | 3.364 |
| route_r3_06_oe_q2 | 4.112 | 4.147 | 4.029 | 2.917 | 2.862 | 2.517 | 3.448 | 3.473 | 3.545 | 3.364 |

##### Paired outcome counts

- `route_r3_04_mc_f1_tf_p0`: wins=40, ties=29, losses=27, correct→wrong=1, wrong→correct=18.
- `route_r3_03_mc_f0_tf_p0`: wins=43, ties=29, losses=24, correct→wrong=1, wrong→correct=17.
- `route_r3_02_mc_f1_safe`: wins=16, ties=62, losses=18, correct→wrong=0, wrong→correct=3.
- `route_r3_01_mc_f0_safe`: wins=19, ties=62, losses=15, correct→wrong=0, wrong→correct=2.
- `route_r3_08_historical`: wins=36, ties=33, losses=27, correct→wrong=0, wrong→correct=2.
- `route_r3_07_oe_q3`: wins=56, ties=0, losses=40, correct→wrong=1, wrong→correct=17.
- `route_r3_05_oe_q1`: wins=59, ties=0, losses=37, correct→wrong=1, wrong→correct=17.
- `route_r3_06_oe_q2`: wins=55, ties=0, losses=41, correct→wrong=1, wrong→correct=17.

### Untouched Holdout48：全部绝对分数与绝对差值

#### Internal holdout confirmation results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r3_02_mc_f1_safe | 7/10 | -0.059 | -0.010 | -0.042 | -0.059 | -0.004 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | route_r3_01_mc_f0_safe | 7/10 | -0.412 | -0.114 | -0.371 | -0.353 | -0.412 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.865 | 3.882 | 3.824 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| route_r3_02_mc_f1_safe | 3.822 | 3.824 | 3.820 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| route_r3_01_mc_f0_safe | 3.494 | 3.529 | 3.412 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |

##### Paired outcome counts

- `route_r3_02_mc_f1_safe`: wins=9, ties=31, losses=8, correct→wrong=1, wrong→correct=1.
- `route_r3_01_mc_f0_safe`: wins=7, ties=31, losses=10, correct→wrong=1, wrong→correct=0.

### 全部候选的相对变化比例

#### All autoresearch relative changes

Each percentage is `100 × (candidate − split Baseline) / split Baseline`. These split-specific percentages are diagnostic and must not be compared as if the splits had identical record composition.

##### Screening24

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_role | +15.203% | +20.690% | +3.226% | -0.287% | -16.668% | +15.361% | +4.351% | -8.000% | -6.667% | -9.999% |
| pareto_p01_boundary | +10.135% | +10.345% | +9.677% | +5.068% | -11.111% | +15.373% | +13.043% | +8.667% | +10.000% | +6.667% |
| pareto_p02_calibration | -6.757% | -6.897% | -6.452% | +2.251% | +0.000% | +15.373% | -4.348% | +13.000% | +15.000% | +10.000% |
| pareto_p03_task_rule | +10.473% | +13.793% | +3.226% | +2.205% | +0.000% | +15.191% | -4.346% | +8.666% | +10.000% | +6.666% |
| pareto_p04_numeric_guard | +1.689% | +6.897% | -9.677% | +0.279% | -16.666% | +30.756% | -4.347% | +2.667% | +0.000% | +6.667% |
| pareto_p05_single_answer | +20.946% | +27.586% | +6.452% | -5.692% | -8.889% | +23.064% | -21.739% | +0.000% | +0.000% | +0.000% |
| pareto_p06_abc | +14.375% | +19.483% | +3.226% | +1.131% | -11.111% | -0.010% | +13.061% | +13.333% | +13.333% | +13.333% |
| pareto_p07_abd | +3.378% | +3.448% | +3.226% | +1.686% | -5.560% | +7.682% | +4.348% | -2.000% | -3.333% | +0.000% |
| pareto_p08_acde | +13.851% | +17.241% | +6.452% | -9.017% | -22.222% | +15.372% | -13.043% | +6.000% | +3.333% | +10.000% |
| pareto_p09_bce | +10.473% | +13.793% | +3.226% | -16.340% | -16.667% | -7.701% | -21.739% | +7.333% | +10.000% | +3.333% |
| pareto_p10_abcde | +9.459% | +13.793% | +0.000% | -7.327% | -11.108% | -0.009% | -8.702% | +9.333% | +13.333% | +3.333% |
| pareto_p11_abce | +9.797% | +17.241% | -6.452% | -7.889% | -22.219% | -0.010% | +0.000% | +2.000% | +3.333% | +0.000% |

##### Validation72

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_role | +5.898% | +7.524% | +2.030% | -5.534% | -8.537% | -7.353% | +0.000% | +20.492% | +16.452% | +27.273% |
| pareto_p01_boundary | +1.076% | +1.944% | -0.990% | -3.162% | -4.878% | -2.941% | -1.282% | +10.310% | +5.620% | +18.182% |
| pareto_p03_task_rule | +1.074% | +1.942% | -0.990% | -14.756% | -19.512% | -23.528% | +0.000% | -4.115% | -6.567% | +0.000% |
| pareto_p06_abc | -2.354% | -0.971% | -5.644% | -12.517% | -14.634% | -14.706% | -7.694% | +3.804% | +4.265% | +3.030% |
| pareto_p07_abd | +2.060% | +0.971% | +4.653% | -7.850% | -12.255% | -10.294% | +0.039% | +22.458% | +18.687% | +28.787% |

##### Development96 R2

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_r2_01_minimal | +5.909% | +6.819% | +3.788% | -14.254% | -18.000% | -18.514% | -5.941% | +13.792% | +12.662% | +15.625% |
| route_r2_02_mc_stable | +6.792% | +7.576% | +4.962% | -18.213% | -21.150% | -23.456% | -9.915% | +12.799% | +9.774% | +17.707% |
| route_r2_03_tf_boundary | +5.746% | +7.576% | +1.477% | -11.691% | -16.000% | -13.567% | -4.959% | +6.246% | +4.959% | +8.333% |
| route_r2_04_oe_old_contract | +8.043% | +7.576% | +9.134% | -1.895% | -11.050% | -2.161% | +8.928% | +13.415% | +9.485% | +19.791% |
| route_r2_05_oe_contract_coverage | +6.591% | +6.818% | +6.061% | -19.899% | -31.000% | -18.519% | -8.366% | +11.048% | +8.966% | +14.427% |
| route_r2_06_full_routed | +8.030% | +9.848% | +3.788% | -18.260% | -30.000% | -14.817% | -7.921% | +7.626% | +7.222% | +8.281% |

##### Development96 R3

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_r3_01_mc_f0_safe | +5.909% | +6.819% | +3.788% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| route_r3_02_mc_f1_safe | +6.792% | +7.576% | +4.962% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| route_r3_03_mc_f0_tf_p0 | +5.909% | +6.819% | +3.788% | +0.000% | +0.000% | +0.000% | +0.000% | +13.792% | +12.662% | +15.625% |
| route_r3_04_mc_f1_tf_p0 | +6.792% | +7.576% | +4.962% | +0.000% | +0.000% | +0.000% | +0.000% | +13.792% | +12.662% | +15.625% |
| route_r3_05_oe_q1 | +5.909% | +6.819% | +3.788% | -6.847% | -10.953% | -5.922% | -2.970% | +13.792% | +12.662% | +15.625% |
| route_r3_06_oe_q2 | +5.909% | +6.819% | +3.788% | -9.664% | -17.000% | -9.878% | -0.990% | +13.792% | +12.662% | +15.625% |
| route_r3_07_oe_q3 | +5.909% | +6.819% | +3.788% | -7.635% | -16.000% | -8.642% | +2.970% | +13.792% | +12.662% | +15.625% |
| route_r3_08_historical | +5.909% | +6.819% | +3.788% | -1.895% | -11.050% | -2.161% | +8.928% | +0.000% | +0.000% | +0.000% |

##### Holdout48

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_r3_02_mc_f1_safe | -1.094% | -1.515% | -0.097% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| route_r3_01_mc_f0_safe | -9.589% | -9.091% | -10.769% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |

## 附录：Autoresearch 使用的全部 Prompt（文档末尾）

以下 catalog 是推理前用固定示例窗口实际渲染并计算 SHA-256 的完整文本。运行时仅替换 sample-specific 的窗口、数值、latent placeholder 数量和问题；每个 Fixed 条件均保留 30 个 `<|fixed_hint|>`。表中出现相同 SHA-256 表示该题型组件被逐字节复用，而不是重新改写。

### Screening Round 1：Fixed role 与 P01–P11

#### Screening round 1 prompt catalog

Rendered before inference with a fixed illustrative window. Runtime 
values, steps, latent placeholders, and questions are substituted per sample.

##### fixed_role / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### fixed_role / open_ended

SHA-256: `22a815d718ac77ccb09e894aa1b62d3fa3c7edc9593bba5117d63385bf551949`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### fixed_role / true_false

SHA-256: `ef8867788295ce4777b8e80b26cbb7b3360e088eb6fc30a1235a75ff28b490f7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p01_boundary / multiple_choice

SHA-256: `6f3678985ec7846f6c8e2352b32ff48471cb383f5c16062eb1c77501816243dd`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p01_boundary / open_ended

SHA-256: `84660aee5bfa6338fed0f7ee3be07e323faa4accab49553662d05f11cb932318`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p01_boundary / true_false

SHA-256: `439b90f2b3c665bb4c22b77ef30173b6838c3141645b7b1148bca52d16b8bea1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p02_calibration / multiple_choice

SHA-256: `e600afd18b1ea8c095156e58e4faaf621a96cdeffafe073804bd0fc0ae351f6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p02_calibration / open_ended

SHA-256: `45a12c90c98dab3ed0209d3c5642d08a6ad467f9390ad6f59ed7d9d7ab9e74aa`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p02_calibration / true_false

SHA-256: `f05e7eaf748ab51189d9b0c8fd0795a1e3f986fb7036b70264679257779e9aad`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p03_task_rule / multiple_choice

SHA-256: `26bac720e9743601acdd3a9a261db5e8197041cb28ec05bd4fe0ff48d368dd91`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p03_task_rule / open_ended

SHA-256: `880e2ae5fd9bddc9bf9537e8582c1ab34d8a852e545d532c48757ea5b62578b6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p03_task_rule / true_false

SHA-256: `2966b574a3060dfcf02e27f14ee8a3e39da50c5ed8bec5fba02b488b0426ba47`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p04_numeric_guard / multiple_choice

SHA-256: `f5946c9225802e32f13dae2e5f8242ef5fe70297873593d3e12c3426814f88f9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p04_numeric_guard / open_ended

SHA-256: `4e30ce8cdd836f6223e51d3f906e1ac982cb890f66a6269bf65123523dbf3968`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p04_numeric_guard / true_false

SHA-256: `6448c49f39fe49564d2d0449c60f782cb16a23fee2b737431003263bb1ce24d9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p05_single_answer / multiple_choice

SHA-256: `0567400d3b90dbf3ac2dd5e9fb143c2cd2a433645cdc1badff59ba10e5af63d9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p05_single_answer / open_ended

SHA-256: `c1311b5567ff9c78c15f38ea3bc6af69aad8d290289cec82e087a3b8a5ec1f3a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p05_single_answer / true_false

SHA-256: `892e216fa2a14f987c1872b3e90fc2c9adfe085300cf8275950d72e0d596b80c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p06_abc / multiple_choice

SHA-256: `99ab4177e85ca491d6d6c78fb692e8545631ce2ac3a7b07ed3564dbb4e620341`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p06_abc / open_ended

SHA-256: `575c6ecb693bc202ef75dcdf8a07708538d337f8013cf2b8a8a89450e264f109`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p06_abc / true_false

SHA-256: `6410c53349710811327402e8379d97b481c4c6b96c62f5d3f16bf4c5907cebe3`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p07_abd / multiple_choice

SHA-256: `d808065164a247810684eb1fcca04cf5b33477793986a6e7382d4297863912d0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p07_abd / open_ended

SHA-256: `c07f585e8ecee409260e07c2ff2480de6a1fbb8cee201fee4821e3160aa88125`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p07_abd / true_false

SHA-256: `0766abfebc778fafef4b1e6f9bdab0b802da6b577c0014c7663a4e98e1cb988a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p08_acde / multiple_choice

SHA-256: `427bcccb202c909b218619fae7d15fd9714cdcd384cfa192a132a95dc66f082a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p08_acde / open_ended

SHA-256: `359d73d8f2c2b233da7eca78d8b6a5d1adc4eca0e44ab76119ffbb0404ddf7e0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p08_acde / true_false

SHA-256: `bf7aa998bc72aae1dffd860e0b0897ea5b41ef33e0b82d97b56623a893062eeb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p09_bce / multiple_choice

SHA-256: `fff5bcbeee210446d7f8c4a776dc098655844569148713a01987bf5494d0f9d5`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p09_bce / open_ended

SHA-256: `3f0e8de389a4f7ec362367495dbc0900f2216650c3c7c6fe78683412f8d45c2a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p09_bce / true_false

SHA-256: `ce52a77289889154c3642bc79374dfa5759c53370ec783aa15f0d8f93ae498f9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p10_abcde / multiple_choice

SHA-256: `0a081ec3dc48eccc591d1d697c6ae659b8883fe7cd4dcf950707d06889ade015`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p10_abcde / open_ended

SHA-256: `1cfefc5bc3a43bd24b0f35ff9a4ad8e78546de49f9874e6fb6325a5a2aee0bda`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p10_abcde / true_false

SHA-256: `282b2c1f38a42cf4213bddc78b1b4116417b159ba316688c810e9a1c1a9dc092`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p11_abce / multiple_choice

SHA-256: `7d66ad6fd4d365020e3452ee190eed79a22e3e880e4cc3a888be1351fb7bc908`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p11_abce / open_ended

SHA-256: `4dad0176a51f018d2660312b7210bb8548f1163b44869393002c59de4b5ef7d4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p11_abce / true_false

SHA-256: `ef67696bd745bbedf43d8604a23442b24946193beb2ccd65ff0e412016446f02`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

### Development Round 2：R2-01–R2-06

#### Development round 2 routed prompt catalog

Rendered before inference with a fixed illustrative window. Runtime 
values, steps, latent placeholders, and questions are substituted per sample.

##### route_r2_01_minimal / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_01_minimal / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_01_minimal / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_02_mc_stable / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_02_mc_stable / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_02_mc_stable / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_03_tf_boundary / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_03_tf_boundary / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_03_tf_boundary / true_false

SHA-256: `e7d3cc6f957bfceb4bcce9aa6b45e77079884e2e5c74d952bd3543ff55335e99`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Use only evidence inside the half-open window [7, 10); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_04_oe_old_contract / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_04_oe_old_contract / open_ended

SHA-256: `85ff0be5d920d2ae5ab9ed0f8cef122e12924436a9dbcae6e53a8014f847ac28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_04_oe_old_contract / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_05_oe_contract_coverage / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_05_oe_contract_coverage / open_ended

SHA-256: `45edabaf3706ef6ae79d1e1c190af5e7860ed052597b1c80d2ecb2264fcbf319`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_05_oe_contract_coverage / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_06_full_routed / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_06_full_routed / open_ended

SHA-256: `45edabaf3706ef6ae79d1e1c190af5e7860ed052597b1c80d2ecb2264fcbf319`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_06_full_routed / true_false

SHA-256: `e7d3cc6f957bfceb4bcce9aa6b45e77079884e2e5c74d952bd3543ff55335e99`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Use only evidence inside the half-open window [7, 10); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

### Development Round 3 与 Holdout：R3-01–R3-08

#### Development round 3 routed prompt catalog

Rendered before inference with a fixed illustrative window. Runtime
values, steps, latent placeholders, and questions are substituted per sample.

##### route_r3_01_mc_f0_safe / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_01_mc_f0_safe / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_01_mc_f0_safe / true_false

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_02_mc_f1_safe / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_02_mc_f1_safe / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_02_mc_f1_safe / true_false

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_03_mc_f0_tf_p0 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_03_mc_f0_tf_p0 / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_03_mc_f0_tf_p0 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_04_mc_f1_tf_p0 / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_04_mc_f1_tf_p0 / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_04_mc_f1_tf_p0 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_05_oe_q1 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_05_oe_q1 / open_ended

SHA-256: `cef28363829aafc3adf63486dddedcea0e9203e08cc7ff4d125d609993fab737`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer the open-ended question in one coherent paragraph. Include: (1) a direct assessment of this window; (2) two or three observed qualitative features that support the assessment, including shape and persistence or recovery; and (3) the boundary/context limitation or additional evidence requested by the question. State location as beginning, middle, or end and describe magnitude relatively unless an exact value or step is unambiguous in the supplied input.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_05_oe_q1 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_06_oe_q2 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_06_oe_q2 / open_ended

SHA-256: `5b8464266ea03cd6ac45ae232cfb3901db9a82728fbe2bce9c8444e24149f580`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Treat the event or pattern named in the question as the behavior to evaluate, not as a predetermined anomaly label. Compare its abruptness, isolation, persistence or recovery, and boundary position with the rest of the supplied window. Report the evidence supporting your conclusion and the strongest evidence against it or remaining uncertainty.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_06_oe_q2 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_07_oe_q3 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_07_oe_q3 / open_ended

SHA-256: `4556449ed061fb4a172f53faafd88fd3a47cfabec392feda67c674a0afa903c6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First answer the assessment or analysis requested by the question. Then apply it to this window using qualitative evidence: local contrast, persistence or recovery, and boundary position. State what supports the conclusion and what observation would refute it or require outside-window context. Use beginning, middle, or end for location when exact step alignment is not explicit.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_07_oe_q3 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_08_historical / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_08_historical / open_ended

SHA-256: `85ff0be5d920d2ae5ab9ed0f8cef122e12924436a9dbcae6e53a8014f847ac28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_08_historical / true_false

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```
