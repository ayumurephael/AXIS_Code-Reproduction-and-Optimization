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
