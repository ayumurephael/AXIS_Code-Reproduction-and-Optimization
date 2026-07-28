# AXIS Baseline 与 Round 9 Prompt：paper140 多 Judge 正式 G-Eval

## 结论

本实验在论文兼容的 `paper140`（140 QA / 70 series）上，对已经审计的
full284 模型回答做精确子集提取，不重新运行 AXIS checkpoint 推理。四个
条件分别由 `deepseek-v4-pro`、`qwen3.5-397b-a17b` 和
`qwen3-30b-a3b-instruct-2507` 独立评分；不同 Judge 的分数不混合。

主要结论如下：

1. 没有一条 Round 9 路由在三位 Judge 下都满足 Table-I 十项不低于各自
   Baseline。结果不支持“跨 Judge 稳健的全面提升”。
2. `Round 9 TF+OE` 是最稳定的路线：TF Final 在三位 Judge 下都提高；
   但 DeepSeek 判定 TF Justification 下降 `-0.0238`，Qwen3.5 判定
   TF Correctness 下降 `-0.0310`。只有 Qwen30B 给出 TF 三项同时提高，
   从而在该 Judge 下达到十项非降、三项严格提高。
3. MC status-then-shape 路线在三位 Judge 下都降低 MC Final 与
   MC Correctness。MC Reasoning 仅在 DeepSeek 与 Qwen30B 上提高，
   Qwen3.5 上反而下降。因此该路线不能作为稳健改进。
4. paper140 不包含 full284 中唯一通过 Round-9 采用门的 OE 样本
   `series_000111:0`。本子集中四个条件的 55 条 OE 回答逐字相同，
   所以所有 Judge 下四项 OE 指标都与 Baseline 完全相同。这不是
   “OE 没有评分波动”，而是同一 Judge 内对字节相同回答精确复用了同一
   canonical 分数；本实验不能据此证明 OE 得到改善。
5. 所有配对 140-record Final 差值的 95% bootstrap CI 均跨过 0。
   即使是 Qwen30B 下通过十项非降的 `Round 9 TF+OE`，其配对均值
   `+0.010` 的区间仍为 `[-0.032, +0.050]`。

## 实验口径

- 数据：`paper140`，140 QA，题型分布为 MC 43、OE 55、TF 42。
- 条件：Baseline、Round 9 MC+OE、Round 9 TF+OE、Round 9 Joint。
- 回答来源：从已审计的 full284 预测中按 paper140 manifest 精确提取；
  checkpoint 推理不重跑。
- 提取后共有 560 条条件记录。按“record ID、题干、标准答案、题型、回答”
  完全相同去重后为 201 条唯一回答、每 Judge 457 个唯一评分维度。
- 每位 Judge 的唯一评分完成后，按 response SHA-256 展开回四个完整条件，
  每位 Judge 均为 1,340 个 Table-I 维度分数。
- 回答变化数：MC+OE 43 条、TF+OE 18 条、Joint 61 条；OE 变化数均为 0。
- Judge 间不复用回答分数，不汇总成单一平均 Judge。
- Table-I 数值保留四位；绝对差值为候选减同一 Judge 的 Baseline；
  相对变化为 `绝对差值 / Baseline × 100%`。

## DeepSeek v4-pro

评分协议：作者 G-Eval rubric；优先使用最终 1–5 score token 的完整
logprob 期望；无法得到完整分布时收集恰好 20 个有效整数评分。

### Table-I

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | 10/10 非降 | 严格提高 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| AXIS Baseline | 4.2000 | 4.3256 | 3.9070 | 3.1318 | 2.9491 | 2.7636 | 3.7746 | 3.6143 | 3.5952 | 3.6429 | — | — |
| Round 9 MC+OE | 4.1199 | 4.2012 | 3.9302 | 3.1318 | 2.9491 | 2.7636 | 3.7746 | 3.6143 | 3.5952 | 3.6429 | 否 | 1 |
| Round 9 TF+OE | 4.2000 | 4.3256 | 3.9070 | 3.1318 | 2.9491 | 2.7636 | 3.7746 | 3.6333 | 3.6429 | 3.6190 | 否 | 2 |
| Round 9 Joint | 4.1199 | 4.2012 | 3.9302 | 3.1318 | 2.9491 | 2.7636 | 3.7746 | 3.6333 | 3.6429 | 3.6190 | 否 | 3 |

### 相对 Baseline 的绝对差值

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 9 MC+OE | -0.0801 | -0.1244 | +0.0233 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Round 9 TF+OE | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0190 | +0.0476 | -0.0238 |
| Round 9 Joint | -0.0801 | -0.1244 | +0.0233 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0190 | +0.0476 | -0.0238 |

### 相对 Baseline 的变化比例

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 9 MC+OE | -1.9075% | -2.8764% | +0.5957% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% |
| Round 9 TF+OE | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.5270% | +1.3245% | -0.6536% |
| Round 9 Joint | -1.9075% | -2.8764% | +0.5957% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.5270% | +1.3245% | -0.6536% |

唯一评分方法计数为：451 个完整 logprob 期望、6 个 exact-20；
展开后分别为 1,320 与 20。

## Qwen3.5-397B-A17B

评分协议：作者 G-Eval rubric，`enable_thinking=false`，seed 72，
`top_logprobs=5`；若缺失 score-token 概率质量上界超过 `1e-6`，
先对同一 judgment 做确定性 A–E JSON readout。唯一仍不能满足上界的
维度按用户单独授权使用 exact-20。

### Table-I

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | 10/10 非降 | 严格提高 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| AXIS Baseline | 4.4116 | 4.4078 | 4.4204 | 3.1316 | 2.9470 | 3.0544 | 3.4371 | 3.8227 | 3.9267 | 3.6667 | — | — |
| Round 9 MC+OE | 4.3159 | 4.3423 | 4.2543 | 3.1316 | 2.9470 | 3.0544 | 3.4371 | 3.8227 | 3.9267 | 3.6667 | 否 | 0 |
| Round 9 TF+OE | 4.4116 | 4.4078 | 4.4204 | 3.1316 | 2.9470 | 3.0544 | 3.4371 | 3.8285 | 3.8958 | 3.7276 | 否 | 2 |
| Round 9 Joint | 4.3159 | 4.3423 | 4.2543 | 3.1316 | 2.9470 | 3.0544 | 3.4371 | 3.8285 | 3.8958 | 3.7276 | 否 | 2 |

### 相对 Baseline 的绝对差值

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 9 MC+OE | -0.0956 | -0.0654 | -0.1661 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Round 9 TF+OE | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0058 | -0.0310 | +0.0609 |
| Round 9 Joint | -0.0956 | -0.0654 | -0.1661 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0058 | -0.0310 | +0.0609 |

### 相对 Baseline 的变化比例

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 9 MC+OE | -2.1679% | -1.4847% | -3.7577% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% |
| Round 9 TF+OE | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.1512% | -0.7890% | +1.6616% |
| Round 9 Joint | -2.1679% | -1.4847% | -3.7577% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.1512% | -0.7890% | +1.6616% |

唯一评分方法计数为：1 个完整 logprob、451 个 bounded logprob、
4 个确定性 readout、1 个 exact-20。展开后分别为 4、1,322、10、4。
exact-20 的唯一原始维度为
`series_000096:0 / base / open_ended accuracy`，20 个样本均为有效
1–5 整数；由于四个条件的该 OE 回答逐字相同，此分数被精确展开为 4 行。

## Qwen3-30B-A3B-Instruct-2507

评分协议：作者 G-Eval rubric，非思考模型，不发送
`enable_thinking` 扩展；seed 72、`top_logprobs=5`、缺失概率质量上界
`1e-6`，必要时采用确定性 A–E JSON readout，不使用 sampling fallback。

### Table-I

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | 10/10 非降 | 严格提高 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| AXIS Baseline | 4.4367 | 4.5365 | 4.2037 | 2.9032 | 2.5513 | 2.8915 | 3.3275 | 3.9269 | 4.0018 | 3.8146 | — | — |
| Round 9 MC+OE | 4.3684 | 4.4236 | 4.2395 | 2.9032 | 2.5513 | 2.8915 | 3.3275 | 3.9269 | 4.0018 | 3.8146 | 否 | 1 |
| Round 9 TF+OE | 4.4367 | 4.5365 | 4.2037 | 2.9032 | 2.5513 | 2.8915 | 3.3275 | 3.9589 | 4.0288 | 3.8541 | 是 | 3 |
| Round 9 Joint | 4.3684 | 4.4236 | 4.2395 | 2.9032 | 2.5513 | 2.8915 | 3.3275 | 3.9589 | 4.0288 | 3.8541 | 否 | 4 |

### 相对 Baseline 的绝对差值

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 9 MC+OE | -0.0683 | -0.1129 | +0.0359 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| Round 9 TF+OE | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0321 | +0.0271 | +0.0395 |
| Round 9 Joint | -0.0683 | -0.1129 | +0.0359 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0321 | +0.0271 | +0.0395 |

### 相对 Baseline 的变化比例

| Prompt | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Round 9 MC+OE | -1.5392% | -2.4891% | +0.8529% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% |
| Round 9 TF+OE | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.8162% | +0.6760% | +1.0367% |
| Round 9 Joint | -1.5392% | -2.4891% | +0.8529% | +0.0000% | +0.0000% | +0.0000% | +0.0000% | +0.8162% | +0.6760% | +1.0367% |

唯一评分方法计数为：4 个完整 logprob、450 个 bounded logprob、
3 个确定性 bounded readout；展开后分别为 14、1,314、12。

## 跨 Judge 对照

| Prompt | DeepSeek 10/10 | Qwen3.5 10/10 | Qwen30B 10/10 | 跨三 Judge 10/10 | 结论 |
|---|:---:|:---:|:---:|:---:|---|
| Round 9 MC+OE | 否 | 否 | 否 | 否 | MC Final/Correctness 在三位 Judge 下均下降 |
| Round 9 TF+OE | 否 | 否 | 是 | 否 | TF Final 一致提高，但子维度存在 Judge 分歧 |
| Round 9 Joint | 否 | 否 | 否 | 否 | 叠加了 MC 路线的退化，不能通过 |

配对 Final 差值及 95% bootstrap CI：

| Judge | Round 9 MC+OE | Round 9 TF+OE | Round 9 Joint |
|---|---:|---:|---:|
| DeepSeek v4-pro | -0.025 [-0.146, +0.096] | +0.006 [-0.040, +0.054] | -0.019 [-0.146, +0.110] |
| Qwen3.5-397B | -0.029 [-0.157, +0.090] | +0.002 [-0.054, +0.064] | -0.028 [-0.166, +0.107] |
| Qwen3-30B | -0.021 [-0.155, +0.106] | +0.010 [-0.032, +0.050] | -0.011 [-0.150, +0.122] |

这些配对统计以 140 个 record 的题型加权 Final 为单位，不把不同 Judge
的量尺混成一个统计量。

## 完整性审计

- 三个 Judge 的正式 `audit.json` 均为 `ok=true`。
- 实验级 `integrity_manifest.json` 二次验证：
  - 560/560 条条件记录、201 条唯一回答；
  - 每 Judge 457/457 个唯一评分维度；
  - 每 Judge 1,340/1,340 个展开评分维度；
  - 无重复 prediction/score key；
  - 模型名、Provider、`enable_thinking` 与协议一致；
  - 所有 Prompt SHA-256 合法；
  - 所有 exact-20 均恰好含 20 个有效整数；
  - 所有 Qwen bounded 分布的缺失质量上界不超过 `1e-6`；
  - Qwen3.5 sampling fallback 的 key 与用户授权的唯一维度严格一致；
  - 实验目录未发现 API-key 形态字符串；
  - 所有交付文件均记录 SHA-256。
- 相关单元测试：19 passed；Python 编译检查通过。
- 所有正式 API Judge 任务都在获授权的 GPU 服务器上运行；未在本机运行
  GPU 运算或训练。
- 远端临时 API key 文件在任务结束后已删除，凭据未写入 Git、评分 JSONL
  或本报告。

## 全部被测 Prompt 与路由

### A. 原 AXIS Baseline Prompt

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

### B. MC status-then-shape Prompt

只用于 MC；其余结构、Local/Fix token 数量、值序列化和生成边界保持
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

### C. TF negative-cue qualitative RE2 Prompt

只有题干 token 命中以下集合时才启用：

```text
no, not, without, absence, lack, neither, nor, cannot, can't,
doesn't, isn't, aren't, wasn't, weren't
```

未命中时逐字使用 Baseline；命中时使用：

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

### D. OE Round-9 两阶段 Prompt

第一遍逐字使用 A 节 Baseline。只有 OE 运行第二遍：

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

第二遍回答只有同时满足以下确定性采用门才替换第一遍：

1. Baseline 包含 `<think>` 且没有 `</think>`；
2. Baseline 没有行首 `Answer:` 或 `Final Answer:`；
3. candidate 没有 think 标签且包含上述 Answer 边界；
4. Baseline 与 candidate 都显式包含至多跨 80 字符的
   `no/without ... anomaly/irregularity`；
5. 按正则 `\b\w+\b` 计词，candidate 词数至少为 Baseline 的 60%。

否则逐字采用 Baseline，并精确复用其 Judge 分数。

### E. 四个正式条件

```text
base:
  multiple_choice -> A
  open_ended      -> A
  true_false      -> A

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

## 限制与可解释边界

1. 本实验评估的是 paper140 上既有回答的 G-Eval，不是新的 checkpoint
   推理重复实验；它隔离了 Judge 差异，但不能测量回答生成随机性。
2. 三位 Judge 的绝对 Baseline 分数差异很大，尤其是 OE completeness 与
   relevance；因此每位 Judge 只能与自己的 Baseline 做配对比较。
3. paper140 缺失唯一实际采用的 OE 修复 case，所以本实验不能验证
   Round 9 的 OE 改进是否跨 Judge 成立。要回答该问题，必须在包含
   `series_000111:0` 的 full284 或专门 OE manifest 上另做多 Judge 审计。
4. Qwen3.5 的 1 个唯一维度使用了 exact-20，仍含有限采样误差；该分数在
   四条件间精确复用，不会制造候选相对 Baseline 的 OE 差值。
