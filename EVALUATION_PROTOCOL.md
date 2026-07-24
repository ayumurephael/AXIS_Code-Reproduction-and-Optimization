# AXIS 三评测协议

本仓库同时支持 DeepSeek v4-pro、Gemini 2.5 Pro 和 Qwen 系列。三者评估同一份 AXIS predictions，但用途和统计口径不同。

## 固定职责

| 场景 | Judge | 用途 | 是否可与论文数值直接对齐 |
|---|---|---|---|
| 日常迭代、消融筛选 | DeepSeek v4-pro | 低成本、同裁判的相对比较 | 否 |
| 最终正式评测 | Gemini 2.5 Pro | 作者 prompt/rubric 下的 Table 1 | 是，仍需报告接口限制 |
| 独立交叉评测 | Qwen 系列 | 直接读取 score-token logprobs，检验结论的 Judge 稳健性 | 否 |
| Gemini 稳定性检查 | Gemini strict sampling | 估计无 logprobs 时的采样波动 | 否，仅诊断 |

不同 Judge 的分数不得拼成一张“混合”表或当作同一量尺。比较两个 checkpoint 时，必须固定 predictions 子集、judge、prompt、model id、endpoint、scoring mode 和脚本 commit。

## 输入协议

三种 judge 均只接收已经完成并审计的 JSONL：

```bash
python -m tools.axis_repro.audit_results \
  --predictions <prediction_dir>/predictions.jsonl \
  --manifest experiments/reproduction/manifests_author42/paper140.json \
  --modes base
```

正式输入必须是 `paper140` 的 140 条 base prediction。不要把 validation、full284、partial journal 或不同 batching 的文件混入。

## DeepSeek v4-pro：日常默认

密钥只放进当前进程环境。官方或兼容中转地址可用 `DEEPSEEK_BASE_URL` 或 `--endpoint` 指定；不要把含密钥的 URL 写入文档。

```bash
export DEEPSEEK_API_KEY='<set outside repository>'
export DEEPSEEK_BASE_URL='https://api.deepseek.com/chat/completions'

python -m tools.axis_repro.geval_resilient \
  --predictions <prediction_dir>/predictions.jsonl \
  --output <score_dir>/geval_deepseek_v4_pro.jsonl \
  --model deepseek-v4-pro \
  --max-tokens 4096 \
  --fallback-samples 20 \
  --primary-workers 8 \
  --fallback-workers 1 \
  --max-retry-rounds 12
```

runner 会优先读取最终 Score token 的 1–5 概率分布；服务端没有完整候选概率时，使用恰好 20 个有效采样分数的均值。`*.fallback_pending.jsonl` 是断点 journal，不是最终 score 文件。原命令重跑会跳过已完成键。

HTTP 401/403/404 等永久请求错误会立即失败；408/409/425/429、5xx
和传输层故障只在有界轮次内重试。这样可以保留断点恢复能力，同时避免认证、
模型名或端点配置错误导致无限请求。

若服务端不支持 logprobs，结果仍可用于同一 DeepSeek 配置下的日常相对比较，但必须记录 `method`，不能宣称与论文 Gemini G-Eval 等价。

## Gemini 2.5 Pro：正式评测

默认兼容 endpoint 是 PackyAPI，也可通过 `GEMINI_BASE_URL` 或 `--endpoint` 显式设置。发送测试问题、参考答案与生成回答属于外部 API 数据传输，执行前应确认授权和预算。

```bash
export GEMINI_API_KEY='<set outside repository>'
export GEMINI_BASE_URL='https://www.packyapi.com/v1/chat/completions'

python -m tools.axis_repro.geval_gemini \
  --predictions <prediction_dir>/predictions.jsonl \
  --output <score_dir>/geval_gemini25pro_author.jsonl \
  --model gemini-2.5-pro \
  --prompt-template author \
  --scoring-mode author \
  --primary-workers 3
```

`--prompt-template author` 使用作者补充 G-Eval 的维度、权重、rubric 与布局。`--scoring-mode author` 的行为是：若接口返回完整 score-token logprobs，则使用概率期望；否则按作者代码可执行的回退语义保存 temperature=0 单次整数分。

PackyAPI 对 Gemini 2.5 Pro 的实测兼容响应不含可用 `choices[].logprobs`，原生路径也不能假定支持。因此正式结果必须保留 `logprobs_requested`、`logprobs_returned`、`method`、`prompt_sha256`、`model`、`endpoint_host` 和 usage 元数据。

## Gemini 严格采样校准

只在独立目录运行，禁止覆盖正式 author-mode 文件：

```bash
python -m tools.axis_repro.geval_gemini \
  --predictions <prediction_dir>/predictions.jsonl \
  --output <calibration_dir>/geval_gemini25pro_strict.jsonl \
  --model gemini-2.5-pro \
  --prompt-template author \
  --scoring-mode strict \
  --fallback-samples 20
```

strict 结果估计服务端无 logprobs 时的波动，不替代论文/作者代码的 author-mode 结果。


## Qwen 系列：直接 logprobs 交叉评测

Qwen 使用阿里云百炼 OpenAI 兼容接口。中国内地（北京）默认完整 endpoint 为
`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`；其他地域的
API key 必须配套对应地域或 Workspace endpoint，并通过 `QWEN_BASE_URL` 或
`--endpoint` 显式覆盖。模型 ID 可自由指定；本实验固定
`qwen3-30b-a3b-instruct-2507`，扩展交叉评测另使用
`qwen3.5-397b-a17b`。

Qwen3 开源模型支持输出 token logprobs，但百炼 `top_logprobs` 上限为 5。
runner 使用作者 prompt，并定位最终 `**Score:**` 后的单个 ASCII 数字 token。
若 top-5 恰好覆盖 1--5，则计算精确归一化期望；若非评分 token 挤占 top-5，
每个未返回数字的 logprob 都不大于已返回第 5 名的 cutoff。runner 据此计算缺失
评分质量的最坏上界，只有上界不超过 `1e-6` 才接受有界截断分布，并记录
`missing_score_mass_upper_bound`、`score_error_upper_bound` 和缺失数字。超过阈值
仍 fail-closed；不能静默把任意缺失候选当作概率 0。只有明确采用 AXIS 回退协议
时才可传 `--fallback-samples 20`。

若原始最终数字的 top-5 仍超过该上界，先写入 pending journal，再使用
`geval_qwen_readout`：它把已经生成的作者式判断原文作为 assistant 历史，只要求
同一模型把其中的整数分数编码为 JSON 单标签 A--E（A=1，…，E=5）。读出标签
必须与原始整数完全一致，仍使用 top-5 直接 logprobs 和相同 `1e-6` 上界；否则
继续 fail-closed。该二阶段只是确定性标签读出，不重新判断答案，也不是 20 次
采样回退。

```bash
export QWEN_API_KEY='<set outside repository>'
export QWEN_BASE_URL='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'

python -m tools.axis_repro.geval_qwen \
  --predictions <prediction_dir>/predictions.jsonl \
  --output <score_dir>/geval_qwen3_30b_a3b_instruct_2507.jsonl \
  --model qwen3-30b-a3b-instruct-2507 \
  --top-logprobs 5 \
  --seed 72 \
  --primary-workers 6 \
  --max-missing-score-mass-upper-bound 1e-6 \
  --max-retry-rounds 12

python -m tools.axis_repro.geval_qwen_readout \
  --pending <score_dir>/fallback_pending.jsonl \
  --output <score_dir>/geval_qwen3_30b_a3b_instruct_2507.jsonl \
  --model qwen3-30b-a3b-instruct-2507 \
  --top-logprobs 5 \
  --seed 72 \
  --max-missing-score-mass-upper-bound 1e-6 \
  --workers 2 \
  --max-retry-rounds 12

python -m tools.axis_repro.audit_results \
  --predictions <prediction_dir>/predictions.jsonl \
  --scores <score_dir>/geval_qwen3_30b_a3b_instruct_2507.jsonl \
  --manifest experiments/reproduction/manifests_author42/paper140.json \
  --modes base \
  --expected-model qwen3-30b-a3b-instruct-2507 \
  --expected-provider qwen \
  --expected-enable-thinking auto \
  --allowed-methods final_score_top_logprobs \
                    final_score_top_logprobs_bounded \
                    final_score_top_logprobs_readout \
                    final_score_top_logprobs_readout_bounded \
  --max-missing-score-mass-upper-bound 1e-6 \
  --require-logprobs \
  --require-prompt-hashes
```

`qwen3-30b-a3b-instruct-2507` 是仅非思考模式模型，默认不要发送
`enable_thinking`。对支持混合模式的其他 Qwen3 模型，可以使用
`--enable-thinking true|false`；`auto` 表示不发送该扩展字段。正式 Qwen
结果应执行上述 fail-closed 审计。

`qwen3.5-397b-a17b` 是支持思考/非思考的混合模式模型。本项目为保证两臂及
两个 Qwen Judge 的 score-token 口径一致，主评分和 JSON readout 都显式传
`--enable-thinking false`，并在最终审计传
`--expected-enable-thinking false`。pending journal、最终 score row 和 readout
元数据必须保存同一模式；发现主判词与 readout 模式不一致时立即 fail-closed。

Qwen API key、模型和 endpoint 具有地域一致性要求。401/403 应检查 key 与权限，
404 应检查模型 ID/地域，429 与 5xx 才进行有界退避。结果必须记录实际返回
model、endpoint host、prompt hash、seed、usage、method、归一化 1--5 分布与
缺失质量/分数误差上界。

## 聚合与完整性审计

```bash
python -m tools.axis_repro.audit_results \
  --predictions <prediction_dir>/predictions.jsonl \
  --scores <score_dir>/geval_*.jsonl \
  --manifest experiments/reproduction/manifests_author42/paper140.json \
  --modes base

python -m tools.axis_repro.table_runner \
  --scores <score_dir>/geval_*.jsonl \
  --output-prefix <score_dir>/table1
```

`audit_results` 必须返回 `ok=true`。paper140/base 的期望数量是 140 条 prediction 和 335 条 dimension score。

## 成本控制

- 基线 predictions 和 judge scores 生成一次后按 checkpoint hash、prompt hash 和 judge 配置缓存。
- 日常首先比较完整 validation loss、forced-choice 准确率、数值/文本代理指标和因果诊断。
- 里程碑候选使用固定小样本 DeepSeek；只有唯一最终候选运行完整 Gemini 335 维评分。
- Qwen 只对已选定 checkpoint 的既有 paper140 predictions 做独立交叉评测，不得参与选模。
- 不能用任何测试 judge 分数选择 epoch、seed、学习率或架构。
