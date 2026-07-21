# AXIS 双评测协议

本仓库同时支持 DeepSeek v4-pro 和 Gemini 2.5 Pro。二者评估同一份 AXIS predictions，但用途和统计口径不同。

## 固定职责

| 场景 | Judge | 用途 | 是否可与论文数值直接对齐 |
|---|---|---|---|
| 日常迭代、消融筛选 | DeepSeek v4-pro | 低成本、同裁判的相对比较 | 否 |
| 最终正式评测 | Gemini 2.5 Pro | 作者 prompt/rubric 下的 Table 1 | 是，仍需报告接口限制 |
| Gemini 稳定性检查 | Gemini strict sampling | 估计无 logprobs 时的采样波动 | 否，仅诊断 |

日常 DeepSeek 分数不得与 Gemini 分数拼成一张“混合”表。比较两个 checkpoint 时，必须固定 predictions 子集、judge、prompt、model id、endpoint、scoring mode 和脚本 commit。

## 输入协议

两种 judge 均只接收已经完成并审计的 JSONL：

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
  --fallback-workers 1
```

runner 会优先读取最终 Score token 的 1–5 概率分布；服务端没有完整候选概率时，使用恰好 20 个有效采样分数的均值。`*.fallback_pending.jsonl` 是断点 journal，不是最终 score 文件。原命令重跑会跳过已完成键。

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
- 不能用任何测试 judge 分数选择 epoch、seed、学习率或架构。