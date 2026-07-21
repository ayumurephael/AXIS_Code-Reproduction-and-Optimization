# AXIS 复现参考结果

本文只记录通过完整性审计、且身份清楚的结果。不同 judge 和不同生成协议的分数是独立结果集，不应混为一个“正式分数”。论文值作为外部参照。

## 论文 Table 1

| 模型/协议 | MC 最终 | MC 正确性 | MC 推理质量 | OE 最终 | OE 准确性 | OE 完整性 | OE 相关性 | TF 最终 | TF 正确性 | TF 合理性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 AXIS / Gemini-2.5 | 4.19 | 4.21 | 4.14 | 3.02 | 2.87 | 2.93 | 3.31 | 3.65 | 3.60 | 3.74 |

## 可审计参考结果

| Phase-II 权重 / 生成 / Judge | MC 最终 | MC 正确性 | MC 推理质量 | OE 最终 | OE 准确性 | OE 完整性 | OE 相关性 | TF 最终 | TF 正确性 | TF 合理性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 本仓库 epoch-3 / per-record / DeepSeek v4-pro | 3.91 | 3.98 | 3.74 | 3.05 | 2.96 | 2.65 | 3.62 | 3.67 | 3.71 | 3.60 |
| 本仓库 epoch-3 / per-record / Gemini 2.5 Pro author | 4.03 | 4.05 | 4.00 | 2.77 | 2.80 | 2.51 | 3.05 | 3.66 | 3.71 | 3.57 |
| 本仓库 epoch-3 / series / Gemini 2.5 Pro author | 4.06 | 4.07 | 4.05 | 2.73 | 2.80 | 2.42 | 3.02 | 3.63 | 3.69 | 3.55 |
| 作者候选 epoch-33 / per-record / Gemini 2.5 Pro author | 4.26 | 4.28 | 4.21 | 2.93 | 2.69 | 2.73 | 3.44 | 3.80 | 3.81 | 3.79 |
| **作者候选 epoch-33 / series / Gemini 2.5 Pro author** | **4.27** | **4.28** | **4.23** | **2.97** | **2.84** | **2.78** | **3.35** | **3.74** | **3.74** | **3.74** |

最后一行相对论文的 MC/OE/TF Final 差值为 `+0.08/-0.05/+0.09`；三个 Final 的平均绝对误差为 0.073，十指标平均绝对误差为 0.074。它证明当前代码路径能够加载作者前缀 checkpoint、执行作者 series batching 并用作者 Gemini prompt 得到接近论文的结果。

作者候选 checkpoint 是外部实验资产，不随 Git 仓库分发，也不能替代从统一协议重新训练三个分支。公平架构比较仍需按 `AUTHOR_COMPATIBLE_TRAINING.md` 从同一 Phase-I 状态训练。

## 三轮基线 checkpoint

历史低预算训练固定为 seed 72、3 epochs、AdamW、LR `1e-4`、weight decay `1e-5`、三卡 global batch 3。三个 checkpoint 在同一 1500-series / 3000-QA validation 集上的 mean teacher-forced loss 为：

| Epoch | Overall | Multiple Choice | Open Ended | True/False |
|---:|---:|---:|---:|---:|
| 1 | 0.818750 | 0.773145 | 0.917631 | 0.766265 |
| 2 | 0.773748 | 0.730721 | 0.871454 | 0.720046 |
| 3 | **0.761679** | **0.721204** | **0.856149** | **0.708736** |

epoch 3 在所有题型仍优于 epoch 2，说明三轮结果没有提供收敛证据。其 inference checkpoint SHA-256 为 `8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b`。

对应 DeepSeek paper140/base 产物曾通过以下审计：140 predictions、335/335 dimension scores、无重复和空回答；333 个 `final_score_top_logprobs`，2 个 `exact_sample_mean_20`；`errors=[]`、`ok=true`。产物身份：

| 产物 | SHA-256 |
|---|---|
| `predictions.jsonl` | `2b9cdb716f73471c9c0732bb6bdbca8012a2c2fcfe38a618672834d35b8231c7` |
| `geval.jsonl` | `c1a6166987280782a4c72b56d1a7e89ce37ed25776bc460458259837af83034b` |
| `table1.json` | `e8f879231a7fac97be43197c9c1e194b6324a5e8737f2c2f11a3bc19a759c6f0` |

## 因素结论

在固定 predictions 上把 DeepSeek 换成 Gemini，不会使所有题型一致地靠近论文：本仓库 epoch-3 的 MC 变近、OE 变远、TF 基本不变。故不能把论文差距简单归因于 judge。

恢复 series batching 会改变部分回答，是必须对齐的实现语义；在现有完整对照中，其聚合 Gemini 变化小于 checkpoint 权重效应。保持 per-record + Gemini 时，从本仓库 epoch-3 切换到作者候选 epoch-33，335 维配对均分提高约 0.176，bootstrap 95% CI 约 `[+0.033,+0.316]`。这支持“三轮训练不足”作为主要解释，但不证明论文明确训练了 33 轮。

PackyAPI 的 Gemini 2.5 Pro 兼容响应没有返回可用 logprobs；表中的 Gemini author 结果采用作者代码可执行的 temperature=0 整数回退语义。它们应连同这一接口限制一起引用。

## 当前复现判定

代码层面已经具备作者兼容的 checkpoint 加载、series-batched 推理、paper140 覆盖、Gemini 作者 prompt 和 fail-closed 审计，因此“运行协议”达到建立公平 baseline 的要求。

训练层面，仓库自身三轮 checkpoint 不足以代表训练充分的作者水平。正式三分支比较应先完成 seed-42、最多 35 epoch、完整验证集选模的 `main` 长预算基线，再以完全相同预算训练改进分支；最终只对验证选出的唯一 checkpoint 使用 Gemini。