# AXIS Baseline 正式复现结果

完成日期：2026-07-14  
正式范围：AXIS 论文 Table 1 第一行；Table 2 不在本轮范围。

## 1. 固定配置

- Phase-II 权重：本项目重新训练的 `epoch_3_inference.pth`，不是作者 released AXIS 权重。
- checkpoint SHA-256：`8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b`。
- 正式测试集：`paper140`，70 series / 140 QA，从同一次 full284/base 推理严格过滤。
- 生成：per-record、beam=5、`max_new_tokens=1000`。
- Judge：`deepseek-v4-pro`，high thinking，temperature=0，最终 score 位置 `top_logprobs=20`。
- 聚合：优先使用 1–5 score-token 概率均值；概率不完整时使用恰好 20 个 temperature=1 有效样本均值。
- 所有训练、推理、validation loss、API/G-Eval 和表格聚合均在 GPU 服务器执行。

## 2. Checkpoint 选择

三个 checkpoint 使用完全相同的 1500-series / 3000-QA validation 集。选择指标为全量
teacher-forced mean loss；测试集和 Table 1 judge 分数不参与选择。

| Epoch | Overall | Multiple Choice | Open Ended | True/False |
|---:|---:|---:|---:|---:|
| 1 | 0.818750 | 0.773145 | 0.917631 | 0.766265 |
| 2 | 0.773748 | 0.730721 | 0.871454 | 0.720046 |
| **3** | **0.761679** | **0.721204** | **0.856149** | **0.708736** |

Epoch 3 在 overall 和三个题型上都最低，选择没有题型冲突。机器可读记录位于
`experiments/reproduction/phase2_torch251_transfer/best_validation_loss.json`。

## 3. Table 1 第一行

| Model | MC Final | MC Corr. | MC Rsn. Qual. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 AXIS / Gemini-2.5 | 4.19 | 4.21 | 4.14 | 3.02 | 2.87 | 2.93 | 3.31 | 3.65 | 3.60 | 3.74 |
| **本 Baseline / DeepSeek-v4-pro** | **3.91** | **3.98** | **3.74** | **3.05** | **2.96** | **2.65** | **3.62** | **3.67** | **3.71** | **3.60** |
| 差值（Baseline − 论文） | -0.28 | -0.23 | -0.40 | +0.03 | +0.09 | -0.28 | +0.31 | +0.02 | +0.11 | -0.14 |

三个题型 Final 的等权 macro 为 `3.5417`；论文对应 macro 为 `3.6200`，差值 `-0.0783`。
OE Final 与 TF Final 分别只差 `+0.03` 和 `+0.02`；主要差距来自 MC，尤其 reasoning quality。
考虑到 judge 从 Gemini-2.5 换成 DeepSeek-v4-pro、作者未公开全部评测代码，这个量级没有显示
“实现完全跑偏”的灾难性错误，但不能据此声称与作者私有流程严格等价。

## 4. 完整性审计

- full284/base：284 predictions，284 unique records，无空回答、无重复。
- paper140/base：140 predictions，140 unique records，无空回答、无重复。
- G-Eval：335/335 dimension scores。
- 方法计数：333 个 `final_score_top_logprobs`；2 个 `exact_sample_mean_20`。
- 两个 fallback 均包含恰好 20 个有效分数；审计 `errors=[]`、`ok=true`。
- 严格 paper140 G-Eval 活跃阶段约 35 分钟，期间一次 SSH 隧道波动；JSONL 断点续跑未重复已落盘结果。

关键产物 SHA-256：

| 产物 | SHA-256 |
|---|---|
| `predictions.jsonl` | `2b9cdb716f73471c9c0732bb6bdbca8012a2c2fcfe38a618672834d35b8231c7` |
| `geval.jsonl` | `c1a6166987280782a4c72b56d1a7e89ce37ed25776bc460458259837af83034b` |
| `table1.json` | `e8f879231a7fac97be43197c9c1e194b6324a5e8737f2c2f11a3bc19a759c6f0` |

正式产物目录：`experiments/reproduction/test_epoch_3_paper140/`。其中
`audit.json`、`table1.json`、`table1.md`、`predictions.jsonl`、`geval.jsonl` 是交付核心。
验证集的历史 partial G-Eval、released-checkpoint 结果及带 `diagnostic` 的文件均未进入本表。

## 5. 如何使用这个 Baseline

后续改架构时，应保持 split、manifest、训练预算、decode 和 judge 配置不变，并首先比较全量
validation loss 与分题型免费指标。Table 1 只在里程碑/最终模型上重跑。更关键的是重跑
local/fixed channel intervention、label-changing evidence counterfactual、option permutation 和
explanation grounding；否则 Table 1 变高仍可能只是换了一种 shortcut。具体协议见 `readme.md`
第 10 节。