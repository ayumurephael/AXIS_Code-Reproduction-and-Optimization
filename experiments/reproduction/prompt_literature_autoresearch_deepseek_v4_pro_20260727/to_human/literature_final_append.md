# 文献驱动 Prompt Autoresearch 最终补充（2026-07-27）

## 结论

本轮完整阅读 `prompt系列改进.md`，结合 Time-LLM、选择题符号绑定与顺序偏差、上下文校准、输出格式偏差、RE2、CoT/小模型提示敏感性等原始论文，随后在不训练模型的前提下完成 30 个新模式的上限搜索。所有正式推理均在授权 GPU 上完成，所有正式评分仅使用 `deepseek-v4-pro`。

最终没有 Prompt 同时通过 development96、screening24、validation72 和 exposed holdout48 的预注册门槛。因而触发停止规则，没有锁定胜者，也没有生成`paper140` 候选输出。该结果不能表述为“找到了正式测试集上的全面提升 Prompt”。

## 最接近的内部结果

| 候选 | 数据层 | MC Final Δ | MC Corr. Δ | MC Rsn. Δ | TF Final Δ | TF Corr. Δ | TF Justif. Δ | c→w | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lit_r4_01_tf_neg_re2` | exposed holdout48 | +0.0000 | +0.0000 | +0.0000 | +0.2000 | +0.1250 | +0.3125 | 1 | Table-I 全非降，但违反零伤害门槛 |
| `lit_r5_01_mc_structured_guard` | exposed holdout48 | +0.5471 | +0.5294 | +0.5882 | +0.0000 | +0.0000 | +0.0000 | 0 | holdout 通过，但 validation72 三项 MC 下降 |

OE 始终复用精确 Baseline，因此四项 OE 指标均为 0 差值。完整的 5 候选 × 4 数据层 × 10 指标、绝对分数、配对胜负和失败 case 见 `prompt_literature_autoresearch_deepseek_v4_pro_20260727/experiments/literature-round-5/analysis.md`。

## 失败 case 归因

- `series_000103:1`：structured guard 把 Baseline 正确的中心向上尖峰改判为缓慢下降后回归，说明阈值规则重新分配了注意力，而非只抑制假阳性。
- `series_000083:1`、`series_000089:1`：status-before-shape 先形成“正常”锚点，随后用正常/振荡选项解释数据，覆盖了 checkpoint 原本正确的边界尖峰识别。
- `series_000111:1`：TF RE2 虽总体修正更多极性错误，却把明确尖峰解释成渐进变化，使 `False→True` 并造成约 −1.60 的 Final 下降。
- `series_000022:1`：完整选项语义比较确实能把 Baseline 的边界尖峰误选修正为局部快速振荡，证明收益机制真实存在；但它与上述伤害机制不能靠当前 prompt 稳定分离。

## 研究判断

失败的共同原因不是 Judge 或解析器，而是 7B checkpoint 对训练时 prompt 分布高度敏感。新增自然语言规则会同时改变“读证据、判异常、匹配选项、组织答案”四个过程。即使 aggregate Table-I 指标上涨，也可能隐藏少数灾难性反转。在当前证据下，发布 Baseline 仍是唯一满足严格零回归要求的 prompt。

## 关键复现配置

- GPU 源提交：`4d3ace60c036cdb231a4c5aeafeec3cdf030aace`。
- checkpoint SHA-256：`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。
- development：192 条原始 GPU 输出、83 个路由分量、136 个新 Judge 维度；组装后 576 predictions / 1326 scores。
- holdout：144 条原始 GPU 输出、44 个路由分量、88 个新 Judge 维度；组装后 288 predictions / 666 scores。
- 审计：模型、provider、方法、prompt SHA-256、record/mode/dimension 唯一性和 manifest 均通过。
- `paper140`：候选调用 0 次；未用于选择或调参。

## 本轮 Prompt 全文索引

下面的最终附录由实现代码直接渲染，覆盖 30 个 literature-guided 模式及全部实际路由分支；每个分支附 SHA-256。为便于后续观察，该附录必须保持为本文档最后一部分。
