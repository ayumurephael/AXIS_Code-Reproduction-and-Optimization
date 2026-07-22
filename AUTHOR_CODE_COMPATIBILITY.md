# AXIS 与作者补充代码的兼容关系

作者提供的 `AnomalyLlava-master` 是有价值的补充实现快照，但它不是论文 AXIS 私有仓库的完整发布版。本仓库以论文 AXIS 语义和现有 `src/models/AXIS/` 为主体，同时吸收补充快照中可验证的训练、checkpoint、推理和 G-Eval 行为。

## 关系概览

| 层面 | 重合 | 差异或冲突 | 本仓库采用的口径 |
|---|---|---|---|
| 模型思想 | 时序 encoder、可训练 Hint Tuner/Perceiver、冻结 LLM、软提示注入 | 补充快照主要命名为 Moirai/AnomalyLlava，不等同论文最终 AXIS 文件树 | 保留 AXIS 模型主体 |
| Phase I | 时序特征进入语言模型前的编码路径 | checkpoint 名称、包装层和配置路径不同 | 外部固定 Phase-I checkpoint，并记录 hash |
| Phase II | 只训练提示相关模块，冻结 encoder/LLM | 作者完整训练 main、实际早停和有效 epoch 未全部公开 | 提供可运行多卡训练和完整验证选模 |
| checkpoint | 都保存可训练模块权重 | 作者 Accelerate 候选 keys 可能统一带 `perceiver.` 前缀 | `model_utils` 严格兼容两类前缀；混合/缺失 key 失败 |
| 测试 batching | beam search、同一数据问答 | 旧复现曾逐 QA；作者测试按同一 series 组成 batch | 正式 `--batching series --skip-loss` |
| 覆盖 | 发布 QA 数据一致 | 作者测试脚本实际只覆盖 70 series / 140 QA | `paper140` 正式；`full284` 只作覆盖诊断 |
| G-Eval | 1–5 分、MC/OE/TF 分维度加权 | prompt 布局、score 概率提取和无 logprobs 回退可能不同 | Gemini 正式使用 author template/mode；DeepSeek 独立日常口径 |

## 模型实现差异

静态对照显示补充快照与 AXIS 主体并非可以直接覆盖替换：

- 模型类、配置名、数据入口和 checkpoint 包装层不同；
- 可训练层初始化存在 Kaiming/Xavier 等实现差异；
- EOS 标签监督、局部 embedding dtype 和推理消融开关并不完全一致；
- 某些补充脚本是更早模型或其他基线的入口，不能视为论文 AXIS 的最终规范；
- 补充 snapshot 的训练参数与论文文字、候选 checkpoint 状态之间仍有不完全可消解的不一致。

因此，本仓库没有把 `AnomalyLlava-master` 整树复制进运行时，也不从其文件名推断“官方真值”。只采用能够由 checkpoint、输出或受控实验验证的行为。

## 已统一的作者兼容行为

### checkpoint 加载

`tools.axis_repro.model_utils` 支持：

1. 本仓库直接保存的 Perceiver state keys；
2. 作者 Accelerate 候选中全部统一带 `perceiver.` 前缀的 keys。

只有当全部 keys 具有一致前缀时才归一化。混合前缀、缺 key、额外 key 或结构不匹配仍按 strict loading 失败，避免静默加载错误权重。

### 推理

`tools.axis_repro.run_inference_cli` 同时支持：

- `per_record`：逐 QA，适合完整 validation loss 和旧结果复核；
- `series`：同一时间序列的 QA batch，正式论文对齐路径；
- `--skip-loss`：正式 test generation 不先计算 teacher-forced loss。

`tools.axis_repro.run_inference_series_legacy` 是作者语义的独立核验入口，不是第二套正式 pipeline。正式命令统一走 `run_inference_cli`。

### 评测

`tools.axis_repro.geval_gemini --prompt-template author --scoring-mode author` 对齐作者补充 G-Eval 的 rubric、权重和可执行回退语义。Gemini 兼容接口没有返回 logprobs 时保存整数分，并在结果中明确记录能力缺失；`strict` 多次采样只作校准。

DeepSeek runner 保留原分维度概率/采样逻辑，服务日常低成本实验。它不是 Gemini 的数值替身，结果必须独立标注。

## 对复现差距的含义

受控结果表明：

- 单独把 DeepSeek judge 换成 Gemini 不会使 MC/OE/TF 全部更接近论文；
- series batching 确实改变生成文本，应作为正式协议修正，但已观察到的总体评分效应较小；
- 作者候选 epoch-33 相对本仓库 epoch-3 的权重效应最大，并使十项指标整体接近论文；
- 本仓库 epoch-3 的 validation loss 仍下降，训练不足是比“模型代码完全错误”更有证据的解释。

结论边界是：当前代码已能执行作者兼容的运行路径；从头训练能否稳定达到相同水平，仍需按固定长预算和完整 validation 选模验证。不能为了逼近测试表格而选择 epoch 或 seed。

## 使用边界

- 不把补充快照称为论文完整官方代码。
- 不直接加载结构不兼容权重。
- 不从 test/Gemini 分数推断训练早停点。
- 不把 per-record 诊断结果当作正式作者协议。
- 不把 DeepSeek 与 Gemini 分数放在同一统计样本中聚合。
- 不将作者参考源码、外部 checkpoint 或凭据提交到任何运行分支。

## Paper, released code, and checkpoint architecture boundary

Section 3.2 uses local time-series states as queries over a prototype bank. Shared learned queries `P_fix` also attend to the same prototype bank. The paper does not define Fixed Hint as a direct soft prompt, and it does not describe a Local-Hint continuous residual bypass or QK-Norm.

The released materials contain a dimension conflict that must be recorded:

| Item | Appendix B.3 | Released code/checkpoint |
|---|---:|---:|
| TS encoder layers | 6 | 8 |
| prototypes | 1024 | 1000 |
| Fixed queries | 8 | 30 |

The released Phase-II checkpoint has `fix_prompt_embeddings` shape `[1, 30, 3584]` and `mapping_layer.weight` shape `[1000, 151667]`, with exactly the original nine Perceiver tensors. When author weights must load strictly, 30/1000/8 is the executable standard. Changing to 8/1024/6 makes that checkpoint structurally incompatible.

## Author checkpoint inventory

The three files under `baseline_author/experiments/checkpoints` were inspected on CPU. Epoch and loss values below come from payloads, not filenames. The supplemental trainers iterate `range(num_epochs)`, so the stored epoch fields are zero-based under that code; raw `epoch=33` can denote the 34th pass of an uninterrupted run. The private training history is not present, so reports should preserve the raw field rather than silently converting it.

The Phase-I payload config names a Llama-8B LLM, but Phase II uses 3584-wide Qwen-7B Hint-Tuner tensors. Phase II only consumes the Phase-I time-series module, so the Phase-I `llm_config.model_name` must not be used to infer the Phase-II backbone.

| File | SHA-256 | Payload and role |
|---|---|---|
| `pretrain_single/pretrain_checkpoint_best.pth` | `2f1507fcc3c3232d375dcb0c18bfcd11f2ec9e184dcb1cc9fb603f0f38c94a2e` | Phase-I best: `epoch=22`, `loss=0.04245388498529792`. Contains 111 TS-encoder/reconstruction/anomaly-head tensors, optimizer state, and training config. Fresh Phase II loads its TS module and freezes it. |
| `axis_qa_by_pretrain_best_accelerate/model_optimizer.pth` | `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7` | Author Phase-II validation-best candidate: `epoch=33`, `avg_loss=0.7127881973981858`. Contains 111 frozen TS tensors and 9 BF16 Hint-Tuner tensors. |
| `axis_qa_by_pretrain_step_accelerate/model_optimizer.pth` | `549b776614edc09336cf00ced13dff9f1cad51c245f4ca54ba1d855b92fce0b7` | Last overwritten periodic step candidate within epoch 33. `avg_loss=0.7364754394601621` is the running training mean at that save. No batch/global step is stored, so the exact step cannot be recovered. Use only if best is missing or an experiment explicitly requests it. |

Despite the filename `model_optimizer.pth`, both Phase-II payloads contain only `epoch`, `model_state_dict`, and `avg_loss`; neither contains `optimizer_state_dict`. They support inference or continued fine-tuning with a newly created optimizer, not restoration of AdamW moments or a scheduler. Equal file size also does not imply equal weights.

The author test code tries `best_accelerate` before `step_accelerate`. This repository likewise forbids silently substituting step when best exists.

## Using the author Phase-II checkpoint to test a new loss

Direct continued training from author `best_accelerate` is useful as a low-cost post-training add-on experiment, but by itself it does not prove that a new loss is better than answer NLL:

1. the author model is already optimized through epoch 33, so auxiliary objectives see a different initial state;
2. the checkpoint has no optimizer state, so continuation resets the optimizer;
3. extra answer-NLL steps can change results even without auxiliary losses;
4. its training history and budget differ from a method trained from Phase I.

The minimum matched add-on experiment clones the same author Phase-II best state twice:

- control: new identical optimizer, answer NLL only;
- treatment: new identical optimizer, answer NLL plus the new auxiliary objective.

Use the same extra steps, data order, learning rates, validation schedule, and selection rule. This answers whether the new loss helps as post-training on an already trained AXIS.

The stronger question--whether adding the loss to Phase-II training is globally better than the author objective--still requires answer-only and loss-final runs from the same Phase-I checkpoint, with identical total budget and full-validation selection. Author best is a loading/inference reference or a matched post-training initializer, not a replacement for that from-Phase-I control.
