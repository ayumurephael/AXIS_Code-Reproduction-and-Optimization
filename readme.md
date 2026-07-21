# AXIS 完整复现代码库

本仓库是论文 **AXIS: Explainable Time Series Anomaly Detection with Large Language Models** 的可运行复现，用于建立可审计的基线并在相同协议下比较后续架构或损失函数改进。模型主体、Phase-II 训练、验证选模、作者兼容推理、G-Eval、完整性审计和表格生成均已纳入仓库；数据、基座模型、API 凭据和大体积 checkpoint 作为外部资产管理。

这不是临时实验目录。第一次接手时，不需要了解代码曾经如何修改，按本文档索引和固定协议即可运行。

## 分支

远端维护四个实验分支：

| 分支 | 用途 | 允许相对 `main` 改变的因素 |
|---|---|---|
| `main` | 论文 AXIS 基线复现 | 无 |
| `loss_resesign` | 损失函数改进 | 仅声明的损失与反事实训练目标 |
| `architecture_redesign` | 架构改进 | 声明的架构因素；需要时包含与该架构配套的训练目标 |
| `architecture_redesign_fixedhint_frozen` | Fixed-Hint 冻结架构改进 | `architecture_redesign` 全架构，并阻断状态损失到 task prompt 的梯度 |

当前分支的精确定义见 `BRANCH_PROFILE.md`。跨分支比较必须共用数据 manifest、Phase-I 权重、训练预算、选模规则、生成协议和 judge 配置。

## 从哪里开始

按以下顺序阅读：

1. `BRANCH_PROFILE.md`：当前分支改变了什么。
2. `REPRODUCTION.md`：从环境、训练到 Table 1 的唯一端到端流程。
3. `EVALUATION_PROTOCOL.md`：DeepSeek v4-pro 日常评测与 Gemini 2.5 Pro 正式评测。
4. `AUTHOR_COMPATIBLE_TRAINING.md`：长预算训练、验证选模和公平比较规则。
5. `REPRODUCTION_RESULTS.md`：经过完整性审计的参考结果及其适用边界。
6. `AUTHOR_CODE_COMPATIBILITY.md`：本实现与作者提供的 `AnomalyLlava-master` 补充快照的对应关系。

改进分支中的研究说明是设计依据，不是运行入口。下列文件在包含它们的分支中保留原文：

- `训练目标(损失函数)改进_new.md`
- `实验结论_AXIS缺陷.md`
- `AXIS架构改进说明.md`

若研究说明中的旧命令与上述五份执行文档冲突，以 `REPRODUCTION.md` 和脚本 `--help` 为准。

## 复现范围

当前正式目标是论文 Table 1 的 AXIS 行：

- `paper140`：作者测试脚本实际覆盖的 70 条序列、140 个 QA，是论文对齐口径；
- `full284`：发布测试集全部 142 条序列、284 个 QA，用于覆盖率和诊断；
- 正式生成：同一序列的 QA 组成 batch，beam size 5，`max_new_tokens=1000`，不计算 teacher-forced test loss；
- 正式评测：Gemini 2.5 Pro、作者 prompt/rubric、`paper140`、335 个题目—维度评分；
- 日常评测：DeepSeek v4-pro，复用同一 predictions，节省 Gemini 成本；
- checkpoint 只能由完整验证集指标选择，不能用 `paper140`、Gemini 或 DeepSeek 测试分数选模。

Table 2 消融工具仍可使用，但不应从不完整或历史实验文件推断论文 Table 2 结果。本仓库不复现其他基线模型或人类评价。

## 代码结构

```text
src/models/AXIS/                 AXIS 模型、时序编码器与数据实现
experiments/configs/             模型与运行配置
tools/axis_repro/                训练、推理、评分、审计与表格工具
experiments/reproduction/        运行产物；不是协议来源
data/                             外部数据，默认不提交
```

核心入口：

- `tools.axis_repro.train_phase2_memory_safe_v3`：`main` 的多卡 Phase-II 训练；
- `tools.axis_repro.run_inference_cli`：验证推理与作者兼容正式推理；
- `tools.axis_repro.geval_resilient`：DeepSeek v4-pro 断点续跑评测；
- `tools.axis_repro.geval_gemini`：Gemini 2.5 Pro 作者兼容评测；
- `tools.axis_repro.audit_results`：prediction/score/manifest 的 fail-closed 审计；
- `tools.axis_repro.table_runner`：生成 Table 1 指标。

改进分支的训练入口和消融参数由 `BRANCH_PROFILE.md` 指定。

## 评测模型政策

DeepSeek 与 Gemini 都受支持，但职责不同：

- 日常开发默认用 DeepSeek v4-pro；基线 predictions 和 scores 应缓存，不重复付费。
- 只有预先通过验证集和免费诊断的最终候选，才用 Gemini 2.5 Pro 做完整正式评测。
- 两个 judge 的绝对分数不可混合、拼接或直接当作同一量尺；跨模型比较时，候选与对应基线必须使用同一 judge、同一 prompt、同一脚本版本。
- PackyAPI 的 Gemini 2.5 Pro 兼容接口已知不返回可用 logprobs。正式作者兼容模式因此记录 temperature=0 的单次整数分；多次采样仅作校准，必须另存目录。
- API key 只从环境变量读取，禁止写入仓库、命令历史、日志或结果文件。

详细命令见 `EVALUATION_PROTOCOL.md`。

## 外部资产与硬件边界

训练和模型推理必须在 CUDA GPU 服务器上执行，推荐多卡；本地 CPU 只用于代码检查、单元测试、结果聚合和文档工作。运行前应准备：

- 基座 LLM（默认配置对应 DeepSeek-R1-Distill-Qwen-7B 或其本地镜像）；
- 训练数据 `data/anomaly_llava_training_dataset/`；
- 测试数据 `data/AXIS_qa_test/`；
- Phase-I checkpoint；
- 通过环境变量注入的 judge 凭据。

Gemini 只是 G-Eval 裁判，不能替换 AXIS 内部需要 embedding 注入和反向传播的本地冻结 LLM。

## 最低验收标准

任何可报告结果必须同时满足：

1. 记录 git commit、branch、环境版本、数据 manifest 和 checkpoint SHA-256；
2. 训练与验证无测试集泄漏；
3. 验证集覆盖完整，并按预先声明规则选唯一 checkpoint；
4. 正式生成使用 `paper140 + batching=series + skip-loss`；
5. `audit_results` 返回 `ok=true`，无缺失、重复、空回答或非法评分；
6. 分数注明 judge、model id、prompt 模板、scoring mode 和 endpoint 语义；
7. 三分支对照除声明因素外保持相同。

只满足“脚本能跑”不等于完成复现；只有协议、覆盖和产物身份全部可审计时，结果才可用于公平比较。