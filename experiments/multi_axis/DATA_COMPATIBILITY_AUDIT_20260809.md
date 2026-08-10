# Multi-AXIS 多模态数据兼容性审计（2026-08-09）

## 结论

- 当前 VLM 训练 manifest 与《说明文件.md》的权威配对、过滤和 grouped 90/10
  划分口径一致，没有训练/验证 `base_sample_id` 泄漏，也没有丢失图像 ID。
- VLM 服务器现已部署全部五个 bias-neutralized 正式测试集及 Teacher-Eval 478
  措辞诊断视图；六个 `eval_*.jsonl` 的条数与题型计数均通过审计，相关图像也已使用
  与训练集相同的 Times New Roman 字体包和 600 DPI 配置完成渲染。因此当前**训练、
  验证和测试数据均完整匹配**。
- 原始 Teacher-Eval 478 与 478new 是同一批 478 个样本的两种问题措辞视图，不能在
  正式五数据集 macro 中作为两个独立数据集重复计权。
- 权威训练数据自身含 52 组重复且冲突的自然语言监督目标。现有 manifest 忠实保留了
  上游权威数据；本次审计不静默删除或改写这些记录。

## 训练数据契约对照

| 检查项 | 说明文件权威值 | 当前 VLM manifest | 结果 |
|---|---:|---:|---|
| question shards | 62 | 62 | 匹配 |
| question pool | 76,998 | 76,998 | 匹配 |
| teacher pairs（过滤前） | 67,820 | 67,820 | 匹配 |
| 直接问题文本匹配 | 67,773 | 67,773 | 匹配 |
| 审计恢复 | 47 | 47 | 匹配 |
| structured-reference matches | 67,820 | 67,820 | 匹配 |
| 空 `model_answer` | 17 | 17 | 匹配 |
| 有效监督记录 | 67,803 | 67,803 | 匹配 |
| train / validation | 61,053 / 6,750 | 61,053 / 6,750 | 匹配 |
| train / validation groups | 15,943 / 1,771 | 15,943 / 1,771 | 匹配 |
| grouped split | seed 42，90/10 | seed 42，90/10 | 匹配 |
| train/validation base overlap | 0 | 0 | 匹配 |
| MC / TF / OE | 22,567 / 22,619 / 22,617 | 22,567 / 22,619 / 22,617 | 匹配 |
| 空 teacher（物化后） | 0 | 0 | 匹配 |
| 缺失 image ID | 0 | 0 | 匹配 |

## 上游重复监督风险

`sample_id` 不是数据记录的全局唯一键；同一 sample 可以有不同题目，因此仅凭
`sample_id` 重复不能判错。本审计进一步使用
`(sample_id, normalized_question_text)` 作为键：

| Split | 重复 sample ID 条数 | 重复“sample + 问题”组 | teacher 答案冲突组 | interval 冲突组 |
|---|---:|---:|---:|---:|
| train | 249 | 49 | 49 | 0 |
| validation | 21 | 3 | 3 | 0 |
| 合计 | 270 | 52 | 52 | 0 |

这 52 组具有相同 sample、相同规范化问题和相同 interval，但 teacher 自然语言答案
不同。它们来源于权威的 67,803 条保留记录，而不是 VLM 图像渲染或 grouped split
引入。由于正式训练的目标契约明确要求忠实使用 67,803 条，本实验继续保留；未来可做
“冲突去重/teacher 仲裁”消融，但必须生成新版本 manifest，不能覆盖本次正式数据。

## 测试视图审计

原始 Teacher-Eval 与 bias-neutralized Teacher-Eval 的 teacher-covered 子集均为
478 条，题型均为 MC 176 / OE 139 / TF 163。按 `sample_id` 对齐后：

- 共享 sample ID：478；任一视图独有：0；
- 数值时间序列与 target interval 完全一致：478/478；
- teacher 长答案完全一致：478/478；
- 规范化问题文本发生变化：29/478，其余 449 条文本相同。

因此 `478new` 是相同样本、相同 teacher 的措辞视图，而不是新增 478 个独立样本。
正式五数据集仍固定为 `478new`、SMD、SWaT、LEMMA-RCA、VTA；`478` 只用于措辞
敏感性诊断。

## VLM 测试部署门槛

2026-08-10 的服务器部署已满足以下门槛：

1. 所选测试集的 question/teacher 原文件 SHA-256 与权威源一致；
2. manifest 数量及 MC/OE/TF 计数与 `tools/multi_axis/datasets.py` 一致；
3. 每条记录均有可解析 `image_id`，渲染图像与训练使用相同的 Times 字体包、600 DPI、
   renderer version、像素预算和半开区间语义；
4. `audit_evaluation.py` 对预测、标签指标与 Judge dimension 任务全部通过；
5. 478 与 478new 同时报表时显式标注 wording-view macro，不纳入正式五数据集 macro
   的重复权重。

## 测试部署结果（2026-08-10）

| Manifest | 总数 | MC | OE | TF | 状态 |
|---|---:|---:|---:|---:|---|
| `eval_478new.jsonl` | 478 | 176 | 139 | 163 | 通过 |
| `eval_478.jsonl` | 478 | 176 | 139 | 163 | 通过（措辞诊断视图） |
| `eval_SMD.jsonl` | 200 | 66 | 67 | 67 | 通过 |
| `eval_SWaT.jsonl` | 184 | 61 | 62 | 61 | 通过 |
| `eval_LEMMA-RCA.jsonl` | 12 | 4 | 4 | 4 | 通过 |
| `eval_VTA.jsonl` | 200 | 67 | 67 | 66 | 通过 |

图像审计共覆盖 34,351 个唯一图像，manifest SHA-256 为
`dbd45e38d343228cf49c374f8c82308113ba7709b0d0b1cd67e1d473e3e1a4c8`。
渲染器记录为 `multi-axis-vl-render-v1`，DPI 为 600，字体族为 Times New Roman，
字体包聚合 SHA-256 为
`1f53e60db60e2eb72b0e52dad9b879290998d1c58bc9510201a6d710145e1147`。
`EVAL_DEPLOY_COMPLETE` 与结构化 `evaluation_deployment_audit.json` 均已生成；后续正式
推理仍须逐项运行预测、标签指标和 Judge 维度审计，不能仅凭部署审计视为测评完成。
