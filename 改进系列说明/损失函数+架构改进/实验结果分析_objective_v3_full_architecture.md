# Objective-v3 全架构实验结果分析：MC 提升，但整体尚未超越 Baseline

> 完成日期：2026-07-18  
> 分支：`architecture_redesign`  
> 代码提交：`c5b3e55537999a67265c24922b375d7b7a7f9e8a`  
> 正式变体：Objective-v3 + QK-Norm + Continuous bypass / prototype sidecar + 直接学习 K=30 task soft-prompt tokens  
> 训练策略：argmin donor（实现元数据为 Top-M 框架且 `M=1`，与 argmin 等价）  
> 正式随机种子：72（单 seed）

## 技术摘要

本轮改进取得了**明确但局部的收益**：Multiple Choice（MC）三项指标均超过正式 Baseline，绝对提升为 `+0.1670` 至 `+0.2061`，相对提升约 `+4.46%` 至 `+5.18%`。第一版 SLR-redesign 中最严重的 MC reasoning collapse（`3.74 → 1.93`）也被修复，本轮 MC reasoning 达到 `3.9070`。

但是，本轮不能被表述为“整体优于 Baseline”：

- 十项正式指标中仅 3 项提升、7 项下降；
- Open Ended（OE）四项全部下降，幅度约 `-4.79%` 至 `-7.08%`；
- True/False（TF）三项全部下降，其中 TF correctness 下降 `-0.4005`（约 `-10.79%`）；
- 三题型 Final 的等权 Macro Final 从 Baseline 报告值 `3.5417` 降至 `3.4486`，下降 `-0.0931`（约 `-2.63%`）；
- 最佳完整验证 teacher-forced mean NLL 从 Baseline 的 `0.761679` 上升至 `0.928742`，恶化 `+21.93%`（NLL 越低越好）；
- 可训练参数增加 `13.05%`，训练时间增加 `30.64%`，全局吞吐下降 `23.45%`。

最值得后续关注的数据瓶颈是：57,000 个训练 QA 中只有 8,636 个有效事实—反事实 pair，有效率仅 `15.15%`；正常 anchor 的异常 donor 只有 419 个被实际使用，单 donor 最大复用 271 次。当前的显式状态目标虽然在训练日志快照上达到约 `93.79%` 的 epoch-3 状态准确率，但它只覆盖有效 pair，且尚无独立 held-out counterfactual-state 测试，因此不能据此宣称 evidence grounding 已被解决。

总体判断是：**当前设计很可能改善了 MC 的答案选择与推理接口，并修复了第一版损失的明显坍缩，但仍存在生成质量、TF 判断和答案似然上的系统性代价。** 下一阶段应优先做组件消融、`beta` 扫描和 donor 多样性实验，而不是直接将全量组合视为最终架构。

## 1. 与正式 Baseline 的十项指标对比

### 1.1 比较口径

正式 Baseline 取自 `readme.md` 与 `REPRODUCTION_RESULTS.md` 的 Table 1 第一行：重新训练的 Baseline epoch-3 checkpoint、固定 paper140、DeepSeek-v4-pro 严格 Appendix-E G-Eval。当前实验使用同一 paper140 manifest、同一题型权重和同一 Judge 模型，因而是当前最可比的对照。

绝对差值和相对变化定义为：

$$
\Delta = \mathrm{Current} - \mathrm{Baseline},
\qquad
\mathrm{Relative\ Change}=\frac{\Delta}{\mathrm{Baseline}}\times100\%.
$$

十项 G-Eval 指标均为 1–5 分，**越高越好**。Baseline 表格只保留两位小数，因此下表差值和相对比例是基于已发布舍入值计算的近似值；当前值保留完整聚合精度。

### 1.2 正式结果

| 指标 | Baseline | 本轮方法 | 绝对差值 | 相对提升/下降 | 方向 |
|---|---:|---:|---:|---:|---|
| MC Final | 3.9100 | 4.1023 | **+0.1923** | **+4.92%** | 提升 |
| MC Correctness | 3.9800 | 4.1861 | **+0.2061** | **+5.18%** | 提升 |
| MC Reasoning Quality | 3.7400 | 3.9070 | **+0.1670** | **+4.46%** | 提升 |
| OE Final | 3.0500 | 2.8673 | -0.1827 | -5.99% | 下降 |
| OE Accuracy | 2.9600 | 2.8182 | -0.1418 | -4.79% | 下降 |
| OE Completeness | 2.6500 | 2.4909 | -0.1591 | -6.00% | 下降 |
| OE Relevance | 3.6200 | 3.3636 | -0.2564 | -7.08% | 下降 |
| TF Final | 3.6700 | 3.3762 | -0.2938 | -8.01% | 下降 |
| TF Correctness | 3.7100 | 3.3095 | **-0.4005** | **-10.79%** | 下降 |
| TF Justification Quality | 3.6000 | 3.4762 | -0.1238 | -3.44% | 下降 |

| 汇总指标 | Baseline | 本轮方法 | 绝对差值 | 相对变化 |
|---|---:|---:|---:|---:|
| 三题型 Final 等权 Macro | 3.5417 | 3.4486 | -0.0931 | -2.63% |
| 提升指标数 | — | 3 / 10 | — | — |
| 下降指标数 | — | 7 / 10 | — | — |

`REPRODUCTION_RESULTS.md` 报告的 Baseline Macro Final 为 `3.5417`，它来自未舍入内部值；若直接对表中两位小数的 `3.91/3.05/3.67` 求平均会得到 `3.5433`。本报告的 Macro 对比使用正式报告值 `3.5417`，而单项相对变化使用表中公开的两位小数值。

### 1.3 分题型解释

**MC 是本轮唯一稳定改善的题型。** correctness 与 reasoning quality 同时提升，说明收益不只是“选对但不解释”。这与第一版 SLR-redesign 的裸选项、短标签和 reasoning collapse 明显不同。

**OE 的下降呈全维度一致性。** accuracy、completeness、relevance 同时下降，说明问题不只出在冗长或格式；更可能是答案内容、覆盖度和问题相关性共同受损。

**TF 的主要问题是判断正确性，而非仅解释风格。** TF correctness 下降约 `10.79%`，大于 justification quality 的 `3.44%` 降幅。显式 normal/anomalous 状态监督并未自动转化为对自然语言命题 True/False 的更好判断，这与“窗口是否异常”和“题干命题是否为真”并非同一标签空间相一致。

### 1.4 与第一版 SLR-redesign 的历史对照（非主要 Baseline）

该表只用于判断 objective-v3 是否修复了第一版损失的已知退化；由于当前实验同时加入架构改造，不能把差异单独归因于损失函数。

| 指标 | 正式 Baseline | 第一版 SLR-redesign | 本轮全量方法 |
|---|---:|---:|---:|
| MC Final | 3.91 | 3.30 | 4.1023 |
| MC Correctness | 3.98 | 3.88 | 4.1861 |
| MC Reasoning Quality | 3.74 | 1.93 | 3.9070 |
| OE Final | 3.05 | 2.87 | 2.8673 |
| OE Accuracy | 2.96 | 2.71 | 2.8182 |
| TF Final | 3.67 | 2.88 | 3.3762 |
| TF Correctness | 3.71 | 2.95 | 3.3095 |
| TF Justification Quality | 3.60 | 2.79 | 3.4762 |
| Macro（报告值） | 3.54 | 3.02 | 3.4486 |

本轮基本修复了第一版 SLR 的 MC reasoning collapse，并部分恢复 TF；但 OE Final 几乎没有恢复到 Baseline 以上，TF 也仍低于 Baseline。

## 2. 评测集合、分母与聚合定义

### 2.1 数据范围

| 阶段 | 数据范围 | series | QA / prediction rows | 审计 |
|---|---|---:|---:|---|
| Phase-II train | 固定 95% series-level split | 28,500 | 57,000 QA | 训练完成 28,500 steps |
| Validation（每个 epoch） | 固定 5% series-level split | 1,500 | 3,000 QA | epoch 1/2/3 均 `ok=true` |
| full284 inference | `AXIS_qa_test` full/base | 142 | 284 | `ok=true` |
| paper140 | 从同一次 full284 严格过滤 | 70 | 140 | `ok=true` |
| DeepSeek G-Eval | paper140 的全部维度 | 70 | 335 dimension scores | `ok=true` |

paper140 的题型分布：

| 题型 | QA 数 | 评分维度数 | 每条记录的维度 |
|---|---:|---:|---|
| Multiple Choice | 43 | 86 | correctness、reasoning quality |
| Open Ended | 55 | 165 | accuracy、completeness、relevance |
| True/False | 42 | 84 | correctness、justification quality |
| **合计** | **140** | **335** | — |

### 2.2 Final 分数权重

每条记录先按以下固定权重计算 Final，再在同题型内求均值；Macro Final 是 MC/OE/TF 三个 Final 的等权平均，而不是按样本数加权。

| 题型 | Final 计算方式 |
|---|---|
| MC | `0.7 × correctness + 0.3 × reasoning_quality` |
| OE | `0.35 × accuracy + 0.35 × completeness + 0.3 × relevance` |
| TF | `0.6 × correctness + 0.4 × justification_quality` |

### 2.3 Checkpoint 选择指标

Checkpoint 只根据完整 1,500-series validation 上的 **teacher-forced mean answer NLL** 选择，测试集与 G-Eval 分数不参与选择。NLL 越低越好。状态 CE 不直接进入 checkpoint 选择指标。

## 3. 方法与正式训练配置

### 3.1 Objective-v3

完整训练目标为：

$$
\mathcal L
=
\mathcal L_{\mathrm{answer}}
+\beta\mathcal L_{\mathrm{state}}.
$$

- `answer` 分支只在真实上下文上训练完整自然语言答案；
- `state` 分支使用与原 QA 独立的固定状态问题；
- 每个有效 pair 同步包含真实与反事实状态，目标分别为 $z_i$ 和 $1-z_i$；
- Window 与 Local 都由同一条完整反事实序列生成，避免来源不一致；
- 冻结 LLM 的 next-token vocabulary logits 只抽取两个 verbalizer，并在两类之间重新归一化；
- normal/anomalous verbalizer 实际选择裸数字 `0/1`，token id 为 `15/16`；
- LLM 和 Phase-I TS encoder 冻结，但不切断从 LLM 输出到 soft embedding 的梯度；
- soft embedding 使用非原地 `index_copy` 注入，避免 autograd 失效。

### 3.2 架构变体

| 组件 | 正式设置 | 作用 |
|---|---|---|
| QK-Norm | 开启 | 对 Q/K 做 L2 normalization，限制 attention 尖锐度由统一尺度控制 |
| QK 序列长度 $L$ | 40（CLI 固定） | 用于初始化可学习尺度 |
| QK scale 初值 | 10.607330 | 等于 $\log_2(40^2-40)$ |
| Continuous bypass | 开启 | 让连续 TS 特征直接投影到 LLM embedding 空间 |
| Prototype sidecar | 开启 | prototype 通道负责语义辅助，不再承载全部连续信息 |
| Gate bias | -2.0 | 初始 sigmoid gate 约为 0.1192，偏向连续 bypass |
| Task soft prompt | 直接学习 | 不再通过 prototype 生成固定常量 |
| Task token 数 $K$ | 30 | 与原 AXIS fixed/task token 数一致 |

### 3.3 训练与优化器

| 配置项 | 本轮正式值 |
|---|---|
| 基座 LLM | DeepSeek-R1-Distill-Qwen-7B |
| Phase-I checkpoint SHA-256 | `2f1507fcc3c3232d375dcb0c18bfcd11f2ec9e184dcb1cc9fb603f0f38c94a2e` |
| seed | 72 |
| epochs / synchronized steps | 3 / 28,500 |
| steps per rank per epoch | 9,500 |
| world size | 3 |
| micro batch / GPU | 1 series |
| global batch | 3 series |
| gradient accumulation | 1 |
| optimizer | AdamW |
| local continuous LR | `5e-5` |
| prototype attention LR | `2e-5` |
| task prompt LR | `5e-5` |
| weight decay | `1e-5` |
| gradient clipping | 1.0 |
| state-loss $\beta$ | 前 10% steps 从 0 线性升至 0.2，之后固定 0.2 |
| precision | bf16 autocast（训练）；fp16 autocast（推理） |
| 冻结模块 | Phase-I TS encoder、LLM |
| checkpoint interval | 每 5,000 synchronized steps，另保存每个 epoch |

`训练目标(损失函数)改进_new.md` 的默认建议是 warm-up 到 `beta=0.1`；本轮按用户正式设置使用 `beta=0.2`。这一区别是后续解释 OE/TF 退化时的重要超参数因素。

### 3.4 可训练参数

| 参数组 | 参数量 | 占全部可训练参数 |
|---|---:|---:|
| Local continuous | 27,543,040 | 11.94% |
| Prototype attention | 203,048,225 | 88.01% |
| Task prompt | 107,520 | 0.05% |
| **合计** | **230,698,785** | **100%** |

Baseline 可训练参数为 204,076,832；本轮增加 26,621,953（`+13.05%`）。新增容量几乎全部来自 continuous/prototype 架构，而不是 30 个 task prompt token。

### 3.5 运行环境

| 项目 | 正式环境 |
|---|---|
| GPU | 3 × NVIDIA A100-PCIE-40GB（GPU 0/1/2） |
| GPU 3 | 始终不用于本实验 |
| Driver | 550.54.15 |
| Python | 3.10.5 |
| PyTorch / CUDA runtime | 2.5.1+cu124 / 12.4 |
| cuDNN | 9.1.0 |
| Transformers | 4.45.2 |
| NumPy | 1.26.1 |

## 4. 反事实索引与显式状态监督覆盖率

### 4.1 构造策略

- 异常 anchor：使用同坐标 paired normal patch 删除异常；
- 正常 anchor：移植异常 donor 的残差；
- Window 距离空间：`round(value × 100)`；
- Local effect：冻结 Phase-I encoder 表示；
- `tau=0.25`；
- `gamma_window=0.454859`，`gamma_local=0.437214`，来自 5% percentile 规则；
- same-length 为硬约束；
- phase fallback 关闭；
- patch 后同时验证 Window 与 Local；
- donor policy 元数据为 `uniform_top_m_by_window_distance`，但 `donor_top_m=1`，因此实际就是确定性最近邻 argmin。

### 4.2 覆盖率与 donor 集中度

| 指标 | 数值 | 解读 |
|---|---:|---|
| 总 series | 28,500 | 训练 split |
| 总 QA anchors | 57,000 | 每 series 两个 QA |
| 正常 anchors | 37,047 | 65.00% |
| 异常 anchors | 19,953 | 35.00% |
| 有效正常→异常残差移植 | 5,269 | 正常 anchor 的 14.22% |
| 有效异常→正常删除 | 3,367 | 异常 anchor 的 16.87% |
| **总有效 pair** | **8,636** | **15.15%** |
| 无效 anchors | 48,364 | 84.85% |
| 实际使用的 unique residual donors | 419 | donor 多样性有限 |
| 单 donor 最大复用 | 271 | 存在明显集中复用 |
| donor 复用次数 p95 | 47.2 | 长尾明显 |
| top donor 占比 | 5.14% | 单 donor 贡献约 1/20 有效移植 |
| effective donors | 160.06 | 显著低于 419 个名义 unique donors |
| 平均 eligible donor 数 | 1.573 | 直接切换 M=8/16 时，大量 anchor 实际池仍小于 M |
| 平均 match ratio | 0.1768 | 背景兼容候选较稀疏 |
| match ratio p95 | 0.2453 | — |
| index 构建时长 | 52.90 s | 非主要运行瓶颈 |

该结果直接支持用户此前对 argmin donor 复用的担忧。Top-M 随机采样是合理方向，但仅把 `M` 从 1 改为 8/16 并不充分：当前平均 eligible donor 只有 1.573，必须同时扩大合法候选池或改进检索表征，否则多数样本仍退化为 M≈1。

反事实索引 SHA-256：`cc0944f9e5bcd70b6b72fdfb398abf8ef439734ed5aa7693ddda337bee1b38d0`。

## 5. 训练过程与资源成本

### 5.1 与 Baseline 的成本对比

| 指标 | Baseline | 本轮方法 | 绝对差值 | 相对变化 | 偏好方向 |
|---|---:|---:|---:|---:|---|
| 训练时长 | 5.6940 h | 7.4388 h | +1.7448 h | +30.64% | 越低越好 |
| step/s/rank | 1.3907 | 1.0645 | -0.3262 | -23.45% | 越高越好 |
| 全局 series/s | 4.1710 | 3.1927 | -0.9783 | -23.45% | 越高越好 |
| 可训练参数 | 204,076,832 | 230,698,785 | +26,621,953 | +13.05% | 视性能收益而定 |
| 最佳 validation NLL | 0.761679 | 0.928742 | +0.167063 | +21.93% | 越低越好 |

正式训练 wall time 为 26,779.73 秒（7 小时 26 分 20 秒），完成 28,500 steps；没有 NaN、Inf、Traceback、OOM 或 NCCL fatal signature。

### 5.2 周期日志快照

训练日志每 20 steps 记录一次聚合快照，共 1,425 个。下表的 state loss 均值包含“该快照没有有效 pair 时记 0”的行；`logged valid pairs` 只统计这些周期快照，不等于全训练所有有效 pair 次数。

| Epoch | 快照数 | Answer loss 均值 | State loss 均值 | Total loss 均值 | Grad norm 均值 / 最大 | Logged valid pairs | 加权 state acc. | 真实 / 反事实 acc. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 475 | 1.0778 | 0.1840 | 1.0841 | 2.7296 / 14.2731 | 436 | 87.61% | 86.70% / 88.53% |
| 2 | 475 | 0.9980 | 0.1176 | 0.9996 | 2.4899 / 7.2342 | 393 | 93.00% | 92.11% / 93.89% |
| 3 | 475 | 0.9597 | 0.1070 | 0.9611 | 2.4165 / 6.8115 | 435 | 93.79% | 91.26% / 96.32% |

观测到 answer/state/total loss 逐 epoch 下降，grad norm 的均值和最大值也下降，说明训练稳定且未发生典型数值爆炸。状态准确率上升主要由反事实侧提升驱动；真实状态侧 epoch 3 为 91.26%，低于反事实侧 96.32%，提示两类上下文仍可能存在难度或构造分布差异。

### 5.3 Beta warm-up 与最后状态

| Step | Beta | 说明 |
|---:|---:|---|
| 20 | 0.001334 | 首个周期日志 |
| 2,800 | 0.196490 | 接近 10% warm-up 终点 |
| 2,840 | 0.199298 | 线性增长正常 |
| 2,860 | 0.200000 | 达到目标 beta |
| 28,500 | 0.200000 | 训练结束 |

最后快照（step 28,500）：answer loss `0.896769`、state loss `0.176068`、total loss `0.898511`、grad norm `2.358206`、state accuracy `1.0`（该快照只有 1 个有效 pair）。单个最后快照不能替代 epoch 聚合统计。

### 5.4 Checkpoint 剥离与审计

| Epoch | Global step | 完整 checkpoint | 推理 checkpoint | 体积减少 | Objective audit | Architecture audit |
|---:|---:|---:|---:|---:|---|---|
| 1 | 9,500 | 2,916,684,896 B | 1,071,080,030 B | 63.28% | `ok=true` | `ok=true` |
| 2 | 19,000 | 2,916,684,896 B | 1,071,080,030 B | 63.28% | `ok=true` | `ok=true` |
| 3 | 28,500 | 2,916,684,896 B | 1,071,080,030 B | 63.28% | `ok=true` | `ok=true` |

epoch-3 inference checkpoint SHA-256：`a24724361fad3647abf20985a9e4d4a38c634bcc241ee6bf33d4087ca0e1a81f`。

## 6. 完整验证与 checkpoint 选择

### 6.1 本轮三个 epoch 的 teacher-forced NLL

| Epoch | Overall | MC | OE | TF | 覆盖审计 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.983609 | 0.920550 | 1.101648 | 0.928745 | 3,000 rows / 1,500 series，`ok=true` |
| 2 | 0.949135 | 0.889380 | 1.064846 | 0.893493 | 3,000 rows / 1,500 series，`ok=true` |
| **3** | **0.928742** | **0.868958** | **1.043933** | **0.873619** | 3,000 rows / 1,500 series，`ok=true` |

Overall 和三个题型均随 epoch 单调下降，因此 epoch 3 是唯一严格最小值，没有题型冲突。该趋势说明 3 epochs 内尚未观察到 validation NLL 回升，但不保证继续训练一定改善 G-Eval。

### 6.2 与 Baseline 的 NLL 轨迹比较

| Epoch | Baseline Overall | 本轮 Overall | 绝对差值 | 相对变化（越低越好） |
|---:|---:|---:|---:|---:|
| 1 | 0.818750 | 0.983609 | +0.164859 | +20.14%（更差） |
| 2 | 0.773748 | 0.949135 | +0.175387 | +22.67%（更差） |
| 3 | 0.761679 | 0.928742 | +0.167063 | +21.93%（更差） |

epoch-3 分题型差距：

| 题型 | Baseline epoch 3 | 本轮 epoch 3 | 绝对差值 | 相对变化 |
|---|---:|---:|---:|---:|
| MC | 0.721204 | 0.868958 | +0.147754 | +20.49% |
| OE | 0.856149 | 1.043933 | +0.187784 | +21.93% |
| TF | 0.708736 | 0.873619 | +0.164883 | +23.26% |

NLL 在所有题型上都比 Baseline 更差，而 MC G-Eval 更高。这表明 teacher-forced answer likelihood 与最终生成/Judge 质量并非单调对应；也可能说明当前架构更偏向提高答案选择与表面推理质量，却牺牲了对参考答案完整 token 分布的拟合。

## 7. 推理、G-Eval 与产物审计

### 7.1 正式执行时间线

| 阶段 | 开始 | 完成 | 观测时长 | 结果 |
|---|---|---|---:|---|
| checkpoint audit/strip | 06:41:30 | 06:41:41 | 11 s | 3 个 epoch 全部通过 |
| validation epoch 1（修复后） | 07:07:36 | 08:23:25 | 1 h 15 m 49 s | 3,000/1,500，`ok=true` |
| validation epoch 2 | 08:23:25 | 09:38:31 | 1 h 15 m 06 s | 3,000/1,500，`ok=true` |
| validation epoch 3 | 09:38:31 | 10:54:11 | 1 h 15 m 40 s | 3,000/1,500，`ok=true` |
| NLL 选择 | 10:54:11 | 10:54:15 | 4 s | 选择 epoch 3 |
| full284 inference | 10:54:15 | 11:03:05 | 8 m 50 s | 284/284，`ok=true` |
| paper140 filter/audit | 11:03:05 | 11:03:05 | <1 s | 140/140，`ok=true` |
| DeepSeek-v4-pro G-Eval | 11:03:05 | 11:22:42 | 19 m 37 s | 335/335，`ok=true` |
| 聚合与 bundle | 11:22:42 | 11:22:49 | 7 s | 完成 |

第一次 validation epoch 1 在 06:41:58 因流水线错误解析 `.venv/bin/python` 为系统 Python、缺少 `transformers` 而失败。修复提交 `c5b3e55` 保留虚拟环境入口后，checkpoint 未重训、有效产物未覆盖，epoch 1 从头安全重跑并通过。该事件不影响模型权重，只影响后训练流水线启动。

### 7.2 生成与 Judge 协议

- 模型生成：per-record、`do_sample=False`、beam=5、`max_new_tokens=1000`；
- 推理：3 GPU，world size 3，fp16 autocast；
- Judge：`deepseek-v4-pro`，严格 Appendix-E prompt；
- `max_tokens=4096`；
- primary workers 8，fallback workers 1；
- 优先使用最终 score token 的 top-logprobs 概率均值；
- 概率不完整时使用恰好 20 个 temperature=1 有效样本均值；
- 本轮 334 个维度使用 `final_score_top_logprobs`，1 个使用 `exact_sample_mean_20`；
- 全部 G-Eval 在 GPU 服务器执行，未在本机 CPU 执行。

Baseline 对应计数为 333 个 top-logprobs + 2 个 20-sample fallback。两轮都使用相同规则，但 fallback 数量略有不同；API/Judge 的剩余随机性不能完全排除。

### 7.3 完整性与哈希

| 对象 | 覆盖 / 哈希 |
|---|---|
| validation epoch 1 | 3,000 rows / 1,500 series，`errors=[]` |
| validation epoch 2 | 3,000 rows / 1,500 series，`errors=[]` |
| validation epoch 3 | 3,000 rows / 1,500 series，`errors=[]` |
| full284 | 284 predictions，`errors=[]` |
| paper140 | 140 predictions，`errors=[]` |
| G-Eval | 335 scores，`errors=[]` |
| 非敏感 bundle | `a95de35d14158f816421847fcff39bcccd4eeedf082cd5d337a334ef82d57686` |
| bundle 大小 | 15,624,596 B |
| 本地清单验证 | 26/26 文件 SHA-256 一致 |

## 8. 结果意味着什么：收益、代价与机制解释

### 8.1 已得到直接支持的结论

1. **第一版 SLR 的严重推理坍缩不是不可避免的。** Objective-v3 使用自洽完整反事实和明确状态标签后，MC reasoning 从第一版的 1.93 恢复到 3.9070，并超过 Baseline 3.74。
2. **当前全量组合对 MC 有真实、跨维度的一致收益。** MC Final、Correctness、Reasoning 三项同时提升约 4.5%–5.2%。
3. **收益没有扩展到 OE 和 TF。** 七项非 MC 指标全部下降；TF correctness 是最大退化项。
4. **当前模型在 3 epochs 内仍在优化。** 训练快照 loss 和完整 validation NLL 均逐 epoch 下降，且无数值异常。
5. **显式状态监督的数据覆盖仍然有限。** 只有 15.15% QA anchor 拥有有效 pair，donor 复用集中。
6. **更高 MC G-Eval 不等于更低 answer NLL。** 当前所有题型 validation NLL 都比 Baseline 高约 20%–23%。

### 8.2 不能由本轮建立的结论

1. 不能确定 MC 改善来自 QK-Norm、continuous bypass、prototype sidecar、直接 task tokens 或 objective-v3 中的哪一项；这些因素在同一正式 run 中同时改变。
2. 不能宣称 QK-Norm 已消除 prototype hard selection；本轮 bundle 没有导出 attention entropy、top-1 mass 或 $N_{\mathrm{eff}}$。
3. 不能宣称 continuous bypass 已让 Local evidence 获得决策权；尚未重跑四格 Window/Local intervention 与 paired counterfactual flip 测试。
4. 不能把训练快照 state accuracy 当成 held-out grounding accuracy；它只来自训练周期快照中的有效 pair。
5. 单 seed 结果不能估计训练方差或统计显著性。

### 8.3 最合理的当前机制解释

结合已有缺陷报告，最保守的解释是：

- QK-Norm 与 continuous bypass 可能改善了 MC 所需的局部证据传输；
- direct task prompt 避免 fixed 分支再经过静态 prototype attention，有助于恢复稳定答题模式；
- objective-v3 避免旧 SLR 反复强化短 answer head，修复了 MC reasoning collapse；
- 但 `beta=0.2`、15.15% 的稀疏状态 pair、状态标签与自然语言 TF 命题的不一致，以及联合架构重新参数化，仍可能损害 OE/TF 的完整答案拟合；
- 当前表现更像“MC 路由改善 + 生成/命题判断代价”，而不是全面 grounding 成功。

## 9. 局限性、稳健性与不确定性

| 局限 | 对结论的影响 | 当前可用证据 |
|---|---|---|
| 只有 seed 72 | 无法估计训练方差；小幅差值可能不稳定 | MC 提升约 0.17–0.21，TF 最大下降 0.40，但无跨 seed CI |
| 多组件同时修改 | 无法做因果归因 | 只有 full variant 正式结果 |
| Baseline 表格仅两位小数 | 单项相对变化是近似值 | Macro 使用 Baseline 报告内部值 3.5417 |
| G-Eval 使用第三方 Judge | 存在模型与 API 漂移、top-logprob/fallback 随机性 | 本轮 334+1，Baseline 333+2；均审计完整 |
| 无 Baseline/Current 原始 paired score 联合文件 | 无法计算逐记录 paired bootstrap CI | 当前 bundle 只有本轮分数 |
| 状态监督覆盖 15.15% | 84.85% anchor 只受 answer loss；状态目标影响不均匀 | index audit 完整 |
| donor 集中复用 | 可能过拟合少数异常形态 | 419 unique、effective 160、max reuse 271 |
| 缺少内部机制导出 | 无法确认 QK entropy、gate、bypass 贡献 | checkpoint 架构审计只能证明组件存在 |
| 未评估额外 epoch | NLL 单调下降，可能尚未收敛 | 但延长训练对 G-Eval 的方向未知 |

## 10. 下一轮改进与消融优先级

### 10.1 P0：先完成可归因的组件消融

保持 seed、split、训练预算、decode 和 Judge 不变，至少执行以下矩阵：

| Run | Objective-v3 | QK-Norm | Continuous bypass + sidecar | Direct K=30 task prompt | 目的 |
|---|---:|---:|---:|---:|---|
| A0 | 否 | 否 | 否 | 否 | 正式 Baseline |
| A1 | 是 | 否 | 否 | 否 | 隔离新损失贡献 |
| A2 | 是 | 是 | 否 | 否 | 隔离 QK-Norm 增量 |
| A3 | 是 | 否 | 是 | 否 | 隔离连续旁路/sidecar 增量 |
| A4 | 是 | 否 | 否 | 是 | 隔离 direct task prompt 增量 |
| A5 | 是 | 是 | 是 | 是 | 已完成的 full variant |

若预算有限，优先 A1、A3、A4；当前证据还不能判断 QK-Norm 是否真正改变了 selector，因此 A2 必须同时导出 manipulation check。

### 10.2 P0：扫描 beta，优先验证 0.1

建议固定 full architecture，比较：

$$
\beta\in\{0.05,0.10,0.20\},
$$

均在前 10% steps 线性 warm-up。`0.10` 是设计文档推荐值，本轮 `0.20` 可能对 OE/TF 完整答案造成过强竞争。每个 run 至少记录：

- answer loss 与 state loss 的梯度范数比；
- 每参数组梯度范数；
- MC/OE/TF validation NLL；
- held-out factual/counterfactual state accuracy；
- 输出长度、answer-first、think rate 与空解释率。

### 10.3 P0：Top-M donor 采样必须与候选池扩展一起做

建议比较 `M=1/8/16`，但先确认每个 anchor 的实际 pool size。当前平均 eligible donor 仅 1.573，直接设置 M=8/16 很可能没有足够候选。

建议同时尝试：

1. 更好的背景检索特征，而非仅依赖量化 Window RMS；
2. 在不破坏 control-gap、same-length 和 dual-source validation 的前提下扩大 donor 候选；
3. 按异常类型/方向/幅度分层采样；
4. 为 donor 使用次数设置软上限或 inverse-frequency sampling。

下一轮至少报告：valid pair rate、mean pool size、unique/effective donors、max/p95 reuse、top donor share，以及异常类型覆盖。

### 10.4 P0：为 OE/TF 增加“答案保持”诊断，而不是重新强化短 head

第一版 SLR 已证明直接重复强化 answer head 会破坏解释。更安全的方向是：

- 保留完整答案 NLL，不额外重复短标签 CE；
- 对 OE/TF 监控 question-type-balanced answer loss；
- 将状态分类与自然语言 TF 命题显式区分，避免把 `has_anomaly` 当成 True/False 标签；
- 先增加生成诊断：解释 token 数、完整句率、与参考证据的覆盖、post-hoc rationalization；
- 若确需辅助目标，优先使用证据一致性或属性监督，而不是原答案概率的单边 margin。

### 10.5 P1：验证架构确实改变了内部机制

每个 checkpoint 应导出：

- QK-Norm 前后的 attention top-1 mass、entropy、$N_{\mathrm{eff}}$；
- 可学习 scale $g$ 的逐层/逐模块分布；
- bypass 与 prototype sidecar 的 gate 均值、方差、分位数及异常/正常差异；
- 将 bypass 或 prototype 分支置零时的 validation NLL 与 paired decision change；
- task prompt token 的范数、相似度和 option/label bias；
- 重新执行 Window/Local 四格实验，报告 $CE_W$、$CE_L$、interaction 和 paired success。

只有 manipulation check 证明 selector 被软化、gate 使用两条通道，才能把效果归因于 QK-Norm 或 bypass。

### 10.6 P1：多 seed 与停止规则

先用 seed 72 做低成本消融，再对最有希望的 2 个配置至少运行 3 个固定 seed，报告均值、标准差和 paired bootstrap CI。建议设置双重停止条件：

1. checkpoint 仍按完整 validation NLL 选择；
2. 若 OE/TF 免费诊断连续恶化，即使 overall NLL 下降也不继续投入 G-Eval。

当前 epoch 1→3 NLL 单调下降，因此可以把 epoch 4 作为低成本诊断；但只有当 validation NLL 继续下降且 OE/TF 生成诊断不恶化时，才值得再次调用 Judge。

### 10.7 建议的下一轮成功标准（提案，不是既有结论）

| 目标 | 建议门槛 |
|---|---|
| 保留 MC 收益 | MC Final ≥ 4.05，MC Reasoning ≥ 3.85 |
| 恢复整体性能 | Macro Final ≥ Baseline 3.5417 |
| 控制题型退化 | 任一题型 Final 相对 Baseline 不低于 -0.05 |
| 改善答案拟合 | 最佳 validation NLL 显著缩小与 0.761679 的差距 |
| 扩大状态监督 | valid pair rate 明显高于 15.15% |
| 降低 donor 集中 | top donor share、max reuse 明显下降，effective donors 上升 |
| 证明 grounding | held-out paired counterfactual flip 与四格干预成功率提升 |

## 11. 仍需回答的关键问题

1. MC 提升主要来自 objective-v3、continuous bypass，还是 direct task prompt？
2. TF correctness 的下降是否由状态标签和自然语言命题标签不一致导致？
3. 将 beta 从 0.2 降到 0.1，能否保留 MC 收益并恢复 OE/TF？
4. 当前 QK-Norm 是否真的降低了 top-1 mass，还是仅改变了参数化而未改变 selector？
5. gate bias=-2 后，训练完成时模型实际使用了多少 prototype sidecar？
6. Top-M=8/16 在当前平均 pool size=1.573 的约束下能覆盖多少 anchor？
7. epoch 4 是否继续降低 NLL，以及是否会改善或进一步损害 G-Eval？
8. MC 提升在至少三个 seed 上是否稳定？
9. 当前 state classifier 在独立 held-out 事实/反事实 pair 上的准确率、校准度和 flip consistency 是多少？

## 12. 可复核产物与来源

### 12.1 本轮机器可读产物

- [正式十项指标](experiments/reproduction/architecture_objective_v3_argmin/test_epoch_3_paper140/table1.json)
- [Table 1 Markdown](experiments/reproduction/architecture_objective_v3_argmin/test_epoch_3_paper140/table1.md)
- [三轮 validation NLL 与选择](experiments/reproduction/architecture_objective_v3_argmin/best_validation_loss.json)
- [选中 checkpoint 及哈希](experiments/reproduction/architecture_objective_v3_argmin/selected_checkpoint.json)
- [epoch-3 objective / architecture audit](experiments/reproduction/architecture_objective_v3_argmin/posttrain_pipeline/epoch_3_checkpoint_audit.json)
- [反事实索引审计](experiments/reproduction/architecture_objective_v3_argmin/index/counterfactual_index_audit.json)
- [full284 审计](experiments/reproduction/architecture_objective_v3_argmin/test_epoch_3_full284/audit_results.json)
- [paper140 prediction 审计](experiments/reproduction/architecture_objective_v3_argmin/test_epoch_3_paper140/audit_predictions.json)
- [G-Eval 审计](experiments/reproduction/architecture_objective_v3_argmin/test_epoch_3_paper140/audit_geval.json)
- [26 文件 SHA-256 清单](experiments/reproduction/architecture_objective_v3_argmin/artifacts_manifest.json)
- [完整非敏感产物包](experiments/reproduction/architecture_objective_v3_argmin/architecture_argmin_v3_nonsecret_artifacts.tar.gz)

### 12.2 设计与历史分析来源

- `readme.md`：正式 Baseline 指标、配置、训练时长与环境；
- `REPRODUCTION_RESULTS.md`：Baseline checkpoint 选择、严格 Table 1 与完整性审计；
- `训练目标(损失函数)改进_new.md`：Objective-v3、反事实构造、状态 CE 与 Top-M donor 设计；
- `第一版损失函数设计存在的问题.md`：第一版 SLR 退化与根因；
- `AXIS架构改进说明.md`：QK-Norm、continuous bypass/prototype sidecar、direct task prompt；
- `实验结论_AXIS缺陷.md`：prototype aggregation 信息损失与 evidence-to-decision routing failure。

## 最终结论

本轮实验是一次**有价值但不均衡的改进**。它成功修复第一版 SLR 最严重的 MC reasoning collapse，并使 MC 三项指标全面超过 Baseline；但 OE、TF、Macro Final、validation NLL 和资源效率均未达到 Baseline。最严谨的结论不是“全架构改进成功”或“失败”，而是：

> Objective-v3 与全架构组合显著改善了 MC 路径，但尚未建立跨题型稳定的 evidence-to-answer 路由；其收益与生成/命题判断代价并存，且当前单一 full run 无法定位贡献来源。

下一步应以可归因消融、`beta=0.1`、donor 候选池扩展 + Top-M、多 seed 和 held-out grounding 诊断为主，而不是继续堆叠未经隔离的新模块。
