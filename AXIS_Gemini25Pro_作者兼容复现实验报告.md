# AXIS Gemini 2.5 Pro 作者兼容复现实验报告

> 日期：2026-07-20。本文记录静态审计之后的完整实证。所有 AXIS 模型推理均在用户指定的 3×A100 40GB 服务器上执行；本机没有执行 GPU 运算或训练。Gemini 调用在用户明确授权将测试内容发送给 PackyAPI 后进行。本文不记录任何密码、API key、SSH 配置或其他凭据。

## 1. 最终结论

目标已经在不针对测试集改标签、不手调逐题分数、也不重新训练的前提下达到：使用服务器中发现的作者 epoch-33 候选权重、作者代码的 series-batched 推理语义，以及 Gemini 2.5 Pro 的作者提示与可执行评分语义，完整 paper140 测试集得到：

| 模型/协议 | MC 最终 | OE 最终 | TF 最终 |
|---|---:|---:|---:|
| 论文 AXIS / Gemini-2.5 | 4.19 | 3.02 | 3.65 |
| 作者候选 epoch-33 / series / Gemini-2.5-Pro | **4.27** | **2.97** | **3.74** |
| 差值（实验 − 论文） | **+0.08** | **−0.05** | **+0.09** |

三个最终分数的平均绝对误差为 0.073；十个报告指标的平均绝对误差为 0.074。相较原 Baseline / DeepSeek-v4-pro 的 0.110 与 0.189，分别下降约 33% 与 61%。

但是，“把 DeepSeek judge 单独替换成 Gemini”并不足以稳定达到目标。对当前 epoch-3 固定预测直接换成作者兼容 Gemini 评分后，MC 从 3.91 变为 4.03，OE 从 3.05 变为 2.77，TF 从 3.67 变为 3.66：MC 更接近，OE 明显更远，TF 基本不变。真正解释大部分差距的是服务器上保留的作者候选权重；series 批处理是必须修正的协议缺口，但其本次聚合评分效应较小。

这里的 Gemini 是 G-Eval 裁判模型，不是 AXIS 的本地可微语言主干。普通 Gemini API 不能直接替换 DeepSeek-R1-Distill-Qwen-7B backbone，因为 AXIS 依赖词嵌入注入、词表映射和反向传播。

## 2. 实验问题与因子设计

为了避免把多个因素混成一个“改进”，完整实验采用 2×2 推理因子，并保留原 DeepSeek 评分作为参照：

| 编号 | Phase-II 权重 | 生成批处理 | Judge | 目的 |
|---|---|---|---|---|
| A | 当前复现 epoch-3 | per-record | DeepSeek-v4-pro | 用户给出的原 Baseline |
| B | 当前复现 epoch-3 | per-record | Gemini 2.5 Pro | 观察 judge 协议整体替换 |
| C | 当前复现 epoch-3 | series | Gemini 2.5 Pro | 隔离当前权重下的批处理效应 |
| D | 作者候选 epoch-33 | per-record | Gemini 2.5 Pro | 隔离权重效应 |
| E | 作者候选 epoch-33 | series | Gemini 2.5 Pro | 最接近作者补充代码的正式组合 |

四份 Gemini 结果均覆盖 140 个回答、335 个题目—维度评分，且由审计器确认唯一键、题型、维度和权重完整，无缺失、重复或非法分数。没有使用中途的 partial 文件建表。

## 3. 运行环境与权重身份

### 3.1 GPU 与软件环境

- 服务器：用户指定列表中的 4×A100 40GB 节点；
- 实际使用：GPU 0/1/2，三卡分布式推理；GPU 3 当时由其他任务占用；
- PyTorch：2.5.1+cu121；
- Transformers：4.45.2；
- NumPy：1.26.1；
- 生成：beam=5、max_new_tokens=1000；
- 子集：paper140、base mode；
- 正式作者兼容路径：batching=series、skip-loss；
- 本轮没有训练，也没有在本机执行模型推理。

### 3.2 两个 Phase-II 权重

| 权重 | epoch | avg_loss | SHA-256 |
|---|---:|---:|---|
| 当前复现 epoch-3 | 3 | 未写入 inference checkpoint | 8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b |
| 服务器作者候选 | 33 | 0.7127881973981858 | d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7 |

作者候选权重文件中的 Moirai trainable keys 统一带 perceiver. 前缀；去除这一历史包装前缀后，其张量名与当前 AXIS/Qwen 路径严格兼容。它不是 AnomalyLlava/Llama checkpoint。代码只在所有 key 都带该前缀时归一化，混合 key 或结构不一致仍会 strict-load 失败。

该权重可称为“作者候选”而不能无条件称为“论文最终权重”：目前缺少作者最终 commit、正式 checkpoint manifest 和论文原始逐题预测来闭合来源链。

## 4. 推理可重复性核验

为排除 PyTorch/CUDA 版本或新 runner 改写造成的差异，作者候选权重先按 per-record 在相同环境重跑，并与服务器历史预测逐题、逐字符比较：

- 140/140 response 完全一致；
- 三题型 exact count 均为全量；
- mean text similarity = 1.0；
- MC/OE/TF 代理指标完全一致。

这证明新环境、数据顺序、checkpoint 加载和 per-record runner 能精确复现历史输出。随后只改变 batching=series，构成有效受控实验。

## 5. 完整结果

| 模型/协议 | MC 最终 | MC 正确性 | MC 推理质量 | OE 最终 | OE 准确性 | OE 完整性 | OE 相关性 | TF 最终 | TF 正确性 | TF 合理性 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 论文 AXIS / Gemini-2.5 | 4.19 | 4.21 | 4.14 | 3.02 | 2.87 | 2.93 | 3.31 | 3.65 | 3.60 | 3.74 |
| 当前 epoch-3 / per-record / DeepSeek | 3.91 | 3.98 | 3.74 | 3.05 | 2.96 | 2.65 | 3.62 | 3.67 | 3.71 | 3.60 |
| 当前 epoch-3 / per-record / Gemini | 4.03 | 4.05 | 4.00 | 2.77 | 2.80 | 2.51 | 3.05 | 3.66 | 3.71 | 3.57 |
| 当前 epoch-3 / series / Gemini | 4.06 | 4.07 | 4.05 | 2.73 | 2.80 | 2.42 | 3.02 | 3.63 | 3.69 | 3.55 |
| 作者候选 epoch-33 / per-record / Gemini | 4.26 | 4.28 | 4.21 | 2.93 | 2.69 | 2.73 | 3.44 | 3.80 | 3.81 | 3.79 |
| **作者候选 epoch-33 / series / Gemini** | **4.27** | **4.28** | **4.23** | **2.97** | **2.84** | **2.78** | **3.35** | **3.74** | **3.74** | **3.74** |
| 差值（最后一行 − 论文） | +0.08 | +0.07 | +0.09 | −0.05 | −0.03 | −0.15 | +0.04 | +0.09 | +0.14 | 0.00 |

### 5.1 与论文的距离

| 实验 | 三个最终分数 MAE | 全部十指标 MAE |
|---|---:|---:|
| 当前 / per-record / DeepSeek | 0.110 | 0.189 |
| 当前 / per-record / Gemini | 0.140 | 0.175 |
| 当前 / series / Gemini | 0.147 | 0.182 |
| 作者候选 / per-record / Gemini | 0.103 | 0.122 |
| **作者候选 / series / Gemini** | **0.073** | **0.074** |

因此：

1. 当前固定预测单独换 judge：十指标 MAE 小幅改善，但三个最终分数 MAE 反而变差；
2. 换到作者候选权重后，MC、OE、TF 同时进入论文附近；
3. 作者候选上恢复 series 语义后，OE 和 TF 比 per-record 更近，十指标 MAE 从 0.122 降至 0.074；
4. 不能把最终接近简单归功于 Gemini，也不能把所有差距归咎于 batching。

## 6. 因果对照

### 6.1 Judge 协议整体替换

对当前 epoch-3 的同一份 per-record 预测：

- DeepSeek 335 维原始均分：3.4237；
- Gemini 作者兼容协议均分：3.3194；
- 配对差：−0.1043；
- 95% bootstrap CI：约 [−0.2030, −0.0114]；
- 平均绝对逐维度评分差：0.5285。

分维度上，Gemini 的 MC reasoning_quality 比 DeepSeek 高约 0.256，而 OE relevance 低约 0.563。这与“MC 提高、OE 降低”的汇总变化一致。

这项对照是 judge stack 对照，而不是只有模型名不同：旧 DeepSeek runner 的 rubric 文本、空行布局和 top-logprob 抽取与作者 AdvancedGEvaluator 并不完全相同；新 Gemini runner使用作者补充代码的提示布局与 rubric。因此它衡量的是“恢复论文/作者评测栈”的总体效应。

### 6.2 权重效应

保持 per-record + 同一 Gemini 作者协议，当前 epoch-3 与作者候选 epoch-33 的 335 维配对比较：

- 当前均分：3.3194；
- 作者候选均分：3.4955；
- 配对提升：+0.1761；
- 95% bootstrap CI：约 [+0.0328, +0.3164]；
- 作者候选 OE completeness +0.218，OE relevance +0.382；
- 140 个回答均发生文本变化。

这是本次对照中唯一总体置信区间不跨 0、且能同时显著缩小论文距离的生成侧因素。它支持“当前三 epoch 复现权重不是作者论文分数所用训练状态”的解释。

### 6.3 Series-batched 推理效应：作者候选权重

从 per-record 改为作者 series 语义：

- 51/140 回答改变：MC 15、OE 24、TF 12；
- 89/140 完全一致；
- 平均文本相似度 0.8795；
- MC exact-option proxy 均为 0.9535；
- OE 文本代理变化 −0.0022；
- TF exact proxy 从 0.5714 增至 0.5952。

独立 Gemini 全量评分：

- per-record 335 维均分 3.4955；
- series 335 维均分 3.5015；
- 差值 +0.0060；
- 95% bootstrap CI 约 [−0.0597, +0.0716]。

51 个改变回答对应的 126 个维度均分变化为 −0.0159；未改变回答对应 209 个相同 prompt 的均分变化为 +0.0191。两者都处于单次 Gemini 噪声内。因此 batching 确定改变生成文本，是必须修正的作者协议差异，但没有证据表明它是本次聚合分差的主要原因。

### 6.4 Series-batched 推理效应：当前权重

- 42/140 回答改变：MC 13、OE 20、TF 9；
- 98/140 完全一致；
- 平均文本相似度 0.9178；
- Gemini 335 维均分变化 −0.0179；
- 95% bootstrap CI 约 [−0.0687, +0.0299]。

当前权重也显示 batching 的聚合效应不显著，并且没有让当前权重更接近论文。这避免了根据作者候选上的一次方向性变化作过度归因。

## 7. PackyAPI、logprobs 与评测噪声

### 7.1 接口能力实测

- OpenAI-compatible chat completions 能正常返回 Gemini 2.5 Pro 文本；
- 请求 logprobs/top_logprobs 时，响应不含 choices[].logprobs；
- Packy 原生 Gemini 兼容接口启用 responseLogprobs/logprobs 时返回 HTTP 400，说明该路由未为此模型开放 logprobs；
- 偶发 RemoteDisconnected 和代理 404，由逐任务持久化与外层重试恢复；
- 四份正式结果均为 335/335，method 全部是 author_temperature0_integer，没有 author_default_3。

作者 AdvancedGEvaluator 在没有概率分布时实际回退为 temperature=0 文本整数分。正式表采用这一可执行语义，因而是“作者代码路径兼容”，但不是论文理想描述中的概率加权 G-Eval。结果中明确记录 logprobs_requested=true、logprobs_returned=false、method、prompt hash、模型名、usage 与 endpoint host。

### 7.2 Temperature=0 仍非完全确定

对同一 6 个回答、14 个维度连续评三次：

- 12/14 维度三次完全一致；
- 2/14 维度发生 1 分范围内变化；
- 三次未加权维度均分为 3.7143、3.8571、3.8571。

因此表格的 0.01–0.05 级差异不能被当作确定性模型收益。作者候选/series 与论文的接近是完整数据上的观测结果，不等于精确复原作者当时的 Gemini 服务快照。

### 7.3 严格采样校准

另选一条 MC、一条 OE、一条 TF，对每个维度执行 20 次 temperature=1 有效采样均值，7/7 维均获得恰好 20 个样本：

| 题型 | 最终/维度分数 |
|---|---|
| MC | Final 4.36；correctness 4.45；reasoning 4.15 |
| OE | Final 3.65；accuracy 3.00；completeness 3.65；relevance 4.40 |
| TF | Final 3.26；correctness 3.30；justification 3.20 |

该小样本只用于确认单次整数回退存在离散化与服务随机性，不能代替 full140 结果，也没有被混入正式表。

## 8. baseline_new 与 AnomalyLlava-master 的实证关系

完整静态差异见《AXIS main 与 AnomalyLlava-master 严谨对比报告》。与本轮实验直接相关的结论是：

1. baseline_new/main 的 AXIS 核心与仓库内 AXIS_original_codes 对齐，不是个人另写的同名模型；
2. AnomalyLlava-master 中与最终 AXIS 最接近的是 Moirai，而不是旧 AnomalyLlava 类；
3. encoder 主体可逐模块对应，AXIS 在 Moirai 之上加入 task-aware / time-aware / prototype 提示、基于语言模型词嵌入初始化 prototype，以及新的训练/测试入口；
4. 作者候选 epoch-33 是 Qwen/AXIS-compatible 权重，不应加载到旧 Llama/AnomalyLlava 入口；
5. 两仓库的实现既有演化重合，也有配置冲突和工程互补。不能用 AnomalyLlava-master 整体覆盖 baseline_new；
6. 本轮找到的实际差距来自 checkpoint/训练状态与评测协议，不支持“baseline 主架构抄错或 encoder 错位”这一假设。

## 9. 已实施的代码修改

分支边界：静态比较对象始终是 Git 的 main ref。作者 checkpoint 兼容、series 推理、Gemini runner 与统计工具现已完整移植到 main；架构改进分支应从同一提交移植这些复现基础设施，保证比较协议一致。

### 9.1 推理与 checkpoint

- tools/axis_repro/model_utils.py：兼容作者 Accelerate checkpoint 的统一 perceiver. 前缀，并保持 strict loading；
- tools/axis_repro/run_inference.py：加入 per_record/series 两种明确语义、skip-loss、按 series 分组、可恢复分片与 manifest 记录；
- tools/axis_repro/run_inference_series_legacy.py：保留作者旧入口语义的独立核验 runner；
- tools/axis_repro/common.py：代理评分兼容 prediction 中的 response 字段。

正式作者兼容配置现在是 batching=series + skip-loss；per_record 保留为诊断对照，而不是继续冒充作者正式协议。

### 9.2 Gemini 与统计

- tools/axis_repro/geval_gemini.py：作者 prompt/rubric、Packy OpenAI-compatible 调用、logprob 检测、作者回退、严格采样、断点续跑、任务级重试与审计元数据；
- tools/axis_repro/compare_inference_protocols.py：key 对齐、逐题文本变化、相似度、题型分层与答案代理指标；
- tools/axis_repro/compare_geval_runs.py：逐维度配对、改变/未改变回答分层、bootstrap CI 与噪声对照；
- readme.md：把正式推理更正为 series + skip-loss，并补充完整命令、Gemini 限制和安全要求。

### 9.3 凭据清理

作者目录内 6 个 API/G-Eval 脚本已改为从环境变量读取凭据。硬编码凭据模式扫描命中 0 个 Python 文件。鉴于凭据曾以明文存在，仍建议用户轮换相关 key。

## 10. 验证

- tools/axis_repro 全测试：72 passed；
- 修改过的作者 API/G-Eval 脚本：py_compile 通过；
- git diff --check：通过；
- 四份正式 Gemini 结果：每份 140 predictions / 335 scores，audit_results 均 ok；
- 作者候选历史预测复跑：140/140 response exact；
- 本机无 GPU 推理，无长时间训练。

## 11. 关键产物

### 11.1 最接近论文的正式组合

- predictions：experiments/reproduction/author_accelerate_candidates/released_epoch33_torch251_cu121_paper140_series_3gpu_v1/predictions.jsonl
- run manifest：同目录 run_manifest.json
- Gemini scores：experiments/reproduction/gemini_calibration/author_epoch33_series_author_gemini25pro.jsonl
- 汇总表：同目录 author_epoch33_series_author_gemini25pro.md
- per/series G-Eval 配对：同目录 author_epoch33_per_vs_series_paired.json

### 11.2 其他因子对照

- 当前 per-record Gemini：baseline_epoch3_per_record_author_gemini25pro.jsonl
- 当前 series Gemini：baseline_epoch3_series_author_gemini25pro.jsonl
- 作者候选 per-record Gemini：author_epoch33_per_record_author_gemini25pro.jsonl
- 当前权重 per/series 配对：current_epoch3_per_vs_series_paired.json
- 当前与作者候选权重配对：current_epoch3_vs_author_epoch33_per_record_paired.json
- DeepSeek 与 Gemini 评测栈配对：current_epoch3_deepseek_vs_gemini_paired.json
- 严格 20 次采样 pilot：pilot3_stratified_strict20_gemini25pro.jsonl

## 12. 推荐的正式复现口径

当前最合理、可审计的主结果应采用：

1. 作者候选 epoch-33 checkpoint；
2. paper140 固定清单；
3. 3 卡 series-batched、beam=5、max_new_tokens=1000、skip-loss；
4. Gemini 2.5 Pro；
5. 作者 prompt/rubric；
6. 服务没有 logprobs 时明确标注 author_temperature0_integer；
7. 保存逐题 predictions、逐维度 judge 内容、prompt hash、checkpoint hash、manifest 与 audit。

不建议为了让 0.08/−0.05/+0.09 进一步“贴表”而根据测试分数调参数或挑 API 重跑结果；这会构成对测试集的后验选择。下一步只有在作者提供最终 checkpoint/commit/Gemini 原始结果后，才有依据判断剩余误差来自服务版本漂移、checkpoint 身份还是论文报告取整。

## 13. 尚未闭合的限制

1. PackyAPI 当前不能返回 Gemini 2.5 Pro logprobs，无法精确重现论文所述概率加权分数；
2. Gemini 服务端即便 temperature=0 也有小幅非确定性；
3. “Gemini 2.5 Pro”没有锁定作者当时的具体服务快照；
4. 作者 epoch-33 checkpoint 的文件位置、结构和性能高度吻合，但缺少作者签名 manifest，不能证明它就是论文最终 checkpoint；
5. 论文文字、作者 main 和较早补充快照在层数、维度、训练轮数等处互相矛盾；
6. 本轮没有因为目标已经通过现有权重达到而重新训练。若未来必须重训，应先获得作者最终训练指纹，并用 validation 而不是 test 选择 checkpoint。

综合判断：baseline_new 的主要代码框架不需要改造成旧 AnomalyLlava；需要的是恢复作者 checkpoint 兼容、series 推理和 Gemini 评测协议。完成这些修改后，结果已经与论文非常接近，同时保留了对中转 logprobs 缺失和评分随机性的诚实边界。
