# AXIS `main` 与作者补充代码 `AnomalyLlava-master` 的严谨对比报告

> 审计日期：2026-07-20
> 审计对象：论文、`baseline_new` 的 Git `main` 提交、作者补充代码 `AnomalyLlava-master`、复现说明与实验结果、现有模型检查点和预测/评分产物。
> 重要边界：静态审计完成后，又在用户指定的 3×A100 服务器上执行了四组无训练推理对照，并在用户明确授权后调用 PackyAPI 的 Gemini 2.5 Pro 完成四组 335 维评分。本机未执行 GPU 运算或训练；凭据未写入报告、代码或实验清单。实证结论见第 13—14 节，完整实验见《AXIS Gemini 2.5 Pro 作者兼容复现实验报告》。

## 1. 结论摘要

### 1.1 最重要的代码溯源结论

1. `baseline_new/main` 的 AXIS 核心模型不是一套脱离官方实现的“个人重写”。Git `main` 提交 `a06884fc8e902cebc69a4223e319cf88d81d5767` 中下列文件，与工作区内 `AXIS/AXIS_original_codes` 的同名文件逐文本比对一致：
   - `src/models/AXIS/AXIS.py`
   - `src/models/AXIS/AXIS_test.py`
   - `src/models/AXIS/Pretrain_ts_encoder.py`
   - `src/models/AXIS/dataset.py`
   - `src/models/AXIS/ts_encoder_bi_bias.py`
   - `requirements.txt`

2. 作者补充的 `AnomalyLlava-master` 不是单一的“最终 AXIS 官方实现”，而是一个更早、更宽的研究代码快照，内部同时包含：
   - `src/models/Moirai`：与最终 AXIS 最接近的直接前身；
   - `src/models/AnomalyLlava`：更早的三提示结构，但其三条提示路径与论文最终 AXIS 不同；
   - 其他 Baselines、GlimpseLLM、CLIP/AnomalyCLIP、ChatTS、G-Eval 与分析 notebook。

3. `baseline_new/main` 与补充代码中真正高度重合的是 `Moirai`，不是名为 `AnomalyLlava` 的模型目录：
   - `AXIS/ts_encoder_bi_bias.py` 与 `Moirai/ts_encoder_bi_bias.py` 完全一致；
   - `AXIS.py` 与 `Moirai.py` 的字符级相似度约为 **0.960**；
   - `AXIS_test.py` 与 `Moirai_test.py` 的相似度约为 **0.794**；
   - `dataset.py` 与 `Moirai/dataset.py` 的相似度约为 **0.867**。

因此，不能把“`baseline_new` 与 `AnomalyLlava-master` 不同”直接解释为“个人复现偏离官方”。更准确的关系是：

```text
较早的 AnomalyLlava 三提示原型
            ↓ 架构收敛/简化
          Moirai
            ↓ 重命名 + 初始化/EOS/解码/消融等修订
      论文版本 AXIS
            ↓ 核心源码被纳入
   baseline_new/main
            + 个人补充的训练、审计、G-Eval 工具
```

### 1.2 最可能造成分数差距的因素

按证据强度和潜在影响排序：

| 优先级 | 因素 | 已确认事实 | 对分数差距的判断 |
|---|---|---|---|
| 高 | 裁判模型与 G-Eval 算法不一致 | 论文/作者脚本使用 Gemini 2.5 Pro；当前正式复现使用 DeepSeek-v4-pro。两套 logprob 取值、缺失回退和提示格式也不相同 | **确定会改变测得分数；很可能解释相当一部分差距，但尚不能证明解释全部差距** |
| 高 | 推理批处理语义不一致 | `AXIS_test.py`/`Moirai_test.py` 会把同一时间序列的 QA 组成 batch 一次生成；旧正式复现按单条 QA 逐条生成。完整 140 题受控实验中，作者 epoch-33 权重有 51/140 输出改变，当前 epoch-3 权重有 42/140 输出改变 | **确定会改变逐题输出，但本次同一 Gemini 协议下作者权重的 335 维总体均分仅变化 +0.006，95% bootstrap CI 跨 0；它是协议缺口，却不是本次聚合分差的主要来源** |
| 高 | Phase II 训练与选模协议不一致 | 作者补充脚本与复现的 DDP、有效 batch、验证集、验证范围、轮数、选模方法均不同，且没有作者最终 checkpoint 可供同权重验证 | **可能产生真实模型能力差异；当前无法定量拆分** |
| 中 | 数据划分 seed 与验证集不同 | 补充脚本未显式传 seed，数据集默认 42；复现显式传 72。两个 1,500 条验证集仅重合 79 条 | **会改变训练样本和最佳 checkpoint 判断；影响可能经选模被放大** |
| 中 | 初始化与停止符监督差异 | 早期 `Moirai` 使用 Kaiming，最终 AXIS 使用 Xavier；早期版本屏蔽了全部 EOS，最终 AXIS 恢复最后一个 EOS 监督 | **若论文实验来自早期 checkpoint，可能显著；若来自最终 AXIS 源码，则不是复现偏差** |
| 中/低 | 数值精度、环境与近离散 prototype 选择 | 正式训练使用 bf16、推理使用 fp16；作者运行环境未完整给出 | **可能导致生成分叉，但现有材料不足以量化** |
| 低 | 测试样本覆盖 | 正式 `paper140` 的随机顺序、跳过前 2 个序列、取 70 个序列、共 140 条 QA 与作者测试逻辑相符 | **不是当前差距的主要来源** |
| 非原因 | TS encoder 核心实现被个人改写 | 编码器文件完全一致，复现 checkpoint 结构也与实际配置吻合 | **证据不支持** |
| 非原因 | 正式复现用了不同 beam/token 上限 | 最终主线源码和论文均为 beam=5、`max_new_tokens=1000`；正式复现遵循该设置 | **不是对最终 AXIS 的差异** |

### 1.3 关于“换成 Gemini 2.5 Pro 会不会更接近论文”

实测结论是：**换成 Gemini 能提高评测协议的可比性，但单独换 judge 不保证三个最终分数整体更近。当前 epoch-3 固定预测由 DeepSeek 改为 Gemini 后，MC 更近、OE 更远、TF 基本不变；只有再使用服务器上发现的作者 epoch-33 候选权重和作者 series 推理，才达到 4.27/2.97/3.74，接近论文的 4.19/3.02/3.65。**

理由：

- 论文明确把 Gemini 2.5 作为 G-Eval 裁判，而当前复现的 DeepSeek-v4-pro 只是裁判，不是 AXIS 的时序问答主干模型。
- 当前差值具有明显的裁判风格特征：DeepSeek 对 MC/OE/TF 的“正确性/准确性”并未全面偏低，但对 MC 推理质量、OE 完整性、TF 合理性的分数更低，同时 OE 相关性更高。它不像单纯的模型整体退化，更像评分校准和偏好差异。
- 当前三大任务最终分数的宏平均约为 3.542，论文约为 3.620，只差约 0.078；OE 与 TF 最终分数已分别只差 +0.03、+0.02，主要差距集中在 MC。重新裁判完全可能使总分更近，也可能使 OE/TF 原本很近的分数反而远离。
- “改 judge”只改变测量，不会改善已经生成的答案。若 Gemini 分数更接近论文，这只能说明评测可比性提高，不能单独证明模型实现已经复现成功。

最严谨的验证方式不是直接覆盖现有分数，而是对**同一份固定预测**做 DeepSeek/Gemini 配对重评，并保留两套结果。

---

## 2. 审计范围、版本与方法

### 2.1 固定审计版本

- `baseline_new` 当前工作树位于 `second_redesign` 且存在未提交改动，因此本报告没有切换或修改分支。
- 所有关于“main”的判断均针对 Git 对象：
  - 分支：`main`
  - 提交：`a06884fc8e902cebc69a4223e319cf88d81d5767`
- 源码通过 `git show main:<path>` 读取；没有把当前物理工作树中的 redesign 配置误当成 main。
- 作者补充代码目录是未跟踪的 `AnomalyLlava-master`，按现有文件原样审计。

### 2.2 使用的核验方式

1. 完整阅读论文抽取文本、仓库说明、复现说明、实验结论和 Gemini 调用说明。
2. 枚举两个代码树，按职责建立模型、数据、训练、推理、评估和配置映射。
3. 使用标准化文本哈希、字符相似度和 token 集相似度区分：
   - 完全相同；
   - 同源改造；
   - 名称相似但语义不同；
   - 仅一方存在的补充模块。
4. 对主线与补充代码的全部 Python 文件做 CPU 侧语法编译检查：共 152 个文件，无语法错误；仅无关的 AnomalyCLIP/Glimpse 代码出现 `SyntaxWarning`。
5. 对复现工具测试执行 CPU 侧 `pytest`，50 个测试通过。该结果证明辅助逻辑通过现有单测，不等价于证明模型与作者最终训练完全一致。
6. 通过 PyTorch `FakeTensorMode` 只读取 checkpoint 的张量名、形状和元数据，不加载到 GPU、不执行推理。
7. 对 paper140 的样本选择、题型数量、训练/测试时间序列重合和不同 seed 的划分交集进行逐项统计。

---

## 3. 代码树的关系：重合、冲突与互补

### 3.1 文件规模与职责

`baseline_new/main` 的 tracked 文件约 54 个，重点是：

- AXIS 模型核心；
- 配置；
- 本地数据/训练/推理入口；
- `tools/axis_repro` 中的正式复现、审计、结果表和 G-Eval 工具；
- 复现文档。

`AnomalyLlava-master` 共 136 个文件，其中约 120 个 Python 文件。它更像作者研究过程的综合快照，除 Moirai/AnomalyLlava 外还包括多类 baseline 和 notebook。大量“只在补充代码中存在”的文件并不是 AXIS 最终实现缺失，而是实验横向对照、早期探索或评估工具。

### 3.2 主要模块映射

| `baseline_new/main` | `AnomalyLlava-master` | 关系 | 结论 |
|---|---|---|---|
| `src/models/AXIS/AXIS.py` | `src/models/Moirai/Moirai.py` | 高度同源，约 0.960 | Moirai 是直接前身；最终 AXIS 在其上修订 |
| `src/models/AXIS/ts_encoder_bi_bias.py` | `src/models/Moirai/ts_encoder_bi_bias.py` | 完全一致 | 编码器核心重合 |
| `src/models/AXIS/dataset.py` | `src/models/Moirai/dataset.py` | 高度同源，约 0.867 | QA 数据读取/缓存/划分逻辑同源 |
| `src/models/AXIS/AXIS_test.py` | `src/models/Moirai/Moirai_test.py` | 明显同源，约 0.794 | 测试覆盖与 series-batched 推理语义相同 |
| `src/models/AXIS/Pretrain_ts_encoder.py` | `src/models/Moirai/Moirai_pretrain_multi.py` | 模型部分同源，后者还含完整训练入口 | 补充代码提供了更多早期 Phase I 训练上下文 |
| 无直接同名最终模块 | `src/models/AnomalyLlava/AnomalyLlava.py` | 仅约 0.696，结构语义不同 | 这是前身/旁支，不应当直接替代 AXIS |
| `tools/axis_repro/geval_*.py` | `src/utils/geval/*.py` | 功能相同、实现不同 | 互为补充，也存在计分冲突 |
| `tools/axis_repro/train_phase2_*.py` | `Moirai_main_Zero2.py` | 都做 Phase II，协议不同 | 不是等价训练脚本 |
| 无 | Baselines、GlimpseLLM、CLIP、AnomalyCLIP、ChatTS、notebooks | 官方补充 | 与 AXIS 主线结果并无直接缺失关系 |

### 3.3 `AnomalyLlava` 与最终 AXIS 的三提示语义不同

名字最容易引起误判。早期 `AnomalyLlava.py` 的三路信息是：

1. 全局时序 embedding 直接投影为 token；
2. 局部 embedding 经 prototype/attention 转为 token；
3. 固定 prompt token 直接参与输入。

Moirai/最终 AXIS 则收敛为论文描述的：

1. 数值窗口文本化提示；
2. 局部上下文 embedding 经 vocabulary prototype 机制映射；
3. 固定/任务提示也经 prototype 机制映射。

最终 AXIS 不再直接插入 AnomalyLlava 的全局 embedding token。因此两者不是“同一个模型少了一个文件”，也不能交叉加载 Phase II 权重。`AnomalyLlava` 对理解架构演化有价值，但不构成最终 AXIS 应逐行照搬的黄金实现。

---

## 4. `AXIS.py` 与 `Moirai.py` 的逐类差异

### 4.1 名称和路径变更

- 类名、配置名、import 路径由 Moirai 迁移为 AXIS。
- 最终 AXIS 增加 Hugging Face token 参数以支持模型下载/鉴权。
- 这些是工程迁移，原则上不改变模型分数。

### 4.2 可训练层初始化：Kaiming → Xavier

- 早期 Moirai 对 mapping、local projection、Q/K/V/out 等层采用 Kaiming uniform（ReLU 假设）。
- 最终 AXIS/main 改为 Xavier uniform。
- 固定提示 embedding 的正态初始化（标准差约 0.02）保持一致。

影响判断：这些层正是 Phase II 的核心可训练模块，初始化差异会改变优化轨迹。若论文结果来自旧 Moirai checkpoint，复现的 Xavier 初始化可能造成差异；但 main 与工作区中的原始 AXIS 代码都采用 Xavier，因此不能仅依据这个旧快照判定 baseline 写错。

### 4.3 EOS 标签监督

- 两版都会把 padding/EOS 相关位置置为 `-100` 以屏蔽 loss。
- 早期 Moirai 屏蔽后没有恢复最后一个 EOS。
- 最终 AXIS/main 显式令最后一个目标位置重新监督 EOS。

影响判断：最终实现更符合自回归训练目标，有助于学习正常停止。这个差异可能改变答案长度、完整性和截断，但同样属于“旧快照 vs 最终主线”，不是 baseline 对最终源码的偏离。

### 4.4 局部 embedding 的 dtype

- 最终 AXIS 在写入语言模型 embedding 序列前显式转换 local embedding dtype。
- 早期 Moirai 依赖张量赋值时的隐式转换。

影响判断：数学语义应相同，但显式转换对混合精度更稳定，降低 dtype 不匹配风险。

### 4.5 消融开关

最终 AXIS 增加：

- `wo_local_hint`
- `wo_fixed_hint`
- `wo_windows`

正常 `none` 模式下它们不改变主模型输出；它们主要用于论文消融实验。补充代码缺失这些开关，说明其快照早于最终论文代码。

### 4.6 解码参数

| 参数 | 早期 Moirai | 最终 AXIS/main | 论文描述/正式复现 |
|---|---:|---:|---:|
| `num_beams` | 5 | 5 | 5 |
| `max_new_tokens` | 200 | 1000 | 1000 |
| `repetition_penalty` | 1.2 | 1.15 | 1.15 |
| `no_repeat_ngram_size` | 未明确 | 3 | 3 |
| `length_penalty` | 未明确 | 1.0 | 1.0 |
| sampling | `do_sample=False`，但保留无效 temperature/top-p/top-k | 关闭 | 关闭 |

结论：正式复现与最终 AXIS/main 对齐；若直接运行补充代码的旧 Moirai，则 200-token 上限更可能伤害 OE 完整性。它不能用于指控正式复现 token 上限错误。

### 4.7 一个继承的非主路径问题

`TimeSeriesPretrainModel` 的实际返回值与某个 `CombinedModel.generate` 包装路径的解包数量不一致。main 延续了这一问题，但正式推理入口直接走 AXIS 实际使用路径，并未触发这个包装器。测试代码中存在捕获生成异常并返回占位文本的做法；正式复现工具则对空输出和错误做 fail-closed 审计。

结论：这是值得修复的工程缺陷，但当前已有正式预测无空结果，不能解释已报告分数差距。

---

## 5. 数据集、划分和评测覆盖

### 5.1 数据逻辑重合

Moirai 与 AXIS 的 QA 数据读取、缓存、文件排序、随机打乱和 train/validation 切分逻辑总体同源。main 删除了旧的 `ChatTSAnomalyPretrainDataset`，并对 QA dataset 重命名；这不会改变 Phase II QA 样本内容。

### 5.2 seed 42 与 seed 72 的实际冲突

- `Moirai/dataset.py` 默认 seed 为 42。
- 补充代码的 `Moirai_main_Zero2.py` 构造 QA train/test dataset 时没有把配置中的 seed 显式传入，因此实际使用默认 42。
- 正式复现训练脚本显式传入 seed 72。

对 30,000 条训练记录做确定性重算：

- 两者 validation 都是 1,500 条；
- validation 集交集只有 **79** 条；
- seed 42 的 validation 中有 1,421 条会进入 seed 72 的 train，反之亦然；
- 两个 train 集交集为 27,079 条，对称差为 2,842 条。

训练总体仍有约九成重合，因此该差异未必单独造成 0.28 的 MC 差距；但它会显著改变验证 loss、最佳 checkpoint 和早停判断，不能忽略。

### 5.3 paper140 覆盖已对齐

作者测试代码的关键行为是：

1. 测试序列按固定顺序和 seed 42 打乱；
2. `train_ratio=0.02` 时先跳过 `int(142×0.02)=2` 个序列；
3. 之后处理 70 个序列；
4. 每个序列 2 个 QA，共 140 个预测。

正式 `paper140` manifest 重现了该覆盖。题型计数为：

- MC：43 条；
- OE：55 条；
- TF：42 条；
- G-Eval 总维度数：335。

因此，报告分数不是因为误把 284 条 full set 当成论文的 140 条，也不是因为题型权重计算错误。

### 5.4 训练/测试时间序列重合的公平性风险

142 个测试 JSON 的底层时间序列在训练数据中都能找到完全相同的序列，但它们的窗口范围、问题和答案并不相同。按 Phase II 划分：

- seed 42 的 train 覆盖 142 个测试底层序列中的 133 个；
- seed 72 覆盖 138 个；
- paper140 对应的 70 个底层序列中，seed 42/72 分别覆盖 67/68 个。

这属于 benchmark 构造层面的潜在泄漏/泛化边界问题，不是 baseline 独有问题。对 paper140 而言两个 seed 只差 1 个底层序列，难以单独解释整体 MC 差距。

---

## 6. Phase I / Phase II 训练协议差异

### 6.1 论文、最终代码与补充快照本身存在不一致

论文附录描述：

- encoder layers = 6；
- vocabulary prototypes = 1024；
- fixed tokens = 8；
- base LLM 为 Qwen2.5-7B-Instruct，hidden size 4096；
- Phase I masking ratio = 0.25；
- Phase II learning rate = 2e-4，weight decay = 0.01。

但 main、补充代码和实际 checkpoint 显示：

- layers = 8；
- prototypes = 1000；
- fixed tokens = 30；
- main 使用 `DeepSeek-R1-Distill-Qwen-7B`，hidden size 3584；
- 补充 Moirai 使用 `DeepSeek-R1-Distill-Llama-8B`；
- 补充 Phase I 代码默认 masking ratio = 0.15；
- main 与 Moirai 配置均为 learning rate = 1e-4，weight decay = 1e-5。

论文 Table 3 中与 Table 1 完全相同的 AXIS 分数对应的是 `Deepseek-Qwen 7B (Instruct)` 行，这又与论文其他段落的 Qwen2.5 表述不完全一致。

结论：材料内部存在无法由 baseline 单方面消除的“论文文字—公开主线—早期补充快照”三方冲突。任何严谨复现都必须明确选择哪一套作为规范，而不能把三者混合为一个所谓官方配置。

### 6.2 实际 checkpoint 证明 baseline 运行的结构

CPU 侧 FakeTensor 元数据核验显示，正式 epoch 3 checkpoint：

- `epoch=3`；
- `global_step=28500`；
- TS encoder 共 8 层；
- `fix_prompt_embeddings` 形状为 `(1, 30, 3584)`；
- prototype mapping 形状为 `(1000, 151667)`；
- local projection 形状为 `(3584, 256)`；
- Q/K/V/out 矩阵均为 `(3584, 3584)`；
- Phase II 精确可训练参数量为 204,076,832。

这证明实际实验确实是 8/1000/30/3584，并非仅配置文件写了某个值却未生效。

Phase I checkpoint 还记录：

- epoch 22；
- best loss 约 0.04245；
- layers 8、patch size 16、`d_model=512`、`d_proj=256`、heads 8；
- batch 64、learning rate 5e-4、计划 50 epochs、seed 72。

补充 pretrain 主程序又会把若干参数临时改为 epochs 30、batch 4。因此，补充代码快照本身也不能直接生成现有 Phase I artifact。

### 6.3 作者补充 Phase II 脚本

`Moirai_main_Zero2.py` 的特点：

- Accelerate + DeepSpeed ZeRO-2；
- 配置 batch size 3、4 processes、gradient accumulation 16、clip 0.7；
- 代码又把 DeepSpeed `train_batch_size` 覆盖为 3，与 world size/micro-batch/accumulation 的组合可能不一致；
- 每次验证只取前 20 个 batch，不是完整 validation set；
- `best_eval_loss` 的生命周期和按 epoch 处理方式使最佳模型语义不够清晰；
- 配置最多 1000 epochs，实际何时停止未由现有材料确定；
- 加载 Phase I 权重，冻结 LLM，并在 TS encoder 不训练时通过 no-grad 路径使用编码器。

这是一份有参考价值但不足以唯一还原论文 checkpoint 的研究脚本。

### 6.4 正式 baseline Phase II

正式复现采用：

- `torchrun` DDP；
- 3 张 GPU；
- 每卡 micro-batch 1 个序列，有效 global batch 3；
- 无 gradient accumulation；
- bf16 autocast；
- AdamW 只优化 perceiver/映射模块；
- learning rate 1e-4，weight decay 1e-5；
- 3 个完整 epoch，共 28,500 step；
- 每个 epoch 在完整 1,500 序列、3,000 QA validation set 上计算 teacher-forced loss；
- 三个验证 loss 约为 0.818750、0.773748、0.761679，选择 epoch 3。

memory-safe 训练把 vocabulary loss 按 64 token 分块，并使用 checkpointing。现有单测支持它与常规 next-token cross entropy 的等价性；v3 还固定冻结 LLM 的 eval 语义。它是面向显存约束的实现补充，不等价于作者补充脚本的训练动态。

### 6.5 权重不兼容

补充 Moirai 使用 Llama-8B，main 使用 Qwen-7B；hidden size、词表和 embedding 矩阵不同。两者的 Phase II checkpoint 不能互相直接加载。因而“把补充代码拷进 baseline”不是有效复现办法，必须先确定论文 Table 1 究竟对应哪种 backbone 和 checkpoint。

---

## 7. 推理协议：当前最值得优先验证的模型侧差异

### 7.1 作者代码实际上是 series-batched

`AXIS_test.py` 与补充的 `Moirai_test.py` 都采用：

- DataLoader batch size 为 1 个时间序列；
- collate 后将该序列的 2 个 QA 展平；
- 一次对这 2 个问题组成的 batch 调用 `generate`。

正式复现则把 manifest 中每条 QA 单独调用生成，即 per-record batch size 1。

两者覆盖的是同一组 140 条 QA，但生成上下文中的 padding、position ids、attention mask、数值精度路径和 beam search batch 组织不同。对理论上完全正确且数值稳定的 causal LLM，这些输出应尽量一致；现实中量化误差、混合精度和生成排序临界点可使 token 序列分叉。

复现文档已有直接证据：比较过的 76 条中，只有 45 条输出完全相同，31 条不同，最低文本相似度约 0.18。这不是可忽略的实现细节。

### 7.2 对当前分数差异的含义

MC 的最终差值 -0.28，推理质量差值 -0.40，是当前最明显的弱项。series-batched 与 per-record 改变了近四成已比较输出，完全可能影响该项。现阶段不能断言哪个输出必然更好，但论文分数若来自作者测试入口，则 series-batched 才是更贴近作者运行路径的对照。

建议在任何重新训练之前先做：

1. 固定同一个 epoch 3 checkpoint；
2. 固定 paper140 manifest 和解码参数；
3. 分别运行 per-record 与严格复刻作者 collate 的 series-batched 推理；
4. 用同一个 Gemini 评分协议进行盲配对比较。

这项实验成本远低于重新训练，且能直接隔离推理协议因素。

---

## 8. G-Eval：作者补充实现与 baseline 的冲突

### 8.1 作者补充代码已经包含 G-Eval

补充代码中的 `src/utils/geval` 明确：

- 评估模型为 `gemini-2.5-pro`；
- temperature 为 0；
- advanced runner 的最大输出 token 为 5000；
- 请求 OpenAI-compatible 接口，并要求 logprobs/top-logprobs；
- prompt 与论文附录 E.4 高度一致；
- 任务维度和最终权重与论文一致。

因此，旧复现文档中“作者未公开 G-Eval 实现”的说法在收到这份补充代码后已经过时，应更新为：“作者后来提供了一个较早的 G-Eval 实现，但其接口、密钥管理和 logprob 聚合仍不足以证明与论文最终运行完全一致。”

### 8.2 作者 scorer 的算法问题

作者 advanced G-Eval 并不是在一个确定的“最终分数 token 位置”读取 1–5 的概率，而是：

1. 扫描输出中所有 token 位置；
2. 收集任何形似 1–5 的 token 及其 top alternatives；
3. 若附近出现 score/rating/final 等词，再人为加一个 context bonus；
4. 聚合得到期望分数。

这会把解释文本中的数字、编号或其他位置的候选 token 混入最终评分分布。若无 logprobs，则直接使用抽取的整数；抽取失败时可能回退为默认 3。它也没有正式复现中的 20 次采样回退和逐条强审计。

### 8.3 baseline scorer

当前正式复现：

- DeepSeek-v4-pro，temperature 0，高 reasoning 配置；
- 默认请求 top 20 logprobs；
- 先定位单一最终 score token，再检查 1–5 候选；
- top-logprobs 不足时，严格做 20 次 temperature 1 的独立采样回退；
- 使用可恢复 JSONL journal、重试、去重、完整性和维度审计。

正式结果中：

- 333/335 个评分维度使用 top-logprobs；
- 2/335 使用 20-sample fallback；
- 无空预测。

这套工程实现更可审计，但与作者 scorer 不等价。换 judge 时必须同时决定：

- 复刻作者的跨位置聚合，以追求“历史脚本一致”；还是
- 保留单一最终 token 的严格概率语义，以追求“统计定义清晰”。

最好的做法是两套都保留，并明确主次，而不是静默覆盖。

### 8.4 prompt 仍有细微差异

两边都基本遵循附录 E.4，但存在：

- `Score 5:` 与 `5:` 等格式差异；
- 维度名空格/下划线差异；
- 换行和最终分数输出格式差异；
- token budget 与 reasoning token 处理差异。

对 logprob 评分，最终输出格式会直接影响 score token 是否进入 top-k，不能当作纯排版差异。

---

## 9. Gemini 2.5 Pro 的替换边界与预期

### 9.1 必须先区分两个“模型”

1. **AXIS 生成 backbone**：当前为本地 `DeepSeek-R1-Distill-Qwen-7B`，需要访问 embedding matrix、`inputs_embeds` 和内部 hidden states。
2. **G-Eval judge**：当前为远程 DeepSeek-v4-pro，只读取 question/reference/prediction 后评分。

Gemini API 可以替换第 2 个 judge；不能作为一个简单 API 参数替换第 1 个 backbone。AXIS 的 prototype-to-vocabulary 和 embedding 注入依赖本地可微语言模型结构，普通 Gemini API 不暴露这些能力。

### 9.2 为什么预计更可比

- 论文明确使用 Gemini 2.5 作为裁判。
- 作者补充 runner 也指定 `gemini-2.5-pro`。
- 本地调用说明已验证兼容接口可调用该模型，并建议 temperature 0、足够的输出 token 上限。
- 实测 PackyAPI 的 OpenAI-compatible Gemini 2.5 Pro 响应不含 logprobs；其原生兼容路由启用 responseLogprobs/logprobs 时返回 HTTP 400。因此当前只能执行作者代码的 temperature=0 整数回退，不能声称得到了概率加权 G-Eval。

官方接口参考：

- <https://ai.google.dev/api/generate-content>
- <https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/content-generation-parameters>

### 9.3 为什么不能承诺一定更接近

- 论文使用的 Gemini 2.5 Pro 具体服务版本、快照日期、system prompt 和代理实现未完全记录；当前 2026 年可调用版本可能已经漂移。
- 作者脚本的 logprob 聚合并不等于严格单位置期望分数。
- Gemini 可能提高 MC 分数，但也可能把当前已经接近论文的 OE/TF 拉远。
- judge 具有随机性和服务端变化；temperature 0 也不能保证跨时间绝对复现。

### 9.4 推荐的最小实验设计

在获得费用授权后分两阶段：

**阶段 A：低成本校准**

- 从 335 个维度中分层抽取，覆盖 MC/OE/TF、当前高分/低分、top-logprob/fallback 情况；
- 同一 prompt 各跑作者 scorer 语义与严格 scorer 语义；
- 检查响应是否真的返回可定位的 1–5 top-logprobs；
- 对少量样本重复调用，估计 judge 重复性。

**阶段 B：完整 paired rejudge**

- 固定现有 140 条预测，不重新生成；
- Gemini 对全部 335 个维度评分；
- 保留每条原始响应、token/logprobs、解析位置、回退方法和重试记录；
- 与 DeepSeek 结果按同一 record/dimension 做配对差值；
- 用 record-level paired bootstrap 给出 95% CI，避免把同一答案的多个维度误当成独立样本。

若阶段 B 证明 judge 不能解释 MC 差异，再执行 series-batched 推理对照。最后才考虑昂贵的 Phase II 重训。

---

## 10. 当前分数的诊断性解读

### 10.1 不是“整体模型都差”

与论文相比：

- MC：最终 -0.28；正确性 -0.23；推理质量 -0.40；
- OE：最终 +0.03；准确性 +0.09；完整性 -0.28；相关性 +0.31；
- TF：最终 +0.02；正确性 +0.11；合理性 -0.14。

如果 baseline 的模型实现整体明显失败，通常会在大多数正确性和最终指标上共同下降。现在表现为：事实/答案正确性并不差，较弱的是解释推理、完整性和合理性，而相关性甚至更高。这与：

- judge 偏好差异；
- 输出更简短；
- 生成 batch 语义改变导致部分解释路径变化；

都相符。

### 10.2 输出统计支持“解释长度/风格”假设

已有正式预测：

- MC exact option accuracy 约 0.884；预测平均约 89.6 词，参考约 96 词；
- OE 预测平均约 87.4 词，参考约 105.8 词；
- TF exact accuracy 约 0.810；预测平均约 76.7 词，参考约 90.5 词。

预测总体短于参考，尤其 OE 和 TF。它可能直接降低完整性/合理性，同时保持或提高相关性。这一现象可以由训练/生成产生，也会被不同 judge 以不同幅度惩罚。

### 10.3 可归因与不可归因

当前可以说：

- **确定**评测裁判和评分实现不一致；
- **确定**正式推理与作者测试入口的 batch 语义不一致；
- **确定**训练/验证协议和 seed 不一致；
- **确定**论文文字、最终 main、较早补充快照三者配置互相矛盾。

当前不能说：

- “换 Gemini 后一定恢复到 4.19/3.02/3.65”；
- “Xavier 一定比 Kaiming 差 0.28”；
- “作者论文分数必然来自补充目录中的某个 checkpoint”；
- “baseline 的 encoder 或三提示主架构实现错误”。

缺失的关键证据是作者最终使用的：commit、checkpoint、完整训练命令、manifest、原始 predictions、Gemini 原始响应与逐维度分数。

---

## 11. GPU 资源核验与后续执行建议

已按要求完整读取指定 GPU 信息文件，并只连接其中列出的服务器。初始资源核验为只读；随后在空闲的 GPU 0/1/2 上完成四组 3×A100 推理对照。没有启动训练，也没有在报告中记录或输出密码、密钥等敏感配置。

检查时的资源概况：

- 4×V100：全部几乎占满；
- 4×A100 40GB 节点：3 张卡基本空闲，1 张占用；
- 两个 8×A100 80GB 节点：一个全部繁忙；另一个仅 1 张相对空闲；
- A40 节点：仅 1 张相对空闲；
- 6×H100：没有安全的连续多卡空闲组合。

本轮实际采用 **3×A100 40GB** 完成无训练推理。Gemini 重评分与批处理对照已完成；作者候选权重已将论文距离降至三最终分数 MAE 0.073，因此当前没有证据支持为“贴近测试分数”启动昂贵重训。若未来重训，仍须重新检查瞬时占卡并以 validation 选 checkpoint。

---

## 12. 安全与复现工程问题

1. GPU 信息文件和 Gemini 调用说明含敏感凭据，作者 G-Eval 脚本也存在硬编码凭据。报告未复制任何敏感值。
2. 建议立即轮换所有曾以明文保存或硬编码在脚本中的密钥，并改用环境变量/仅本机 secret store。
3. 作者 advanced runner 依赖文件名最后一段判断“最新文件”，只比较时间片段而可能跨日期选错；正式复现应使用完整时间戳或 manifest ID。
4. 任何 Gemini 重评都应写入新结果目录，不能覆盖 DeepSeek 结果；应记录 model id、endpoint 类型、日期、prompt hash、scorer 版本和原始响应 hash。

---

## 13. 执行顺序与完成状态

1. **已完成** Gemini 固定预测配对重评：当前预测的 MC 更近、OE 更远、TF 基本不变。
2. **已完成** 两个 checkpoint 的 per-record vs series-batched 完整 140 题对照及 335 维 Gemini 重评。
3. **向作者索取最终实验指纹**：commit、checkpoint hash、训练命令、seed、有效 global batch、实际 backbone、Gemini 版本和原始逐维度结果。
4. **暂不重训**：现有作者候选权重已达到 4.27/2.97/3.74；为继续贴测试表而重训会引入后验选择。
5. **已完成** paired delta、bootstrap CI、改变/未改变回答噪声对照，并保留逐题与逐维度结果。

## 14. 最终判断

`baseline_new/main` 与作者代码并非“互相冲突的两个 AXIS 实现”。核心关系是：main 的 AXIS 与最终原始 AXIS 源码一致，并由补充代码里的 Moirai 演化而来；`AnomalyLlava` 则是更早的架构前身。baseline 真正新增的是更完整、可审计的训练/推理/G-Eval 工具。

实证结果把原因排序进一步收窄：

1. **权重/训练状态是主要生成侧因素**：相同 per-record + Gemini 协议下，作者候选 epoch-33 相对当前 epoch-3 的 335 维均分提升 +0.176，95% bootstrap CI 约 [+0.033, +0.316]；
2. **Gemini 提高可比性但不是充分条件**：单独替换当前固定预测的 judge stack，MC 更近而 OE 更远；
3. **series batching 是真实协议缺口但本次不是聚合主因**：作者权重 51/140 回答改变，335 维均分仅 +0.006，95% CI 跨 0；
4. **核心主架构没有发现足以解释分差的错位**：作者候选权重能够被 main 严格兼容加载并取得接近论文的结果。

最终作者兼容组合为作者候选 epoch-33 + series + Gemini 2.5 Pro，得到 **4.27/2.97/3.74**，相对论文 **+0.08/−0.05/+0.09**。完整实验、限制和产物路径见 `AXIS_Gemini25Pro_作者兼容复现实验报告.md`。
