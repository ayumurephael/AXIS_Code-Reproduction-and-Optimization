> 现在已经不能再把 AXIS 的问题简单概括成“Fixed 在起作用，Local 完全没用”。最新实验已经证明：Local 中存在真实、可测量、方向正确的样本证据信号；但这部分信号先在 prototype value aggregation 中被压缩，进入 LLM 后又被更强的任务先验、Fixed 控制状态和标签偏置压过，最终无法稳定控制离散决策。

所以，更准确的故事不是旧标题式的 “Control Tokens, Not Evidence”，而是：

> Evidence Enters, but Does Not Control the Decision：证据进入了模型，却没有获得决策权。

这是一个“两级串联故障”：

1. 表示层故障：证据在近 one-hot 的 prototype aggregation 中发生有损压缩。
2. 决策层故障：残余证据信号进入 LLM 后，被强烈的 Fixed/题面/标签先验压制，形成 routing collapse。

此外还有第三层训练原因：当前 answer-NLL 目标并不要求模型建立不可绕过的 evidence→answer 因果链。

------

# 一、四份报告合起来，真正证明了什么？

建议把结论分为“已经直接证明”“强机制推断”“仍未证明”三层。

| 结论                                     | 主要证据                                                     | 证据强度         |
| ---------------------------------------- | ------------------------------------------------------------ | ---------------- |
| TS encoder 里存在异常信息                | has-anomaly AUROC 约 0.70–0.75，幅度/位置 Spearman 显著高于 raw statistics | 已直接支持       |
| 最初 Local projection 不是主要损失点     | Local projection 后 position rho 反而由 0.278/0.387 上升到 0.496/0.548 | 已直接支持       |
| Q→Value aggregation 是位置语义瓶颈       | 两种 pooling 下 position rho 分别下降 0.349、0.342，simultaneous CI 均低于 0 | 已直接支持       |
| Local 最终仍含有因果信号                 | 四格实验 Raw $CE_L=0.1182$，CI $[0.0319,0.2045]$；norm matching 后几乎不变 | 已直接支持       |
| Local 信号没有控制最终标签               | 98 个 series 中 97 个在四种证据条件下始终预测 True           | 已直接支持       |
| Fixed 主要是生成状态控制器               | 删除 Fixed 后绝对 NLL、格式、长度、think rate 崩溃；但 exact-label accuracy 不降反升 | 强支持           |
| Fixed 不是干净的样本标签语义             | Fixed 与样本无关；移除后 label ranking 没有一致恶化；Fixed shuffle 大体保留功能 | 强支持           |
| prototype attention 近似硬选择           | 每个 query 的 $N_{\rm eff}\approx1$，top-1 mass 接近 1       | 已直接支持       |
| 近 one-hot selector 是信息损失的唯一原因 | 阶段位置与现象吻合，但 temperature 没有成功软化 selector     | 尚未证明         |
| Shared prototype 导致梯度冲突            | endpoint 梯度 cosine 接近 0；容量对照削弱了该解释            | 当前不支持       |
| Fixed dropout 能让模型学会使用 Local     | Fixed 梯度下降，但 paired success 仍接近 0                   | 已被当前实验反驳 |
| 所有 AXIS checkpoint、backbone 都会这样  | 目前只有 released DeepSeek checkpoint                        | 尚未证明         |

这一区分非常重要。现在最严谨的表述应当是：

- Local 不是“无信息”；
- Local 也不是“被充分使用”；
- 它具有小而真实的因果影响，但没有成为离散决策的充分或主导原因。

------

# 二、最新实验如何修正早期结论？

早期报告中的三个结论需要更新，而不是完全推翻。

## 1. “删除 Local 几乎不影响”不等于“Local 没有信息”

旧实验中：

- 删除 Local，accuracy 只下降约 2.2 个百分点；
- random Local 基本不伤；
- reverse、shuffle、roll 几乎不改变 loss。

这证明的是：

> 在原始任务、原始阈值和其他证据通道存在时，Local 不是决策必要条件。

它没有证明：

> 改变 Local 的样本内容完全不会改变模型内部概率。

新的 Window/Local 四格实验使用真实异常 Local 和配对正常 Local，而且保留通道位置、token 数、RoPE 和 prompt 结构，避免了删除/随机向量造成的 OOD 混杂。结果发现：
$$
CE_L=0.1182,\qquad 95\%\ \mathrm{CI}=[0.0319,0.2045].
$$
因此 Local 确实移动了模型的异常 margin，只是移动得远远不够。

类比来说：旧实验问的是“拔掉温度传感器，报警器会不会改变最终红灯”；新实验问的是“温度变化时，报警电压是否发生变化”。现在的答案是：

- 电压确实变化；
- 但报警器几乎永远亮红灯；
- 所以传感器不是坏的，坏的是信号强度、阈值或后续接线。

## 2. “Fixed 很重要”不等于“Fixed 携带正确标签”

自由生成中删除 Fixed 会发生巨大崩溃：

- 离散准确率约从 0.805 降到 0.265；
- 平均输出长度从 682 增到 3659；
- think rate 从 0.099 增到 0.968；
- answer-first 从 0.620 降到 0；
- gold answer NLL 增加约 1。

但 constrained exact-label 实验排除了格式和 parser：

| 条件           | Exact-label accuracy |
| -------------- | -------------------- |
| Base           | 0.378                |
| Fixed zero     | 0.519                |
| 无 Fixed token | 0.503                |

删除 Fixed 后，正确标签字符串的绝对概率变低，但正确标签相对于错误标签的排序并没有崩，甚至有所改善。

原因可以写成：
$$
\operatorname{NLL}(\text{Answer: True})
=
\operatorname{NLL}(\text{Answer: / 格式状态})
+
\operatorname{NLL}(\text{True}\mid\text{格式状态}).
$$
Fixed 很可能主要降低了第一部分：它让模型相信“现在应该短答、不要 think、先输出 Answer、使用训练答案措辞”。这会让 True 和 False 两个紧凑答案整体都更容易生成，却不保证两者的相对语义排序更正确。

所以 Fixed 更像“答题模式开关”，不是“答案密钥”。

## 3. Temperature 负结果不能证明 saturation 无关

在 $\tau=1024$ 时：

- test median $N_{\rm eff}=1.0067$；
- top-1 mass 仍为 0.9983；
- amplitude/position probe 只改善约 0.023/0.036；
- paired success 改善为 0。

这说明：

> 简单推理时 temperature 没有修复问题。

但不能推出：

> one-hot saturation 不是问题。

因为 manipulation check 没成功——你希望 selector 从“一张卡片”变成“混合四张卡片”，实际仍然约等于只选一张。这个实验否定的是“普通 temperature 是有效修复旋钮”，没有否定 saturation 的因果作用。

------

# 三、最核心的失败机制：强偏置压过弱证据

最新四格实验是整个证据包中最有解释力的部分。

定义：
$$
m(W,L)
=
\operatorname{NLL}(\text{False})
-
\operatorname{NLL}(\text{True}),
$$
因此：

- $m>0$：更支持 True；
- $m<0$：更支持 False。

四个条件为：

| 条件 | Window | Local | Margin |
| ---- | ------ | ----- | ------ |
| $++$ | 异常   | 异常  | 2.8831 |
| $+-$ | 异常   | 正常  | 2.7669 |
| $-+$ | 正常   | 异常  | 2.8137 |
| $--$ | 正常   | 正常  | 2.6934 |

可以近似写成：
$$
m(W,L)
\approx
2.6934
+0.0735W
+0.1203L
-0.0041WL,
$$
其中 $W,L\in\{0,1\}$，0 表示正常，1 表示异常。

这个式子揭示了整个故障：

- Window 的异常证据贡献约 $+0.07$；
- Local 的异常证据贡献约 $+0.12$；
- 两者联合贡献约 $+0.19$；
- 但在二者都正常时，模型已经有 $+2.69$ 的 True 偏置。

由于 margin 是两个候选序列的 log-likelihood ratio，$m_{--}=2.693$ 大约意味着：
$$
\frac{P(\text{True})}{P(\text{False})}
\approx e^{2.693}
\approx 14.8.
$$
也就是说，在 Window 和 Local 都是正常证据时，模型仍然以约 14.8:1 的相对似然偏向 True。异常证据同时加入后，这个比值只从约 14.8:1 增加到约 17.9:1。

因此，当前系统不是完全不响应证据，而是：

> 证据只能轻轻推动方向盘，但汽车的转向被一个更大的固定偏置锁死了。

这也解释了为什么：

- $CE_W>0$；
- $CE_L>0$；
- 但 paired success 只有 1.02%；
- 没有 Window follower，也没有 Local follower；
- 97/98 个 series 都是 Always True。

同时，interaction 接近 0：
$$
I_{W\times L}\approx-0.004.
$$
这意味着没有证据表明两条通道发生了显著协同融合或直接冲突。它们更像两个独立的小幅加法信号，都被同一个巨大下游偏置压住。

------

# 四、当前 AXIS 会遇到哪些具体失败场景？

## 1. 正常反事实仍被判断为异常

这是目前最明确的失败场景。

当异常区间被配对的 normal series 替换，Window 和 Local 都成为正常证据时，模型仍几乎全部回答 True。

这意味着在以下场景中很危险：

- 异常已经恢复，但模型仍报告异常；
- 同一模板下输入正常设备，模型仍沿用异常结论；
- 数据分布略微变化时，模型依赖任务先验而不是证据；
- 高异常比例训练集部署到低异常率真实场景时，误报率可能很高。

不过，这里还要保留一个重要限定：标准化 presence question 与原 benchmark 问法不同，而且样本来自异常样本的配对反事实。因此它直接证明的是“反事实、统一问题下的决策失败”，不等于原始 benchmark 每道题都恒 True。

## 2. 幅度、位置、方向等细粒度问题容易失败

Encoder 尚能读出：

- amplitude rho 约 0.37–0.47；
- position rho 约 0.28–0.55；
- has-anomaly AUROC 约 0.70–0.75。

但经过 Q→Value aggregation 后：

- amplitude rho 接近 0；
- position rho 从约 0.49 降到约 0.04–0.16；
- direction BAcc 多数接近或低于随机水平。

所以 AXIS 更可能保留“像某类异常”的粗类别模板，却无法可靠保存：

- 峰值到底有多高；
- 异常发生在窗口前部还是后部；
- 是先上升再下降，还是先下降再恢复；
- 异常持续多少步；
- 两个相似异常的细微幅度差；
- compound anomaly 中多个属性的组合。

这类失败会在开放解释里表现为：异常类别说得像，但位置、数值、方向或持续时间错误。

## 3. 所谓 step-aligned hint 可能退化为弱模板提示

Local token 数与时间步对齐，并不等于模型使用了这种对齐。

已有现象包括：

- reverse、roll、shuffle Local 几乎不改变原任务 loss；
- LLM 中后层对 Local 的平均注意力密度很低；
- position 在 value aggregation 处显著下降；
- Final Local 只产生小幅 margin 移动。

因此当前 Local 更可能是：

> 一串带有少量异常类别信息的连续提示，

而不是：

> 每个 token 都忠实对应一个具体时间步，LLM 可以据此定位和推理。

需要注意，普通 shuffle/reverse 有时并不会改变异常类别或数值集合，所以它不是最强的时间顺序检验。后续应使用“相同数值多重集合、只有顺序不同、标签必须改变”的最小对照，例如：

- rise→fall 与 fall→rise；
- early spike 与 late spike；
- upward shift 与 downward recovery；
- 相同极值、不同发生位置。

## 4. Window 与 Local 冲突时没有证据仲裁能力

四格实验中：

- Window 异常、Local 正常；
- Window 正常、Local 异常；

两种冲突条件仍然几乎全部输出 True，没有形成明确的 Window follower 或 Local follower。

这表明当前架构没有学到“两个证据源冲突时如何判断谁更可靠”。它们只是各自给 logit 一个小增量，并不存在显式的一致性检查、可靠性估计或 question-conditioned gating。

真实系统中，这会对应：

- 数字文本被截断，但 encoder 正常；
- encoder 遇到 OOD 信号，但数值统计正常；
- 传感器噪声导致两种表示不一致；
- 多变量之间出现互相矛盾的证据。

## 5. 输出格式和语义判断严重纠缠

Fixed 删除后自由生成崩溃，但 constrained label ranking 不崩。这说明 AXIS 的 benchmark 分数混合了两类能力：

1. 是否知道正确答案；
2. 是否按 parser 喜欢的格式输出。

因此模型可能：

- 语义排序没有明显变差，却因为进入 think 模式而被判错；
- 通过 Fixed 提高 answer-first 和可解析性，从而看起来“推理能力提高”；
- 换一个 backbone、system prompt 或 decoding policy 后性能突然变化；
- 在 JSON、自然语言、A/B/C/D 等不同输出协议下结果不一致。

这很可能与 released checkpoint 使用 DeepSeek-R1-Distill-Qwen-7B 有关：Fixed 可能在很大程度上负责压制其长 reasoning 风格。需要在 Qwen2.5、普通 instruct 模型上复现，才能判断是否是 AXIS 通用机制。

## 6. 选项位置、标签字母和 prompt 布局敏感

多选题全排列后：

- text-answer accuracy 约 0.572；
- label-only accuracy 约 0.338；
- label-only 旧字母黏附更明显。

而 matched reverse 虽然几乎不改变 reference NLL，却使 MC exact-label accuracy 增加约 0.138。

这意味着：

- 模型对答案文本理解强于对抽象标签映射；
- 标签字母、选项位置、Local/Fixed 在 prompt 中的相对位置会影响排序；
- 原始固定选项顺序可能掩盖位置先验。

这里统计时要注意：2232 个排列不是 2232 个独立问题，推断区间必须以原始 93 道题为 cluster，而不能把所有排列当作独立样本。

## 7. 训练容易坍缩为恒标签策略

formal-lite 的 12 个 runs 中，经常出现：

- 几乎恒 False：target-following=1，但 positive recall≈0；
- 几乎恒 True：positive recall=1，但 target-following≈0；
- paired success 接近 0；
- original accuracy 仍可达到 0.6–0.75。

这说明原始准确率、target-following、invariance 都可能被恒标签模型“刷高”。

机制不是简单的训练时间不足，而是训练目标存在退化解：

- answer NLL 可以通过类别先验降低；
- Fixed 和题面提供稳定梯度；
- Local 是样本相关信号，梯度方差更大；
- dropout 只关闭捷径，没有告诉模型应该如何使用正确证据；
- checkpoint 仍按原始 validation NLL 选择，而不是按 paired grounding 选择。

formal-lite 训练确实不足，因此不能断言完整训练必然坍缩；但它证明了当前目标函数存在容易到达的退化路线。

## 8. prototype 层存在“名义容量大、有效容量小”的问题

AXIS 有 1000 个 prototype，但对单个 query：
$$
N_{\rm eff}\approx1.
$$
于是：
$$
z(q)=\sum_j\alpha_j(q)v_j
\approx v_{j^*(q)}.
$$
如果没有保留 $q$ 的 residual bypass，那么落入同一个 prototype 区域的不同输入会变成几乎相同的输出。幅度 1.2 和 1.8、位置 30% 和 40%，可能都被翻译成同一张模板卡片。

但这里还缺一个关键诊断：

> $N_{\rm eff}=1$ 只说明每次选一个 prototype，不说明全数据是否总选同一个 prototype。

需要区分：

- 全局 codebook collapse：所有样本只使用极少几个 prototype；
- hard vector quantization：不同样本使用不同 prototype，但每个样本硬选一个；
- 有效语义量化：top-1 index 真能编码属性；
- 无意义硬路由：index 与异常属性无关。

当前报告主要测了 per-query entropy，还没有完整报告 dataset-level top-1 prototype 使用直方图、Gini、codebook perplexity 及其与幅度/位置的条件关系。

## 9. 数值内核与精度可能造成不连续行为

在近硬 argmax 的 attention 下，FP16、BF16、FP32 或不同 fused kernel 的微小误差可能改变 winner prototype。

你们已经观察到：

- 显式 FP32 与 released fused path 的多数差异很小；
- 但少数样本 relative error 超过 $10^{-3}$，最大达到 0.058。

这意味着潜在的：

- GPU/内核复现差异；
- FP16/BF16 切换不稳定；
- top-1 prototype 突然跳变；
- 输入极小扰动导致离散表示突变。

当前只有少量异常样本，因此它是明确的数值风险信号，还不是已经证明的大规模性能失败。

## 10. 开放题可能出现 post-hoc rationalization

现有证据说明：

- 删除 Window 会提高 answer NLL，但离散标签不变；
- Local 对开放答案 likelihood 影响很小；
- Fixed 强烈控制答案长度和格式；
- 已有案例出现输入中无法支持的具体数值。

因此模型可能先依赖模板/先验选择一个异常叙述，再用 Window 数字拼接一段看似具体的理由。

但当前开放题还缺少完整、可执行的属性级 faithfulness audit，因此应写成“强风险与待验证机制”，而不是最终定论。

------

# 五、失败为什么会发生？

```

```

## 1. prototype attention 缺少连续信息的安全通道

在标准 Transformer block 中，常见设计是：
$$
z=q+\operatorname{Attn}(q,K,V),
$$
或至少对 attention 输出与输入做 residual/gating。

AXIS 当前路线更接近把样本信息完全交给 prototype value aggregation：
$$
z=\sum_j\alpha_jv_j.
$$
当 $\alpha$ 近似 one-hot 时，$q$ 中的连续变化只能通过 winner index 传递。如果没有 $Uq$ residual，prototype 选择以后，未被 winner identity 表达的细节就直接消失。

这与 Q→Value 的 position probe 断崖高度一致。

## 2. Q/K 或 prototype 范数失控导致 softmax 饱和

Local、Fixed 和中间 attention 向量的范数远高于普通词向量。prototype mapping 又缺乏明确的 Q/K normalization、logit scale 限制和 entropy 约束。

因此：
$$
\alpha_j=
\operatorname{softmax}
\left(
\frac{q^\top k_j}{\sqrt d}
\right)
$$
中的 logit gap 会非常大。即便把 query 除以 1024，top-1 mass 仍约为 0.9983，说明原始 gap 已经到了普通温度缩放很难处理的量级。

## 3. Fixed 的梯度比 Local 更稳定

Fixed 对所有样本共享，因此它收到的梯度会稳定地学习：

- Answer-first；
- 不进入 think；
- 输出格式；
- 常见异常话术；
- 数据集类别与选项先验；
- 参考答案文本风格。

Local 每个样本不同，需要从答案中少数真正依赖证据的 token 获得监督，梯度更稀疏、更噪声。

因此优化器自然优先找到 Fixed/task-prior 路线。

## 4. 全答案 NLL 没有证据可识别性

训练目标是：
$$
\mathcal L_{\rm ans}
=
-\sum_t\log p(y_t\mid Q,W,L,F,y_{<t}).
$$
它只要求最终答案文本概率高，并不要求：
$$
p(y\mid Q,W,L,F)
\neq
p(y\mid Q,W,L',F)
$$
在证据改变时发生正确变化。

当 $Q,W,L,F$ 相互相关时，模型使用任意一个捷径都能降低 loss。于是“答对”无法识别究竟是哪条通道起作用。

## 5. 去掉捷径不等于建立正确路线

Fixed dropout 确实降低了 Fixed 梯度、提高了 Local 梯度，但 paired success 没有改善。

一个直观比喻是：关闭高速公路并不会自动修好乡间公路。模型失去 Fixed 后，如果没有 counterfactual supervision、属性监督和证据路由目标，仍可能选择最容易的恒标签路线。

这也解释了为什么单纯：

- Fixed dropout；
- Local/Fixed prototype separation；
- 增大 temperature；

都没有自动解决 grounding。



# 最终结论

你们的实验已经把 AXIS 的问题从模糊的“模型似乎没看数据”，推进到了一个相当清楚的机制框架：

1. TS encoder 中存在有限但真实的异常信息。
2. Local projection 尚能保存甚至增强部分位置可读性。
3. 近 one-hot prototype value aggregation 对位置造成确认性的可解码性损失，对幅度也有明显损失趋势。
4. Final Local 仍保留小而真实、与范数无关的因果信号。
5. 这份信号进入 LLM 后被远强于它的 Fixed/题面/标签偏置压住。
6. 当前 answer-NLL 目标允许模型使用格式、先验和恒标签策略绕过证据。
7. 所以 AXIS 的核心缺陷是“表示压缩 + evidence-to-decision routing failure”，不是单一的 encoder 失败、Fixed 捷径、prototype 共享或 temperature 设置错误。

最需要立刻避免的表述是“Local 完全没用”。最准确的表述是：

> Local evidence is causally detectable but decision-ineffective: it moves the model’s score in the correct direction, yet the movement is too weak and too compressed to overcome a much stronger task-conditioned decision bias.