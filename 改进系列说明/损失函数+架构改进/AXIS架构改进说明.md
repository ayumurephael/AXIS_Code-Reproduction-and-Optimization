# AXIS 架构改进

## 1：Prototype Aggregation 会损失信息

>Enoder / Local Projection 中存在位置、幅度信息，Q→Value Aggregation损失位置信息和幅度信息
>- TS Encoder 里存在**异常信息**
>- 最初 Local Projection 不是主要损失点
>- Q→ Value Aggregation 是主要信息损失：接近 One-Hot selector 是导致信息损失的主要原因

解决方案可以分成 2 个层次：
1. 不让所有信息必须经过 prototype；
2. 防止 attention 过度饱和

**方案 1：QK Normalization → 防止 attention 过度饱和**

请阅读本地工作目录下的论文：

- "实习2\AXIS\Baseline_Optimization_Paper\Query-Key Normalization\paper\2010.04245v1.pdf"



单个 head 中：
$$
\ell_j = \frac{q^{\top}k_j}{\sqrt{d_h}},\quad \alpha_j = \frac{e^{\ell_j}}{\sum_{i}e^{\ell_i}},\quad z = \sum_j \alpha_j v_j.
$$
其中：$\text{logits} = \frac{QK^{\top}}{\sqrt{d_k}}$，也称 attention scores 注意力分数

设 $j^*$ 是 最大 logit，那么：
$$
\alpha_{j^*} = \frac{e^{\ell_{j^*}}}{\sum_{i}e^{\ell_{i}}} = \frac{1}{1+ \sum_{j \neq j^*} e^{(\ell_j - \ell_{j^*})}} = \frac{1}{1+ \sum_{j \neq j^*} e^{-(\ell_{j^*} - \ell_{j})}} = \frac{1}{1+ \sum_{j \neq j^*} e^{-\Delta_j}}
$$
因此真正决定 one-hot 程度的不是 logits 的绝对值，而是 top-1 和其他 logits 之间的间隔：
$$
\Delta_j = \ell_{j^*} - \ell_{j}
$$
当 $\Delta_j$ 很大时，softmax 必然饱和。因此，logits 注意力分数必须 **平滑**。

标准的 缩放因子 $\frac{1}{\sqrt{d_k}}$ 只是假设 Q、K 各维方差受控时，补偿维度增长。
它不能控制训练后增长的：$\|q\|$、$\|k_j\|$、$W_Q,W_K$ 的谱范数等。


普通注意力：
$$
q^\top k
=
\underbrace{\|q\|\|k\|}_{\text{局部、不可控的尺度}}
\underbrace{\cos\theta}_{\text{方向相似度}}
$$
每个 token 都可能通过增大自身范数，改变 softmax 的尖锐程度。



**QKNorm**：
$$
g\hat q^\top\hat k
=
\underbrace{g}_{\text{统一且可学习的尺度}}
\underbrace{\cos\theta}_{\text{方向相似度}}
$$
它把控制权重新组织成：
- **方向**决定谁与谁相关；
- **一个学习到的标量**决定整体有多果断。



**对Q、K 做 L2 归一化**：对每个 token 的每个注意力头：
$$
\hat q_i
=
\frac{q_i}{\|q_i\|_2}
$$
于是：
$$
\hat q_i^\top\hat k_j
=
\frac{q_i^\top k_j}
{\|q_i\|_2\|k_j\|_2}
=
\cos\theta_{ij}
$$
因此：
$$
\hat Q\hat K^\top
$$
中的每个元素都是余弦相似度，范围是：
$$
[-1,1]
$$

注：**不能归一化 V！**
Value 不参与 softmax 打分，它携带的是实际被聚合的信息。如果归一化 V：
$$
\hat v_j=\frac{v_j}{\|v_j\|}
$$
会删除 Value 向量中的幅度信息。





**可学习参数 $g$ 及其初始化**

QKNorm 将标准公式：
$$
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}}
\right)V
$$
改为：
$$
\boxed{
\operatorname{softmax}
\left(
g\hat Q\hat K^\top
\right)V
}
$$
其中 $g$ 是可学习标量。

注意：作者代码不是“每个头一个 $g$”，而是：

> 每个 `MultiheadAttention` 模块有一个标量 $g$，由该模块中的所有注意力头共享。

编码器每层的自注意力、解码器每层的自注意力和交叉注意力分别创建 `MultiheadAttention` 对象，因此各模块可以学到不同的 $g$，但一个模块内部各头共享该参数。

**g 的初始化**：作者使用：
$$
g_0=\log_2(L^2-L)
$$
其中 $L$ 是训练集中源语言和目标语言序列长度的第 97.5 百分位数。

- 当 $L=72$ 时，$g_0 \approx 12.32$
- 当 $L=75$ 时，$g_0 \approx 12.44$
- 当 $L = 79$ 时，$g_0 \approx 12.59$







**方案 2：Continuous bypass + prototype sidecar**
最重要的修复原则：

>不要把“软化 attention”当做唯一方案，不能依赖 prototype 通道来承载所有信息。

即便把 $\alpha$ 调得更加均匀，也只是：
$$
z = \sum_j \alpha_j v_j
$$
对静态模板做平均。
>如果 $v_j$ 本来没有精确位置和幅度，平均更多模板也不能凭空恢复这些信息，甚至可能产生更加模糊的平均表示。

其核心思想可以用一句话概括为：
>让“原始连续信息”走一条直通高速路，让 prototype 只负责“语义翻译”的辅助工作（prototype sidecar），最后通过门控融合两者。

这样，即使 prototype 通道因为 one-hot 而丢失了连续细节，原始信息仍然通过 bypass 保留了下来。

**Step 1: 连续证据通道（Continuous ByPass）**
$$
C_t = W_c H_t
$$
- $H_t$：TS Encoder 输出的第 $t$ 个时间步的表示
- $W_c$：一个动态投影矩阵
- $C_t$：直接投影到 LLM 的 embedding 维度，**保留原始连续信息**

**Step 2：Prototype 语义通道**
$$
P_t = \text{Attn} (W_q H_t, W_k S_{\text{proto}}, W_vS_{\text{proto}})
$$
- $P_t$：将时序证据翻译成 LLM 更熟悉的语义方向

**Step 3：门控融合（Gated Fusion） +  残差融合 + LayerNorm**
$$
\begin{gather*}
g_t = \sigma (W_g [C_t; P_t]) \\
L_t = \text{LN}(C_t + g_t \odot P_t)
\end{gather*}
$$
- $[C_t;P_t]$：将连续通道和 prototype通道拼接
- $W_g$：学习如何平衡两者，$g_t$：门控值，决定多少比例的 prototype 信号被融合；$\sigma$：Sigmoid 激活函数。



## 2. Fixed Hint 是否真的需要 ProtoType？

AXIS 的 fixed 分支是：
$$
\widetilde{F} = \text{Attn}(P_{\text{fix}},S_{\text{proto}},S_{\text{proto}})
$$
其中：$\widetilde{F}$、$P_{\text{fix}}$、$S_{\text{proto}}$、cross-attention 的参数在训练完成后就是固定的，因此 Fixed 输出也是常量。

更合理的选择：
>直接学习 $K$ 个 task soft-prompt token $F_{\text{task}} \in \mathbb{R}^{K\times d}$。($K=30$)