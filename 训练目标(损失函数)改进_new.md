# 全新的损失函数设计【改进请严格依照此版本进行】

# 先把核心思想说清楚

“完整一致反事实 + 显式 normal/anomalous 目标”不是把旧 SLR 换一个名字，而是同时修正两个根本问题：

1. **反事实输入必须自洽**：反事实 Local 和反事实 Window 必须来自同一条反事实时间序列，不能一个表示异常、另一个仍表示正常。
2. **反事实监督必须有明确方向**：不能只要求“原答案概率降低”，而要明确告诉模型反事实窗口究竟是 `NORMAL` 还是 `ANOMALOUS`。

最核心的约束是：
$$
\boxed{
\exists X_i^-:
\quad
W_i^-=\operatorname{Fmt}(X_i^-[I_i]),
\qquad
L_i^-=E_0(X_i^-)[I_i]
}
$$
即：必须存在一条具体、可检查的反事实完整序列 $X_i^-$，使得 $W_i^-$ 和 $L_i^-$ 都从它产生。

这就是“完整一致”的严格定义。

------

# 沿用原来的符号

对第 $i$ 个 QA 窗口，定义：

| 符号                                   | 含义                                         |
| -------------------------------------- | -------------------------------------------- |
| $X_i$                                  | 当前 `time_series`，即模型原始看到的完整序列 |
| $N_i$                                  | 与 $X_i$ 配对的 `normal_series`              |
| $I_i=[s_i,e_i)$                        | 当前 QA 关注的时间窗口                       |
| $x_i=X_i[I_i]$                         | 当前窗口                                     |
| $n_i=N_i[I_i]$                         | 当前窗口的正常版本                           |
| $\delta_i=x_i-n_i$                     | 当前窗口相对正常版本的变化                   |
| $z_i\in\{0,1\}$                        | 窗口状态；0 表示正常，1 表示异常             |
| $Q_i$                                  | 原始自然语言问题                             |
| $Y_i$                                  | 原始完整答案                                 |
| $F$                                    | Fixed hint                                   |
| $E_0$                                  | 冻结的 Phase-I TS Encoder                    |
| $L_i=E_0(X_i)[I_i]$                    | 原始 Local 表示                              |
| $W_i=\operatorname{Fmt}(x_i)$          | 原始 Window 文本                             |
| $X_i^-$                                | 与当前窗口状态相反的完整反事实序列           |
| $L_i^-=E_0(X_i^-)[I_i]$                | 反事实 Local                                 |
| $W_i^-=\operatorname{Fmt}(X_i^-[I_i])$ | 反事实 Window                                |

这里的上标 “$-$” 不再表示“没有正确标签的负样本”，而表示：

> 与 anchor 状态相反的、拥有明确标签的反事实样本。

因此：
$$
\boxed{
z_i^- = 1-z_i
}
$$

------

# 为什么要从完整反事实序列 $X_i^-$ 出发

你原来的构造是分别定义：
$$
L_i^-,
\qquad
W_i^-.
$$
但分别定义会留下一个隐患：两个来源可能并不对应同一条真实序列。

例如：
$$
W_i^-=\operatorname{Fmt}(x_j),
$$
其中 $j\neq k$。这时 Window 来自 donor $j$，Local 来自 donor $k$，模型收到的是现实中不可能同时出现的证据组合。

即使 $j=k$，如果直接使用：
$$
L_i^-=E_0(X_j)[I_j],
$$
它依然包含：

- donor 的完整序列上下文；
- donor 的绝对位置；
- donor 的周边趋势；
- donor 的 patch 上下文。

而 Window 却被放在 anchor 的 `Steps s_i to e_i` 上。

所以更严格的做法是：

1. 先在 anchor 的时间坐标中构造一条完整 $X_i^-$；
2. 再从同一个 $X_i^-$ 同时产生 Local 和 Window。

定义 Patch 操作：
$$
\operatorname{Patch}(X,I,u)[t]
=
\begin{cases}
u_{t-s}, & t\in I=[s,e),\\
X[t], & t\notin I.
\end{cases}
$$
也就是说，只把 $X$ 在窗口 $I$ 中的值替换为 $u$，其他位置保持不变。

------

# 异常锚点 $z_i=1$：删除异常

当：
$$
z_i=1,
$$
原始窗口 $x_i$ 是异常窗口，配对正常窗口为：
$$
n_i=N_i[I_i].
$$

## 反事实完整序列

不要默认直接把完整 $N_i$ 送进模型，而是优先构造：
$$
\boxed{
X_i^-=
\operatorname{Patch}(X_i,I_i,n_i)
}
$$
即：
$$
X_i^-[t]
=
\begin{cases}
N_i[t], & t\in I_i,\\
X_i[t], & t\notin I_i.
\end{cases}
$$
它的含义非常清楚：

> 保持原完整序列和窗口外上下文不变，只删除目标窗口中的异常成分。

然后定义：
$$
\boxed{
W_i^-=\operatorname{Fmt}(n_i)
}
$$
以及：
$$
\boxed{
L_i^-=E_0(X_i^-)[I_i]
}
$$
注意这里与旧设计有一个重要区别：
$$
L_i^-\neq E_0(N_i)[I_i]
$$
默认推荐的是：
$$
L_i^-=E_0\big(
\operatorname{Patch}(X_i,I_i,n_i)
\big)[I_i].
$$
这样可以避免 `normal_series` 在窗口外的缩放、裁剪或生成差异污染 Local。

## 正负上下文及目标

原始异常上下文：
$$
C_i^{z,+}
=
(Q^z,L_i,W_i,\bar F),
$$
目标：
$$
z_i=1
\quad\Longrightarrow\quad
\texttt{ANOMALOUS}.
$$
正常反事实上下文：
$$
C_i^{z,-}
=
(Q^z,L_i^-,W_i^-,\bar F),
$$
目标：
$$
1-z_i=0
\quad\Longrightarrow\quad
\texttt{NORMAL}.
$$
其中：

- $Q^z$ 是固定的状态分类问题，后面详细定义；
- $\bar F$ 表示数值保持不变、参数冻结的 Fixed hint。

## 为什么优先使用局部 Patch，而不是完整 $N_i$

完整使用 $N_i$：
$$
X_i^-=N_i
$$
会同时改变：

- 目标窗口；
- 窗口外正常区域；
- 全局均值和方差；
- TS Encoder 的全局上下文。

而局部 Patch 保证：
$$
X_i^-[\overline{I_i}]
=
X_i[\overline{I_i}],
$$
因此对 Local 表示的改变主要来源于目标窗口。

只有当你验证：
$$
X_i\approx N_i
\quad\text{在所有正常区域成立}
$$
时，才可以直接使用完整 $N_i$。



------

# 正常锚点 $z_i=0$

当：
$$
z_i=0,
$$
原始上下文目标为：
$$
\texttt{NORMAL}.
$$
需要构造异常反事实 $X_i^-$，目标为：
$$
\texttt{ANOMALOUS}.
$$
这里的目标是：
$$
\underbrace{(L_i,W_i)}_{\texttt{NORMAL}}
\quad\longrightarrow\quad
\underbrace{(L_i^-,W_i^-)}_{\texttt{ANOMALOUS}}.
$$


------

## 方法 ：移植异常残差

通过检索
$$
j^* = \arg \min_j d_W(x_i,n_j)
$$
约束条件为：
$$
\begin{cases}
z_j = 1, \\
K_i = K_j, \\
\frac{d_S(i,j)}{a_S(j)} \le \tau \\
\frac{c_S(j)}{a_S(j)} \leq \tau.
\end{cases}
$$


假设异常是可加残差，定义 donor 异常残差：
$$
\delta_{j^*}=x_{j^*}-n_{j^*}.
$$
构造：
$$
X_i^-[I_i]
=
x_i+\delta_{j^*}
$$
即：
$$
X_i^-=
\operatorname{Patch}
\left(
X_i,I_i,x_i+\delta_{j^*}
\right)
$$


然后：
$$
W_i^-=
\operatorname{Fmt}(x_i+\delta_{j^*}), \qquad L_i^- = E_0(X_i^-)[I_i].
$$
该方法的优点是：
$$
X_i^-[I_i]-x_i=\delta_j,
$$
没有 donor 正常基线 $n_j-x_i$ 的污染。



我们选择一个完整配对 donor $(X_j,N_j,I_j)$，然后由它计算：
$$
x_j = X_k[I_j],\qquad n_j = N_j[I_j],\qquad \delta_j = x_j - n_j.
$$
一个可用 donor 必须满足：
$$
\boxed{z_j = 1,\qquad K_j = K_i}
$$
并且 $x_j,n_j$ 必须是同一条序列、同一个窗口的异常/正常配对。

不能：

- 从样本 $j$ 取 $x_j$；
- 从另一个样本 $k$ 取 $n_k$；
- 然后计算 $x_j-n_k$。

因为那样得到的不是异常残差，而是两个完全不同基线的差异。



更稳妥的候选集合是：
$$
\boxed{
\begin{aligned}
\mathcal A_i^{\mathrm{res}}
=
\{j:\quad
&z_j=1,\\
&K_j=K_i,\\
&a_W(j)>\gamma_W,\\
&c_W(j)/a_W(j)\le\tau_c,\\
&d_W(i,j)/a_W(j)\le\tau_b,\\
&\operatorname{Valid}(x_i+\delta_j)=1
\}.
\end{aligned}
}
$$
其中：
$$
d_W(i,j)
=
\operatorname{RMS}
\left(
q(x_i)-q(n_j)
\right).
$$
这里保留 $d_W$ 的目的已经不是消除 $n_j-x_i$——残差移植已经消除了它——而是：

> 检查 donor 异常残差产生时的正常背景，与 anchor 背景是否处于相近的数据尺度和形态。



#### 如何从候选集合里选出 $j^*$

**最保守版本：选择最近的 donor**
$$
\boxed{
j^*
=
\arg\min_{j\in\mathcal A_i^{\mathrm{res}}}
d_W(i,j)
}
$$
优点：

- 最简单；
- 可复现；
- 背景兼容性最高。

缺点：

- 某些 donor 可能被重复使用很多次；
- 异常多样性不足。







**推荐版本：Top-M 随机采样**

先找：
$$
\operatorname{TopM}_i
=
\text{按 }d_W(i,j)\text{ 最小排序的前 }M\text{ 个 donor}.
$$
然后：
$$
\boxed{
j^*
\sim
\operatorname{Uniform}
(\operatorname{TopM}_i)
}
$$
推荐：
$$
M=8
\quad\text{或}\quad
M=16.
$$
优点：

- 仍然保证背景兼容；
- 增加异常类型和形态多样性；
- 防止单个 donor 成为“万能 donor”。



# 十二、完整样本构造步骤

对正常 anchor $i$：

## 步骤 1：确认 anchor 可靠正常

要求：
$$
z_i=0,
\qquad
\texttt{anomaly\_descriptions}_i=[].
$$
还可检查：
$$
\operatorname{RMS}
\left(
q(x_i)-q(n_i)
\right)
\le\gamma_{\rm normal}.
$$
如果标注正常窗口与自己的 normal_series 差异很大，先排除。

## 步骤 2：确定窗口长度

$$
K_i=e_i-s_i.
$$

只在：
$$
K_j=K_i
$$
的 donor 桶中选择。

## 步骤 3：选择异常 donor

从可靠 donor 池中寻找：
$$
j^*\in\mathcal A_i^{\mathrm{res}}.
$$
第一版使用：
$$
j^*=\arg\min d_W(i,j).
$$

## 步骤 4：计算异常残差

$$
\boxed{
\delta_{j^*}
=
x_{j^*}-n_{j^*}
}
$$

## 步骤 5：构造异常窗口

$$
\boxed{
x_i^-
=
x_i+\delta_{j^*}
}
$$

## 步骤 6：构造完整反事实序列

$$
\boxed{
X_i^-=
\operatorname{Patch}
(X_i,I_i,x_i^-)
}
$$

## 步骤 7：重新产生 Window 和 Local

$$
\boxed{
W_i^-=
\operatorname{Fmt}(x_i^-)
}
$$

## 步骤 8：赋予明确状态标签

原始状态样本：
$$
(Q^z,L_i,W_i,\bar F)
\longrightarrow
\texttt{NORMAL}.
$$
反事实状态样本：
$$
(Q^z,L_i^-,W_i^-,\bar F)
\longrightarrow
\texttt{ANOMALOUS}.
$$
注意：不再使用原始自然语言答案 $Y_i$ 监督反事实上下文。





------

# 什么叫“显式 normal/anomalous 目标”

旧方法用原始答案 $Y_i$：
$$
P(Y_i\mid C^+)\uparrow,
\qquad
P(Y_i\mid C^-)\downarrow.
$$
新方法不再要求反事实上下文回答原始 MC/TF/OE 问题，而是定义一个固定辅助问题 $Q^z$：

```
Classify only the target time-series window.

NORMAL means that the target window contains no anomaly.
ANOMALOUS means that the target window contains an anomaly.

Return exactly one label.
State:
```

对所有样本，$Q^z$ 完全相同。

原始问题 $Q_i$ 只用于原始答案损失，绝不用于状态辅助损失。

这样可以避免：

- TF 命题正负极性；
- MC 正确选项不确定；
- OE 没有唯一反义答案；
- 问题模板泄露答案。

------

# 状态标签最好使用两个单 token verbalizer

直接输出完整的 `NORMAL` 和 `ANOMALOUS` 可能存在 token 数不同的问题。

启动时检查：

```
pairs = [
    (" normal", " anomalous"),
    (" N", " A"),
    (" 0", " 1"),
]

for normal_text, anomaly_text in pairs:
    normal_ids = tokenizer.encode(
        normal_text,
        add_special_tokens=False
    )
    anomaly_ids = tokenizer.encode(
        anomaly_text,
        add_special_tokens=False
    )

    if len(normal_ids) == 1 and len(anomaly_ids) == 1:
        normal_id = normal_ids[0]
        anomaly_id = anomaly_ids[0]
        break
else:
    raise RuntimeError("No one-token state verbalizers found")
```

如果使用 `0/1`，prompt 明确定义：

```
0 = NORMAL
1 = ANOMALOUS
Return exactly one token.
State:
```

语义仍然是显式的，只是输出 token 更容易处理。

------

# 二分类概率如何从冻结 LLM 中得到

设状态 prompt 最后一个位置的隐藏状态为：
$$
h_\theta(C)\in\mathbb R^{d_{\rm LLM}}.
$$
LLM 的输出层中，对应 NORMAL 和 ANOMALOUS token 的权重分别为：
$$
u_0,\qquad u_1.
$$
定义两个 logits：
$$
\ell_0(C)=u_0^\top h_\theta(C),
$$
然后只在这两个 logits 上做 softmax：
$$
P_\theta(z=1\mid C)
=
\frac{\exp\ell_1(C)}
{\exp\ell_0(C)+\exp\ell_1(C)}.
$$
单个上下文的状态损失为：
$$
\boxed{
\ell_z(C,r)
=
-\log
\frac{\exp\ell_r(C)}
{\exp\ell_0(C)+\exp\ell_1(C)}
}
$$
其中 $r\in\{0,1\}$ 是明确状态标签。

这样：

- 不生成长答案；
- 不监督 EOS；
- 不存在 answer-head 长度问题；
- 不需要 margin；
- 不需要额外正 head CE；
- 直接得到可解释的 normal/anomalous 概率。

------

# 一个 pair 的完整状态损失

对原始上下文：
$$
C_i^{z,+}
=
(Q^z,L_i,W_i,\bar F),
$$
目标：
$$
z_i.
$$
对反事实上下文：
$$
C_i^{z,-}
=
(Q^z,L_i^-,W_i^-,\bar F),
$$
目标：
$$
1-z_i.
$$
定义：
$$
\boxed{
\mathcal L_{\mathrm{state},i}
=
\frac12
\left[
\ell_z(C_i^{z,+},z_i)
+
\ell_z(C_i^{z,-},1-z_i)
\right]
}
$$
因为每个 pair 一定同时包含一个正常和一个异常标签，所以状态损失在 pair 内天然类别平衡。

------

# 完整训练目标

原始答案损失保持不变：
$$
\boxed{
\mathcal L_{\mathrm{answer},i}
=
\operatorname{CE}
\left(
Y_i\mid Q_i,L_i,W_i,F
\right)
}
$$
注意：

- 只在原始真实上下文上训练完整答案；
- 不为 $C_i^{z,-}$ 构造自然语言反事实答案；
- 不对反事实上下文使用原始 $Y_i$。

总损失：
$$
\boxed{
\mathcal L_i
=
\mathcal L_{\mathrm{answer},i}
+
\beta
\mathcal L_{\mathrm{state},i}
}
$$
batch 中若有效反事实 pair 集合为 $\mathcal V$，则：
$$
\boxed{
\mathcal L_{\mathrm{state}}
=
\frac{1}{|\mathcal V|}
\sum_{i\in\mathcal V}
\mathcal L_{\mathrm{state},i}
}
$$
若某个 batch 没有有效 pair：
$$
\mathcal L_{\mathrm{state}}=0,
$$
只训练：
$$
\mathcal L=\mathcal L_{\mathrm{answer}}.
$$
绝对不能对无效 pair 使用“原样本复制一份但标签取反”，否则会给完全相同的输入赋两个相反标签





# 具体 batch 构造

假设一个 batch 中有 $B$ 个有效 anchor。

需要准备：

```
positive_local       # [B, K, d]
counterfactual_local # [B, K, d]

positive_window       # list length B
counterfactual_window # list length B

z                     # [B], 0 or 1
```

把两个状态上下文合并：

```
state_local = torch.cat(
    [positive_local, counterfactual_local],
    dim=0,
)

state_window = (
    positive_window
    + counterfactual_window
)

state_targets = torch.cat(
    [z, 1 - z],
    dim=0,
)
```

这里：

- 前 $B$ 行是原始上下文；
- 后 $B$ 行是反事实上下文；
- 每个 pair 的两个状态都参与同一次 CE。

训练伪代码：

```
answer_loss = model.answer_nll(
    questions=Q,
    answers=Y,
    local_embeddings=positive_local,
    window_values=positive_window,
    fixed_hint=True,
)

state_logits = model.state_logits(
    question=Q_state,
    local_embeddings=state_local,
    window_values=state_window,
    fixed_hint_frozen=True,
)

state_loss = F.cross_entropy(
    state_logits,
    state_targets,
)

loss = answer_loss + beta * state_loss
```

不再需要：

```
mode = random.choice(...)
```

也不再需要：

- `active_hinges`
- `margin`
- answer-head mask
- 正负 head NLL
- 每个 rank 随机不同模式

------

# `state_logits()` 的高效实现

不需要生成完整词表在所有时间位置上的 logits。

```
def state_logits(
    axis,
    input_ids,
    attention_mask,
    local_embeddings,
    starts,
    ends,
    normal_id,
    anomalous_id,
):
    embeds = axis.get_hint_embeddings(
        input_ids=input_ids,
        local_embeddings=local_embeddings,
        start_indices=starts,
        end_indices=ends,
    )

    hidden = axis.model.model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state

    # 适用于右侧 padding
    last_index = attention_mask.sum(dim=1) - 1
    batch_index = torch.arange(
        hidden.shape[0],
        device=hidden.device,
    )

    final_hidden = hidden[batch_index, last_index]

    selected_weight = axis.model.lm_head.weight[
        [normal_id, anomalous_id]
    ]

    logits = F.linear(
        final_hidden,
        selected_weight,
    )

    return logits
```

输出形状为：

```
[B, 2]
```

注意：

- LLM 参数可以 `requires_grad=False`；
- 但不能把整个 LLM forward 放进 `torch.no_grad()`；
- 否则梯度无法从状态损失传回 Local Perceiver；
- TS Encoder $E_0$ 可以并且应该在 `no_grad()` 中运行。



------

# 训练参数建议

当前状态损失是标准二分类 CE，尺度已经比旧 answer-head SLR 清楚很多。

建议 warm-start baseline 时：

```
beta: 0 → 0.1，前 10% steps 线性增长
local_word_proj LR: 5e-5
local_attention LR: 2e-5
Fixed branch LR: 0
LLM LR: 0
TS Encoder LR: 0
gradient clipping: 1.0
```



先给出最直接的答案：

> “二分类概率”不需要额外训练一个二分类头，也不需要让冻结 LLM 生成完整答案。
> 我们给冻结 LLM 一个固定的状态判断 prompt，然后读取它在下一个 token 位置上对 `normal` 和 `anomalous` 两个候选 token 给出的 logits，再只在这两个 logits 之间做 Softmax。

而且：

> 状态分类 prompt 与原 AXIS 问答 prompt 使用同一个冻结 LLM、同一套 Window/Local/Fixed 证据注入机制，但不建议拼成同一条 prompt。它们是两个独立的训练任务、两次独立前向传播；工程上可以放进同一个 batch 并行计算。

------

# 一、自顶向下：为什么需要二分类概率？

## 1. 原 AXIS 的训练目标究竟在做什么？

AXIS 原始 Phase II 的训练目标是生成完整答案：
$$
\mathcal L_{\mathrm{answer}}
=
-\frac{1}{|Y_i|}
\sum_{t=1}^{|Y_i|}
\log P_\Theta
\left(
Y_{i,t}
\mid
Y_{i,<t},Q_i,L_i,W_i,F
\right).
$$
其中：

- $i$：第 $i$ 个训练样本；
- $Q_i$：原始 QA 问题，例如判断题、选择题或开放题；
- $Y_i$：对应的完整自然语言答案；
- $W_i$：窗口中的数值文本；
- $L_i$：由整条时间序列编码后截取出的 Local 表示；
- $F$：Fixed task-prior hint；
- $\Theta$：冻结 LLM 的参数。

这个目标只要求：

> 在真实证据下，完整答案 $Y_i$ 的概率尽可能高。

但它没有明确告诉模型：

> 当前窗口究竟是 normal 还是 anomalous。

完整答案可能有几十个 token，例如：

> False. There is no evidence of an anomaly occurring within the specified interval. The values remain relatively stable...

其中真正决定异常状态的可能只有第一个 `False`，后面大量 token 都是语言组织、复述和解释。

因此，模型可能通过以下捷径降低答案损失：

- 问题模板先验；
- 选项分布；
- Fixed hint；
- 常见答案句式；
- 训练集标签比例；
- 前几个答案 token 的统计规律。

这时即使 Local 或 Window 没被真正使用，完整答案损失也可能下降。

------

## 2. 显式二分类目标做的是什么？

我们增加一个非常明确的辅助任务：

> 给定 Window、Local、Fixed，当前区间是正常还是异常？

定义标签：
$$
z_i=
\begin{cases}
0,& \text{normal}\\
1,& \text{anomalous}.
\end{cases}
$$
状态分类任务不要求生成长答案，只要求输出：

```
0
```

或者：

```
1
```

其中固定约定：

```
0 = normal
1 = anomalous
```

这样做的核心价值是：

- 输出空间只有两个明确状态；
- 反事实证据对应的正确目标也很明确；
- 不再只是“降低原答案概率”；
- 而是明确要求反事实样本从 anomalous 变成 normal，或从 normal 变成 anomalous。

------

## 3. 为什么需要“真实 + 反事实”成对监督？

对于每个样本 $i$，构造：

- 真实完整序列 $X_i$；
- 完整反事实序列 $X_i^-$。

二者必须对应相反状态：
$$
z(X_i^-)=1-z_i.
$$
然后从两条完整序列分别生成一致的 Window 和 Local：
$$
W_i=\operatorname{Fmt}(X_i[I_i]),
\qquad
L_i=E_0(X_i)[I_i],
$$
其中：

- $I_i=[s_i,e_i)$：目标区间；
- $E_0$：冻结的时间序列编码器；
- $\operatorname{Fmt}$：AXIS 的数值文本格式化过程。

真实证据的目标是 $z_i$，反事实证据的目标是 $1-z_i$：
$$
(Q^z,L_i,W_i,F)\longrightarrow z_i,
$$
这里 $Q^z$ 是所有样本共用的固定状态判断问题。

这样，同一个 $Q^z$、同一个 $F$，必须在两组时间序列证据下输出相反状态。只依赖固定问题或 Fixed hint 就无法同时做对两个样本。

------

# 二、它是否与原 AXIS prompt 一起输入？

需要区分两个“一起”。

## 1. 是否使用同一个冻结 LLM？

是。

原答案任务和状态任务都经过同一个冻结 LLM：
$$
G_\Theta,
\qquad
\Theta\text{ 不更新}.
$$
它们也共用：

- 同一个 tokenizer；
- 同一个词嵌入层；
- 同一个 LM head；
- 同一个 Local Hint Tuner；
- 同一个 Fixed Hint Tuner；
- 同样的 placeholder embedding 替换机制。

------

## 2. 是否拼成同一条 prompt？

不建议。

推荐结构是：

| 任务             | 问题       | 证据            | 监督目标       |
| ---------------- | ---------- | --------------- | -------------- |
| 原 AXIS 答案任务 | 原始 $Q_i$ | $L_i,W_i,F$     | 完整答案 $Y_i$ |
| 真实状态任务     | 固定 $Q^z$ | $L_i,W_i,F$     | $z_i$          |
| 反事实状态任务   | 固定 $Q^z$ | $L_i^-,W_i^-,F$ | $1-z_i$        |

也就是说，每个 anchor 最清楚的实现是三个输入序列：
$$
\mathcal P_i^{\mathrm{answer}},
\qquad
\mathcal P_i^{z,+},
\qquad
\mathcal P_i^{z,-}.
$$
它们可以在一次 model call 中沿 batch 维拼起来，但彼此是三个独立序列，没有相互注意力。

------

## 3. 为什么不能把状态问题接在原答案后面？

假如构造：

```
原始 AXIS 问题
原始 gold answer
请再判断它是 normal 还是 anomalous
```

训练时使用 teacher forcing，状态判断可以直接从前面的 gold answer 中读取答案。

例如前文已经写了：

```
There is no evidence of an anomaly.
```

那么后面的 `normal` 完全不需要读取时间序列。辅助任务反而变成了文本理解任务。

即使不放入 gold answer，同时放置两个问题，也可能出现：

- 两个任务指令相互干扰；
- 原始问题本身泄漏标签；
- 不同 QA 类型的语义不统一；
- 很难确定应该在哪个位置读取状态 logits。

所以最干净的设计是：

> 复用 AXIS 的证据模板，但把原始任意问题 $Q_i$ 替换为固定状态问题 $Q^z$。

------

# 三、状态分类 prompt 具体长什么样？

一个可执行的设计如下：

```
System:
You are an expert time-series anomaly analyst.

User:
Analyze the following time-series evidence.

### Time Series Data
- Window: Steps {s_i} to {e_i}
- Values (scaled by 100): {W_i}

### Contextual Hints
- Per-Step Analysis:
  <|local hint|> <|local hint|> ... <|local hint|>

- Fixed Hints:
  <|fixed hint|> <|fixed hint|> ... <|fixed hint|>

### State Classification
Classify the target interval using only the evidence above.

Output exactly one label:
0 = normal
1 = anomalous

Label:

Assistant:
```

LLM 接下来要预测的第一个 token 就是 `0` 或 `1`。

选择 `0/1` 而不是直接选择 `normal/anomalous`，主要是为了避免 tokenizer 把 `anomalous` 分成多个 token。

如果检查后发现 `" normal"` 和 `" anomalous"` 都恰好是单 token，也可以直接使用它们。

------

# 四、Window、Local、Fixed 是怎样进入冻结 LLM 的？

## 1. Window：正常文本 token

窗口：
$$
x_i=X_i[I_i]
$$
经过 AXIS 的格式化过程：
$$
W_i=\operatorname{Fmt}(x_i).
$$
例如：

```
103, 98, 101, 99, 442, 105
```

这些文本经过 tokenizer 得到普通 token ID，再通过冻结 LLM 的 embedding lookup 得到词向量。

------

## 2. Local：先放 placeholder，再替换 embedding

首先让整条序列经过冻结时间序列编码器：
$$
H_i=E_0(X_i).
$$
然后截取目标窗口：
$$
L_i=H_i[I_i].
$$
注意，$L_i$ 还不一定处于 LLM 的 hidden space。令可训练 Hint Tuner 为 $T_\phi$，则：
$$
\widetilde L_i=T_\phi(L_i)
\in\mathbb R^{K_i\times d_h},
$$
其中：

- $K_i$：Local token 数量；
- $d_h$：LLM hidden dimension；
- $\phi$：可训练 Hint Tuner 参数。

Prompt 中先放置 $K_i$ 个：

```
<|local hint|>
```

token。完成普通 embedding lookup 后，再将这些 placeholder 的 embedding 替换成 $\widetilde L_i$。

------

## 3. Fixed：同样通过 placeholder 注入

Fixed soft hint 表示为：
$$
\widetilde F_\phi
\in\mathbb R^{K_F\times d_h},
$$
其中 $K_F$ 是配置中的 Fixed token 数量。

Prompt 中放入 $K_F$ 个：

```
<|fixed hint|>
```

然后用 $\widetilde F_\phi$ 替换这些位置的词嵌入。

因此，最终送入 LLM 的不是纯文本，而是：
$$
U_i=
[
U_{\mathrm{text}};
\widetilde L_i;
\widetilde F_\phi;
U_{Q^z}
].
$$
其中具体顺序由 AXIS 模板决定。

------

# 五、自底向上：冻结 LLM 怎样产生二分类概率？

## 1. 冻结 LLM 仍然会输出 vocabulary logits

假设状态 prompt 一共有 $T$ 个输入位置，其输入 embedding 为：
$$
U=(u_1,u_2,\dots,u_T).
$$
冻结 Transformer 计算：
$$
(h_1,h_2,\dots,h_T)=G_\Theta(U).
$$
其中 $\Theta$ 冻结，但前向运算仍然正常进行。

在最后一个 prompt 位置 $T$，LM head 计算：
$$
r_T=W_{\mathrm{LM}}h_T+b.
$$
其中：
$$
r_T\in\mathbb R^{|\mathcal V|}
$$
是整个词表上每个 token 的 logit。

例如词表大小为 152,000，那么 $r_T$ 就有 152,000 个数。

这些 logits 表示：

> 在当前状态 prompt 后面，每个 vocabulary token 作为下一个 token 的相对合理程度。

------

## 2. 取出两个候选标签的 logits

定义：

- $v_0$：标签 `0` 对应的 token ID；
- $v_1$：标签 `1` 对应的 token ID。

从完整 vocabulary logits 中取出：
$$
s_0=r_T[v_0],
\qquad
s_1=r_T[v_1].
$$
于是得到两个类别的原始分数：
$$
\mathbf s(C)=
\begin{bmatrix}
s_0\\
s_1
\end{bmatrix}.
$$
这里 $C$ 表示当前完整上下文，例如：
$$
C=(Q^z,L_i,W_i,F).
$$
这两个分数来自冻结 LLM 原有的 LM head，不是新建的 `nn.Linear(hidden_size, 2)`。

------

## 3. 为什么需要只在两个类别之间重新归一化？

冻结 LLM 原本的全词表概率是：
$$
P_{\mathrm{vocab}}(v\mid C)
=
\frac{\exp(r_T[v])}
{\sum_{u\in\mathcal V}\exp(r_T[u])}.
$$
但当前任务明确规定只能输出 `0` 或 `1`，因此我们关心的是：
$$
P(z=c\mid C,z\in\{0,1\}).
$$
于是只对两个 logits 做 Softmax：
$$
P(z=0\mid C)
=
\frac{\exp(s_0)}
{\exp(s_0)+\exp(s_1)},
$$
进一步可写成：
$$
\boxed{
P(z=1\mid C)=\sigma(s_1-s_0)
}
$$
其中：
$$
\sigma(a)=\frac{1}{1+\exp(-a)}
$$
是 Sigmoid 函数。

因此真正决定异常概率的是一个非常简单的量：
$$
\boxed{
\Delta s(C)=s_1-s_0
}
$$

- $\Delta s>0$：更倾向 anomalous；
- $\Delta s<0$：更倾向 normal；
- $\Delta s=0$：二者各为 0.5。

------

## 4. 数值例子

假设真实异常证据下：
$$
s_0=2.0,\qquad s_1=4.0.
$$
那么：
$$
P(z=1\mid C)
=
\sigma(4-2)
=
\sigma(2)
\approx0.881.
$$
模型认为异常概率约为 88.1%。

把异常删除，得到完整正常反事实后：
$$
s_0^-=3.5,\qquad s_1^-=1.5.
$$
于是：
$$
P(z=1\mid C^-)
=
\sigma(1.5-3.5)
=
\sigma(-2)
\approx0.119.
$$
这正是我们希望看到的：
$$
X_i\longrightarrow 88.1\%\text{ anomalous},
$$

------

# 六、显式状态损失怎样定义？

对某个上下文 $C$ 和标签 $z$，定义二分类交叉熵：
$$
\ell_z(C,z)
=
-\log P(z\mid C).
$$
等价写法是：
$$
\ell_z(C,z)
=
-s_z
+
\log\left(\exp(s_0)+\exp(s_1)\right).
$$
真实和反事实组成一个配对：
$$
C_i^{z,+}=(Q^z,L_i,W_i,F),
$$
状态损失为：
$$
\boxed{
\mathcal L_{\mathrm{state},i}
=
\frac12
\left[
\ell_z(C_i^{z,+},z_i)
+
\ell_z(C_i^{z,-},1-z_i)
\right]
}
$$
这里要特别注意：

- 上标 $+$ 表示真实 factual context；
- 上标 $-$ 表示反事实 context；
- 它们不代表正类别和负类别。

例如异常 anchor $z_i=1$：

| 输入            | 目标           |
| --------------- | -------------- |
| $(L_i,W_i)$     | anomalous，$1$ |
| $(L_i^-,W_i^-)$ | normal，$0$    |

正常 anchor $z_i=0$：

| 输入                         | 目标           |
| ---------------------------- | -------------- |
| $(L_i,W_i)$                  | normal，$0$    |
| 残差移植后的 $(L_i^-,W_i^-)$ | anomalous，$1$ |

最终目标：
$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{answer}}
+
\beta\mathcal L_{\mathrm{state}}
}
$$
其中原答案损失和状态损失都应按各自监督 token 数量取平均，不能把“几十个答案 token 的 NLL 总和”与“一个分类 token 的 CE”直接相加。

初始实现可以先用：
$$
\beta=1,
$$
然后根据验证集和两部分梯度大小调整。

------

# 七、为什么 LLM 冻结后，这个损失仍然能训练？

“冻结 LLM”只表示：
$$
\Theta\leftarrow\Theta
$$
即 LLM 参数不被优化器更新。

它不表示切断从 LLM 输出到输入 embedding 的梯度。

设可训练 Hint Tuner 参数为 $\phi$，则：
$$
\widetilde L_i=T_\phi(L_i),
\qquad
\widetilde F=F_\phi.
$$
状态损失的梯度路径为：
$$
\frac{\partial\mathcal L_{\mathrm{state}}}{\partial\phi}
=
\frac{\partial\mathcal L_{\mathrm{state}}}{\partial(s_0,s_1)}
\frac{\partial(s_0,s_1)}{\partial U}
\frac{\partial U}{\partial(\widetilde L,\widetilde F)}
\frac{\partial(\widetilde L,\widetilde F)}{\partial\phi}.
$$
LLM 是中间一个固定但可微的函数。

可以把它类比为一把被冻结的尺子：

- 尺子的刻度不改变；
- 但你仍然可以根据读数调整被测物体的位置；
- 这里被调整的是 Hint Tuner 产生的软提示向量。

对于交叉熵，有：
$$
\frac{\partial\ell_z}{\partial s_c}
=
P(z=c\mid C)-\mathbf 1[c=z].
$$
如果真实标签为 anomalous，但 $P(z=1)$ 太小，梯度会推动输入 soft hint 发生变化，使冻结 LLM 的：
$$
s_1-s_0
$$
变大。

------

## 一个非常重要的代码问题

正确冻结方式：

```
for p in llm.parameters():
    p.requires_grad_(False)

llm.eval()
```

但训练 Hint Tuner 时不能这样做：

```
with torch.no_grad():
    outputs = llm(inputs_embeds=inputs_embeds)
```

后者会切断 LLM 输出到 Local/Fixed soft embedding 的梯度。

所以：

> 冻结参数不等于使用 `torch.no_grad()` 包裹 LLM 前向传播。

时间序列编码器可以放在 `no_grad()` 中，因为它前面通常没有可训练模块：

```
with torch.no_grad():
    H = ts_encoder(X)
```

但 Hint Tuner 和 LLM 前向必须保留计算图：

```
local_soft = hint_tuner(H)
outputs = llm(inputs_embeds=inputs_embeds)
```

------

# 八、具体 PyTorch 实现

下面给出单样本版本，比较适合 AXIS 当前常见的 batch size 1 设置。

```
import torch
import torch.nn.functional as Fnn


def get_one_token_id(tokenizer, text):
    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )
    if len(ids) != 1:
        raise ValueError(
            f"{text!r} is tokenized into {ids}, "
            "please use another verbalizer."
        )
    return ids[0]


# Prompt 的 Label: 后面输出一个带空格的数字
normal_token_id = get_one_token_id(tokenizer, " 0")
anomaly_token_id = get_one_token_id(tokenizer, " 1")
```

不要直接写：

```
tokenizer.convert_tokens_to_ids("0")
```

因为 BPE/SentencePiece 中，`"0"` 与 `" 0"` 可能对应不同 token。

------

## 1. 将 soft embedding 写入 placeholder 位置

```
def inject_soft_tokens(
    base_embeddings,     # [1, S, D]
    input_ids,           # [1, S]
    placeholder_id,
    soft_embeddings,     # [K, D]
):
    B, S, D = base_embeddings.shape
    assert B == 1

    positions = (
        input_ids[0] == placeholder_id
    ).nonzero(as_tuple=False).flatten()

    if positions.numel() != soft_embeddings.shape[0]:
        raise ValueError(
            f"Found {positions.numel()} placeholders, "
            f"but got {soft_embeddings.shape[0]} soft embeddings."
        )

    # 使用非原地 index_copy，保留到 soft_embeddings 的梯度
    flat = base_embeddings.reshape(S, D)
    flat = torch.index_copy(
        flat,
        dim=0,
        index=positions,
        source=soft_embeddings,
    )

    return flat.unsqueeze(0)
```

------

## 2. 从冻结 LLM 中取得两个 logits

```
def get_state_binary_logits(
    llm,
    input_ids,           # [1, S]
    attention_mask,      # [1, S]
    local_soft,          # [K_local, D]
    fixed_soft,          # [K_fixed, D]
    local_placeholder_id,
    fixed_placeholder_id,
    normal_token_id,
    anomaly_token_id,
):
    # 普通文本 token 的 embedding 来自冻结 LLM
    inputs_embeds = llm.get_input_embeddings()(input_ids)

    # 替换 Local placeholder
    inputs_embeds = inject_soft_tokens(
        inputs_embeds,
        input_ids,
        local_placeholder_id,
        local_soft,
    )

    # 替换 Fixed placeholder
    inputs_embeds = inject_soft_tokens(
        inputs_embeds,
        input_ids,
        fixed_placeholder_id,
        fixed_soft,
    )

    # 至少 Local/Fixed 中一条路径应当可训练
    assert inputs_embeds.requires_grad

    # 注意：这里不能使用 torch.no_grad()
    outputs = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )

    # 找到最后一个非 padding prompt token
    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).unsqueeze(0)

    decision_position = (
        positions * attention_mask
    ).amax(dim=1)[0]

    # 该位置的 logits 用来预测下一个 token
    vocabulary_logits = outputs.logits[
        0, decision_position, :
    ]

    binary_logits = vocabulary_logits[
        [normal_token_id, anomaly_token_id]
    ]

    # binary_logits[0] = normal logit
    # binary_logits[1] = anomalous logit
    return binary_logits
```

------

## 3. 计算真实和反事实状态损失

```
real_logits = get_state_binary_logits(
    llm=llm,
    input_ids=real_state_input_ids,
    attention_mask=real_state_attention_mask,
    local_soft=real_local_soft,
    fixed_soft=fixed_soft,
    local_placeholder_id=local_placeholder_id,
    fixed_placeholder_id=fixed_placeholder_id,
    normal_token_id=normal_token_id,
    anomaly_token_id=anomaly_token_id,
)

counterfactual_logits = get_state_binary_logits(
    llm=llm,
    input_ids=cf_state_input_ids,
    attention_mask=cf_state_attention_mask,
    local_soft=cf_local_soft,
    fixed_soft=fixed_soft,
    local_placeholder_id=local_placeholder_id,
    fixed_placeholder_id=fixed_placeholder_id,
    normal_token_id=normal_token_id,
    anomaly_token_id=anomaly_token_id,
)

z = torch.tensor(
    [z_i],
    dtype=torch.long,
    device=real_logits.device,
)

z_cf = 1 - z

loss_real = Fnn.cross_entropy(
    real_logits.unsqueeze(0),
    z,
)

loss_cf = Fnn.cross_entropy(
    counterfactual_logits.unsqueeze(0),
    z_cf,
)

loss_state = 0.5 * (loss_real + loss_cf)

loss = loss_answer + beta * loss_state

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

得到概率：

```
real_probs = torch.softmax(real_logits, dim=-1)

p_normal = real_probs[0]
p_anomalous = real_probs[1]
```

训练阶段不需要调用：

```
llm.generate(...)
```

因为：

- `generate()` 包含离散采样或贪心选择；
- 不适合直接反向传播；
- 我们只需要一次普通 forward 得到 logits。

------

# 九、完整反事实怎样接入状态分类？

这里的状态分类主目标应该使用完整一致反事实，而不是 Local-only 或 Window-only 混搭。

## 异常 anchor：$z_i=1$

真实序列：
$$
X_i
$$
正常反事实：
$$
X_i^-=N_i.
$$
分别计算：
$$
W_i=\operatorname{Fmt}(X_i[I_i]),
\qquad
L_i=E_0(X_i)[I_i],
$$
目标为：
$$
(L_i,W_i)\rightarrow1,
$$

------

## 正常 anchor：$z_i=0$

通过异常残差移植构造：
$$
\delta_j=x_j-n_j,
$$
再将它补回完整序列：
$$
X_i^-=\operatorname{Patch}(X_i,I_i,x_i^-).
$$
然后：
$$
W_i^-=\operatorname{Fmt}(X_i^-[I_i]),
$$
目标为：
$$
(L_i,W_i)\rightarrow0,
$$
关键要求仍然是：

> $W_i^-$ 和 $L_i^-$ 必须来自同一条完整反事实序列 $X_i^-$。

不能使用：
$$
W_i^-=\operatorname{Fmt}(X_i^-[I_i]),
\qquad
L_i^-=E_0(X_j)[I_j],
$$
因为这会让两个证据来源描述不同的时间序列。

------

# 十、这个概率到底是不是“真实异常概率”？

训练中得到的：
$$
P(z=1\mid C)
=
\operatorname{Softmax}(s_0,s_1)_1
$$
首先是一个“在 `normal/anomalous` 两个 verbalizer 之间归一化的模型置信度”。

它不自动等于统计意义上完全校准的真实异常后验概率，原因包括：

- 冻结 LLM 的 logits 本身可能未校准；
- 反事实训练严格形成 1:1 类别比例；
- 实际测试数据中的异常比例可能远低于 50%；
- 残差移植生成的异常和真实异常分布可能不完全一致。

但如果它主要用于辅助训练，这不构成严重问题。它最重要的作用是提供正确的有方向监督：
$$
\text{真实异常}\Rightarrow s_1-s_0\uparrow,
$$
如果以后将它作为最终异常概率输出，再在真实验证集上进行温度校准：
$$
P_\tau(z=1\mid C)
=
\sigma\left(\frac{s_1-s_0}{\tau}\right).
$$

------

# 十一、最容易出现的实现错误

需要重点检查以下几项：

1. **把冻结等同于 `torch.no_grad()`**

   这会切断状态损失到 Hint Tuner 的梯度。

2. **读取错位置的 logits**

   状态类别由最后一个 prompt token 位置的 logits 预测，而不是读取输入标签 token 自己位置的 logits。

3. **`normal/anomalous` 实际是多 token**

   优先使用经过检查的单 token `0/1`。

4. **将标签提前放入 prompt**

   计算 logits 时，`0/1` 是待预测目标，不应该已经出现在输入中。

5. **将原始 gold answer 放入状态 prompt**

   会产生严重的答案泄漏。

6. **真实和反事实只替换 Window 或只替换 Local**

   主状态损失应使用完整一致的 $(L,W)$ 对；单来源替换留作诊断实验。

7. **真实与反事实没有成对进入同一训练 step**

   最好每个 anchor 同时计算一条 normal 和一条 anomalous，使固定问题 $Q^z$ 与 Fixed hint 无法依赖类别比例投机。

8. **长答案损失使用 sum，状态损失使用 mean**

   两者必须先分别归一化，否则 $\beta$ 没有稳定含义。

------

# 最后总结

这部分完整流程可以压缩成：
$$
X
\rightarrow
(W,L,F)
\rightarrow
\text{固定状态 prompt}
\rightarrow
\text{冻结 LLM}
\rightarrow
(s_{\mathrm{normal}},s_{\mathrm{anomalous}})
$$
状态 prompt 与原 AXIS prompt 的关系则是：

> 共用 AXIS 的证据模板、soft-token 注入方式和冻结 LLM；但用固定状态问题 $Q^z$ 替换原始问题 $Q_i$，形成独立前向序列，而不是把两个问题和答案拼在同一条 prompt 中。
