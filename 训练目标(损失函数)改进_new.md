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



------

# 最终推荐的第一版方案

建议精确实现：
$$
\begin{gather*}
X_i^- = \text{Patch}(X_i,I_i,N_i[I_i]),\qquad z_i = 1 \\
W_i^- = \text{Fmt}(X_i^-[I_i])\qquad L_i^- = E_0(X_i^-)[I_i] \\
 \mathcal { L } = \mathcal { L } _ { \mathrm { a n s w e r } } + \beta \frac { 1 } { 2 } \left[ \mathrm { C E } _ { 2 } ( z _ { i } \mid Q ^ { z } , L _ { i } , W _ { i } , \bar { F } ) + \mathrm { C E } _ { 2 } ( 1 - z _ { i } \mid Q ^ { z } , L _ { i } ^ { - } , W _ { i } ^ { - } , \bar { F } ) \right] 
\end{gather*}
$$


这版最简单，却已经修复了当前方案最核心的三个问题：

1. Local/Window 不再互相冲突；
2. 反事实拥有明确正确目标；
3. 不再通过极强的第一句监督破坏完整解释生成。