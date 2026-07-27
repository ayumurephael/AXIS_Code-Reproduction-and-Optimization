# AXIS Prompt 阶段 A 消融实验结果与分析

## 结论摘要

本报告在作者发布 checkpoint、固定 `paper140` 清单和同一云端运行环境下比较 1 个 Baseline 与 7 个 prompt-only 条件。按三类题型 Final 的未加权宏平均，最佳条件为 `base`（Baseline），宏平均 3.6613，相对 Baseline 的绝对变化为 +0.0000。

2026-07-26 又按相同论文口径完成了 2 个确认性 prompt 实验和 1 个修订版 Evidence Contract 实验；其完整结果、相对 Baseline 的变化和运行审计见文末对应章节。

各单项指标的最高值及其相对 Baseline 的绝对变化如下：

- MC Final: Fixed 角色重命名，Δ=+0.0679
- MC Corr.: Fixed 角色重命名，Δ=+0.0471
- MC Rsn.: Fixed 角色重命名，Δ=+0.1163
- OE Final: 角色重命名 + Evidence Contract，Δ=+0.1578
- OE Acc.: 题型协议，Δ=+0.3453
- OE Comp.: 角色重命名 + Evidence Contract，Δ=+0.0264
- OE Rel.: 角色重命名 + Evidence Contract，Δ=+0.1772
- TF Final: Baseline，Δ=+0.0000
- TF Corr.: Baseline，Δ=+0.0000
- TF Justif.: Fixed 角色重命名，Δ=+0.0714

## 关键判断

- **没有任何 prompt-only 条件超过 Baseline 的总体分数。** Baseline 的三题型未加权 Macro Final 为 3.6613；最接近的是 Fixed 角色重命名 3.6604（Δ=-0.0009），其 140 条 QA 配对差异区间为 [-0.1391, +0.1442]，不能据此声称总体提升。
- **`EOS + Answer:` 是强格式控制，不是内容增益。** MC/TF strict-parse 从 Baseline 的 13.95%/9.52% 升至 97.67%/100.00%，但 Macro Final 下降 -0.0626。
- **Fixed hint 对旧 checkpoint 是必要条件。** 在相同答案边界下移除 Fixed 后 Macro Final 从 3.5986 降至 1.8813；平均输出从 646.5 增至 2961.2 字符，且 55.00% 回答不少于 3000 字符。该条件应视为明确的负面对照。
- **Evidence Contract 呈题型迁移而非全局提升。** 相对 Fixed 角色重命名，它使 OE Final 改变 +0.1721，但 MC/TF Final 分别改变 -0.3586/-0.3000；不能把 OE 收益外推为通用收益。
- **组合存在明显交互。** 在综合 2+3+4 上加入边界对齐会恢复 MC Final（Δ=+0.4682）并略升 TF Final（Δ=+0.0333），同时降低 OE Final（Δ=-0.2252）。因此不应按单因素结果做简单加和预测。

## Table-I 指标汇总

所有分数均由唯一 Judge `deepseek-v4-pro` 按论文 G-Eval 量表给出。

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 4.2558 | 4.3256 | 4.0930 | 3.0859 | 2.8365 | 2.7700 | 3.7455 | 3.6421 | 3.6417 | 3.6429 |
| EOS + Answer: | 4.1372 | 4.1860 | 4.0232 | 3.0420 | 2.8909 | 2.7091 | 3.6066 | 3.6168 | 3.5929 | 3.6526 |
| EOS + Answer: / w/o Fixed | 2.2605 | 2.3721 | 2.0000 | 1.7434 | 1.5219 | 1.5717 | 2.2022 | 1.6400 | 1.6905 | 1.5643 |
| 题型协议 | 3.9581 | 3.9302 | 4.0233 | 3.0038 | 3.1818 | 2.5328 | 3.3455 | 3.3428 | 3.3333 | 3.3570 |
| Fixed 角色重命名 | 4.3237 | 4.3727 | 4.2093 | 3.0716 | 2.9273 | 2.7818 | 3.5782 | 3.5857 | 3.5000 | 3.7143 |
| 角色重命名 + Evidence Contract | 3.9651 | 4.0000 | 3.8837 | **3.2437** | **3.1091** | **2.7964** | **3.9227** | 3.2857 | 3.1429 | 3.5000 |
| 综合 2+3+4 | 3.6322 | 3.6739 | 3.5349 | 3.1173 | 3.1272 | 2.6734 | 3.6236 | 3.5428 | 3.5238 | 3.5714 |
| EOS + Answer: + 综合 2+3+4 | 4.1004 | 4.1733 | 3.9302 | 2.8921 | 2.7134 | 2.5419 | 3.5091 | 3.5762 | 3.5476 | 3.6190 |

## 相对 Baseline 的绝对差值

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| EOS + Answer: | -0.1186 | -0.1395 | -0.0698 | -0.0439 | +0.0544 | -0.0609 | -0.1388 | -0.0254 | -0.0488 | +0.0097 |
| EOS + Answer: / w/o Fixed | -1.9954 | -1.9535 | -2.0930 | -1.3425 | -1.3146 | -1.1983 | -1.5432 | -2.0021 | -1.9512 | -2.0786 |
| 题型协议 | -0.2977 | -0.3953 | -0.0698 | -0.0822 | +0.3453 | -0.2372 | -0.4000 | -0.2994 | -0.3083 | -0.2859 |
| **Fixed 角色重命名** | +0.0679 | +0.0471 | +0.1163 | -0.0143 | +0.0908 | +0.0118 | -0.1673 | -0.0564 | -0.1417 | +0.0714 |
| **角色重命名 + Evidence Contract** | -0.2907 | -0.3256 | -0.2093 | **+0.1578** | **+0.2726** | **+0.0264** | **+0.1772** | -0.3564 | -0.4988 | -0.1429 |
| 综合 2+3+4 | -0.6236 | -0.6517 | -0.5581 | +0.0314 | +0.2907 | -0.0966 | -0.1218 | -0.0993 | -0.1179 | -0.0714 |
| EOS + Answer: + 综合 2+3+4 | -0.1554 | -0.1523 | -0.1628 | -0.1939 | -0.1231 | -0.2282 | -0.2364 | -0.0660 | -0.0940 | -0.0238 |

## 相对 Baseline 的相对变化

相对变化定义为 `(variant - baseline) / baseline × 100%`。

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| EOS + Answer: | -2.79% | -3.23% | -1.70% | -1.42% | +1.92% | -2.20% | -3.71% | -0.70% | -1.34% | +0.27% |
| EOS + Answer: / w/o Fixed | -46.89% | -45.16% | -51.14% | -43.50% | -46.34% | -43.26% | -41.20% | -54.97% | -53.58% | -57.06% |
| 题型协议 | -6.99% | -9.14% | -1.70% | -2.66% | +12.17% | -8.56% | -10.68% | -8.22% | -8.47% | -7.85% |
| Fixed 角色重命名 | +1.59% | +1.09% | +2.84% | -0.46% | +3.20% | +0.43% | -4.47% | -1.55% | -3.89% | +1.96% |
| 角色重命名 + Evidence Contract | -6.83% | -7.53% | -5.11% | +5.11% | +9.61% | +0.95% | +4.73% | -9.79% | -13.70% | -3.92% |
| 综合 2+3+4 | -14.65% | -15.07% | -13.63% | +1.02% | +10.25% | -3.49% | -3.25% | -2.73% | -3.24% | -1.96% |
| EOS + Answer: + 综合 2+3+4 | -3.65% | -3.52% | -3.98% | -6.28% | -4.34% | -8.24% | -6.31% | -1.81% | -2.58% | -0.65% |

## 实验因素矩阵

| 实验 | EOS + Answer: | 移除 Fixed | 题型协议 | 角色重命名 | Evidence Contract |
|---|---|---|---|---|---|
| Baseline | — | — | — | — | — |
| EOS + Answer: | ✓ | — | — | — | — |
| EOS + Answer: / w/o Fixed | ✓ | ✓ | — | — | — |
| 题型协议 | — | — | ✓ | — | — |
| Fixed 角色重命名 | — | — | — | ✓ | — |
| 角色重命名 + Evidence Contract | — | — | — | ✓ | ✓ |
| 综合 2+3+4 | — | — | ✓ | ✓ | ✓ |
| EOS + Answer: + 综合 2+3+4 | ✓ | — | ✓ | ✓ | ✓ |

## 输出行为与中间诊断

| 实验 | MC answer-first | TF answer-first | MC strict parse | TF strict parse | OE direct-start | 无 think 标签 | 平均字符数 | ≥3000 字符 |
|---|---|---|---|---|---|---|---|---|
| Baseline | 13.95% | 9.52% | 13.95% | 9.52% | 10.91% | 69.29% | 703.9 | 0.71% |
| EOS + Answer: | 97.67% | 100.00% | 97.67% | 100.00% | 100.00% | 100.00% | 646.5 | 0.00% |
| EOS + Answer: / w/o Fixed | 81.40% | 0.00% | 81.40% | 0.00% | 40.00% | 34.29% | 2961.2 | 55.00% |
| 题型协议 | 27.91% | 38.10% | 27.91% | 38.10% | 30.91% | 72.86% | 848.8 | 3.57% |
| Fixed 角色重命名 | 4.65% | 7.14% | 4.65% | 7.14% | 5.45% | 67.86% | 694.1 | 0.00% |
| 角色重命名 + Evidence Contract | 6.98% | 0.00% | 6.98% | 0.00% | 1.82% | 58.57% | 796.5 | 0.00% |
| 综合 2+3+4 | 46.51% | 45.24% | 46.51% | 45.24% | 12.73% | 59.29% | 888.9 | 2.86% |
| EOS + Answer: + 综合 2+3+4 | 97.67% | 100.00% | 97.67% | 100.00% | 100.00% | 100.00% | 596.6 | 0.00% |

该表的格式指标由确定性规则计算，不替代 G-Eval。`answer-first` 只检查原始生成是否直接以题型规定答案起始；`strict parse` 对 MC 要求首行以 `A)`–`D)` 起始，对 TF 要求首 token 为 `True.` 或 `False.`。`OE direct-start` 还排除选项字母、True/False、`Answer:` 与 `<think>` 起始。

## 配对差异

下列区间把 140 条 QA 的题型 Final 差值（variant − Baseline）合并后执行 10,000 次配对 bootstrap（seed=72）。该均值按 QA 数量加权，与上文三题型等权的 `Macro Final` 不是同一统计量：

- EOS + Answer:: -0.0613 [-0.1911, +0.0680]
- EOS + Answer: / w/o Fixed: -1.7409 [-1.9994, -1.4811]
- 题型协议: -0.2135 [-0.4140, -0.0118]
- Fixed 角色重命名: -0.0017 [-0.1391, +0.1442]
- 角色重命名 + Evidence Contract: -0.1342 [-0.3163, +0.0479]
- 综合 2+3+4: -0.2090 [-0.4193, -0.0003]
- EOS + Answer: + 综合 2+3+4: -0.1437 [-0.3201, +0.0324]

## 预注册机制对比

下表按实验设计中可直接解释的 Treatment − Control 计算。`Evidence Contract` 没有脱离角色重命名单独运行，因此其增量只能解释为“在已重命名条件上加入 Contract”，不能声称是完全独立主效应。

| 预注册对比 | Treatment | Control | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 边界对齐 | EOS + Answer: | Baseline | -0.1186 | -0.1395 | -0.0698 | -0.0439 | +0.0544 | -0.0609 | -0.1388 | -0.0254 | -0.0488 | +0.0097 | -0.0626 |
| 边界对齐下移除 Fixed | EOS + Answer: / w/o Fixed | EOS + Answer: | -1.8767 | -1.8140 | -2.0232 | -1.2985 | -1.3690 | -1.1374 | -1.4044 | -1.9768 | -1.9024 | -2.0883 | -1.7173 |
| 仅题型协议 | 题型协议 | Baseline | -0.2977 | -0.3953 | -0.0698 | -0.0822 | +0.3453 | -0.2372 | -0.4000 | -0.2994 | -0.3083 | -0.2859 | -0.2264 |
| 仅 Fixed 角色重命名 | Fixed 角色重命名 | Baseline | +0.0679 | +0.0471 | +0.1163 | -0.0143 | +0.0908 | +0.0118 | -0.1673 | -0.0564 | -0.1417 | +0.0714 | -0.0009 |
| 在角色重命名上增加 Evidence Contract | 角色重命名 + Evidence Contract | Fixed 角色重命名 | -0.3586 | -0.3727 | -0.3256 | +0.1721 | +0.1818 | +0.0146 | +0.3445 | -0.3000 | -0.3571 | -0.2143 | -0.1622 |
| 综合 2+3+4 | 综合 2+3+4 | Baseline | -0.6236 | -0.6517 | -0.5581 | +0.0314 | +0.2907 | -0.0966 | -0.1218 | -0.0993 | -0.1179 | -0.0714 | -0.2305 |
| 在综合 2+3+4 上增加边界对齐 | EOS + Answer: + 综合 2+3+4 | 综合 2+3+4 | +0.4682 | +0.4994 | +0.3953 | -0.2252 | -0.4138 | -0.1316 | -0.1145 | +0.0333 | +0.0238 | +0.0476 | +0.0921 |

## 面向下一阶段的改进建议

1. **把 Fixed 角色重命名作为唯一的通用确认候选，Baseline 作为主控制。** 它的总体分数与 Baseline 实质持平，同时改善 MC 三项、OE Accuracy/Completeness 和 TF Justification；下一阶段应在完整 284 条测试集和重新训练后确认，而不是宣称本阶段已提升。
2. **将 `EOS + Answer:` 定位为可选的部署格式层。** 它把 MC/TF 首行可解析率推近 100%，但未提升总体内容分；若下游必须稳定解析，可启用该边界，否则不应把它列为质量优化。
3. **保留 Fixed 或设计有监督替代，停止直接移除。** `w/o Fixed` 在所有题型上显著崩溃并产生超长回答。若要减少固定提示依赖，应在训练阶段逐步 dropout/蒸馏，而不是只在推理时删除。
4. **把格式收益与证据收益拆开。** 下一轮同时报告程序化 MC/TF 正确率、首行可解析率、答案长度，以及 G-Eval 内容维度；若 Final 提升主要随 parse 改善而非 Accuracy/Justification 改善，应将结论限定为输出控制收益。
5. **修正 Evidence Contract 与布局矛盾，并做题型专用版本。** Stage A 按规范保留了“same row”文字和两块式 Values/Local 布局。Phase II 应预注册两条互斥路线：将文字改为“corresponds by order”，或真正采用逐步交错序列化；两者不能在同一条件中同时改变。鉴于当前收益集中在 OE，还应比较 OE-only Contract 与全题型 Contract。
6. **扩展稳健性验证。** 在 284 条完整测试集上复核方向，并对多个训练 seed/checkpoint 重复最佳条件。单 Judge 结果还应在不用于选择方案的前提下，增加独立 Judge 或人工盲评作为确认性分析。

## 完整实验配置与可审计数据

- 实验代码提交：`bb1410fa8568769affbe99ac9dd4b3d25fef82b5`
- 代码基点：`origin/main@e8c1aee59bb98cda9b37445bf2eb23619f374d54`
- Baseline 预测先使用同 checkpoint、同 paper140、同 3-rank series batching、同 torch/CUDA 协议的历史锁定产物；其 SHA-256 为 `fc687e2ad4c24e66ef00fc4a381885df90c18e26d4edcc6eb230ce94052fa71e`。本次同运行 base 必须 140/140 逐条完全一致，否则替换并重评。
- checkpoint SHA-256：`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`
- checkpoint 元数据：epoch 33，保存时训练平均 loss `0.7127881973981858`，文件大小 `482319394` bytes
- 原训练集：30000 条 series JSON、每条 2 个窗口 QA，共 60000 QA；题型分布为 TF=20204、MC=19885、OE=19911；本阶段不重新训练，该信息仅用于 checkpoint 数据血缘
- 原训练集没有官方 train/validation/test 划分；任何后续重训应按`sample_id` 做样本级切分，避免同一 series 的两个窗口跨集合泄漏
- prompt 规范 SHA-256：`e1752b96b1c31a979e8788398b319485b6816d7f2da1ffd625c8fcc407895f06`
- local hint 与数值序列保持论文/作者代码的原始两块式 prompt 布局，没有在本阶段改成交错布局
- tokenizer 边界审计：LlamaTokenizerFast；EOS token id `151643`；字面量 `Answer:` 为 3 tokens；Baseline 前缀 141 tokens，`EOS + Answer:` 前缀 145 tokens；移除 Fixed 后固定占位符 30→0
- 子集：`paper140`
- batching：`series`
- skip loss：`True`
- world size：`3`
- QA 数：140
- series 数：70
- 题型数：`{'open_ended': 55, 'multiple_choice': 43, 'true_false': 42}`
- paper140 异常标签分布：`has_anomaly=false` 94，`has_anomaly=true` 46；完整候选测试集为 284 条 QA
- 论文 Final 聚合权重：MC=`0.7×Correctness + 0.3×Reasoning`；OE=`0.35×Accuracy + 0.35×Completeness + 0.3×Relevance`；TF=`0.6×Correctness + 0.4×Justification`
- 各模式预测数：`{'answer_boundary': 140, 'answer_boundary_combined_234': 140, 'answer_boundary_wo_fixed': 140, 'base': 140, 'combined_234': 140, 'fixed_role': 140, 'fixed_role_evidence_contract': 140, 'task_protocol': 140}`
- 生成：`do_sample=False`, `num_beams=5`, `max_new_tokens=1000`, `repetition_penalty=1.15`, `no_repeat_ngram_size=3`, `length_penalty=1`
- Python：`3.10.5`
- PyTorch：`2.5.1+cu121`
- PyTorch CUDA：`12.1`
- Transformers：`4.45.2`
- GPU：3×NVIDIA A100 40GB，`CUDA_VISIBLE_DEVICES=0,1,2`，每个分布式 rank 固定映射一张卡；启动时三卡空闲并已预留，运行期间后来出现其他用户共享进程，导致吞吐波动，但本任务未终止或修改任何外部进程
- 推理运行时警告：Transformers 重复报告 `Setting pad_token_id to eos_token_id:None`；未出现 OOM、空输出或缺 shard，最终合并 1120/1120
- Judge：仅 DeepSeek；请求模型 `deepseek-v4-pro`；thinking enabled；reasoning effort high；max tokens 4096；请求 top-20 logprobs；缺失完整 1–5 分布时执行 20 次精确分数回退
- TLS：保持证书验证开启，并显式使用 GPU 节点的系统 CA bundle；批量前真实 G-Eval 探活必须同时满足模型 ID、非空正文、可解析分数与完整 1–5 score-token 分布。
- Judge 行数：2680
- Judge 返回模型：`{'deepseek-v4-pro': 2680}`
- Judge provider：`{'deepseek': 2680}`
- Judge 评分方法：`{'final_score_top_logprobs': 2658, 'exact_sample_mean_20': 22}`
- Judge endpoint host：`{'api.deepseek.com': 2680}`
- Judge system fingerprint：`{'fp_9954b31ca7_prod0820_fp8_kvcache_20260402': 2680}`
- 唯一 Judge prompt 哈希数：2662
- Judge token 用量汇总：`{'prompt_tokens': 1849037, 'completion_tokens': 4007460, 'total_tokens': 5856497, 'prompt_cache_hit_tokens': 672512, 'prompt_cache_miss_tokens': 1176525}`
- predictions SHA-256：`32a29dcc5e592d2dbbb83cf711f3e06b376e79dd296380b67494c49c5938e005`
- scores SHA-256：`703ce79b6dec082986789007ef0207c87e8b0d09a466cbbe35ee82a55ba28e80`
- run manifest SHA-256：`6e567c154ad0b0bbd34a75d91608d4b9b5d89d69b32b62b8779de722aa360e72`
- diagnostics SHA-256：`d1c99e46ada1d7351b02a6762f7e2165adba4992de652c00b3a28363aca945b6`
- 完整性审计：`ok=True`，`{'prediction_rows': 1120, 'records': 140, 'prediction_modes': {'answer_boundary': 140, 'answer_boundary_combined_234': 140, 'answer_boundary_wo_fixed': 140, 'base': 140, 'combined_234': 140, 'fixed_role': 140, 'fixed_role_evidence_contract': 140, 'task_protocol': 140}, 'score_rows': 2680, 'score_methods': {'final_score_top_logprobs': 2658, 'exact_sample_mean_20': 22}, 'score_models': {'deepseek-v4-pro': 2680}, 'score_providers': {'deepseek': 2680}, 'score_enable_thinking': {'auto': 2680}, 'logprobs_returned': 2658, 'valid_prompt_hashes': 2680, 'errors': [], 'ok': True}`

## 解释边界与局限性

- 阶段 A 只改变推理 prompt，旧 checkpoint 未针对新文字或新边界重新训练；因此结果只能回答“旧权重能否即时受益”，不能替代 Phase II 重新训练后的结论。
- Evidence Contract 按控制文档逐字保留“same row”，但本阶段又按要求保留原始 Values/Local 两块式排布。该文字与实际布局不完全一致，是实验条件的一部分，也是解释结果时必须披露的混杂因素。
- 题型协议同时改变格式、长度与内容约束。Final 提升若伴随 parse/answer-first 改善，不能全部归因于时间序列证据利用增强。
- 正式集为论文口径的 140 条 QA，而非 284 条全测试集；未执行多 seed 模型训练或 checkpoint 重复。
- 按用户要求只使用一个 Judge。虽然保留分布、回退样本、system fingerprint 与 prompt 哈希，评分仍可能包含单 Judge 偏差和托管 API 的服务端非确定性。

## 文件索引

- 推理结果：`predictions.jsonl`
- G-Eval 明细：`all_scores.jsonl`
- 推理 manifest：`run_manifest.json`
- 输出诊断：`output_diagnostics.json`
- 运行环境：`runtime_environment.json`
- GPU 节点固定清单审计：`all_audit_manifest.json`
- 聚合指标：`all_table.json` / `all_table.md`
- 回退条目：`all_fallback_pending.jsonl`
- 推理日志：`formal_inference.log`
- Judge 终轮日志：`all_judge.log`

## 追加确认实验（2026-07-26）

### 实验定义

两项实验继续使用作者发布 checkpoint，不训练或更新任何模型参数，也不增加 `EOS + Answer:` 或题型输出协议。

- **实验 1：专家开头 + Contract(1–3) + Fixed 独立节。** 将原开头替换为 `You are an expert time-series anomaly analyst. Produce one precise, evidence-grounded answer to the question.`；加入只含前 3 条的 Evidence Contract；保留原 `Time Series Data` 和 `Contextual Hints / Per-Step Analysis`；30 个 Fixed token 单独放在 `### Learned Task Guidance/Shared Task-Control Tokens` 下。
- **实验 2：实验 1 + 逐步交错证据。** 用 `Time-Series Evidence` 替换原 `Time Series Data` 与 `Contextual Hints` 两块。只输入格式说明和真实的 `{aligned_rows}`，不输入四行示意模板。每行按 `Step {step:04d} | value={value:+06d} | context=<|local_hint|>` 序列化；`value` 先采用原代码 `(x * 100):.0f` 舍入，再转为整数。30 个 Fixed token 仍保留在独立节中。

Evidence Contract 第 2 条使用本轮锁定文字：“Each Per-Step Context token is sample-specific and corresponds to the value on the same row. Use it to determine whether that local behavior is unexpected relative to the full temporal pattern.” 因此实验 1 同时保留了原两块式布局和 “same row” 描述；该不完全一致是实验条件的一部分。实验 2 才使文字与逐行布局一致。

### 正式评分结果

所有新分数均由唯一 Judge `deepseek-v4-pro` 按与 Stage A 相同的 G-Eval 量表产生。Baseline 使用同一批已锁定评分，不重新调用 Judge。

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 4.2558 | 4.3256 | 4.0930 | 3.0859 | 2.8365 | 2.7700 | 3.7455 | 3.6421 | 3.6417 | 3.6429 | 3.6613 |
| 实验 1：Contract(1–3) + Fixed 独立节 | 4.0349 | 4.0698 | 3.9535 | 2.8772 | 2.5946 | 2.7273 | 3.3818 | 3.4619 | 3.5001 | 3.4047 | 3.4580 |
| 实验 2：实验 1 + 逐步交错证据 | 4.0327 | 4.1628 | 3.7291 | 2.6685 | 2.1818 | 2.5964 | 3.3206 | 3.2862 | 3.2548 | 3.3333 | 3.3291 |

Macro Final 是 MC/OE/TF 三个 Final 的未加权平均；不是论文表中的独立 Judge 维度。

### 相对 Baseline 的绝对差值

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 实验 1 | -0.2209 | -0.2558 | -0.1395 | -0.2087 | -0.2419 | -0.0427 | -0.3636 | -0.1802 | -0.1416 | -0.2382 | -0.2033 |
| 实验 2 | -0.2231 | -0.1628 | -0.3640 | -0.4174 | -0.6547 | -0.1737 | -0.4249 | -0.3559 | -0.3869 | -0.3095 | -0.3322 |

### 相对 Baseline 的相对变化

相对变化定义为 `(experiment - baseline) / baseline × 100%`。

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 实验 1 | -5.19% | -5.91% | -3.41% | -6.76% | -8.53% | -1.54% | -9.71% | -4.95% | -3.89% | -6.54% | -5.55% |
| 实验 2 | -5.24% | -3.76% | -8.89% | -13.53% | -23.08% | -6.27% | -11.34% | -9.77% | -10.62% | -8.50% | -9.07% |

### 实验 2 相对实验 1 的增量

该对比隔离“把原两块式 Values/Local 布局改为真实逐步交错布局”的联合增量；开头、Contract(1–3)、Fixed 数量和 Fixed 独立节均保持不变。

| MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -0.0022 | +0.0930 | -0.2244 | -0.2087 | -0.4128 | -0.1310 | -0.0613 | -0.1757 | -0.2453 | -0.0714 | -0.1289 |

逐步交错几乎没有改变 MC Final，但其内部发生方向相反的迁移：Correctness +0.0930、Reasoning -0.2244。OE 和 TF 的所有正式维度均下降。因此，在旧 checkpoint 上，本轮没有观察到布局对齐带来的总体收益。

### 配对差异与统计解释

下列区间使用与 Stage A 相同的方法：对 140 条 QA 的题型 Final 差值执行 10,000 次配对 bootstrap（seed=72）。它按 QA 数量加权，与三题型等权的 Macro Final 不同。

- 实验 1 − Baseline：-0.2039，95% CI [-0.3720, -0.0361]
- 实验 2 − Baseline：-0.3393，95% CI [-0.5042, -0.1707]
- 实验 2 − 实验 1：-0.1354，95% CI [-0.3205, +0.0492]

前两个区间均完全低于 0：在本次 `paper140`、单 Judge 和旧 checkpoint 口径下，两项实验相对 Baseline 的下降不能解释为仅有方向不明的抽样波动。实验 2 相对实验 1 的区间跨 0，因此不能仅凭本次样本断言交错布局的额外下降可稳定复现。

### 输出行为与中间诊断

| 实验 | MC answer-first | TF answer-first | MC strict parse | TF strict parse | OE direct-start | 无 think 标签 | 平均字符数 | ≥3000 字符 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 实验 1 | 4.65% | 2.38% | 4.65% | 2.38% | 25.45% | 66.43% | 819.8 | 0.00% |
| 实验 2 | 4.65% | 2.38% | 4.65% | 2.38% | 41.82% | 71.43% | 837.2 | 0.71% |

这些是原始输出起始格式的严格诊断，不是 G-Eval 分数，也不是“一般可解析率”。本轮有意不加入 `EOS + Answer:`，所以模型仍可能先输出推理、`<think>` 或其他正文；低 answer-first/strict-parse 与该生成边界一致。作为交叉检查，在能够严格匹配 MC 首行的少量样本中，选项正文精确匹配率均为 100%；宽松启发式 MC 正确率为实验 1 的 88.37% 和实验 2 的 95.35%。这说明低 strict-parse 不是由选项映射代码整体失效造成的，但宽松启发式正确率也不能替代正式 G-Eval。

实验 2 的严格 OE direct-start 和“无 think 标签”比例上升，却没有转化为 OE 内容分数提升。启发式 unsupported-number fraction 的整体均值也几乎不变（42.44%→42.89%）；其中 MC 从 31.63% 降至 20.75%，OE 却从 54.63% 升至 69.61%。该指标只检查回答数字是否能在问题、窗口或边界中找到，不判断数字语义，因而只能用于定位输出行为，不能单独解释质量变化。

### 关键结论

1. **两项确认实验都没有超过 Baseline。** 实验 1 的三个题型 Final 全部下降，Macro Final -5.55%；实验 2 的 Macro Final -9.07%。
2. **把 Contract 缩短到前 3 条并修正开头、Fixed 节位置，仍未使旧 checkpoint 即时受益。** 实验 1 的 OE Completeness 降幅最小（-0.0427），但其余正式维度均有更明显下降。
3. **逐步交错布局只改善了 MC Correctness，未改善综合质量。** 相对实验 1，MC Correctness +0.0930，但 MC Reasoning -0.2244，OE Final -0.2087，TF Final -0.1757。
4. **结果更支持“训练—推理 prompt 分布偏移”解释，而不是布局设计本身无效。** 作者 checkpoint 是用原 prompt 训练的；本轮只改推理 prompt。要检验新布局的真实上限，需要在相同 prompt 上重新训练或至少进行受控微调，不能把旧权重的即时退化外推为架构性结论。
5. **格式改善与内容质量仍须分开。** 实验 2 的严格 OE direct-start 提升 16.36 个百分点，但 OE Final 反而下降 0.2087，说明直接起答不是内容增益的充分条件。

### 完整配置、运行过程与审计

- 运行时代码基于 `prompt@4d29da2d3e645861d67cc98a5eac96f94af68803` 的未提交 follow-up 修改部署；四个部署文件与本地逐文件 SHA-256 完全一致。prompt 规范 SHA-256 为 `966a0c2a78811cadf5630d718ebfb7d79e7b38699164bbf0bf6963687a0b0387`。具体文件哈希见 `experiment_manifest.json`。
- checkpoint SHA-256：`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`；未执行训练、反向传播或参数更新。
- 固定清单：`paper140`，SHA-256 `c2aefed015869d153ab3f5632442f83ac3ffccf140685229ffab7a3a8e9cc1a6`；140 条 QA、70 个 series；MC=43、OE=55、TF=42。
- 推理：3×NVIDIA A100 40GB，`world_size=3`，`series` batching，`skip_loss=True`；每个模式 140 条，共 280/280 条。生成参数为 `do_sample=False`、`num_beams=5`、`max_new_tokens=1000`、`repetition_penalty=1.15`、`no_repeat_ngram_size=3`、`length_penalty=1`。
- 运行环境：Python 3.10.5、PyTorch 2.5.1+cu121、CUDA 12.1、Transformers 4.45.2、NumPy 1.26.4。
- 推理节点上只有本地 checkpoint、测试数据和可用的模型目录。第一次启动尝试在线解析模型名，得到 0 条预测后立即停止，未进入正式统计；改为已核验本地模型并设置离线模式后完成 280 条正式预测。正式运行无 OOM、无空 shard、无 traceback；Transformers 重复给出 `pad_token_id` 到 EOS 的既有警告。
- 资产所在 A100 节点启动时 GPU 0–2 已有其他用户进程并显示高利用率，但各有约 34GB 空闲显存；本任务只使用空闲显存，没有终止、修改或接管任何外部进程。该共享状态影响吞吐稳定性，不改变输入、权重或聚合公式。
- Judge 仅使用 DeepSeek；请求与返回模型均为 `deepseek-v4-pro`，`enable_thinking=auto`，`max_tokens=4096`，primary workers=8，fallback workers=1，fallback samples=20，最大重试轮次=12。
- 推理节点的第一次 Judge 连接预运行因出站网络错误产生 0 条评分，未纳入统计。随后在已验证网络连通的 GPU 节点上正式完成 670/670 条：`final_score_top_logprobs` 661 条，`exact_sample_mean_20` 9 条；精确日志审计未发现鉴权错误、永久错误或 traceback。
- Judge 返回 provider 为 `deepseek` 670/670，endpoint host 为 `api.deepseek.com` 670/670，system fingerprint 为 `fp_9954b31ca7_prod0820_fp8_kvcache_20260402` 670/670；有效且唯一的 Judge prompt 哈希均为 670。
- Judge token 用量：prompt 433,278，completion 1,061,500，total 1,494,778；cache hit 176,640，cache miss 256,638。
- Baseline 使用 Stage A 已锁定的 335 条分数，SHA-256 `c288f3ed3c162f4df10f9f0b38b50ac1d2e9e6831c54e5a93770e49d3bbebcc0`；只在聚合时合并，未重新请求 Judge。
- 预测审计：280 条、两个模式各 140 条、错误为空、`ok=True`。Judge 审计：140 个 record、670 条评分、661 条 logprobs、9 条 20 次精确采样回退、670 个有效 prompt 哈希、错误为空、`ok=True`。
- 本地测试：`python -m pytest tools/axis_repro/test_prompt_stage_a.py -q` 为 12 passed + 2 subtests；`python -m pytest tools/axis_repro -q` 为 53 passed + 2 subtests；`git diff --check` 通过。

### 新实验产物与哈希

新实验目录为 `../prompt_followup_deepseek_v4_pro_20260726/`。大体积原始预测、评分和日志保留在本地并由该目录 `.gitignore` 排除提交；小型审计与汇总文件可随代码版本化。

- `experiment_manifest.json`：去除机器路径后的完整运行配置、源文件哈希与产物哈希。
- `table_followup.json` / `table_followup.md`：Baseline 与两项实验的聚合结果及 Baseline 配对区间。
- `prediction_audit.json`：推理覆盖率审计。
- `audit.json`：Judge 覆盖率、模型、provider、方法和 prompt 哈希审计。
- `output_diagnostics.json`：确定性输出行为汇总。
- `final_metadata.json`：Judge 模型、端点、fingerprint、方法计数和核心产物哈希。
- `paper140.json`：本轮固定清单。
- predictions SHA-256：`6b06ae4acfb3c09186d3a6ca74ee3e3d1020795687f4c8cd1a51ab5a897bb449`
- scores SHA-256：`1e706c4c3b1f28c9b6cba1ef9a82da4b00c84a33592ccb3ff98844bd25e63bf4`
- run manifest SHA-256：`9bceca4651a132d44d909235e3ae3e3c06d3340a522976a8092b3a90945282c3`
- Judge audit SHA-256：`0faa456c2be2f98ebbbdd6d557c7c6e37752d70561d447a0ff5931297519145b`
- diagnostics SHA-256：`8d359177d25f0621bfe0bda0dd5b7c9b9eb7249a672fb2c3e5c12fb3fce66529`

### 解释边界

- 本轮结论只回答“作者旧 checkpoint 在两个新推理 prompt 下的即时表现”，不回答用新 prompt 重新训练后能否提升。
- 正式集仍为论文口径的 140 条 QA，不是完整 284 条候选测试数据；没有模型 seed 或 checkpoint 重复。
- 按用户要求只使用一个 Judge。配对区间刻画 QA 抽样不确定性，不包含托管 Judge 的模型间差异或服务端非确定性。
- 实验 1 同时改变开头、Contract 内容和 Fixed 节位置，不是单因素消融；实验 2 相对实验 1 才是对交错布局增量的受控比较。

## 修订版 Evidence Contract 补充实验（2026-07-26）

### 实验定义

本实验继续使用作者发布 checkpoint，只改变推理 prompt，不执行训练、反向传播或参数更新。相对 `Fixed 角色重命名 + 旧完整 Evidence Contract`，开头、原始两块式 `Time Series Data` / `Contextual Hints` 布局、30 个 Fixed token、作者原 generation boundary 均保持不变；只替换 Evidence Contract：

1. 明确 Window Values 是按原代码舍入得到的整数编码；只有问题显式要求近似未缩放数值时才除以 100 一次。
2. 明确 Local token 与 Window Values 按序列位置一一对应，而不是按文本行对应。
3. 保留 Shared Task-Control token 的角色边界。
4. 保留基于全局时序模式判断异常的定义。

本条件不增加 `EOS + Answer:`，不增加题型协议，也不采用逐步交错布局。完整逐字 prompt 见本文最末附录的模板 I。

### 正式评分结果

所有分数均由唯一 Judge `deepseek-v4-pro` 按与 Stage A 相同的 G-Eval 量表产生；Baseline 和既有实验沿用各自已锁定分数，不重新调用 Judge。

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 4.2558 | 4.3256 | 4.0930 | 3.0859 | 2.8365 | 2.7700 | 3.7455 | 3.6421 | 3.6417 | 3.6429 | 3.6613 |
| Fixed 角色重命名 | 4.3237 | 4.3727 | 4.2093 | 3.0716 | 2.9273 | 2.7818 | 3.5782 | 3.5857 | 3.5000 | 3.7143 | 3.6604 |
| Fixed 角色重命名 + 旧完整 Contract | 3.9651 | 4.0000 | 3.8837 | 3.2437 | 3.1091 | 2.7964 | 3.9227 | 3.2857 | 3.1429 | 3.5000 | 3.4982 |
| 追加实验 1：Contract(1–3) + Fixed 独立节 | 4.0349 | 4.0698 | 3.9535 | 2.8772 | 2.5946 | 2.7273 | 3.3818 | 3.4619 | 3.5001 | 3.4047 | 3.4580 |
| 追加实验 2：实验 1 + 逐步交错证据 | 4.0327 | 4.1628 | 3.7291 | 2.6685 | 2.1818 | 2.5964 | 3.3206 | 3.2862 | 3.2548 | 3.3333 | 3.3291 |
| **Fixed 角色重命名 + 修订版 Contract** | **4.0186** | **4.0465** | **3.9535** | **2.9047** | **2.7277** | **2.6727** | **3.3818** | **3.5333** | **3.5714** | **3.4762** | **3.4856** |

Macro Final 是 MC/OE/TF 三个 Final 的未加权平均，不是论文表中的独立 Judge 维度。

### 相对 Baseline 的绝对差值与相对变化

相对变化定义为 `(新版 Contract - Baseline) / Baseline × 100%`。

| 变化 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 绝对差值 | -0.2372 | -0.2791 | -0.1395 | -0.1812 | -0.1088 | -0.0973 | -0.3636 | -0.1088 | -0.0702 | -0.1667 | -0.1757 |
| 相对变化 | -5.57% | -6.45% | -3.41% | -5.87% | -3.83% | -3.51% | -9.71% | -2.99% | -1.93% | -4.58% | -4.80% |

三个题型的所有正式维度均低于 Baseline。新版 Contract 的三题型等权 Macro Final 为 3.4856，相对 Baseline 下降 0.1757（-4.80%）。

### 相对旧完整 Contract 的受控增量

该对比保持角色名、Fixed token、两块式证据布局和生成边界不变，隔离“重写第 1、2 条，移除旧第 4 条并保留异常定义”的联合增量。

| 变化 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | Macro Final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 绝对差值 | +0.0535 | +0.0465 | +0.0698 | -0.3390 | -0.3814 | -0.1237 | -0.5408 | +0.2476 | +0.4286 | -0.0238 | -0.0126 |
| 相对变化 | +1.35% | +1.16% | +1.80% | -10.45% | -12.27% | -4.42% | -13.79% | +7.54% | +13.64% | -0.68% | -0.36% |

修订达到了预期的局部方向：相对旧 Contract，MC 三个维度全部恢复，TF Final 和 TF Correctness 分别恢复 0.2476 和 0.4286；但 OE 三个维度全部下降，尤其 OE Relevance 下降 0.5408。结果是 Macro Final 仍微降 0.0126，不能称为总体改进。

作为另一条必要控制，新版 Contract 相对只做 Fixed 角色重命名的条件，MC/OE/TF Final 分别为 -0.3051/-0.1669/-0.0524，Macro Final 为 -0.1748。这说明在作者旧 checkpoint 上，即使修正缩放和对齐文字，加入整段 Contract 仍没有表现出通用收益。

### 配对差异与统计解释

下列区间对相同 140 条 QA 的题型 Final 差值执行 10,000 次配对 bootstrap（seed=72）。配对均值按 QA 数量加权，与三题型等权 Macro Final 不同。

- 新版 Contract − Baseline：-0.1767，95% CI [-0.3434, -0.0109]；win/tie/loss=62/0/78。
- 新版 Contract − Fixed 角色重命名：-0.1750，95% CI [-0.3567, +0.0011]；win/tie/loss=64/0/76。
- 新版 Contract − 旧完整 Contract：-0.0425，95% CI [-0.2182, +0.1336]；win/tie/loss=67/0/73。
- 新版 Contract − 追加实验 1：+0.0272，95% CI [-0.1432, +0.2037]；win/tie/loss=69/0/71。
- 新版 Contract − 追加实验 2：+0.1626，95% CI [-0.0196, +0.3420]；win/tie/loss=82/0/58。

新版相对 Baseline 的区间完全低于 0；在当前 `paper140` 口径下，总体下降不能解释为仅有方向不明的 QA 抽样波动。新版相对旧 Contract 的区间跨 0，而且题型方向相反，因此不能断言新版取得稳定的总体提升或下降。

### 输出行为与中间诊断

| 实验 | MC answer-first | TF answer-first | MC strict parse | TF strict parse | OE direct-start | 无 think 标签 | 平均字符数 | ≥3000 字符 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 13.95% | 9.52% | 13.95% | 9.52% | 10.91% | 69.29% | 703.9 | 0.71% |
| Fixed 角色重命名 | 4.65% | 7.14% | 4.65% | 7.14% | 5.45% | 67.86% | 694.2 | 0.00% |
| 旧完整 Contract | 6.98% | 0.00% | 6.98% | 0.00% | 1.82% | 58.57% | 796.5 | 0.00% |
| 修订版 Contract | 6.98% | 0.00% | 6.98% | 0.00% | 7.27% | 67.86% | 775.4 | 0.00% |

这里沿用确定性诊断器对原始输出的定义；它们不是 G-Eval 分数。新版没有加入答案边界，因此 MC/TF 仍常先输出推理正文，低 answer-first/strict-parse 不等于答案无法从全文识别。严格匹配到 MC 首行的 3 条回答中，选项正文精确匹配率为 100%；宽松启发式 MC 正确率从旧 Contract 的 88.37% 回升到 93.02%，TF 从 47.62% 回升到 61.90%。这些方向与 MC/TF 正式 Final 的恢复一致，但启发式统计不能替代 G-Eval。

新版输出平均长度由旧 Contract 的 796.5 降至 775.4 字符，“无 think 标签”由 58.57% 升至 67.86%，严格 OE direct-start 由 1.82% 升至 7.27%。然而 OE Final 同时下降，说明起答更直接或标签更少都不是内容质量提升的充分条件。

启发式 unsupported-number fraction 在 MC/TF 上由旧 Contract 的 32.32%/41.01% 变为 41.81%/38.02%，OE 则由 45.05% 升至 51.01%。该规则只检查数字能否在输入表面找到，不验证数字语义；OE 上升可作为进一步抽样审阅线索，但不能据此证明缩放条款导致 OE 退化。

### 关键结论

1. **新版第 1、2 条修正了旧 Contract 的两个明确文本问题，并恢复了 MC/TF。** 相对旧 Contract，MC Final +0.0535，TF Final +0.2476；尤其 TF Correctness +0.4286。
2. **恢复不是总体提升。** OE Final -0.3390 抵消了 MC/TF 收益，使 Macro Final 相对旧 Contract 仍为 -0.0126；相对 Baseline 为 -0.1757。
3. **不能把结果归因到单独某一条。** 本实验同时重写第 1 条、第 2 条并删除旧第 4 条，正式结果只能解释为该联合修改的增量。若要回答哪一条负责 OE 下降，必须继续做“只改第 1 条”和“只改第 2 条”的正交消融。
4. **最稳妥的即时推理条件仍是 Baseline 或仅 Fixed 角色重命名。** 新版相对 Fixed 角色重命名的三个题型 Final 均下降；对旧 checkpoint 加长且改变语义的 Contract 仍存在训练—推理 prompt 分布偏移。
5. **下一步应优先保留位置对齐修正，并简化数值条款。** 若继续验证，可预注册：位置对齐条款保持新版文字；数值条款只说明“整数用于相对形态比较，不推断单位”，把“除以 100”限制到确实询问原始尺度的 OE 子集，避免在所有题型上引入额外解码任务。

### 完整配置、运行过程与审计

- 运行时代码基于 `prompt@4d29da2d3e645861d67cc98a5eac96f94af68803` 的未提交修订版 Contract 修改部署；四个部署文件与本地逐文件 SHA-256 完全一致。prompt 规范 SHA-256 为 `e1aa401e762949d35d06f9e53c1b42d4f6f3b93ee9aacefe77618ee58a131f34`。
- checkpoint SHA-256：`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`；未执行训练、反向传播或参数更新。
- 固定清单：`paper140`，SHA-256 `c2aefed015869d153ab3f5632442f83ac3ffccf140685229ffab7a3a8e9cc1a6`；140 条 QA、70 个 series；MC=43、OE=55、TF=42。
- 推理：3×NVIDIA A100 40GB，`world_size=3`，`series` batching，`skip_loss=True`；完成 140/140 条。生成参数为 `do_sample=False`、`num_beams=5`、`max_new_tokens=1000`、`repetition_penalty=1.15`、`no_repeat_ngram_size=3`、`length_penalty=1`。
- 运行环境：Python 3.10.5、PyTorch 2.5.1+cu121、CUDA 12.1、Transformers 4.45.2、NumPy 1.26.4。正式推理无 OOM、无空 shard、无 traceback；只有既有的 `pad_token_id` 警告。
- Judge 仅使用 DeepSeek；请求与返回模型均为 `deepseek-v4-pro`，`enable_thinking=auto`，`max_tokens=4096`，primary workers=8，fallback workers=1，fallback samples=20，最大重试轮次=12。
- Judge 完成 335/335 条评分：`final_score_top_logprobs` 333 条，`exact_sample_mean_20` 2 条；provider、endpoint host、system fingerprint 和有效 Judge prompt 哈希均为 335/335，致命日志行数为 0。
- Judge token 用量：prompt 212,078，completion 505,021，total 717,099；cache hit 92,032，cache miss 120,046。
- Baseline 使用 Stage A 已锁定的 335 条分数，SHA-256 `c288f3ed3c162f4df10f9f0b38b50ac1d2e9e6831c54e5a93770e49d3bbebcc0`；只在聚合时合并，未重新请求 Judge。
- 预测审计：140 条、唯一模式 140 条、错误为空、`ok=True`。Judge 审计：140 个 record、335 条评分、333 条 logprobs、2 条 20 次精确采样回退、335 个有效 prompt 哈希、错误为空、`ok=True`。
- 本地测试：`python -m pytest tools/axis_repro/test_prompt_stage_a.py -q` 为 13 passed + 2 subtests；`python -m pytest tools/axis_repro -q` 为 54 passed + 2 subtests；`python -m compileall` 与 `git diff --check` 通过。

### 新实验产物与哈希

新实验目录为 `../prompt_revised_contract_deepseek_v4_pro_20260726/`。大体积原始预测、评分和日志保留在本地并由目录内 `.gitignore` 排除提交；小型审计与汇总文件可随代码版本化。

- `experiment_manifest.json`：去除机器路径后的完整配置、源文件哈希与产物哈希。
- `table_revised_contract.json` / `table_revised_contract.md`：六个相关条件的聚合结果、差值及配对区间。
- `prediction_audit.json`：推理覆盖率审计。
- `audit.json`：Judge 覆盖率、模型、provider、方法和 prompt 哈希审计。
- `output_diagnostics.json`：确定性输出行为汇总。
- `final_metadata.json`：Judge 模型、端点、fingerprint、方法计数和核心产物哈希。
- `paper140.json`：本轮固定清单。
- predictions SHA-256：`80e045229149d3101c9ee967604b6f9e7f1e0b0d67eddd6740b2c5eaeea3ba5b`
- scores SHA-256：`4650c46637a82394d182877d9a1297e1080c21bcf4318fbca83bd2c4977ac06b`
- run manifest SHA-256：`028040cc5d16b96f4c87cd0f7eb12aee90ae7a4fe7839e263581ef1e25e1b4a6`
- Judge audit SHA-256：`5ce25f0df3b9c057f78a84af5358ab7e809e1e942bef7ca36897734cb892f94b`
- diagnostics SHA-256：`3f2dd77f94423a2ce23078504c67bb2ce5687491c44ef50fabce272fc2228963`

### 解释边界

- 本实验只回答作者旧 checkpoint 对修订版 Contract 的即时响应，不代表在该 prompt 上重新训练后的结果。
- 正式集为论文口径的 140 条 QA，不是完整 284 条候选测试数据；没有多 seed 或 checkpoint 重复。
- 按用户要求只使用一个 Judge。配对区间不包含 Judge 模型间差异或服务端非确定性。
- 本条件同时修改多条 Contract；MC/TF 恢复和 OE 退化都不能归因到单独一条。

## 附录：全部正式实验 Prompt 模板

本附录覆盖报告中已经正式运行的 11 个模式。为避免把完全相同的文本重复多次，先给出“模式→模板”映射，再完整列出每一个唯一文本模板。`answer_boundary` 与 `answer_boundary_combined_234` 的变化发生在 generation prefix，不改变 question prompt 文本。

统一占位符含义：

- `{start}` / `{end}`：窗口的半开区间边界；最后一个实际步号是 `{end_minus_1}`。
- `{serialized_values}`：原实现按 `(x * 100):.0f` 生成的逗号分隔整数。
- `{local_hint_tokens}`：与窗口长度相同数量的 `<|local_hint|>`。
- `{fixed_hint_tokens}`：30 个 `<|fixed_hint|>`；只有移除 Fixed 的条件为空串。
- `{question}`：当前测试问题及其选项。
- `{ACTIVE_OUTPUT_PROTOCOL}`：根据题型选择本附录末尾列出的 MC、TF 或 OE 协议。
- `{aligned_rows}`：逐步交错的真实行，不包含示意模板行。

### 模式与模板映射

| 正式模式 | 使用模板 | generation boundary |
|---|---|---|
| `base` | 模板 A | 作者原边界 |
| `answer_boundary` | 模板 A | question prefix 后追加 EOS token 与字面量 `Answer:` |
| `answer_boundary_wo_fixed` | 模板 B | question prefix 后追加 EOS token 与字面量 `Answer:` |
| `task_protocol` | 模板 C | 作者原边界 |
| `fixed_role` | 模板 D | 作者原边界 |
| `fixed_role_evidence_contract` | 模板 E | 作者原边界 |
| `combined_234` | 模板 F | 作者原边界 |
| `answer_boundary_combined_234` | 模板 F | question prefix 后追加 EOS token 与字面量 `Answer:` |
| `expert_contract3_fixed_section` | 模板 G | 作者原边界 |
| `expert_contract3_interleaved` | 模板 H | 作者原边界 |
| `fixed_role_evidence_contract_revised` | 模板 I | 作者原边界 |

### 模板 A：Baseline / EOS + Answer:

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
{question}
```

`answer_boundary` 使用完全相同的文本，只在 embedding prefix 末尾追加一个 EOS token 和 tokenizer 编码后的字面量 `Answer:`。

### 模板 B：EOS + Answer: / w/o Fixed

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:**

### Question
{question}
```

该模板在 embedding prefix 末尾追加 EOS token 与 `Answer:`；`Overall Summary Hints` 标签仍存在，但 Fixed token 数为 0。

### 模板 C：题型协议

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
{question}

### Active Output Protocol
{ACTIVE_OUTPUT_PROTOCOL}
```

### 模板 D：Fixed 角色重命名

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Learned Task Guidance/Shared Task-Control Tokens:** {fixed_hint_tokens}

### Question
{question}
```

### 模板 E：Fixed 角色重命名 + 旧完整 Evidence Contract

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [{start}, {end}), i.e. steps {start} through {end_minus_1}.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Learned Task Guidance/Shared Task-Control Tokens:** {fixed_hint_tokens}

### Question
{question}
```

### 模板 F：综合 2+3+4 / EOS + Answer: + 综合 2+3+4

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [{start}, {end}), i.e. steps {start} through {end_minus_1}.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Learned Task Guidance/Shared Task-Control Tokens:** {fixed_hint_tokens}

### Question
{question}

### Active Output Protocol
{ACTIVE_OUTPUT_PROTOCOL}
```

`answer_boundary_combined_234` 使用相同文本，但额外在 embedding prefix 末尾追加 EOS token 与 `Answer:`。

### 模板 G：专家开头 + Contract 前三条 + Fixed 独立节

```text
You are an expert time-series anomaly analyst. Produce one precise, evidence-grounded answer to the question.

### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [{start}, {end}), i.e. steps {start} through {end_minus_1}.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}

### Learned Task Guidance/Shared Task-Control Tokens

{fixed_hint_tokens}

### Question
{question}
```

### 模板 H：专家开头 + Contract 前三条 + 逐步交错证据

```text
You are an expert time-series anomaly analyst. Produce one precise, evidence-grounded answer to the question.

### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [{start}, {end}), i.e. steps {start} through {end_minus_1}.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

### Time-Series Evidence

Each row has the format:
step | observed value | aligned per-step latent evidence

{aligned_rows}

### Learned Task Guidance/Shared Task-Control Tokens

{fixed_hint_tokens}

### Question
{question}
```

其中 `{aligned_rows}` 的每一行均为真实输入：

```text
Step {step:04d} | value={rounded_scaled_integer:+06d} | context=<|local_hint|>
```

### 模板 I：修订版四条 Evidence Contract（本次实验）

```text
You are an expert time series analyst. Analyze the provided data and answer the question.

### Evidence Contract
1. Window Values are rounded integer encodings of sample-specific
   observations for the half-open interval [{start}, {end}), listed in
   chronological order from step {start} through step {end_minus_1}.
   Use the displayed scale consistently when comparing shape, direction,
   relative change, and local deviations. Do not infer physical units.
   If the question explicitly requires an approximate unscaled numerical
   value, divide the displayed integer by 100 exactly once; otherwise,
   do not rescale the values.

2. Per-Step Context tokens are sample-specific and align one-to-one with
   Window Values by sequence position, not by text row. The first token
   corresponds to the first value at step {start}; each subsequent token
   corresponds to the next value; and the last token corresponds to the
   last value at step {end_minus_1}. Use each token only with its aligned
   value when judging whether that local behavior is unexpected relative
   to the complete temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Learned Task Guidance/Shared Task-Control Tokens:** {fixed_hint_tokens}

### Question
{question}
```

### `{ACTIVE_OUTPUT_PROTOCOL}` 的三种实际内容

#### Multiple Choice

```text
This is a single-answer multiple-choice question.

Silently evaluate every option against the time-series evidence. Do not use option position, option-letter frequency, or the wording of the question as evidence.

Output requirements:

1. The first line must be exactly: <LETTER>) <EXACT TEXT OF THE SELECTED OPTION>
2. Copy the selected option text verbatim. Do not paraphrase it and do not output only the letter.
3. On the next line, write "Explanation:" followed by 3–5 focused sentences.
4. The explanation must:

   * state the observed pattern that positively supports the selected option;
   * identify its relevant location, shape, persistence, or recovery behavior;
   * explain briefly why the strongest competing option does not fit.
5. Do not enumerate every option. Do not write "the correct answer is" after already giving the answer.
```

#### True/False

```text
This is a true-or-false question.

Judge the truth of the complete proposition, not merely whether an anomaly exists. Silently identify and evaluate every essential clause, including negation, "all", "only", "no", anomaly shape, temporal position, duration, boundary containment, and conjunctions. If an essential required clause is contradicted, the complete proposition is false.

Output requirements:

1. The first token must be exactly "True." or "False."
2. Continue with 3–5 natural sentences without headings such as "Explanation:" or "Evidence:".
3. If the answer is False, explicitly identify which part of the proposition is incorrect and state the evidence-consistent correction.
4. Support the judgment with the relevant observed pattern, location, duration or recovery behavior, and contextual consistency.
5. Never map anomaly presence mechanically to False or anomaly absence mechanically to True.
```

#### Open-ended

```text
This is an open-ended question.

Answer the actual diagnostic question rather than giving a generic anomaly-detection tutorial. Identify every requested component of the question, such as anomaly status, anomaly type, temporal location, relation to window boundaries, supporting evidence, persistence, recovery, or analytical indicators.

Output requirements:

1. Begin with a direct diagnostic conclusion in the first sentence.
2. Do not begin with A, B, C, D, True, or False unless the question explicitly requests such a format.
3. Write one coherent paragraph of normally 4–6 focused sentences.
4. Cover every requested component, but do not add unrelated background.
5. Ground the answer in the observed shape, location, duration or recovery, and contextual consistency.
6. If the question asks what evidence or techniques should be examined, first assess the supplied window and then name only the indicators relevant to that assessment.
7. End when the requested conclusion and justification are complete; do not pad the answer with generic advice.
```

## 大规模 Prompt Pareto 自研究与确认实验（2026-07-26 至 2026-07-27）

### 最终结论

本轮在已发布 checkpoint 上共完成 35 个 prompt 条件/候选的分阶段研究，
但**没有任何候选通过预注册的 untouched holdout 确认**。因此，没有 prompt
被锁定，也没有对 `paper140` 正式集运行候选。这里的“没有运行”不是资源或
网络原因，而是预注册停止条件：若两个冻结候选均未在 holdout 达到 10/10
指标非下降并通过 correct→wrong 防护，则禁止从 holdout 选较不差者、禁止
继续调参，也禁止触碰最终测试集。

最终确认结果为：

| Holdout 模式 | 非下降指标 | 最差差值 | 十指标平均差值 | MC correct→wrong | 是否确认 |
|---|---:|---:|---:|---:|:---:|
| `route_r3_02_mc_f1_safe` | 7/10 | -0.058823 | -0.010482 | 1 | 否 |
| `route_r3_01_mc_f0_safe` | 7/10 | -0.411745 | -0.113527 | 1 | 否 |

`route_r3_02_mc_f1_safe` 的 MC Final/Correctness/Reasoning 相对 holdout
Baseline 分别下降 0.042289、0.058823、0.003710（相对下降
1.094%、1.515%、0.097%）。`route_r3_01_mc_f0_safe` 分别下降
0.370582、0.352940、0.411745（相对下降 9.589%、9.091%、10.769%）。
两个候选的 OE/TF 七个指标与 Baseline 严格相等，因为它们逐条复用了同一
holdout Baseline 的回答和 Judge 分数。

### 研究漏斗与数据隔离

| 阶段 | 数据 | 候选/模式 | GPU 预测 | 新 Judge 维度 | 结果 |
|---|---|---:|---:|---:|---|
| Screening | 24 QA / 12 series | Baseline + 12 | 312 | 715 | 最好 9/10，未成功 |
| Validation | 72 QA / 36 series | Baseline + 5 | 432 | 996 | 无候选通过 gate |
| Development R2 | 暴露后的 96 QA / 48 series | 6 | 576 | 1,326 | 最好 7/10 |
| Development R3 | 同一 development96 | 8 | 288 个新预测并做组件复用 | 261 个新分数；合成 1,989 | 4 个数值 10/10；2 个通过决策防护 |
| Untouched holdout | 48 QA / 24 series | Baseline + 2 | 144 | 179；合成 333 | 两候选均失败 |
| Final `paper140` | 140 QA | 0 | 0 | 0 | 按停止规则保持未触碰 |

数据拆分按 series 互斥。screening 与 validation 在形成新候选后合并为暴露的
development96；holdout 在 R3-01/R3-02 冻结以前从未被读取或评分。
`paper140` 只允许在 holdout 锁定唯一胜者后运行一次，因此本轮没有使用其
候选结果。论文正式 Baseline 仍为：

| MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.255815 | 4.325584 | 4.093021 | 3.085931 | 2.836509 | 2.770042 | 3.745460 | 3.642142 | 3.641666 | 3.642856 |

这些 paper140 数字只作为研究开始前已经锁定的 Baseline，不曾用于本轮
候选选择。

### 关键失败 case 与失败原因

#### R3-02：一致性规则可以修复矛盾，也会冻结错误的初始判断

在 `series_000022:1` 中，Baseline 先选 D，解释又说没有任何选项正确；
R3-02 改为金标 B 并保持解释一致，单题 Final 从 1.000001 升到 4.700000。
这说明 “select exactly one / do not revise” 确实能修复回答后改口。

但在正常窗口 `series_000141:1` 中，Baseline 正确选择 A，R3-02 将普通的
正负交替误读为“急跌并立即恢复”，错误选择异常选项 D，Final 从 4.700000
降到 1.299998。该规则约束的是**选择后的承诺**，并没有提高证据到选项的
初始映射；初始判断错误时，它反而阻止模型自我纠正。一个 +3.70 的纠错和
一个 -3.40 的退化基本抵消，其余若干正确选项又因解释变短而丢失 reasoning
分，最终三个 MC 聚合指标全部略降。

#### R3-01：对 learned fixed tokens 的角色重命名不是语义中性的

在正常窗口 `series_000139:0` 中，Baseline 正确选择 D。R3-01 却选择描述
异常波动的 C，但解释正文明确说这是 normal variability，并以 “no evidence
of anomalous behavior” 结束。Final 从 4.999999 降到 1.300001。

30 个 learned fixed tokens 是在原 `Overall Summary Hints` 标题下共同训练
出来的。把标题改成 `Learned Task Guidance/Shared Task-Control Tokens`
虽然对人更准确，却改变了 checkpoint 解码这些 latent tokens 的文本条件，
造成选项标签与后续理由脱钩。因此不能把 header/角色重命名视为仅格式修改。

#### OE：新增文字主要改变诊断，而不是补充证据覆盖

R2 的旧 Contract 只在 OE Relevance 上提高，但 Accuracy/Completeness 下降；
Q1/Q2/Q3 三个正向 OE 规则在 R3 的 Final/Accuracy/Completeness 分别全面
下降。典型错误包括：否认输入中清晰的 downward event、编造窗口外 step 或
数值、把正常可逆波动改判为异常、以及对方法/证据类问题强行给出诊断闭环。
“写得更长”或“明确列出必须覆盖的内容”没有解决问题，因为新增规则首先改变
了模型的内容选择与异常判断。

#### TF：直接极性规则的平均收益被真实异常退化否决

TF-P0 在 development96 上修复多条否定命题映射错误，但把
`series_000132:1` 的真实 spike + sustained increase 从正确 True 改成
False。包含 TF-P0 的 R3-03/R3-04 虽然数值达到 10/10，仍因这一条
Baseline-correct→candidate-wrong 被预注册防护淘汰。聚合均值上升不能解释
或抵消一个未解决的真实异常漏检。

### 综合机制结论

1. 发布模型对训练 prompt 具有明显共同适配；latent-token 标题、任务语义和
   生成行为不是可独立替换的模块。
2. 长 Contract 同时激活缩放、对齐、异常定义与回答控制，导致 instruction
   competition；修改其中的事实错误仍不能消除竞争。
3. anomaly feature 列表即使以否定形式出现，也会提高这些特征的显著性，
   在正常高波动窗口上增加 false positive。
4. 输出一致性约束只能防止改口，无法保证初始选项正确；错误时会形成
   premature commitment。
5. 题型路由能够把 OE/TF 保护为 Baseline，但本轮没有找到可在 series-disjoint
   holdout 上同时保住 MC 三指标的文本变化。
6. development 的改善由少数灾难性 case 修复驱动时，必须检查对称的灾难性
   新错误；仅看均值或 10/10 的零差组合会高估泛化能力。

### 正式配置与审计汇总

- checkpoint：作者发布权重，SHA-256
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。
- 所有正式推理均在授权 GPU 上执行，series batching、beam size 5、
  `--skip-loss`；没有在本机 CPU 上运行模型推理或训练。
- Judge 仅使用 `deepseek-v4-pro`，主读数为最终 Score token 的 1–5
  概率期望；缺少完整概率分布时使用严格 20 次独立采样均值。
- Holdout 成功推理为 144/144；选择性 Judge 为 179/179，其中 178 个概率
  读数、1 个 exact-20；合成审计为 144 个预测、333 个维度，全部 prompt
  hash 有效，无缺失键、重复键、模型/供应商混用。
- 首次 holdout 三卡加载期间有无关进程抢占 GPU 0，造成零预测 OOM。失败
  日志和 preflight 均保留；进程退出后使用原三卡协议在新目录成功重跑，
  没有混合失败产物。
- 完整研究状态、时间线与结论分别在
  `prompt_autoresearch_deepseek_v4_pro_20260726/research-state.yaml`、
  `research-log.md`、`findings.md`。
- Holdout 完整分析和逐题配对数据在
  `experiments/holdout-confirmation/analysis.md` 与
  `analysis_output/paired_cases.jsonl`。

### 停止条件与后续研究边界

本轮不存在满足“MC/OE/TF 全部 Table-I 指标不低于 Baseline”的确认 prompt。
若继续使用当前 holdout 迭代，就会把确认集变成开发集；若在没有胜者时直接
测试 `paper140`，又会把最终测试集用于选择。二者都会破坏本轮声明的泛化
证据。因此当前可复现结论是**负确认**：在既定数据预算和 checkpoint 下，
已测试的 prompt-only 改动没有全面优于 Baseline。

未来若继续，应先获得一个新的、series-disjoint、此前未查看的选择集，然后
重新预注册候选数量与唯一 final-run 规则；不能在本轮 holdout failure cases
上修改文字后继续报告同一个 holdout 的“提升”。

## 全部 Autoresearch 指标表

下列五组表保留每个候选的绝对分数与相对同阶段 Baseline 的绝对差值；随后
给出逐指标相对百分比。不同 split 的 Baseline 组成不同，因此只能在同一
阶段内比较，不应跨表直接比较绝对数值。

### Screening24：全部绝对分数与绝对差值

#### Screening round 1 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pareto_p01_boundary | 9/10 | -0.286 | +0.253 | +0.375 | +0.375 | +0.375 | +0.129 | -0.286 | +0.286 | +0.429 | +0.289 | +0.333 | +0.222 |
| 2 | pareto_p03_task_rule | 8/10 | -0.143 | +0.205 | +0.387 | +0.500 | +0.125 | +0.056 | -0.000 | +0.282 | -0.143 | +0.289 | +0.333 | +0.222 |
| 3 | pareto_p06_abc | 8/10 | -0.286 | +0.287 | +0.532 | +0.706 | +0.125 | +0.029 | -0.286 | -0.000 | +0.429 | +0.444 | +0.444 | +0.444 |
| 4 | pareto_p07_abd | 7/10 | -0.143 | +0.038 | +0.125 | +0.125 | +0.125 | +0.043 | -0.143 | +0.143 | +0.143 | -0.067 | -0.111 | +0.000 |
| 5 | pareto_p04_numeric_guard | 7/10 | -0.429 | +0.026 | +0.063 | +0.250 | -0.375 | +0.007 | -0.429 | +0.571 | -0.143 | +0.089 | +0.000 | +0.222 |
| 6 | pareto_p08_acde | 7/10 | -0.571 | +0.109 | +0.513 | +0.625 | +0.250 | -0.229 | -0.571 | +0.286 | -0.429 | +0.200 | +0.111 | +0.333 |
| 7 | pareto_p10_abcde | 6/10 | -0.286 | +0.096 | +0.350 | +0.500 | +0.000 | -0.186 | -0.286 | -0.000 | -0.286 | +0.311 | +0.444 | +0.111 |
| 8 | pareto_p11_abce | 6/10 | -0.571 | +0.014 | +0.362 | +0.625 | -0.250 | -0.200 | -0.571 | -0.000 | +0.000 | +0.067 | +0.111 | +0.000 |
| 9 | pareto_p05_single_answer | 6/10 | -0.714 | +0.137 | +0.775 | +1.000 | +0.250 | -0.144 | -0.229 | +0.428 | -0.714 | +0.000 | +0.000 | -0.000 |
| 10 | pareto_p09_bce | 6/10 | -0.714 | +0.000 | +0.388 | +0.500 | +0.125 | -0.414 | -0.429 | -0.143 | -0.714 | +0.244 | +0.333 | +0.111 |
| 11 | pareto_p02_calibration | 5/10 | -0.250 | +0.072 | -0.250 | -0.250 | -0.250 | +0.057 | -0.000 | +0.286 | -0.143 | +0.433 | +0.500 | +0.333 |
| 12 | fixed_role | 5/10 | -0.429 | +0.061 | +0.562 | +0.750 | +0.125 | -0.007 | -0.429 | +0.285 | +0.143 | -0.267 | -0.222 | -0.333 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| pareto_p01_boundary | 4.075 | 4.000 | 4.250 | 2.664 | 2.286 | 2.143 | 3.714 | 3.622 | 3.667 | 3.556 |
| pareto_p03_task_rule | 4.087 | 4.125 | 4.000 | 2.592 | 2.571 | 2.139 | 3.143 | 3.622 | 3.667 | 3.556 |
| pareto_p06_abc | 4.232 | 4.331 | 4.000 | 2.564 | 2.286 | 1.857 | 3.715 | 3.778 | 3.778 | 3.778 |
| pareto_p07_abd | 3.825 | 3.750 | 4.000 | 2.579 | 2.428 | 2.000 | 3.429 | 3.267 | 3.222 | 3.333 |
| pareto_p04_numeric_guard | 3.763 | 3.875 | 3.500 | 2.543 | 2.143 | 2.429 | 3.143 | 3.422 | 3.333 | 3.556 |
| pareto_p08_acde | 4.213 | 4.250 | 4.125 | 2.307 | 2.000 | 2.143 | 2.857 | 3.533 | 3.444 | 3.667 |
| pareto_p10_abcde | 4.050 | 4.125 | 3.875 | 2.350 | 2.286 | 1.857 | 3.000 | 3.644 | 3.778 | 3.444 |
| pareto_p11_abce | 4.063 | 4.250 | 3.625 | 2.336 | 2.000 | 1.857 | 3.286 | 3.400 | 3.444 | 3.333 |
| pareto_p05_single_answer | 4.475 | 4.625 | 4.125 | 2.391 | 2.343 | 2.286 | 2.571 | 3.333 | 3.333 | 3.333 |
| pareto_p09_bce | 4.088 | 4.125 | 4.000 | 2.121 | 2.143 | 1.714 | 2.571 | 3.578 | 3.667 | 3.444 |
| pareto_p02_calibration | 3.450 | 3.375 | 3.625 | 2.593 | 2.571 | 2.143 | 3.143 | 3.767 | 3.833 | 3.667 |
| fixed_role | 4.262 | 4.375 | 4.000 | 2.529 | 2.143 | 2.143 | 3.429 | 3.067 | 3.111 | 3.000 |

##### Paired outcome counts

- `pareto_p01_boundary`: wins=15, ties=0, losses=9, correct→wrong=0, wrong→correct=1.
- `pareto_p03_task_rule`: wins=13, ties=0, losses=11, correct→wrong=2, wrong→correct=5.
- `pareto_p06_abc`: wins=16, ties=0, losses=8, correct→wrong=1, wrong→correct=5.
- `pareto_p07_abd`: wins=11, ties=0, losses=13, correct→wrong=0, wrong→correct=1.
- `pareto_p04_numeric_guard`: wins=12, ties=0, losses=12, correct→wrong=0, wrong→correct=1.
- `pareto_p08_acde`: wins=9, ties=0, losses=15, correct→wrong=1, wrong→correct=5.
- `pareto_p10_abcde`: wins=12, ties=0, losses=12, correct→wrong=1, wrong→correct=5.
- `pareto_p11_abce`: wins=13, ties=0, losses=11, correct→wrong=1, wrong→correct=5.
- `pareto_p05_single_answer`: wins=12, ties=0, losses=12, correct→wrong=1, wrong→correct=5.
- `pareto_p09_bce`: wins=8, ties=0, losses=16, correct→wrong=1, wrong→correct=5.
- `pareto_p02_calibration`: wins=14, ties=0, losses=10, correct→wrong=1, wrong→correct=0.
- `fixed_role`: wins=12, ties=0, losses=12, correct→wrong=1, wrong→correct=1.

### Validation72：全部绝对分数与绝对差值

#### Screening round 1 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pareto_p07_abd | 7/10 | -0.457 | +0.128 | +0.081 | +0.038 | +0.181 | -0.271 | -0.457 | -0.318 | +0.001 | +0.662 | +0.575 | +0.792 |
| 2 | fixed_role | 6/10 | -0.318 | +0.173 | +0.232 | +0.298 | +0.079 | -0.191 | -0.318 | -0.227 | -0.000 | +0.604 | +0.506 | +0.750 |
| 3 | pareto_p01_boundary | 5/10 | -0.182 | +0.063 | +0.042 | +0.077 | -0.038 | -0.109 | -0.182 | -0.091 | -0.045 | +0.304 | +0.173 | +0.500 |
| 4 | pareto_p03_task_rule | 4/10 | -0.727 | -0.221 | +0.042 | +0.077 | -0.038 | -0.509 | -0.727 | -0.727 | +0.000 | -0.121 | -0.202 | +0.000 |
| 5 | pareto_p06_abc | 3/10 | -0.545 | -0.173 | -0.093 | -0.038 | -0.219 | -0.432 | -0.545 | -0.455 | -0.273 | +0.112 | +0.131 | +0.083 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| pareto_p07_abd | 4.020 | 4.000 | 4.065 | 3.179 | 3.270 | 2.773 | 3.547 | 3.608 | 3.652 | 3.542 |
| fixed_role | 4.171 | 4.260 | 3.963 | 3.259 | 3.409 | 2.864 | 3.545 | 3.550 | 3.583 | 3.500 |
| pareto_p01_boundary | 3.981 | 4.039 | 3.846 | 3.341 | 3.545 | 3.000 | 3.500 | 3.250 | 3.250 | 3.250 |
| pareto_p03_task_rule | 3.981 | 4.038 | 3.846 | 2.941 | 3.000 | 2.364 | 3.545 | 2.825 | 2.875 | 2.750 |
| pareto_p06_abc | 3.846 | 3.923 | 3.665 | 3.018 | 3.182 | 2.636 | 3.273 | 3.058 | 3.208 | 2.833 |

##### Paired outcome counts

- `pareto_p07_abd`: wins=44, ties=0, losses=28, correct→wrong=0, wrong→correct=5.
- `fixed_role`: wins=44, ties=0, losses=28, correct→wrong=1, wrong→correct=7.
- `pareto_p01_boundary`: wins=39, ties=0, losses=33, correct→wrong=0, wrong→correct=3.
- `pareto_p03_task_rule`: wins=33, ties=0, losses=39, correct→wrong=4, wrong→correct=10.
- `pareto_p06_abc`: wins=35, ties=0, losses=37, correct→wrong=4, wrong→correct=10.

### Development96 R2：全部绝对分数与绝对差值

#### Development round 2 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r2_04_oe_old_contract | 7/10 | -0.381 | +0.205 | +0.312 | +0.294 | +0.355 | -0.061 | -0.381 | -0.060 | +0.311 | +0.409 | +0.298 | +0.576 |
| 2 | route_r2_03_tf_boundary | 6/10 | -0.552 | -0.032 | +0.223 | +0.294 | +0.057 | -0.378 | -0.552 | -0.379 | -0.173 | +0.191 | +0.156 | +0.242 |
| 3 | route_r2_01_minimal | 6/10 | -0.621 | +0.011 | +0.229 | +0.265 | +0.147 | -0.460 | -0.621 | -0.517 | -0.207 | +0.421 | +0.398 | +0.455 |
| 4 | route_r2_02_mc_stable | 6/10 | -0.729 | -0.035 | +0.264 | +0.294 | +0.193 | -0.588 | -0.729 | -0.655 | -0.345 | +0.391 | +0.308 | +0.515 |
| 5 | route_r2_06_full_routed | 6/10 | -1.034 | -0.077 | +0.312 | +0.382 | +0.147 | -0.590 | -1.034 | -0.414 | -0.276 | +0.233 | +0.227 | +0.241 |
| 6 | route_r2_05_oe_contract_coverage | 6/10 | -1.069 | -0.073 | +0.256 | +0.265 | +0.235 | -0.643 | -1.069 | -0.517 | -0.291 | +0.337 | +0.282 | +0.420 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r2_04_oe_old_contract | 4.195 | 4.176 | 4.237 | 3.168 | 3.067 | 2.733 | 3.794 | 3.461 | 3.445 | 3.485 |
| route_r2_03_tf_boundary | 4.105 | 4.176 | 3.940 | 2.852 | 2.897 | 2.414 | 3.310 | 3.242 | 3.303 | 3.152 |
| route_r2_01_minimal | 4.112 | 4.147 | 4.029 | 2.769 | 2.828 | 2.276 | 3.276 | 3.473 | 3.545 | 3.364 |
| route_r2_02_mc_stable | 4.146 | 4.176 | 4.075 | 2.641 | 2.719 | 2.138 | 3.137 | 3.442 | 3.455 | 3.424 |
| route_r2_06_full_routed | 4.194 | 4.265 | 4.029 | 2.640 | 2.414 | 2.379 | 3.207 | 3.285 | 3.374 | 3.150 |
| route_r2_05_oe_contract_coverage | 4.138 | 4.147 | 4.118 | 2.587 | 2.379 | 2.276 | 3.191 | 3.389 | 3.429 | 3.329 |

##### Paired outcome counts

- `route_r2_04_oe_old_contract`: wins=59, ties=0, losses=37, correct→wrong=1, wrong→correct=17.
- `route_r2_03_tf_boundary`: wins=46, ties=0, losses=50, correct→wrong=2, wrong→correct=15.
- `route_r2_01_minimal`: wins=53, ties=0, losses=43, correct→wrong=1, wrong→correct=17.
- `route_r2_02_mc_stable`: wins=49, ties=0, losses=47, correct→wrong=1, wrong→correct=18.
- `route_r2_06_full_routed`: wins=53, ties=0, losses=43, correct→wrong=2, wrong→correct=16.
- `route_r2_05_oe_contract_coverage`: wins=52, ties=0, losses=44, correct→wrong=1, wrong→correct=17.

### Development96 R3：全部绝对分数与绝对差值

#### Development round 3 results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r3_04_mc_f1_tf_p0 | 10/10 | +0.000 | +0.202 | +0.264 | +0.294 | +0.193 | +0.000 | +0.000 | +0.000 | +0.000 | +0.421 | +0.398 | +0.455 |
| 2 | route_r3_03_mc_f0_tf_p0 | 10/10 | +0.000 | +0.192 | +0.229 | +0.265 | +0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.421 | +0.398 | +0.455 |
| 3 | route_r3_02_mc_f1_safe | 10/10 | +0.000 | +0.075 | +0.264 | +0.294 | +0.193 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | route_r3_01_mc_f0_safe | 10/10 | +0.000 | +0.064 | +0.229 | +0.265 | +0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | route_r3_08_historical | 7/10 | -0.381 | +0.045 | +0.229 | +0.265 | +0.147 | -0.061 | -0.381 | -0.060 | +0.311 | +0.000 | +0.000 | +0.000 |
| 6 | route_r3_07_oe_q3 | 7/10 | -0.552 | +0.098 | +0.229 | +0.265 | +0.147 | -0.247 | -0.552 | -0.241 | +0.103 | +0.421 | +0.398 | +0.455 |
| 7 | route_r3_05_oe_q1 | 6/10 | -0.378 | +0.105 | +0.229 | +0.265 | +0.147 | -0.221 | -0.378 | -0.165 | -0.103 | +0.421 | +0.398 | +0.455 |
| 8 | route_r3_06_oe_q2 | 6/10 | -0.586 | +0.071 | +0.229 | +0.265 | +0.147 | -0.312 | -0.586 | -0.276 | -0.034 | +0.421 | +0.398 | +0.455 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_04_mc_f1_tf_p0 | 4.146 | 4.176 | 4.075 | 3.229 | 3.448 | 2.793 | 3.483 | 3.473 | 3.545 | 3.364 |
| route_r3_03_mc_f0_tf_p0 | 4.112 | 4.147 | 4.029 | 3.229 | 3.448 | 2.793 | 3.483 | 3.473 | 3.545 | 3.364 |
| route_r3_02_mc_f1_safe | 4.146 | 4.176 | 4.075 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_01_mc_f0_safe | 4.112 | 4.147 | 4.029 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_08_historical | 4.112 | 4.147 | 4.029 | 3.168 | 3.067 | 2.733 | 3.794 | 3.052 | 3.147 | 2.909 |
| route_r3_07_oe_q3 | 4.112 | 4.147 | 4.029 | 2.983 | 2.897 | 2.552 | 3.586 | 3.473 | 3.545 | 3.364 |
| route_r3_05_oe_q1 | 4.112 | 4.147 | 4.029 | 3.008 | 3.071 | 2.628 | 3.379 | 3.473 | 3.545 | 3.364 |
| route_r3_06_oe_q2 | 4.112 | 4.147 | 4.029 | 2.917 | 2.862 | 2.517 | 3.448 | 3.473 | 3.545 | 3.364 |

##### Paired outcome counts

- `route_r3_04_mc_f1_tf_p0`: wins=40, ties=29, losses=27, correct→wrong=1, wrong→correct=18.
- `route_r3_03_mc_f0_tf_p0`: wins=43, ties=29, losses=24, correct→wrong=1, wrong→correct=17.
- `route_r3_02_mc_f1_safe`: wins=16, ties=62, losses=18, correct→wrong=0, wrong→correct=3.
- `route_r3_01_mc_f0_safe`: wins=19, ties=62, losses=15, correct→wrong=0, wrong→correct=2.
- `route_r3_08_historical`: wins=36, ties=33, losses=27, correct→wrong=0, wrong→correct=2.
- `route_r3_07_oe_q3`: wins=56, ties=0, losses=40, correct→wrong=1, wrong→correct=17.
- `route_r3_05_oe_q1`: wins=59, ties=0, losses=37, correct→wrong=1, wrong→correct=17.
- `route_r3_06_oe_q2`: wins=55, ties=0, losses=41, correct→wrong=1, wrong→correct=17.

### Untouched Holdout48：全部绝对分数与绝对差值

#### Internal holdout confirmation results

##### Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r3_02_mc_f1_safe | 7/10 | -0.059 | -0.010 | -0.042 | -0.059 | -0.004 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | route_r3_01_mc_f0_safe | 7/10 | -0.412 | -0.114 | -0.371 | -0.353 | -0.412 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

##### Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.865 | 3.882 | 3.824 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| route_r3_02_mc_f1_safe | 3.822 | 3.824 | 3.820 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| route_r3_01_mc_f0_safe | 3.494 | 3.529 | 3.412 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |

##### Paired outcome counts

- `route_r3_02_mc_f1_safe`: wins=9, ties=31, losses=8, correct→wrong=1, wrong→correct=1.
- `route_r3_01_mc_f0_safe`: wins=7, ties=31, losses=10, correct→wrong=1, wrong→correct=0.

### 全部候选的相对变化比例

#### All autoresearch relative changes

Each percentage is `100 × (candidate − split Baseline) / split Baseline`. These split-specific percentages are diagnostic and must not be compared as if the splits had identical record composition.

##### Screening24

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_role | +15.203% | +20.690% | +3.226% | -0.287% | -16.668% | +15.361% | +4.351% | -8.000% | -6.667% | -9.999% |
| pareto_p01_boundary | +10.135% | +10.345% | +9.677% | +5.068% | -11.111% | +15.373% | +13.043% | +8.667% | +10.000% | +6.667% |
| pareto_p02_calibration | -6.757% | -6.897% | -6.452% | +2.251% | +0.000% | +15.373% | -4.348% | +13.000% | +15.000% | +10.000% |
| pareto_p03_task_rule | +10.473% | +13.793% | +3.226% | +2.205% | +0.000% | +15.191% | -4.346% | +8.666% | +10.000% | +6.666% |
| pareto_p04_numeric_guard | +1.689% | +6.897% | -9.677% | +0.279% | -16.666% | +30.756% | -4.347% | +2.667% | +0.000% | +6.667% |
| pareto_p05_single_answer | +20.946% | +27.586% | +6.452% | -5.692% | -8.889% | +23.064% | -21.739% | +0.000% | +0.000% | +0.000% |
| pareto_p06_abc | +14.375% | +19.483% | +3.226% | +1.131% | -11.111% | -0.010% | +13.061% | +13.333% | +13.333% | +13.333% |
| pareto_p07_abd | +3.378% | +3.448% | +3.226% | +1.686% | -5.560% | +7.682% | +4.348% | -2.000% | -3.333% | +0.000% |
| pareto_p08_acde | +13.851% | +17.241% | +6.452% | -9.017% | -22.222% | +15.372% | -13.043% | +6.000% | +3.333% | +10.000% |
| pareto_p09_bce | +10.473% | +13.793% | +3.226% | -16.340% | -16.667% | -7.701% | -21.739% | +7.333% | +10.000% | +3.333% |
| pareto_p10_abcde | +9.459% | +13.793% | +0.000% | -7.327% | -11.108% | -0.009% | -8.702% | +9.333% | +13.333% | +3.333% |
| pareto_p11_abce | +9.797% | +17.241% | -6.452% | -7.889% | -22.219% | -0.010% | +0.000% | +2.000% | +3.333% | +0.000% |

##### Validation72

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_role | +5.898% | +7.524% | +2.030% | -5.534% | -8.537% | -7.353% | +0.000% | +20.492% | +16.452% | +27.273% |
| pareto_p01_boundary | +1.076% | +1.944% | -0.990% | -3.162% | -4.878% | -2.941% | -1.282% | +10.310% | +5.620% | +18.182% |
| pareto_p03_task_rule | +1.074% | +1.942% | -0.990% | -14.756% | -19.512% | -23.528% | +0.000% | -4.115% | -6.567% | +0.000% |
| pareto_p06_abc | -2.354% | -0.971% | -5.644% | -12.517% | -14.634% | -14.706% | -7.694% | +3.804% | +4.265% | +3.030% |
| pareto_p07_abd | +2.060% | +0.971% | +4.653% | -7.850% | -12.255% | -10.294% | +0.039% | +22.458% | +18.687% | +28.787% |

##### Development96 R2

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_r2_01_minimal | +5.909% | +6.819% | +3.788% | -14.254% | -18.000% | -18.514% | -5.941% | +13.792% | +12.662% | +15.625% |
| route_r2_02_mc_stable | +6.792% | +7.576% | +4.962% | -18.213% | -21.150% | -23.456% | -9.915% | +12.799% | +9.774% | +17.707% |
| route_r2_03_tf_boundary | +5.746% | +7.576% | +1.477% | -11.691% | -16.000% | -13.567% | -4.959% | +6.246% | +4.959% | +8.333% |
| route_r2_04_oe_old_contract | +8.043% | +7.576% | +9.134% | -1.895% | -11.050% | -2.161% | +8.928% | +13.415% | +9.485% | +19.791% |
| route_r2_05_oe_contract_coverage | +6.591% | +6.818% | +6.061% | -19.899% | -31.000% | -18.519% | -8.366% | +11.048% | +8.966% | +14.427% |
| route_r2_06_full_routed | +8.030% | +9.848% | +3.788% | -18.260% | -30.000% | -14.817% | -7.921% | +7.626% | +7.222% | +8.281% |

##### Development96 R3

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_r3_01_mc_f0_safe | +5.909% | +6.819% | +3.788% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| route_r3_02_mc_f1_safe | +6.792% | +7.576% | +4.962% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| route_r3_03_mc_f0_tf_p0 | +5.909% | +6.819% | +3.788% | +0.000% | +0.000% | +0.000% | +0.000% | +13.792% | +12.662% | +15.625% |
| route_r3_04_mc_f1_tf_p0 | +6.792% | +7.576% | +4.962% | +0.000% | +0.000% | +0.000% | +0.000% | +13.792% | +12.662% | +15.625% |
| route_r3_05_oe_q1 | +5.909% | +6.819% | +3.788% | -6.847% | -10.953% | -5.922% | -2.970% | +13.792% | +12.662% | +15.625% |
| route_r3_06_oe_q2 | +5.909% | +6.819% | +3.788% | -9.664% | -17.000% | -9.878% | -0.990% | +13.792% | +12.662% | +15.625% |
| route_r3_07_oe_q3 | +5.909% | +6.819% | +3.788% | -7.635% | -16.000% | -8.642% | +2.970% | +13.792% | +12.662% | +15.625% |
| route_r3_08_historical | +5.909% | +6.819% | +3.788% | -1.895% | -11.050% | -2.161% | +8.928% | +0.000% | +0.000% | +0.000% |

##### Holdout48

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| route_r3_02_mc_f1_safe | -1.094% | -1.515% | -0.097% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |
| route_r3_01_mc_f0_safe | -9.589% | -9.091% | -10.769% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% | +0.000% |

## 附录：Autoresearch 使用的全部 Prompt（文档末尾）

以下 catalog 是推理前用固定示例窗口实际渲染并计算 SHA-256 的完整文本。运行时仅替换 sample-specific 的窗口、数值、latent placeholder 数量和问题；每个 Fixed 条件均保留 30 个 `<|fixed_hint|>`。表中出现相同 SHA-256 表示该题型组件被逐字节复用，而不是重新改写。

### Screening Round 1：Fixed role 与 P01–P11

#### Screening round 1 prompt catalog

Rendered before inference with a fixed illustrative window. Runtime 
values, steps, latent placeholders, and questions are substituted per sample.

##### fixed_role / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### fixed_role / open_ended

SHA-256: `22a815d718ac77ccb09e894aa1b62d3fa3c7edc9593bba5117d63385bf551949`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### fixed_role / true_false

SHA-256: `ef8867788295ce4777b8e80b26cbb7b3360e088eb6fc30a1235a75ff28b490f7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p01_boundary / multiple_choice

SHA-256: `6f3678985ec7846f6c8e2352b32ff48471cb383f5c16062eb1c77501816243dd`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p01_boundary / open_ended

SHA-256: `84660aee5bfa6338fed0f7ee3be07e323faa4accab49553662d05f11cb932318`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p01_boundary / true_false

SHA-256: `439b90f2b3c665bb4c22b77ef30173b6838c3141645b7b1148bca52d16b8bea1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p02_calibration / multiple_choice

SHA-256: `e600afd18b1ea8c095156e58e4faaf621a96cdeffafe073804bd0fc0ae351f6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p02_calibration / open_ended

SHA-256: `45a12c90c98dab3ed0209d3c5642d08a6ad467f9390ad6f59ed7d9d7ab9e74aa`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p02_calibration / true_false

SHA-256: `f05e7eaf748ab51189d9b0c8fd0795a1e3f986fb7036b70264679257779e9aad`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p03_task_rule / multiple_choice

SHA-256: `26bac720e9743601acdd3a9a261db5e8197041cb28ec05bd4fe0ff48d368dd91`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p03_task_rule / open_ended

SHA-256: `880e2ae5fd9bddc9bf9537e8582c1ab34d8a852e545d532c48757ea5b62578b6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p03_task_rule / true_false

SHA-256: `2966b574a3060dfcf02e27f14ee8a3e39da50c5ed8bec5fba02b488b0426ba47`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p04_numeric_guard / multiple_choice

SHA-256: `f5946c9225802e32f13dae2e5f8242ef5fe70297873593d3e12c3426814f88f9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p04_numeric_guard / open_ended

SHA-256: `4e30ce8cdd836f6223e51d3f906e1ac982cb890f66a6269bf65123523dbf3968`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p04_numeric_guard / true_false

SHA-256: `6448c49f39fe49564d2d0449c60f782cb16a23fee2b737431003263bb1ce24d9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p05_single_answer / multiple_choice

SHA-256: `0567400d3b90dbf3ac2dd5e9fb143c2cd2a433645cdc1badff59ba10e5af63d9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p05_single_answer / open_ended

SHA-256: `c1311b5567ff9c78c15f38ea3bc6af69aad8d290289cec82e087a3b8a5ec1f3a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p05_single_answer / true_false

SHA-256: `892e216fa2a14f987c1872b3e90fc2c9adfe085300cf8275950d72e0d596b80c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p06_abc / multiple_choice

SHA-256: `99ab4177e85ca491d6d6c78fb692e8545631ce2ac3a7b07ed3564dbb4e620341`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p06_abc / open_ended

SHA-256: `575c6ecb693bc202ef75dcdf8a07708538d337f8013cf2b8a8a89450e264f109`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p06_abc / true_false

SHA-256: `6410c53349710811327402e8379d97b481c4c6b96c62f5d3f16bf4c5907cebe3`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p07_abd / multiple_choice

SHA-256: `d808065164a247810684eb1fcca04cf5b33477793986a6e7382d4297863912d0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p07_abd / open_ended

SHA-256: `c07f585e8ecee409260e07c2ff2480de6a1fbb8cee201fee4821e3160aa88125`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p07_abd / true_false

SHA-256: `0766abfebc778fafef4b1e6f9bdab0b802da6b577c0014c7663a4e98e1cb988a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p08_acde / multiple_choice

SHA-256: `427bcccb202c909b218619fae7d15fd9714cdcd384cfa192a132a95dc66f082a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p08_acde / open_ended

SHA-256: `359d73d8f2c2b233da7eca78d8b6a5d1adc4eca0e44ab76119ffbb0404ddf7e0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p08_acde / true_false

SHA-256: `bf7aa998bc72aae1dffd860e0b0897ea5b41ef33e0b82d97b56623a893062eeb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p09_bce / multiple_choice

SHA-256: `fff5bcbeee210446d7f8c4a776dc098655844569148713a01987bf5494d0f9d5`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p09_bce / open_ended

SHA-256: `3f0e8de389a4f7ec362367495dbc0900f2216650c3c7c6fe78683412f8d45c2a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p09_bce / true_false

SHA-256: `ce52a77289889154c3642bc79374dfa5759c53370ec783aa15f0d8f93ae498f9`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p10_abcde / multiple_choice

SHA-256: `0a081ec3dc48eccc591d1d697c6ae659b8883fe7cd4dcf950707d06889ade015`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p10_abcde / open_ended

SHA-256: `1cfefc5bc3a43bd24b0f35ff9a4ad8e78546de49f9874e6fb6325a5a2aee0bda`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p10_abcde / true_false

SHA-256: `282b2c1f38a42cf4213bddc78b1b4116417b159ba316688c810e9a1c1a9dc092`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.
            - Use displayed values for qualitative shape, order, direction, relative change, and local contrast. Do not convert or quote exact numbers unless the question explicitly asks for an approximate numerical value in the original scale.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### pareto_p11_abce / multiple_choice

SHA-256: `7d66ad6fd4d365020e3452ee190eed79a22e3e880e4cc3a888be1351fb7bc908`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the evidence. The selected option must match anomaly status, shape, direction, temporal location, persistence or recovery, and boundary relation. Reject an option that requires an unsupported event.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### pareto_p11_abce / open_ended

SHA-256: `4dad0176a51f018d2660312b7210bb8548f1163b44869393002c59de4b5ef7d4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer every component requested by the question. Start from a diagnostic conclusion, then state the observed shape and location and why they support or refute an anomaly. Discuss boundary uncertainty only when relevant, and distinguish observed evidence from evidence that would still be needed.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### pareto_p11_abce / true_false

SHA-256: `ef67696bd745bbedf43d8604a23442b24946193beb2ccd65ff0e412016446f02`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Use
            - Use only evidence inside the half-open window [7, 10); never invent or cite a step outside it.
            - Call a behavior anomalous only when it is unexpected relative to the complete window and supported by Per-Step Analysis. A large or small value, sign change, smooth trend, ordinary peak or trough, or variability is not anomalous by itself; check local contrast, persistence, and recovery.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Evaluate the truth of the complete proposition, including negation and every required clause. Decide the underlying evidence claim first, then map it to True or False. Anomaly presence does not mechanically imply either label.

            ### Response Rule
            Give one final answer only. Do not repeat, revise, or contradict it; stop after the shortest explanation that fully supports the answer.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

### Development Round 2：R2-01–R2-06

#### Development round 2 routed prompt catalog

Rendered before inference with a fixed illustrative window. Runtime 
values, steps, latent placeholders, and questions are substituted per sample.

##### route_r2_01_minimal / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_01_minimal / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_01_minimal / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_02_mc_stable / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_02_mc_stable / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_02_mc_stable / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_03_tf_boundary / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_03_tf_boundary / open_ended

SHA-256: `e97b4a198a9d956275c778eb9dc569c5786b0a17d9b98c31ca8cb4b4a3a78ffe`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_03_tf_boundary / true_false

SHA-256: `e7d3cc6f957bfceb4bcce9aa6b45e77079884e2e5c74d952bd3543ff55335e99`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Use only evidence inside the half-open window [7, 10); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_04_oe_old_contract / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_04_oe_old_contract / open_ended

SHA-256: `85ff0be5d920d2ae5ab9ed0f8cef122e12924436a9dbcae6e53a8014f847ac28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_04_oe_old_contract / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_05_oe_contract_coverage / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_05_oe_contract_coverage / open_ended

SHA-256: `45edabaf3706ef6ae79d1e1c190af5e7860ed052597b1c80d2ecb2264fcbf319`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_05_oe_contract_coverage / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

##### route_r2_06_full_routed / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
            
```

##### route_r2_06_full_routed / open_ended

SHA-256: `45edabaf3706ef6ae79d1e1c190af5e7860ed052597b1c80d2ecb2264fcbf319`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.

            ### Question
            What evidence supports or refutes an anomaly in this window?
            
```

##### route_r2_06_full_routed / true_false

SHA-256: `e7d3cc6f957bfceb4bcce9aa6b45e77079884e2e5c74d952bd3543ff55335e99`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Use only evidence inside the half-open window [7, 10); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
            
```

### Development Round 3 与 Holdout：R3-01–R3-08

#### Development round 3 routed prompt catalog

Rendered before inference with a fixed illustrative window. Runtime
values, steps, latent placeholders, and questions are substituted per sample.

##### route_r3_01_mc_f0_safe / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_01_mc_f0_safe / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_01_mc_f0_safe / true_false

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_02_mc_f1_safe / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_02_mc_f1_safe / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_02_mc_f1_safe / true_false

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_03_mc_f0_tf_p0 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_03_mc_f0_tf_p0 / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_03_mc_f0_tf_p0 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_04_mc_f1_tf_p0 / multiple_choice

SHA-256: `0295f82d90bd24c4994d62c5fbaaabf5a2e497e8fcb26d0c523c080cf1c3ba17`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_04_mc_f1_tf_p0 / open_ended

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_04_mc_f1_tf_p0 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_05_oe_q1 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_05_oe_q1 / open_ended

SHA-256: `cef28363829aafc3adf63486dddedcea0e9203e08cc7ff4d125d609993fab737`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Answer the open-ended question in one coherent paragraph. Include: (1) a direct assessment of this window; (2) two or three observed qualitative features that support the assessment, including shape and persistence or recovery; and (3) the boundary/context limitation or additional evidence requested by the question. State location as beginning, middle, or end and describe magnitude relatively unless an exact value or step is unambiguous in the supplied input.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_05_oe_q1 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_06_oe_q2 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_06_oe_q2 / open_ended

SHA-256: `5b8464266ea03cd6ac45ae232cfb3901db9a82728fbe2bce9c8444e24149f580`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Treat the event or pattern named in the question as the behavior to evaluate, not as a predetermined anomaly label. Compare its abruptness, isolation, persistence or recovery, and boundary position with the rest of the supplied window. Report the evidence supporting your conclusion and the strongest evidence against it or remaining uncertainty.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_06_oe_q2 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_07_oe_q3 / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_07_oe_q3 / open_ended

SHA-256: `4556449ed061fb4a172f53faafd88fd3a47cfabec392feda67c674a0afa903c6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            First answer the assessment or analysis requested by the question. Then apply it to this window using qualitative evidence: local contrast, persistence or recovery, and boundary position. State what supports the conclusion and what observation would refute it or require outside-window context. Use beginning, middle, or end for location when exact step alignment is not explicit.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_07_oe_q3 / true_false

SHA-256: `cf96965a03fefdc51d9afebd852945b313d59e85aa2091f0ce8805d32d73ff81`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Task Rule
            Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

##### route_r3_08_historical / multiple_choice

SHA-256: `117b06954b0b34f72bf6770eaae86c8c038427b6032e275eb33c9099ed158159`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

##### route_r3_08_historical / open_ended

SHA-256: `85ff0be5d920d2ae5ab9ed0f8cef122e12924436a9dbcae6e53a8014f847ac28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Evidence Contract
1. Window values are exact sample-specific observations for the half-open
   interval [7, 10), i.e. steps 7 through 9.
   They are serialized as model-input values multiplied by 100.
   Do not infer physical units.

2. Each Per-Step Context token is sample-specific and corresponds to the
   value on the same row. Use it to determine whether that local behavior
   is unexpected relative to the full temporal pattern.

3. Shared Task-Control tokens are identical across samples. They specify
   how to answer, but they are NOT evidence that the current sample is
   normal or anomalous.

4. The wording of the question and the order of answer options are
   hypotheses, not evidence. If they conflict with the time-series
   evidence, follow the evidence.

5. An anomaly is an unexpected deviation relative to the global temporal
   pattern. A value is not anomalous merely because it is large, small,
   increasing, decreasing, or variable.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Learned Task Guidance/Shared Task-Control Tokens:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

##### route_r3_08_historical / true_false

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

---

# 文献驱动 Prompt Autoresearch 最终补充（2026-07-27）

## 结论

本轮完整阅读 `prompt系列改进.md`，结合 Time-LLM、选择题符号绑定与顺序偏差、上下文校准、输出格式偏差、RE2、CoT/小模型提示敏感性等原始论文，随后在不训练模型的前提下完成 30 个新模式的上限搜索。所有正式推理均在授权 GPU 上完成，所有正式评分仅使用 `deepseek-v4-pro`。

最终没有 Prompt 同时通过 development96、screening24、validation72 和 exposed holdout48 的预注册门槛。因而触发停止规则，没有锁定胜者，也没有生成`paper140` 候选输出。该结果不能表述为“找到了正式测试集上的全面提升 Prompt”。

## 最接近的内部结果

| 候选 | 数据层 | MC Final Δ | MC Corr. Δ | MC Rsn. Δ | TF Final Δ | TF Corr. Δ | TF Justif. Δ | c→w | 判定 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `lit_r4_01_tf_neg_re2` | exposed holdout48 | +0.0000 | +0.0000 | +0.0000 | +0.2000 | +0.1250 | +0.3125 | 1 | Table-I 全非降，但违反零伤害门槛 |
| `lit_r5_01_mc_structured_guard` | exposed holdout48 | +0.5471 | +0.5294 | +0.5882 | +0.0000 | +0.0000 | +0.0000 | 0 | holdout 通过，但 validation72 三项 MC 下降 |

OE 始终复用精确 Baseline，因此四项 OE 指标均为 0 差值。完整的 5 候选 × 4 数据层 × 10 指标、绝对分数、配对胜负和失败 case 见 `prompt_literature_autoresearch_deepseek_v4_pro_20260727/experiments/literature-round-5/analysis.md`。

## 失败 case 归因

- `series_000103:1`：structured guard 把 Baseline 正确的中心向上尖峰改判为缓慢下降后回归，说明阈值规则重新分配了注意力，而非只抑制假阳性。
- `series_000083:1`、`series_000089:1`：status-before-shape 先形成“正常”锚点，随后用正常/振荡选项解释数据，覆盖了 checkpoint 原本正确的边界尖峰识别。
- `series_000111:1`：TF RE2 虽总体修正更多极性错误，却把明确尖峰解释成渐进变化，使 `False→True` 并造成约 −1.60 的 Final 下降。
- `series_000022:1`：完整选项语义比较确实能把 Baseline 的边界尖峰误选修正为局部快速振荡，证明收益机制真实存在；但它与上述伤害机制不能靠当前 prompt 稳定分离。

## 研究判断

失败的共同原因不是 Judge 或解析器，而是 7B checkpoint 对训练时 prompt 分布高度敏感。新增自然语言规则会同时改变“读证据、判异常、匹配选项、组织答案”四个过程。即使 aggregate Table-I 指标上涨，也可能隐藏少数灾难性反转。在当前证据下，发布 Baseline 仍是唯一满足严格零回归要求的 prompt。

## 关键复现配置

- GPU 源提交：`4d3ace60c036cdb231a4c5aeafeec3cdf030aace`。
- checkpoint SHA-256：`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。
- development：192 条原始 GPU 输出、83 个路由分量、136 个新 Judge 维度；组装后 576 predictions / 1326 scores。
- holdout：144 条原始 GPU 输出、44 个路由分量、88 个新 Judge 维度；组装后 288 predictions / 666 scores。
- 审计：模型、provider、方法、prompt SHA-256、record/mode/dimension 唯一性和 manifest 均通过。
- `paper140`：候选调用 0 次；未用于选择或调参。

## 本轮 Prompt 全文索引

下面的最终附录由实现代码直接渲染，覆盖 30 个 literature-guided 模式及全部实际路由分支；每个分支附 SHA-256。为便于后续观察，该附录必须保持为本文档最后一部分。

---

# All literature-guided AXIS prompt templates

Rendered from the frozen implementation with representative evidence. Router modes include every distinct lexical branch used in experiments.

## `lit_r1_01_mc_semantic_bind`

### multiple_choice / default

SHA-256: `3f3c8df43cfa0dd3ef81b40b0e630652abe24ab0f583aa01b8f01d79d41c5553`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_02_mc_pointwise`

### multiple_choice / default

SHA-256: `3c39225d2d988dfd90e89db98ab9678e1c182518541aea18c3df796376c8e0ed`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Test each option's complete claim independently against the supplied evidence. Choose the best-supported option by its text, then report its attached letter and a brief reason.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_03_mc_re2`

### multiple_choice / default

SHA-256: `274997d12d8950b8f3d3c28b2d1f3ce71035dbb865a912fc6b2adf5e4da1c17c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic

            Read the question again:
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_04_tf_minimal`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `116ebc1425118aa96591965ffbab669e1a85028ca0018317b4fa52064a3b0ef4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Judge whether the complete statement as written is true. Report True or False once, followed by a brief reason that supports the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_05_tf_clause`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `92f67a6bd48ed68399badacd7f4d64366209d161bf0aa5608a766ca095bdf5ee`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Check every required clause and negation in the statement. It is false if any required clause is contradicted; report one truth label and a brief consistent reason.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_06_tf_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `e50bfa34b2bcc26387738fd2a19b7923caddefb09db9eaf4c284762a3f4e266f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_07_oe_direct`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `625bd1ab3fbe1a3ae2beb0dc55c0b5877598b0afb38a8467f6b070ff9c9e1b6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Answer exactly what the question asks, using evidence from this window. Include the requested conclusion or assessment and the brief reason needed to support it.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_08_oe_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `df2259a7bf131b369aa922c9f7ef0f84a13cdf61f458d69b941a442b96a34df6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?

            Read the question again:
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_09_triplet_minimal`

### multiple_choice / default

SHA-256: `3f3c8df43cfa0dd3ef81b40b0e630652abe24ab0f583aa01b8f01d79d41c5553`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `625bd1ab3fbe1a3ae2beb0dc55c0b5877598b0afb38a8467f6b070ff9c9e1b6b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Answer exactly what the question asks, using evidence from this window. Include the requested conclusion or assessment and the brief reason needed to support it.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `116ebc1425118aa96591965ffbab669e1a85028ca0018317b4fa52064a3b0ef4`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Judge whether the complete statement as written is true. Report True or False once, followed by a brief reason that supports the same truth value.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_10_triplet_re2`

### multiple_choice / default

SHA-256: `274997d12d8950b8f3d3c28b2d1f3ce71035dbb865a912fc6b2adf5e4da1c17c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic

            Read the question again:
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `df2259a7bf131b369aa922c9f7ef0f84a13cdf61f458d69b941a442b96a34df6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?

            Read the question again:
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `e50bfa34b2bcc26387738fd2a19b7923caddefb09db9eaf4c284762a3f4e266f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_11_decoupled`

### multiple_choice / default

SHA-256: `897db1dd97f1faa9067b7c8ddfe2503769657eee5de022cc9dbecef3917ab662`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First determine which complete option text is best supported by the evidence. Then report its attached letter once and give one brief reason.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `ef58122ce0a933b7612a3c7464414d96395618f64d29da0210926fe75851e99c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First determine the analysis requested by the question. Then answer it directly with a brief reason from the supplied evidence.

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `4cbf2443f3f85689a459a8fa3b61298a1e9d8814bab165885de76d2f9e787da3`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First determine whether the complete statement is supported. Then report True or False once and give one brief reason consistent with it.

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r1_12_mc_bind_re2`

### multiple_choice / default

SHA-256: `ed8e444eb1af3637b216a39e818f7b5e44e1992b13d606fafedd1c95bbc21c2f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Report that option once and give a brief reason from the supplied evidence.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic

            Read the question again:
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r2_01_mc_tf_re2_safe`

### multiple_choice / default

SHA-256: `274997d12d8950b8f3d3c28b2d1f3ce71035dbb865a912fc6b2adf5e4da1c17c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic

            Read the question again:
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `e50bfa34b2bcc26387738fd2a19b7923caddefb09db9eaf4c284762a3f4e266f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_01_mc_pointwise_qual`

### multiple_choice / default

SHA-256: `e2d3e1cdbb25cc08a76562fd1f75810af9c08766295891f10ec17f43a4f51b7b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Test each option's complete claim independently against the supplied evidence, then choose the best-supported option by its text. Explain which observed pattern and continuation or recovery support it and which part of the strongest alternative is absent. Use qualitative evidence; do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_02_mc_semantic_qual`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_03_mc_salient_aftermath`

### multiple_choice / default

SHA-256: `5643e4407a55395cf1e13b46af951b4e4fed3ca27133b43fb723db5575934c28`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Match the options to the most salient local change and what happens immediately afterward. Select the option whose complete description matches both, then explain the local change and its continuation or recovery using qualitative evidence. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_04_tf_re2_verdict`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `ecd906c41dad15c8cee77b1644fc8927c7a9993c257a8f8eeca6e0fd6ac9629f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_05_tf_re2_qual_verdict`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `52ee71d54d97723c619bf6baecb1bbd2f012fa12f4b47d44c104244fbae888b7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_06_joint_pointwise_tf_qual`

### multiple_choice / default

SHA-256: `e2d3e1cdbb25cc08a76562fd1f75810af9c08766295891f10ec17f43a4f51b7b`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Test each option's complete claim independently against the supplied evidence, then choose the best-supported option by its text. Explain which observed pattern and continuation or recovery support it and which part of the strongest alternative is absent. Use qualitative evidence; do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `52ee71d54d97723c619bf6baecb1bbd2f012fa12f4b47d44c104244fbae888b7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r3_07_joint_semantic_tf_qual`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `52ee71d54d97723c619bf6baecb1bbd2f012fa12f4b47d44c104244fbae888b7`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window contains an anomalous deviation.

            Read the question again:
            True or False: The window contains an anomalous deviation.
```

## `lit_r4_01_tf_neg_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

## `lit_r4_02_joint_semantic_tf_neg_re2`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

## `lit_r4_03_tf_nonanomaly_re2`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

### true_false / normality_word

SHA-256: `40cba28623f4ea577cb2b16706b81c44d635f50e05eb21a345f4e4306bec3b00`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window shows stable normal behavior.

            Read the question again:
            True or False: The window shows stable normal behavior.
```

## `lit_r4_04_joint_semantic_tf_nonanomaly_re2`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

### true_false / normality_word

SHA-256: `40cba28623f4ea577cb2b16706b81c44d635f50e05eb21a345f4e4306bec3b00`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: The window shows stable normal behavior.

            Read the question again:
            True or False: The window shows stable normal behavior.
```

## `lit_r4_05_tf_neg_prefix`

### multiple_choice / default

SHA-256: `e5e6454eb69a27dd04bb960c9b2c276f354973330f9225149521ae472dd369a6`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `7cfdc00dd06d34be8a8d7cd577003f16da543871fb049a28d934f2e0fa9be71a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Preserve the proposition's polarity exactly as written. Decide whether the complete statement is supported, and begin with exactly "Answer: True." or "Answer: False." Then give a qualitative reason consistent with that label. Do not quote exact values or step numbers unless the question asks for them.

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r4_06_joint_semantic_tf_neg_prefix`

### multiple_choice / default

SHA-256: `e5bb143cb12cde0a99fc5008b78a1160b36a6b11ede6e14fa0dfd252a35f0a5c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Choose the option by its complete text before mapping it to A, B, C, or D. Explain the observed qualitative pattern and its persistence or recovery that distinguish the selected option from the strongest alternative. Do not quote exact values or step numbers unless the question explicitly asks for them.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `7cfdc00dd06d34be8a8d7cd577003f16da543871fb049a28d934f2e0fa9be71a`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Preserve the proposition's polarity exactly as written. Decide whether the complete statement is supported, and begin with exactly "Answer: True." or "Answer: False." Then give a qualitative reason consistent with that label. Do not quote exact values or step numbers unless the question asks for them.

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_01_mc_structured_guard`

### multiple_choice / default

SHA-256: `bb5c687deb0a81584c681aba304103a61dc2fbda4b4f57b85f22d3f79e606ab1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Compare every option by its complete meaning. Treat alternating signs, ordinary peaks or troughs, isolated large or small values, and irregular-looking fluctuation as normal variability unless the supplied evidence supports the option's specific structured anomaly signature, such as a localized contrast, persistence, recovery, or boundary pattern. Select an anomalous option only when that defining signature is supported; otherwise select the normal option. Explain the decisive qualitative pattern without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `f352829131025c30cf5824a43d9cf510a9229cd3f2a3022e41376120fa26f4c0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_02_mc_status_then_shape`

### multiple_choice / default

SHA-256: `36ec916e9cb9004c6f91c7601978c35839103319a490740a3c4e24ff817f9eca`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `f352829131025c30cf5824a43d9cf510a9229cd3f2a3022e41376120fa26f4c0`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_03_joint_structured_tf_neg_re2`

### multiple_choice / default

SHA-256: `bb5c687deb0a81584c681aba304103a61dc2fbda4b4f57b85f22d3f79e606ab1`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Compare every option by its complete meaning. Treat alternating signs, ordinary peaks or troughs, isolated large or small values, and irregular-looking fluctuation as normal variability unless the supplied evidence supports the option's specific structured anomaly signature, such as a localized contrast, persistence, recovery, or boundary pattern. Select an anomalous option only when that defining signature is supported; otherwise select the normal option. Explain the decisive qualitative pattern without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```

## `lit_r5_04_joint_status_tf_neg_re2`

### multiple_choice / default

SHA-256: `36ec916e9cb9004c6f91c7601978c35839103319a490740a3c4e24ff817f9eca`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.

            ### Question
            Which description best matches the window?
A) Stable
B) Spike
C) Trend
D) Cyclic
```

### open_ended / default

SHA-256: `2839337c01b253c2b75f5e0d48b885f9470ba9b7794c7b4e62e669165cc7436c`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            What evidence supports or refutes an anomaly in this window?
```

### true_false / default

SHA-256: `653ca40c2f82fb18a791765e7bcca91ef4494426daa42c0dc72bda9237f99bdb`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Question
            True or False: The window contains an anomalous deviation.
```

### true_false / explicit_negative

SHA-256: `338b25c325009fbea04d0547df5760bc1b3bf8720d4bc67fa0e2a4f42d09204f`

```text

            You are an expert time series analyst. Analyze the provided data and answer the question.

            ### Time Series Data
            - **Window:** Steps 7 to 10
            - **Values (scaled by 100):** 10, 11, 42

            ### Contextual Hints
            - **Per-Step Analysis:** <|local_hint|><|local_hint|><|local_hint|>
            - **Overall Summary Hints:** <|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|><|fixed_hint|>

            ### Answering Rule
            Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.

            ### Question
            True or False: There is no evidence of an anomaly.

            Read the question again:
            True or False: There is no evidence of an anomaly.
```
