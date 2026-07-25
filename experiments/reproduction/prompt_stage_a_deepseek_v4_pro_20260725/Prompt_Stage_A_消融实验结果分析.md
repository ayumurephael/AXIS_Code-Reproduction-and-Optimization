# AXIS Prompt 阶段 A 消融实验结果与分析

## 结论摘要

本报告在作者发布 checkpoint、固定 `paper140` 清单和同一云端运行环境下比较 1 个 Baseline 与 7 个 prompt-only 条件。按三类题型 Final 的未加权宏平均，最佳条件为 `base`（Baseline），宏平均 3.6613，相对 Baseline 的绝对变化为 +0.0000。

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
| 角色重命名 + Evidence Contract | 3.9651 | 4.0000 | 3.8837 | 3.2437 | 3.1091 | 2.7964 | 3.9227 | 3.2857 | 3.1429 | 3.5000 |
| 综合 2+3+4 | 3.6322 | 3.6739 | 3.5349 | 3.1173 | 3.1272 | 2.6734 | 3.6236 | 3.5428 | 3.5238 | 3.5714 |
| EOS + Answer: + 综合 2+3+4 | 4.1004 | 4.1733 | 3.9302 | 2.8921 | 2.7134 | 2.5419 | 3.5091 | 3.5762 | 3.5476 | 3.6190 |

## 相对 Baseline 的绝对差值

| 实验 | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| EOS + Answer: | -0.1186 | -0.1395 | -0.0698 | -0.0439 | +0.0544 | -0.0609 | -0.1388 | -0.0254 | -0.0488 | +0.0097 |
| EOS + Answer: / w/o Fixed | -1.9954 | -1.9535 | -2.0930 | -1.3425 | -1.3146 | -1.1983 | -1.5432 | -2.0021 | -1.9512 | -2.0786 |
| 题型协议 | -0.2977 | -0.3953 | -0.0698 | -0.0822 | +0.3453 | -0.2372 | -0.4000 | -0.2994 | -0.3083 | -0.2859 |
| Fixed 角色重命名 | +0.0679 | +0.0471 | +0.1163 | -0.0143 | +0.0908 | +0.0118 | -0.1673 | -0.0564 | -0.1417 | +0.0714 |
| 角色重命名 + Evidence Contract | -0.2907 | -0.3256 | -0.2093 | +0.1578 | +0.2726 | +0.0264 | +0.1772 | -0.3564 | -0.4988 | -0.1429 |
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
