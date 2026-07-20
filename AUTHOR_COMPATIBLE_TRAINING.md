# AXIS 作者兼容训练与公平架构比较协议

## 结论先行

当前默认的 3-epoch Phase-II 训练应视为低预算复现基线，不应再视为已经匹配作者训练充分度的最终基线。证据是：

- epoch 1/2/3 的完整 validation loss 为 0.818750、0.773748、0.761679，第三轮仍在下降；
- 当前 epoch-3 与服务器作者候选 epoch-33 使用相同 Gemini 协议时，335 个评分维度的配对均分差为 +0.176，95% bootstrap CI 约为 [+0.033, +0.316]；
- 作者候选 epoch-33 配合 series 推理和 Gemini 得到 4.27/2.97/3.74，接近论文 4.19/3.02/3.65。

这支持“3 epoch 很可能训练不足”，但不能反推“论文一定训练了 33 epoch”。作者补充脚本的最大 epoch、实际早停、有效 batch 和验证逻辑彼此存在冲突；候选 checkpoint 的 avg_loss 也不是当前完整 validation loss，不能直接横向比较。

## 推荐的作者接近型训练

在 3×A100 40GB 上显式运行，不改变当前已经对齐的优化器参数：

    torchrun --standalone --nproc_per_node=3 \
      -m tools.axis_repro.train_phase2_memory_safe_v3 \
      --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
      --data data/anomaly_llava_training_dataset \
      --output experiments/reproduction/phase2_author_compatible_seed42 \
      --epochs 35 \
      --seed 42 \
      --lr 1e-4 \
      --weight-decay 1e-5

选择 35 只是为了覆盖已发现的 epoch-33 候选状态，并不是声称 35 是论文真实轮数。按已有 3-epoch 实测速度线性估算，35 epoch 约需 66 小时；实际提交前应重新估算。

建议对 epoch 3、8、15、20、25、30、33、35 做同一完整 validation 集的 teacher-forced loss。只有 validation loss 可以选 checkpoint；paper140、Gemini 或任何测试分数均不得参与选模。如果最佳点早于 35，应报告实际最佳 epoch，不应为了接近论文表格而选择 epoch 33。

## 哪些参数先保持不变

- backbone、Phase-I encoder 和 Hint Tuner 初始化逻辑；
- AdamW；
- learning rate 1e-4；
- weight decay 1e-5；
- 每卡 1 个 series、3 卡 global batch 3；
- BF16、同一数据版本和同一 generation 配置；
- beam=5、max_new_tokens=1000；
- 正式测试使用 series batching + skip-loss；
- Gemini 使用作者 prompt/rubric，并明确记录中转无 logprobs 时的整数回退。

在没有新的 validation 证据前，不建议同时改 learning rate、batch、seed、epoch 和初始化。那样即使结果接近，也无法判断是哪一个因素产生作用。

## Seed 的两种用途

作者补充数据集的实际默认 seed 更接近 42；原三轮复现显式使用 72。因此：

1. 作者接近型基线：使用 seed 42，目的是提高与作者运行路径的可比性；
2. 架构公平比较：baseline 和每个改进分支必须使用完全相同的 split manifest 和 seed；
3. 稳健性结论：至少运行 3 个预先声明的 seed，且应包含 42 和 72，报告 paired mean 与置信区间。

不能在看到 test/Gemini 分数后挑选 seed。

## 三分支公平比较

main、architecture_redesign 和 loss_resesign 必须共同固定：

- Phase-I checkpoint SHA-256；
- train/validation manifest；
- seed；
- 最大 epoch 与 validation checkpoint 日程；
- global batch、学习率、weight decay；
- checkpoint 选择规则；
- paper140 manifest；
- series-batched generation；
- Gemini prompt、model id、endpoint 语义和评分脚本版本。

改进分支只能改变其声明的架构或损失因素。建议先在 main 完成长预算 seed-42 基线，再用相同预算运行两个改进分支；不要拿改进分支的 3-epoch 结果与 main 的 35-epoch 结果比较。

## 验收层次

1. 每轮：训练 loss、完整 validation NLL、题型分层 NLL；
2. 候选 checkpoint：MC/TF forced-choice、OE token/numeric 指标与已有因果诊断；
3. 最终唯一 checkpoint：paper140 series generation；
4. 最后一次：Gemini 335 维完整评分；
5. 任何架构结论：逐题 paired bootstrap，并同时报告生成质量和 evidence dependence。

作者候选 checkpoint 可用于验证推理与评测协议，但不能替代从同一训练协议得到的三分支公平 checkpoint。
