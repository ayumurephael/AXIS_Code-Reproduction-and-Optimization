# AXIS 作者兼容训练与公平比较协议

## 结论

3-epoch Phase-II checkpoint 是低预算回归基线，不是训练充分的正式基线。其完整 validation loss 在 epoch 1/2/3 持续下降；作者候选 epoch-33 在相同 Gemini 协议下显著优于本仓库 epoch-3，并能在作者兼容生成下接近论文。

这支持增加训练预算，但不能反推论文一定训练了 33 或 35 epoch。最大 35 epoch 仅用于覆盖已观察到的候选状态，最终 checkpoint 必须由验证集选择。

## 作者接近型基线

`main` 推荐三卡命令：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_memory_safe_v3 \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/phase2_author_seed42 \
  --epochs 35 \
  --seed 42 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --num-workers 2 \
  --save-every 5000
```

已有三轮训练在 3×A100 40GB 上约 5.7 小时。按线性比例估算 35 epoch 约 66 小时；实际时间取决于节点、I/O 和验证频率，提交前应做短时 benchmark。

## 预先声明的选模日程

建议在 epoch `3, 8, 15, 20, 25, 30, 33, 35` 上运行同一完整验证集。若训练成本允许，可每 epoch 验证，但三分支必须使用完全相同日程。

唯一主选模指标是完整 1500-series / 3000-QA validation mean teacher-forced loss。同步报告 MC/OE/TF 分层 loss 和免费代理指标，用于发现题型退化，但不得用 paper140、DeepSeek 或 Gemini test score 选 checkpoint。

若最佳 validation loss 位于 epoch 35：

- 报告尚未观察到收敛；
- 在三分支共同扩大上限前，不单独给某一分支追加预算；
- 不能因为某个较早 checkpoint 更接近论文表格就选择它。

## 首轮保持不变的参数

- 同一 backbone、Phase-I encoder checkpoint 和初始化逻辑；
- AdamW、LR `1e-4`、weight decay `1e-5`；
- 每卡 1 series、三卡 global batch 3、gradient accumulation 1；
- BF16、同一数据版本和 manifest；
- beam 5、`max_new_tokens=1000`；
- 正式测试 `paper140 + batching=series + skip-loss`；
- 同一 judge prompt、model、endpoint 语义和评分脚本 commit。

不要在一次实验中同时改变学习率、batch、seed、epoch 上限和初始化；否则无法归因。

## Seed

- 作者接近型单次基线使用 seed 42，因为作者补充数据路径更接近该随机状态。
- 三分支公平比较必须直接复用同一 seed-42 split manifest。
- 稳健性结论至少使用三个预先声明 seed，并包含 42 与 72；报告 paired mean、分题型结果和置信区间。
- 禁止观察 test/judge 分数后选择 seed。

单次 seed-42 结果用于论文接近性检查；多 seed 结果用于方法改进的稳健性主张，两种目标要区分。

## 跨分支对照

`main`、`loss_resesign`、`architecture_redesign`、`architecture_redesign_fixedhint_frozen` 与 `loss_final` 共同固定 Phase-I SHA-256、数据、manifest、seed、global batch、优化器、训练上限、验证日程、选模规则、paper140、生成协议和 judge。每个改进分支只能改变 `BRANCH_PROFILE.md` 声明的因素。

推荐顺序：

1. 完成 `main` seed-42 长预算训练和验证选模；
2. 冻结所有公共配置及哈希；
3. 对各改进分支复用相同训练预算；
4. 用免费验证指标筛查实现错误和明显退化；
5. 对每个分支验证选出的唯一 checkpoint 生成 paper140；
6. 日常/预筛使用 DeepSeek，最终一次使用 Gemini；
7. 报告逐题 paired bootstrap，并保留分题型结果。

不能拿改进分支的 3-epoch 结果与 `main` 的 35-epoch 结果比较。

## 验收层次

1. 训练过程：loss、非有限值、梯度、参数更新范围、checkpoint 完整性；
2. 每个候选：完整 validation loss 和分题型 loss；
3. 唯一候选：forced-choice、数值/文本代理和 evidence intervention；
4. 正式生成：paper140、series batching、140/140 审计；
5. 日常评测：同一 DeepSeek 配置；
6. 最终评测：Gemini author mode、335/335 审计；
7. 方法结论：paired 统计、生成质量和 evidence dependence 同时报告。

作者候选 checkpoint 可用于验证加载、推理和评测协议，但不能作为任一实验分支的训练结果参与公平比较。

## Loss-final initialization policy

`loss_final` must use the author `loss_only` architecture. Any earlier checkpoint whose metadata says `architecture.variant=full` contains QK-Norm, residual-bypass, and direct-task-prompt parameters and is not a valid initializer after the architecture restoration.

Two different research questions require two different matched controls:

1. Primary objective comparison: initialize both answer-only and loss-final from the same Phase-I checkpoint; keep data, total optimization budget, and validation selection identical.
2. Low-cost post-training add-on: initialize both arms from the same author Phase-II best checkpoint, reset the optimizer in both arms, and compare answer-only continuation against answer-plus-auxiliary continuation for the same extra budget.

Comparing author best before training with loss-final after additional training is not a controlled loss ablation. Continuing only the treatment arm confounds auxiliary loss with optimizer reset and extra answer-NLL updates.
