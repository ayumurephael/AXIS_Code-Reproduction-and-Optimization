# AXIS 损失函数改进分支复现协议

本分支在原始 AXIS 架构上使用 coherent counterfactual state objective。正式配置是 `architecture-variant=loss_only`；不启用架构改进分支的 QK-Norm、continuous bypass 或 direct task prompt。

## 目标

对真实上下文监督完整 answer NLL。对每个有效 factual/counterfactual pair，使用冻结 LLM 输出头中两个单 token verbalizer 的 next-token logits 计算二分类 state CE：

```text
answer_loss = full-answer teacher-forced NLL(real context)
state_loss  = 0.5 * [CE2(real, z) + CE2(counterfactual, 1-z)]
total_loss  = answer_loss + beta(step) * state_loss
```

`beta` 在同步更新的前 10% 从 0 线性增加到 0.2。无有效反事实的 anchor 保留 answer NLL，但不伪造翻转标签。

## 1. 构造并审计 counterfactual index

index 生成需要一次单 GPU encoder 前向；正式训练前完成。三个相关对照必须复用同一文件和 SHA-256：

```bash
export CUDA_VISIBLE_DEVICES=0
python -m tools.axis_repro.build_counterfactual_index \
  --data data/anomaly_llava_training_dataset \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --output experiments/reproduction/shared_author42/counterfactual_index.json \
  --seed 42 --train-ratio 0.95 --tau 0.25 \
  --donor-top-m 1 --gamma-percentile 5

python -m tools.axis_repro.audit_counterfactual_index \
  --data data/anomaly_llava_training_dataset \
  --index experiments/reproduction/shared_author42/counterfactual_index.json \
  --seed 42 --train-ratio 0.95 --expected-donor-top-m 1 \
  --output experiments/reproduction/shared_author42/counterfactual_index_audit.json
```

审计必须确认 split、Phase-I hash、donor policy、Window/Local control gap 和 patched effect 一致。审计失败时禁止训练。

## 2. 多卡 smoke

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_loss_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/shared_author42/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/loss_resesign/smoke \
  --epochs 1 --max-steps 2 --save-every 0 --seed 42
```

smoke 必须检查三 rank、finite answer/state/total loss、有效 pair 数、factual/counterfactual accuracy、梯度范围和 metadata；其 checkpoint 不参与结果。

## 3. 正式长预算训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_loss_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/shared_author42/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/loss_resesign/phase2_author42 \
  --epochs 35 --seed 42 \
  --local-lr 5e-5 --attention-lr 2e-5 --prompt-lr 5e-5 \
  --weight-decay 1e-5 --gradient-clip 1.0 \
  --beta 0.2 --beta-warmup-ratio 0.1 \
  --num-workers 2 --save-every 5000
```

35 epoch 是共同训练上限，不是固定选择点。预先声明 epoch `3,8,15,20,25,30,33,35` 等候选，并与 `main` 和架构分支保持相同验证日程。

## 4. checkpoint 审计与选模

每个候选先审计目标和 index 身份：

```bash
python -m tools.axis_repro.audit_loss_objective_checkpoint \
  --checkpoint <checkpoint.pth> \
  --expected-donor-top-m 1
```

随后按 `REPRODUCTION.md` 在固定 1500-series / 3000-QA validation 集上运行完整 teacher-forced loss，只按预先声明规则选择唯一 checkpoint。test、DeepSeek 和 Gemini 不参与选模。

## 5. 正式生成与双评测

对唯一 checkpoint 运行 `paper140 + batching=series + skip-loss`。日常对照使用 DeepSeek v4-pro；最终正式结果使用 Gemini 2.5 Pro 的 author prompt/mode。两种 scores 分开保存，完整命令见 `EVALUATION_PROTOCOL.md`。

## 公平性控制

直接与 `main` 比较时，本分支改变了 auxiliary objective 和参数组 LR，结论应称为“loss redesign recipe”效果。若要将改进归因于 state objective，预先增加同入口 `--beta 0` 的 optimizer control，并保持 index、初始化、LR groups、seed、预算、选模和生成完全相同。

`architecture_redesign` 的 `loss_only` 变体应与本分支在相同代码版本和 metadata 下等价；可用短 smoke/checkpoint 审计验证，但不能用 test score 决定采用哪个实现。