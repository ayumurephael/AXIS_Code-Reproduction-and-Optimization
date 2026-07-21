# AXIS 架构改进分支复现协议

本分支在 coherent counterfactual state objective 上实现三个独立架构因素。正式 bundle 使用 `full`；因子归因必须用相同目标下的 2×2×2 变体，而不是直接把 `full` 与 `main` 的原始目标比较。

## 正式架构

1. **Prototype cross-attention QK-Norm**：Q/K 按 head 维 L2 归一化，V 不归一化；每个 attention 拥有可学习 scale。训练 split 上的 Local hint 长度统计决定初始化序列长度，并写入 checkpoint metadata。
2. **Continuous bypass + prototype sidecar**：encoder state 的连续投影作为主路径，prototype attention 作为门控残差 sidecar；gate bias 默认 `-2`。
3. **Direct task prompt**：30 个 fixed/task 位置由直接可学习的 `[1,30,d_llm]` 参数填充，不实例化旧 fixed-query prototype attention。

LLM 和 Phase-I encoder 冻结。训练目标同时包含真实上下文 answer NLL 与 factual/counterfactual state CE。

## 变体

| `--architecture-variant` | QK-Norm | bypass/sidecar | direct task prompt |
|---|---:|---:|---:|
| `loss_only` | 0 | 0 | 0 |
| `qk_only` | 1 | 0 | 0 |
| `bypass_only` | 0 | 1 | 0 |
| `task_prompt_only` | 0 | 0 | 1 |
| `qk_bypass` | 1 | 1 | 0 |
| `qk_task_prompt` | 1 | 0 | 1 |
| `bypass_task_prompt` | 0 | 1 | 1 |
| `full` | 1 | 1 | 1 |

## 1. 构造并审计 counterfactual index

必须与 `loss_resesign` 复用同一个 seed-42 index 文件和 SHA-256；不要在分支间分别生成不同 donor 集：

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

审计通过后冻结 index 与 Phase-I hash。

## 2. 多卡 smoke

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_architecture_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/shared_author42/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/architecture_redesign/full/smoke \
  --architecture-variant full \
  --epochs 1 --max-steps 2 --save-every 0 --seed 42
```

检查三 rank、finite loss/gradient、只更新声明参数、index/checkpoint metadata 一致。smoke 产物不能用于评分。

## 3. 正式长预算训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_architecture_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/shared_author42/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/architecture_redesign/full/phase2_author42 \
  --architecture-variant full \
  --epochs 35 --seed 42 \
  --local-lr 5e-5 --attention-lr 2e-5 --prompt-lr 5e-5 \
  --weight-decay 1e-5 --gradient-clip 1.0 \
  --beta 0.2 --beta-warmup-ratio 0.1 \
  --num-workers 2 --save-every 5000
```

训练时不传 `--qk-norm-seq-len`，由固定 training split 计算并写入 `reproduction_meta.architecture.qk_norm_seq_len`。

## 4. checkpoint 审计、验证和选模

每个预先声明的候选先运行：

```bash
python -m tools.axis_repro.audit_architecture_checkpoint \
  --checkpoint <checkpoint.pth> --expected-variant full

python -m tools.axis_repro.audit_loss_objective_checkpoint \
  --checkpoint <checkpoint.pth> --expected-donor-top-m 1
```

随后按 `REPRODUCTION.md` 在同一 1500-series 验证集上计算完整 loss。推理命令必须额外传入：

```text
--architecture-variant full
--qk-norm-seq-len <checkpoint metadata 中的精确整数>
--gate-bias -2
```

只按预先声明的完整 validation 指标选择 checkpoint，不能用 test/judge score。

## 5. 正式生成与评测

对验证选出的唯一 checkpoint 运行 `paper140 + batching=series + skip-loss`，并传入同一组 architecture 参数。日常使用 DeepSeek v4-pro，最终正式使用 Gemini 2.5 Pro author mode。命令、审计和 335 维完整度要求见 `REPRODUCTION.md` 与 `EVALUATION_PROTOCOL.md`。

## 因子归因

若研究问题是“架构是否有效”，至少共同训练 `loss_only` 与 `full`；严谨定位三个因素则运行预先声明的 2×2×2 设计。所有变体复用同一 index、初始化、seed、预算和选模日程。不要把 bundle 相对 `main` 的差异全部归因于某一个架构模块。