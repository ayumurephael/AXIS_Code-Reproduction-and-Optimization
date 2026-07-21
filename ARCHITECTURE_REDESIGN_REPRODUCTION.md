# AXIS Fixed-Hint 冻结架构分支复现协议

本文件是 `architecture_redesign_fixedhint_frozen` 的分支专用执行协议；与共享 `REPRODUCTION.md` 冲突时，以本文件对本轮实验的明确覆盖项为准。设计说明不是可直接执行的命令。

## 1. 已冻结的正式实验身份

本轮只训练和评测完整架构，不进行消融；启动后不得根据测试或 Judge 分数回改配置。

| 项目 | 固定值 |
|---|---|
| architecture variant | `full` |
| seed / epochs | 72 / 6 |
| world size / micro batch / global batch | 3 / 每卡 1 series / 3 series |
| GPU | 3 张 H100，当前物理卡 2、3、4 |
| Phase-I SHA-256 | `2f1507fcc3c3232d375dcb0c18bfcd11f2ec9e184dcb1cc9fb603f0f38c94a2e` |
| counterfactual index | objective-v3、schema v3、Top-M=1 |
| index SHA-256 | `cc0944f9e5bcd70b6b72fdfb398abf8ef439734ed5aa7693ddda337bee1b38d0` |
| optimizer | AdamW |
| local / attention / task-prompt LR | `5e-5` / `1e-4` / `5e-5` |
| weight decay / gradient clip | `1e-5` / `1.0` |
| state-loss beta | 前 10% synchronized steps从 0 线性升至 `0.1` |
| precision | BF16 autocast |
| checkpoint schedule | epoch 1–6 全部候选 |
| selection metric | 完整 1500-series / 3000-QA validation 的 mean answer NLL 最小 |
| formal generation | `paper140 + batching=series + skip-loss` |
| formal Judge | Gemini 2.5 Pro，author prompt + author scoring |

历史 schema-v1 索引即使 seed 与 Phase-I hash 相同也不得用于本轮训练。索引、数据 split、训练 seed 必须同时为 72。虽然服务器还有更多空闲 H100，但本轮保持 world size 3 和 global batch 3，以免改变优化轨迹；其余 GPU 只用于不改变模型状态的验证/推理并行任务。

## 2. Fixed/task-prompt 梯度契约

完整架构包含 QK-Norm、continuous bypass + prototype sidecar，以及直接学习的 30-token task prompt。LLM 和 Phase-I encoder 冻结。

- `L_state` 必须通过 `detach_fixed_hint=True` 把 task-prompt 输出视为常量，不得更新 `task_prompt_embeddings`。
- `L_state` 仍更新 mapping/local attention、continuous projection、gate/fusion 等局部证据模块。
- `L_answer` 使用非 detach task-prompt 路径，继续更新 `task_prompt_embeddings`。
- checkpoint 必须记录 `fixed_hint_state_gradient=false`、`fixed_hint_answer_gradient=true`、`fixed_hint_freeze_scope=state_loss_only`、`fixed_hint_state_isolation=direct_task_prompt_output_stop_gradient`。
- objective audit 对这些字段 fail closed。CPU 回归同时覆盖 state-only 和 answer-only 的实际 hint-injection 梯度路径。

## 3. 输入资产审计

```bash
RUN_ROOT=experiments/reproduction/architecture_redesign_fixedhint_frozen_seed72
INDEX=$RUN_ROOT/index/counterfactual_index.json
PHASE1=experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth

sha256sum "$PHASE1" "$INDEX"
python -m tools.axis_repro.audit_counterfactual_index \
  --data data/anomaly_llava_training_dataset --index "$INDEX" \
  --seed 72 --train-ratio 0.95 --expected-donor-top-m 1 \
  --output "$RUN_ROOT/index/counterfactual_index_audit.json"
```

索引审计必须得到 `version=3`、57,000 records、28,500 train series、`ok=true`，且索引中的 Phase-I hash 与实算值一致。复用上述精确索引，不重新抽 donor。

## 4. 三卡 smoke

```bash
export CUDA_VISIBLE_DEVICES=2,3,4
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_architecture_redesign \
  --phase1 "$PHASE1" --counterfactual-index "$INDEX" \
  --data data/anomaly_llava_training_dataset \
  --output "$RUN_ROOT/smoke" --architecture-variant full \
  --epochs 1 --max-steps 2 --save-every 0 --seed 72 \
  --local-lr 5e-5 --attention-lr 1e-4 --prompt-lr 5e-5 \
  --weight-decay 1e-5 --gradient-clip 1.0 \
  --beta 0.1 --beta-warmup-ratio 0.1 --num-workers 2
```

smoke 必须确认三 rank 同步、loss/gradient 有限、BF16 可用、索引和 Phase-I hash 一致。smoke 产物不能参与选模或评分。

## 5. 正式 6-epoch Phase-II

```bash
export CUDA_VISIBLE_DEVICES=2,3,4
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_architecture_redesign \
  --phase1 "$PHASE1" --counterfactual-index "$INDEX" \
  --data data/anomaly_llava_training_dataset \
  --output "$RUN_ROOT/formal" --architecture-variant full \
  --epochs 6 --seed 72 \
  --local-lr 5e-5 --attention-lr 1e-4 --prompt-lr 5e-5 \
  --weight-decay 1e-5 --gradient-clip 1.0 \
  --beta 0.1 --beta-warmup-ratio 0.1 \
  --num-workers 2 --save-every 5000
```

不传 `--qk-norm-seq-len`；程序使用固定 training split 的 Local 长度 p97.5 初始化并写入 checkpoint。保存日志、runtime、每轮 inference checkpoint、周期 checkpoint、commit、环境和 GPU 身份。正式训练启动后以稳定窗口的实测 synchronized steps/s 计算 ETA。

## 6. 双重审计、逐轮 validation 与选模

对 epoch 1–6 的每个 inference checkpoint 先执行：

```bash
python -m tools.axis_repro.audit_architecture_checkpoint \
  --checkpoint <epoch_N_inference.pth> --expected-variant full
python -m tools.axis_repro.audit_loss_objective_checkpoint \
  --checkpoint <epoch_N_inference.pth> --expected-donor-top-m 1
```

从 checkpoint metadata 读取精确 `qk_norm_seq_len`，再运行：

```bash
export CUDA_VISIBLE_DEVICES=2,3,4
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint <epoch_N_inference.pth> \
  --data data/anomaly_llava_training_dataset --subset full \
  --series-split-manifest experiments/reproduction/manifests/phase2_split.json \
  --series-split-key val_series --modes base \
  --architecture-variant full \
  --qk-norm-seq-len <checkpoint metadata integer> --gate-bias -2 \
  --output "$RUN_ROOT/validation/epoch_N"

python -m tools.axis_repro.audit_validation \
  --predictions "$RUN_ROOT/validation/epoch_N/predictions.jsonl" \
  --series-manifest experiments/reproduction/manifests/phase2_split.json \
  --series-keys val_series
```

每轮必须为 3,000 rows、1,500 series、每 series 2 QA、无错误且 `ok=true`。将六个 predictions 文件一次性传给 `select_best_loss`，只按 mean answer NLL 的 argmin 选择唯一 checkpoint。state accuracy、测试输出和 Judge 分数只作事后诊断。

## 7. paper140 正式生成与 Gemini 评分

仅对 validation 选出的 checkpoint：

```bash
export CUDA_VISIBLE_DEVICES=2,3,4
torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint <selected_inference_checkpoint.pth> \
  --data data/AXIS_qa_test --subset paper140 --modes base \
  --batching series --skip-loss \
  --architecture-variant full \
  --qk-norm-seq-len <checkpoint metadata integer> --gate-bias -2 \
  --output "$RUN_ROOT/test_selected_paper140"

python -m tools.axis_repro.audit_results \
  --predictions "$RUN_ROOT/test_selected_paper140/predictions.jsonl" \
  --manifest experiments/reproduction/manifests/paper140.json --modes base
```

预测审计通过后才可调用正式 Judge：

```bash
python -m tools.axis_repro.geval_gemini \
  --predictions "$RUN_ROOT/test_selected_paper140/predictions.jsonl" \
  --output "$RUN_ROOT/test_selected_paper140/geval_gemini25pro_author.jsonl" \
  --model gemini-2.5-pro --prompt-template author --scoring-mode author \
  --primary-workers 3
```

PackyAPI 没有可用 score-token logprobs 时，author mode 按作者代码的可执行回退语义使用 temperature=0 单次整数分。必须保留 `logprobs_requested`、`logprobs_returned`、`method`、usage、prompt hash、模型和 endpoint host；不得混入 DeepSeek 分数。

最终 `audit_results` 与 `table_runner` 必须证明 140 predictions、335 dimension scores、无重复/缺失/非法值、`ok=true`。

## 8. 报告与停止规则

报告必须包含 Table 1 十项 Gemini 指标及 Baseline 差值/相对变化、六轮训练与 validation、选模依据、answer/state 指标、valid-pair 覆盖、梯度与吞吐、全部超参数和参数量、QK/gate 配置、环境、所有关键 SHA-256、全部审计以及 Gemini logprobs/fallback/usage。

若最佳 validation 点落在 epoch 6，必须报告“在当前预算上限仍可能未收敛”；不得借测试或 Gemini 分数追加定向调参。
