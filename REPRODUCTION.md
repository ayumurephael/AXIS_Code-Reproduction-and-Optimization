# AXIS 端到端复现协议

本文是当前有效分支共同遵守的执行规范。命令默认从仓库根目录运行；训练与模型推理只在已授权的 CUDA GPU 服务器执行，推荐三卡。`<...>` 表示必须由运行者填写的外部路径。

## 1. 固定实验身份

每次运行先记录：

```bash
git branch --show-current
git rev-parse HEAD
python --version
python -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"
nvidia-smi
```

已验证的软件基线为 Python 3.10、PyTorch 2.5.1+cu124、Transformers 4.45.2、NumPy 1.26.4。安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

基座模型可用环境变量指向服务器本地镜像。密钥与 SSH 信息不得进入仓库。

## 2. 外部资产

仓库期望以下布局：

```text
data/
├── anomaly_llava_training_dataset/
└── AXIS_qa_test/
experiments/checkpoints/pretrain_single/
└── pretrain_checkpoint_best.pth
```

在运行前记录数据版本、Phase-I checkpoint 的绝对来源和 SHA-256。三分支必须使用同一个 Phase-I 文件。

## 3. 固定 manifest

作者接近型复现使用 seed 42，并将 manifest 写入独立目录：

```bash
python -m tools.axis_repro.make_manifests \
  --test-data data/AXIS_qa_test \
  --train-data data/anomaly_llava_training_dataset \
  --seed 42 \
  --output experiments/reproduction/manifests_author42
```

验收数量：`paper140=140`、`full=284`、`train_series=28500`、`val_series=1500`。生成后冻结文件和哈希；三分支直接复制同一 manifest，不得各自重新随机生成。

历史 seed-72 manifest 只用于复核已存在的三轮结果，不能与 seed-42 长预算结果混为同一次实验。

## 4. Phase-II 训练

先阅读 `BRANCH_PROFILE.md`，使用当前分支指定的训练入口。`main` 的作者接近型命令为：

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

固定项：AdamW、LR `1e-4`、weight decay `1e-5`、每卡 micro-batch 1、三卡 global batch 3、BF16、冻结 Phase-I encoder 与 LLM。35 epoch 是覆盖已发现 epoch-33 作者候选状态的上限，不代表论文明确声明了 35 epoch。

训练不得因测试分数提前停止。保存 runtime、每轮 checkpoint、训练日志、commit、manifest hash 和 Phase-I hash。

## 5. 完整验证集选模

预先声明验证 checkpoint 日程，例如 epoch 3、8、15、20、25、30、33、35。每个候选在同一 1500-series / 3000-QA 验证集上运行 teacher-forced loss；验证推理使用默认 per-record 路径且不能加 `--skip-loss`：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint <candidate_inference_checkpoint.pth> \
  --data data/anomaly_llava_training_dataset \
  --subset full \
  --series-split-manifest experiments/reproduction/manifests_author42/phase2_split.json \
  --series-split-key val_series \
  --modes base \
  --output <validation_output_dir>

python -m tools.axis_repro.audit_validation \
  --predictions <validation_output_dir>/predictions.jsonl \
  --series-manifest experiments/reproduction/manifests_author42/phase2_split.json \
  --series-keys val_series
```

每个 checkpoint 必须得到 3000 rows、1500 series、每 series 2 QA、`ok=true`。用全部候选文件选择 mean validation loss 最低者：

```bash
python -m tools.axis_repro.select_best_loss \
  <val_epoch_3>/predictions.jsonl \
  <val_epoch_8>/predictions.jsonl \
  <other_predeclared_candidates...>/predictions.jsonl \
  --output <training_dir>/best_validation_loss.json
```

若最佳点仍位于训练上限，应扩大预算或将“未收敛”作为限制报告；不能直接按最接近论文的 epoch 选择。

## 6. 作者兼容正式生成

只对验证集选出的唯一 checkpoint 运行测试：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint <validation_selected_checkpoint.pth> \
  --data data/AXIS_qa_test \
  --subset paper140 \
  --modes base \
  --batching series \
  --skip-loss \
  --output <prediction_dir>
```

`batching=series` 对齐作者 `AXIS_test.py`：同一时间序列的 QA 组成 batch 后 beam search。正式测试不计算 teacher-forced loss。`per_record` 只用于验证选模、兼容性诊断或复核旧结果。

立即审计：

```bash
python -m tools.axis_repro.audit_results \
  --predictions <prediction_dir>/predictions.jsonl \
  --manifest experiments/reproduction/manifests_author42/paper140.json \
  --modes base
```

预期 140 条唯一、非空的 base prediction，且 `ok=true`。

## 7. 评分

日常默认 DeepSeek，正式使用 Gemini。完整参数、endpoint 与无 logprobs 处理见 `EVALUATION_PROTOCOL.md`。

日常：

```bash
export DEEPSEEK_API_KEY='<set outside repository>'
python -m tools.axis_repro.geval_resilient \
  --predictions <prediction_dir>/predictions.jsonl \
  --output <score_dir>/geval_deepseek_v4_pro.jsonl \
  --model deepseek-v4-pro --max-tokens 4096
```

正式：

```bash
export GEMINI_API_KEY='<set outside repository>'
python -m tools.axis_repro.geval_gemini \
  --predictions <prediction_dir>/predictions.jsonl \
  --output <score_dir>/geval_gemini25pro_author.jsonl \
  --model gemini-2.5-pro \
  --prompt-template author \
  --scoring-mode author
```

两个输出必须分目录或明确命名，不能覆盖、合并或跨 judge 计算差值后声称为同一评价量尺。

## 8. 最终审计与 Table 1

```bash
python -m tools.axis_repro.audit_results \
  --predictions <prediction_dir>/predictions.jsonl \
  --scores <score_dir>/geval_gemini25pro_author.jsonl \
  --manifest experiments/reproduction/manifests_author42/paper140.json \
  --modes base

python -m tools.axis_repro.table_runner \
  --scores <score_dir>/geval_gemini25pro_author.jsonl \
  --output-prefix <score_dir>/table1
```

正式完整度是 140 predictions、335 dimension scores、无重复/缺失/非法值、`ok=true`。报告必须附：checkpoint 与 prediction/score 的 SHA-256、branch/commit、环境、manifest、选模文件、judge model、prompt/scoring mode、logprobs 是否返回、审计 JSON。

## 9. 三分支公平比较

同一批对照必须固定：

- Phase-I checkpoint 和数据版本；
- manifest、seed 和 train/validation split；
- global batch、优化器、LR、weight decay、precision；
- 最大 epoch、验证 checkpoint 日程和选模指标；
- beam、最大生成长度、series batching；
- paper140 manifest；
- judge、prompt、endpoint 语义和评分脚本 commit。

唯一可变项是 `BRANCH_PROFILE.md` 声明的架构或损失因素。不能比较一个分支的 3-epoch checkpoint 与另一个分支的 35-epoch checkpoint，也不能用 Gemini 测试分数回头调整超参数。

## 10. `loss_final` architecture gate

Before any new `loss_final` run, verify the branch profile and CLI default:

```bash
python -m tools.axis_repro.train_phase2_loss_final --help
```

The only permitted architecture is `--architecture-variant loss_only`; omit QK length. The trainer fails before CUDA initialization if a redesign variant is requested. The corresponding checkpoint audit must confirm all of the following:

- `variant=loss_only`;
- `qk_norm=false`;
- `continuous_bypass=false`;
- `direct_task_prompt=false`;
- 30 learned Fixed queries routed through shared prototype cross-attention;
- exactly the nine author-compatible Perceiver tensors.

Do not resume checkpoints from the aborted full-architecture Answer-Only or Joint runs. Restart both matched arms from the predeclared common initializer described in `AUTHOR_COMPATIBLE_TRAINING.md`.
