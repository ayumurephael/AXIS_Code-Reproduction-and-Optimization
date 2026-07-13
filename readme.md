# AXIS Baseline 复现与交接文档

> 本文档面向第一次接手代码的 AI 或开发者。除数据/凭据等明确标记为
> `[待补充]` 的外部信息外，仅依赖本文档即可安装环境、准备数据、训练 Phase II、
> 生成并审计 Table 1 的 AXIS 第一行。
>
> 所有命令默认从本目录（`AXIS/baseline_new`）在 GPU 服务器上执行。不要从仓库上级目录
> 直接运行，否则相对路径会失效。

> `baseline_new` 是从完整实验目录 `baseline` 筛选出的后续研究版：保留训练/测试数据、
> Phase-I 初始化权重、最终 epoch-3 Baseline 权重、固定 manifests、正式 Table 1 产物与当前
> 生产入口；不包含 epoch-1/2、中间 optimizer checkpoint、历史 released-checkpoint 实验、
> validation 原始输出、API smoke/partial/diagnostic 结果和日志。详细边界见
> `FILTERED_COPY.md`。

## 1. Baseline 简介

本代码库复现论文 **AXIS: Explainable Time Series Anomaly Detection with Large Language
Models** 中供后续架构研究使用的最小基线：

- 当前正式交付只复现 Table 1 第一行 `AXIS`，不运行其他论文基线，也不把 Table 2
  的历史 partial/diagnostic 文件当作结果。
- 使用作者发布的 Phase-I encoder，从头训练 Phase-II Hint Tuner；冻结 encoder 与 LLM。
- 三个 epoch 都在同一完整验证集推理，以全量 teacher-forced validation loss 选择 epoch 3。
- `paper140` 复刻作者 `AXIS_test.py` 的实际覆盖方式（70 series、140 QA），用于正式 Table 1。
- `full284` 覆盖作者发布测试集全部 142 series、284 QA，用于覆盖率审计和后续诊断。
- 核心推理入口仍支持四路推理时消融，供研究使用，但 Table 2 不在本轮正式结果范围。

### 1.1 能做到什么

- 可重复构造 seed=72 的 95/5 series-level Phase-II train/validation split。
- 可在 3 卡上完成 Phase-II DDP 训练；只训练 Hint Tuner，冻结时序 encoder 与 LLM。
- 可用本项目最终 epoch-3 checkpoint 做逐 QA、beam-5、可断点续跑推理。
- 可做三路输入的推理时消融，并输出每条 QA 的 response、loss、模式与 checkpoint hash。
- 可按论文 Appendix E.2/E.4 用 `deepseek-v4-pro` 做分维度 G-Eval。
- 可审计 prediction/score 的覆盖率、重复项、空输出和 20-sample fallback 完整性。
- 可生成 Table 1 AXIS 行、Table 2 四行和 paired bootstrap CI。

### 1.2 不能严格宣称什么

- 论文 Table 1/2 使用 **Gemini-2.5** judge；本项目按需求使用
  **DeepSeek-v4-pro**。因此最终数值是“模型复现 + 替代 judge 复现”，不能仅凭与论文数值
  不同就判断模型未复现。
- 作者未发布其完整 Phase-II training main、G-Eval 实现、human evaluation、所有 baseline
  runner 和完整采样清单。本仓库补齐了这些流程，但不能证明内部实现与作者私有代码逐行一致。
- 论文没有充分说明 Table 2 的模型是否分别重训。本仓库采用**同一 checkpoint 的推理时
  channel removal**，这是可审计的 controlled ablation；不要把它写成“已复现作者私有的
  ablation training protocol”。
- 本流程不重训论文 Phase I；默认加载作者发布的 Phase-I checkpoint。
- 不复现 Table 1 的 Image LLM、ChatTS、LLMAD、ChatTime、AnomLLM，也不复现人类评价。

### 1.3 当前状态（2026-07-14）

| 项目 | 状态 | 可审计证据 |
|---|---:|---|
| 新 Phase-II 训练 | 完成 | 3 epochs，28,500 steps，5.694 h |
| epoch 1/2/3 全量验证推理 | 完成 | 每个 1500 series / 3000 QA，覆盖一致 |
| checkpoint 选择 | 完成 | mean loss：0.818750 / 0.773748 / **0.761679**，选择 epoch 3 |
| epoch-3 full284/base 推理 | 完成 | 284 rows，审计 `ok=true` |
| epoch-3 paper140/base 推理 | 完成 | 140 rows，由同一 full284 run 过滤，审计通过 |
| 严格 DeepSeek G-Eval | 完成 | 335 scores；333 概率均值 + 2×20-sample fallback |
| Table 1 AXIS 第一行 | 完成 | 10 项数值与完整审计见 `REPRODUCTION_RESULTS.md` |
| Table 2 | 本轮不执行 | runner 保留；partial/diagnostic 文件不进入正式结果 |

最终数值、差值、hash 与审计见 `REPRODUCTION_RESULTS.md`；协议摘要见 `REPRODUCTION.md`。

正式 AXIS 行（DeepSeek-v4-pro judge）：

| MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.91 | 3.98 | 3.74 | 3.05 | 2.96 | 2.65 | 3.62 | 3.67 | 3.71 | 3.60 |

## 2. 与作者开源代码的关系

本目录不是带 Git 历史的 fork，而是从 `AXIS/AXIS_original_codes` **完整复制后增量补齐**。
作者模型主体仍位于 `src/models/AXIS/`，没有重新实现一套不同模型。

主要修改如下：

| 修改 | 文件/目录 | 原因 |
|---|---|---|
| 增加三路消融参数 | `src/models/AXIS/AXIS.py` | 分离 local hint、fixed/task hint、window text |
| LLM 路径可配置 | `experiments/configs/axis_config.py` | 支持 `AXIS_MODEL_NAME` 指向本地离线模型 |
| 固定依赖版本 | `requirements.txt` | 排除已观察到的版本漂移 |
| 补齐 Phase-II DDP | `tools/axis_repro/train_phase2_*.py` | 作者未提供可直接运行的完整入口 |
| 内存安全词表 loss | `train_phase2_memory_safe_v3.py` | 避免一次物化全部 sequence×vocab logits |
| 逐记录、可恢复、多卡推理 | `run_inference_cli.py` | 生成正式 Table 1/2 数据 |
| G-Eval 与断点 journal | `geval_resilient.py` | 复现 Appendix E 并抵抗 API/网络波动 |
| manifests、merge、audit、tables | `tools/axis_repro/` | 明确样本集合并 fail closed |

`experiments/configs/deepspeed_config.json` 与 `accelerate_config.yaml` 来自原路径，但本次
**正式 Phase-II 结果使用 torchrun + DDP**，不是该 Deepspeed 配置。不要误把其中
`gradient_accumulation_steps=32` 当作正式训练超参数。

## 3. 环境安装与 GPU 说明

### 3.1 已实测环境

| 用途 | GPU | 驱动 | Python | PyTorch / CUDA runtime | cuDNN | Transformers | NumPy |
|---|---|---|---|---|---|---|---|
| 正式训练、验证 tail | NVIDIA A100-PCIE-40GB；节点 4 卡，使用 3 卡 | 550.54.15 | 3.10.5 | 2.5.1+cu124 / 12.4 | 9.1.0 | 4.45.2 | 1.26.1 |
| epoch-2 推理 | NVIDIA A100 80GB PCIe；使用 2 卡 | 550.54.15 | 3.10.14 | 2.5.1+cu124 / 12.4 | 9.1.0 | 4.45.2 | 1.26.4 |
| epoch-3 推理 | NVIDIA A100 80GB PCIe；使用 2 卡 | 550.54.14 | 3.10.12 | 2.5.1+cu124 / 12.4 | 9.1.0 | 4.45.2 | 1.26.4 |

仓库固定 `numpy==1.26.4`。正式训练节点当前实际查询为 1.26.1；两者均已成功运行，
新环境统一安装 1.26.4。系统级 CUDA toolkit 与 nvcc 版本未纳入结果依赖；PyTorch wheel
自带 CUDA 12.4 runtime。

### 3.2 推荐安装步骤

```bash
conda create -n axis-baseline python=3.10 -y
conda activate axis-baseline

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt

# 仅运行测试时需要
python -m pip install pytest
```

验证环境：

```bash
python - <<'PY'
import torch, transformers, numpy
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY

python -m pytest tools/axis_repro -q
```

当前预期测试输出：

```text
......                                                                   [100%]
6 passed
```

### 3.3 基座 LLM 与 Hugging Face

默认模型为：

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

联网环境可让 Transformers 自动下载；离线服务器建议先下载并指定本地路径：

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /path/to/DeepSeek-R1-Distill-Qwen-7B

export AXIS_MODEL_NAME=/path/to/DeepSeek-R1-Distill-Qwen-7B
export HF_HOME=/path/to/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

不要把 Hugging Face token 写进代码。若必须鉴权，只通过 `HF_TOKEN` 环境变量传入。

## 4. 数据与 checkpoint 准备

### 4.1 预期目录

```text
baseline_new/
├── data/
│   ├── anomaly_llava_training_dataset/
│   │   ├── metadata.json
│   │   └── series/                 # 30,000 个 series_*.json
│   └── AXIS_qa_test/
│       ├── progress.json
│       └── series/                 # 142 个 series_*.json，284 QA
└── experiments/
    └── checkpoints/
        ├── pretrain_single/
        │   └── pretrain_checkpoint_best.pth
```

训练数据与测试数据的公开下载 URL 在现有材料中没有唯一可验证来源：
`[待补充：数据集公开仓库/下载链接及许可证]`。在当前工作区中，数据来自
`AXIS_original_codes/data/`，复制 baseline 时已经带入。

### 4.2 作者 checkpoint

作者 README 给出的下载方式为：

```bash
huggingface-cli download thu-sail-lab/TimeSemantic checkpoints.zip \
  --local-dir ./experiments
unzip ./experiments/checkpoints.zip -d ./experiments
```

当前精简版实际保留的关键 hash：

| checkpoint | SHA-256 |
|---|---|
| released Phase-I | `2f1507fcc3c3232d375dcb0c18bfcd11f2ec9e184dcb1cc9fb603f0f38c94a2e` |
| 新训练 epoch 3 inference | `8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b` |

epoch-1/2 与作者 released AXIS checkpoint 未复制；如需历史诊断，请回到未清理的 `baseline`。

### 4.3 生成固定 manifest

```bash
python -m tools.axis_repro.make_manifests
```

预期输出：

```json
{"paper140": 140, "full": 284, "train_series": 28500, "val_series": 1500}
```

生成文件位于 `experiments/reproduction/manifests/`。所有正式实验必须复用这些文件，
不要每次重新随机抽样。

## 5. 使用文档

### 5.1 正式 Phase-II 训练

正式超参数：

| 参数 | 值 |
|---|---:|
| train/val split | 95/5，series-level |
| seed | 72 |
| epochs | 3 |
| optimizer | AdamW |
| learning rate | 1e-4 |
| weight decay | 1e-5 |
| micro batch / GPU | 1 series |
| global batch | 3 series（3 GPU） |
| gradient accumulation | 1 |
| precision | bf16 autocast |
| trainable module | `axis.perceiver` / Hint Tuner |
| trainable parameters | 204,076,832 |
| frozen modules | Phase-I TS encoder、LLM |
| checkpoint interval | 5000 synchronized steps |

启动命令：

```bash
conda activate axis-baseline
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_memory_safe_v3 \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/phase2_torch251 \
  --epochs 3 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --seed 72 \
  --num-workers 2 \
  --save-every 5000
```

开始时应打印类似：

```json
{"world_size": 3, "lr": 0.0001, "weight_decay": 1e-05, "epochs": 3,
 "seed": 72, "trainable_parameters": 204076832,
 "train_series": 28500, "steps_per_rank_epoch": 9500}
```

完成后应有：

```text
experiments/reproduction/phase2_torch251/
├── epoch_1.pth
├── epoch_2.pth
├── epoch_3.pth
├── step_5000.pth ... step_25000.pth
└── runtime.json
```

移除 optimizer state，生成约 965 MB 的推理 checkpoint：

```bash
python -m tools.axis_repro.strip_checkpoints \
  experiments/reproduction/phase2_torch251/epoch_1.pth \
  experiments/reproduction/phase2_torch251/epoch_2.pth \
  experiments/reproduction/phase2_torch251/epoch_3.pth
```

### 5.2 本精简版不含作者 released checkpoint

作者 released AXIS 权重及其兼容 runner 只服务历史诊断，已从 `baseline_new` 排除。
后续模型训练与比较均使用本项目 Phase-I 初始化和新训练 checkpoint。

模式与论文名称的对应关系：

| CLI mode | Table 2 名称 | 删除内容 |
|---|---|---|
| `base` | AXIS | 无 |
| `wo_fixed_hint` | w/o-task-hint | 30 个 learned fixed/task tokens |
| `wo_local_hint` | w/o-context-hint | step-aligned local hint tokens |
| `wo_windows` | w/o-windows | prompt 中的数值 window text |

full284 四模式预期：

```text
wrote 1136 rows to .../predictions.jsonl
```

运行审计：

```bash
python -m tools.axis_repro.audit_results \
  --predictions experiments/reproduction/released_torch251_full284/predictions.jsonl \
  --manifest experiments/reproduction/manifests/full.json
```

成功必须包含：

```json
{"prediction_rows": 1136, "records": 284, "errors": [], "ok": true}
```

### 5.3 新训练 checkpoint 的全量验证

每个 epoch 都必须在同一 1500-series 验证集上只运行 `base`：

```bash
for epoch in 1 2 3; do
  export CUDA_VISIBLE_DEVICES=0,1,2
  torchrun --standalone --nproc_per_node=3 \
    -m tools.axis_repro.run_inference_cli \
    --checkpoint "experiments/reproduction/phase2_torch251/epoch_${epoch}_inference.pth" \
    --data data/anomaly_llava_training_dataset \
    --subset full \
    --series-split-manifest experiments/reproduction/manifests/phase2_split.json \
    --series-split-key val_series \
    --modes base \
    --output "experiments/reproduction/val_epoch_${epoch}"

  python -m tools.axis_repro.audit_validation \
    --predictions "experiments/reproduction/val_epoch_${epoch}/predictions.jsonl" \
    --series-manifest experiments/reproduction/manifests/phase2_split.json \
    --series-keys val_series
done
```

每个 epoch 预期：3000 rows、1500 series、每 series 2 QA、`ok=true`。

### 5.4 Appendix-E DeepSeek G-Eval

论文 judge 是 Gemini-2.5；本项目按需求使用 `deepseek-v4-pro`。正式 runner 使用：

- Appendix E.2 原维度、描述、1–5 rubric 与权重；
- Appendix E.4 完整输出格式；
- `thinking={type: enabled, level: high}`；
- temperature=0、`top_logprobs=20`；
- 只读取最终 `**Score:**` 位置的 score-token distribution；
- 若 top-20 未包含 1–5 全部候选，执行恰好 20 个 temperature=1 的有效样本；
- `max_tokens=4096`，避免高思考在输出 `**Score:**` 前被旧的 1200-token 上限截断。

所有 judge 命令在 GPU 服务器上运行。不要把 API key 写到命令、配置或仓库；凭据文件
必须位于仓库外并限制权限：

```bash
python -m tools.axis_repro.run_with_secret \
  --credential-file /secure/path/credential.md \
  python -m tools.axis_repro.geval_resilient \
  --predictions experiments/reproduction/test_epoch_3_paper140/predictions.jsonl \
  --output experiments/reproduction/test_epoch_3_paper140/geval.jsonl \
  --model deepseek-v4-pro \
  --fallback-samples 20 \
  --max-tokens 4096 \
  --primary-workers 8 \
  --fallback-workers 1
```

`geval.fallback_pending.jsonl` 是断点 journal，不是最终 score。网络中断后原命令重跑即可。
正式 paper140/base 应有 140 条 prediction 和 335 条 dimension score。

评估完成后做 fail-closed 审计：

```bash
python -m tools.axis_repro.audit_results \
  --predictions experiments/reproduction/released_torch251_full284/predictions.jsonl \
  --scores experiments/reproduction/released_torch251_full284/geval.jsonl \
  --manifest experiments/reproduction/manifests/full.json
```

### 5.5 选择最佳 epoch

三个 epoch 必须先完成同一 1500-series / 3000-QA validation inference，然后用全量
teacher-forced loss 选择 checkpoint；不要看测试集：

```bash
python -m tools.axis_repro.select_best_loss \
  experiments/reproduction/val_epoch_1/predictions.jsonl \
  experiments/reproduction/val_epoch_2/predictions.jsonl \
  experiments/reproduction/val_epoch_3/predictions.jsonl \
  --output experiments/reproduction/phase2_torch251/best_validation_loss.json
```

本次 mean loss 为 epoch 1=`0.818750`、epoch 2=`0.773748`、epoch 3=`0.761679`。
epoch 3 在 MC、OE、TF 三类上也都最低，因此选 `epoch_3_inference.pth`。验证集 G-Eval
不是 Table 1 必需步骤；历史中断的 validation G-Eval journal 只作诊断，不能进入正式表格。

### 5.6 最佳新模型 Table 1

本次最佳权重固定为 epoch 3。先在 full284 上做 per-record base 推理，再按 paper140 manifest
过滤，保证正式子集来自同一次生成：

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 \
  -m tools.axis_repro.run_inference_cli \
  --checkpoint experiments/reproduction/phase2_torch251/epoch_3_inference.pth \
  --data data/AXIS_qa_test --subset full --modes base \
  --output experiments/reproduction/test_epoch_3_full284

python -m tools.axis_repro.filter_subset \
  --predictions experiments/reproduction/test_epoch_3_full284/base_predictions.jsonl \
  --manifest experiments/reproduction/manifests/paper140.json \
  --output experiments/reproduction/test_epoch_3_paper140/predictions.jsonl \
  --modes base
```

严格 G-Eval 和审计完成后生成 Table 1：

```bash
python -m tools.axis_repro.table_runner \
  --scores experiments/reproduction/test_epoch_3_paper140/geval.jsonl \
  --output-prefix experiments/reproduction/test_epoch_3_paper140/table1
```

`table1.md` 中 `AXIS` 一行依次给出论文 Table 1 的 10 个指标。Table 2 不在本轮范围。

### 5.7 paper140

可以独立推理 `--subset paper140`；本次采用更严格的方式，从同一次 full284/base 输出按
manifest 过滤，然后只对该固定文件调用 judge：

```bash
python -m tools.axis_repro.filter_subset \
  --predictions experiments/reproduction/test_epoch_3_full284/base_predictions.jsonl \
  --manifest experiments/reproduction/manifests/paper140.json \
  --output experiments/reproduction/test_epoch_3_paper140/predictions.jsonl \
  --modes base

python -m tools.axis_repro.audit_results \
  --predictions experiments/reproduction/test_epoch_3_paper140/predictions.jsonl \
  --manifest experiments/reproduction/manifests/paper140.json \
  --modes base
```

当前正式 base-only 口径预期：140 prediction rows、335 score rows；四模式的 560/1340 只属于 Table 2 扩展口径。

## 6. 代码库架构

```text
baseline_new/
├── data/                              # 训练/测试数据
├── experiments/
│   ├── checkpoints/                   # 作者 Phase-I 初始化权重（仅保留必需文件）
│   ├── configs/                       # 模型、Accelerate、DeepSpeed 配置
│   └── reproduction/                  # manifests、checkpoint、prediction、score、table
├── src/models/AXIS/
│   ├── AXIS.py                        # AXIS + Hint Tuner + prompt/消融逻辑
│   ├── AXIS_test.py                   # 作者原测试逻辑与 collate_fn
│   ├── dataset.py                     # series-level 95/5 dataset split
│   ├── Pretrain_ts_encoder.py         # Phase-I 相关模型代码
│   └── ts_encoder_bi_bias.py          # time-series encoder
├── tools/axis_repro/
│   ├── make_manifests.py              # paper140/full284/train/val 固定清单
│   ├── train_phase2_ddp.py            # DDP 主循环
│   ├── train_phase2_memory_safe_v3.py # 正式训练入口
│   ├── strip_checkpoints.py            # 删除 optimizer state
│   ├── run_inference_cli.py            # 新训练 checkpoint 正式推理
│   ├── geval_resilient.py              # 正式 G-Eval；可断点恢复
│   ├── audit_results.py                # prediction/score 完整性审计
│   ├── audit_validation.py             # validation series 覆盖审计
│   ├── filter_subset.py                # 按 manifest 过滤
│   ├── table_runner.py                 # 聚合 Table 1/2 与 paired CI
│   └── select_best_loss.py             # 以完整验证集 loss 选择最佳 epoch
├── requirements.txt
├── REPRODUCTION.md                     # 协议摘要
└── readme.md                           # 本交接文档
```

以下历史文件未复制到 `baseline_new`，也不应用于最终表：

- `run_inference_series*.py`：series batching 会改变正式 per-record response。
- `train_phase2_memory_safe.py`、`*_v2.py`：中间调试版本。
- `geval_two_stage.py`、`geval_parallel*.py`、`geval_correct_async.py`：早期 runner；
  网络中断时不如 `geval_resilient.py` 安全。
- 任意目录/文件名含 `diagnostic`、`partial`、`slow_diagnostic` 的输出。

## 7. 已踩过的坑与常见问题

### 7.1 普通 Phase-II forward OOM

症状：在 A100 40GB 上训练时，原始 forward 为整段 hidden state 一次生成完整词表 logits，
容易出现 `CUDA out of memory` 或进程被系统终止。

解决：只使用 `train_phase2_memory_safe_v3`。它按 64-token chunk 计算 lm-head cross entropy，
启用 gradient checkpointing，并保留与原 teacher-forcing objective 相同的 loss。

### 7.2 PyTorch 2.1/2.2 与正式生成不等价

PyTorch 2.1 环境曾产生退化 response；2.2 只保留为诊断。正式结果固定 PyTorch 2.5.1、
Transformers 4.45.2。不要混合不同版本产生的 shard。

### 7.3 released checkpoint 不在精简版范围

`baseline_new` 不含作者 released AXIS 权重和兼容 runner。若要复查该历史兼容问题，
请使用未清理的 `baseline`；后续架构实验不要混用两类 checkpoint。

### 7.4 series batching 看似快，但输出改变

实测在 76 个重叠 base 样本中，series-batched 与 per-record 只有 45 个 response 完全相同，
最低文本相似度约 0.18。正式表只能使用 per-record runner；不要为提速切换 batching 语义。

### 7.5 paper140 不是 full284

作者原测试流程先按 seed=42 shuffle，跳过前 2%，再取 70 个 series，即 140 QA。
完整发布测试集是 142 series / 284 QA。报告时必须写清口径，不能把二者混称“test set”。

### 7.6 `pad_token_id` warning

可能反复看到：

```text
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
```

当前正式运行中它是 warning，不会导致空输出；最终以 `audit_results` 是否通过为准。

### 7.7 多卡 rank 与 GPU 编号

`LOCAL_RANK` 映射的是 `CUDA_VISIBLE_DEVICES` 内部编号。例如：

```bash
CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 ...
```

两个进程看到的本地设备仍是 0 和 1。不要在代码中把它们硬编码回物理 0 和 2。

### 7.8 API 网络波动、重复请求与 parser 错误

早期 runner 会在 fallback 阶段中断后重做 primary；早期 parser 也可能漏掉
`**Score:** 4`。正式 `geval_resilient.py` 已：

- 先把 fallback primary response 写入 journal；
- 逐任务重试 URL/timeout/API 异常；
- 只解析最终 score 位置；
- 支持 markdown score 格式；
- 审计每个 fallback 是否恰好有 20 个有效 score。

DeepSeek 当前 endpoint 对 `n=2` 返回 HTTP 400，因此 20-sample fallback 必须是 20 个独立
请求，不能用一次请求返回多个 completion 来缩短往返。

### 7.9 G-Eval prompt 必须是 Appendix E.4 完整格式

只写 rubric、但省略 `Step-by-step Analysis`、`Comparison with Expected Answer`、
`Final Assessment` 三段，不能视为精确 prompt 复现。此类旧 score 已改名为
`*_nonexact_prompt_diagnostic.jsonl`，禁止进入最终表。

### 7.10 离线服务器找不到模型

设置：

```bash
export AXIS_MODEL_NAME=/absolute/local/model/path
export HF_HOME=/absolute/cache/path
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

若仍报 `No such file or directory`，先确认模型目录包含 config、tokenizer 和全部 shard。

### 7.11 JSON/文本编码

所有新增 JSON/JSONL/Markdown 使用 UTF-8。源数据中已有的 mojibake 不应在复现阶段静默
“修复”，否则 expected answer 改变会污染 G-Eval。需要清洗时应建立新数据版本与 manifest。

## 8. 时长与资源估算

### 8.1 已测训练

正式训练：3×A100-PCIE-40GB，batch=1/GPU，global batch=3，3 epochs。

| 指标 | 实测 |
|---|---:|
| synchronized steps | 28,500 |
| wall time | 20,497.7 s |
| 总时长 | 5.694 h |
| throughput | 1.3907 step/s/rank |
| global series throughput | 4.171 series/s |

训练峰值显存没有在原始 run 中写入日志：`[待补充：用 nvidia-smi/DCGM 重新记录峰值]`。
已知 3×40GB 能稳定完成正式配置。

### 8.2 已测/观测推理

生成是 per-record、beam=5、`max_new_tokens=1000`，batch=1/GPU，耗时强烈依赖输出长度。

| 任务 | GPU | 观测/估计 |
|---|---|---:|
| 1000 validation series / 2000 base QA | 3×A100 40GB | 约 53–60 min |
| 500 validation series / 1000 base QA | 3×A100 40GB | 约 30–35 min |
| 完整 1500 series / 3000 base QA | 3×A100 40GB | 约 1.4–1.6 h |
| full284 四模式 / 1136 generations | 2×A100 80GB | 当前约 3–4 h，`[待本轮完成后回填]` |

四模式测试比 3000 条 base 验证更慢的原因是 `w/o-task-hint` 经常生成到 1000-token 上限；
beam=5 还意味着每个 decoding step 同时维护 5 条候选。不要只按 QA 数线性估时。

推理峰值显存同样未单独落盘：`[待补充：按 base 与 w/o-task-hint 分别记录]`。

### 8.3 G-Eval 时长

正式 G-Eval 在 GPU 服务器进程中执行；本地只提供 loopback-only SSH proxy 和文件同步。
paper140/base 共 335 个 dimension：本次 8 个 primary worker 完成 333 个概率均值，2 个维度
各执行 20 次有效采样，活跃阶段约 35 分钟。单个 fallback 明显慢于普通 primary，因为内部需
分批完成 20 个 high-thinking 请求。不要对三个完整 validation epoch 反复做 judge 评分；本项目
用全量 validation loss 选 checkpoint，仅对最终 paper140 运行一次严格 G-Eval。

## 9. 注意事项与交接检查单

- 固定 seed：训练/验证 split 使用 72；paper140 的作者测试抽样使用 42。
- 训练 split 必须按 series，不可把同一 series 的 QA 拆到 train/val 两边。
- 正式推理必须 per-record；不要使用 series-batched 诊断脚本。
- 正式生成固定 beam=5、`max_new_tokens=1000`，不要为提速偷偷缩短。
- checkpoint 选择只看完整 validation teacher-forced loss；test 与 Table 1 judge 分数不参与选 epoch。
- 所有输出先审计，再建表；审计失败时不得手工补数。
- released 与 newly-trained 结果分目录，不能混合 score JSONL。
- DeepSeek 与 Gemini 的 judge 差异必须在论文/报告中显式说明。
- API key、SSH 密码、HF token 不得写入仓库、shell history、manifest 或日志。
- `rank*.jsonl` 是 append-only shard；重跑会根据 `(record_id, mode)` 续跑。
- `predictions.jsonl` 与 `run_manifest.json` 只有 rank 0 在 barrier 后生成；看到 shard 数满但
  manifest 尚未出现时，等待合并完成，不要重复启动第二个作业。
- 完成复现后至少保存：requirements、manifests、checkpoint hash、run_manifest、
  predictions、geval scores、tables、audit 输出与 runtime。

最终交付前逐项确认：

```text
[x] epoch 1/2/3 validation: 每个 3000 rows / 1500 series，覆盖一致
[x] best epoch 由完整 validation mean loss 选择，不看 test
[x] epoch-3 full284/base prediction: 284 rows，audit = ok
[x] paper140/base 从同一 full284 run 严格过滤：140 rows
[x] paper140 G-Eval: 335 scores；fallback 均为恰好 20 个有效样本
[x] Table 1 AXIS 第一行已写入 REPRODUCTION_RESULTS.md
[x] 所有 diagnostic/partial/validation-judge 输出均未进入表格
[x] 文档和正式产物中无密码、API key、SSH 私钥或 token
[ ] [待补充] 用 DCGM/nvidia-smi 重新记录训练与推理峰值显存
```

## 10. 后续架构改进的低成本评估

Table 1 只能回答“最终文本质量是否变化”，不能证明模型是否更依赖时间序列证据。后续每个
架构版本应使用同一 split、manifest、decode 配置和 seed，与本 Baseline 做 paired comparison。
推荐按以下成本层级执行。

### 10.1 每次训练都跑：无 API

- 全量 validation teacher-forced NLL：报告 overall 以及 MC/OE/TF 分题型均值。
- MC/TF：forced-choice accuracy、gold-vs-best-wrong margin、parse rate；paired McNemar test。
- OE：numeric span F1、数值误差、ROUGE-L/语义相似度作为辅助，不能只报一个 macro。
- 训练与推理成本：参数量、峰值显存、训练 wall time、每 QA 延迟、生成 token 数。
- 至少 3 个 seed；报告 paired bootstrap 95% CI，而不是只比较一次点估计。

### 10.2 每个候选都跑：证据因果诊断，无 API

现有实验已经指出 fixed/task hint 强而 local evidence 弱。改进是否有效，核心应看以下指标：

- `local_zero`、`local_shuffle`、`local_random/donor` 相对 base 的 `ΔNLL`、accuracy drop、flip rate。
  有效改进应使**样本相关 local 干预产生更强且方向正确的退化**。
- `fixed_zero` 与 `fixed_shuffle` 的影响，以及 `local ΔNLL / fixed ΔNLL`。目标不是让 fixed 完全
  无用，而是降低“fixed 控制一切、local 几乎可删”的失衡。
- label-changing evidence counterfactual：替换窗口后，答案跟随新证据的比例；同时报告
  question-negation flip，避免把语言模板敏感性误当作 evidence sensitivity。
- MC 全 24 种 option permutation：text-answer/label-only accuracy、gold margin、old-letter
  stick rate。真正的改进应对选项位置更稳健。
- OE explanation grounding：回答中的位置、方向、幅度、异常类型能否被窗口支持；报告
  evidence precision/recall 和 numeric F1，而不只看语言流畅性。

已有报告中的 `local-zero ΔNLL≈0.041`、fixed-all-zero `ΔNLL≈1.0096`、TF donor-evidence
flip≈0.319、negation flip≈0.681 等数值来自 released checkpoint，是失败机制参考，不是新训练
epoch-3 的正式基线。改架构前应先在 epoch 3 上重跑同一诊断，再与新模型逐样本配对比较。

### 10.3 需要时跑：内部机制

- encoder→Perceiver 前后对 anomaly family、幅度和位置做同预算 linear probe，检查信息保留。
- 对 question-conditioned retrieval/attention 报告正确 evidence span 的 mass、top-k recall。
- 用 CKA 或 paired representation distance 比较 base、证据替换、问题改写，验证表示变化主要
  跟随 evidence，而不是只跟随题面。

### 10.4 API 预算策略

- 缓存 Baseline 的 predictions 和 judge scores，后续永不重复评分 Baseline。
- 日常迭代只使用上述无 API 指标；里程碑候选在固定、分题型的 50–100 QA 上做一次盲化
  pairwise A/B judge。每题一个胜负判断通常比 7 个绝对维度更省且对小改进更敏感。
- 只有通过 causal diagnostics、分题型无明显退化且统计区间支持提升的最终模型，才执行一次
  完整 paper140 严格 G-Eval。不要用测试集反复挑架构或超参数。

最重要的接受标准不是“Table 1 macro 更高”，而是：Table 1 不退化或提升，同时 local evidence
必要性、label-changing counterfactual consistency 和 explanation grounding 明显增强，且 fixed
控制与位置先验减弱。否则很可能只是换了一种捷径。

