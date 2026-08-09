# 多变量 Multi-AXIS 评测协议

本文只描述本仓库的多变量 AXIS，包括数值/文本单模态路径与图像增强的
VLM 路径。它不再使用原论文单变量 AXIS 的 `paper140`、`full284` 或
`tools.axis_repro` 口径。

## 1. 不可变原则

1. checkpoint 只能由 grouped validation 的 `validation_answer_nll` 选择；禁止用
   测试集标签、G-Eval 或 Judge 分数挑 epoch、seed、超参数或架构。
2. 正式五数据集结果统一使用 bias-neutralized 视图：478new、SMD、SWaT、
   LEMMA-RCA、VTA。
3. Teacher-Eval 478 与 478new 是同一批 478 个合成样本的两种问题文本视图，
   不是两个独立数据集。可以同时报告敏感性，但不得把二者当成 956 个独立样本，
   也不得在正式五数据集 macro 中双重计权。
4. MC、TF 报最终标签 exact-match accuracy；OE 报可解析率。G-Eval 分数是独立的
   1–5 语义质量指标，不能与标签准确率混称为 accuracy。
5. 推理、正式 Judge 调用与评分审计必须在 GPU 服务器执行。API Judge 本身不消耗
   本地 CUDA 算力，但其任务、断点文件与审计产物仍必须落在正式 GPU 实验目录。
6. 比较模型时固定数据视图、问题顺序、teacher、rubric、Judge model ID、endpoint、
   scoring method、生成参数和代码 commit。

## 2. 数据集与权威来源

| 入口名 | 视图 | N | MC/OE/TF | Question | Teacher |
|---|---|---:|---:|---|---|
| `478new` | 合成、去偏置问法 | 478 | 176/139/163 | `question/question_eval_bias_neutralized_20260710b/*/questions_1000.jsonl` | `teacheranswer/teacher_eval_bias_neutralized_20260710b/*/teacher_gpt55.answers.jsonl` |
| `478` | 同一合成样本、原始问法 | 478 | 176/139/163 | `question/question_eval/*/questions_1000.jsonl` | `teacheranswer/teacher_eval/*/teacher_gpt55.answers.jsonl` |
| `SMD` | 真实、去偏置 | 200 | 66/67/67 | `question/question_smd_axis_v1_no_root_200_gpt54_bias_neutralized_20260713c/questions_200.jsonl` | `teacheranswer/teacher_smd_axis_v1_no_root_200_gpt54_bias_neutralized_20260713a/teacher_gpt54.answers.jsonl` |
| `SWaT` | 真实、去偏置 | 184 | 61/62/61 | `question/question_swat_axis_v1_eval92_regular_184_gpt54_bias_neutralized_20260712c/questions_184.jsonl` | `teacheranswer/teacher_swat_axis_v1_eval92_regular_184_gpt54_20260712a/teacher_gpt54.answers.jsonl` |
| `LEMMA-RCA` | 真实、去偏置 | 12 | 4/4/4 | `question/question_lemma_rca_cloud_curated_onset_v1_no_root_12_gpt54_bias_neutralized_openfirst_20260713b/questions_12.jsonl` | `teacheranswer/teacher_lemma_rca_cloud_curated_onset_v1_no_root_12_gpt54_bias_neutralized_openfirst_20260713a/teacher_gpt54.answers.jsonl` |
| `VTA` | 真实、去偏置 | 200 | 67/67/66 | `question/question_vta_articulary_axis_v1_no_root_200_gpt54_llm2_bias_neutralized_openfirst_20260713b/questions_200.jsonl` | `teacheranswer/teacher_vta_articulary_axis_v1_no_root_200_gpt54_llm2_bias_neutralized_openfirst_20260713a/teacher_gpt54.answers.jsonl` |

478/478new 的问题池各有 2,000 行，但只有 478 行具有 teacher 长答案；本协议的
G-Eval 和表格均使用这 478 行。manifest builder 以规范化问题文本在各自视图内部
匹配 teacher，禁止跨视图拿原始问题匹配去偏置 teacher。

训练数据的权威契约是 62 个 question/teacher shard：67,820 对，监督目标字段为
teacher `model_answer`；过滤 17 条空目标后保留 67,803 条，并按 `base_sample_id`
以 seed 42 做 grouped 90/10 划分。train/validation 的 `base_sample_id` 交集必须为 0。

## 3. 构建与审计 manifest

```bash
export PYTHONPATH="$PWD"
python tools/multi_axis/build_manifests.py \
  --data-root /path/to/multi-axis-assets \
  --output-dir /path/to/manifests \
  --seed 42 --validation-fraction 0.10
```

若已有正式 train/validation 与五个去偏置测试 manifest，只增补原始问法视图时使用：

```bash
python tools/multi_axis/build_manifests.py \
  --data-root /path/to/multi-axis-assets \
  --output-dir /path/to/existing-manifests \
  --only eval --eval-datasets 478
```

该命令只写 `eval_478*` 并合并 `eval_summary.json`，不重建或覆盖训练划分。

必须得到：

- paired 67,820；direct match 67,773；audited recovery 47；structured-reference
  match 67,820；过滤空 teacher 17；最终 67,803；
- train 61,053、validation 6,750；train/validation group overlap 0；
- 六个可选择评测入口的数量与上表完全一致；
- 每个 manifest 保存 question/teacher 源路径、行号或字节偏移、输入 SHA-256、
  `sample_id`、`base_sample_id` 与题型。

VLM 还必须为所选评测 manifest 渲染图像，并执行字体、600 DPI、半开区间边界、
无标签/异常分数泄漏以及 PNG SHA-256 审计。单模态路径忽略图像字段。

## 4. checkpoint 与推理

读取训练目录的 `best_checkpoint.json`。文件必须声明：

```json
{
  "selection_metric": "validation_answer_nll",
  "best_epoch": 10,
  "checkpoint": "hint_epoch_10.pt"
}
```

其中数值只是示例；实际值以当前已完成 epoch 为准。不得等待或窥视测试分数后改选。

`infer.py` 默认运行五个 bias-neutralized 数据集。诊断原始/去偏置问法敏感性时显式
传入两个视图。推理 world size 可以与训练不同，但必须显式登记并与实际
`torchrun` world 完全一致：

```bash
torchrun --standalone --nproc_per_node=3 tools/multi_axis/infer.py \
  --config experiments/multi_axis/formal_deepseek_20epochs_4gpu.json \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/manifests \
  --timercd-checkpoint /path/to/pretrain_checkpoint_best_multi.pth \
  --hint-checkpoint /path/to/hint_epoch_10.pt \
  --output-dir /path/to/predictions \
  --expected-inference-world-size 3 \
  --datasets 478new 478
```

每个 `<dataset>.predictions.jsonl` 必须逐行对应 manifest，并至少包含：
`index`、`sample_id`、`dataset`、`question_group`、`raw_response`。推理 manifest
必须记录 checkpoint epoch/SHA-256、TimeRCD SHA-256、模型、生成参数、world size、
代码来源与数据集顺序。

### 4.1 长上下文与禁止截断

数值提示必须完整保留，禁止为适配显存而截断 channel、window 或问题文本。运行前应以
正式 tokenizer 审计全部 prompt。当前两个 478 视图各有 18/478 条超过 32,768 tokens，
最大为 38,325；因此本次 Epoch-10 诊断使用
`experiments/multi_axis/eval_deepseek_epoch10_context131072.json`，只把推理上下文上限
提升到模型原生 YaRN 131,072 上限，不改变 checkpoint、beam-5 或其他生成参数。

```bash
python tools/multi_axis/audit_prompt_lengths.py \
  --config experiments/multi_axis/eval_deepseek_epoch10_context131072.json \
  --model-path /path/to/deepseek-r1-0528-qwen3-8b \
  --data-root /path/to/multi-axis-assets \
  --manifest-dir /path/to/manifests \
  --datasets 478new 478 \
  --output /path/to/prompt_length_group_audit.json
```

混合显存推理必须先做边界探针。本次审计中，A100 40GB 对 23,883-token prompt
即使启用 `expandable_segments` 仍然 OOM，而完整 15,801-token 两样本探针成功；
正式分片进一步采用 14,000-token 安全阈值，所有更长 group 只分配给 80GB GPU。
该阈值是本次硬件/beam-5 配置的容量记录，不是通用模型常数。

### 4.2 保持 group 的异构 GPU 分片

外部分片以 `ManifestDataset.groups` 的 group number 为单位，不能拆开同一
`base_sample_id`。每个 worker 显式登记完整分片总数和自己承担的 group residue：

```bash
torchrun --nproc_per_node=1 tools/multi_axis/infer.py \
  ... \
  --datasets 478new 478 \
  --expected-inference-world-size 1 \
  --inference-shard-count 216 \
  --inference-shard-indices 0 10 17 19
```

各 worker 输出不可直接拼接。必须验证 checkpoint、TimeRCD、生成参数、LLM/attention
配置一致，分片互斥且覆盖全部 group，并按原 manifest `index` 还原顺序：

```bash
python tools/multi_axis/merge_sharded_inference_outputs.py \
  --source /path/to/worker-a \
  --source /path/to/worker-b \
  --manifest-dir /path/to/manifests \
  --output-dir /path/to/merged_predictions \
  --datasets 478new 478
```

正式报告必须保存 prompt-length audit、worker plan、失败/成功容量探针和合并 manifest。

## 5. 确定性任务指标

```bash
python tools/multi_axis/label_metrics.py \
  --manifest-dir /path/to/manifests \
  --predictions-dir /path/to/predictions \
  --output-dir /path/to/label_metrics \
  --datasets 478new 478
```

- MC：解析最终 A–F 标签并做 exact match；无法解析计错，同时单列不可解析数。
- TF：规范化 Yes/No/True/False 后做 exact match；无法解析计错。
- OE：非空且非错误占位文本记为可解析；它不是开放题事实正确率。
- 同时报告 overall、分题型、分数据集。对 478/478new 的 combined overall 只作
  两视图诊断，不解释为独立样本总体。

## 6. G-Eval：GPT-5.4 基线对齐口径

先生成 Judge 输入：

```bash
python tools/multi_axis/prepare_judge_inputs.py \
  --manifest-dir /path/to/manifests \
  --predictions-dir /path/to/predictions \
  --output-dir /path/to/judge_inputs \
  --datasets 478new 478
```

rubric 固定为 `experiments/multi_axis/geval_rubrics.json`：

| 题型 | 维度与权重 |
|---|---|
| MC | correctness 0.7 + reasoning_quality 0.3 |
| OE | accuracy 0.35 + completeness 0.35 + relevance 0.3 |
| TF | correctness 0.6 + justification_quality 0.4 |

与 `BASELINE_TABLES.md` 比较必须使用 `gpt-5.4` profile，不得用其他 Judge 的
结果替代。密钥只由 GPU 服务器的私有环境文件注入，不写入仓库、日志、命令行或
报告。`gpt-5.4` profile 还要求 `GPT54_CA_BUNDLE` 指向服务器信任的 CA bundle；
不得用 `--insecure`、未校验 SSL context 或全局关闭 TLS 校验代替。报告只记录 CA
文件 SHA-256，不记录 endpoint 明文：

```bash
export GPT54_CA_BUNDLE=/path/to/audited-ca-bundle.pem
python tools/multi_axis/geval_runner.py \
  --input /path/to/judge_inputs/478new.judge_input.jsonl \
  --rubrics experiments/multi_axis/geval_rubrics.json \
  --profiles experiments/multi_axis/judge_profiles.json \
  --judge gpt-5.4 \
  --output /path/to/geval/gpt-5.4/478new.jsonl
```

对 `478` 重复同一命令。断点续跑必须复用同一输出文件；完整任务数为每个视图
1,095 个 dimension scores。GPT-5.4 的当前接口口径是单次整数 1–5 分，结果必须
记录 `method=integer_score`、prompt hash、请求/返回模型、匿名 endpoint 指纹、
usage 与 latency。

## 7. 十指标、micro/macro 与双视图纪律

```bash
python tools/multi_axis/aggregate_geval.py \
  --results-root /path/to/geval \
  --output-dir /path/to/aggregate \
  --judges gpt-5.4 \
  --datasets 478new 478
```

每个“数据视图 × Judge”输出：MC Final/Corr./Rsn.、OE Final/Acc./Comp./Rel.、
TF Final/Corr./Justif.。micro 按题目数加权，macro 是两个视图表的等权诊断均值。
由于两视图共享底层样本，此处 macro 必须标注为 wording-view macro；正式五数据集
macro 只能使用 `478new SMD SWaT LEMMA-RCA VTA`。

## 8. 完整性审计与交付文件

```bash
python tools/multi_axis/audit_evaluation.py \
  --manifest-dir /path/to/manifests \
  --predictions-dir /path/to/predictions \
  --judge-input-dir /path/to/judge_inputs \
  --geval-results-root /path/to/geval \
  --label-metrics-dir /path/to/label_metrics \
  --datasets 478new 478 \
  --judges gpt-5.4 \
  --expected-checkpoint-epoch 10 \
  --output /path/to/evaluation_audit.json
```

审计必须 `passed=true`。结果 Markdown 至少包含：数据与代码来源、checkpoint
选择证据、生成配置、GPU/world、每视图标签指标、每视图十项 G-Eval、micro/macro、
与基线的绝对差及相对变化、失败/重试/缺失任务、API usage、数据视图相关性 caveat、
可复现文件路径和 SHA-256。

任何 partial JSONL、pending journal、first-N smoke、未通过身份/数量检查的文件都不得
进入正式表格。
