# baseline_new 筛选与清理清单

`baseline_new` 是 `baseline` 的后续架构研究版。源目录未被修改；本目录只保留重新训练、推理、评估和与既有 Baseline 做配对比较所需的闭环文件。

## 保留内容

- 模型源码：`src/models/AXIS/` 全部 5 个 Python 文件。
- 生产脚本：Phase-II DDP/显存安全训练、逐记录推理、严格 G-Eval、manifest、审计、聚合表格、验证 loss 选 epoch、checkpoint 精简和测试。
- 配置与依赖：`experiments/configs/`、`requirements.txt` 及交接文档。
- 数据：30,000-series Phase-II 训练集和 142-series/284-QA 正式测试集。
- 固定样本清单：`phase2_split.json`、`paper140.json`、`full.json`。
- 必需权重：
  - Phase-I 初始化：`experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth`
  - 最终 Baseline：`experiments/reproduction/phase2_torch251/epoch_3_inference.pth`
- 正式结果：epoch-3 full284/base predictions，以及 paper140 的 predictions、G-Eval scores、audit、Table 1 JSON/Markdown。
- checkpoint 选择摘要：`best_validation_loss.json`。该文件保留 epoch-1/2/3 的完整验证集统计，但其指向的历史 validation JSONL 没有复制。

## 未复制内容

- `MCQ_datasets/`、`QA_datasets/`：与已经整理好的 `AXIS_qa_test/` 重复的上游测试数据副本。
- epoch-1/2 inference 权重、训练 optimizer checkpoint、step checkpoint、运行时 tar/tar.gz。
- 作者 released AXIS 权重及其历史兼容实验；它们不是后续“自训练 Baseline vs 改进模型”的比较对象。
- validation 的 raw/head/tail predictions 和中断的 validation G-Eval；最佳 epoch 的统计摘要已经保留。
- API smoke、released-checkpoint 诊断、series-batching 试验、partial/fallback-pending journal、日志与空日志。
- `__pycache__/`、`.pytest_cache/`、`.pyc`，以及已被生产入口替代的旧版/临时脚本。
- `README.original.md`：当前 `readme.md` 已完整说明与原仓库关系和复现流程。
- 任何凭据、API key、SSH 配置和服务器连接信息均未复制。

## 体积变化

- 原目录：30,689 个文件，5,980,297,533 bytes（约 5.57 GiB）。
- 精简目录：约 30,198 个文件，约 1.89 GiB（其中绝大多数是必需训练数据与两份权重）。

## 使用边界

本目录可从 Phase-I 权重重新训练 Phase II，也可直接用 epoch-3 权重复核正式 Table 1，并为后续模型保存同格式 predictions 后做配对比较。它不用于复查作者 released checkpoint、历史失败任务或原始数据转换过程；如确有需要，从未修改的 `../baseline/` 单独取回对应文件，不要混入正式结果目录。
