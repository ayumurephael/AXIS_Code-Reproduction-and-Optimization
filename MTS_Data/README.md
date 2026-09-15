# MTS_Data：多变量时间序列问答数据生成代码

本目录只保留从“底层多变量时间序列”到“带 GPT 教师答案的 MC / TF / OE 数据”的完整生成链。训练、模型评测、服务器运维、代码下载器、历史输出样例和重复的上游仓库不在本代码库中。

## 1. 完整流程

```text
Datasets-RCD 底层生成器
  │  合成正常序列、随机 DAG、通道属性，并注入异常
  ▼
01_generate_series.py
  │  将底层输出转换成统一真值结构，写入 raw_series.jsonl
  ▼
02_sample_windows.py
  │  从长序列采样候选窗口，判断窗口真值并绘制 PNG
  ▼
03_organize_frames.py
  │  重建窗口样本；每个物理窗口复制为 anomaly_frame / normal_frame 两行
  ▼
04_generate_questions.py
  │  选择 regular / hard 题库、MC / TF / OE 和设问角度，调用 GPT-5.5 出题
  ▼
05_generate_answers.py
     调用 GPT-5.4 读取题目、图、序列与真值，生成教师答案
```

这里必须区分两个概念：

- `target_output.fact_check.is_anomalous` 是窗口的客观真值，来自逐时刻、逐通道异常标签。
- `question_frame_override` 是出题视角。`anomaly_frame` 要求从异常分析角度设问，`normal_frame` 要求从正常性核验或反证角度设问。复制 frame 行不会改写窗口真值。

因此，一个物理窗口可以形成两道不同视角的题，但两行仍共享同一段序列、图片和底层真值，并通过 `question_pair_id` 关联。

## 2. 目录结构

```text
MTS_Data/
├── configs/
│   ├── doflow.json              # 可选 DoFlow 底层生成器配置
│   ├── gpt55_question.json      # GPT-5.5 题干生成配置
│   └── gpt54_answer.json        # GPT-5.4 答案生成配置
├── scripts/
│   ├── 01_generate_series.py    # 底层合成、异常注入、真值转换
│   ├── 02_sample_windows.py     # 候选窗口抽取与绘图
│   ├── 03_organize_frames.py    # frame 组织
│   ├── 04_generate_questions.py # GPT-5.5 题干生成
│   ├── 05_generate_answers.py   # GPT-5.4 教师答案生成
│   ├── generate_doflow.py       # 可选 DoFlow 合成入口
│   └── _bootstrap.py            # 统一 Python 导入路径
├── src/mvaxis/
│   ├── legacy_adapter.py        # Datasets-RCD 输出 → 统一真值结构
│   ├── legacy_tsad_generator.py # 调用 vendored 底层生成器
│   ├── frame_builder.py         # 窗口重建、选择和双 frame 展开
│   ├── question_template.py     # regular 题库、MC/TF/OE 轴与设问角度
│   ├── question_template_hard.py# hard 题库及其设问角度
│   ├── question_provider.py     # 视觉 prompt、题干解析和题目写回
│   ├── teacher_answer_provider.py # GPT-5.4 答案 prompt
│   ├── llm_client.py            # OpenAI-compatible Chat Completions 客户端
│   ├── causal_graph.py          # DoFlow 图结构和 SCM 模拟
│   ├── generator.py             # DoFlow 数据生成
│   ├── data_schema.py           # 统一字段、通道尺度和校验
│   ├── evidence.py              # 窗口证据特征
│   ├── html_reports.py          # 题目/答案审阅页
│   ├── progress.py              # 进度输出
│   └── utils.py                 # JSONL、随机种子等公共函数
├── third_party/datasets_rcd/
│   ├── src/                     # Datasets-RCD 多变量序列合成与异常注入实现
│   └── config/synthetic.json    # 底层通道属性配置
├── requirements.txt
└── README.md
```

`third_party/datasets_rcd` 是实际运行所需的精简 vendored 代码，不需要再克隆 Time-RCD 或 Datasets-RCD。

## 3. 环境安装

建议使用 Python 3.10–3.12。Windows PowerShell 示例：

```powershell
cd "D:\桌面\多元AXIS\MTS_Data"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

GPT 调用只使用 Python 标准库，不需要安装 OpenAI SDK。密钥只放在环境变量中，不要写进 JSON 或提交到 Git：

```powershell
$env:OPENAI_API_KEY = "你的密钥"
```

如果同一机器使用多把密钥，不需要复制配置文件，可直接覆盖变量名：

```powershell
python scripts/04_generate_questions.py `
  --frames outputs/frames/frames.jsonl `
  --api-key-env OPENAI_API_KEY2
```

命令会读取 `$env:OPENAI_API_KEY2`。答案阶段同样支持 `--api-key-env`。

## 4. 分阶段运行

以下命令构造一个很小的演示批次。正式生成时增大样本数、序列长度和通道数即可。

### 阶段 1：底层序列合成、异常注入与真值转换

```powershell
python scripts/01_generate_series.py `
  --output-dir outputs/series `
  --num-samples 2 `
  --seq-len 128 `
  --num-features 3 `
  --anomaly-ratio 1.0 `
  --seed 531
```

底层生成器先产生：

- `normal_time_series`：未注入异常的正常对照序列；
- `time_series`：注入异常后的多变量序列；
- `labels`：逐时刻、逐通道标签矩阵；
- `attribute`：每个通道的信号属性、异常参数、随机 DAG 和内生/外生异常信息。

`legacy_adapter.py` 随后把这些字段整理为统一结构，包括 `series.values`、`series.labels`、`channels`、`causal_graph`、`root_cause` 和 `synthetic_label`。这一步不是重新生成异常，而是把底层真值变成后续窗口与问答代码都能稳定读取的格式。

主要输出：

- `outputs/series/raw_series.jsonl`
- `outputs/series/dataset_summary.json`

### 阶段 2：候选窗口抽取与绘图

```powershell
python scripts/02_sample_windows.py `
  --source outputs/series/raw_series.jsonl `
  --output-dir outputs/windows `
  --samples-per-series 2 `
  --min-window-size 60 `
  --max-window-size 80 `
  --anomaly-ratio 0.5 `
  --seed 531
```

程序逐条读取长序列，根据 `series.labels` 找到异常区间和可作为对照的正常区间，再按 `anomaly-ratio` 组织候选池。每个窗口记录原序列行号、起止点、是否含异常、受影响通道、根因通道、异常类型和图片路径；同时把所有通道绘制到一张 PNG 中，并高亮目标区间。

输出文件名包含窗口总数。例如 2 条序列、每条 2 个窗口时，得到：

- `outputs/windows/windows_4.jsonl`
- `outputs/windows/images/*.png`
- `outputs/windows/summary.json`

### 阶段 3：重建窗口并组织 frame

```powershell
python scripts/03_organize_frames.py `
  --windows outputs/windows/windows_4.jsonl `
  --raw-series outputs/series/raw_series.jsonl `
  --output outputs/frames/frames.jsonl `
  --num-windows 2 `
  --seed 601
```

窗口 JSONL 只保存轻量元数据，所以此阶段用 `source_index` 找回阶段 1 的完整长序列，再用 `target_interval` 重建可直接出题的窗口样本。之后每个窗口复制成两行：

```text
同一物理窗口 W
├── W__anomaly_frame  （从异常识别、定位、传播等角度设问）
└── W__normal_frame   （从正常性、证据不足、排除误判等角度设问）
```

两行的 `question_pair_id`、`source_window_sample_id` 相同；`question_frame_override` 不同；`target_output.fact_check` 保持原窗口真值不变。主要输出为 `frames.jsonl` 和 `frames.summary.json`。

### 阶段 4：GPT-5.5 生成题干

```powershell
python scripts/04_generate_questions.py `
  --frames outputs/frames/frames.jsonl `
  --output-dir outputs/questions `
  --question-bank regular `
  --llm-config configs/gpt55_question.json `
  --question-seed 602 `
  --stop-on-error
```

此阶段先确定 `question-bank`（`regular` 或 `hard`），再为每行分配 MC、TF、OE 之一。`question_provider.py` 根据 frame、题型和题库进入相应的设问角度池；池中的每个条目包含：

- `key`：设问角度的稳定标识；
- `template`：希望问题覆盖的语义目标；
- `instruction`：对措辞、观察范围以及禁问内容的约束；
- `choices`：仅 MC 使用的选项语义锚点。

它们不是依次再次随机选择的四层对象，而是同一个“设问角度条目”的四个字段。选中一个条目后，程序把图片、目标区间、frame、MC/TF/OE、regular/hard、条目字段和样本上下文一次性组装为视觉 prompt，交给 GPT-5.5。模型输出经题型解析和校验后写入样本；MC 必须成功解析出指定数量的选项。

主要输出：

- `outputs/questions/questions.jsonl`
- `outputs/questions/questions.html`
- `outputs/questions/summary.json`

### 阶段 5：GPT-5.4 生成教师答案

```powershell
python scripts/05_generate_answers.py `
  --data outputs/questions/questions.jsonl `
  --output outputs/answers/teacher_answers.jsonl `
  --llm-config configs/gpt54_answer.json `
  --stop-on-error
```

GPT-5.4 不负责重新出题。它接收已经生成的题干、题型和选项，同时获得目标窗口数值、通道信息、窗口图像、目标区间、异常真值、根因/受影响通道、异常类型及参考答案等信息。因此，相比题干生成阶段，它多得到的是用于判定正确答案和组织解释的完整真值与参考答案上下文。

主要输出：

- `teacher_answers.jsonl`：原题目行加上 `teacher_answer_llm`；
- `teacher_answers.answers.jsonl`：便于阅读的题目、参考答案、模型答案；
- 对应 HTML、prompt 预览和 summary 文件。

只检查答案 prompt、不调用 API：

```powershell
python scripts/05_generate_answers.py `
  --data outputs/questions/questions.jsonl `
  --output outputs/answers/preview.jsonl `
  --preview-only `
  --limit 1
```

## 5. regular / hard、frame、题型和设问角度的关系

程序真正执行的顺序是：

1. 阶段 3 已为每个物理窗口固定 `anomaly_frame` / `normal_frame`。
2. 运行阶段 4 时由参数固定整个批次使用 `regular` 还是 `hard` 题库。
3. `sample_question_axes_for_bank` 为每行平衡分配 MC、TF、OE。
4. 用“题库 + frame + 题型”定位对应设问角度池。
5. 从池中抽取一个完整条目；`key/template/instruction/choices` 随条目一起确定。
6. 把该条目和样本信息组装成 GPT-5.5 输入，模型写出最终自然语言题干。

所以不是先分别抽 `key`、再抽 `template`、再抽 `instruction`、最后抽 `choices`。它们属于同一个预先定义的模板条目，必须一起使用，才能保证语义目标、约束和 MC 选项锚点互相一致。

## 6. 可选 DoFlow 合成器

`third_party/datasets_rcd` 是主流程默认底层生成器。若需要 DoFlow 风格的 SCM、图结构和节点/边/子图异常，可运行：

```powershell
python scripts/generate_doflow.py --config configs/doflow.json
```

DoFlow 是替代性的底层数据来源，不是主流程的额外必经步骤。

## 7. 可复现性与安全事项

- 固定各阶段 `--seed` / `--question-seed`，可复现窗口选择、题型轴和模板抽样；GPT 输出仍可能受远端服务影响。
- `outputs/`、虚拟环境、缓存和本机服务器说明均被 `.gitignore` 排除。
- 配置中只写密钥环境变量名，绝不写真实密钥。
- `--resume` 用于续跑 GPT 阶段；大批量生成建议同时使用 `--stop-on-error`，避免跳过失败行后造成行号错位。
- 若更换 OpenAI-compatible 网关，只需修改两份 GPT JSON 中的 `base_url` 和 `model`，题库与 prompt 逻辑无需改动。
