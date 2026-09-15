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

`third_party/datasets_rcd` 是按“生成运行时”裁出的 vendored 副本，而不是功能压缩版。目录共保留 9 个上游文件：`src/` 下全部 6 个实现文件、`config/synthetic.json`、`requirements.txt` 和上游 README；只省略与执行无关的上游 `.gitignore`。正常过程、20 类局部异常、22 类可选季节异常、DAG、滞后 ARX、内生/外生注入和多进程入口均在。与 Datasets-RCD 提交 `73eb733f3026e32bf9edfde1d4c3b787358908e7` 比较时，其中 8 个文件逐字节一致；唯一代码差异位于 `src/generate_dataset.py`，即 Windows 下默认 worker 数由 80 改为 1，以规避 Windows `ProcessPoolExecutor` 限制。主入口仍显式提供 `--num-workers`，无需再克隆 Time-RCD 或 Datasets-RCD。

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
  --num-workers 1 `
  --seed 531
```

底层生成器先产生：

- `normal_time_series`：未注入异常的正常对照序列；
- `time_series`：注入异常后的多变量序列；
- `labels`：底层生成器原生的一维逐时刻标签；
- `attribute`：每个通道的信号属性、异常参数、随机 DAG 和内生/外生异常信息。

`legacy_adapter.py` 随后把这些字段整理为统一结构，包括 `series.values`、二维 `series.labels`、`channels`、`causal_graph`、`root_cause` 和 `synthetic_label`。二维标签由“观测序列与正常反事实的逐通道差异”在原生时间标签范围内恢复；这一步不重新注入异常。

底层有六类持续水平偏移模板：`sudden increase`、`sudden decrease` 以及四类 `* after * spike`。这些模板的数值偏移持续到序列末端，但原生 `position_end` / `labels` 只覆盖尖峰和过渡段；适配器严格保留该标签语义，不把后续尾部自行扩标。研究者若希望把持续尾部也算作异常，应另行定义标签修订规则，并与原始 `legacy_generator_record` 区分。

为了不让统一化过程丢失底层信息，输出同时保留以下两层：

- `legacy_generator_record` 保存原生一维 `labels`、完整 `attribute`，以及 `normal_time_series` / `time_series` 对应字段的原始 dtype 和 shape；NumPy 类型和 tuple 的 JSON 转换路径记录在 `attribute_type_manifest` 中。
- `series.values`、`normal_series` 和 `original_data` 使用 Python/JSON 的完整浮点表示，不再执行 6 位小数截断。
- `generation_record` 在每条样本中保存种子、计划长度、计划通道数、异常比例、属性模式、worker 数和生成器源码树哈希。

完整 `attribute` 不只是“异常类型”四个字。它包含顶层 `attribute_list`、`num_features`、`is_endogenous`、`dag`；每个通道还保存趋势、周期、频率、噪声、异常区间，以及 `full_attribute_pool` 中的幅度、周期、分段、异常参数、背景周期尖峰、周期噪声调制和统计量。

主要输出：

- `outputs/series/raw_series.jsonl`
- `outputs/series/dataset_summary.json`
- `outputs/series/generation_schedule.json`：逐样本记录计划的长度和通道数；
- `outputs/series/generation_provenance.json`：记录完整 CLI 参数、Python/依赖版本、平台、全部底层运行文件 SHA-256 及输出文件 SHA-256。

默认 `--num-workers 1` 是有意的：底层 Python/NumPy 随机调用按固定顺序执行，便于在相同源码和环境下重放。设置大于 1 的值可提高吞吐量，但底层按 future 完成顺序收集结果，样本行顺序可能随调度变化；该值会被明确写入 provenance。

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
- `focus_type`：便于归类设问语义的粗粒度类别；
- `template`：希望问题覆盖的语义目标；
- `instruction`：对措辞、观察范围以及禁问内容的约束；
- `choices`：仅 MC 使用的选项语义锚点。

它们不是依次再次随机选择的多层对象，而是同一个“设问角度条目”的绑定字段。选中后，frame、regular/hard 和 `key` 在程序侧决定题池、条件规则与问题 ID；实际发送给 GPT-5.5 的文本直接包含目标区间、题型、事实摘要、`template`、`instruction` 以及 MC 的 `choices` 锚点，并附上图片。`focus_type` 不以字段名写入 Prompt。模型输出经题型解析后写入样本；MC 必须成功解析出指定数量的选项。

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

GPT-5.4 不负责重新出题。它接收已经生成的题干、题型和选项，同时获得目标窗口图像、逐位置观测值与正常反事实值、目标区间，以及异常真值、根因/受影响通道、异常类型、范围、异常边和传播辅助字段。它并不直接接收 `windows[0].answer` 这一结构化参考答案；该字段只在生成完成后与模型答案一起写入 answer-only 文件。相比 GPT-5.5，GPT-5.4 最关键的新增输入是逐点数值—反事实对和更完整的教师真值 JSON。

默认数值表覆盖目标窗口全部时刻并保留三位小数；答案请求默认最多尝试 3 次。脚本会清理 think 标签，但不会程序化验证 MC/TF 首行或 OE 三段式的语义与格式，因此正式发布前仍需对教师答案作格式审计和内容抽查。

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

- 固定各阶段 `--seed` / `--question-seed`，并在阶段 1 使用 `--num-workers 1`，可复现底层随机调用顺序、窗口选择、题型轴和模板抽样；GPT 输出仍可能受远端服务版本与采样影响。
- 发布或归档 `raw_series.jsonl` 时应同时保留 `generation_schedule.json`、`generation_provenance.json` 和 `dataset_summary.json`；三者共同给出逐样本配置、源码/环境身份和文件完整性。
- `outputs/`、虚拟环境、缓存和本机服务器说明均被 `.gitignore` 排除。
- 配置中只写密钥环境变量名，绝不写真实密钥。
- `--resume` 用于续跑 GPT 阶段；大批量生成建议同时使用 `--stop-on-error`，避免跳过失败行后造成行号错位。
- 若更换 OpenAI-compatible 网关，只需修改两份 GPT JSON 中的 `base_url` 和 `model`，题库与 prompt 逻辑无需改动。
