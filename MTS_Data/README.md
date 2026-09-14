# 多变量 AXIS 数据生成代码库

## 1. 当前结论

这个文件夹现在已经包含从“底层多变量时间序列合成”到“GPT 教师答案生成”的完整代码链。此前服务器工程缺少的历史 TSAD 底层生成器，已经由 `Datasets-RCD` 的 `clean_version` 分支补齐，并放到了服务器代码原本默认寻找的位置：

`server_pipeline/legacy/TSAD_dataset_gen-axis`

两个 GitHub 仓库的核对结果如下：

- `Datasets-RCD` 确实包含缺失的底层生成器，包括 `generate_dataset.py`、`ts_generator.py`、`ts_multi_generator.py`、属性池和依赖文件。当前克隆提交为 `73eb733f3026e32bf9edfde1d4c3b787358908e7`，分支为 `clean_version`。
- `Time-RCD` 的 `main` 分支主要提供预训练异常检测模型和推理 API，本身不是底层时间序列合成器。它的 README 指向 `TSAD_dataset_gen_public` 作为数据生成代码；远程核对表明该仓库的 `clean_version` 与本地 `Datasets-RCD` 指向同一个提交 `73eb733f...`，因此不需要再保存一份内容相同的第三个克隆。当前 `Time-RCD` 提交为 `372bb980426b2f67007311c6f3165ab789c79bef`。
- THU-IE 服务器上能找到底层结构适配、窗口抽取与绘图、GPT 题干生成、GPT 答案生成以及 DoFlow 备用合成器，但没有找到 `TSAD_dataset_gen-axis`/`Datasets-RCD` 的完整底层源码。因此，本目录采用“服务器快照 + 官方 GitHub 底层生成器”的组合。

## 2. 目录结构

```text
MTS_Data/
├── README.md
├── requirements-generation.txt
├── requirements-local-llm-optional.txt
├── THU_IE_GPU.md                         # 仅限本机，GitHub 提交中明确排除
├── server_pipeline/
│   ├── src/mvaxis/                       # 六阶段流水线的核心 Python 模块
│   ├── scripts/                          # 生成、抽窗、绘图、出题、答案及历史实验入口
│   ├── configs/                          # DoFlow、GPT-5.5、GPT-5.4 和历史模型配置
│   ├── legacy/AXIS_repo/                 # 服务器已有的历史 AXIS 模型代码
│   ├── legacy/TSAD_dataset_gen-axis/     # 从 Datasets-RCD 补入的兼容副本
│   ├── SOURCE_MANIFEST.json              # 服务器原始快照逐文件哈希
│   ├── LOCAL_PATCHES.md                  # 本地可运行性补丁说明
│   └── smoke_outputs/                    # 本次小样本验证产物
├── upstream/
│   ├── Datasets-RCD/                     # 官方底层生成器的原样 Git 克隆
│   └── Time-RCD/                         # 官方 Time-RCD 推理代码的原样 Git 克隆
├── tools/
│   ├── download_server_snapshot.py       # SFTP 只读下载与哈希清单工具
│   └── build_code_index.py               # 重建全量代码索引
└── docs/
    └── ALL_CODE_INDEX.md                 # 317 个代码/配置文件的逐文件作用说明
```

`server_pipeline` 保存了 THU-IE 2228 节点项目中 `src`、`scripts`、`configs` 和 `legacy/AXIS_repo` 的代码快照，共 288 个服务器原始文件。没有下载原始数据、生成结果、模型权重、缓存或压缩包。服务器来源、远程修改时间、文件大小和下载时 SHA-256 均记录在 `SOURCE_MANIFEST.json`。

需要逐一查看所有 Python、PowerShell、Shell 和 JSON 配置文件的作用时，请读 `docs/ALL_CODE_INDEX.md`。下面重点解释真正构成数据生成主链的代码。

## 3. 完整数据生成链

| 阶段 | 接收什么 | 做什么 | 主要输出 | 核心代码 |
|---|---|---|---|---|
| 1. 底层序列合成与异常注入 | 样本数、序列长度、通道数、异常比例和属性池 | 生成正常多变量序列，采样 DAG 和通道依赖，在根通道注入异常并传播到后继通道 | `normal_time_series`、`time_series`、时间标签、`attribute` | `legacy/TSAD_dataset_gen-axis/src/*.py` |
| 2. 统一真值转换 | 第一阶段的数组、时间标签、每通道属性、内生异常标记和 DAG | 把历史结构转成 MVAXIS 统一 schema；恢复异常通道、根因通道、异常区间、受影响通道和因果图 | `raw_series.jsonl`；每行含 `series`、`channels`、`root_cause`、`synthetic_label` 等 | `legacy_adapter.py`、`data_schema.py`、`evidence.py` |
| 3. 候选窗口抽取与绘图 | 一条长序列及其统一真值 | 分别采样含异常窗口和正常窗口，裁剪数值、换算全局位置，绘制带高亮区域的多通道折线图 | `windows_N.jsonl`、`images/*.png`、`summary.json` | `mvaxis_sample_windows_to_images.py`、`question_provider.py` |
| 4. frame 样本组织 | 候选窗口、目标异常比例和每序列窗口数 | 将窗口组织成 `anomaly_frame` 或 `normal_frame`；可保留多个候选或每序列选一个 | 带 `target_interval`、`target_output.fact_check` 和图像路径的样本 | `mvaxis_select_one_window_per_series.py`、`mvaxis_generate_variable_feature_windows.py` |
| 5. GPT-5.5 题干生成 | frame 样本、图像、MC/TF/OE 题型轴、regular/hard 题库、设问角度 | 从对应模板池抽取 `key/template/instruction/choices`，组装视觉 prompt，调用 LLM，解析并校验题干/选项 | 题目 JSONL、series 旁车文件、prompt/响应记录和审阅 HTML | `question_template*.py`、`question_provider.py`、`mvaxis_generate_bank_questions_from_windows.py` |
| 6. GPT-5.4 教师答案生成 | 最终题干、题型、窗口数值/图像，以及题干阶段不可见的完整真值 | 组装答案 prompt，要求模型给出与题型一致、受真值约束的答案和解释 | 完整答案 JSONL、answer-only JSONL、prompt 样例、summary 和 HTML | `teacher_answer_provider.py`、`mvaxis_generate_teacher_answers.py` |

第一阶段输出与第二阶段的关系非常直接：第二阶段不重新合成序列，也不重新注入异常；它只解释并规范第一阶段给出的 `labels + attribute + DAG`，把它们变成后续抽窗、出题和答案阶段都能稳定读取的字段。第三阶段以后不应直接猜测历史 `attribute` 的含义，而应读取第二阶段形成的统一真值。

## 4. 底层 TSAD 生成器各文件的作用

兼容副本位于 `server_pipeline/legacy/TSAD_dataset_gen-axis`，官方原样版本位于 `upstream/Datasets-RCD`。

| 文件 | 作用 |
|---|---|
| `src/generate_dataset.py` | 总入口 `generate_dataset(...)`。控制样本数、长度、异常样本比例、单/多变量模式、通道数和 worker，最后返回样本列表。每个样本含正常序列、异常后序列、标签和 attribute。 |
| `src/ts_generator.py` | 单通道生成内核。合成趋势、周期、噪声和局部/季节异常，记录异常种类与位置。 |
| `src/ts_multi_generator.py` | 多变量组织层。随机生成 DAG，为每个节点生成基础序列，通过系数和滞后形成父子依赖；选择内生异常通道，并把异常效应和标签传播到后继通道。 |
| `src/trend_utils.py` | 随机控制点、插值趋势、趋势文字和 ARIMA 趋势辅助函数。 |
| `src/config.py` | `ALL_ATTRIBUTE_SET` 及底层默认常量，定义可采样的趋势、周期、噪声、局部异常和季节异常类型。 |
| `src/attribute_utils.py` | 在 metric 模式下，把 `synthetic.json` 中的 metric 映射为受控属性。 |
| `config/synthetic.json` | metric 模式的大型属性组合池；使用 `use_attribute_set=True` 时主要改用 `ALL_ATTRIBUTE_SET`。 |

历史多变量样本的关键结构为：

```text
normal_time_series : 未注入异常的 T×C 数组
time_series        : 注入并传播异常后的 T×C 数组
labels             : 时间级或通道级异常标签
attribute:
  attribute_list   : 每个通道的趋势、周期、噪声和异常描述
  num_features     : 通道数
  is_endogenous    : 哪个通道被直接注入异常
  dag              : 通道依赖边
```

## 5. 服务器流水线核心代码

### 5.1 合成与真值

| 文件 | 作用 |
|---|---|
| `src/mvaxis/legacy_tsad_generator.py` | 动态加载历史 TSAD 生成器；可屏蔽缺少可选包时的 ARIMA/小波分支；把样本送入统一转换。 |
| `src/mvaxis/legacy_adapter.py` | 解释历史 labels、`attribute_list`、`is_endogenous` 和 DAG，构造通道级标签、根因、受影响通道、异常类型与区间。 |
| `src/mvaxis/data_schema.py` | 定义统一 schema、通道元信息、尺度统计和校验函数。 |
| `src/mvaxis/evidence.py` | 从数值计算局部突变、趋势、相关性、滞后等可观察证据。 |
| `src/mvaxis/generator.py` | 在 `legacy_tsad` 与 `doflow` 后端之间切换，输出统一数据。 |
| `src/mvaxis/causal_graph.py` | DoFlow 备用后端的四类 DAG 和结构因果模型。 |

### 5.2 窗口与图像

| 文件 | 作用 |
|---|---|
| `scripts/mvaxis_generate_ts_only.py` | 推荐的“只生成长序列”入口；支持固定或随机长度、固定或随机通道数、JSONL 分片。 |
| `scripts/mvaxis_generate_variable_feature_windows.py` | 串联“长序列生成”和“窗口抽取/绘图”的总入口。 |
| `scripts/mvaxis_sample_windows_to_images.py` | 从已有 `raw_series.jsonl` 采样 anomaly/normal 窗口并绘图。 |
| `scripts/mvaxis_select_one_window_per_series.py` | 每条源序列只保留一个最合适窗口。 |
| `scripts/mvaxis_generate_question_images.py` | 按源序列范围批量生成出题图像及元数据。 |
| `scripts/mvaxis_prepare_dataset_visuals.py` | 把高亮图、异常分数图等产物回填到 QA 数据。 |
| `scripts/mvaxis_generate_anomaly_score_images.py` | 使用异常检测模型输出逐点异常分数图。 |

### 5.3 题干

| 文件 | 作用 |
|---|---|
| `src/mvaxis/question_template.py` | regular 题库；定义 MC/TF/OE 题型轴、anomaly/normal frame 下的设问角度及 `QuestionSpec`。 |
| `src/mvaxis/question_template_hard.py` | hard 题库；重点覆盖关系破坏、传播阶段、根因与跟随通道区分等问题。 |
| `src/mvaxis/question_provider.py` | 判定 frame，抽题型与 focus item，把 `key/template/instruction/choices` 和窗口上下文组装为 GPT 输入，解析返回题干并构造结构化参考答案。 |
| `src/mvaxis/llm_client.py` | OpenAI-compatible API 与本地 Hugging Face/Qwen-VL 的统一客户端。 |
| `scripts/mvaxis_generate_bank_questions_from_windows.py` | 当前 regular/hard 统一入口；从保存窗口恢复样本、均衡题型轴、调用 LLM、断点续跑并写审阅 HTML。 |
| `scripts/mvaxis_generate_hard_questions_from_windows.py` | 较早的 hard 专用入口，保留用于复现历史批次。 |
| `scripts/mvaxis_generate_bank_questions_from_existing_series.py` | 从已有长序列直接完成抽窗和出题。 |
| `scripts/mvaxis_generate_bank_questions_one_per_series.py` | 每条源序列限制一道题。 |

### 5.4 教师答案与数据集装配

| 文件 | 作用 |
|---|---|
| `src/mvaxis/teacher_answer_provider.py` | 生成答案阶段的系统提示、题型模板、窗口数值表、图像消息、真值字段解释和 prompt。 |
| `scripts/mvaxis_generate_teacher_answers.py` | 批量调用教师 LLM；支持 preview、resume、重试、answer-only、prompt 样例和 HTML。 |
| `scripts/mvaxis_build_multilevel_qa_dataset.py` | 合并不同难度和题型来源，形成多层次 QA 数据集。 |
| `scripts/mvaxis_build_easy_medium_teacher60.py` | 构造历史 easy/medium 教师样本集。 |
| `scripts/mvaxis_build_fixed60_qa600.py` | 从固定 60 条序列和 frame 规格构造 600 题实验集。 |
| `scripts/sync_teacheranswer_train.py` | 把教师答案产物同步到训练数据目录。 |

## 6. 推荐运行方式

以下命令均从 `D:\桌面\多元AXIS\MTS_Data\server_pipeline` 执行。

### 6.1 安装核心依赖

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r ..\requirements-generation.txt
```

### 6.2 只生成底层多变量长序列

```powershell
python scripts\mvaxis_generate_ts_only.py `
  --num-samples 100 `
  --seq-len 180 `
  --num-features 10 `
  --anomaly-ratio 1.0 `
  --seed 531 `
  --output-dir outputs\ts_demo
```

默认会读取 `legacy/TSAD_dataset_gen-axis`，输出 `outputs/ts_demo/raw_series.jsonl` 和 `dataset_summary.json`。

### 6.3 一次完成长序列、候选窗口和绘图

```powershell
python scripts\mvaxis_generate_variable_feature_windows.py `
  --output-dir outputs\window_demo `
  --num-samples 100 `
  --seq-len-min 160 --seq-len-max 240 `
  --num-features-min 3 --num-features-max 10 `
  --samples-per-series 2 `
  --min-window-size 80 --max-window-size 120 `
  --window-anomaly-ratio 0.5
```

### 6.4 从已有长序列重新抽窗和绘图

```powershell
python scripts\mvaxis_sample_windows_to_images.py `
  --source outputs\ts_demo\raw_series.jsonl `
  --output-dir outputs\window_demo `
  --samples-per-series 2 `
  --min-window-size 80 --max-window-size 120 `
  --anomaly-ratio 0.5
```

### 6.5 GPT-5.5 生成题干

先设置配置文件要求的环境变量，例如 `gpt55_question_llm.json` 使用 `OPENAI_API_KEY`。不要把密钥写进 JSON 或脚本。

```powershell
$env:OPENAI_API_KEY = "你的密钥"
python scripts\mvaxis_generate_bank_questions_from_windows.py `
  --windows outputs\window_demo\windows_200.jsonl `
  --raw-series outputs\window_demo\raw_series.jsonl `
  --output-dir outputs\questions_regular `
  --question-bank regular `
  --llm-config configs\gpt55_question_llm.json
```

hard 题只需把 `--question-bank regular` 改成 `--question-bank hard`。窗口文件名中的数量应以实际 `summary.json` 为准。

### 6.6 GPT-5.4 生成教师答案

`gpt54_question_llm1.json` 到 `gpt54_question_llm5.json` 虽沿用历史文件名，但配置中的模型是 GPT-5.4，可用于答案生成。各文件读取的密钥环境变量可能不同，运行前应检查 `api_key_env`。

```powershell
$env:OPENAI_API_KEY1 = "你的密钥"
python scripts\mvaxis_generate_teacher_answers.py `
  --data outputs\questions_regular\questions.jsonl `
  --llm-config configs\gpt54_question_llm1.json `
  --output outputs\answers\questions_with_teacher.jsonl `
  --resume
```

若只想检查答案 prompt，不调用 API，可增加 `--preview-only --no-image --limit 3`。

## 7. 本地补丁与历史脚本注意事项

为保证本机可运行，做了两个最小补丁：

1. `src/mvaxis/generator.py` 原来从错误模块导入 `sample_question_specs`；现改为从实际定义它的 `question_template.py` 导入。
2. Datasets-RCD 兼容副本在 Windows 下默认使用单进程，避免上游默认 80 workers 超过 Windows 上限，以及动态加载模块在 spawn 模式下无法稳定序列化。Linux 仍保留 80 workers。

详细说明见 `server_pipeline/LOCAL_PATCHES.md` 和 `server_pipeline/legacy/TSAD_dataset_gen-axis/LOCAL_PATCHES.md`。`upstream/Datasets-RCD` 与 `upstream/Time-RCD` 始终保持原样，未打补丁。

服务器中的 `.ps1`/`.sh` 是历史实验启动器，许多带有旧的 `X:\...`、Linux 服务器目录、固定 GPU 编号或特定数据分片。它们用于追溯当时怎样批量运行，不应在本机直接照搬；当前推荐使用上面列出的 Python 入口并显式传参。

## 8. 已完成的验证

本机默认 Python 为 3.13.7。已通过清华 PyPI 镜像安装 `scipy`、`tqdm`、`PyWavelets` 和 `statsmodels`；未降级现有 NumPy 2.5.1。

| 检查 | 结果 |
|---|---|
| 历史 TSAD 合成 + 异常注入 + 统一真值转换 | 通过；默认 Python 生成 2 条 `128×3` 多变量异常序列，2 条均含异常。 |
| 候选窗口生成与绘图 | 通过；从 2 条长序列生成 4 个窗口和 4 张图，其中 anomaly/normal 各 2 个。 |
| DoFlow 备用合成器 | 通过；2 个基础系统、`alpha=[0,1]`，生成 4 条 `128×10` 变体并写出 train/val/test。 |
| GPT 题干 prompt 组装 | 通过；regular/hard × anomaly/normal frame × MC/TF/OE 共 12 种组合均成功生成 prompt。 |
| GPT 教师答案 prompt | 通过；`--preview-only` 成功生成答案 prompt 和审阅文件。 |
| 真实 GPT API | 未执行；避免未经确认消耗 API 额度。代码、配置和 preview 路径均已验证。 |
| 明文密钥扫描 | 未发现常见 `sk-`、GitHub token 或私钥头格式。 |

小样本产物保存在 `server_pipeline/smoke_outputs`，可直接查看 JSONL、summary、PNG 和 prompt 预览。由于本机已经完成核心实测，本次没有再占用 THU-IE GPU 做重复测试。

另外，默认 Python 中原有的 `torch 2.10.0` 声明缺少可选本地 LLM 模式所需的 `jinja2`。清华 PyPI 镜像在本次请求中未返回该包，因此它没有被安装；这不影响上述底层合成、真值转换、窗口绘图、OpenAI-compatible GPT API 或 prompt preview。只有运行 `local_hf`/`local_hf_vl` 时才需要先补齐 `requirements-local-llm-optional.txt`。

## 9. 如何确认代码没有遗漏

- 服务器原始快照：查看 `server_pipeline/SOURCE_MANIFEST.json`，其中有 288 个原文件的远程来源和哈希。
- 当前所有代码与配置：查看 `docs/ALL_CODE_INDEX.md`，其中逐一列出 317 个文件及作用。
- 重新生成索引：运行 `python tools\build_code_index.py`。
- 重新下载服务器快照：运行 `python tools\download_server_snapshot.py --help`，凭据由本机的 `THU_IE_GPU.md` 在运行时读取，不会写入脚本或清单。该凭据文件受 `.gitignore` 保护，不包含在 GitHub 版本中。
