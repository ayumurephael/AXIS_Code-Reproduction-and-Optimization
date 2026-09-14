# 全量代码与配置索引

本索引覆盖 `MTS_Data` 中除 `.git`、测试输出和凭据说明外的全部代码/配置文件。
核心生成文件的说明经过人工核对；服务器中与训练、评估、运维有关的历史文件按用途分类说明。

## requirements-generation.txt

| 文件 | 作用 |
|---|---|
| `requirements-generation.txt` | 依赖、包元数据或运行配置。 |

## requirements-local-llm-optional.txt

| 文件 | 作用 |
|---|---|
| `requirements-local-llm-optional.txt` | 依赖、包元数据或运行配置。 |

## server_pipeline

| 文件 | 作用 |
|---|---|
| `server_pipeline/configs/523trail_qwen25_7b_axis_embedding.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/configs/deepseek_geval.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/configs/deepseek_llm.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/configs/doflow_axis_qa_compare10.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/configs/gpt54_question_llm1.json` | GPT-5.4 教师答案生成的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt54_question_llm2.json` | GPT-5.4 教师答案生成的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt54_question_llm3.json` | GPT-5.4 教师答案生成的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt54_question_llm4.json` | GPT-5.4 教师答案生成的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt54_question_llm5.json` | GPT-5.4 教师答案生成的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_geval.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm1.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm10.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm2.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm3.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm4.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm5.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm6.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm7.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm8.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm9.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/gpt55_question_llm_openai.json` | GPT-5.5 题干生成或 G-Eval 的 OpenAI-compatible API 配置；密钥由环境变量读取。 |
| `server_pipeline/configs/local_axis_llm_template.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/configs/local_qwen25_3b_cuda.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_3b_cuda_remote.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_axis_style_smoke_bridge.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_bridge.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_embedding_clean.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_embedding_clean_qa100_v1.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_fast_json.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_fast_text_cache.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_overfit64_bridge.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_semantic_truth_bridge.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_timercd_dproj256_523truth600.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_timercd_dproj256_523truth600_fast_eval.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_timercd_dproj256_523truth600_schema_exact_fast_eval.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_7b_cuda_remote_timercd_dproj256_typeheads_v1.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl3b_cuda_remote.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_question1000_0603_eval.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_question1000_0603_eval_2gpu.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_trainsmall.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_2gpu_shard.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_ct16_scale005.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_cuda_remote_timercd_dproj256_typeheads_v1_4gpu_ct8_scale010.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_fixed60_qa600_globalhint_eval_scoreimage_ct12_scale010_1gpu_20260621a.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_fixed60_qa600_globalhint_train_score_ct12_scale010_1gpu_20260620b.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_fixed60_qa600_globalhybrid_ct12_scale010_1gpu_20260620a.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_ts61_vl_allhints_ct12_scale010_8gpu.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_qwen25_vl7b_ts61_vl_noglobalhints_highlight_ct12_scale010_8gpu.json` | 本地 Qwen 文本/视觉模型的历史推理或训练配置。 |
| `server_pipeline/configs/local_smoke_doflow.json` | 本地 DoFlow 小样本验证配置。 |
| `server_pipeline/configs/mvaxis_geval_rubrics.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/configs/smoke.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/configs/torch_fixed60_qa600_timercd.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/configs/torch_legacy_axis50_original.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/configs/torch_relation_stress_v1.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/configs/torch_semantic_scale.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/configs/torch_semantic_scale_nocf_schema.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/configs/torch_smoke.json` | AXIS/Time-RCD 训练或评估实验配置，不是底层数据生成配置。 |
| `server_pipeline/legacy/AXIS_repo/src/models/AXIS/AXIS.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/legacy/AXIS_repo/src/models/AXIS/AXIS_test.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/legacy/AXIS_repo/src/models/AXIS/dataset.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/legacy/AXIS_repo/src/models/AXIS/Pretrain_ts_encoder.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/legacy/AXIS_repo/src/models/AXIS/ts_encoder_bi_bias.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/config/synthetic.json` | 服务器项目配置文件；具体用途由文件名及调用它的脚本决定。 |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/requirements.txt` | 依赖、包元数据或运行配置。 |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/src/attribute_utils.py` | Attribute utilities for time series generation. |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/src/config.py` | Configuration module for time series generation. |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/src/generate_dataset.py` | Dataset generation module for time series anomaly detection. |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/src/trend_utils.py` | Trend generation utilities for time series. |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/src/ts_generator.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/legacy/TSAD_dataset_gen-axis/src/ts_multi_generator.py` | Multivariate time series generation module. |
| `server_pipeline/scripts/01_generate_synthetic.ps1` | Windows 历史启动器：运行合成数据入口。 |
| `server_pipeline/scripts/02_train_encoder.ps1` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/03_train_hint_tuner.ps1` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/04_evaluate.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/_bootstrap.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/build_legacy_qa_review_and_compare.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/build_local_exports_eval_index.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/build_teacher_student_live_compare.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/build_tsdata61_hparam_docx.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/cleanup_axis_duplicates.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/continue_chatts_upload_and_resubmit.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/create_baseline_eval_metrics_docx.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/evaluate.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/evaluate_axis_interval_proposal.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/evaluate_swat_tsad_reference_metrics.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/evaluate_swat_window_anomaly_head.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/evaluate_timercd_interval_proposal.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/evaluate_timercd_tsad_metrics.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/export_qa_comparison_txt.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/generate_from_legacy_axis.py` | 较早的 Datasets-RCD 直连脚本；加载底层生成器、转换并切分 JSONL。 |
| `server_pipeline/scripts/generate_synthetic.py` | 读取 JSON 配置并调用统一生成入口。 |
| `server_pipeline/scripts/monitor_ts61_scale100_pull_20260609b.ps1` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/mvaxis_build_answer_review_html.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/mvaxis_build_easy_medium_teacher60.py` | 构造 easy/medium 教师集的历史脚本。 |
| `server_pipeline/scripts/mvaxis_build_eval_summary_html.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/mvaxis_build_fixed60_qa600.py` | 从固定 60 条序列和 frame 规则构造 600 题实验集。 |
| `server_pipeline/scripts/mvaxis_build_hard200_review_pages.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/mvaxis_build_multilevel_qa_dataset.py` | 把不同难度/题型来源合并为多层次 QA 数据集。 |
| `server_pipeline/scripts/mvaxis_build_swat_axis_review_html.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/mvaxis_build_training_report_html.py` | 数据/报告/实验产物构建脚本；是否进入主链取决于具体实验。 |
| `server_pipeline/scripts/mvaxis_compare_student_runs.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/mvaxis_compute_f1_score_auc.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/mvaxis_compute_metrics_with_open_decision.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/mvaxis_dump_fixed_question_specs.py` | 导出固定题目规格，便于检查题型/设问角度分布。 |
| `server_pipeline/scripts/mvaxis_dump_promptcap_examples.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/mvaxis_eval_answer_nll.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/mvaxis_eval_answer_semantic_alignment.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/mvaxis_eval_teacher_aligned_student_runs.py` | 模型/答案评估脚本；随服务器快照保留，不参与生成原始 QA。 |
| `server_pipeline/scripts/mvaxis_fill_missing_evidence_card.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/mvaxis_generate_anomaly_score_images.py` | 用异常检测模型生成逐点分数图和图像编码。 |
| `server_pipeline/scripts/mvaxis_generate_bank_questions_from_existing_series.py` | 直接从已有完整序列抽窗口并生成题目。 |
| `server_pipeline/scripts/mvaxis_generate_bank_questions_from_windows.py` | 从已保存窗口恢复统一样本，按 regular/hard 题库、frame 和 MC/TF/OE 轴调用 LLM 生成题干。 |
| `server_pipeline/scripts/mvaxis_generate_bank_questions_one_per_series.py` | 限制为每条源序列一道题的题干生成入口。 |
| `server_pipeline/scripts/mvaxis_generate_doflow_dataset.py` | DoFlow 生成入口别名，转交 generate_synthetic.py。 |
| `server_pipeline/scripts/mvaxis_generate_hard_questions_from_windows.py` | hard 题干生成的历史专用入口。 |
| `server_pipeline/scripts/mvaxis_generate_legacy_axis_dataset.py` | legacy_tsad 生成入口别名，转交 generate_from_legacy_axis.py。 |
| `server_pipeline/scripts/mvaxis_generate_question_images.py` | 按 source 范围选择序列/窗口并生成题目所需图像和旁车文件。 |
| `server_pipeline/scripts/mvaxis_generate_teacher_answers.py` | 读取最终 QA，调用教师 LLM 生成答案，并保存完整、answer-only、prompt 和 HTML 结果。 |
| `server_pipeline/scripts/mvaxis_generate_ts_only.py` | 只生成长多变量序列及异常真值，不抽窗口、不出题；支持长度/通道数随机化和分片。 |
| `server_pipeline/scripts/mvaxis_generate_variable_feature_windows.py` | 总编排入口：必要时生成变长/变通道序列，再调用窗口抽样与绘图。 |
| `server_pipeline/scripts/mvaxis_prepare_dataset_visuals.py` | 把高亮图、异常分数图等视觉产物回填到 QA 数据。 |
| `server_pipeline/scripts/mvaxis_prepare_real_benchmark_collection.py` | 数据或模型输入准备脚本。 |
| `server_pipeline/scripts/mvaxis_prepare_structured100_repair_dataset.py` | 数据或模型输入准备脚本。 |
| `server_pipeline/scripts/mvaxis_prepare_swat_axis_dataset.py` | 数据或模型输入准备脚本。 |
| `server_pipeline/scripts/mvaxis_profile_training_tokens.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/mvaxis_prompt_template_sweep.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/mvaxis_run_denoised_student_answer_eval.py` | 推理或实验执行脚本；不是底层合成入口。 |
| `server_pipeline/scripts/mvaxis_run_hint_ablation.py` | 推理或实验执行脚本；不是底层合成入口。 |
| `server_pipeline/scripts/mvaxis_run_llm_explanation.py` | 推理或实验执行脚本；不是底层合成入口。 |
| `server_pipeline/scripts/mvaxis_run_raw_qwen_answers.py` | 推理或实验执行脚本；不是底层合成入口。 |
| `server_pipeline/scripts/mvaxis_run_student_prompt_channel_experiments.py` | 推理或实验执行脚本；不是底层合成入口。 |
| `server_pipeline/scripts/mvaxis_run_truth_interval_with_model_hints.py` | 推理或实验执行脚本；不是底层合成入口。 |
| `server_pipeline/scripts/mvaxis_sample_windows_to_images.py` | 从 raw_series 生成 anomaly/normal 候选窗口、窗口 JSONL 和高亮折线图。 |
| `server_pipeline/scripts/mvaxis_score_with_geval.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/mvaxis_select_one_window_per_series.py` | 每条长序列从候选窗口中选一个代表窗口。 |
| `server_pipeline/scripts/mvaxis_shard_ts_jsonl.py` | 把大规模时间序列 JSONL 切成稳定分片。 |
| `server_pipeline/scripts/mvaxis_train_anomaly_head.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/mvaxis_train_embedding_bridge.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/mvaxis_train_globalhint_warmup.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/mvaxis_train_nexttoken_hints.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/mvaxis_watch_move_and_resume.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/plot_clean_embedding_cases.py` | 实验结果绘图脚本；用于诊断或论文图，不是题目窗口的核心绘图入口。 |
| `server_pipeline/scripts/plot_complex_relation_cases.py` | 实验结果绘图脚本；用于诊断或论文图，不是题目窗口的核心绘图入口。 |
| `server_pipeline/scripts/plot_legacy_tsad_anomaly_head.py` | 实验结果绘图脚本；用于诊断或论文图，不是题目窗口的核心绘图入口。 |
| `server_pipeline/scripts/plot_stress_anomaly_head.py` | 实验结果绘图脚本；用于诊断或论文图，不是题目窗口的核心绘图入口。 |
| `server_pipeline/scripts/plot_timercd_detection_examples.py` | 实验结果绘图脚本；用于诊断或论文图，不是题目窗口的核心绘图入口。 |
| `server_pipeline/scripts/rebuild_mvaxis_questions_from_qwen_raw.py` | 从 Qwen 原始输出重新解析和重建题目记录。 |
| `server_pipeline/scripts/remote_concat_question4000_to_ts61_train20k_20260610.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_daemon_eval_then_train_20k_8gpu_20260610.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_eval_chattime_ts61_20k_8gpu_20260618a.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_eval_chatts_chattime_swat_160_20260610.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_prepare_highlight_then_train_noglobalhints_8gpu_20260619.sh` | 历史云端数据/视觉产物准备脚本；包含服务器绝对路径，使用前需改路径。 |
| `server_pipeline/scripts/remote_prepare_ts61_20k_highlight_visuals_20260618.sh` | 历史云端数据/视觉产物准备脚本；包含服务器绝对路径，使用前需改路径。 |
| `server_pipeline/scripts/remote_prepare_ts61_20k_score_visuals_20260610.sh` | 历史云端数据/视觉产物准备脚本；包含服务器绝对路径，使用前需改路径。 |
| `server_pipeline/scripts/remote_prepare_ts61_score_visuals_20260608.sh` | 历史云端数据/视觉产物准备脚本；包含服务器绝对路径，使用前需改路径。 |
| `server_pipeline/scripts/remote_resume_question1000_two_gpu.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_resume_swat_160_eval_groups_20260605c.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_run_ts61_vl_allhints_pipeline_8gpu_20260608.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_runner_hold_h800_auto_20260614.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_start_eval_parallel_20260604.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_fixed60_qa600_globalhint_eval_scoreimage_20260621a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_fixed60_qa600_globalhint_g0g1_20260620b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_fixed60_qa600_globalhybrid_eval_20260620a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question1000_hard200_smoke_eval_20260603.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question1000_nexttoken_2x2gpu_sharded_scoretext_hints_noimage_20260603.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question1000_nexttoken_4gpu.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question1000_nexttoken_4gpu_scoretext_hints_noimage_20260603.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question1000_promptcap_smoke_2x2.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question4000_first4k_allhints_sharded_20260609a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_question4000_ts61_eval_20260609a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat160_ts61g1_scoreimage_8gpu_20260621c.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat160_ts61g1_scoreimage_zw1_20260621.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_160_ct16_scale005_rerun_20260607a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_160_eval_20260605c.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_160_eval_20260606a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_160_extra_noglobal_score_20260606b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_axis_smoke_eval_0605a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_controls_eval_20260605b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_noglobal_eval_20260605b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_scorepromptfix_eval_20260607b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_smoke_0606b_eval_20260606a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_smoke_0606b_eval_20260606b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_smoke_0606b_eval_20260606c.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_smoke_0606b_extra_noglobal_score_20260606d.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_swat_smoke_0606b_hintformat_ablation_20260606f.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_scale100_only_swat_half_full_20260609b.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_swat_scale100_half_20260609.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_vl_allhints_nexttoken_full_8gpu_20260608.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_vl_allhints_nexttoken_full_8gpu_20k_20260610.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_vl_allhints_nexttoken_smoke_8gpu_20260608.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_vl_allhints_nexttoken_smoke_8gpu_20k_20260610.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_vl_noglobalhints_highlight_nexttoken_full_8gpu_20k_20260618.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_ts61_vl_noglobalhints_highlight_nexttoken_smoke_8gpu_20k_20260618.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_start_tsdata61_smoke_hparam_eval_20260608a.sh` | 历史云端实验启动/恢复脚本；主要服务训练或评估，不属于核心数据生成链。 |
| `server_pipeline/scripts/remote_tools/axis_askpass.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/axis_scp_from_remote.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/axis_scp_to_remote.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/axis_ssh.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/download_zw1_qwen25_7b.sh` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/make_prompt_preview.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/scripts/remote_tools/run_axis_style_smoke.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_tools/save_password.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/setup_zw1_axis_env.sh` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/start_aux_semantic_shared.sh` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/start_axis_style_smoke.sh` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/status_aux_semantic_shared.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_tools/status_axis_style_smoke.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_tools/status_zw1_axis_setup.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/remote_tools/train_aux_semantic_shared.sh` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/remote_tools/zw1_askpass.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/zw1_scp_from_remote.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/zw1_scp_to_remote.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/remote_tools/zw1_ssh.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/run_523trail_50.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_523truth600_eval.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_523truth600_eval_fast128.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_523truth600_eval_schema_exact_fast96.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_baseline_tsdata61_smoke_0606c.ps1` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_hard_question_generation.ps1` | Windows 历史启动器：批量生成 hard 题。 |
| `server_pipeline/scripts/run_llm_inference.py` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_llm_on_proposals.py` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_qgen_final300_8group_20260528.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_qgen_final300_vl7b_image_hints_20260528.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_question_part00_regular_even_0619fulla.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part00_regular_odd_0619fulla.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part01_regular_even_0619fulla.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part01_regular_odd_0619fulla.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part02_hard_odd_0619fullb.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part02_regular_even_0619fulla.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part03_hard_even_0619fullb.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part03_hard_odd_0619fullb.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part04_hard_even_0619fullb.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_question_part04_hard_odd_0619fullb.ps1` | GPT 题干生成的历史分片 PowerShell 启动器；文件名编码分片、regular/hard 和奇偶窗口范围。 |
| `server_pipeline/scripts/run_smoke.py` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/run_teacheranswer_part00_seq_a_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part00_seq_b_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part01_seq_a_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part01_seq_b_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part02_seq_a_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part02_seq_b_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part03_seq_a_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part03_seq_b_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part04_seq_a_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_part04_seq_b_0621ta.ps1` | GPT 教师答案生成的历史分片 PowerShell 启动器；文件名编码分片与顺序组。 |
| `server_pipeline/scripts/run_teacheranswer_sequence.ps1` | Windows 历史启动器：按分片顺序生成教师答案。 |
| `server_pipeline/scripts/status_523trail_50.sh` | 历史运行、监控或状态脚本；保存用于追溯，通常含原服务器路径或特定实验参数。 |
| `server_pipeline/scripts/sync_teacheranswer_train.py` | 同步教师答案产物到训练数据目录。 |
| `server_pipeline/scripts/train_axis_embedding_bridge.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/train_axis_interval_proposal.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/train_encoder.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/train_hint_tuner.py` | 模型训练脚本；随服务器快照保留，不属于六阶段数据生成主链。 |
| `server_pipeline/scripts/upload_chatts_checkpoint_parts.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/upload_chatts_checkpoint_sequential.ps1` | 历史自动化启动或运维脚本；使用前需核对绝对路径、设备和环境变量。 |
| `server_pipeline/scripts/watch_question_then_teacheranswer.ps1` | 监控题干分片完成后触发对应教师答案任务。 |
| `server_pipeline/SOURCE_MANIFEST.json` | 服务器快照来源、下载时间以及原始文件大小/mtime/SHA-256 清单。 |
| `server_pipeline/src/__init__.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `server_pipeline/src/mvaxis/__init__.py` | Multivariate AXIS smoke workflow. |
| `server_pipeline/src/mvaxis/anomaly_score_context.py` | 聚合外部异常分数并附加窗口内/外统计上下文。 |
| `server_pipeline/src/mvaxis/answer_schema.py` | 规范 MC/TF 等答案标签和目标答案结构。 |
| `server_pipeline/src/mvaxis/axis_interval.py` | AXIS 异常区间提议模型及训练/评估。 |
| `server_pipeline/src/mvaxis/causal_graph.py` | DoFlow 备用底层生成器：定义 tree/diamond/fc_layer/chain DAG 并模拟结构因果模型。 |
| `server_pipeline/src/mvaxis/data_schema.py` | 定义并校验统一样本结构、通道元数据和模型输入。 |
| `server_pipeline/src/mvaxis/eval.py` | 闭环模型评估。 |
| `server_pipeline/src/mvaxis/evidence.py` | 根据窗口数值计算变化幅度、趋势、相关性和滞后等 evidence_card。 |
| `server_pipeline/src/mvaxis/features.py` | 从统一样本提取通道、证据和标签特征。 |
| `server_pipeline/src/mvaxis/generator.py` | 统一生成入口；选择 legacy_tsad 或 doflow 后端，组织异常注入、真值、问题和 train/val/test 输出。 |
| `server_pipeline/src/mvaxis/geval.py` | LLM-as-judge 评分 prompt、解析和汇总。 |
| `server_pipeline/src/mvaxis/hint_heads.py` | 通道池化、全局提示和时间细化网络头。 |
| `server_pipeline/src/mvaxis/html_reports.py` | 把题干或教师答案结果输出为便于人工复核的 HTML。 |
| `server_pipeline/src/mvaxis/legacy_adapter.py` | 把 Datasets-RCD 的 normal_time_series/time_series/labels/attribute 转成 MVAXIS 统一样本结构和通道级真值。 |
| `server_pipeline/src/mvaxis/legacy_tsad_generator.py` | 动态加载 Datasets-RCD，并把其样本交给统一真值适配器；也负责数据切分和 QA 窗口构建。 |
| `server_pipeline/src/mvaxis/llm_client.py` | 统一 LLM 客户端；支持 OpenAI-compatible API、本地 Hugging Face 文本模型和 Qwen-VL。 |
| `server_pipeline/src/mvaxis/llm_eval.py` | 把 LLM 结构化输出与真值比较。 |
| `server_pipeline/src/mvaxis/models.py` | NumPy 版编码器和 hint tuner。 |
| `server_pipeline/src/mvaxis/progress.py` | 长批处理的进度、耗时和 ETA 打印。 |
| `server_pipeline/src/mvaxis/prompts.py` | 学生推理/解释阶段的提示词和 JSON 输出规范。 |
| `server_pipeline/src/mvaxis/proposal.py` | 异常区间和受影响通道的候选提议、指标与答案解析。 |
| `server_pipeline/src/mvaxis/question_provider.py` | 窗口抽样、绘图、frame 判定、题型/设问角度抽样、GPT 题干 prompt 组装、题干解析及参考答案结构化。 |
| `server_pipeline/src/mvaxis/question_quality_expert.py` | 题干质量规则检查与问题报告。 |
| `server_pipeline/src/mvaxis/question_template.py` | regular 题库：MC/TF/OE 的题型轴、anomaly/normal frame 设问角度及 QuestionSpec。 |
| `server_pipeline/src/mvaxis/question_template_hard.py` | hard 题库：更复杂的关系破坏、传播阶段、根因与跟随者等设问角度。 |
| `server_pipeline/src/mvaxis/randomhint.py` | 构造随机提示，用于对照/消融实验。 |
| `server_pipeline/src/mvaxis/semantic_alignment.py` | 答案语义相似度和对齐统计。 |
| `server_pipeline/src/mvaxis/student_answer_provider.py` | 学生模型答案 prompt、消息和输出解析。 |
| `server_pipeline/src/mvaxis/student_answer_provider_experimental.py` | 实验版学生答案提示结构。 |
| `server_pipeline/src/mvaxis/teacher_answer_provider.py` | 把题目、窗口数值、图像和完整真值组装为教师答案 prompt/messages。 |
| `server_pipeline/src/mvaxis/torch_models.py` | PyTorch 版 AXIS 编码器和 hint tuner。 |
| `server_pipeline/src/mvaxis/training.py` | 选择模型后端并组织编码器/hint tuner 训练。 |
| `server_pipeline/src/mvaxis/utils.py` | JSON/JSONL、随机种子、路径和文本清洗通用函数。 |

## tools

| 文件 | 作用 |
|---|---|
| `tools/build_code_index.py` | 生成本文件的代码/配置全量索引。 |
| `tools/download_server_snapshot.py` | 从 THU_IE_GPU.md 读取凭据，通过 SFTP 只读下载代码快照并生成哈希清单。 |

## upstream

| 文件 | 作用 |
|---|---|
| `upstream/Datasets-RCD/config/synthetic.json` | metric 模式使用的大型属性配置池。 |
| `upstream/Datasets-RCD/requirements.txt` | Datasets-RCD 官方依赖声明。 |
| `upstream/Datasets-RCD/src/attribute_utils.py` | 把 synthetic.json 中的 metric 映射为受控生成属性。 |
| `upstream/Datasets-RCD/src/config.py` | 底层生成器默认参数与 ALL_ATTRIBUTE_SET 异常/形态属性池。 |
| `upstream/Datasets-RCD/src/generate_dataset.py` | 底层数据集总入口；按参数生成单变量/多变量样本，控制异常样本比例，并返回正常序列、注入后序列、标签和 attribute。 |
| `upstream/Datasets-RCD/src/trend_utils.py` | 趋势辅助函数；生成控制点、插值趋势和 ARIMA 序列。 |
| `upstream/Datasets-RCD/src/ts_generator.py` | 单通道波形与异常实现；合成趋势、周期、噪声、局部/季节异常并生成属性描述。 |
| `upstream/Datasets-RCD/src/ts_multi_generator.py` | 多变量生成器；采样 DAG，把父通道经滞后和系数组合到子通道，注入内生异常并传播标签。 |
| `upstream/Time-RCD/examples/quickstart.py` | 使用合成输入调用 Time-RCD 推理 API 的最小示例。 |
| `upstream/Time-RCD/pyproject.toml` | Time-RCD Python 包元数据与依赖。 |
| `upstream/Time-RCD/time_rcd/__init__.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `upstream/Time-RCD/time_rcd/_core/__init__.py` | Python 模块/脚本；随完整服务器或上游快照保留。 |
| `upstream/Time-RCD/time_rcd/_core/dataset.py` | Time-RCD 训练/推理所需时间序列数据集封装。 |
| `upstream/Time-RCD/time_rcd/_core/full_reconstruction.py` | 掩码重建预训练模型与批处理逻辑。 |
| `upstream/Time-RCD/time_rcd/_core/time_rcd_config.py` | Time-RCD 模型与推理超参数。 |
| `upstream/Time-RCD/time_rcd/_core/TimeRCD_pretrain_multi.py` | 多变量 Time-RCD 预训练数据整理和模型代码。 |
| `upstream/Time-RCD/time_rcd/_core/ts_encoder_bi_bias.py` | Time-RCD 时间序列 Transformer 编码器。 |
| `upstream/Time-RCD/time_rcd/_inference.py` | Time-RCD 推理预处理、分块和异常分数计算。 |
| `upstream/Time-RCD/time_rcd/detector.py` | Time-RCD 用户 API；加载检查点并输出逐时间点异常分数。 |
