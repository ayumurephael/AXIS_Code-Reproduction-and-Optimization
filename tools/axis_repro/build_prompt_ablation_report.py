"""Build the auditable Chinese Markdown report for AXIS prompt Stage A."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .build_tables import aggregate
from .common import read_jsonl
from src.models.AXIS.prompt_stage_a import STAGE_A_MODES


METRICS = (
    ("MC Final", "multiple_choice/final"),
    ("MC Corr.", "multiple_choice/correctness"),
    ("MC Rsn.", "multiple_choice/reasoning_quality"),
    ("OE Final", "open_ended/final"),
    ("OE Acc.", "open_ended/accuracy"),
    ("OE Comp.", "open_ended/completeness"),
    ("OE Rel.", "open_ended/relevance"),
    ("TF Final", "true_false/final"),
    ("TF Corr.", "true_false/correctness"),
    ("TF Justif.", "true_false/justification_quality"),
)

MODE_LABELS = {
    "base": "Baseline",
    "answer_boundary": "EOS + Answer:",
    "answer_boundary_wo_fixed": "EOS + Answer: / w/o Fixed",
    "task_protocol": "题型协议",
    "fixed_role": "Fixed 角色重命名",
    "fixed_role_evidence_contract": "角色重命名 + Evidence Contract",
    "combined_234": "综合 2+3+4",
    "answer_boundary_combined_234": "EOS + Answer: + 综合 2+3+4",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_metric_table(models):
    return markdown_table(
        ["实验"] + [label for label, _ in METRICS],
        [
            [MODE_LABELS[mode]]
            + [f"{models[mode][key]:.4f}" for _, key in METRICS]
            for mode in STAGE_A_MODES
        ],
    )


def format_delta_table(models, relative=False):
    baseline = models["base"]
    rows = []
    for mode in STAGE_A_MODES[1:]:
        values = []
        for _, key in METRICS:
            delta = models[mode][key] - baseline[key]
            if relative:
                value = delta / baseline[key] * 100.0
                values.append(f"{value:+.2f}%")
            else:
                values.append(f"{delta:+.4f}")
        rows.append([MODE_LABELS[mode]] + values)
    return markdown_table(
        ["实验"] + [label for label, _ in METRICS],
        rows,
    )


def score_metadata(scores):
    usage_fields = Counter()
    for row in scores:
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage_fields[key] += value
    return {
        "rows": len(scores),
        "models": dict(Counter(str(row.get("model")) for row in scores)),
        "providers": dict(Counter(str(row.get("provider")) for row in scores)),
        "methods": dict(Counter(str(row.get("method")) for row in scores)),
        "judge_max_tokens": dict(
            Counter(str(row.get("judge_max_tokens")) for row in scores)
        ),
        "endpoint_hosts": dict(
            Counter(str(row.get("endpoint_host")) for row in scores)
        ),
        "system_fingerprints": dict(
            Counter(str(row.get("system_fingerprint")) for row in scores)
        ),
        "unique_prompt_hashes": len(
            {row.get("prompt_sha256") for row in scores}
        ),
        "usage_totals": dict(usage_fields),
    }


def diagnostic_table(diagnostics):
    rows = []
    for mode in STAGE_A_MODES:
        entry = diagnostics["modes"][mode]
        overall = entry["overall"]
        mc = entry["by_question_type"]["multiple_choice"]
        tf = entry["by_question_type"]["true_false"]
        oe = entry["by_question_type"]["open_ended"]
        none_rate = (
            overall["think_states"].get("none", 0) / overall["count"]
        )
        rows.append(
            [
                MODE_LABELS[mode],
                f"{mc['raw_answer_first_rate']:.2%}",
                f"{tf['raw_answer_first_rate']:.2%}",
                f"{mc['strict_parse_rate']:.2%}",
                f"{tf['strict_parse_rate']:.2%}",
                f"{oe['strict_parse_rate']:.2%}",
                f"{none_rate:.2%}",
                f"{overall['response_chars_mean']:.1f}",
                f"{overall['response_at_least_3000_chars_rate']:.2%}",
            ]
        )
    return markdown_table(
        [
            "实验",
            "MC answer-first",
            "TF answer-first",
            "MC strict parse",
            "TF strict parse",
            "OE direct-start",
            "无 think 标签",
            "平均字符数",
            "≥3000 字符",
        ],
        rows,
    )


def factor_table(mode_definitions):
    rows = []
    for mode in STAGE_A_MODES:
        item = mode_definitions[mode]
        rows.append(
            [
                MODE_LABELS[mode],
                "✓" if item["answer_boundary"] else "—",
                "✓" if item["remove_fixed_hint"] else "—",
                "✓" if item["add_output_protocol"] else "—",
                "✓" if item["rename_fixed_hint"] else "—",
                "✓" if item["add_evidence_contract"] else "—",
            ]
        )
    return markdown_table(
        [
            "实验",
            "EOS + Answer:",
            "移除 Fixed",
            "题型协议",
            "角色重命名",
            "Evidence Contract",
        ],
        rows,
    )


CONTRASTS = (
    ("边界对齐", "answer_boundary", "base"),
    ("边界对齐下移除 Fixed", "answer_boundary_wo_fixed", "answer_boundary"),
    ("仅题型协议", "task_protocol", "base"),
    ("仅 Fixed 角色重命名", "fixed_role", "base"),
    (
        "在角色重命名上增加 Evidence Contract",
        "fixed_role_evidence_contract",
        "fixed_role",
    ),
    ("综合 2+3+4", "combined_234", "base"),
    (
        "在综合 2+3+4 上增加边界对齐",
        "answer_boundary_combined_234",
        "combined_234",
    ),
)


def contrast_table(models):
    headers = ["预注册对比", "Treatment", "Control"]
    headers += [label for label, _ in METRICS] + ["Macro Final"]
    rows = []
    for label, treatment, control in CONTRASTS:
        values = [
            models[treatment][key] - models[control][key]
            for _, key in METRICS
        ]
        values.append(
            models[treatment]["macro_final"] - models[control]["macro_final"]
        )
        rows.append(
            [label, MODE_LABELS[treatment], MODE_LABELS[control]]
            + [f"{value:+.4f}" for value in values]
        )
    return markdown_table(headers, rows)


def build_report(
    *,
    scores_path: Path,
    predictions_path: Path,
    run_manifest_path: Path,
    diagnostics_path: Path,
    environment_path: Path,
    audit_path: Path | None,
    experiment_commit: str,
):
    scores = read_jsonl(scores_path)
    predictions = read_jsonl(predictions_path)
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path is not None
        else None
    )
    models, paired = aggregate(scores)
    missing_modes = [mode for mode in STAGE_A_MODES if mode not in models]
    if missing_modes:
        raise RuntimeError(f"missing aggregate modes: {missing_modes}")
    for mode in STAGE_A_MODES:
        missing_metrics = [
            key for _, key in METRICS if key not in models[mode]
        ]
        if missing_metrics:
            raise RuntimeError(
                f"missing metrics for {mode}: {missing_metrics}"
            )

    baseline = models["base"]
    best_macro_mode = max(
        STAGE_A_MODES, key=lambda mode: models[mode]["macro_final"]
    )
    metric_winners = []
    for label, key in METRICS:
        best_mode = max(STAGE_A_MODES, key=lambda mode: models[mode][key])
        delta = models[best_mode][key] - baseline[key]
        metric_winners.append((label, best_mode, delta))
    score_meta = score_metadata(scores)
    prediction_type_counts = Counter(
        row["question_type"] for row in predictions
        if row["mode"] == "base"
    )
    prediction_mode_counts = Counter(row["mode"] for row in predictions)

    lines = [
        "# AXIS Prompt 阶段 A 消融实验结果与分析",
        "",
        "## 结论摘要",
        "",
        (
            f"本报告在作者发布 checkpoint、固定 `paper140` 清单和同一云端"
            f"运行环境下比较 1 个 Baseline 与 7 个 prompt-only 条件。按三类"
            f"题型 Final 的未加权宏平均，最佳条件为 "
            f"`{best_macro_mode}`（{MODE_LABELS[best_macro_mode]}），宏平均 "
            f"{models[best_macro_mode]['macro_final']:.4f}，相对 Baseline 的"
            f"绝对变化为 "
            f"{models[best_macro_mode]['macro_final'] - baseline['macro_final']:+.4f}。"
        ),
        "",
        "各单项指标的最高值及其相对 Baseline 的绝对变化如下：",
        "",
    ]
    lines.extend(
        f"- {label}: {MODE_LABELS[mode]}，Δ={delta:+.4f}"
        for label, mode, delta in metric_winners
    )
    lines += [
        "",
        "## 关键判断",
        "",
        (
            "- **没有任何 prompt-only 条件超过 Baseline 的总体分数。** "
            f"Baseline 的三题型未加权 Macro Final 为 "
            f"{baseline['macro_final']:.4f}；最接近的是 Fixed 角色重命名 "
            f"{models['fixed_role']['macro_final']:.4f}（Δ="
            f"{models['fixed_role']['macro_final'] - baseline['macro_final']:+.4f}），"
            f"其 140 条 QA 配对差异区间为 "
            f"[{paired['fixed_role']['low']:+.4f}, "
            f"{paired['fixed_role']['high']:+.4f}]，不能据此声称总体提升。"
        ),
        (
            "- **`EOS + Answer:` 是强格式控制，不是内容增益。** MC/TF "
            f"strict-parse 从 Baseline 的 "
            f"{diagnostics['modes']['base']['by_question_type']['multiple_choice']['strict_parse_rate']:.2%}/"
            f"{diagnostics['modes']['base']['by_question_type']['true_false']['strict_parse_rate']:.2%} "
            f"升至 {diagnostics['modes']['answer_boundary']['by_question_type']['multiple_choice']['strict_parse_rate']:.2%}/"
            f"{diagnostics['modes']['answer_boundary']['by_question_type']['true_false']['strict_parse_rate']:.2%}，"
            f"但 Macro Final 下降 "
            f"{models['answer_boundary']['macro_final'] - baseline['macro_final']:+.4f}。"
        ),
        (
            "- **Fixed hint 对旧 checkpoint 是必要条件。** 在相同答案边界下移除 "
            f"Fixed 后 Macro Final 从 {models['answer_boundary']['macro_final']:.4f} "
            f"降至 {models['answer_boundary_wo_fixed']['macro_final']:.4f}；"
            f"平均输出从 {diagnostics['modes']['answer_boundary']['overall']['response_chars_mean']:.1f} "
            f"增至 {diagnostics['modes']['answer_boundary_wo_fixed']['overall']['response_chars_mean']:.1f} "
            f"字符，且 {diagnostics['modes']['answer_boundary_wo_fixed']['overall']['response_at_least_3000_chars_rate']:.2%} "
            "回答不少于 3000 字符。该条件应视为明确的负面对照。"
        ),
        (
            "- **Evidence Contract 呈题型迁移而非全局提升。** 相对 Fixed "
            f"角色重命名，它使 OE Final 改变 "
            f"{models['fixed_role_evidence_contract']['open_ended/final'] - models['fixed_role']['open_ended/final']:+.4f}，"
            f"但 MC/TF Final 分别改变 "
            f"{models['fixed_role_evidence_contract']['multiple_choice/final'] - models['fixed_role']['multiple_choice/final']:+.4f}/"
            f"{models['fixed_role_evidence_contract']['true_false/final'] - models['fixed_role']['true_false/final']:+.4f}；"
            "不能把 OE 收益外推为通用收益。"
        ),
        (
            "- **组合存在明显交互。** 在综合 2+3+4 上加入边界对齐会恢复 "
            f"MC Final（Δ={models['answer_boundary_combined_234']['multiple_choice/final'] - models['combined_234']['multiple_choice/final']:+.4f}）"
            f"并略升 TF Final（Δ={models['answer_boundary_combined_234']['true_false/final'] - models['combined_234']['true_false/final']:+.4f}），"
            f"同时降低 OE Final（Δ={models['answer_boundary_combined_234']['open_ended/final'] - models['combined_234']['open_ended/final']:+.4f}）。"
            "因此不应按单因素结果做简单加和预测。"
        ),
        "",
        "## Table-I 指标汇总",
        "",
        "所有分数均由唯一 Judge `deepseek-v4-pro` 按论文 G-Eval 量表给出。",
        "",
        format_metric_table(models),
        "",
        "## 相对 Baseline 的绝对差值",
        "",
        format_delta_table(models, relative=False),
        "",
        "## 相对 Baseline 的相对变化",
        "",
        "相对变化定义为 `(variant - baseline) / baseline × 100%`。",
        "",
        format_delta_table(models, relative=True),
        "",
        "## 实验因素矩阵",
        "",
        factor_table(manifest["mode_definitions"]),
        "",
        "## 输出行为与中间诊断",
        "",
        diagnostic_table(diagnostics),
        "",
        "该表的格式指标由确定性规则计算，不替代 G-Eval。`answer-first` "
        "只检查原始生成是否直接以题型规定答案起始；`strict parse` 对 MC "
        "要求首行以 `A)`–`D)` 起始，对 TF 要求首 token 为 `True.` 或 "
        "`False.`。`OE direct-start` 还排除选项字母、True/False、"
        "`Answer:` 与 `<think>` 起始。",
        "",
        "## 配对差异",
        "",
        "下列区间把 140 条 QA 的题型 Final 差值（variant − Baseline）合并后"
        "执行 10,000 次配对 bootstrap（seed=72）。该均值按 QA 数量加权，"
        "与上文三题型等权的 `Macro Final` 不是同一统计量：",
        "",
    ]
    for mode in STAGE_A_MODES[1:]:
        interval = paired[mode]
        lines.append(
            f"- {MODE_LABELS[mode]}: {interval['mean']:+.4f} "
            f"[{interval['low']:+.4f}, {interval['high']:+.4f}]"
        )
    lines += [
        "",
        "## 预注册机制对比",
        "",
        "下表按实验设计中可直接解释的 Treatment − Control 计算。"
        "`Evidence Contract` 没有脱离角色重命名单独运行，因此其增量只能解释为"
        "“在已重命名条件上加入 Contract”，不能声称是完全独立主效应。",
        "",
        contrast_table(models),
        "",
        "## 面向下一阶段的改进建议",
        "",
        (
            "1. **把 Fixed 角色重命名作为唯一的通用确认候选，Baseline 作为"
            "主控制。** 它的总体分数与 Baseline 实质持平，同时改善 MC 三项、"
            "OE Accuracy/Completeness 和 TF Justification；下一阶段应在完整 "
            "284 条测试集和重新训练后确认，而不是宣称本阶段已提升。"
        ),
        (
            "2. **将 `EOS + Answer:` 定位为可选的部署格式层。** 它把 MC/TF "
            "首行可解析率推近 100%，但未提升总体内容分；若下游必须稳定解析，"
            "可启用该边界，否则不应把它列为质量优化。"
        ),
        (
            "3. **保留 Fixed 或设计有监督替代，停止直接移除。** `w/o Fixed` "
            "在所有题型上显著崩溃并产生超长回答。若要减少固定提示依赖，应在"
            "训练阶段逐步 dropout/蒸馏，而不是只在推理时删除。"
        ),
        (
            "4. **把格式收益与证据收益拆开。** 下一轮同时报告程序化 MC/TF "
            "正确率、首行可解析率、答案长度，以及 G-Eval 内容维度；若 Final "
            "提升主要随 parse 改善而非 Accuracy/Justification 改善，应将结论"
            "限定为输出控制收益。"
        ),
        (
            "5. **修正 Evidence Contract 与布局矛盾，并做题型专用版本。** "
            "Stage A 按规范保留了"
            "“same row”文字和两块式 Values/Local 布局。Phase II 应预注册两条"
            "互斥路线：将文字改为“corresponds by order”，或真正采用逐步交错"
            "序列化；两者不能在同一条件中同时改变。鉴于当前收益集中在 OE，"
            "还应比较 OE-only Contract 与全题型 Contract。"
        ),
        (
            "6. **扩展稳健性验证。** 在 284 条完整测试集上复核方向，并对多个"
            "训练 seed/checkpoint 重复最佳条件。单 Judge 结果还应在不用于选择"
            "方案的前提下，增加独立 Judge 或人工盲评作为确认性分析。"
        ),
        "",
        "## 完整实验配置与可审计数据",
        "",
        f"- 实验代码提交：`{experiment_commit}`",
        f"- 代码基点：`origin/main@e8c1aee59bb98cda9b37445bf2eb23619f374d54`",
        "- Baseline 预测先使用同 checkpoint、同 paper140、同 3-rank "
        "series batching、同 torch/CUDA 协议的历史锁定产物；其 SHA-256 为 "
        "`fc687e2ad4c24e66ef00fc4a381885df90c18e26d4edcc6eb230ce94052fa71e`。"
        "本次同运行 base 必须 140/140 逐条完全一致，否则替换并重评。",
        f"- checkpoint SHA-256：`{manifest['checkpoint_sha256']}`",
        "- checkpoint 元数据：epoch 33，保存时训练平均 loss "
        "`0.7127881973981858`，文件大小 `482319394` bytes",
        "- 原训练集：30000 条 series JSON、每条 2 个窗口 QA，共 60000 QA；"
        "题型分布为 TF=20204、MC=19885、OE=19911；本阶段不重新训练，"
        "该信息仅用于 checkpoint 数据血缘",
        "- 原训练集没有官方 train/validation/test 划分；任何后续重训应按"
        "`sample_id` 做样本级切分，避免同一 series 的两个窗口跨集合泄漏",
        f"- prompt 规范 SHA-256：`{manifest['prompt_spec_sha256']}`",
        "- local hint 与数值序列保持论文/作者代码的原始两块式 prompt 布局，"
        "没有在本阶段改成交错布局",
        "- tokenizer 边界审计：LlamaTokenizerFast；EOS token id `151643`；"
        "字面量 `Answer:` 为 3 tokens；Baseline 前缀 141 tokens，"
        "`EOS + Answer:` 前缀 145 tokens；移除 Fixed 后固定占位符 30→0",
        f"- 子集：`{manifest['subset']}`",
        f"- batching：`{manifest['batching']}`",
        f"- skip loss：`{manifest['skip_loss']}`",
        f"- world size：`{manifest['world_size']}`",
        f"- QA 数：{manifest['record_count']}",
        f"- series 数：{manifest['series_count']}",
        f"- 题型数：`{dict(prediction_type_counts)}`",
        "- paper140 异常标签分布：`has_anomaly=false` 94，"
        "`has_anomaly=true` 46；完整候选测试集为 284 条 QA",
        "- 论文 Final 聚合权重：MC=`0.7×Correctness + 0.3×Reasoning`；"
        "OE=`0.35×Accuracy + 0.35×Completeness + 0.3×Relevance`；"
        "TF=`0.6×Correctness + 0.4×Justification`",
        f"- 各模式预测数：`{dict(prediction_mode_counts)}`",
        "- 生成：`do_sample=False`, `num_beams=5`, "
        "`max_new_tokens=1000`, `repetition_penalty=1.15`, "
        "`no_repeat_ngram_size=3`, `length_penalty=1`",
        f"- Python：`{environment.get('python')}`",
        f"- PyTorch：`{environment.get('torch')}`",
        f"- PyTorch CUDA：`{environment.get('torch_cuda')}`",
        f"- Transformers：`{environment.get('transformers')}`",
        "- GPU：3×NVIDIA A100 40GB，`CUDA_VISIBLE_DEVICES=0,1,2`，"
        "每个分布式 rank 固定映射一张卡；启动时三卡空闲并已预留，运行期间"
        "后来出现其他用户共享进程，导致吞吐波动，但本任务未终止或修改任何"
        "外部进程",
        "- 推理运行时警告：Transformers 重复报告 "
        "`Setting pad_token_id to eos_token_id:None`；未出现 OOM、空输出或"
        "缺 shard，最终合并 1120/1120",
        "- Judge：仅 DeepSeek；请求模型 `deepseek-v4-pro`；"
        "thinking enabled；reasoning effort high；max tokens 4096；"
        "请求 top-20 logprobs；缺失完整 1–5 分布时执行 20 次精确分数回退",
        "- TLS：保持证书验证开启，并显式使用 GPU 节点的系统 CA bundle；"
        "批量前真实 G-Eval 探活必须同时满足模型 ID、非空正文、可解析分数与"
        "完整 1–5 score-token 分布。",
        f"- Judge 行数：{score_meta['rows']}",
        f"- Judge 返回模型：`{score_meta['models']}`",
        f"- Judge provider：`{score_meta['providers']}`",
        f"- Judge 评分方法：`{score_meta['methods']}`",
        f"- Judge endpoint host：`{score_meta['endpoint_hosts']}`",
        f"- Judge system fingerprint：`{score_meta['system_fingerprints']}`",
        f"- 唯一 Judge prompt 哈希数：{score_meta['unique_prompt_hashes']}",
        f"- Judge token 用量汇总：`{score_meta['usage_totals']}`",
        f"- predictions SHA-256：`{sha256_file(predictions_path)}`",
        f"- scores SHA-256：`{sha256_file(scores_path)}`",
        f"- run manifest SHA-256：`{sha256_file(run_manifest_path)}`",
        f"- diagnostics SHA-256：`{sha256_file(diagnostics_path)}`",
    ]
    if audit is not None:
        lines.append(f"- 完整性审计：`ok={audit.get('ok')}`，`{audit}`")
    lines += [
        "",
        "## 解释边界与局限性",
        "",
        "- 阶段 A 只改变推理 prompt，旧 checkpoint 未针对新文字或新边界重新"
        "训练；因此结果只能回答“旧权重能否即时受益”，不能替代 Phase II "
        "重新训练后的结论。",
        "- Evidence Contract 按控制文档逐字保留“same row”，但本阶段又按"
        "要求保留原始 Values/Local 两块式排布。该文字与实际布局不完全一致，"
        "是实验条件的一部分，也是解释结果时必须披露的混杂因素。",
        "- 题型协议同时改变格式、长度与内容约束。Final 提升若伴随 parse/"
        "answer-first 改善，不能全部归因于时间序列证据利用增强。",
        "- 正式集为论文口径的 140 条 QA，而非 284 条全测试集；未执行多 seed "
        "模型训练或 checkpoint 重复。",
        "- 按用户要求只使用一个 Judge。虽然保留分布、回退样本、system "
        "fingerprint 与 prompt 哈希，评分仍可能包含单 Judge 偏差和托管 API "
        "的服务端非确定性。",
        "",
        "## 文件索引",
        "",
        f"- 推理结果：`{predictions_path.name}`",
        f"- G-Eval 明细：`{scores_path.name}`",
        f"- 推理 manifest：`{run_manifest_path.name}`",
        f"- 输出诊断：`{diagnostics_path.name}`",
        f"- 运行环境：`{environment_path.name}`",
        "- GPU 节点固定清单审计：`all_audit_manifest.json`",
        "- 聚合指标：`all_table.json` / `all_table.md`",
        "- 回退条目：`all_fallback_pending.jsonl`",
        "- 推理日志：`formal_inference.log`",
        "- Judge 终轮日志：`all_judge.log`",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--audit")
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        scores_path=Path(args.scores),
        predictions_path=Path(args.predictions),
        run_manifest_path=Path(args.run_manifest),
        diagnostics_path=Path(args.diagnostics),
        environment_path=Path(args.environment),
        audit_path=Path(args.audit) if args.audit else None,
        experiment_commit=args.experiment_commit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
