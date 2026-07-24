"""Build the audited loss_e2e_0723 Control/Treatment Markdown report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping


METRICS = (
    ("multiple_choice/final", "MC Final"),
    ("multiple_choice/correctness", "MC Corr."),
    ("multiple_choice/reasoning_quality", "MC Rsn."),
    ("open_ended/final", "OE Final"),
    ("open_ended/accuracy", "OE Acc."),
    ("open_ended/completeness", "OE Comp."),
    ("open_ended/relevance", "OE Rel."),
    ("true_false/final", "TF Final"),
    ("true_false/correctness", "TF Corr."),
    ("true_false/justification_quality", "TF Justif."),
    ("macro_final", "Macro Final"),
)
JUDGES = ("gemini", "deepseek")
ARMS = ("control", "treatment")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_download(raw: Path) -> dict:
    manifest = read_json(raw / "download_manifest.json")
    assert manifest["file_count"] == len(manifest["files"])
    assert manifest["total_bytes"] == sum(
        int(row["bytes"]) for row in manifest["files"]
    )
    for row in manifest["files"]:
        path = raw / row["path"]
        assert path.is_file(), path
        assert path.stat().st_size == row["bytes"], path
        assert sha256(path) == row["sha256"], path
    return manifest


def validate_pipeline(raw: Path) -> dict:
    pipeline = read_json(raw / "pipeline_final_audit.json")
    assert pipeline["pipeline_complete"] is True
    assert pipeline["cross_arm"]["temporary_secret_files_absent"] is True
    assert pipeline["cross_arm"]["selection_used_test_or_judge_metrics"] is False
    for arm in ARMS:
        arm_payload = pipeline["arms"][arm]
        assert arm_payload["selected_step"] == 19000
        assert arm_payload["prediction_rows"] == 140
        predictions = (
            raw / f"{arm}_evaluation/predictions/predictions.jsonl"
        )
        assert sha256(predictions) == arm_payload["predictions_sha256"]
        for judge in JUDGES:
            judge_payload = arm_payload["judges"][judge]
            assert judge_payload["score_rows"] == 335
            table = read_json(
                raw / f"{arm}_evaluation/{judge}/table1.json"
            )
            assert table["models"]["base"] == judge_payload["table"]
    return pipeline


def table_metrics(raw: Path, arm: str, judge: str) -> dict[str, float]:
    payload = read_json(raw / f"{arm}_evaluation/{judge}/table1.json")
    values = payload["models"]["base"]
    return {key: float(value) for key, value in values.items()}


def comparison_rows(
    control: Mapping[str, float],
    treatment: Mapping[str, float],
) -> list[dict]:
    rows = []
    for key, label in METRICS:
        baseline = float(control[key])
        candidate = float(treatment[key])
        delta = candidate - baseline
        relative = delta / baseline * 100.0
        rows.append(
            {
                "key": key,
                "metric": label,
                "control": baseline,
                "treatment": candidate,
                "absolute_delta": delta,
                "relative_change_pct": relative,
                "direction": (
                    "improved"
                    if delta > 0
                    else "declined"
                    if delta < 0
                    else "unchanged"
                ),
            }
        )
    return rows


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def pct(value: float) -> str:
    return f"{value:+.2f}%"


def metric_table(rows: list[dict]) -> str:
    lines = [
        "| 指标 | Control | Treatment | 绝对差值 T−C | 相对变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['control']:.4f} | "
            f"{row['treatment']:.4f} | "
            f"{signed(row['absolute_delta'])} | "
            f"{pct(row['relative_change_pct'])} |"
        )
    return "\n".join(lines)


def selection_table(
    control: Mapping,
    treatment: Mapping,
) -> str:
    lines = [
        "| Step | Control validation NLL | Treatment validation NLL | T−C |",
        "|---:|---:|---:|---:|",
    ]
    for step in ("4750", "9500", "14250", "19000"):
        control_value = float(control["candidate_global_token_nll"][step])
        treatment_value = float(treatment["candidate_global_token_nll"][step])
        lines.append(
            f"| {step} | {control_value:.6f} | {treatment_value:.6f} | "
            f"{treatment_value - control_value:+.6f} |"
        )
    return "\n".join(lines)


def judge_default_rows(raw: Path) -> list[dict]:
    path = (
        raw
        / "treatment_evaluation/gemini/geval_gemini25pro_author.jsonl"
    )
    return [
        row
        for row in read_jsonl(path)
        if row["method"] == "author_default_3"
    ]


def generate(raw: Path) -> tuple[str, dict]:
    download = validate_download(raw)
    pipeline = validate_pipeline(raw)
    tables = {
        judge: {
            arm: table_metrics(raw, arm, judge)
            for arm in ARMS
        }
        for judge in JUDGES
    }
    comparisons = {
        judge: comparison_rows(
            tables[judge]["control"],
            tables[judge]["treatment"],
        )
        for judge in JUDGES
    }

    control_selection = read_json(raw / "control_selection_audit.json")
    treatment_selection = read_json(raw / "treatment_selection_audit.json")
    control_training = read_json(raw / "control/training_summary.json")
    treatment_training = read_json(raw / "treatment/training_summary.json")
    control_manifest = read_json(raw / "control/run_manifest.json")
    treatment_manifest = read_json(raw / "treatment/run_manifest.json")
    control_gemini = read_json(raw / "control_gemini_audit.json")
    treatment_gemini = read_json(raw / "treatment_gemini_audit.json")
    control_deepseek = read_json(raw / "control_deepseek_audit.json")
    treatment_deepseek = read_json(raw / "treatment_deepseek_audit.json")

    validation_control = float(
        control_selection["selected_global_token_nll"]
    )
    validation_treatment = float(
        treatment_selection["selected_global_token_nll"]
    )
    validation_delta = validation_treatment - validation_control
    validation_relative = validation_delta / validation_control * 100.0

    control_drift = float(
        control_training["optimization_diagnostics"][
            "parameter_relative_change_by_step"
        ]["19000"]["trainable_perceiver_parameters"]
    )
    treatment_drift = float(
        treatment_training["optimization_diagnostics"][
            "parameter_relative_change_by_step"
        ]["19000"]["trainable_perceiver_parameters"]
    )
    drift_delta_pct = (treatment_drift - control_drift) / control_drift * 100.0
    control_grad = control_training["optimization_diagnostics"]
    treatment_grad = treatment_training["optimization_diagnostics"]
    training_nll_control = float(
        control_training["metrics"]["global_token_nll"]
    )
    training_nll_treatment = float(
        treatment_training["metrics"]["global_token_nll"]
    )
    training_nll_relative = (
        (training_nll_treatment - training_nll_control)
        / training_nll_control
        * 100.0
    )

    direction_lines = [
        "| 指标 | Gemini 相对变化 | DeepSeek 相对变化 | 方向一致性 |",
        "|---|---:|---:|---|",
    ]
    for gemini, deepseek in zip(
        comparisons["gemini"],
        comparisons["deepseek"],
        strict=True,
    ):
        assert gemini["key"] == deepseek["key"]
        same = gemini["direction"] == deepseek["direction"]
        direction_lines.append(
            f"| {gemini['metric']} | "
            f"{pct(gemini['relative_change_pct'])} | "
            f"{pct(deepseek['relative_change_pct'])} | "
            f"{'一致' if same else '不一致'} |"
        )

    default_rows = judge_default_rows(raw)
    assert len(default_rows) == 2
    default_descriptions = "；".join(
        f"`{row['record_id']}/{row['dimension']}`"
        for row in default_rows
    )

    report = f"""# AXIS loss_e2e_0723 双臂实验结果分析

## 技术摘要

- **分段联合损失在整体指标上有效，但不是全题型无条件改进。** Treatment 的 Macro Final 相对 Control 在 Gemini 2.5 Pro 下提高 {comparisons['gemini'][-1]['relative_change_pct']:.2f}%，在 DeepSeek V4 Pro 下提高 {comparisons['deepseek'][-1]['relative_change_pct']:.2f}%。
- **收益集中在 OE 与 TF。** 两位 Judge 对 OE 的 4 个指标和 TF 的 3 个指标均给出正向变化；其中 Gemini 的 TF Final 提高 {next(row for row in comparisons['gemini'] if row['key'] == 'true_false/final')['relative_change_pct']:.2f}%，DeepSeek 提高 {next(row for row in comparisons['deepseek'] if row['key'] == 'true_false/final')['relative_change_pct']:.2f}%。
- **MC 存在明确权衡。** MC Final 和 MC Correctness 在两位 Judge 下均下降；MC Reasoning 仅 Gemini 判为上升，而 DeepSeek 判为下降。因此当前方法不能宣称全面改善 MC。
- **无泄漏选模证据与测试结果方向一致。** seed=72 validation 的 teacher-forced global token NLL 从 {validation_control:.6f} 降至 {validation_treatment:.6f}（{validation_relative:+.2f}%），两臂都由预先声明的四个候选选中 step_19000；选模未读取 paper140 或 Judge 分数。
- **判定：** 若成功标准是提高宏平均并重点改善 OE/TF，则本次改进有效；若成功标准要求 MC/OE/TF 全部不退化，则当前证据不足，下一版应优先修复 MC conclusion/correctness 的回退。

## 两种 Judge 的 Table 1 第一行逐指标对比

本节严格读取四个 `table1.json` 的 `models.base`（Table 1 第一行）。绝对差值定义为 `Treatment − Control`，相对变化定义为 `(Treatment − Control) / Control × 100%`；正值表示 Treatment 更高。

### Gemini 2.5 Pro author-mode

{metric_table(comparisons['gemini'])}

**解读。** Gemini 判断 Macro Final 提升 {comparisons['gemini'][-1]['absolute_delta']:+.4f}（{comparisons['gemini'][-1]['relative_change_pct']:+.2f}%）。OE/TF 全部改善；MC Reasoning 改善，但 MC Final 与 MC Correctness 分别下降 {abs(comparisons['gemini'][0]['relative_change_pct']):.2f}% 和 {abs(comparisons['gemini'][1]['relative_change_pct']):.2f}%。

### DeepSeek V4 Pro

{metric_table(comparisons['deepseek'])}

**解读。** DeepSeek 判断 Macro Final 提升 {comparisons['deepseek'][-1]['absolute_delta']:+.4f}（{comparisons['deepseek'][-1]['relative_change_pct']:+.2f}%）。OE/TF 的收益方向与 Gemini 一致，但 MC 的三个指标全部下降，MC Final 相对下降 {abs(comparisons['deepseek'][0]['relative_change_pct']):.2f}%。

### 两位 Judge 的方向一致性

{chr(10).join(direction_lines)}

**最稳健的结论**是 OE/TF 改善与宏平均提升，因为两位 Judge 方向一致；MC Reasoning 的方向不一致，不能作为确定收益。精确值来自 [Gemini Control](artifacts/control_evaluation/gemini/table1.json)、[Gemini Treatment](artifacts/treatment_evaluation/gemini/table1.json)、[DeepSeek Control](artifacts/control_evaluation/deepseek/table1.json) 和 [DeepSeek Treatment](artifacts/treatment_evaluation/deepseek/table1.json)。

## 无测试泄漏的四里程碑选模

{selection_table(control_selection, treatment_selection)}

- 两臂候选固定为 step_4750 / 9500 / 14250 / 19000。
- 选模集为 seed=72 的 5% validation：1,500 series、3,000 QA rows、376,696 个有效 answer tokens。
- 指标为有效 answer token 加权的 teacher-forced global token NLL，越低越好；两臂均选中 step_19000。
- Treatment 最佳 NLL 比 Control 低 {abs(validation_delta):.6f}（{abs(validation_relative):.2f}%）。四个里程碑上 Treatment 均低于 Control，说明该方向不是由单一 checkpoint 偶然造成。
- paper140 固定使用 seed=42 manifest；两臂均为 140 条 base predictions、335 个 G-Eval 维度分数。

## 实验设计、配置与数据审计

| 项目 | Control | Treatment |
|---|---|---|
| 初始化 | 同一作者 checkpoint（SHA-256 相同） | 同 Control |
| 优化器 | AdamW，weight decay=1e-5 | 同 Control |
| LR 日程 | epoch 1: 1e-4；epoch 2: 3e-5 | 同 Control |
| 冻结 LLM 精度 | epoch 1: FP16；epoch 2: BF16 | 同 Control |
| 训练 | 3×A100 80GB，2 epochs，19,000 steps，无早停 | 同 Control |
| seed | 训练/validation=72；paper140 manifest=42 | 同 Control |
| 目标 | 全局 continuation-token NLL | conclusion 0.40 + explanation 0.60 的分段目标 |
| fixed hint | dynamic | step 0 缓存 F0，shape=30×3584，重复误差=0 |
| 可训练参数 | {control_manifest['trainable_parameters']:,} | {treatment_manifest['trainable_parameters']:,} |
| 训练 QA | 57,000 总行；56,987 纳入；13 排除 | 完全相同的排除 manifest |
| answer tokenization | fast tokenizer；truncation=False；0 个截断 | 同 Control |
| max answer tokens | 234（model limit 16,384） | 同 Control |

题型纳入量为 MC={control_manifest['data_counts']['included_question_type_counts']['multiple_choice']:,}、OE={control_manifest['data_counts']['included_question_type_counts']['open_ended']:,}、TF={control_manifest['data_counts']['included_question_type_counts']['true_false']:,}。Treatment 的确定性切段计数为：MC marker={treatment_manifest['data_counts']['segmentation_rule_counts']['multiple_choice/mc_marker']:,}、MC first line={treatment_manifest['data_counts']['segmentation_rule_counts']['multiple_choice/mc_first_line']:,}、MC blank line={treatment_manifest['data_counts']['segmentation_rule_counts']['multiple_choice/mc_blank_line']:,}、MC first sentence/fallback={treatment_manifest['data_counts']['segmentation_rule_counts']['multiple_choice/mc_first_sentence']:,}、OE first sentence={treatment_manifest['data_counts']['segmentation_rule_counts']['open_ended/oe_first_sentence']:,}、OE short-merge={treatment_manifest['data_counts']['segmentation_rule_counts']['open_ended/oe_short_merge_second']:,}、TF label+first statement={treatment_manifest['data_counts']['segmentation_rule_counts']['true_false/tf_label_plus_first_statement']:,}。

## 训练中间结果与稳定性

| 诊断量 | Control | Treatment | 解释 |
|---|---:|---:|---|
| 训练 global token NLL | {training_nll_control:.6f} | {training_nll_treatment:.6f} | Treatment 低 {abs(training_nll_relative):.2f}%；与 validation 方向一致 |
| Treatment segmented objective | 不适用 | {treatment_training['metrics']['objective']:.6f} | 与 Control token-mean objective 归一化不同，不直接比较绝对值 |
| conclusion NLL | 不适用 | {treatment_training['metrics']['conclusion_nll']:.6f} | conclusion 明显易于 explanation |
| explanation NLL | 不适用 | {treatment_training['metrics']['explanation_nll']:.6f} | 仍是主要误差来源 |
| mean gradient L2 | {control_grad['gradient_l2_norm_mean']:.6f} | {treatment_grad['gradient_l2_norm_mean']:.6f} | 目标归一化不同，只用于臂内稳定性判断 |
| max gradient L2 | {control_grad['gradient_l2_norm_max']:.6f} | {treatment_grad['gradient_l2_norm_max']:.6f} | Treatment 有更尖锐的稀有峰值 |
| final gradient L2 | {control_grad['gradient_l2_norm_last']:.6f} | {treatment_grad['gradient_l2_norm_last']:.6f} | 两者最终均为有限值 |
| non-finite steps | {control_grad['nonfinite_gradient_steps']} | {treatment_grad['nonfinite_gradient_steps']} | 正式轨迹均为 0 |
| step_19000 参数相对漂移 | {control_drift:.6f} | {treatment_drift:.6f} | Treatment 比 Control 高 {drift_delta_pct:.2f}% |
| 吞吐（step/s/rank） | {control_training['steps_per_second_per_rank']:.4f} | {treatment_training['steps_per_second_per_rank']:.4f} | 接近一致 |

Treatment epoch 2 只从完整的 step_9500 epoch 边界恢复，恢复 optimizer 与累计指标，未跳过中途 batch；最终 checkpoint 链记录了该恢复来源。正式汇总的 19,000 个 gradient norm 全部有限。较大的参数漂移与 MC 退化同时出现，提示下一版需要约束 conclusion 权重或加入臂内 drift/MC proxy 监控，但这只是诊断关联，不能据此声称因果。

## Judge 协议与完整性审计

| Judge / Arm | Scores | 方法 | Logprobs | 完整性 |
|---|---:|---|---:|---|
| Gemini / Control | 335 | temperature=0 整数：335 | {control_gemini['judge']['logprobs_returned_count']} | final_audit ok=true，键唯一 |
| Gemini / Treatment | 335 | temperature=0 整数：333；默认 3：2 | {treatment_gemini['judge']['logprobs_returned_count']} | final_audit ok=true，键唯一 |
| DeepSeek / Control | 335 | top-logprobs：333；20 次采样均值：2 | {control_deepseek['judge']['logprobs_returned_count']} | final_audit ok=true，键唯一 |
| DeepSeek / Treatment | 335 | top-logprobs：334；20 次采样均值：1 | {treatment_deepseek['judge']['logprobs_returned_count']} | final_audit ok=true，键唯一 |

- Gemini 使用 author prompt 与 author scoring；兼容接口未返回 score-token logprobs。Control 的 335 条均为 temperature=0 整数；Treatment 有 2 条服务回退为默认 3：{default_descriptions}。
- 每个受影响维度有 55 个 OE 样本；将默认 3 的真实值在 1–5 极端范围内变化，只会使对应 component mean 变化 ±0.0364。即使取最不利值，Treatment 的 OE Completeness 与 Relevance 相对 Control 仍保持正向，因此核心 OE 方向结论不变。
- DeepSeek 使用官方端点：Treatment 为 334 条 top-logprobs + 1 条精确 20 次采样均值；Control 为 333 + 2。所有 fallback 都恰好 20 个有效整数分，符合预定协议。
- 两臂 Judge 各自读取同一个 paper140 manifest，但 predictions 分别独立生成；每个 Judge 内 Control/Treatment 使用同一模型、prompt 代码和聚合脚本。

## 限制、不确定性与结论边界

1. 本报告是固定 seed、固定 paper140 的描述性双臂比较，没有多 seed 方差或预注册显著性检验；不能把百分比变化等同于统计显著性。
2. Gemini 两条默认 3 造成轻微的臂间方法不完全对称，尽管极端界限分析不改变 OE 的方向。
3. Gemini 与 DeepSeek 对 MC Reasoning 的方向不一致，说明该维度对 Judge 选择敏感；MC 的结论应以两者共同确认的 Final/Correctness 下降为主。
4. Treatment 的训练 objective 是分段 row-mean，Control 是 token-mean；两者 objective 绝对值不能直接比较。可比较的选模指标是两臂统一计算的 validation global token NLL。
5. 本实验只验证当前 alpha=0.40 与两阶段精度/LR 日程；不能外推到其他 alpha、学习率、epoch 或未冻结 LLM。

## 结论与下一步

**当前损失设计通过“宏平均 + OE/TF 定向改善”的有效性标准，但没有通过“所有题型不退化”的更强标准。** 下一步建议按以下优先级推进：

1. 在不使用 paper140 的前提下，为 MC 增加 seed=72 validation proxy（例如 forced-choice label NLL/accuracy），与 global token NLL 共同监控。
2. 只在 validation 上扫描 MC conclusion 权重或采用题型自适应 alpha，目标是保留 OE/TF 收益同时回收 MC Correctness。
3. 增加至少 3 个训练 seed，并对同一 paper140 记录执行 paired bootstrap 或随机化检验，报告置信区间。
4. 对参数漂移设置预声明诊断阈值，保留 no-clipping 主实验，同时增加受控 clipping/regularization 消融。
5. 若 Gemini 接口仍无 logprobs，预先规定失败重试与缺失值策略，避免单臂出现 `author_default_3`。

## 数据与复现材料

本地保存了 {download['file_count']} 个服务器结果文件，共 {download['total_bytes']:,} bytes；下载清单逐文件记录大小和 SHA-256。明确排除了模型 checkpoint、PID 和所有临时密钥文件。主要入口：

- [全链路最终审计](artifacts/pipeline_final_audit.json)
- [可发布证据清单与 SHA-256](artifacts/artifact_manifest.json)
- [Control 训练审计](artifacts/control_training_audit.json) / [Treatment 训练审计](artifacts/treatment_training_audit.json)
- [Control 选模审计](artifacts/control_selection_audit.json) / [Treatment 选模审计](artifacts/treatment_selection_audit.json)
- [Control 推理审计](artifacts/control_inference_audit.json) / [Treatment 推理审计](artifacts/treatment_inference_audit.json)
- [Control Gemini 审计](artifacts/control_gemini_audit.json) / [Treatment Gemini 审计](artifacts/treatment_gemini_audit.json)
- [Control DeepSeek 审计](artifacts/control_deepseek_audit.json) / [Treatment DeepSeek 审计](artifacts/treatment_deepseek_audit.json)
- 完整 predictions、335-row score JSONL、训练日志、选模中间 JSON、Table 1 Markdown/JSON 与 manifests 均保存在同目录的 `artifacts/` 树中；发布树将服务器绝对路径统一替换为 `<remote-root>`，且不含密钥、checkpoint 或 PID 文件。

定量信息使用表格而非图形，是因为本报告只有两个实验臂、两个 Judge 和固定的离散指标集合；精确值、差值及方向一致性比图形位置更适合审计。
"""

    derived = {
        "schema_version": 1,
        "experiment": "loss_e2e_0723",
        "comparison_definition": {
            "absolute_delta": "treatment - control",
            "relative_change_pct": (
                "(treatment - control) / control * 100"
            ),
        },
        "table1_comparisons": comparisons,
        "selection": {
            "control": control_selection,
            "treatment": treatment_selection,
            "selected_nll_absolute_delta": validation_delta,
            "selected_nll_relative_change_pct": validation_relative,
        },
        "training_diagnostics": {
            "control_global_token_nll": training_nll_control,
            "treatment_global_token_nll": training_nll_treatment,
            "global_token_nll_relative_change_pct": training_nll_relative,
            "control_step19000_parameter_drift": control_drift,
            "treatment_step19000_parameter_drift": treatment_drift,
            "parameter_drift_relative_change_pct": drift_delta_pct,
        },
        "judge_caveats": {
            "treatment_gemini_author_default_3": [
                {
                    "record_id": row["record_id"],
                    "dimension": row["dimension"],
                    "score": row["score"],
                }
                for row in default_rows
            ],
            "control_deepseek_methods": control_deepseek["judge"][
                "method_counts"
            ],
            "treatment_deepseek_methods": treatment_deepseek["judge"][
                "method_counts"
            ],
        },
        "source_pipeline_audit_sha256": sha256(
            raw / "pipeline_final_audit.json"
        ),
        "source_download_manifest_sha256": sha256(
            raw / "download_manifest.json"
        ),
    }
    return report, derived


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--raw", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    result.add_argument("--derived", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    report, derived = generate(args.raw)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    args.derived.write_text(
        json.dumps(derived, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "report": str(args.report),
        "derived": str(args.derived),
        "report_sha256": sha256(args.report),
        "derived_sha256": sha256(args.derived),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
