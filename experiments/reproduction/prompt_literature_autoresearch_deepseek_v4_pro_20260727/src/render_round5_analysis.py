"""Render the frozen Round-5 results and the Chinese main-report appendix."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "experiments/reproduction/"
    "prompt_literature_autoresearch_deepseek_v4_pro_20260727"
)
ROUND = ROOT / "experiments/literature-round-5"
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
SPLITS = (
    ("development96", "development_pooled96"),
    ("screening24", "development_screening24"),
    ("validation72", "development_validation72"),
    ("exposed holdout48", "exposed_holdout48"),
)
MODES = (
    "lit_r4_01_tf_neg_re2",
    "lit_r5_01_mc_structured_guard",
    "lit_r5_02_mc_status_then_shape",
    "lit_r5_03_joint_structured_tf_neg_re2",
    "lit_r5_04_joint_status_tf_neg_re2",
)


def load_summary(result_dir: str) -> dict:
    path = ROUND / "results" / result_dir / "screening_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def by_mode(summary: dict) -> dict[str, dict]:
    return {row["mode"]: row for row in summary["rankings"]}


def delta_table(summary: dict) -> list[str]:
    rows = by_mode(summary)
    lines = [
        "| Mode | Nonneg. | Worst Δ | Mean Δ | c→w | w→c | "
        + " | ".join(name for name, _ in METRICS)
        + " |",
        "|" + "---|" * 16,
    ]
    for mode in MODES:
        row = rows[mode]
        pair = summary["pair_summary"][mode]
        values = [
            mode,
            f"{row['nonnegative_dimensions']}/10",
            f"{row['worst_delta']:+.4f}",
            f"{row['mean_delta']:+.4f}",
            str(pair["correct_to_wrong"]),
            str(pair["wrong_to_correct"]),
        ]
        values.extend(f"{row['deltas'][key]:+.4f}" for _, key in METRICS)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def absolute_table(summary: dict) -> list[str]:
    rows = by_mode(summary)
    lines = [
        "| Mode | " + " | ".join(name for name, _ in METRICS) + " |",
        "|" + "---|" * 11,
    ]
    for mode in ("base",) + MODES:
        metrics = summary["models"]["base"] if mode == "base" else rows[mode]["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [mode] + [f"{metrics[key]:.4f}" for _, key in METRICS]
            )
            + " |"
        )
    return lines


def gate_table(summaries: dict[str, dict]) -> list[str]:
    lines = [
        "| Mode | development96 | screening24 | validation72 | "
        "exposed holdout48 | Eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        cells = []
        eligible = True
        for label, _ in SPLITS:
            summary = summaries[label]
            row = by_mode(summary)[mode]
            c2w = summary["pair_summary"][mode]["correct_to_wrong"]
            passed = row["all_nonnegative"] and c2w == 0
            eligible &= passed
            cells.append(
                ("PASS" if passed else "FAIL")
                + f" ({row['nonnegative_dimensions']}/10, c→w={c2w})"
            )
        lines.append(
            "| "
            + " | ".join([mode] + cells + ["YES" if eligible else "NO"])
            + " |"
        )
    return lines


def paired_case(result_dir: str, mode: str, record_id: str) -> dict:
    path = ROUND / "results" / result_dir / "paired_cases.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["mode"] == mode and row["record_id"] == record_id:
            return row
    raise KeyError((result_dir, mode, record_id))


def compact(text: str, limit: int = 420) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def render_analysis(summaries: dict[str, dict]) -> str:
    dev = summaries["development96"]
    holdout = summaries["exposed holdout48"]
    case_103 = paired_case(
        "development_pooled96",
        "lit_r5_01_mc_structured_guard",
        "series_000103:1",
    )
    case_083 = paired_case(
        "development_pooled96",
        "lit_r5_02_mc_status_then_shape",
        "series_000083:1",
    )
    case_111 = paired_case(
        "exposed_holdout48",
        "lit_r4_01_tf_neg_re2",
        "series_000111:1",
    )
    case_022 = paired_case(
        "exposed_holdout48",
        "lit_r5_01_mc_structured_guard",
        "series_000022:1",
    )
    lines = [
        "# Literature Round 5 — conservative decision calibration",
        "",
        "## Outcome",
        "",
        "Round 5 completed all frozen GPU inference and `deepseek-v4-pro` Judge "
        "work. No candidate passed every preregistered gate. The registered "
        "30-new-mode stopping rule therefore fired, no winner was locked, and "
        "`paper140` candidate outputs were not generated.",
        "",
        "This is stronger than an average-score rejection: several routes had "
        "all ten nonnegative aggregate deltas, but each failed either a held-out "
        "split or the zero correct→wrong guard. The released Baseline remains "
        "the only verified strict no-regression deployment prompt.",
        "",
        "## Frozen execution",
        "",
        "- Checkpoint SHA-256: "
        "`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.",
        "- Source commit used by GPU inference: `4d3ace60c036cdb231a4c5aeafeec3cdf030aace`.",
        "- Inference: authorized GPU only, one selected GPU, series batching, "
        "beam size 5, `--skip-loss`; no local GPU and no training.",
        "- Judge: `deepseek-v4-pro` only, OpenAI-compatible official endpoint "
        "recorded as `api.deepseek.com` in every new score row.",
        "- Development: 192 raw predictions; 83 routed components; 68 newly "
        "judged predictions / 136 dimensions (135 top-logprob, 1 exact-20).",
        "- Holdout: 144 raw predictions; 44 routed components; 44 newly judged "
        "predictions / 88 dimensions (all top-logprob).",
        "- Audited assembled totals: development 576 predictions / 1326 scores; "
        "holdout 288 predictions / 666 scores.",
        "- Exact reuse: unchanged MC/OE/TF records reuse canonical Baseline "
        "responses and scores; development TF reuse comes from the already "
        "audited Round-4 component.",
        "",
        "## Gate matrix",
        "",
        *gate_table(summaries),
        "",
        "A cell is `PASS` only when all ten split-level Table-I deltas are "
        "nonnegative **and** no Baseline-correct parsed MC/TF decision becomes "
        "wrong. No row is eligible.",
        "",
    ]
    for label, _ in SPLITS:
        summary = summaries[label]
        lines.extend(
            [
                f"## {label}: delta versus exact split Baseline",
                "",
                *delta_table(summary),
                "",
            ]
        )
    lines.extend(
        [
            "## Absolute scores — development96",
            "",
            *absolute_table(dev),
            "",
            "## Absolute scores — exposed holdout48",
            "",
            *absolute_table(holdout),
            "",
            "## Concrete failure and success cases",
            "",
            "### Failure 1 — the structured guard suppresses a true central spike",
            "",
            f"- Record: `{case_103['record_id']}`; gold/base/candidate = "
            f"`{case_103['gold_decision']}/{case_103['base_decision']}/"
            f"{case_103['candidate_decision']}`; final Δ = "
            f"`{case_103['delta']:+.4f}`.",
            f"- Question: {compact(case_103['question'])}",
            f"- Baseline: {compact(case_103['base_response'])}",
            f"- Candidate: {compact(case_103['candidate_response'])}",
            "- Mechanism: the added list of ordinary variation and structured "
            "signatures does not merely calibrate a threshold. It redirects "
            "attention from the checkpoint's learned local-hint interpretation "
            "toward a broad decline/recovery narrative, changing the anomaly "
            "shape and the option.",
            "",
            "### Failure 2 — status-before-shape creates a premature normality decision",
            "",
            f"- Record: `{case_083['record_id']}`; gold/base/candidate = "
            f"`{case_083['gold_decision']}/{case_083['base_decision']}/"
            f"{case_083['candidate_decision']}`; final Δ = "
            f"`{case_083['delta']:+.4f}`.",
            f"- Baseline: {compact(case_083['base_response'])}",
            f"- Candidate: {compact(case_083['candidate_response'])}",
            "- Mechanism: forcing an anomaly-status decision before option "
            "comparison separates two computations that the finetuned model "
            "learned jointly. Once the first stage declares ordinary behavior, "
            "the second stage rationalizes a normal/recovery distractor and "
            "ignores the correct boundary spike.",
            "",
            "### Failure 3 — RE2 fixes polarity but can rewrite the evidence",
            "",
            f"- Record: `{case_111['record_id']}`; gold/base/candidate = "
            f"`{case_111['gold_decision']}/{case_111['base_decision']}/"
            f"{case_111['candidate_decision']}`; final Δ = "
            f"`{case_111['delta']:+.4f}`.",
            f"- Question: {compact(case_111['question'])}",
            f"- Baseline: {compact(case_111['base_response'])}",
            f"- Candidate: {compact(case_111['candidate_response'])}",
            "- Mechanism: the TF rule often repairs proposition polarity, but "
            "it is not a deterministic label normalizer. On this case it also "
            "changes the time-series interpretation, calling a sharp excursion "
            "gradual and non-anomalous; the forced verdict then becomes "
            "internally consistent but wrong.",
            "",
            "### Success case — complete-option comparison recovers a localized shake",
            "",
            f"- Record: `{case_022['record_id']}`; gold/base/candidate = "
            f"`{case_022['gold_decision']}/{case_022['base_decision']}/"
            f"{case_022['candidate_decision']}`; final Δ = "
            f"`{case_022['delta']:+.4f}`.",
            f"- Baseline: {compact(case_022['base_response'])}",
            f"- Candidate: {compact(case_022['candidate_response'])}",
            "- The intervention succeeds when the error is true semantic option "
            "binding: it maps localized oscillation to the complete option "
            "rather than defaulting to a boundary-spike label. This benefit is "
            "real, but it is not uniformly separable from the failure modes.",
            "",
            "## Failure synthesis",
            "",
            "1. Long or multi-clause guidance competes with the checkpoint's "
            "finetuned prompt distribution. It changes evidence interpretation, "
            "not just response formatting.",
            "2. Normality priors and anomaly-threshold language are unsafe: "
            "they prevent false positives on some normal windows but suppress "
            "true localized spikes on others.",
            "3. Decomposing status and shape is not modular for this model. The "
            "first generated judgment anchors the second and promotes post-hoc "
            "rationalization.",
            "4. RE2-style TF rules repair many polarity errors but occasionally "
            "alter the underlying evidence narrative; aggregate gains can hide "
            "a catastrophic per-case reversal.",
            "5. OE was left byte-identical to Baseline. That is why its four "
            "metrics are exactly tied rather than improved; no tested OE rule "
            "survived earlier validation.",
            "",
            "## Frozen prompt rules",
            "",
            "All routes preserve the released opening, Values → Per-Step "
            "Analysis → `Overall Summary Hints` order, 30 Fixed tokens and "
            "question-ending boundary. Only the following `### Answering Rule` "
            "text is conditionally inserted.",
            "",
            "### MC structured-signature guard",
            "",
            "```text",
            "Compare every option by its complete meaning. Treat alternating "
            "signs, ordinary peaks or troughs, isolated large or small values, "
            "and irregular-looking fluctuation as normal variability unless "
            "the supplied evidence supports the option's specific structured "
            "anomaly signature, such as a localized contrast, persistence, "
            "recovery, or boundary pattern. Select an anomalous option only "
            "when that defining signature is supported; otherwise select the "
            "normal option. Explain the decisive qualitative pattern without "
            "quoting exact values or step numbers unless asked.",
            "```",
            "",
            "### MC status then shape",
            "",
            "```text",
            "First decide anomaly status from the supplied Per-Step Analysis "
            "and Overall Summary Hints, independently of the option wording. "
            "Ordinary variance, alternating signs, isolated highs or lows, and "
            "irregular-looking fluctuation are not anomalies by themselves. "
            "Then compare only options consistent with that status and choose "
            "the one whose complete text best matches the observed shape, "
            "persistence, recovery, and boundary behavior. Explain the decisive "
            "qualitative evidence without quoting exact values or step numbers "
            "unless asked.",
            "```",
            "",
            "### Explicit-negation TF RE2",
            "",
            "Activated only for the frozen lexical negation cues.",
            "",
            "```text",
            'Evaluate the complete proposition exactly as written. Use '
            "qualitative shape, direction, persistence, and recovery; do not "
            "quote exact values or step numbers unless the question explicitly "
            'asks for them. End with exactly "Your answer: True." or "Your '
            'answer: False.", and keep the explanation consistent with that '
            "verdict.",
            "```",
            "",
            "## Final decision",
            "",
            "The search completed 30 new modes across at least three distinct "
            "mechanistic cycles. Because no candidate passed all four frozen "
            "gates, the registered stopping criterion is met. Selecting the "
            "largest mean gain or relaxing the correct→wrong guard after seeing "
            "holdout would be post-hoc selection. `paper140` therefore remains "
            "untouched and there is no claim of a prompt-only Pareto improvement.",
            "",
        ]
    )
    return "\n".join(lines)


def render_main_append(summaries: dict[str, dict]) -> str:
    holdout = summaries["exposed holdout48"]
    r4 = by_mode(holdout)["lit_r4_01_tf_neg_re2"]
    structured = by_mode(holdout)["lit_r5_01_mc_structured_guard"]
    lines = [
        "# 文献驱动 Prompt Autoresearch 最终补充（2026-07-27）",
        "",
        "## 结论",
        "",
        "本轮完整阅读 `prompt系列改进.md`，结合 Time-LLM、选择题符号绑定与顺序偏差、"
        "上下文校准、输出格式偏差、RE2、CoT/小模型提示敏感性等原始论文，随后在不训练"
        "模型的前提下完成 30 个新模式的上限搜索。所有正式推理均在授权 GPU 上完成，"
        "所有正式评分仅使用 `deepseek-v4-pro`。",
        "",
        "最终没有 Prompt 同时通过 development96、screening24、validation72 和 "
        "exposed holdout48 的预注册门槛。因而触发停止规则，没有锁定胜者，也没有生成"
        "`paper140` 候选输出。该结果不能表述为“找到了正式测试集上的全面提升 Prompt”。",
        "",
        "## 最接近的内部结果",
        "",
        "| 候选 | 数据层 | MC Final Δ | MC Corr. Δ | MC Rsn. Δ | "
        "TF Final Δ | TF Corr. Δ | TF Justif. Δ | c→w | 判定 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        "| `lit_r4_01_tf_neg_re2` | exposed holdout48 | +0.0000 | "
        "+0.0000 | +0.0000 | "
        f"{r4['deltas']['true_false/final']:+.4f} | "
        f"{r4['deltas']['true_false/correctness']:+.4f} | "
        f"{r4['deltas']['true_false/justification_quality']:+.4f} | "
        f"{holdout['pair_summary']['lit_r4_01_tf_neg_re2']['correct_to_wrong']} | "
        "Table-I 全非降，但违反零伤害门槛 |",
        "| `lit_r5_01_mc_structured_guard` | exposed holdout48 | "
        f"{structured['deltas']['multiple_choice/final']:+.4f} | "
        f"{structured['deltas']['multiple_choice/correctness']:+.4f} | "
        f"{structured['deltas']['multiple_choice/reasoning_quality']:+.4f} | "
        "+0.0000 | +0.0000 | +0.0000 | "
        f"{holdout['pair_summary']['lit_r5_01_mc_structured_guard']['correct_to_wrong']} | "
        "holdout 通过，但 validation72 三项 MC 下降 |",
        "",
        "OE 始终复用精确 Baseline，因此四项 OE 指标均为 0 差值。完整的 5 候选 × "
        "4 数据层 × 10 指标、绝对分数、配对胜负和失败 case 见 "
        "`prompt_literature_autoresearch_deepseek_v4_pro_20260727/"
        "experiments/literature-round-5/analysis.md`。",
        "",
        "## 失败 case 归因",
        "",
        "- `series_000103:1`：structured guard 把 Baseline 正确的中心向上尖峰改判为"
        "缓慢下降后回归，说明阈值规则重新分配了注意力，而非只抑制假阳性。",
        "- `series_000083:1`、`series_000089:1`：status-before-shape 先形成“正常”"
        "锚点，随后用正常/振荡选项解释数据，覆盖了 checkpoint 原本正确的边界尖峰识别。",
        "- `series_000111:1`：TF RE2 虽总体修正更多极性错误，却把明确尖峰解释成渐进"
        "变化，使 `False→True` 并造成约 −1.60 的 Final 下降。",
        "- `series_000022:1`：完整选项语义比较确实能把 Baseline 的边界尖峰误选修正为"
        "局部快速振荡，证明收益机制真实存在；但它与上述伤害机制不能靠当前 prompt "
        "稳定分离。",
        "",
        "## 研究判断",
        "",
        "失败的共同原因不是 Judge 或解析器，而是 7B checkpoint 对训练时 prompt "
        "分布高度敏感。新增自然语言规则会同时改变“读证据、判异常、匹配选项、组织答案”"
        "四个过程。即使 aggregate Table-I 指标上涨，也可能隐藏少数灾难性反转。"
        "在当前证据下，发布 Baseline 仍是唯一满足严格零回归要求的 prompt。",
        "",
        "## 关键复现配置",
        "",
        "- GPU 源提交：`4d3ace60c036cdb231a4c5aeafeec3cdf030aace`。",
        "- checkpoint SHA-256："
        "`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。",
        "- development：192 条原始 GPU 输出、83 个路由分量、136 个新 Judge 维度；"
        "组装后 576 predictions / 1326 scores。",
        "- holdout：144 条原始 GPU 输出、44 个路由分量、88 个新 Judge 维度；"
        "组装后 288 predictions / 666 scores。",
        "- 审计：模型、provider、方法、prompt SHA-256、record/mode/dimension 唯一性"
        "和 manifest 均通过。",
        "- `paper140`：候选调用 0 次；未用于选择或调参。",
        "",
        "## 本轮 Prompt 全文索引",
        "",
        "下面的最终附录由实现代码直接渲染，覆盖 30 个 literature-guided 模式及全部"
        "实际路由分支；每个分支附 SHA-256。为便于后续观察，该附录必须保持为本文档"
        "最后一部分。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summaries = {label: load_summary(path) for label, path in SPLITS}
    (ROUND / "analysis.md").write_text(
        render_analysis(summaries), encoding="utf-8"
    )
    to_human = ROOT / "to_human"
    to_human.mkdir(parents=True, exist_ok=True)
    (to_human / "literature_final_append.md").write_text(
        render_main_append(summaries), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
