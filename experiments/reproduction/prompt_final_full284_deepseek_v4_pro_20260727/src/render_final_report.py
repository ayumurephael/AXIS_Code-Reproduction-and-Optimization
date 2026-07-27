"""Render the frozen full-284 result report and paired improvement cases."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.axis_repro.build_tables import DIMS  # noqa: E402
from tools.axis_repro.common import read_jsonl  # noqa: E402
from tools.axis_repro.table_runner import aggregate  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / "artifacts"
ASSEMBLED = ARTIFACTS / "assembled"
PREDICTIONS = ASSEMBLED / "predictions.jsonl"
SCORES = ASSEMBLED / "scores.jsonl"
INVENTORY = PROJECT / "screening_inventory.md"
RESULTS_JSON = PROJECT / "final_results.json"
CASES_JSONL = PROJECT / "paired_improvement_cases.jsonl"
REPORT = PROJECT / "prompt_final.md"

BASE = "base"
OE_ONLY = "lit_r1_08_oe_re2"
TRIPLET = "lit_r1_10_triplet_re2"
MODES = (BASE, OE_ONLY, TRIPLET)
MODE_LABELS = {
    BASE: "AXIS Baseline",
    OE_ONLY: "OE-only RE2",
    TRIPLET: "MC/OE/TF RE2",
}
QUESTION_LABELS = {
    "multiple_choice": "MC",
    "open_ended": "OE",
    "true_false": "TF",
}
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
OE_KEYS = {
    "open_ended/final",
    "open_ended/accuracy",
    "open_ended/completeness",
    "open_ended/relevance",
}
TOLERANCE = 1e-6

BASE_TEMPLATE = """You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
{question}"""

RE2_TEMPLATE = """You are an expert time series analyst. Analyze the provided data and answer the question.

### Time Series Data
- **Window:** Steps {start} to {end}
- **Values (scaled by 100):** {serialized_values}

### Contextual Hints
- **Per-Step Analysis:** {local_hint_tokens}
- **Overall Summary Hints:** {fixed_hint_tokens}

### Question
{question}

Read the question again:
{question}"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def final_scores(rows: list[dict]) -> dict[tuple[str, str], float]:
    dimensions: dict[tuple[str, str], dict[str, tuple[float, float]]] = (
        defaultdict(dict)
    )
    question_types: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["record_id"], row["mode"])
        dimensions[key][row["dimension"]] = (
            float(row["score"]),
            float(row["weight"]),
        )
        question_types[key] = row["question_type"]
    output = {}
    for key, values in dimensions.items():
        qtype = question_types[key]
        expected = DIMS[qtype]
        if any(dimension not in values for dimension in expected):
            raise RuntimeError(f"Incomplete score dimensions for {key}")
        output[key] = sum(
            values[dimension][0] * values[dimension][1]
            for dimension in expected
        )
    return output


def success_decision(metrics: dict[str, dict[str, float]]) -> dict[str, dict]:
    base = metrics[BASE]
    decisions = {}
    for mode in (OE_ONLY, TRIPLET):
        deltas = {
            key: metrics[mode][key] - base[key] for _, key in METRICS
        }
        nonnegative = sum(delta >= -TOLERANCE for delta in deltas.values())
        improved = sum(delta > TOLERANCE for delta in deltas.values())
        oe_improved = sum(
            deltas[key] > TOLERANCE for key in OE_KEYS
        )
        decisions[mode] = {
            "deltas": deltas,
            "nonnegative": nonnegative,
            "improved": improved,
            "oe_prompt_changed": True,
            "oe_improved": oe_improved,
            "passed": (
                nonnegative == len(METRICS)
                and improved >= 3
                and oe_improved >= 1
            ),
        }
    return decisions


def extract_cases(
    predictions: list[dict],
    score_rows: list[dict],
    decisions: dict[str, dict],
) -> list[dict]:
    prediction_index = {
        (row["record_id"], row["mode"]): row for row in predictions
    }
    record_scores = final_scores(score_rows)
    dimension_scores = {
        (row["record_id"], row["mode"], row["dimension"]): float(row["score"])
        for row in score_rows
    }
    output = []
    for mode in (OE_ONLY, TRIPLET):
        if not decisions[mode]["passed"]:
            continue
        for question_type in QUESTION_LABELS:
            candidates = []
            for (record_id, candidate_mode), candidate in prediction_index.items():
                if candidate_mode != mode:
                    continue
                if candidate["question_type"] != question_type:
                    continue
                baseline = prediction_index[(record_id, BASE)]
                if candidate["response"] == baseline["response"]:
                    continue
                delta = (
                    record_scores[(record_id, mode)]
                    - record_scores[(record_id, BASE)]
                )
                if delta <= TOLERANCE:
                    continue
                candidates.append((delta, record_id, baseline, candidate))
            if not candidates:
                continue
            delta, record_id, baseline, candidate = max(candidates)
            dimensions = {}
            for dimension in DIMS[question_type]:
                base_value = dimension_scores[(record_id, BASE, dimension)]
                candidate_value = dimension_scores[(record_id, mode, dimension)]
                dimensions[dimension] = {
                    "baseline": base_value,
                    "candidate": candidate_value,
                    "delta": candidate_value - base_value,
                }
            output.append(
                {
                    "mode": mode,
                    "question_type": question_type,
                    "record_id": record_id,
                    "question": baseline["question"],
                    "gold_answer": baseline["answer"],
                    "baseline_response": baseline["response"],
                    "candidate_response": candidate["response"],
                    "baseline_final": record_scores[(record_id, BASE)],
                    "candidate_final": record_scores[(record_id, mode)],
                    "final_delta": delta,
                    "dimensions": dimensions,
                }
            )
    return output


def score_table(
    metrics: dict[str, dict[str, float]],
    decisions: dict[str, dict],
) -> list[str]:
    lines = [
        "| Prompt | " + " | ".join(label for label, _ in METRICS) + " |",
        "|" + "---|" * (len(METRICS) + 1),
    ]
    for mode in MODES:
        values = [MODE_LABELS[mode]]
        values.extend(f"{metrics[mode][key]:.4f}" for _, key in METRICS)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "| Prompt | 10项非降 | 严格提升项数 | OE严格提升项数 | 最终判定 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for mode in (OE_ONLY, TRIPLET):
        decision = decisions[mode]
        lines.append(
            f"| `{mode}` | {decision['nonnegative']}/10 | "
            f"{decision['improved']} | {decision['oe_improved']} | "
            f"{'PASS' if decision['passed'] else 'FAIL'} |"
        )
    return lines


def delta_table(
    metrics: dict[str, dict[str, float]],
    decisions: dict[str, dict],
    relative: bool,
) -> list[str]:
    title = "相对变化（%）" if relative else "绝对差值"
    lines = [
        f"| Prompt（{title}） | "
        + " | ".join(label for label, _ in METRICS)
        + " |",
        "|" + "---|" * (len(METRICS) + 1),
    ]
    base = metrics[BASE]
    for mode in (OE_ONLY, TRIPLET):
        values = [MODE_LABELS[mode]]
        for _, key in METRICS:
            delta = decisions[mode]["deltas"][key]
            value = 100.0 * delta / base[key] if relative else delta
            values.append(f"{value:+.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def route_table() -> list[str]:
    return [
        "| Prompt | MC | OE | TF |",
        "|---|---|---|---|",
        "| AXIS Baseline | 原论文 Prompt | 原论文 Prompt | 原论文 Prompt |",
        "| `lit_r1_08_oe_re2` | 原论文 Prompt（精确复用） | "
        "RE2 Prompt | 原论文 Prompt（精确复用） |",
        "| `lit_r1_10_triplet_re2` | RE2 Prompt | RE2 Prompt | RE2 Prompt |",
    ]


def case_sections(cases: list[dict], decisions: dict[str, dict]) -> list[str]:
    lines = ["## 具体回答质量改善案例", ""]
    successful = [
        mode for mode in (OE_ONLY, TRIPLET) if decisions[mode]["passed"]
    ]
    if not successful:
        return lines + [
            "没有候选通过最终十项指标门槛，因此不能把局部高分回答包装成"
            "“全面提升 Prompt”的成功案例。",
            "",
        ]
    by_key = {(row["mode"], row["question_type"]): row for row in cases}
    for mode in successful:
        lines.extend([f"### `{mode}`", ""])
        for question_type, label in QUESTION_LABELS.items():
            row = by_key.get((mode, question_type))
            if row is None:
                lines.extend(
                    [
                        f"#### {label}",
                        "",
                        "该题型的 Prompt/回答与 Baseline 字节级相同，或不存在"
                        "单题 Judge Final 严格上升的已改变回答；不能虚构改善案例。",
                        "",
                    ]
                )
                continue
            lines.extend(
                [
                    f"#### {label} — `{row['record_id']}`",
                    "",
                    f"- 单题 Final：Baseline `{row['baseline_final']:.4f}`；"
                    f"改进 `{row['candidate_final']:.4f}`；"
                    f"Δ `{row['final_delta']:+.4f}`。",
                    f"- 题干：{row['question']}",
                    f"- 标准答案：{row['gold_answer']}",
                    "",
                    "**原 AXIS Prompt 的回答：**",
                    "",
                    row["baseline_response"].strip(),
                    "",
                    "**改进 Prompt 的回答：**",
                    "",
                    row["candidate_response"].strip(),
                    "",
                    "Judge 分项："
                    + "；".join(
                        f"{name} {values['baseline']:.4f}→"
                        f"{values['candidate']:.4f} "
                        f"({values['delta']:+.4f})"
                        for name, values in row["dimensions"].items()
                    )
                    + "。",
                    "",
                ]
            )
    return lines


def main() -> None:
    predictions = read_jsonl(PREDICTIONS)
    score_rows = read_jsonl(SCORES)
    if Counter(row["mode"] for row in predictions) != Counter(
        {mode: 284 for mode in MODES}
    ):
        raise RuntimeError("Assembled prediction mode counts are not 284 each")
    metrics, paired = aggregate(score_rows)
    if set(metrics) != set(MODES):
        raise RuntimeError(f"Unexpected result modes: {sorted(metrics)}")
    decisions = success_decision(metrics)
    cases = extract_cases(predictions, score_rows, decisions)
    CASES_JSONL.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
        encoding="utf-8",
    )

    metadata = {
        "record_count_per_mode": 284,
        "prediction_rows": len(predictions),
        "score_rows": len(score_rows),
        "question_type_counts": dict(
            Counter(
                row["question_type"]
                for row in predictions
                if row["mode"] == BASE
            )
        ),
        "judge_models": dict(Counter(row["model"] for row in score_rows)),
        "judge_providers": dict(
            Counter(row["provider"] for row in score_rows)
        ),
        "score_methods": dict(Counter(row["method"] for row in score_rows)),
        "predictions_sha256": sha256(PREDICTIONS),
        "scores_sha256": sha256(SCORES),
    }
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "models": metrics,
                "paired_vs_axis": paired,
                "decisions": decisions,
                "metadata": metadata,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    inventory = INVENTORY.read_text(encoding="utf-8").strip()
    lines = [
        "# AXIS Prompt 最终 full-284 实验结果",
        "",
        "## 结论",
        "",
    ]
    successful = [
        mode for mode in (OE_ONLY, TRIPLET) if decisions[mode]["passed"]
    ]
    if successful:
        lines.append(
            "按“Table-I 十项全部不低于 Baseline、至少三项严格提高、"
            "OE Prompt 确实改变且至少一个 OE 指标提高”的冻结标准，"
            "通过的 Prompt 为："
            + "、".join(f"`{mode}`" for mode in successful)
            + "。"
        )
    else:
        lines.append(
            "两个宽口径候选均未通过最终 full-284 门槛；本轮没有可宣称为"
            "全面提升的 Prompt。已确认失败的 Prompt 没有重复运行。"
        )
    lines.extend(
        [
            "",
            "这里已彻底删除 `zero correct→wrong` 门槛；MC/TF 决策迁移"
            "只可用于诊断，不参与 PASS/FAIL。",
            "",
            "## 正式 Table-I 十项结果",
            "",
            *score_table(metrics, decisions),
            "",
            "## 相对 Baseline 的绝对差值",
            "",
            *delta_table(metrics, decisions, relative=False),
            "",
            "## 相对 Baseline 的变化比例",
            "",
            *delta_table(metrics, decisions, relative=True),
            "",
            "相对比例按 `(候选−Baseline)/Baseline×100%` 计算；最终判定"
            "使用未四舍五入浮点值和 `1e-6` 容差。",
            "",
            "## 评测口径与完整性",
            "",
            "- 数据：作者发布 `AXIS_qa_test` 的全部 142 个 series、284 QA；"
            "不是论文 Table I 的 `paper140`。",
            "- 推理：发布 checkpoint；远端授权 GPU；series batching；"
            "beam size 5；`max_new_tokens=1000`；`skip_loss=true`。",
            "- Judge：仅 `deepseek-v4-pro`；Baseline 与候选采用同一 G-Eval "
            "rubric、同一评分读出。",
            "- 发布 checkpoint SHA-256："
            "`d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`。",
            "- GPU 推理源提交："
            "`c113d3080375f05cec2b73702e8c1d2f5bcffa0f`；后续提交只增加"
            "精确复用、审计、报告与 Judge 断线恢复代码，未改变正式 Prompt。",
            f"- 组装后预测 `{metadata['prediction_rows']}` 行；评分 "
            f"`{metadata['score_rows']}` 行；每个 Prompt 284 个回答。",
            f"- Baseline 题型分布：`{metadata['question_type_counts']}`。",
            "- 原始 Judge 方法：`final_score_top_logprobs=1322`，"
            "`exact_sample_mean_20=12`；精确复用组装后方法分布："
            f"`{metadata['score_methods']}`。",
            f"- 预测 SHA-256：`{metadata['predictions_sha256']}`。",
            f"- 评分 SHA-256：`{metadata['scores_sha256']}`。",
            "",
            "## 最终实验 Prompt 路由",
            "",
            *route_table(),
            "",
            "`lit_r1_08_oe_re2` 的 OE 与 `lit_r1_10_triplet_re2` 的 OE "
            "Prompt 完全相同；前者 MC/TF 与 Baseline 完全相同。因此只生成"
            "两套不同的原始回答，再按题型精确复用，避免把随机重复生成误报"
            "为 Prompt 效果。",
            "",
            *case_sections(cases, decisions),
            "## 全部正式实验 Prompt",
            "",
            "下面是去除 Python f-string 固有缩进后的等价完整模板。"
            "`{local_hint_tokens}` 与 `{fixed_hint_tokens}` 的数量及注入位置"
            "保持作者实现不变。",
            "",
            "### 原 AXIS Baseline Prompt",
            "",
            "```text",
            BASE_TEMPLATE,
            "```",
            "",
            "### RE2 Prompt",
            "",
            "```text",
            RE2_TEMPLATE,
            "```",
            "",
            "RE2 唯一新增行为是：在原问题之后加入 `Read the question "
            "again:` 并原样重复问题；未加入 Evidence Contract、数值解释、"
            "新角色名或额外输出协议。",
            "",
            "## 完整 69-mode 筛选清单",
            "",
            "以下清单包含 Baseline、65 个 Prompt 候选和 3 个发布消融对照。"
            "已确认失败项依据已有审计结果排除，没有重复运行。",
            "",
            inventory,
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "successful_modes": successful,
                "case_count": len(cases),
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
