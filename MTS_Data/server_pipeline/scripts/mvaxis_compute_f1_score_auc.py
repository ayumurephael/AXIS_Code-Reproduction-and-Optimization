from __future__ import annotations

"""
Supplementary anomaly-sensitive eval metrics.

This script is the companion to `mvaxis_compute_metrics_with_open_decision.py`.
For regular eval reporting, keep the focus on:

- judgment_true_f1
- open_decision_anomalous_f1
- interval_max_auroc (score-bearing runs only)

The script can compute more fields, but those are the default extra metrics we
intend to maintain across datasets unless an experiment explicitly asks for
more detail.

For hint-alignment smoke checks, the standard comparison is:

- no hint
- true hint at fixed `ct` with multiple scales
- shuffled hint at the same `ct` and scales

The goal is to verify that real hints outperform no hint, while shuffled hints
do not produce the same improvement.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from _bootstrap import add_project_root

ROOT = add_project_root()

from scripts.mvaxis_eval_teacher_aligned_student_runs import parse_bool, parse_choice, _teacher_answer


GROUPS = [
    "all_hints",
    "no_global_hints",
    "no_channel_hints",
    "score_image_note_all_hints",
    "score_text_all_hints",
    "raw_image",
    "score_text_image_no_global_hints",
    "score_text_image_no_channel_hints",
    "score_text_nohints",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_answer_type(row: Dict[str, Any]) -> str:
    fact = (row.get("target_output") or {}).get("fact_check") or {}
    return str(
        row.get("question_answer_type")
        or fact.get("question_answer_type")
        or row.get("question_type")
        or ""
    ).lower()


def _parse_open_decision(text: str) -> Optional[bool]:
    text = text or ""
    decision_block = re.search(
        r"(?is)\bdecision\s*:\s*(.+?)(?:\n\s*\n|\n\s*main evidence\s*:|\n\s*interpretation\s*:|$)",
        text,
    )
    search_space = decision_block.group(1) if decision_block else text
    if re.search(r"(?i)\bthis interval is anomalous\b", search_space):
        return True
    if re.search(r"(?i)\bthis interval is normal\b", search_space):
        return False
    if re.search(r"(?i)\banomal(?:y|ous)\b", search_space) and not re.search(r"(?i)\bnormal\b", search_space):
        return True
    if re.search(r"(?i)\bnormal\b", search_space) and not re.search(r"(?i)\banomal(?:y|ous)\b", search_space):
        return False
    return None


def _teacher_open_decision(question_row: Dict[str, Any], teacher_row: Dict[str, Any]) -> Optional[bool]:
    teacher_answer = _teacher_answer(teacher_row)
    parsed = _parse_open_decision(teacher_answer)
    if parsed is not None:
        return parsed
    fact = (question_row.get("target_output") or {}).get("fact_check") or {}
    is_anomalous = fact.get("is_anomalous")
    return bool(is_anomalous) if is_anomalous is not None else None


def _target_choice_label(question_row: Dict[str, Any]) -> Optional[str]:
    fact = (question_row.get("target_output") or {}).get("fact_check") or {}
    val = fact.get("choice_answer")
    if val:
        return str(val).strip().upper()
    for candidate in [
        fact.get("answer_label"),
        (question_row.get("target_output") or {}).get("question_answer"),
        (question_row.get("target_output") or {}).get("final_answer"),
    ]:
        if candidate is None:
            continue
        parsed = parse_choice(str(candidate))
        if parsed:
            return parsed
    return None


def _safe_div(num: float, den: float) -> Optional[float]:
    return (num / den) if den else None


def _binary_f1(tp: int, fp: int, fn: int) -> Optional[float]:
    den = 2 * tp + fp + fn
    return (2 * tp / den) if den else None


def _multiclass_f1(pairs: Sequence[Tuple[Any, Any]], labels: Sequence[Any]) -> Tuple[Optional[float], Optional[float], Dict[Any, Optional[float]]]:
    if not pairs:
        return None, None, {}
    tp_total = fp_total = fn_total = 0
    per_label: Dict[Any, Optional[float]] = {}
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        per_label[label] = _binary_f1(tp, fp, fn)
    micro = _binary_f1(tp_total, fp_total, fn_total)
    macro_vals = [v for v in per_label.values() if v is not None]
    macro = (sum(macro_vals) / len(macro_vals)) if macro_vals else None
    return micro, macro, per_label


def _binary_curve_points(labels: Sequence[int], scores: Sequence[float]) -> List[Tuple[float, float]]:
    order = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return []
    tp = fp = 0
    points: List[Tuple[float, float]] = [(0.0, 0.0)]
    i = 0
    while i < len(order):
        score = order[i][0]
        while i < len(order) and order[i][0] == score:
            _, y = order[i]
            if y:
                tp += 1
            else:
                fp += 1
            i += 1
        points.append((fp / neg, tp / pos))
    return points


def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    points = _binary_curve_points(labels, scores)
    if not points:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def _pr_curve_points(labels: Sequence[int], scores: Sequence[float]) -> List[Tuple[float, float]]:
    order = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    pos = sum(labels)
    if pos == 0:
        return []
    tp = fp = 0
    points: List[Tuple[float, float]] = [(0.0, 1.0)]
    i = 0
    while i < len(order):
        score = order[i][0]
        while i < len(order) and order[i][0] == score:
            _, y = order[i]
            if y:
                tp += 1
            else:
                fp += 1
            i += 1
        recall = tp / pos
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        points.append((recall, precision))
    return points


def _pr_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    points = _pr_curve_points(labels, scores)
    if not points:
        return None
    area = 0.0
    for (r0, p0), (r1, p1) in zip(points, points[1:]):
        area += (r1 - r0) * (p0 + p1) / 2.0
    return area


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def compute_for_run(
    run_dir: Path,
    questions: List[Dict[str, Any]],
    teachers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = read_jsonl(run_dir / "qwen_raw_answers.jsonl")
    choice_pairs: List[Tuple[str, Optional[str]]] = []
    judgment_pairs: List[Tuple[bool, Optional[bool]]] = []
    open_pairs: List[Tuple[bool, Optional[bool]]] = []
    anomaly_labels: List[int] = []
    interval_max_scores: List[float] = []
    interval_mean_scores: List[float] = []

    for row in rows:
        idx = int(row["index"])
        question_row = questions[idx]
        teacher_row = teachers[idx]
        raw = str(row.get("raw_response") or "")
        answer_type = _question_answer_type(question_row)

        if "choice" in answer_type:
            gold = parse_choice(_teacher_answer(teacher_row)) or _target_choice_label(question_row)
            if gold is not None:
                choice_pairs.append((gold, parse_choice(raw)))
        elif "judgment" in answer_type or "true_false" in answer_type or "true-false" in answer_type:
            gold = parse_bool(_teacher_answer(teacher_row))
            if gold is None:
                fact = (question_row.get("target_output") or {}).get("fact_check") or {}
                if fact.get("is_anomalous") is not None:
                    gold = bool(fact["is_anomalous"])
            if gold is not None:
                judgment_pairs.append((gold, parse_bool(raw)))
        else:
            gold = _teacher_open_decision(question_row, teacher_row)
            if gold is not None:
                open_pairs.append((gold, _parse_open_decision(raw)))

        stats = row.get("anomaly_score_stats") or {}
        fact = (question_row.get("target_output") or {}).get("fact_check") or {}
        is_anomalous = fact.get("is_anomalous")
        if is_anomalous is not None and "interval_max" in stats and "interval_mean" in stats:
            anomaly_labels.append(1 if bool(is_anomalous) else 0)
            interval_max_scores.append(float(stats["interval_max"]))
            interval_mean_scores.append(float(stats["interval_mean"]))

    choice_micro, choice_macro, _ = _multiclass_f1(choice_pairs, ["A", "B", "C", "D", "E"])
    judgment_micro, judgment_macro, judgment_per = _multiclass_f1(judgment_pairs, [False, True])
    open_micro, open_macro, open_per = _multiclass_f1(open_pairs, [False, True])

    result: Dict[str, Any] = {
        "status": "ok",
        "choice_micro_f1": choice_micro,
        "choice_macro_f1": choice_macro,
        "choice_n": len(choice_pairs),
        "choice_parse_rate": _safe_div(sum(1 for _, pred in choice_pairs if pred is not None), len(choice_pairs)),
        "judgment_micro_f1": judgment_micro,
        "judgment_macro_f1": judgment_macro,
        "judgment_true_f1": judgment_per.get(True),
        "judgment_n": len(judgment_pairs),
        "judgment_parse_rate": _safe_div(sum(1 for _, pred in judgment_pairs if pred is not None), len(judgment_pairs)),
        "open_decision_micro_f1": open_micro,
        "open_decision_macro_f1": open_macro,
        "open_decision_anomalous_f1": open_per.get(True),
        "open_n": len(open_pairs),
        "open_parse_rate": _safe_div(sum(1 for _, pred in open_pairs if pred is not None), len(open_pairs)),
    }

    if anomaly_labels and interval_max_scores and interval_mean_scores:
        result.update(
            {
                "score_positive_rate": _safe_div(sum(anomaly_labels), len(anomaly_labels)),
                "score_n": len(anomaly_labels),
                "interval_max_auroc": _roc_auc(anomaly_labels, interval_max_scores),
                "interval_max_auprc": _pr_auc(anomaly_labels, interval_max_scores),
                "interval_mean_auroc": _roc_auc(anomaly_labels, interval_mean_scores),
                "interval_mean_auprc": _pr_auc(anomaly_labels, interval_mean_scores),
                "score_auc_note": "computed from anomaly_score_stats interval_max/interval_mean",
            }
        )
    else:
        result.update(
            {
                "score_positive_rate": None,
                "score_n": 0,
                "interval_max_auroc": None,
                "interval_max_auprc": None,
                "interval_mean_auroc": None,
                "interval_mean_auprc": None,
                "score_auc_note": "anomaly_score_stats not stored in this run",
            }
        )
    return result


def write_outputs(root: Path, dataset_label: str, results: Dict[str, Dict[str, Any]]) -> None:
    headers = [
        "run",
        "status",
        "choice_micro_f1",
        "choice_macro_f1",
        "choice_n",
        "judgment_micro_f1",
        "judgment_macro_f1",
        "judgment_true_f1",
        "judgment_n",
        "open_decision_micro_f1",
        "open_decision_macro_f1",
        "open_decision_anomalous_f1",
        "open_n",
        "interval_max_auroc",
        "interval_max_auprc",
        "interval_mean_auroc",
        "interval_mean_auprc",
        "score_n",
        "score_positive_rate",
        "score_auc_note",
    ]
    lines = ["\t".join(headers)]
    for run in GROUPS:
        item = results.get(run)
        if not item:
            continue
        lines.append("\t".join(_fmt(item.get(h)) for h in headers))
    (root / "qa_label_f1_score_auc_summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {"dataset_label": dataset_label, "results": results}
    (root / "qa_label_f1_score_auc_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def table(cols: Sequence[str], title: str) -> str:
        head = "".join(f"<th>{c}</th>" for c in cols)
        body_rows = []
        for run in GROUPS:
            item = results.get(run)
            if not item:
                continue
            tds = "".join(f"<td>{_fmt(item.get(c))}</td>" for c in cols)
            body_rows.append(f"<tr>{tds}</tr>")
        body = "\n".join(body_rows)
        return f"<h2>{title}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{dataset_label} F1 / AUC Summary</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    table {{ border-collapse: collapse; margin: 18px 0 28px; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #f6f8fa; }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ margin: 8px 0 16px; }}
    code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{dataset_label} F1 / AUC Summary</h1>
  <p>This page reports QA-label F1 for choice/judgment/open-decision and interval-score AUROC/AUPRC for score-bearing runs.</p>
  {table(
      [
          "run",
          "choice_micro_f1",
          "choice_macro_f1",
          "choice_n",
          "judgment_micro_f1",
          "judgment_macro_f1",
          "judgment_true_f1",
          "judgment_n",
          "open_decision_micro_f1",
          "open_decision_macro_f1",
          "open_decision_anomalous_f1",
          "open_n",
      ],
      "QA Label F1",
  )}
  {table(
      [
          "run",
          "interval_max_auroc",
          "interval_max_auprc",
          "interval_mean_auroc",
          "interval_mean_auprc",
          "score_n",
          "score_positive_rate",
          "score_auc_note",
      ],
      "Anomaly Score AUC",
  )}
</body>
</html>
"""
    (root / "f1_score_auc_summary.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--groups", nargs="*", default=GROUPS)
    args = parser.parse_args()

    root = Path(args.root)
    questions = read_jsonl(Path(args.questions))
    teachers = read_jsonl(Path(args.teacher))
    results: Dict[str, Dict[str, Any]] = {}
    for run in args.groups:
        jsonl_path = root / run / "qwen_raw_answers.jsonl"
        if not jsonl_path.exists():
            results[run] = {"run": run, "status": "missing_jsonl"}
            continue
        item = compute_for_run(root / run, questions, teachers)
        item["run"] = run
        results[run] = item
    write_outputs(root, args.dataset_label, results)
    print(json.dumps({"dataset_label": args.dataset_label, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
