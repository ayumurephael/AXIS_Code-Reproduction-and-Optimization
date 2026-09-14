from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
LOCAL_EXPORTS = ROOT / "outputs" / "local_exports"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _metric_block(run_payload: Dict[str, Any]) -> Dict[str, Any]:
    return ((run_payload.get("overall") or {}).get("label") or {})


def _category_metric(run_payload: Dict[str, Any], key: str) -> Dict[str, Any]:
    return (((run_payload.get("by_category") or {}).get(key) or {}).get("label") or {})


def _row_from_run(summary_path: Path, run_name: str, run_payload: Dict[str, Any]) -> Dict[str, Any]:
    rel_parent = summary_path.parent.relative_to(LOCAL_EXPORTS)
    rel_parts = rel_parent.parts
    top_group = rel_parts[0] if rel_parts else ""
    overall = _metric_block(run_payload)
    script_report = run_payload.get("script_report") or {}
    workflow = script_report.get("student_workflow") or {}
    experiment = script_report.get("experiment") or {}
    llm_cfg = script_report.get("llm_config") or {}

    row: Dict[str, Any] = {
        "source_top_group": top_group,
        "source_relative_dir": str(rel_parent).replace("\\", "/"),
        "summary_file": str(summary_path),
        "run_name": run_name,
        "is_archive": "runs_archive" in rel_parts,
        "run_root": script_report.get("html_report") or str((summary_path.parent / run_name).resolve()),
        "question_path": script_report.get("data_path"),
        "provider": llm_cfg.get("provider"),
        "model": llm_cfg.get("model"),
        "label_n": _safe_int(overall.get("n")),
        "label_parsed": _safe_int(overall.get("parsed")),
        "label_correct": _safe_int(overall.get("correct")),
        "label_parse_rate": _safe_float(overall.get("parse_rate")),
        "label_acc": _safe_float(overall.get("accuracy")),
        "label_acc_on_parsed": _safe_float(overall.get("accuracy_on_parsed")),
        "explicit_n": _safe_int(overall.get("explicit_n")),
        "explicit_parsed": _safe_int(overall.get("explicit_parsed")),
        "explicit_answer_rate": _safe_float(overall.get("explicit_answer_rate")),
        "choice_acc": _safe_float(_category_metric(run_payload, "answer_type=choice").get("accuracy")),
        "judgment_acc": _safe_float(_category_metric(run_payload, "answer_type=judgment").get("accuracy")),
        "question_frame_anomaly_acc": _safe_float(_category_metric(run_payload, "difficulty=anomaly_frame").get("accuracy")),
        "question_frame_normal_acc": _safe_float(_category_metric(run_payload, "difficulty=normal_frame").get("accuracy")),
        "simple_acc": _safe_float(_category_metric(run_payload, "difficulty=simple").get("accuracy")),
        "medium_acc": _safe_float(_category_metric(run_payload, "difficulty=medium").get("accuracy")),
        "complex_acc": _safe_float(_category_metric(run_payload, "difficulty=complex").get("accuracy")),
        "hide_global_hints": workflow.get("hide_global_hints"),
        "hide_channel_hints": workflow.get("hide_channel_hints"),
        "include_anomaly_score_text": workflow.get("include_anomaly_score_text"),
        "include_anomaly_score_image_note": workflow.get("include_anomaly_score_image_note"),
        "include_raw_image_note": workflow.get("include_raw_image_note"),
        "image_consumed_by_vl_qwen": workflow.get("image_consumed_by_vl_qwen"),
        "prompt_style": workflow.get("prompt_style") or experiment.get("prompt_style"),
        "experiment_group": experiment.get("group"),
        "experiment_name": experiment.get("name"),
        "notes": "",
    }
    if row["choice_acc"] is None and row["judgment_acc"] is None and row["simple_acc"] is None:
        row["notes"] = "overall-only summary"
    return row


def collect_teacher_aligned_runs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(LOCAL_EXPORTS.rglob("teacher_aligned_summary.json")):
        payload = _load_json(summary_path)
        runs = payload.get("runs") or {}
        for run_name, run_payload in runs.items():
            rows.append(_row_from_run(summary_path, str(run_name), run_payload))
    return rows


def collect_curated_summaries() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(LOCAL_EXPORTS.rglob("summary.json")):
        try:
            payload = _load_json(summary_path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        if not payload or not isinstance(payload[0], dict):
            continue
        if "run" not in payload[0] or "label_acc" not in payload[0]:
            continue
        rel_parent = summary_path.parent.relative_to(LOCAL_EXPORTS)
        for item in payload:
            rows.append(
                {
                    "source_relative_dir": str(rel_parent).replace("\\", "/"),
                    "summary_file": str(summary_path),
                    "dataset": item.get("dataset"),
                    "run": item.get("run"),
                    "label_acc": _safe_float(item.get("label_acc")),
                    "label_acc_on_explicit": _safe_float(item.get("label_acc_on_explicit")),
                    "explicit_answer_rate": _safe_float(item.get("explicit_answer_rate")),
                    "explicit_n": _safe_int(item.get("explicit_n")),
                    "label_n": _safe_int(item.get("label_n")),
                }
            )
    return rows


def main() -> None:
    output_dir = LOCAL_EXPORTS / "research_report_20260603"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_teacher_aligned_runs()
    curated = collect_curated_summaries()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "local_exports_root": str(LOCAL_EXPORTS),
        "teacher_aligned_runs": runs,
        "curated_summary_rows": curated,
    }
    out_path = output_dir / "local_exports_eval_index.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "run_count": len(runs), "curated_count": len(curated)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
