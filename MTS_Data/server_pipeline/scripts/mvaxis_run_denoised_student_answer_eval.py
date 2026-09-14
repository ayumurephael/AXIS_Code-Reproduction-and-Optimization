from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

import mvaxis_eval_teacher_aligned_student_runs as teacher_eval
from src.mvaxis.utils import load_json, save_json


GROUP_SPECS: Dict[str, Dict[str, Any]] = {
    "all_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": False,
        "hide_channel_hints": False,
        "include_anomaly_score_text": False,
        "include_anomaly_score_image_note": False,
        "include_raw_image_note": False,
    },
    "no_global_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": True,
        "hide_channel_hints": False,
        "include_anomaly_score_text": False,
        "include_anomaly_score_image_note": False,
        "include_raw_image_note": False,
    },
    "no_channel_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": False,
        "hide_channel_hints": True,
        "include_anomaly_score_text": False,
        "include_anomaly_score_image_note": False,
        "include_raw_image_note": False,
    },
    "score_image_note_all_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": False,
        "hide_channel_hints": False,
        "include_anomaly_score_text": False,
        "include_anomaly_score_image_note": True,
        "include_raw_image_note": False,
    },
    "score_text_all_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": False,
        "hide_channel_hints": False,
        "include_anomaly_score_text": True,
        "include_anomaly_score_image_note": False,
        "include_raw_image_note": False,
    },
    "raw_image": {
        "use_axis_embedding_hints": False,
        "hide_global_hints": False,
        "hide_channel_hints": False,
        "include_anomaly_score_text": False,
        "include_anomaly_score_image_note": False,
        "include_raw_image_note": True,
    },
    "score_text_image_no_global_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": True,
        "hide_channel_hints": False,
        "include_anomaly_score_text": True,
        "include_anomaly_score_image_note": True,
        "include_raw_image_note": False,
    },
    "score_text_image_no_channel_hints": {
        "use_axis_embedding_hints": True,
        "hide_global_hints": False,
        "hide_channel_hints": True,
        "include_anomaly_score_text": True,
        "include_anomaly_score_image_note": True,
        "include_raw_image_note": False,
    },
    "score_text_nohints": {
        "use_axis_embedding_hints": False,
        "hide_global_hints": False,
        "hide_channel_hints": False,
        "include_anomaly_score_text": True,
        "include_anomaly_score_image_note": False,
        "include_raw_image_note": False,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full 9-group student-answer ablation with denoised channel-hint settings."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--interval-proposer-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--score-manifest", required=True)
    parser.add_argument("--raw-manifest", required=True)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(GROUP_SPECS.keys()),
        choices=sorted(GROUP_SPECS.keys()),
    )
    parser.add_argument("--limit", type=int, default=0, help="0 means full dataset.")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-examples", type=int, default=4)
    parser.add_argument("--raw-examples", type=int, default=8)
    parser.add_argument("--max-window-rows", type=int, default=0)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--progress-percent-step", type=int, default=5)
    parser.add_argument("--max-channel-tokens", type=int, default=8)
    parser.add_argument("--channel-injection-scale", type=float, default=0.05)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _write_llm_configs(
    output_root: Path,
    *,
    llm_config_path: Path,
    max_tokens: int,
    max_channel_tokens: int,
    channel_injection_scale: float,
) -> Dict[str, Path]:
    base = load_json(llm_config_path)
    if max_tokens:
        base["max_tokens"] = int(max_tokens)

    axis_cfg = dict(base.get("axis_hints") or {})
    axis_cfg["enabled"] = True
    axis_cfg["max_channel_tokens"] = int(max_channel_tokens)
    axis_cfg["channel_injection_scale"] = float(channel_injection_scale)
    axis_on = dict(base)
    axis_on["axis_hints"] = axis_cfg

    axis_off = dict(base)
    axis_off_cfg = dict(base.get("axis_hints") or {})
    axis_off_cfg["enabled"] = False
    axis_off["axis_hints"] = axis_off_cfg

    config_dir = output_root / "_llm_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    axis_on_path = config_dir / "llm_axis_denoised.json"
    axis_off_path = config_dir / "llm_no_axis.json"
    axis_on_path.write_text(json.dumps(axis_on, ensure_ascii=False, indent=2), encoding="utf-8")
    axis_off_path.write_text(json.dumps(axis_off, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"axis_on": axis_on_path, "axis_off": axis_off_path}


def _run_group(
    group: str,
    spec: Dict[str, Any],
    *,
    limit_value: int,
    data_path: Path,
    llm_config_axis_on: Path,
    llm_config_axis_off: Path,
    axis_config_path: Path,
    interval_checkpoint: Path,
    output_root: Path,
    score_manifest: Path,
    raw_manifest: Path,
    args: argparse.Namespace,
) -> None:
    output_dir = output_root / group
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-B",
        str(ROOT / "scripts" / "mvaxis_run_raw_qwen_answers.py"),
        "--data",
        str(data_path),
        "--llm-config",
        str(llm_config_axis_on if spec["use_axis_embedding_hints"] else llm_config_axis_off),
        "--output-dir",
        str(output_dir),
        "--limit",
        str(int(limit_value)),
        "--max-tokens",
        str(int(args.max_tokens)),
        "--prompt-examples",
        str(int(args.prompt_examples)),
        "--raw-examples",
        str(int(args.raw_examples)),
        "--max-window-rows",
        str(int(args.max_window_rows)),
        "--digits",
        str(int(args.digits)),
        "--progress-percent-step",
        str(int(args.progress_percent_step)),
    ]
    if spec["use_axis_embedding_hints"]:
        cmd.extend(
            [
                "--use-axis-embedding-hints",
                "--config",
                str(axis_config_path),
                "--interval-proposer-checkpoint",
                str(interval_checkpoint),
            ]
        )
    elif spec["include_anomaly_score_text"] or spec["include_anomaly_score_image_note"]:
        cmd.extend(
            [
                "--config",
                str(axis_config_path),
                "--interval-proposer-checkpoint",
                str(interval_checkpoint),
            ]
        )
    if spec["hide_global_hints"]:
        cmd.append("--hide-global-hints")
    if spec["hide_channel_hints"]:
        cmd.append("--hide-channel-hints")
    needs_score_manifest = False
    if spec["include_anomaly_score_text"]:
        cmd.append("--include-anomaly-score-text")
        needs_score_manifest = True
    if spec["include_anomaly_score_image_note"]:
        cmd.append("--include-anomaly-score-image-note")
        needs_score_manifest = True
    if needs_score_manifest:
        cmd.extend(["--score-image-manifest", str(score_manifest)])
    if spec["include_raw_image_note"]:
        cmd.extend(["--include-raw-image-note", "--raw-image-manifest", str(raw_manifest)])

    print(f"[denoised-run] starting {group}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[denoised-run] finished {group}", flush=True)


def main() -> None:
    args = _parse_args()
    data_path = _resolve(args.data)
    teacher_path = _resolve(args.teacher)
    llm_config_path = _resolve(args.llm_config)
    axis_config_path = _resolve(args.config)
    output_root = _resolve(args.output_root)
    score_manifest = _resolve(args.score_manifest)
    raw_manifest = _resolve(args.raw_manifest)
    interval_checkpoint = _resolve(args.interval_proposer_checkpoint)
    output_root.mkdir(parents=True, exist_ok=True)
    questions = teacher_eval._read_jsonl(data_path)
    limit_value = len(questions) if int(args.limit) <= 0 else min(int(args.limit), len(questions))

    cfgs = _write_llm_configs(
        output_root,
        llm_config_path=llm_config_path,
        max_tokens=int(args.max_tokens),
        max_channel_tokens=int(args.max_channel_tokens),
        channel_injection_scale=float(args.channel_injection_scale),
    )

    for group in args.groups:
        _run_group(
            group,
            GROUP_SPECS[group],
            limit_value=limit_value,
            data_path=data_path,
            llm_config_axis_on=cfgs["axis_on"],
            llm_config_axis_off=cfgs["axis_off"],
            axis_config_path=axis_config_path,
            interval_checkpoint=interval_checkpoint,
            output_root=output_root,
            score_manifest=score_manifest,
            raw_manifest=raw_manifest,
            args=args,
        )

    questions = questions[:limit_value]
    teacher_rows = teacher_eval._read_jsonl(teacher_path)
    teacher_by_sample = {
        str((row.get("sample_id") or row.get("sample_id_hint") or row.get("question_pair_id") or "")).strip(): teacher_eval._teacher_answer(row)
        for row in teacher_rows
        if str((row.get("sample_id") or row.get("sample_id_hint") or row.get("question_pair_id") or "")).strip()
    }

    summary = {"runs": {}}
    run_names = list(args.groups)
    for run_name in run_names:
        summary["runs"][run_name] = teacher_eval.evaluate_run(
            output_root / run_name,
            questions=questions,
            teacher_by_sample=teacher_by_sample,
        )
    (output_root / "teacher_aligned_summary.txt").write_text(
        teacher_eval._format_table(summary, run_names),
        encoding="utf-8",
    )
    (output_root / "metrics_table.tsv").write_text(
        teacher_eval._format_metrics_tsv(summary, run_names),
        encoding="utf-8",
    )
    save_json(
        {
            "data": str(data_path),
            "teacher": str(teacher_path),
            "score_manifest": str(score_manifest),
            "raw_manifest": str(raw_manifest),
            "denoise": {
                "max_channel_tokens": int(args.max_channel_tokens),
                "channel_injection_scale": float(args.channel_injection_scale),
            },
            "runs": summary["runs"],
        },
        output_root / "teacher_aligned_summary.json",
    )
    print(teacher_eval._format_table(summary, run_names), flush=True)


if __name__ == "__main__":
    main()
