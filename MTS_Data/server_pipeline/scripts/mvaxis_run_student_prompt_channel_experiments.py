from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

import mvaxis_eval_teacher_aligned_student_runs as teacher_eval
import mvaxis_run_raw_qwen_answers as raw_base
from src.mvaxis.anomaly_score_context import (
    aggregate_anomaly_scores,
    attach_anomaly_score_context,
    score_stats,
)
from src.mvaxis.html_reports import write_answer_generation_html
from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.proposal import sample_with_interval
from src.mvaxis.student_answer_provider_experimental import (
    build_student_answer_messages_experimental,
    parse_student_answer,
    prompt_from_messages,
)
from src.mvaxis.utils import save_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run small prompt/channel-hint experiments for student answers."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--interval-proposer-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--prompt-examples", type=int, default=4)
    parser.add_argument("--raw-examples", type=int, default=6)
    parser.add_argument("--max-window-rows", type=int, default=0)
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--progress-percent-step", type=int, default=5)
    parser.add_argument(
        "--include-question-first-style",
        action="store_true",
        help="Also run question_first_hints_data prompt-style variants.",
    )
    return parser.parse_args()


def _build_experiments(include_question_first: bool) -> List[Dict[str, Any]]:
    experiments = [
        {
            "name": "all_hints_max_channel_tokens_8",
            "group": "all_hints",
            "prompt_style": "default_strict",
            "axis_overrides": {"max_channel_tokens": 8},
            "include_anomaly_score_text": False,
            "include_anomaly_score_image_note": False,
            "hide_global_hints": False,
            "hide_channel_hints": False,
        },
        {
            "name": "all_hints_channel_injection_scale_005",
            "group": "all_hints",
            "prompt_style": "default_strict",
            "axis_overrides": {"channel_injection_scale": 0.05},
            "include_anomaly_score_text": False,
            "include_anomaly_score_image_note": False,
            "hide_global_hints": False,
            "hide_channel_hints": False,
        },
        {
            "name": "score_image_note_all_hints_max_channel_tokens_8",
            "group": "score_image_note_all_hints",
            "prompt_style": "default_strict",
            "axis_overrides": {"max_channel_tokens": 8},
            "include_anomaly_score_text": False,
            "include_anomaly_score_image_note": True,
            "hide_global_hints": False,
            "hide_channel_hints": False,
        },
        {
            "name": "score_image_note_all_hints_channel_injection_scale_005",
            "group": "score_image_note_all_hints",
            "prompt_style": "default_strict",
            "axis_overrides": {"channel_injection_scale": 0.05},
            "include_anomaly_score_text": False,
            "include_anomaly_score_image_note": True,
            "hide_global_hints": False,
            "hide_channel_hints": False,
        },
    ]
    if include_question_first:
        experiments.extend(
            [
                {
                    "name": "all_hints_question_first_hints_data",
                    "group": "all_hints",
                    "prompt_style": "question_first_hints_data",
                    "axis_overrides": {},
                    "include_anomaly_score_text": False,
                    "include_anomaly_score_image_note": False,
                    "hide_global_hints": False,
                    "hide_channel_hints": False,
                },
                {
                    "name": "score_image_note_all_hints_question_first_hints_data",
                    "group": "score_image_note_all_hints",
                    "prompt_style": "question_first_hints_data",
                    "axis_overrides": {},
                    "include_anomaly_score_text": False,
                    "include_anomaly_score_image_note": True,
                    "hide_global_hints": False,
                    "hide_channel_hints": False,
                },
            ]
        )
    return experiments


def _llm_config_with_overrides(
    llm_config_path: Path,
    *,
    max_tokens: int,
    axis_overrides: Dict[str, Any],
) -> Dict[str, Any]:
    config = raw_base._load_llm_config(
        llm_config_path,
        model_path=None,
        max_tokens=max_tokens,
        use_axis_embedding_hints=True,
    )
    axis = dict(config.get("axis_hints") or {})
    axis.update(axis_overrides)
    config["axis_hints"] = axis
    return config


def _run_single_experiment(
    spec: Dict[str, Any],
    *,
    rows: List[Dict[str, Any]],
    inputs: List[Any],
    data_path: Path,
    output_root: Path,
    llm_config_path: Path,
    axis_config_path: Path,
    interval_checkpoint: str,
    limit: int,
    max_tokens: int,
    prompt_examples_limit: int,
    raw_examples_limit: int,
    max_window_rows: int,
    digits: int,
    progress_percent_step: int,
    score_image_manifest: Dict[str, Dict[str, Any]],
) -> None:
    llm_config = _llm_config_with_overrides(
        llm_config_path,
        max_tokens=max_tokens,
        axis_overrides=spec.get("axis_overrides") or {},
    )
    run_config = raw_base.load_json(axis_config_path)
    encoder = raw_base._load_encoder_for_hints(axis_config_path, interval_checkpoint)
    client = create_llm_client(llm_config)
    client_supports_images = bool(hasattr(client, "complete_with_images"))
    output_dir = output_root / spec["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_examples: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    parsed_outputs: List[Dict[str, Any]] = []
    max_window_rows_value = None if int(max_window_rows) <= 0 else int(max_window_rows)
    total_items = min(len(rows), len(inputs), limit)
    progress_step = max(1, min(100, int(progress_percent_step)))
    next_progress_pct = progress_step
    print(f"[exp:{spec['name']}] 0% (0/{total_items}) started", flush=True)

    for idx, (row, model_input) in enumerate(zip(rows[:limit], inputs[:limit])):
        prompt_row = row
        prompt_interval = list(model_input.interval)
        proposal = None
        anomaly_score_stats = None
        score_image_path = raw_base._score_image_for_row(score_image_manifest, idx, row)

        start, end = prompt_interval
        axis_cfg = llm_config.get("axis_hints") or {}
        hint_bundle = encoder.make_embedding_hints(
            prompt_row,
            start,
            end,
            top_k_channels=axis_cfg.get("top_k_channels"),
            max_channel_tokens=axis_cfg.get("max_channel_tokens", axis_cfg.get("max_local_tokens")),
            max_global_tokens=axis_cfg.get(
                "max_global_tokens",
                axis_cfg.get("num_global_tokens", axis_cfg.get("num_fixed_tokens")),
            ),
        )
        hint_bundle = raw_base._filter_embedding_hints(
            hint_bundle,
            include_global_hints=not bool(spec.get("hide_global_hints")),
            include_channel_hints=not bool(spec.get("hide_channel_hints")),
        )

        if spec.get("include_anomaly_score_image_note") or spec.get("include_anomaly_score_text"):
            channel_probs = encoder.anomaly_probabilities(prompt_row)
            time_scores = aggregate_anomaly_scores(
                channel_probs,
                prompt_row.get("channels") or [],
                aggregation="max",
            )
            anomaly_score_stats = score_stats(time_scores, int(prompt_interval[0]), int(prompt_interval[1]))
            prompt_row = attach_anomaly_score_context(
                prompt_row,
                time_scores,
                aggregation="max",
                image_path=score_image_path,
                copy_sample=True,
            )

        messages = build_student_answer_messages_experimental(
            prompt_row,
            prompt_style=str(spec.get("prompt_style") or "default_strict"),
            embedding_hint_trace=hint_bundle["trace"],
            use_axis_hints=True,
            include_global_hints=not bool(spec.get("hide_global_hints")),
            include_channel_hints=not bool(spec.get("hide_channel_hints")),
            include_evidence_card=False,
            include_anomaly_score_text=bool(spec.get("include_anomaly_score_text")),
            include_anomaly_score_image_note=bool(spec.get("include_anomaly_score_image_note")),
            include_raw_image_note=False,
            max_window_rows=max_window_rows_value,
            digits=int(digits),
        )
        prompt = prompt_from_messages(messages)
        if len(prompt_examples) < prompt_examples_limit:
            prompt_examples.append(
                {
                    "index": idx,
                    "question": model_input.question,
                    "prompt": prompt,
                    "experiment": spec,
                    "embedding_hint_trace": hint_bundle["trace"],
                    "global_embedding_shape": list(hint_bundle["embeddings"]["global"].shape)
                    if "global" in hint_bundle["embeddings"]
                    else None,
                    "channel_embedding_shape": list(hint_bundle["embeddings"]["channel"].shape)
                    if "channel" in hint_bundle["embeddings"]
                    else None,
                    "anomaly_score_stats": anomaly_score_stats,
                    "score_image_path": score_image_path,
                }
            )

        image_path_for_vl = score_image_path if spec.get("include_anomaly_score_image_note") else None
        if image_path_for_vl and hasattr(client, "complete_with_images_and_axis_hints"):
            response = client.complete_with_images_and_axis_hints(
                messages,
                [image_path_for_vl],
                hint_bundle["embeddings"],
            )
        else:
            response = client.complete_with_axis_hints(messages, hint_bundle["embeddings"])

        parsed = parse_student_answer(
            response.content,
            question_type=model_input.question_type,
            question=model_input.question,
        )
        parsed_outputs.append(parsed)
        records.append(
            {
                "index": idx,
                "sample_id": f"{data_path.name}:{idx}",
                "question_type": model_input.question_type,
                "question": model_input.question,
                "prompt": prompt,
                "raw_response": response.content,
                "parsed_response": parsed,
                "parsed_student_answer": parsed,
                "target_output": model_input.target_output,
                "latency_seconds": response.latency_seconds,
                "axis_embedding_hints_used": True,
                "embedding_hint_trace": hint_bundle["trace"],
                "global_embedding_shape": list(hint_bundle["embeddings"]["global"].shape)
                if "global" in hint_bundle["embeddings"]
                else None,
                "channel_embedding_shape": list(hint_bundle["embeddings"]["channel"].shape)
                if "channel" in hint_bundle["embeddings"]
                else None,
                "interval_source": "target",
                "prompt_interval": prompt_interval,
                "proposal": proposal,
                "anomaly_score_stats": anomaly_score_stats,
                "score_image_path": score_image_path,
                "raw_image_path": None,
                "image_not_consumed_by_text_qwen": bool(
                    spec.get("include_anomaly_score_image_note") and not client_supports_images
                ),
                "llm_raw_metadata": response.raw,
                "experiment": spec,
            }
        )
        completed = idx + 1
        pct = int(completed * 100 / total_items)
        if completed == total_items or pct >= next_progress_pct:
            print(
                f"[exp:{spec['name']}] {min(100, pct)}% ({completed}/{total_items}) last_latency={response.latency_seconds:.2f}s",
                flush=True,
            )
            while next_progress_pct <= pct:
                next_progress_pct += progress_step

    metrics = raw_base._student_answer_metrics(rows[: len(parsed_outputs)], parsed_outputs)
    metrics["experiment"] = spec
    metrics["data_path"] = str(data_path)
    metrics["llm_config"] = {
        "provider": llm_config.get("provider"),
        "model": llm_config.get("model"),
        "axis_hints_enabled": bool((llm_config.get("axis_hints") or {}).get("enabled")),
        "max_tokens": llm_config.get("max_tokens"),
    }
    metrics["student_workflow"] = {
        "prompt_style": spec.get("prompt_style"),
        "interval_source": "target",
        "hide_global_hints": bool(spec.get("hide_global_hints")),
        "hide_channel_hints": bool(spec.get("hide_channel_hints")),
        "include_anomaly_score_text": bool(spec.get("include_anomaly_score_text")),
        "include_anomaly_score_image_note": bool(spec.get("include_anomaly_score_image_note")),
        "axis_overrides": spec.get("axis_overrides") or {},
    }

    answers_jsonl_path = output_dir / "qwen_raw_answers.jsonl"
    write_jsonl(records, answers_jsonl_path)
    html_path = write_answer_generation_html(answers_jsonl_path, output_dir / "qwen_raw_answers.html")
    metrics["html_report"] = str(html_path)
    save_json(metrics, output_dir / "qwen_raw_report.json")
    save_json({"examples": prompt_examples}, output_dir / "qwen_prompt_examples.json")
    save_json({"examples": records[: raw_examples_limit]}, output_dir / "qwen_raw_examples.json")
    (output_dir / "qwen_case_study.txt").write_text(
        "\n\n" + ("=" * 80) + "\n\n".join(raw_base._case_text(r) for r in records[: raw_examples_limit]),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    teacher_path = Path(args.teacher)
    if not teacher_path.is_absolute():
        teacher_path = ROOT / teacher_path
    llm_config_path = Path(args.llm_config)
    if not llm_config_path.is_absolute():
        llm_config_path = ROOT / llm_config_path
    axis_config_path = Path(args.config)
    if not axis_config_path.is_absolute():
        axis_config_path = ROOT / axis_config_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows, inputs = raw_base._load_rows_and_inputs(data_path, args.limit)
    experiments = _build_experiments(bool(args.include_question_first_style))
    score_manifest_path = output_root / "score_images_manifest.json"
    score_manifest = raw_base._load_score_image_manifest(str(score_manifest_path)) if score_manifest_path.exists() else {}

    if not score_manifest:
        raise FileNotFoundError(
            f"Expected pre-generated score image manifest at {score_manifest_path}. "
            "Generate score images first and copy/symlink manifest here."
        )

    for spec in experiments:
        _run_single_experiment(
            spec,
            rows=rows,
            inputs=inputs,
            data_path=data_path,
            output_root=output_root,
            llm_config_path=llm_config_path,
            axis_config_path=axis_config_path,
            interval_checkpoint=args.interval_proposer_checkpoint,
            limit=args.limit,
            max_tokens=args.max_tokens,
            prompt_examples_limit=args.prompt_examples,
            raw_examples_limit=args.raw_examples,
            max_window_rows=args.max_window_rows,
            digits=args.digits,
            progress_percent_step=args.progress_percent_step,
            score_image_manifest=score_manifest,
        )

    questions = teacher_eval._read_jsonl(data_path)[: int(args.limit)]
    teacher_rows = teacher_eval._read_jsonl(teacher_path)
    teacher_by_sample = {
        str((row.get("sample_id") or row.get("sample_id_hint") or row.get("question_pair_id") or "")).strip(): teacher_eval._teacher_answer(row)
        for row in teacher_rows
        if str((row.get("sample_id") or row.get("sample_id_hint") or row.get("question_pair_id") or "")).strip()
    }
    summary = {"runs": {}}
    run_names = [spec["name"] for spec in experiments]
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
    save_json(summary, output_root / "teacher_aligned_summary.json")
    print(teacher_eval._format_table(summary, run_names))


if __name__ == "__main__":
    main()
