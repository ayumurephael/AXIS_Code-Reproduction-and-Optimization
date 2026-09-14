from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.data_schema import attach_channel_scales
from src.mvaxis.html_reports import write_question_generation_html
from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.question_provider import (
    assign_question,
    duplicate_samples_with_question_frames,
    generate_question_spec_with_visual_workflow,
    sample_question_axes_for_bank,
)
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root


def _dated_output_dir(prefix: str, count: int) -> str:
    return f"outputs/{prefix}_{int(count)}_{datetime.now().strftime('%m%d')}"


def _normalize_source_range(total: int, start: int, end: int | None) -> tuple[int, int]:
    range_start = max(0, int(start))
    range_end = total if end is None else int(end)
    range_end = min(total, max(range_start, range_end))
    if range_start >= total:
        raise ValueError(f"source-start-index {range_start} is outside the available range 0..{max(total - 1, 0)}")
    return range_start, range_end


def _filter_windows_by_source_range(
    rows: List[Dict[str, Any]],
    *,
    source_start_index: int,
    source_end_index: int | None,
) -> List[Dict[str, Any]]:
    if source_start_index <= 0 and source_end_index is None:
        return rows
    max_source_index = max(int(row.get("source_index", -1)) for row in rows) + 1 if rows else 0
    range_start, range_end = _normalize_source_range(max_source_index, source_start_index, source_end_index)
    filtered = [
        copy.deepcopy(row)
        for row in rows
        if range_start <= int(row.get("source_index", -1)) < range_end
    ]
    if not filtered:
        raise ValueError(
            f"No windows remain after filtering source_index in [{range_start}, {range_end})"
        )
    return filtered


class TrackingClient:
    def __init__(self, client: Any, *, max_retries: int = 3, retry_sleep: float = 8.0) -> None:
        self.client = client
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep = max(0.0, float(retry_sleep))
        self.last_response = None
        self.last_messages: List[Dict[str, Any]] = []

    def complete(self, messages: List[Dict[str, Any]]):
        self.last_messages = messages
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self.last_response = self.client.complete(messages)
                if not str(self.last_response.content or "").strip():
                    raise RuntimeError("LLM returned empty question content")
                return self.last_response
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                print(f"LLM request failed on attempt {attempt}/{self.max_retries}: {exc}", flush=True)
                time.sleep(self.retry_sleep * attempt)
        raise last_error


def _safe_name(value: str, limit: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned[:limit] or "sample"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _series_record(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "idx": idx,
        "sample_id": sample.get("sample_id"),
        "base_sample_id": sample.get("base_sample_id"),
        "source_sample_id": sample.get("source_sample_id"),
        "target_interval": sample.get("target_interval"),
        "channels": sample.get("channels"),
        "series": sample.get("series"),
        "normal_series": sample.get("normal_series"),
        "original_data": sample.get("original_data"),
        "target_output": sample.get("target_output"),
        "windows": sample.get("windows"),
    }


def _choose_window_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    shuffled = [copy.deepcopy(row) for row in rows]
    rng.shuffle(shuffled)
    return shuffled[:n]


def _load_raw_rows_by_index(path: Path, wanted_indices: List[int]) -> Dict[int, Dict[str, Any]]:
    wanted = set(int(idx) for idx in wanted_indices)
    found: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx not in wanted:
                continue
            found[idx] = json.loads(line)
            if len(found) == len(wanted):
                break
    missing = sorted(wanted.difference(found))
    if missing:
        raise ValueError(f"Missing raw series rows for source indices: {missing[:10]}")
    return found


def _window_sample(raw_row: Dict[str, Any], window_row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(raw_row)
    start = int((window_row.get("target_interval") or {}).get("start", 0))
    end = int((window_row.get("target_interval") or {}).get("end", 0))
    source_window_id = str(window_row.get("sample_id") or f"window_{window_row.get('source_index', 0):05d}")
    root_channel = window_row.get("root_cause_channel")
    affected = list(window_row.get("affected_channels") or [])
    is_anom = bool(window_row.get("has_anomaly"))
    fact = {
        "is_anomalous": is_anom,
        "root_cause_channel": root_channel,
        "root_cause_channels": [root_channel] if root_channel else [],
        "affected_channels": affected,
        "anomaly_type": window_row.get("anomaly_type") if is_anom else None,
        "root_anomaly_name": (out.get("synthetic_label") or {}).get("root_anomaly_name") if is_anom else None,
        "anomaly_scope": window_row.get("anomaly_scope") if is_anom else None,
        "abnormal_edges": list(((out.get("synthetic_label") or {}).get("abnormal_edges") or [])) if is_anom else [],
        "causal_path": list(((out.get("synthetic_label") or {}).get("causal_path") or [])) if is_anom else [],
    }
    out["base_sample_id"] = str(out.get("base_sample_id") or out.get("sample_id") or source_window_id)
    out["source_sample_id"] = out.get("sample_id")
    out["source_window_sample_id"] = source_window_id
    out["sample_id"] = source_window_id
    out["target_interval"] = {"start": start, "end": end}
    out["root_cause"] = {
        "channel_id": root_channel,
        "time_index": start if root_channel else None,
        "interval": [start, end] if root_channel else None,
        "provided_by": "saved_window_metadata",
    }
    previous = dict(out.get("target_output") or {})
    previous["fact_check"] = fact
    previous["abnormality_score"] = 1.0 if is_anom else 0.0
    previous["answer_confidence"] = previous.get("answer_confidence", 0.72 if is_anom else 0.76)
    out["target_output"] = previous
    out["windows"] = [
        {
            "window_range": [start, end],
            "question": out.get("question"),
            "answer": previous.get("final_answer", ""),
            "question_type": out.get("question_type"),
            "has_anomaly": is_anom,
            "mv_label": {
                "root_cause_channel": root_channel,
                "affected_channels": affected,
                "anomaly_type": fact.get("anomaly_type"),
                "root_anomaly_name": fact.get("root_anomaly_name"),
                "anomaly_scope": fact.get("anomaly_scope"),
            },
        }
    ]
    out["question_generation_artifacts"] = {
        "image_path": str(window_row.get("image_path") or ""),
    }
    attach_channel_scales(out)
    return out


def _summary(
    *,
    windows_path: Path,
    raw_series_path: Path,
    selected_windows: List[Dict[str, Any]],
    completed: int,
    failed: int,
    args: argparse.Namespace,
    source_start_index: int,
    source_end_index: int | None,
    output_dir: Path,
    questions_path: Path,
    series_jsonl_path: Path,
    txt_path: Path,
    failures_path: Path,
    summary_path: Path,
    html_path: Path,
    llm_config: Dict[str, Any],
    label_counts: Dict[str, int],
    latencies: List[float],
    started: float,
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "windows_source": str(windows_path),
        "raw_series_source": str(raw_series_path),
        "windows_used": len(selected_windows),
        "source_start_index": int(source_start_index),
        "source_end_index": None if source_end_index is None else int(source_end_index),
        "generated_questions": completed,
        "failed_questions": failed,
        "requested_questions": int(args.num_questions),
        "output_dir": str(output_dir),
        "output_jsonl": str(questions_path),
        "output_series_jsonl": str(series_jsonl_path),
        "output_txt": str(txt_path),
        "output_html": str(html_path),
        "failures_jsonl": str(failures_path),
        "summary": str(summary_path),
        "llm_base_url": llm_config.get("base_url"),
        "llm_model": llm_config.get("model"),
        "question_provider": "visual_llm(question_provider.py, bank=hard)",
        "label_counts": label_counts,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="outputs/ts_data/ts_531_windows/windows_20000.jsonl")
    parser.add_argument("--raw-series", default="outputs/ts_data/ts_531/raw_series.jsonl")
    parser.add_argument("--llm-config", default="configs/gpt55_question_llm.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument(
        "--source-start-index",
        type=int,
        default=0,
        help="Inclusive raw tsdata row index to start from when filtering windows by source_index.",
    )
    parser.add_argument(
        "--source-end-index",
        type=int,
        default=None,
        help="Exclusive raw tsdata row index to stop at when filtering windows by source_index.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = _dated_output_dir("question_hard", int(args.num_questions))

    windows_path = Path(args.windows)
    raw_series_path = Path(relative_to_root(ROOT, args.raw_series))
    output_dir = Path(relative_to_root(ROOT, args.output_dir))
    series_dir = output_dir / "series"
    output_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    questions_path = output_dir / f"questions_{args.num_questions}.jsonl"
    series_jsonl_path = output_dir / f"series_{args.num_questions}.jsonl"
    txt_path = output_dir / f"questions_{args.num_questions}.txt"
    failures_path = output_dir / f"failures_{args.num_questions}.jsonl"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / f"questions_{args.num_questions}.html"
    questions_path.write_text("", encoding="utf-8")
    series_jsonl_path.write_text("", encoding="utf-8")
    txt_path.write_text("", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    window_rows = read_jsonl(windows_path)
    window_rows = _filter_windows_by_source_range(
        window_rows,
        source_start_index=int(args.source_start_index),
        source_end_index=args.source_end_index,
    )
    pair_count = max(1, (int(args.num_questions) + 1) // 2)
    if len(window_rows) < pair_count:
        raise ValueError(
            f"Requested {pair_count} windows for {int(args.num_questions)} questions, "
            f"but only {len(window_rows)} windows remain after source range filtering."
        )
    selected_windows = _choose_window_rows(window_rows, pair_count, int(args.seed))
    source_indices = [int(row["source_index"]) for row in selected_windows]
    raw_rows_by_index = _load_raw_rows_by_index(raw_series_path, source_indices)
    sampled_windows = [_window_sample(raw_rows_by_index[int(row["source_index"])], row) for row in selected_windows]
    paired_rows = duplicate_samples_with_question_frames(sampled_windows)[: int(args.num_questions)]

    llm_config = load_json(ROOT / args.llm_config)
    client = TrackingClient(
        create_llm_client(llm_config),
        max_retries=int(args.max_retries),
        retry_sleep=float(args.retry_sleep),
    )
    pair_axes = sample_question_axes_for_bank(len(sampled_windows), int(args.seed) + 17, question_bank="hard")
    axes: List[Dict[str, Any]] = []
    for axis in pair_axes:
        axes.append(dict(axis))
        axes.append(dict(axis))
    axes = axes[: len(paired_rows)]

    completed = 0
    failed = 0
    latencies: List[float] = []
    label_counts = {"anomalous": 0, "normal": 0}
    examples: List[Dict[str, Any]] = []
    started = time.perf_counter()
    progress = ProgressPrinter(
        len(paired_rows),
        label="question-hard",
        step_percent=float(args.progress_step_percent),
    )
    progress.start(extra=f"generated={completed} failed={failed}")
    summary = _summary(
        windows_path=windows_path,
        raw_series_path=raw_series_path,
        selected_windows=selected_windows,
        completed=completed,
        failed=failed,
        args=args,
        source_start_index=int(args.source_start_index),
        source_end_index=args.source_end_index,
        output_dir=output_dir,
        questions_path=questions_path,
        series_jsonl_path=series_jsonl_path,
        txt_path=txt_path,
        failures_path=failures_path,
        summary_path=summary_path,
        html_path=html_path,
        llm_config=llm_config,
        label_counts=label_counts,
        latencies=latencies,
        started=started,
        examples=examples,
    )

    for idx, (sample, axis) in enumerate(zip(paired_rows, axes), start=1):
        interval = sample["target_interval"]
        start, end = int(interval["start"]), int(interval["end"])
        difficulty = sample.get("question_frame_override") or sample.get("question_difficulty") or "anomaly_frame"
        try:
            spec = generate_question_spec_with_visual_workflow(
                sample,
                client,
                difficulty=str(difficulty),
                answer_type=axis["answer_type"],
                seed=int(args.seed) + 17,
                index=idx,
                question_bank="hard",
            )
            if spec is None:
                raise ValueError(
                    f"Hard visual question generation produced no valid {axis['answer_type']} question for "
                    f"{sample.get('sample_id') or idx}"
                )
        except Exception as exc:
            failed += 1
            failure = {
                "idx": idx,
                "sample_id": sample.get("sample_id"),
                "interval": sample.get("target_interval"),
                "difficulty": difficulty,
                "answer_type": axis["answer_type"],
                "llm_model": llm_config.get("model"),
                "llm_base_url": llm_config.get("base_url"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _append_jsonl(failures_path, failure)
            summary = _summary(
                windows_path=windows_path,
                raw_series_path=raw_series_path,
                selected_windows=selected_windows,
                completed=completed,
                failed=failed,
                args=args,
                output_dir=output_dir,
                questions_path=questions_path,
                series_jsonl_path=series_jsonl_path,
                txt_path=txt_path,
                failures_path=failures_path,
                summary_path=summary_path,
                html_path=html_path,
                llm_config=llm_config,
                label_counts=label_counts,
                latencies=latencies,
                started=started,
                examples=examples,
            )
            _write_json(summary_path, summary)
            progress.update(idx, extra=f"generated={completed} failed={failed}")
            print(f"[question-hard failed {failed}] idx={idx} {sample.get('sample_id')} error={exc}", flush=True)
            if args.stop_on_error:
                raise
            continue

        qa_sample = assign_question(
            sample,
            spec,
            index=None,
            copy_sample=False,
            preserve_source_question=True,
        )
        safe_id = _safe_name(str(qa_sample.get("sample_id", f"sample_{idx:04d}")))
        series_path = series_dir / f"{idx:04d}_{safe_id}.json"
        series_record = _series_record(qa_sample, idx)
        _write_json(series_path, series_record)

        fact = (qa_sample.get("target_output") or {}).get("fact_check") or {}
        has_anomaly = bool(fact.get("is_anomalous"))
        label_counts["anomalous" if has_anomaly else "normal"] += 1
        latency = float(getattr(client.last_response, "latency_seconds", 0.0) or 0.0)
        latencies.append(latency)

        artifacts = dict(qa_sample.get("question_generation_artifacts") or {})
        artifacts.update(
            {
                "idx": idx,
                "series_path": str(series_path),
                "llm_model": llm_config.get("model"),
                "llm_base_url": llm_config.get("base_url"),
                "latency_seconds": latency,
                "workflow": "hard",
                "question_bank": "hard",
            }
        )
        qa_sample["question_generation_artifacts"] = artifacts
        _append_jsonl(questions_path, qa_sample)
        _append_jsonl(series_jsonl_path, series_record)
        with txt_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{idx:04d}] {qa_sample.get('sample_id')} "
                f"{start}-{end} anomaly={has_anomaly} "
                f"{qa_sample.get('question_type')}\n"
                f"{qa_sample.get('question')}\n"
                f"image: {artifacts.get('image_path', '')}\n"
                f"series: {series_path}\n\n"
            )

        completed += 1
        if len(examples) < 10:
            examples.append(
                {
                    "idx": idx,
                    "sample_id": qa_sample.get("sample_id"),
                    "interval": qa_sample.get("target_interval"),
                    "question_type": qa_sample.get("question_type"),
                    "question_difficulty": qa_sample.get("question_difficulty"),
                    "question_answer_type": qa_sample.get("question_answer_type"),
                    "question": qa_sample.get("question"),
                    "has_anomaly": has_anomaly,
                    "image_path": artifacts.get("image_path"),
                    "series_path": str(series_path),
                }
            )

        summary = _summary(
            windows_path=windows_path,
            raw_series_path=raw_series_path,
            selected_windows=selected_windows,
            completed=completed,
            failed=failed,
            args=args,
            output_dir=output_dir,
            questions_path=questions_path,
            series_jsonl_path=series_jsonl_path,
            txt_path=txt_path,
            failures_path=failures_path,
            summary_path=summary_path,
            html_path=html_path,
            llm_config=llm_config,
            label_counts=label_counts,
            latencies=latencies,
            started=started,
            examples=examples,
        )
        _write_json(summary_path, summary)
        progress.update(idx, extra=f"generated={completed} failed={failed}")

    write_question_generation_html(questions_path, html_path)
    summary = _summary(
        windows_path=windows_path,
        raw_series_path=raw_series_path,
        selected_windows=selected_windows,
        completed=completed,
        failed=failed,
        args=args,
        output_dir=output_dir,
        questions_path=questions_path,
        series_jsonl_path=series_jsonl_path,
        txt_path=txt_path,
        failures_path=failures_path,
        summary_path=summary_path,
        html_path=html_path,
        llm_config=llm_config,
        label_counts=label_counts,
        latencies=latencies,
        started=started,
        examples=examples,
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

