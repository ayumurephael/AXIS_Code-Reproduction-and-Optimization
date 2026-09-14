from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.html_reports import write_question_generation_html
from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.question_provider import (
    ANOMALY_RATIO,
    MAX_WINDOW_SIZE,
    MIN_WINDOW_SIZE,
    SAMPLES_PER_SERIES,
    _generate_multichannel_time_series_image,
    assign_question,
    duplicate_samples_with_question_frames,
    expand_samples_with_analysis_windows,
    generate_question_spec_with_visual_workflow,
    question_frame_for_sample,
)
from src.mvaxis.question_provider import sample_question_axes
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root


def _dated_output_dir(prefix: str, count: int) -> str:
    """Return the standard dated output directory name."""

    return f"outputs/{prefix}_{int(count)}_{datetime.now().strftime('%m%d')}"


def _normalize_source_range(total: int, start: int, end: int | None) -> tuple[int, int]:
    range_start = max(0, int(start))
    range_end = total if end is None else int(end)
    range_end = min(total, max(range_start, range_end))
    if range_start >= total:
        raise ValueError(f"source-start-index {range_start} is outside the available range 0..{max(total - 1, 0)}")
    return range_start, range_end


class TrackingClient:
    """Record metadata from the latest LLM completion without changing the client API."""

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


def _choose_source_rows(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """Sample source series deterministically, repeating only if necessary."""

    rng = random.Random(seed)
    shuffled = [copy.deepcopy(row) for row in rows]
    rng.shuffle(shuffled)
    if n <= len(shuffled):
        return shuffled[:n]
    selected: List[Dict[str, Any]] = []
    while len(selected) < n:
        batch = [copy.deepcopy(row) for row in shuffled]
        rng.shuffle(batch)
        selected.extend(batch)
    return selected[:n]


def _safe_name(value: str, limit: int = 120) -> str:
    """Return a filesystem-safe compact name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return cleaned[:limit] or "sample"


def _pair_image_key(sample: Dict[str, Any]) -> str:
    """Return a stable key so paired questions can share one rendered interval image."""

    return str(
        sample.get("question_pair_id")
        or sample.get("source_window_sample_id")
        or sample.get("sample_id")
        or "sample"
    )


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write one JSON file with stable UTF-8 formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSONL record."""

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _image_message_count(messages: List[Dict[str, Any]]) -> int:
    """Count image_url message parts in one OpenAI-style message list."""

    count = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")
    return count


def _series_record(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Extract the full source time-series payload saved beside each question."""

    return {
        "idx": idx,
        "sample_id": sample.get("sample_id"),
        "base_sample_id": sample.get("base_sample_id"),
        "source_sample_id": sample.get("source_sample_id"),
        "target_interval": sample.get("target_interval"),
        "channels": sample.get("channels"),
        "series": sample.get("series"),
        "original_data": sample.get("original_data"),
        "target_output": sample.get("target_output"),
        "windows": sample.get("windows"),
    }


def _summary(
    *,
    source_path: Path,
    source_rows: List[Dict[str, Any]],
    selected_rows: List[Dict[str, Any]],
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
    per_series: int,
    label_counts: Dict[str, int],
    image_messages_built: int,
    latencies: List[float],
    started: float,
    examples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the generation progress summary written after every record."""

    return {
        "source": str(source_path),
        "source_rows": len(source_rows),
        "source_rows_used": len(selected_rows),
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
        "window_sampling": {
            "samples_per_series": per_series,
            "anomaly_ratio": float(args.anomaly_ratio),
            "min_window_size": int(args.min_window_size),
            "max_window_size": int(args.max_window_size),
        },
        "label_counts": label_counts,
        "image_messages_built": image_messages_built,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "elapsed_seconds": time.perf_counter() - started,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="outputs/runs/fixed60_qa600_legacy_dag/data/raw_series.jsonl",
    )
    parser.add_argument("--llm-config", default="configs/gpt55_question_llm.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument(
        "--source-start-index",
        type=int,
        default=0,
        help="Inclusive raw tsdata row index to start from before sampling source series.",
    )
    parser.add_argument(
        "--source-end-index",
        type=int,
        default=None,
        help="Exclusive raw tsdata row index to stop at before sampling source series.",
    )
    parser.add_argument("--samples-per-series", type=int, default=SAMPLES_PER_SERIES)
    parser.add_argument("--min-window-size", type=int, default=MIN_WINDOW_SIZE)
    parser.add_argument("--max-window-size", type=int, default=MAX_WINDOW_SIZE)
    parser.add_argument("--anomaly-ratio", type=float, default=ANOMALY_RATIO)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = _dated_output_dir("question", int(args.num_questions))

    source_path = Path(relative_to_root(ROOT, args.source))
    output_dir = Path(relative_to_root(ROOT, args.output_dir))
    images_dir = output_dir / "images"
    series_dir = output_dir / "series"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    series_dir.mkdir(parents=True, exist_ok=True)

    questions_path = output_dir / f"questions_{args.num_questions}.jsonl"
    series_jsonl_path = output_dir / f"series_{args.num_questions}.jsonl"
    txt_path = output_dir / f"questions_{args.num_questions}.txt"
    failures_path = output_dir / f"failures_{args.num_questions}.jsonl"
    summary_path = output_dir / "summary.json"
    html_path = output_dir / f"questions_{args.num_questions}.html"
    if not args.resume:
        questions_path.write_text("", encoding="utf-8")
        series_jsonl_path.write_text("", encoding="utf-8")
        txt_path.write_text("", encoding="utf-8")
        failures_path.write_text("", encoding="utf-8")

    source_rows = read_jsonl(source_path)
    range_start, range_end = _normalize_source_range(len(source_rows), int(args.source_start_index), args.source_end_index)
    source_rows = source_rows[range_start:range_end]
    per_series = max(1, int(args.samples_per_series))
    pair_count = max(1, math.ceil(int(args.num_questions) / 2))
    source_needed = max(1, math.ceil(pair_count / per_series))
    selected_rows = _choose_source_rows(source_rows, source_needed, int(args.seed))
    sampled_windows = expand_samples_with_analysis_windows(
        selected_rows,
        seed=int(args.seed),
        samples_per_series=per_series,
        min_window_size=int(args.min_window_size),
        max_window_size=int(args.max_window_size),
        anomaly_ratio=float(args.anomaly_ratio),
    )[:pair_count]
    window_rows = duplicate_samples_with_question_frames(sampled_windows)[: int(args.num_questions)]

    llm_config = load_json(ROOT / args.llm_config)
    client = TrackingClient(
        create_llm_client(llm_config),
        max_retries=int(args.max_retries),
        retry_sleep=float(args.retry_sleep),
    )
    pair_axes = sample_question_axes(len(sampled_windows), int(args.seed) + 17)
    axes: List[Dict[str, Any]] = []
    for axis in pair_axes:
        axes.append(dict(axis))
        axes.append(dict(axis))
    axes = axes[: len(window_rows)]

    completed = 0
    failed = 0
    image_messages_built = 0
    latencies: List[float] = []
    label_counts = {"anomalous": 0, "normal": 0}
    examples: List[Dict[str, Any]] = []
    rendered_images: Dict[str, Path] = {}
    started = time.perf_counter()
    progress = ProgressPrinter(
        len(window_rows),
        label="question-gen",
        step_percent=float(args.progress_step_percent),
    )
    progress.start(extra=f"generated={completed} failed={failed}")
    summary = _summary(
        source_path=source_path,
        source_rows=source_rows,
        selected_rows=selected_rows,
        completed=completed,
        failed=failed,
        args=args,
        source_start_index=range_start,
        source_end_index=range_end,
        output_dir=output_dir,
        questions_path=questions_path,
        series_jsonl_path=series_jsonl_path,
        txt_path=txt_path,
        failures_path=failures_path,
        summary_path=summary_path,
        html_path=html_path,
        llm_config=llm_config,
        per_series=per_series,
        label_counts=label_counts,
        image_messages_built=image_messages_built,
        latencies=latencies,
        started=started,
        examples=examples,
    )

    for idx, (sample, axis) in enumerate(zip(window_rows, axes), start=1):
        interval = sample["target_interval"]
        start, end = int(interval["start"]), int(interval["end"])
        difficulty = question_frame_for_sample(sample)
        try:
            spec = generate_question_spec_with_visual_workflow(
                sample,
                client,
                difficulty=difficulty,
                answer_type=axis["answer_type"],
                seed=int(args.seed) + 17,
                index=idx,
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
                source_path=source_path,
                source_rows=source_rows,
                selected_rows=selected_rows,
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
                per_series=per_series,
                label_counts=label_counts,
                image_messages_built=image_messages_built,
                latencies=latencies,
                started=started,
                examples=examples,
            )
            _write_json(summary_path, summary)
            progress.update(idx, extra=f"generated={completed} failed={failed}")
            print(f"[question-gen failed {failed}] idx={idx} {sample.get('sample_id')} error={exc}", flush=True)
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

        pair_key = _pair_image_key(qa_sample)
        image_path = rendered_images.get(pair_key)
        if image_path is None:
            safe_id = _safe_name(pair_key)
            image_path = images_dir / f"{safe_id}.png"
            image_base64 = _generate_multichannel_time_series_image(
                qa_sample,
                window_start=start,
                window_end=end,
            )
            if image_base64:
                image_path.write_bytes(base64.b64decode(image_base64))
                rendered_images[pair_key] = image_path
            else:
                image_path = Path("")

        safe_id = _safe_name(str(qa_sample.get("sample_id", f"sample_{idx:04d}")))
        series_path = series_dir / f"{idx:04d}_{safe_id}.json"
        series_record = _series_record(qa_sample, idx)
        _write_json(series_path, series_record)

        fact = (qa_sample.get("target_output") or {}).get("fact_check") or {}
        has_anomaly = bool(fact.get("is_anomalous"))
        label_counts["anomalous" if has_anomaly else "normal"] += 1
        latency = float(getattr(client.last_response, "latency_seconds", 0.0) or 0.0)
        latencies.append(latency)
        images_in_prompt = _image_message_count(client.last_messages)
        image_messages_built += images_in_prompt

        qa_sample["question_generation_artifacts"] = {
            "idx": idx,
            "image_path": str(image_path),
            "series_path": str(series_path),
            "llm_model": llm_config.get("model"),
            "llm_base_url": llm_config.get("base_url"),
            "latency_seconds": latency,
            "image_messages_built": images_in_prompt,
        }
        _append_jsonl(questions_path, qa_sample)
        _append_jsonl(series_jsonl_path, series_record)
        with txt_path.open("a", encoding="utf-8") as f:
            f.write(
                f"[{idx:04d}] {qa_sample.get('sample_id')} "
                f"{start}-{end} anomaly={has_anomaly} "
                f"{qa_sample.get('question_type')}\n"
                f"{qa_sample.get('question')}\n"
                f"image: {image_path}\n"
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
                    "image_path": str(image_path),
                    "series_path": str(series_path),
                }
            )

        summary = _summary(
            source_path=source_path,
            source_rows=source_rows,
            selected_rows=selected_rows,
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
            per_series=per_series,
            label_counts=label_counts,
            image_messages_built=image_messages_built,
            latencies=latencies,
            started=started,
            examples=examples,
        )
        _write_json(summary_path, summary)
        progress.update(idx, extra=f"generated={completed} failed={failed}")

    write_question_generation_html(questions_path, html_path)
    summary = _summary(
        source_path=source_path,
        source_rows=source_rows,
        selected_rows=selected_rows,
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
        per_series=per_series,
        label_counts=label_counts,
        image_messages_built=image_messages_built,
        latencies=latencies,
        started=started,
        examples=examples,
    )
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
