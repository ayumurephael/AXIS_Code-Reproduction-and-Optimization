from __future__ import annotations

import argparse
import copy
import json
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
    generate_question_spec_with_visual_workflow,
    question_frame_for_sample,
    sample_question_axes_for_bank,
)
from src.mvaxis.utils import load_json, read_jsonl, relative_to_root, save_json


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


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _series_record(sample: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "idx": idx,
        "sample_id": sample.get("sample_id"),
        "base_sample_id": sample.get("base_sample_id"),
        "source_sample_id": sample.get("source_sample_id"),
        "source_window_sample_id": sample.get("source_window_sample_id"),
        "target_interval": sample.get("target_interval"),
        "channels": sample.get("channels"),
        "series": sample.get("series"),
        "normal_series": sample.get("normal_series"),
        "original_data": sample.get("original_data"),
        "target_output": sample.get("target_output"),
        "windows": sample.get("windows"),
        "question_generation_artifacts": sample.get("question_generation_artifacts"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one visual question per selected window/series.")
    parser.add_argument("--windows", required=True)
    parser.add_argument("--raw-series", required=True)
    parser.add_argument("--llm-config", default="configs/gpt55_question_llm.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-bank", choices=("regular", "hard"), default="regular")
    parser.add_argument("--seed", type=int, default=6051)
    parser.add_argument("--question-seed", type=int, default=6052)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--progress-step-percent", type=float, default=5.0)
    parser.add_argument("--max-spec-attempts", type=int, default=3)
    args = parser.parse_args()

    windows_path = Path(args.windows)
    if not windows_path.is_absolute():
        windows_path = ROOT / windows_path
    raw_series_path = Path(relative_to_root(ROOT, args.raw_series))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    window_rows = read_jsonl(windows_path)
    questions_path = output_dir / f"questions_{len(window_rows)}.jsonl"
    series_jsonl_path = output_dir / f"series_{len(window_rows)}.jsonl"
    txt_path = output_dir / f"questions_{len(window_rows)}.txt"
    failures_path = output_dir / f"failures_{len(window_rows)}.jsonl"
    html_path = output_dir / f"questions_{len(window_rows)}.html"
    summary_path = output_dir / "summary.json"
    questions_path.write_text("", encoding="utf-8")
    series_jsonl_path.write_text("", encoding="utf-8")
    txt_path.write_text("", encoding="utf-8")
    failures_path.write_text("", encoding="utf-8")

    source_indices = [int(row["source_index"]) for row in window_rows]
    raw_rows_by_index = _load_raw_rows_by_index(raw_series_path, source_indices)
    samples = [_window_sample(raw_rows_by_index[int(row["source_index"])], row) for row in window_rows]

    llm_config = load_json(ROOT / args.llm_config)
    client = TrackingClient(
        create_llm_client(llm_config),
        max_retries=int(args.max_retries),
        retry_sleep=float(args.retry_sleep),
    )
    axes = sample_question_axes_for_bank(len(samples), int(args.question_seed), question_bank=str(args.question_bank))

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    failed = 0
    label_counts = {"anomalous": 0, "normal": 0}
    answer_type_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    progress = ProgressPrinter(len(samples), label=f"question-{args.question_bank}", step_percent=float(args.progress_step_percent))
    progress.start(extra=f"samples={len(samples)}")
    with txt_path.open("a", encoding="utf-8") as txt_handle:
        for idx, (sample, axis) in enumerate(zip(samples, axes), start=1):
            difficulty = question_frame_for_sample(sample)
            spec = None
            error = None
            for attempt in range(1, max(1, int(args.max_spec_attempts)) + 1):
                try:
                    spec = generate_question_spec_with_visual_workflow(
                        sample,
                        client,
                        difficulty=difficulty,
                        answer_type=axis["answer_type"],
                        seed=int(args.question_seed) + attempt * 1000,
                        index=idx,
                        question_bank=str(args.question_bank),
                    )
                    if spec is not None:
                        break
                    error = RuntimeError("Question spec generation returned None")
                except Exception as exc:
                    error = exc
            if spec is None:
                failed += 1
                _append_jsonl(
                    failures_path,
                    {
                        "idx": idx,
                        "sample_id": sample.get("sample_id"),
                        "source_window_sample_id": sample.get("source_window_sample_id"),
                        "question_bank": args.question_bank,
                        "difficulty": difficulty,
                        "answer_type": axis.get("answer_type"),
                        "error": str(error) if error else "unknown",
                    },
                )
                progress.update(idx, extra=f"generated={len(rows)} failed={failed}")
                continue

            qa_row = assign_question(sample, spec, copy_sample=True, preserve_source_question=True)
            rows.append(qa_row)
            _append_jsonl(questions_path, qa_row)
            _append_jsonl(series_jsonl_path, _series_record(qa_row, idx))
            answer = ((qa_row.get("target_output") or {}).get("final_answer") or "").strip()
            txt_handle.write(f"[{idx:04d}] {qa_row.get('sample_id')}\n")
            txt_handle.write(f"Question: {qa_row.get('question', '')}\n")
            txt_handle.write(f"Answer: {answer}\n\n")

            fact = (qa_row.get("target_output") or {}).get("fact_check") or {}
            has_anomaly = bool(fact.get("is_anomalous"))
            label_counts["anomalous" if has_anomaly else "normal"] += 1
            answer_type = str(qa_row.get("question_answer_type") or "unknown")
            answer_type_counts[answer_type] = answer_type_counts.get(answer_type, 0) + 1
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
            if len(examples) < 12:
                examples.append(
                    {
                        "idx": idx,
                        "sample_id": qa_row.get("sample_id"),
                        "source_window_sample_id": qa_row.get("source_window_sample_id"),
                        "question_type": qa_row.get("question_type"),
                        "question_answer_type": qa_row.get("question_answer_type"),
                        "question": qa_row.get("question"),
                        "has_anomaly": has_anomaly,
                        "image_path": ((qa_row.get("question_generation_artifacts") or {}).get("image_path")),
                    }
                )
            progress.update(idx, extra=f"generated={len(rows)} failed={failed}")

    write_question_generation_html(questions_path, html_path, title=f"{str(args.question_bank).title()} SWaT Questions: {questions_path.name}")
    summary = {
        "windows_source": str(windows_path),
        "raw_series_source": str(raw_series_path),
        "generated_questions": len(rows),
        "failed_questions": failed,
        "selected_windows": len(window_rows),
        "output_dir": str(output_dir),
        "output_jsonl": str(questions_path),
        "output_series_jsonl": str(series_jsonl_path),
        "output_txt": str(txt_path),
        "output_html": str(html_path),
        "failures_jsonl": str(failures_path),
        "summary": str(summary_path),
        "question_provider": f"visual_llm(question_provider.py, bank={args.question_bank}, one_per_series)",
        "question_bank": str(args.question_bank),
        "llm_base_url": llm_config.get("base_url"),
        "llm_model": llm_config.get("model"),
        "seed": int(args.seed),
        "question_seed": int(args.question_seed),
        "label_counts": label_counts,
        "answer_type_counts": answer_type_counts,
        "difficulty_counts": difficulty_counts,
        "elapsed_seconds": time.perf_counter() - started,
        "examples": examples,
    }
    save_json(summary, summary_path)
    progress.update(len(samples), extra=f"generated={len(rows)} failed={failed}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
