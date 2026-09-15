from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.html_reports import write_question_generation_html
from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.question_provider import (
    assign_question,
    generate_question_spec_with_visual_workflow,
    question_frame_for_sample,
    sample_question_axes_for_bank,
)
from src.mvaxis.utils import load_json, read_jsonl, save_json


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _normalize_question_bank(value: str) -> str:
    bank = str(value or "regular").strip().lower()
    if bank not in {"regular", "hard"}:
        raise ValueError(f"Unsupported question bank: {value!r}")
    return bank


def _axes_for_rows(rows: Sequence[Dict[str, Any]], seed: int, question_bank: str) -> List[Dict[str, str]]:
    """Preserve paired-frame axis sharing for the hard bank and regular balancing otherwise."""

    bank = _normalize_question_bank(question_bank)
    if bank != "hard":
        return sample_question_axes_for_bank(len(rows), int(seed), question_bank=bank)

    pair_ids: List[str] = []
    for row in rows:
        pair_id = str(row.get("question_pair_id") or row.get("source_window_sample_id") or row.get("sample_id"))
        if pair_id not in pair_ids:
            pair_ids.append(pair_id)
    pair_axes = sample_question_axes_for_bank(len(pair_ids), int(seed), question_bank=bank)
    by_pair = dict(zip(pair_ids, pair_axes))
    return [
        dict(by_pair[str(row.get("question_pair_id") or row.get("source_window_sample_id") or row.get("sample_id"))])
        for row in rows
    ]


class TrackingClient:
    def __init__(self, client: Any, *, max_retries: int, retry_sleep: float) -> None:
        self.client = client
        self.max_retries = max(1, int(max_retries))
        self.retry_sleep = max(0.0, float(retry_sleep))

    def complete(self, messages: List[Dict[str, Any]]):
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.complete(messages)
                if not str(response.content or "").strip():
                    raise RuntimeError("LLM returned empty question content")
                return response
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    print(f"LLM request failed on attempt {attempt}/{self.max_retries}: {exc}", flush=True)
                    time.sleep(self.retry_sleep * attempt)
        assert last_error is not None
        raise last_error


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate MC/TF/OE question stems from prepared frame rows through GPT-5.5."
    )
    parser.add_argument("--frames", required=True, help="Frame JSONL produced by stage 03.")
    parser.add_argument("--output-dir", default="outputs/questions")
    parser.add_argument("--num-questions", type=int, default=None, help="Default: process every frame row.")
    parser.add_argument("--question-seed", type=int, default=602)
    parser.add_argument("--question-bank", choices=["regular", "hard"], default="regular")
    parser.add_argument("--llm-config", default="configs/gpt55_question.json")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Override api_key_env in the JSON config, e.g. OPENAI_API_KEY2.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bank = _normalize_question_bank(args.question_bank)
    frames_path = _resolve(args.frames)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "questions.jsonl"
    html_path = output_dir / "questions.html"
    summary_path = output_dir / "summary.json"

    frame_rows = read_jsonl(frames_path)
    if args.num_questions is not None:
        if int(args.num_questions) <= 0:
            raise ValueError("num-questions must be greater than zero.")
        frame_rows = frame_rows[: int(args.num_questions)]
    axes = _axes_for_rows(frame_rows, int(args.question_seed), bank)

    existing = read_jsonl(output_path) if args.resume and output_path.exists() else []
    if len(existing) > len(frame_rows):
        raise ValueError("Existing output contains more rows than the current frame selection.")
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    llm_config = load_json(_resolve(args.llm_config))
    if args.api_key_env:
        llm_config["api_key_env"] = str(args.api_key_env)
    client = TrackingClient(
        create_llm_client(llm_config),
        max_retries=int(args.max_retries),
        retry_sleep=float(args.retry_sleep),
    )

    rows = list(existing)
    errors: List[Dict[str, Any]] = []
    started = time.perf_counter()
    progress = ProgressPrinter(
        len(frame_rows),
        label=f"question-{bank}-gpt55",
        step_percent=float(args.progress_step_percent),
    )
    progress.start(extra=f"generated={len(rows)} errors=0")
    for index, (sample, axis) in enumerate(zip(frame_rows, axes), start=1):
        if index <= len(existing):
            continue
        try:
            frame = question_frame_for_sample(sample)
            spec = generate_question_spec_with_visual_workflow(
                sample,
                client,
                difficulty=frame,
                answer_type=axis["answer_type"],
                seed=int(args.question_seed),
                index=index,
                question_bank=bank,
            )
            if spec is None:
                raise ValueError(
                    f"GPT-5.5 produced no valid {axis['answer_type']} question for {sample.get('sample_id')}"
                )
            qa_row = assign_question(
                sample,
                spec,
                index=None,
                copy_sample=True,
                preserve_source_question=True,
            )
            rows.append(qa_row)
            _append_jsonl(output_path, qa_row)
        except Exception as exc:
            errors.append(
                {
                    "index": index,
                    "sample_id": sample.get("sample_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"[question failed {len(errors)}] idx={index} error={exc}", flush=True)
            if args.stop_on_error:
                raise
        progress.update(index, extra=f"generated={len(rows)} errors={len(errors)}")

    write_question_generation_html(
        output_path,
        html_path,
        title=f"{bank.title()} GPT-5.5 Questions",
    )
    answer_type_counts = Counter(str(row.get("question_answer_type") or "unknown") for row in rows)
    frame_counts = Counter(str(row.get("question_difficulty") or "unknown") for row in rows)
    summary = {
        "frames_source": str(frames_path),
        "output": str(output_path),
        "html": str(html_path),
        "question_bank": bank,
        "model": llm_config.get("model"),
        "api_key_env": llm_config.get("api_key_env"),
        "requested": len(frame_rows),
        "generated": len(rows),
        "answer_type_counts": dict(answer_type_counts),
        "question_frame_counts": dict(frame_counts),
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(summary, summary_path)
    print(json.dumps({**summary, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
