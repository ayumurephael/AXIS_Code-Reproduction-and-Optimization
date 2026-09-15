from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from _bootstrap import add_project_root

ROOT = add_project_root()

from src.mvaxis.llm_client import create_llm_client
from src.mvaxis.html_reports import write_answer_generation_html
from src.mvaxis.progress import ProgressPrinter
from src.mvaxis.teacher_answer_provider import (
    build_teacher_answer_messages,
    build_teacher_answer_prompt,
    prompt_from_messages,
)
from src.mvaxis.utils import load_json, read_jsonl, save_json, strip_think_blocks, write_jsonl


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return ROOT / p


def _load_llm_config(
    path: Path,
    max_tokens: int | None,
    temperature: float | None,
    api_key_env: str | None,
) -> Dict[str, Any]:
    config = load_json(path)
    if max_tokens is not None:
        config["max_tokens"] = int(max_tokens)
    if temperature is not None:
        config["temperature"] = float(temperature)
    if api_key_env:
        config["api_key_env"] = str(api_key_env)
    return config


def _with_teacher_answer(
    row: Dict[str, Any],
    *,
    answer: str,
    raw: Dict[str, Any],
    latency_seconds: float,
    llm_config: Dict[str, Any],
    prompt_preview: str | None,
) -> Dict[str, Any]:
    out = dict(row)
    out["teacher_answer_llm"] = {
        "answer": answer,
        "provider": llm_config.get("provider", "openai_compatible"),
        "model": llm_config.get("model"),
        "latency_seconds": latency_seconds,
        "raw": raw,
    }
    if prompt_preview is not None:
        out["teacher_answer_llm"]["prompt_preview"] = prompt_preview
    return out


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _first_window_answer(row: Dict[str, Any]) -> str:
    """Return the first QA-window reference answer from the source JSONL row."""

    windows = row.get("windows") or []
    if not windows or not isinstance(windows[0], dict):
        return ""
    answer = windows[0].get("answer")
    return "" if answer is None else str(answer)


def _row_question(row: Dict[str, Any]) -> str:
    """Return the question text associated with the source JSONL row."""

    question = row.get("question")
    if question is None:
        windows = row.get("windows") or []
        if windows and isinstance(windows[0], dict):
            question = windows[0].get("question")
    return "" if question is None else str(question)


def _answer_only(answer: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """Build one readable answer JSONL row with question, reference, and model answer."""

    question = _row_question(row)
    reference_answer = _first_window_answer(row)
    model_answer = strip_think_blocks("" if answer is None else str(answer))
    combined = "\n\n".join(
        [
            f"Question: {question}",
            f"windows[0].answer: {reference_answer}",
            f"model_answer: {model_answer}",
        ]
    )
    return {
        "answer": combined,
        "question": question,
        "windows_0_answer": reference_answer,
        "model_answer": model_answer,
    }


def _successful_prefix(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the contiguous successful prefix so resume restarts from first incomplete row."""

    prefix: List[Dict[str, Any]] = []
    for row in rows:
        teacher = row.get("teacher_answer_llm") or {}
        answer = teacher.get("answer")
        if answer is None or row.get("teacher_answer_error"):
            break
        prefix.append(row)
    return prefix


class TrackingClient:
    """Record the latest LLM call and retry transient API failures."""

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
                return self.last_response
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                print(f"LLM request failed on attempt {attempt}/{self.max_retries}: {exc}", flush=True)
                time.sleep(self.retry_sleep * attempt)
        raise last_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AXIS-style natural-language teacher answers.")
    parser.add_argument("--data", required=True, help="Input QA JSONL.")
    parser.add_argument("--llm-config", default="configs/gpt54_answer.json")
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Override api_key_env in the JSON config, e.g. OPENAI_API_KEY2.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL with teacher_answer_llm attached.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-window-rows", type=int, default=0, help="0 means include the full target window.")
    parser.add_argument("--digits", type=int, default=3)
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--prompt-examples", type=int, default=3)
    parser.add_argument("--save-prompts", default=None, help="Optional JSON file for prompt previews.")
    parser.add_argument("--summary", default=None, help="Optional summary JSON path.")
    parser.add_argument("--answers-output", default=None, help="Optional answer-only JSONL path.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--progress-step-percent", type=float, default=2.5)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the first incomplete row in an existing output JSONL.")
    args = parser.parse_args()

    data_path = _resolve(args.data)
    output_path = _resolve(args.output)
    llm_config_path = _resolve(args.llm_config)
    prompt_path = _resolve(args.save_prompts) if args.save_prompts else output_path.with_suffix(".prompts.json")
    summary_path = _resolve(args.summary) if args.summary else output_path.with_suffix(".summary.json")
    answers_path = _resolve(args.answers_output) if args.answers_output else output_path.with_suffix(".answers.jsonl")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    rows = read_jsonl(data_path)
    selected = rows[int(args.start_index) :]
    if args.limit is not None:
        selected = selected[: int(args.limit)]

    max_window_rows = None if int(args.max_window_rows) <= 0 else int(args.max_window_rows)
    prompt_examples: List[Dict[str, Any]] = []
    if args.resume and output_path.exists():
        completed_rows = _successful_prefix(read_jsonl(output_path))
        write_jsonl(completed_rows, output_path)
        answer_outputs = [
            _answer_only((row.get("teacher_answer_llm") or {}).get("answer"), row)
            for row in completed_rows
        ]
        write_jsonl(answer_outputs, answers_path)
    else:
        completed_rows = []
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_path.write_text("", encoding="utf-8")
    generated = 0
    skipped = len(completed_rows)
    processed_count = len(completed_rows)
    answer_count = len(completed_rows)
    errors: List[Dict[str, Any]] = []

    llm_config = _load_llm_config(
        llm_config_path,
        args.max_tokens,
        args.temperature,
        args.api_key_env,
    )
    client = (
        None
        if args.preview_only
        else TrackingClient(
            create_llm_client(llm_config),
            max_retries=int(args.max_retries),
            retry_sleep=float(args.retry_sleep),
        )
    )
    progress = ProgressPrinter(
        len(selected),
        label="teacher-answer",
        step_percent=float(args.progress_step_percent),
    )
    progress.start(extra=f"generated={generated} skipped={skipped} errors={len(errors)}")

    pending = selected[len(completed_rows) :]
    for offset, row in enumerate(pending, start=len(completed_rows)):
        index = int(args.start_index) + offset
        current = offset + 1
        messages = build_teacher_answer_messages(
            row,
            include_image=not args.no_image,
            max_window_rows=max_window_rows,
            digits=int(args.digits),
        )
        prompt_preview = prompt_from_messages(messages)
        keep_prompt_preview = len(prompt_examples) < int(args.prompt_examples)
        if keep_prompt_preview:
            prompt_examples.append(
                {
                    "index": index,
                    "sample_id": row.get("sample_id"),
                    "question": row.get("question"),
                    "prompt": prompt_preview,
                }
            )

        if row.get("teacher_answer_llm") and not args.overwrite:
            skipped += 1
            processed_count += 1
            answer_count += 1
            progress.update(current, extra=f"generated={generated} skipped={skipped} errors={len(errors)}")
            continue

        if args.preview_only:
            preview_row = dict(row)
            preview_row["teacher_answer_prompt_preview"] = build_teacher_answer_prompt(
                row,
                max_window_rows=max_window_rows,
                digits=int(args.digits),
            )
            _append_jsonl(output_path, preview_row)
            processed_count += 1
            progress.update(current, extra=f"previewed={processed_count}")
            continue

        try:
            response = client.complete(messages)  # type: ignore[union-attr]
            cleaned_answer = strip_think_blocks(response.content)
            answer_row = _answer_only(cleaned_answer, row)
            output_row = _with_teacher_answer(
                row,
                answer=cleaned_answer,
                raw=response.raw,
                latency_seconds=response.latency_seconds,
                llm_config=llm_config,
                prompt_preview=prompt_preview if keep_prompt_preview else None,
            )
            _append_jsonl(output_path, output_row)
            _append_jsonl(answers_path, answer_row)
            generated += 1
            processed_count += 1
            answer_count += 1
            progress.update(current, extra=f"generated={generated} skipped={skipped} errors={len(errors)}")
        except Exception as exc:
            failed = dict(row)
            failed["teacher_answer_error"] = str(exc)
            _append_jsonl(output_path, failed)
            processed_count += 1
            errors.append(
                {
                    "index": index,
                    "sample_id": row.get("sample_id"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            progress.update(current, extra=f"generated={generated} skipped={skipped} errors={len(errors)}")
            print(f"[failed {len(errors)}] idx={index} {row.get('sample_id')} error={exc}", flush=True)
            if args.stop_on_error:
                raise

    html_path = write_answer_generation_html(output_path, output_path.with_suffix(".html"))
    answers_html_path = write_answer_generation_html(answers_path, answers_path.with_suffix(".html"))
    save_json({"prompts": prompt_examples}, prompt_path)
    save_json(
        {
            "input": str(data_path),
            "output": str(output_path),
            "answers_output": str(answers_path),
            "html_output": str(html_path),
            "answers_html_output": str(answers_html_path),
            "llm_config": str(llm_config_path),
            "model": llm_config.get("model"),
            "llm_base_url": llm_config.get("base_url"),
            "preview_only": bool(args.preview_only),
            "include_image": not args.no_image,
            "start_index": int(args.start_index),
            "requested_limit": args.limit,
            "processed": processed_count,
            "generated": generated,
            "skipped": skipped,
            "answers_saved": answer_count,
            "max_retries": int(args.max_retries),
            "retry_sleep": float(args.retry_sleep),
            "errors": errors,
        },
        summary_path,
    )
    if prompt_examples:
        _write_text(output_path.with_suffix(".prompt_example.txt"), prompt_examples[0]["prompt"])


if __name__ == "__main__":
    main()
