from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DATA_BLOCK_RE = re.compile(
    r"- Data \[current_value(?:\(normal_value\))?\]: \[(.*?)\]\s*(?:\n\n|\n### |\nQuestion:)",
    re.DOTALL,
)
ROW_RE = re.compile(r"t=(\d+):\s*(.*)")
CHANNEL_RE = re.compile(r"([A-Za-z0-9_]+)=(-?\d+(?:\.\d+)?)")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_data_block(prompt: str) -> Tuple[List[List[float]], List[str]]:
    match = DATA_BLOCK_RE.search(prompt)
    if not match:
        raise ValueError("Could not locate Data block in prompt.")

    body = match.group(1).strip()
    values: List[List[float]] = []
    channel_ids: List[str] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row_match = ROW_RE.match(line)
        if not row_match:
            continue
        payload = row_match.group(2)
        entries = CHANNEL_RE.findall(payload)
        if not entries:
            raise ValueError(f"Could not parse channel values from line: {line}")
        if not channel_ids:
            channel_ids = [name for name, _ in entries]
        values.append([float(value) for _, value in entries])

    if not values or not channel_ids:
        raise ValueError("Parsed empty series from prompt Data block.")
    return values, channel_ids


def _question_answer_type(row: Dict[str, Any]) -> str:
    parsed = row.get("parsed_response") or {}
    target = row.get("target_output") or {}
    fact = target.get("fact_check") or {}
    return str(
        row.get("question_answer_type")
        or parsed.get("question_answer_type")
        or fact.get("question_answer_type")
        or "open"
    ).lower()


def _question_difficulty(row: Dict[str, Any]) -> str | None:
    target = row.get("target_output") or {}
    fact = target.get("fact_check") or {}
    value = row.get("question_difficulty") or fact.get("question_difficulty")
    return None if value is None else str(value)


def _series_map_key(series_row: Dict[str, Any]) -> str:
    return str(
        series_row.get("sample_id")
        or series_row.get("base_sample_id")
        or series_row.get("source_sample_id")
        or ""
    )


def _window_meta(
    index: int,
    windows_rows: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not windows_rows:
        return None
    if not (0 <= index < len(windows_rows)):
        raise IndexError(f"Window index {index} out of range for {len(windows_rows)} rows.")
    return windows_rows[index]


def _full_series_row(
    window_row: Optional[Dict[str, Any]],
    raw_series_map: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not window_row or not raw_series_map:
        return None
    candidates = [
        str(window_row.get("source_sample_id") or ""),
        str(window_row.get("sample_id") or "").split("_w", 1)[0],
        f"ts_data_61_{int(window_row.get('source_index')):05d}" if window_row.get("source_index") is not None else "",
    ]
    for candidate in candidates:
        if candidate and candidate in raw_series_map:
            return raw_series_map[candidate]
    return None


def _rebuild_row(
    row: Dict[str, Any],
    *,
    index: int,
    windows_rows: Optional[List[Dict[str, Any]]] = None,
    raw_series_map: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    window_values, parsed_channel_ids = _parse_data_block(prompt)
    interval = row.get("prompt_interval") or [0, len(window_values)]
    start = int(interval[0])
    end = int(interval[1])
    window_row = _window_meta(index, windows_rows)
    series_row = _full_series_row(window_row, raw_series_map)

    if series_row is not None:
        full_values = (
            ((series_row.get("original_data") or {}).get("time_series"))
            or ((series_row.get("series") or {}).get("values"))
            or []
        )
        channels = series_row.get("channels") or []
        image_path = window_row.get("image_path") if window_row else row.get("raw_image_path")
        sample_id = window_row.get("sample_id") if window_row else row.get("sample_id")
    else:
        full_values = window_values
        channels = [{"channel_id": channel_id} for channel_id in parsed_channel_ids]
        image_path = row.get("raw_image_path")
        sample_id = row.get("sample_id")

    rebuilt = {
        "sample_id": sample_id,
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "question_answer_type": _question_answer_type(row),
        "question_difficulty": _question_difficulty(row),
        "target_interval": {"start": start, "end": end},
        "series": {"values": full_values},
        "channels": channels,
        "target_output": row.get("target_output"),
        "raw_image_path": image_path,
        "question_generation_artifacts": {
            "image_path": image_path,
            "recovered_from": "qwen_raw_answers.jsonl",
        },
        "global_information": "Recovered from historical qwen_raw_answers.jsonl prompt.",
    }
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild mvaxis question JSONL from historical qwen_raw_answers.jsonl."
    )
    parser.add_argument("--input", required=True, help="Path to qwen_raw_answers.jsonl")
    parser.add_argument("--output", required=True, help="Path to rebuilt question JSONL")
    parser.add_argument("--windows", default=None, help="Optional windows JSONL used to recover sample ids and image paths.")
    parser.add_argument("--raw-series", default=None, help="Optional raw-series JSONL used to recover full sequences.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = _read_jsonl(input_path)
    windows_rows = _read_jsonl(Path(args.windows)) if args.windows else None
    raw_series_rows = _read_jsonl(Path(args.raw_series)) if args.raw_series else None
    raw_series_map = (
        {_series_map_key(row): row for row in raw_series_rows if _series_map_key(row)}
        if raw_series_rows
        else None
    )
    rebuilt = [
        _rebuild_row(
            row,
            index=index,
            windows_rows=windows_rows,
            raw_series_map=raw_series_map,
        )
        for index, row in enumerate(rows)
    ]
    _write_jsonl(output_path, rebuilt)

    summary = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": len(rebuilt),
        "used_windows": bool(windows_rows),
        "used_raw_series": bool(raw_series_map),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
