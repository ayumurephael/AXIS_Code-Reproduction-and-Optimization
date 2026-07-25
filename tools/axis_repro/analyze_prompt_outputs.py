"""Deterministic format and response-behavior diagnostics for prompt ablations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .common import extract_choice, extract_numbers, extract_true_false, read_jsonl


THINK_OPEN = re.compile(r"<think>", flags=re.I)
THINK_CLOSE = re.compile(r"</think>", flags=re.I)
MC_FIRST = re.compile(r"^\s*([A-D])\)\s*(\S[^\r\n]*)", flags=re.I)
TF_FIRST = re.compile(r"^\s*(True|False)\.", flags=re.I)
OPTION_LINE = re.compile(
    r"^\s*([A-D])[\).]\s*(.+?)\s*$",
    flags=re.I | re.M,
)


def think_state(text: str) -> str:
    has_open = THINK_OPEN.search(text or "") is not None
    has_close = THINK_CLOSE.search(text or "") is not None
    if has_open and has_close:
        return "open_and_close"
    if has_open:
        return "open_only"
    if has_close:
        return "close_only"
    return "none"


def sentence_count(text: str) -> int:
    return len(re.findall(r"(?<=[.!?])(?:\s+|$)", text.strip()))


def question_options(question: str) -> dict[str, str]:
    return {
        label.upper(): value.strip()
        for label, value in OPTION_LINE.findall(question or "")
    }


def unsupported_number_fraction(row: dict) -> float | None:
    response_numbers = extract_numbers(row.get("response", ""))
    if not response_numbers:
        return None
    allowed = set(extract_numbers(row.get("question", "")))
    allowed.update(
        {
            str(row.get("start_index")),
            str(row.get("end_index")),
            str(int(row.get("end_index", 0)) - 1),
        }
    )
    start = int(row.get("start_index", 0))
    end = int(row.get("end_index", 0))
    for value in row.get("time_series", [])[start:end]:
        allowed.add(f"{value * 100:.0f}")
    unsupported = sum(value not in allowed for value in response_numbers)
    return unsupported / len(response_numbers)


def diagnose(row: dict) -> dict:
    response = str(row.get("response", ""))
    qtype = str(row.get("question_type", "unknown"))
    stripped = response.lstrip()
    state = think_state(response)
    first_line = stripped.splitlines()[0].strip() if stripped else ""
    raw_answer_first = False
    strict_parse = False
    selected_text_exact = None
    heuristic_correct = None

    if qtype == "multiple_choice":
        match = MC_FIRST.match(stripped)
        raw_answer_first = match is not None
        strict_parse = raw_answer_first
        predicted = match.group(1).upper() if match else extract_choice(response)
        expected = extract_choice(str(row.get("answer", "")))
        if predicted is not None and expected is not None:
            heuristic_correct = predicted == expected
        if match:
            options = question_options(str(row.get("question", "")))
            option_text = options.get(match.group(1).upper())
            selected_text_exact = (
                option_text is not None
                and match.group(2).strip() == option_text
            )
    elif qtype == "true_false":
        match = TF_FIRST.match(stripped)
        raw_answer_first = match is not None
        strict_parse = raw_answer_first
        predicted = (
            match.group(1).lower() == "true"
            if match
            else extract_true_false(response)
        )
        expected = extract_true_false(str(row.get("answer", "")))
        if predicted is not None and expected is not None:
            heuristic_correct = predicted == expected
    elif qtype == "open_ended":
        forbidden_prefix = re.match(
            r"^\s*(?:[A-D][\).:]|True\.|False\.|<think>|Answer\s*:)",
            stripped,
            flags=re.I,
        )
        raw_answer_first = bool(stripped) and state == "none"
        strict_parse = raw_answer_first and forbidden_prefix is None

    return {
        "record_id": row.get("record_id"),
        "mode": row.get("mode"),
        "question_type": qtype,
        "response_chars": len(response),
        "response_words": len(re.findall(r"\S+", response)),
        "sentence_count": sentence_count(response),
        "think_state": state,
        "raw_answer_first": raw_answer_first,
        "strict_parse": strict_parse,
        "selected_option_text_exact": selected_text_exact,
        "heuristic_correct": heuristic_correct,
        "response_at_least_3000_chars": len(response) >= 3000,
        "unsupported_number_fraction": unsupported_number_fraction(row),
        "first_line_chars": len(first_line),
    }


def mean(items):
    values = [float(item) for item in items if item is not None]
    return sum(values) / len(values) if values else None


def summarize(items: list[dict]) -> dict:
    chars = [item["response_chars"] for item in items]
    return {
        "count": len(items),
        "response_chars_mean": mean(chars),
        "response_chars_median": statistics.median(chars) if chars else None,
        "response_words_mean": mean(item["response_words"] for item in items),
        "sentence_count_mean": mean(item["sentence_count"] for item in items),
        "raw_answer_first_rate": mean(
            item["raw_answer_first"] for item in items
        ),
        "strict_parse_rate": mean(item["strict_parse"] for item in items),
        "selected_option_text_exact_rate": mean(
            item["selected_option_text_exact"] for item in items
        ),
        "heuristic_correct_rate": mean(
            item["heuristic_correct"] for item in items
        ),
        "response_at_least_3000_chars_rate": mean(
            item["response_at_least_3000_chars"] for item in items
        ),
        "unsupported_number_fraction_mean": mean(
            item["unsupported_number_fraction"] for item in items
        ),
        "think_states": dict(
            sorted(Counter(item["think_state"] for item in items).items())
        ),
    }


def analyze(rows: list[dict]) -> tuple[dict, list[dict]]:
    details = [diagnose(row) for row in rows]
    by_mode = defaultdict(list)
    by_mode_type = defaultdict(list)
    for item in details:
        by_mode[item["mode"]].append(item)
        by_mode_type[(item["mode"], item["question_type"])].append(item)
    summary = {
        "rows": len(details),
        "modes": {
            mode: {
                "overall": summarize(items),
                "by_question_type": {
                    qtype: summarize(by_mode_type[(mode, qtype)])
                    for qtype in (
                        "multiple_choice",
                        "open_ended",
                        "true_false",
                    )
                    if by_mode_type[(mode, qtype)]
                },
            }
            for mode, items in sorted(by_mode.items())
        },
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    summary, details = analyze(read_jsonl(args.predictions))
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prefix.with_suffix(".jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False) + "\n"
            for item in details
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(details), "modes": len(summary["modes"])}))


if __name__ == "__main__":
    main()
