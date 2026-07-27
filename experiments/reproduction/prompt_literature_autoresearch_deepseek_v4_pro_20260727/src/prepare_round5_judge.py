"""Build deterministic JSONL inputs for the Round-5 component Judge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.axis_repro.common import read_jsonl


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def row_key(row: dict) -> tuple:
    key = (row["record_id"], row["mode"])
    if "dimension" in row:
        key += (row["dimension"],)
    return key


def merge(args: argparse.Namespace) -> None:
    indexed: dict[tuple, dict] = {}
    for input_path in args.inputs:
        for row in read_jsonl(input_path):
            key = row_key(row)
            if key in indexed and indexed[key] != row:
                raise RuntimeError(f"Conflicting duplicate row: {key}")
            indexed[key] = row
    rows = sorted(indexed.values(), key=row_key)
    write_jsonl(args.output, rows)
    print(json.dumps({"rows": len(rows), "output": args.output}))


def select(args: argparse.Namespace) -> None:
    excluded = set(args.exclude_modes or [])
    included = set(args.include_modes or [])
    rows = []
    for row in read_jsonl(args.predictions):
        mode = row["mode"]
        if mode in excluded:
            continue
        if included and mode not in included:
            continue
        rows.append(row)
    rows.sort(key=row_key)
    if len({row_key(row) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate prediction key in Judge input")
    write_jsonl(args.output, rows)
    print(json.dumps({
        "rows": len(rows),
        "modes": sorted({row["mode"] for row in rows}),
        "output": args.output,
    }))


def align_scores(args: argparse.Namespace) -> None:
    prediction_keys = {
        (row["record_id"], row["mode"])
        for row in read_jsonl(args.predictions)
    }
    indexed: dict[tuple, dict] = {}
    for input_path in args.scores:
        for row in read_jsonl(input_path):
            key = row_key(row)
            if key[:2] not in prediction_keys:
                continue
            if key in indexed and indexed[key] != row:
                raise RuntimeError(f"Conflicting duplicate score: {key}")
            indexed[key] = row
    rows = sorted(indexed.values(), key=row_key)
    covered = {(row["record_id"], row["mode"]) for row in rows}
    missing = prediction_keys - covered
    if missing:
        raise RuntimeError(f"Predictions without scores: {sorted(missing)}")
    write_jsonl(args.output, rows)
    print(json.dumps({
        "prediction_keys": len(prediction_keys),
        "score_rows": len(rows),
        "output": args.output,
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", nargs="+", required=True)
    merge_parser.add_argument("--output", required=True)
    merge_parser.set_defaults(func=merge)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--predictions", required=True)
    select_parser.add_argument("--output", required=True)
    select_parser.add_argument("--exclude-modes", nargs="*")
    select_parser.add_argument("--include-modes", nargs="*")
    select_parser.set_defaults(func=select)

    align_parser = subparsers.add_parser("align-scores")
    align_parser.add_argument("--predictions", required=True)
    align_parser.add_argument("--scores", nargs="+", required=True)
    align_parser.add_argument("--output", required=True)
    align_parser.set_defaults(func=align_scores)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
