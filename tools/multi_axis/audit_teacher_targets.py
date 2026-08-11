from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, Sequence

from src.models.MultiAXIS.data import structured_is_anomalous
from src.models.MultiAXIS.response_contracts import (
    canonicalize_teacher_answer,
    response_contract_error,
)


SPLITS = ("train", "validation")
QUESTION_GROUPS = ("MC", "TF", "OE")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                record = json.loads(line)
                record["_manifest_line"] = line_number
                yield record


class QuestionReader:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.handles: Dict[str, BinaryIO] = {}

    def read(self, record: Dict[str, Any]) -> Dict[str, Any]:
        relative = str(record["question_path"])
        handle = self.handles.get(relative)
        if handle is None:
            handle = (self.data_root / relative).open("rb")
            self.handles[relative] = handle
        handle.seek(int(record["question_offset"]))
        raw = handle.read(int(record["question_length"]))
        row = json.loads(raw)
        if str(row.get("question")) != str(record.get("question")):
            raise RuntimeError("manifest/question text mismatch")
        return row

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __enter__(self) -> "QuestionReader":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def audit_split(
    manifest_path: Path,
    data_root: Path,
    *,
    max_recorded_failures: int = 100,
) -> Dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    canonicalized: collections.Counter[str] = collections.Counter()
    failures = []
    total_failures = 0
    with QuestionReader(data_root) as questions:
        for record in read_jsonl(manifest_path):
            group = str(record.get("question_group"))
            counts[group] += 1
            try:
                if group not in QUESTION_GROUPS:
                    raise ValueError(f"unsupported question group {group!r}")
                answer = record.get("teacher_answer")
                row = questions.read(record)
                normalized = canonicalize_teacher_answer(
                    answer,
                    group,
                    structured_is_anomalous=structured_is_anomalous(row),
                )
                error = response_contract_error(normalized, group)
                if error is not None:
                    raise AssertionError(
                        f"normalized target violates contract: {error}"
                    )
                if normalized != str(answer).strip().replace("\r\n", "\n"):
                    canonicalized[group] += 1
            except Exception as error:  # Record every failure count before fail-closed exit.
                total_failures += 1
                if len(failures) < max_recorded_failures:
                    failures.append(
                        {
                            "manifest_line": int(record["_manifest_line"]),
                            "sample_id": record.get("sample_id"),
                            "question_group": group,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "examples": sum(counts.values()),
        "question_type_counts": dict(counts),
        "canonicalized_teacher_targets": dict(canonicalized),
        "canonicalized_total": sum(canonicalized.values()),
        "failures": total_failures,
        "recorded_failures": failures,
        "passed": total_failures == 0,
    }


def audit(
    manifest_dir: Path,
    data_root: Path,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, Any]:
    results = {
        split: audit_split(manifest_dir / f"{split}.jsonl", data_root)
        for split in splits
    }
    return {
        "protocol": "canonicalize-then-contract-check-v1",
        "splits": results,
        "examples": sum(item["examples"] for item in results.values()),
        "failures": sum(item["failures"] for item in results.values()),
        "passed": all(item["passed"] for item in results.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit every formal train/validation teacher target."
    )
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.manifest_dir, args.data_root, tuple(args.splits))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
