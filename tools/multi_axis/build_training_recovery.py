from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple


def teacher_model_answer(row: Dict[str, Any]) -> str:
    value = row.get("model_answer")
    return value.strip() if isinstance(value, str) else ""


def normalized_question(row: Dict[str, Any]) -> str:
    value = row.get("question")
    if value is None:
        windows = row.get("windows") or []
        if windows and isinstance(windows[0], dict):
            value = windows[0].get("question")
    return "" if value is None else str(value).strip().lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with path.open("rb") as handle:
        index = 0
        for raw in handle:
            if not raw.strip():
                continue
            yield index, json.loads(raw)
            index += 1


def build_recovery(source_root: Path, output_dir: Path) -> Dict[str, Any]:
    question_root = source_root / "question" / "question_train"
    teacher_root = source_root / "teacheranswer" / "teacher_train"
    summary_path = teacher_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = summary.get("entries") or []
    if len(entries) != 62:
        raise RuntimeError(f"Expected 62 teacher shards, got {len(entries)}")

    recovered = []
    raw_sources = {}
    shard_records = []
    total = direct = empty = 0
    for entry in entries:
        shard = str(entry["name"])
        expected = int(entry["line_count"])
        candidates = sorted((question_root / shard).glob("questions_*.jsonl"))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one question shard for {shard}, got {len(candidates)}")
        question_path = candidates[0]
        teacher_path = teacher_root / shard / "teacher_gpt55.answers.jsonl"
        by_question = collections.defaultdict(collections.deque)
        question_count = 0
        for question_index, row in iter_jsonl(question_path):
            by_question[normalized_question(row)].append(question_index)
            question_count += 1

        missing = []
        teacher_count = 0
        for teacher_index, row in iter_jsonl(teacher_path):
            if teacher_index >= expected:
                raise RuntimeError(f"Teacher file exceeds summary count for {shard}")
            key = normalized_question(row)
            queue = by_question[key]
            if key and queue:
                queue.popleft()
                direct += 1
            else:
                missing.append((teacher_index, key))
            empty += int(not teacher_model_answer(row))
            teacher_count += 1
            total += 1
        if teacher_count != expected:
            raise RuntimeError(
                f"Teacher count mismatch for {shard}: {teacher_count} != {expected}"
            )

        if missing:
            raw_path = teacher_root / shard / "teacher_gpt55.jsonl"
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            wanted = dict(missing)
            found = set()
            for raw_index, row in iter_jsonl(raw_path):
                if raw_index not in wanted:
                    continue
                if normalized_question(row) != wanted[raw_index]:
                    raise RuntimeError(
                        f"Raw recovery question mismatch for {shard} index {raw_index}"
                    )
                recovered_row = dict(row)
                recovered_row["_multi_axis_recovery"] = {
                    "source_shard": shard,
                    "source_index": raw_index,
                    "source_file": (
                        f"teacheranswer/teacher_train/{shard}/teacher_gpt55.jsonl"
                    ),
                }
                recovered.append(recovered_row)
                found.add(raw_index)
            if found != set(wanted):
                raise RuntimeError(
                    f"Raw recovery rows missing for {shard}: {sorted(set(wanted) - found)}"
                )
            raw_sources[shard] = {
                "path": f"teacheranswer/teacher_train/{shard}/teacher_gpt55.jsonl",
                "size": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
                "recovered_indices": sorted(wanted),
            }
        shard_records.append(
            {
                "name": shard,
                "expected": expected,
                "question_file": question_path.name,
                "question_rows": question_count,
                "teacher_rows": teacher_count,
                "direct_matches": expected - len(missing),
                "recovered_matches": len(missing),
            }
        )

    if (total, direct, len(recovered), empty) != (67820, 67773, 47, 17):
        raise RuntimeError(
            "Recovery audit mismatch: "
            f"total={total} direct={direct} recovered={len(recovered)} empty={empty}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "recovered_questions.jsonl"
    temporary = questions_path.with_suffix(questions_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in recovered:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, questions_path)
    manifest = {
        "format": "multi-axis-training-recovery-v1",
        "policy": (
            "question-text one-to-one matching within authoritative teacher summary shard; "
            "unmatched teacher rows recovered at the same source index from teacher_gpt55.jsonl"
        ),
        "teacher_summary": {
            "path": "teacheranswer/teacher_train/summary.json",
            "size": summary_path.stat().st_size,
            "sha256": sha256_file(summary_path),
        },
        "total_teacher_rows": total,
        "direct_question_matches": direct,
        "recovered_question_rows": len(recovered),
        "empty_teacher_answers": empty,
        "recovered_output": {
            "path": "derived/training_recovery/recovered_questions.jsonl",
            "size": questions_path.stat().st_size,
            "sha256": sha256_file(questions_path),
        },
        "raw_sources": raw_sources,
        "shards": shard_records,
    }
    manifest_path = output_dir / "recovery_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the audited 47-row training-question recovery bundle"
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = build_recovery(Path(args.source_root).resolve(), Path(args.output_dir).resolve())
    print(
        json.dumps(
            {
                "status": "RECOVERY_BUILD_OK",
                "total": manifest["total_teacher_rows"],
                "direct": manifest["direct_question_matches"],
                "recovered": manifest["recovered_question_rows"],
                "empty": manifest["empty_teacher_answers"],
            }
        )
    )


if __name__ == "__main__":
    main()