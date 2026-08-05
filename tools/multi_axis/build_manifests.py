from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from src.models.MultiAXIS.data import question_type_group, teacher_answer


EVAL_SPECS = {
    "478new": {
        "questions": ["question/question_eval_bias_neutralized_20260710b/*/questions_1000.jsonl"],
        "teachers": ["teacheranswer/teacher_eval_bias_neutralized_20260710b/*/teacher_gpt55.answers.jsonl"],
        "expected": 478,
    },
    "SMD": {
        "questions": ["question/question_smd_axis_v1_no_root_200_gpt54_bias_neutralized_20260713c/questions_200.jsonl"],
        "teachers": ["teacheranswer/teacher_smd_axis_v1_no_root_200_gpt54_bias_neutralized_20260713a/teacher_gpt54.answers.jsonl"],
        "expected": 200,
    },
    "SWaT": {
        "questions": ["question/question_swat_axis_v1_eval92_regular_184_gpt54_bias_neutralized_20260712c/questions_184.jsonl"],
        "teachers": ["teacheranswer/teacher_swat_axis_v1_eval92_regular_184_gpt54_20260712a/teacher_gpt54.answers.jsonl"],
        "expected": 184,
    },
    "LEMMA-RCA": {
        "questions": ["question/question_lemma_rca_cloud_curated_onset_v1_no_root_12_gpt54_bias_neutralized_openfirst_20260713b/questions_12.jsonl"],
        "teachers": ["teacheranswer/teacher_lemma_rca_cloud_curated_onset_v1_no_root_12_gpt54_bias_neutralized_openfirst_20260713a/teacher_gpt54.answers.jsonl"],
        "expected": 12,
    },
    "VTA": {
        "questions": ["question/question_vta_articulary_axis_v1_no_root_200_gpt54_llm2_bias_neutralized_openfirst_20260713b/questions_200.jsonl"],
        "teachers": ["teacheranswer/teacher_vta_articulary_axis_v1_no_root_200_gpt54_llm2_bias_neutralized_openfirst_20260713a/teacher_gpt54.answers.jsonl"],
        "expected": 200,
    },
}


def iter_jsonl(path: Path):
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            if not raw.strip():
                continue
            yield line_number, offset, len(raw), json.loads(raw)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets: List[int] = []
    with path.open("wb") as handle:
        for record in records:
            offsets.append(handle.tell())
            handle.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    path.with_suffix(path.suffix + ".idx.json").write_text(
        json.dumps(offsets, separators=(",", ":")), encoding="utf-8"
    )
    grouped = collections.defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["base_sample_id"]].append(index)
    path.with_suffix(path.suffix + ".groups.json").write_text(
        json.dumps(list(grouped.values()), separators=(",", ":")), encoding="utf-8"
    )



def teacher_rows(paths: Sequence[Path]):
    rows = []
    for path in paths:
        for line, offset, length, row in iter_jsonl(path):
            rows.append({"path": path, "line": line, "offset": offset, "length": length, "row": row})
    return rows


def match_teachers(question_rows, teachers):
    by_question = collections.defaultdict(collections.deque)
    for index, item in enumerate(teachers):
        by_question[str(item["row"].get("question", ""))].append(index)
    used = set()
    matches = []
    for position, question_item in enumerate(question_rows):
        question = str(question_item["row"].get("question", ""))
        chosen = None
        if position < len(teachers) and position not in used and str(teachers[position]["row"].get("question", "")) == question:
            chosen = position
        else:
            queue = by_question[question]
            while queue and queue[0] in used:
                queue.popleft()
            if queue:
                chosen = queue.popleft()
        if chosen is not None:
            used.add(chosen)
            matches.append((question_item, teachers[chosen]))
    return matches, used


def train_record(data_root: Path, question_item, teacher_item) -> Dict[str, Any]:
    row = question_item["row"]
    return {
        "question_path": question_item["path"].relative_to(data_root).as_posix(),
        "question_offset": question_item["offset"],
        "question_length": question_item["length"],
        "question_line": question_item["line"],
        "teacher_path": teacher_item["path"].relative_to(data_root).as_posix(),
        "teacher_line": teacher_item["line"],
        "sample_id": str(row["sample_id"]),
        "base_sample_id": str(row.get("base_sample_id") or row["sample_id"]),
        "question": str(row["question"]),
        "question_group": question_type_group(row),
        "interval": [int(row["target_interval"]["start"]), int(row["target_interval"]["end"])],
        "teacher_answer": teacher_answer(teacher_item["row"]),
    }


def build_train(data_root: Path, output_dir: Path, seed: int, validation_fraction: float, hash_inputs: bool):
    question_root = data_root / "question" / "question_train"
    teacher_root = data_root / "teacheranswer" / "teacher_train"
    question_files = sorted(question_root.glob("*/questions_1000.jsonl"))
    if not question_files:
        raise FileNotFoundError(f"No training questions beneath {question_root}")
    records = []
    empty_answers = 0
    unmatched_questions = 0
    unused_teachers = 0
    paired_before_empty = 0
    source_files = []
    for question_path in question_files:
        teacher_path = teacher_root / question_path.parent.name / "teacher_gpt55.answers.jsonl"
        if not teacher_path.exists():
            raise FileNotFoundError(f"Missing aligned teacher shard: {teacher_path}")
        questions = [
            {"path": question_path, "line": line, "offset": offset, "length": length, "row": row}
            for line, offset, length, row in iter_jsonl(question_path)
        ]
        teachers = teacher_rows([teacher_path])
        matches, used = match_teachers(questions, teachers)
        paired_before_empty += len(matches)
        unmatched_questions += len(questions) - len(matches)
        unused_teachers += len(teachers) - len(used)
        for question_item, teacher_item in matches:
            record = train_record(data_root, question_item, teacher_item)
            if not record["teacher_answer"]:
                empty_answers += 1
                continue
            records.append(record)
        for path in (question_path, teacher_path):
            source_files.append({"path": path.relative_to(data_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path) if hash_inputs else None})
    if paired_before_empty != 67820:
        raise RuntimeError(f"Expected 67820 aligned training rows, got {paired_before_empty}")
    if empty_answers != 17:
        raise RuntimeError(f"Expected exactly 17 empty teacher answers, got {empty_answers}")
    if unmatched_questions:
        raise RuntimeError(f"Found {unmatched_questions} questions without a teacher match")

    groups = sorted({record["base_sample_id"] for record in records})
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    validation_groups = set(shuffled[: max(1, round(len(shuffled) * validation_fraction))])
    train = [record for record in records if record["base_sample_id"] not in validation_groups]
    validation = [record for record in records if record["base_sample_id"] in validation_groups]
    train_groups = {record["base_sample_id"] for record in train}
    val_groups = {record["base_sample_id"] for record in validation}
    if train_groups & val_groups:
        raise AssertionError("base_sample_id leakage across train/validation split")
    write_manifest(train, output_dir / "train.jsonl")
    write_manifest(validation, output_dir / "validation.jsonl")
    summary = {
        "protocol": "grouped-base_sample_id-90-10",
        "seed": seed,
        "validation_fraction": validation_fraction,
        "question_shards": len(question_files),
        "paired_before_empty_filter": paired_before_empty,
        "empty_teacher_answers_filtered": empty_answers,
        "unused_teacher_rows": unused_teachers,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "train_groups": len(train_groups),
        "validation_groups": len(val_groups),
        "group_overlap": 0,
        "question_type_counts": dict(collections.Counter(record["question_group"] for record in records)),
        "source_files": source_files,
    }
    (output_dir / "train_split_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def expand_patterns(data_root: Path, patterns: Sequence[str]) -> List[Path]:
    paths = []
    for pattern in patterns:
        paths.extend(data_root.glob(pattern))
    return sorted(set(paths))


def build_eval(data_root: Path, output_dir: Path, hash_inputs: bool):
    all_summary = {}
    for dataset, spec in EVAL_SPECS.items():
        question_paths = expand_patterns(data_root, spec["questions"])
        teacher_paths = expand_patterns(data_root, spec["teachers"])
        questions = []
        for path in question_paths:
            questions.extend({"path": path, "line": line, "offset": offset, "length": length, "row": row} for line, offset, length, row in iter_jsonl(path))
        teachers = teacher_rows(teacher_paths)
        matches, used = match_teachers(questions, teachers)
        records = []
        for question_item, teacher_item in matches:
            answer = teacher_answer(teacher_item["row"])
            if not answer:
                continue
            row = question_item["row"]
            record = train_record(data_root, question_item, teacher_item)
            record["dataset"] = dataset
            record["label_reference"] = {
                "target_output": row.get("target_output"),
                "question_choices": row.get("question_choices"),
                "teacher_short_answer": teacher_item["row"].get("answer"),
                "question_answer_type": row.get("question_answer_type"),
            }
            records.append(record)
        if len(records) != spec["expected"]:
            raise RuntimeError(f"{dataset}: expected {spec['expected']} teacher-covered rows, got {len(records)}")
        write_manifest(records, output_dir / f"eval_{dataset}.jsonl")
        source_files = []
        for path in question_paths + teacher_paths:
            source_files.append({"path": path.relative_to(data_root).as_posix(), "size": path.stat().st_size, "sha256": sha256_file(path) if hash_inputs else None})
        summary = {
            "examples": len(records),
            "question_pool": len(questions),
            "teacher_rows": len(teachers),
            "matched_teacher_rows": len(used),
            "question_type_counts": dict(collections.Counter(record["question_group"] for record in records)),
            "source_files": source_files,
        }
        (output_dir / f"eval_{dataset}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        all_summary[dataset] = summary
    (output_dir / "eval_summary.json").write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--skip-input-hashes", action="store_true")
    parser.add_argument("--only", choices=["all", "train", "eval"], default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_inputs = not args.skip_input_hashes
    if args.only in ("all", "train"):
        print(json.dumps({"train": build_train(data_root, output_dir, args.seed, args.validation_fraction, hash_inputs)}, ensure_ascii=False))
    if args.only in ("all", "eval"):
        print(json.dumps({"eval": build_eval(data_root, output_dir, hash_inputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
