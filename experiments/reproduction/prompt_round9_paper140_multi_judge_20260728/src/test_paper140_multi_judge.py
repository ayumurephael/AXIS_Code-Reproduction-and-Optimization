from __future__ import annotations

import json
from pathlib import Path

from experiments.reproduction.prompt_round9_paper140_multi_judge_20260728.src.paper140_multi_judge import (
    MODES,
    expand_scores,
    prepare,
)


class Args:
    pass


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prepare_and_expand(monkeypatch, tmp_path: Path) -> None:
    record_ids = [f"series_{index:06d}:0" for index in range(140)]
    manifest = tmp_path / "paper140.json"
    manifest.write_text(
        json.dumps({"record_count": 140, "record_ids": record_ids}),
        encoding="utf-8",
    )
    source_rows = []
    for record_id in record_ids:
        for mode in MODES:
            response = "base"
            if mode != "base" and record_id.endswith("000000:0"):
                response = "changed"
            source_rows.append(
                {
                    "record_id": record_id,
                    "mode": mode,
                    "question_type": "multiple_choice",
                    "question": "q",
                    "answer": "a",
                    "response": response,
                }
            )
    source = tmp_path / "source.jsonl"
    write_jsonl(source, source_rows)
    all_predictions = tmp_path / "all.jsonl"
    unique_predictions = tmp_path / "unique.jsonl"
    mapping = tmp_path / "mapping.json"

    args = Args()
    args.source = str(source)
    args.manifest = str(manifest)
    args.output_all = str(all_predictions)
    args.output_unique = str(unique_predictions)
    args.output_mapping = str(mapping)
    monkeypatch.setattr(
        "experiments.reproduction.prompt_round9_paper140_multi_judge_20260728.src.paper140_multi_judge.DIMS",
        {"multiple_choice": ("correctness", "reasoning")},
    )
    prepare(args)

    all_rows = [
        json.loads(line)
        for line in all_predictions.read_text(encoding="utf-8").splitlines()
    ]
    unique_rows = [
        json.loads(line)
        for line in unique_predictions.read_text(encoding="utf-8").splitlines()
    ]
    assert len(all_rows) == 560
    assert len(unique_rows) == 141

    raw_scores = []
    for row in unique_rows:
        for dimension in ("correctness", "reasoning"):
            raw_scores.append(
                {
                    "record_id": row["record_id"],
                    "mode": row["mode"],
                    "dimension": dimension,
                    "score": 4.0,
                }
            )
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, raw_scores)
    expanded = tmp_path / "expanded.jsonl"
    expand_args = Args()
    expand_args.predictions = str(all_predictions)
    expand_args.mapping = str(mapping)
    expand_args.raw_scores = str(raw)
    expand_args.output = str(expanded)
    monkeypatch.setattr(
        "experiments.reproduction.prompt_round9_paper140_multi_judge_20260728.src.paper140_multi_judge.DIMS",
        {"multiple_choice": ("correctness", "reasoning")},
    )
    monkeypatch.setattr(
        "experiments.reproduction.prompt_round9_paper140_multi_judge_20260728.src.paper140_multi_judge.read_jsonl",
        lambda path: [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
        ],
    )
    # The production paper140 total is fixed at 1,340; this synthetic fixture
    # exercises mapping and reuse up to that final production-only count guard.
    try:
        expand_scores(expand_args)
    except RuntimeError as exc:
        assert "Expected 1,340 expanded scores" in str(exc)
    else:
        raise AssertionError("Synthetic fixture should stop at production count guard")

