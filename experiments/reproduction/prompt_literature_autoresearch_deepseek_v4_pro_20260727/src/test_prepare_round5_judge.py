import argparse
import json

from experiments.reproduction.prompt_literature_autoresearch_deepseek_v4_pro_20260727.src import (
    prepare_round5_judge as target,
)


def dump(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def load(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_merge_deduplicates_identical_rows(tmp_path):
    row = {"record_id": "s:0", "mode": "m", "response": "x"}
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "merged.jsonl"
    dump(first, [row])
    dump(second, [row])

    target.merge(
        argparse.Namespace(inputs=[str(first), str(second)], output=str(output))
    )

    assert load(output) == [row]


def test_align_scores_keeps_only_prediction_keys(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.jsonl"
    output = tmp_path / "aligned.jsonl"
    dump(
        predictions,
        [{"record_id": "s:0", "mode": "wanted", "response": "x"}],
    )
    dump(
        scores,
        [
            {
                "record_id": "s:0",
                "mode": "wanted",
                "dimension": "correctness",
                "score": 5,
            },
            {
                "record_id": "s:0",
                "mode": "other",
                "dimension": "correctness",
                "score": 1,
            },
        ],
    )

    target.align_scores(
        argparse.Namespace(
            predictions=str(predictions),
            scores=[str(scores)],
            output=str(output),
        )
    )

    assert [(row["mode"], row["score"]) for row in load(output)] == [
        ("wanted", 5)
    ]
