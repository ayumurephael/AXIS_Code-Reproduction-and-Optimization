import json

from .compare_geval_runs import compare_geval_runs


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compare_geval_runs_separates_changed_and_unchanged(tmp_path) -> None:
    left_predictions = tmp_path / "left_predictions.jsonl"
    right_predictions = tmp_path / "right_predictions.jsonl"
    left_scores = tmp_path / "left_scores.jsonl"
    right_scores = tmp_path / "right_scores.jsonl"
    _write(left_predictions, [
        {"record_id": "a", "mode": "base", "response": "same"},
        {"record_id": "b", "mode": "base", "response": "old"},
    ])
    _write(right_predictions, [
        {"record_id": "a", "mode": "base", "response": "same"},
        {"record_id": "b", "mode": "base", "response": "new"},
    ])
    _write(left_scores, [
        {"record_id": "a", "mode": "base", "dimension": "correctness", "score": 3, "prompt_sha256": "x"},
        {"record_id": "b", "mode": "base", "dimension": "correctness", "score": 2, "prompt_sha256": "y"},
    ])
    _write(right_scores, [
        {"record_id": "a", "mode": "base", "dimension": "correctness", "score": 4, "prompt_sha256": "x"},
        {"record_id": "b", "mode": "base", "dimension": "correctness", "score": 5, "prompt_sha256": "z"},
    ])

    report = compare_geval_runs(
        left_scores, right_scores, left_predictions, right_predictions,
        bootstrap_samples=100, seed=1,
    )

    assert report["overall"]["count"] == 2
    assert report["changed_response_tasks"]["count"] == 1
    assert report["unchanged_response_tasks"]["count"] == 1
    assert report["unchanged_prompt_hash_equal"] == 1
    assert report["changed_response_tasks"]["delta_right_minus_left"]["mean"] == 3.0