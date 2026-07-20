import json

import pytest

from .compare_inference_protocols import compare_predictions


def _write(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_compare_predictions_reports_protocol_changes(tmp_path) -> None:
    common = {
        "mode": "base",
        "question_type": "true_false",
        "answer": "True",
    }
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [dict(common, record_id="a", response="True"), dict(common, record_id="b", response="False")])
    _write(right, [dict(common, record_id="a", response="True"), dict(common, record_id="b", response="True")])

    report = compare_predictions(left, right)

    assert report["overall"]["count"] == 2
    assert report["overall"]["changed_response_count"] == 1
    assert report["overall"]["left_mean_proxy_score"] == 0.5
    assert report["overall"]["right_mean_proxy_score"] == 1.0


def test_compare_predictions_rejects_key_mismatch(tmp_path) -> None:
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    _write(left, [{"record_id": "a", "mode": "base", "question_type": "open_ended", "answer": "x", "response": "x"}])
    _write(right, [{"record_id": "b", "mode": "base", "question_type": "open_ended", "answer": "x", "response": "x"}])

    with pytest.raises(ValueError, match="prediction key mismatch"):
        compare_predictions(left, right)