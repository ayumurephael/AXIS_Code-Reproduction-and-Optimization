from __future__ import annotations

import json
import sys

import pytest

from .audit_results import main


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def qwen_rows():
    distribution = {
        "1": 0.05,
        "2": 0.05,
        "3": 0.10,
        "4": 0.30,
        "5": 0.50,
    }
    common = {
        "record_id": "r1",
        "mode": "base",
        "question_type": "multiple_choice",
        "method": "final_score_top_logprobs",
        "model": "qwen3-30b-a3b-instruct-2507",
        "provider": "qwen",
        "enable_thinking": False,
        "logprobs_returned": True,
        "distribution": distribution,
        "fallback_scores": [],
        "prompt_sha256": "a" * 64,
        "missing_score_mass_upper_bound": 0.0,
    }
    return [
        {
            **common,
            "dimension": "correctness",
            "weight": 0.7,
            "score": 4.15,
        },
        {
            **common,
            "dimension": "reasoning_quality",
            "weight": 0.3,
            "score": 4.15,
            "prompt_sha256": "b" * 64,
        },
    ]


def test_qwen_fail_closed_audit(tmp_path, monkeypatch, capsys):
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.jsonl"
    write_jsonl(predictions, [{
        "record_id": "r1",
        "mode": "base",
        "question_type": "multiple_choice",
        "response": "answer",
    }])
    write_jsonl(scores, qwen_rows())
    monkeypatch.setattr(sys, "argv", [
        "audit_results",
        "--predictions", str(predictions),
        "--scores", str(scores),
        "--modes", "base",
        "--expected-model", "qwen3-30b-a3b-instruct-2507",
        "--expected-provider", "qwen",
        "--expected-enable-thinking", "false",
        "--allowed-methods", "final_score_top_logprobs",
        "--require-logprobs",
        "--require-prompt-hashes",
    ])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["logprobs_returned"] == 2
    assert result["score_enable_thinking"] == {"false": 2}


def test_qwen_audit_rejects_missing_distribution(
    tmp_path, monkeypatch, capsys,
):
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.jsonl"
    write_jsonl(predictions, [{
        "record_id": "r1",
        "mode": "base",
        "question_type": "multiple_choice",
        "response": "answer",
    }])
    rows = qwen_rows()
    rows[0]["distribution"] = None
    rows[0]["logprobs_returned"] = False
    write_jsonl(scores, rows)
    monkeypatch.setattr(sys, "argv", [
        "audit_results",
        "--predictions", str(predictions),
        "--scores", str(scores),
        "--modes", "base",
        "--require-logprobs",
    ])
    with pytest.raises(SystemExit):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert any("incomplete logprobs" in error for error in result["errors"])

def test_audit_rejects_non_twenty_sample_fallback(
    tmp_path, monkeypatch, capsys,
):
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.jsonl"
    write_jsonl(predictions, [{
        "record_id": "r1",
        "mode": "base",
        "question_type": "multiple_choice",
        "response": "answer",
    }])
    rows = qwen_rows()
    for row in rows:
        row["method"] = "exact_sample_mean_5"
        row["score"] = 4.0
        row["distribution"] = None
        row["logprobs_returned"] = False
        row["fallback_scores"] = [4, 4, 4, 4, 4]
    write_jsonl(scores, rows)
    monkeypatch.setattr(sys, "argv", [
        "audit_results",
        "--predictions", str(predictions),
        "--scores", str(scores),
        "--modes", "base",
    ])
    with pytest.raises(SystemExit):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert any(
        "fallback must contain exactly 20 samples" in error
        for error in result["errors"]
    )


def test_audit_rejects_excessive_missing_score_mass(
    tmp_path, monkeypatch, capsys,
):
    predictions = tmp_path / "predictions.jsonl"
    scores = tmp_path / "scores.jsonl"
    write_jsonl(predictions, [{
        "record_id": "r1",
        "mode": "base",
        "question_type": "multiple_choice",
        "response": "answer",
    }])
    rows = qwen_rows()
    for row in rows:
        row["method"] = "final_score_top_logprobs_bounded"
        row["missing_score_mass_upper_bound"] = 1e-3
    write_jsonl(scores, rows)
    monkeypatch.setattr(sys, "argv", [
        "audit_results",
        "--predictions", str(predictions),
        "--scores", str(scores),
        "--modes", "base",
        "--require-logprobs",
        "--max-missing-score-mass-upper-bound", "1e-6",
    ])
    with pytest.raises(SystemExit):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert any(
        "score-mass bound exceeds limit" in error
        for error in result["errors"]
    )
