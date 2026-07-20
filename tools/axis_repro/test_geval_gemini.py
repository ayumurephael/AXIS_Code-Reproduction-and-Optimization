from __future__ import annotations

import math

from .geval_gemini import author_distribution, author_prompt_for


def test_author_integer_fallback_without_logprobs():
    response = {"choices": [{"message": {"content": "**Score:** 4"}}]}
    score, distribution, method = author_distribution(response)
    assert score == 4.0
    assert distribution is None
    assert method == "author_temperature0_integer"


def test_author_default_is_explicit():
    response = {"choices": [{"message": {"content": "No final score"}}]}
    score, distribution, method = author_distribution(response)
    assert score == 3.0
    assert distribution is None
    assert method == "author_default_3"


def test_author_cross_position_weighting_matches_supplied_code():
    response = {
        "choices": [{
            "message": {"content": "**Score:** 4"},
            "logprobs": {"content": [
                {"token": "Score", "logprob": -0.1, "top_logprobs": []},
                {
                    "token": "4", "logprob": -0.2,
                    "top_logprobs": [{"token": "3", "logprob": -1.2}],
                },
            ]},
        }],
    }
    score, distribution, method = author_distribution(response)
    expected_p4 = math.exp(0.3) / (math.exp(0.3) + math.exp(-1.2))
    assert method == "author_cross_position_logprobs"
    assert distribution is not None
    assert abs(distribution["4"] - expected_p4) < 1e-12
    assert abs(score - (4 * expected_p4 + 3 * (1 - expected_p4))) < 1e-12


def test_author_prompt_has_exact_score_prefix_and_layout():
    prompt = author_prompt_for(
        {"question": "Q", "answer": "A", "response": "R"},
        "correctness", "D", ["5: excellent", "1: poor"],
    )
    assert "Score 5: excellent" in prompt
    assert "**Evaluation Criterion: correctness**" in prompt
    assert "Control the Maximum Length to 500 words." in prompt
