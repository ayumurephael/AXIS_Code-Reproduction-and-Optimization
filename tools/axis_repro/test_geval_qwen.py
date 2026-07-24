from __future__ import annotations

import json
from unittest import mock

import pytest

from .geval_qwen import (
    api_call,
    bounded_final_score_distribution,
    request_body,
)
from .geval_correct import final_score_distribution


def test_qwen_request_has_direct_logprobs_and_fixed_seed():
    body = request_body(
        "qwen3-30b-a3b-instruct-2507",
        "prompt",
        0.0,
        2048,
        True,
        5,
        72,
        None,
    )
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5
    assert body["seed"] == 72
    assert "enable_thinking" not in body


def test_qwen_thinking_extension_is_explicit_only():
    body = request_body(
        "qwen3-30b-a3b",
        "prompt",
        0.0,
        2048,
        True,
        5,
        72,
        False,
    )
    assert body["enable_thinking"] is False


@pytest.mark.parametrize("value", [-1, 6])
def test_qwen_rejects_out_of_range_top_logprobs(value):
    with pytest.raises(ValueError, match="between 0 and 5"):
        request_body(
            "qwen3-30b-a3b-instruct-2507",
            "prompt",
            0.0,
            2048,
            True,
            value,
            72,
            None,
        )


def test_qwen_api_uses_configured_endpoint_without_persisting_key():
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices": []}'

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data)
        seen["authorized"] = request.get_header("Authorization") is not None
        return FakeResponse()

    endpoint = "https://judge.example/compatible-mode/v1/chat/completions"
    with mock.patch(
        "tools.axis_repro.geval_qwen.urllib.request.urlopen",
        fake_urlopen,
    ):
        response = api_call(
            "secret",
            endpoint,
            "qwen3-30b-a3b-instruct-2507",
            "prompt",
            0.0,
            2048,
            True,
            5,
            72,
            None,
        )
    assert response == {"choices": []}
    assert seen["url"] == endpoint
    assert seen["timeout"] == 240
    assert seen["authorized"] is True
    assert seen["body"]["top_logprobs"] == 5


def test_qwen_final_score_uses_complete_score_token_distribution():
    alternatives = [
        {"token": str(score), "logprob": 0.0 if score == 4 else -2.0}
        for score in range(1, 6)
    ]
    response = {
        "choices": [{
            "message": {"content": "**Score:** 4"},
            "logprobs": {"content": [
                {"token": "**", "top_logprobs": []},
                {"token": "Score", "top_logprobs": []},
                {"token": ":**", "top_logprobs": []},
                {"token": " 4", "top_logprobs": alternatives},
            ]},
        }],
    }
    distribution = final_score_distribution(response)
    assert distribution is not None
    score, probabilities = distribution
    assert score > 3.5
    assert set(probabilities) == {"1", "2", "3", "4", "5"}


def qwen_probe_response(last_logprob=-24.5625):
    alternatives = [
        {"token": "4", "logprob": -0.000003},
        {"token": "3", "logprob": -13.25},
        {"token": "5", "logprob": -13.50},
        {"token": "�", "logprob": -21.375},
        {"token": "2", "logprob": last_logprob},
    ]
    return {
        "choices": [{
            "message": {"content": "**Score:** 4"},
            "logprobs": {"content": [
                {"token": "**", "top_logprobs": []},
                {"token": "Score", "top_logprobs": []},
                {"token": ":**", "top_logprobs": []},
                {"token": " ", "top_logprobs": []},
                {"token": "4", "top_logprobs": alternatives},
            ]},
        }],
    }


def test_qwen_bounded_distribution_accepts_negligible_missing_mass():
    result = bounded_final_score_distribution(
        qwen_probe_response(), 5, 1e-6
    )
    assert result is not None
    score, probabilities, metadata = result
    assert 3.99 < score <= 4.01
    assert set(probabilities) == set("12345")
    assert probabilities["1"] == 0.0
    assert metadata["method"] == "final_score_top_logprobs_bounded"
    assert metadata["missing_score_tokens"] == ["1"]
    assert metadata["missing_score_mass_upper_bound"] < 1e-6
    assert metadata["score_error_upper_bound"] < 4e-6


def test_qwen_bounded_distribution_rejects_material_missing_mass():
    response = qwen_probe_response()
    alternatives = response["choices"][0]["logprobs"]["content"][-1][
        "top_logprobs"
    ]
    for index, alternative in enumerate(alternatives, start=1):
        alternative["logprob"] = -0.1 * index
    result = bounded_final_score_distribution(
        response, 5, 1e-6
    )
    assert result is None


def test_qwen_bounded_distribution_rejects_short_topk():
    response = qwen_probe_response()
    response["choices"][0]["logprobs"]["content"][-1][
        "top_logprobs"
    ].pop()
    assert bounded_final_score_distribution(
        response, 5, 1e-6
    ) is None
