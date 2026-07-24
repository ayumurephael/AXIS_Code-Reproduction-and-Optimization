from __future__ import annotations

from .geval_qwen_readout import (
    bounded_label_distribution,
    request_body,
)


def response_with_missing_a(cutoff=-26.25):
    return {
        "model": "qwen3-30b-a3b-instruct-2507",
        "choices": [{
            "message": {"content": '{"score":"D"}'},
            "logprobs": {"content": [
                {"token": '{"', "top_logprobs": []},
                {"token": "score", "top_logprobs": []},
                {"token": '":"', "top_logprobs": []},
                {
                    "token": "D",
                    "top_logprobs": [
                        {"token": "D", "logprob": 0.0},
                        {"token": "C", "logprob": -21.5},
                        {"token": "E", "logprob": -22.75},
                        {"token": "B", "logprob": -23.0},
                        {"token": "4", "logprob": cutoff},
                    ],
                },
                {"token": '"}', "top_logprobs": []},
            ]},
        }],
    }


def test_readout_request_is_json_and_direct_logprobs():
    body = request_body(
        "qwen3-30b-a3b-instruct-2507",
        "author prompt",
        "**Score:** 4",
        20,
        5,
        72,
    )
    assert body["response_format"] == {"type": "json_object"}
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5
    assert body["seed"] == 72
    assert body["messages"][1]["role"] == "assistant"


def test_readout_bounded_distribution_maps_labels_to_scores():
    result = bounded_label_distribution(
        response_with_missing_a(), 5, 1e-6
    )
    assert result is not None
    score, probabilities, metadata = result
    assert 3.99 < score <= 4.01
    assert set(probabilities) == set("12345")
    assert probabilities["1"] == 0.0
    assert metadata["chosen_label"] == "D"
    assert metadata["missing_score_tokens"] == ["A"]
    assert (
        metadata["method"]
        == "final_score_top_logprobs_readout_bounded"
    )
    assert metadata["missing_score_mass_upper_bound"] < 1e-6


def test_readout_rejects_material_missing_mass():
    response = response_with_missing_a()
    alternatives = response["choices"][0]["logprobs"]["content"][3][
        "top_logprobs"
    ]
    for index, alternative in enumerate(alternatives, start=1):
        alternative["logprob"] = -0.1 * index
    assert bounded_label_distribution(
        response, 5, 1e-6
    ) is None


def test_readout_rejects_non_json_or_short_topk():
    response = response_with_missing_a()
    response["choices"][0]["message"]["content"] = "D"
    assert bounded_label_distribution(
        response, 5, 1e-6
    ) is None
    response = response_with_missing_a()
    response["choices"][0]["logprobs"]["content"][3][
        "top_logprobs"
    ].pop()
    assert bounded_label_distribution(
        response, 5, 1e-6
    ) is None
