from .common import score_prediction


def test_score_prediction_accepts_inference_response_field() -> None:
    row = {
        "record_id": "series_000001:0",
        "question_type": "multiple_choice",
        "answer": "B",
        "response": "The correct option is B.",
    }

    score = score_prediction(row)

    assert score["gold_choice"] == "B"
    assert score["pred_choice"] == "B"
    assert score["choice_correct"] == 1.0
