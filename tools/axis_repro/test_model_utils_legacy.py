from types import SimpleNamespace

import torch

from .model_utils import load_axis_payload


def test_load_axis_payload_strips_author_perceiver_prefix():
    ts_model = torch.nn.Linear(2, 2)
    perceiver = torch.nn.Linear(3, 2)
    model = SimpleNamespace(
        ts_pretrain_model=ts_model,
        axis=SimpleNamespace(
            perceiver=perceiver,
            config=SimpleNamespace(
                llm_config=SimpleNamespace(architecture_variant="loss_only")
            ),
        ),
    )
    ts_state = {
        key: torch.full_like(value, 2.0)
        for key, value in ts_model.state_dict().items()
    }
    legacy_state = {
        f"perceiver.{key}": torch.full_like(value, 3.0)
        for key, value in perceiver.state_dict().items()
    }
    payload = {"model_state_dict": {
        "ts_pretrain_model": ts_state,
        "moirai_trainable": legacy_state,
    }}
    load_axis_payload(model, payload)
    assert all(torch.all(value == 2) for value in ts_model.state_dict().values())
    assert all(torch.all(value == 3) for value in perceiver.state_dict().values())
