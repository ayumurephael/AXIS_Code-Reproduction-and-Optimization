"""Fail-closed audit for matched author-checkpoint post-training arms."""
from __future__ import annotations

import argparse
import json

import torch


AUTHOR_KEYS = {
    "fix_prompt_embeddings",
    "mapping_layer.weight",
    "mapping_layer.bias",
    "local_word_proj.weight",
    "local_word_proj.bias",
    "local_attention.q_proj.weight",
    "local_attention.k_proj.weight",
    "local_attention.v_proj.weight",
    "local_attention.out_proj.weight",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_author_posttrain_checkpoint(payload: dict, *, expected_arm: str) -> dict:
    _require(expected_arm in {"control", "treatment"}, "unknown post-training arm")
    metadata = payload.get("reproduction_meta")
    _require(isinstance(metadata, dict), "checkpoint has no reproduction metadata")
    _require(int(metadata.get("objective_version", -1)) == 6, "objective version mismatch")
    _require(metadata.get("stage") == "author_checkpoint_posttrain", "stage mismatch")
    _require(metadata.get("arm") == expected_arm, "arm mismatch")
    objective = (
        "author_posttrain_answer_control_v1"
        if expected_arm == "control" else "author_posttrain_question_type_joint_v1"
    )
    _require(metadata.get("objective") == objective, "objective mismatch")
    _require(metadata.get("initializer_kind") == "released_author_phase2_best", "initializer mismatch")
    _require(bool(metadata.get("init_phase2_sha256")), "initializer SHA is missing")
    _require(metadata.get("optimizer_reset") is True, "optimizer was not reset")
    _require(metadata.get("resume_optimizer") is False, "author optimizer was incorrectly resumed")
    _require(int(metadata.get("world_size", -1)) == 3, "formal run must use three GPUs")
    _require(int(metadata.get("epochs", -1)) == 1, "formal run must contain one auxiliary epoch")
    _require(int(metadata.get("steps_per_epoch", -1)) == 1600, "step budget mismatch")
    _require(int(metadata.get("seed", -1)) == 72, "seed mismatch")
    _require(metadata.get("fixed_hint_answer_gradient") is True, "answer must update Fixed Hint")
    _require(metadata.get("fixed_hint_auxiliary_gradient") is False, "auxiliary updates Fixed Hint")
    architecture = metadata.get("architecture", {})
    _require(architecture.get("variant") == "loss_only", "architecture is not loss_only")
    _require(architecture.get("qk_norm") is False, "QK-Norm is enabled")
    _require(architecture.get("continuous_bypass") is False, "residual bypass is enabled")
    _require(architecture.get("direct_task_prompt") is False, "direct task prompt is enabled")
    _require(int(architecture.get("fixed_query_tokens", -1)) == 30, "Fixed query count mismatch")

    if expected_arm == "control":
        _require(metadata.get("auxiliary_sampling") is None, "control has an auxiliary sampler")
        _require(metadata.get("beta_calibration") is None, "control calibrated auxiliary betas")
        _require(all(float(value) == 0.0 for value in metadata["calibrated_betas"].values()),
                 "control has nonzero auxiliary beta")
    else:
        sampling = metadata.get("auxiliary_sampling", {})
        _require(sampling.get("ratio_mc_tf_oe") == [4, 3, 1], "stratified ratio mismatch")
        _require(sampling.get("global_exposure_per_epoch") == {
            "mc": 2400, "tf": 1800, "oe": 600
        }, "global exposure mismatch")
        _require(sampling.get("valid_pool_counts") == {
            "mc": 2822, "tf": 1048, "oe": 198
        }, "valid pool count mismatch")
        _require(int(metadata.get("beta_calibration_steps", -1)) == 200,
                 "beta calibration budget mismatch")
        _require(set(metadata.get("calibrated_betas", {})) == {"mc", "tf", "oe"},
                 "calibrated beta keys mismatch")
        _require(int(metadata.get("counterfactual_index_version", -1)) == 3,
                 "counterfactual index version mismatch")
        _require(int(metadata.get("supervision_cache_version", -1)) == 1,
                 "supervision cache version mismatch")

    state = payload.get("model_state_dict", {}).get("moirai_trainable")
    _require(isinstance(state, dict), "checkpoint has no Perceiver state")
    prefix = "perceiver."
    keys = {
        key[len(prefix):] if key.startswith(prefix) else key
        for key in state
    }
    _require(keys == AUTHOR_KEYS, f"author Perceiver keys mismatch: {sorted(keys ^ AUTHOR_KEYS)}")
    return {
        "ok": True,
        "arm": expected_arm,
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "initializer_sha256": metadata.get("init_phase2_sha256"),
        "calibrated_betas": metadata.get("calibrated_betas"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(json.dumps(audit_author_posttrain_checkpoint(payload, expected_arm=args.arm), indent=2))


if __name__ == "__main__":
    main()
