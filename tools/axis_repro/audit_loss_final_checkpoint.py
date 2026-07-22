"""Fail-closed audit for answer-only and final-joint Phase-II checkpoints."""
from __future__ import annotations

import argparse
import json

import torch


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_loss_final_checkpoint(payload: dict, *, expected_stage: str) -> dict:
    metadata = payload.get("reproduction_meta")
    _require(isinstance(metadata, dict), "checkpoint has no reproduction metadata")
    _require(int(metadata.get("objective_version", -1)) == 5, "objective version is not 5")
    _require(metadata.get("stage") == expected_stage, "checkpoint stage mismatch")
    expected_objective = (
        "answer_only_recovery_v1"
        if expected_stage == "answer_only"
        else "question_type_routed_joint_v1"
    )
    _require(metadata.get("objective") == expected_objective, "objective name mismatch")
    _require(metadata.get("fixed_hint_answer_gradient") is True, "answer loss must update Fixed")
    _require(metadata.get("fixed_hint_auxiliary_gradient") is False, "auxiliary loss updates Fixed")
    _require(
        metadata.get("fixed_hint_auxiliary_isolation")
        == "processed_fixed_hint_output_stop_gradient",
        "Fixed isolation mechanism mismatch",
    )
    _require(metadata.get("component_backward") == "sequential_same_optimizer_step", "component schedule mismatch")
    _require(int(metadata.get("world_size", -1)) == 3, "formal run must use world size 3")
    _require(int(metadata.get("epochs", -1)) == 6, "formal stage must contain six epochs")
    _require(int(metadata.get("seed", -1)) == 72, "formal seed mismatch")
    architecture = metadata.get("architecture", {})
    _require(architecture.get("variant") == "loss_only", "architecture is not author loss_only")
    _require(architecture.get("qk_norm") is False, "loss_final enables QK-Norm")
    _require(architecture.get("continuous_bypass") is False, "loss_final enables residual bypass")
    _require(architecture.get("direct_task_prompt") is False, "loss_final enables direct task prompt")
    _require(architecture.get("qk_norm_seq_len") is None, "author architecture has a QK length")
    _require(int(architecture.get("fixed_query_tokens", -1)) == 30, "author fixed-query count is not 30")
    _require(architecture.get("task_prompt_tokens") is None, "direct task tokens are present")
    _require(
        architecture.get("fixed_hint_path")
        == "learned_queries_to_shared_prototype_cross_attention",
        "Fixed Hint does not use the author prototype-attention path",
    )

    if expected_stage == "answer_only":
        _require(metadata.get("init_phase2") is None, "answer-only unexpectedly warm-started")
        _require(metadata.get("resume_optimizer") is False, "answer-only resumed an optimizer")
        _require(metadata.get("counterfactual_index") is None, "answer-only materialized counterfactuals")
        for key in ("beta_mc_target", "beta_tf_target", "beta_oe_target", "beta_warmup_ratio"):
            _require(float(metadata.get(key, -1.0)) == 0.0, f"answer-only {key} is not zero")
    elif expected_stage == "joint":
        _require(bool(metadata.get("init_phase2")), "joint stage has no Phase-II initializer")
        _require(bool(metadata.get("init_phase2_sha256")), "joint parent SHA is missing")
        _require(metadata.get("resume_optimizer") is True, "joint stage did not resume optimizer moments")
        _require(metadata.get("auxiliary_warmup_restarted") is True, "joint warm-up was not restarted")
        _require(float(metadata.get("beta_mc_target", -1.0)) == 0.10, "MC beta mismatch")
        _require(float(metadata.get("beta_tf_target", -1.0)) == 0.05, "TF beta mismatch")
        _require(float(metadata.get("beta_oe_target", -1.0)) == 0.05, "OE beta mismatch")
        _require(float(metadata.get("beta_warmup_ratio", -1.0)) == 0.10, "warm-up ratio mismatch")
        _require(int(metadata.get("counterfactual_index_version", -1)) == 3, "counterfactual index mismatch")
        _require(int(metadata.get("supervision_cache_version", -1)) == 1, "supervision cache mismatch")
        _require(
            int(metadata.get("counterfactual_policy", {}).get("donor_top_m", -1)) == 1,
            "formal donor Top-M is not 1",
        )
        _require(
            metadata.get("ddp_auxiliary_denominator") == "global_valid_rows_per_component",
            "DDP auxiliary denominator mismatch",
        )
        _require(
            metadata.get("oe_qcar_target_tokens")
            == "answer_content_only_no_prefix_eos_padding",
            "OE target mask mismatch",
        )
        _require(
            metadata.get("oe_qcar_counterfactual_policy")
            == "abnormal_self_normal_patch_only",
            "OE counterfactual policy mismatch",
        )
        _require(metadata.get("question_type_routing") == {
            "multiple_choice": "answer+state_ce",
            "true_false": "answer+tf_q_ce",
            "open_ended": "answer+oe_qcar",
        }, "question-type routing mismatch")
    else:
        raise ValueError(f"unknown expected stage: {expected_stage}")

    state = payload.get("model_state_dict", {})
    _require("moirai_trainable" in state, "checkpoint has no Perceiver state")
    perceiver_state = state["moirai_trainable"]
    _require(isinstance(perceiver_state, dict), "Perceiver state is not a mapping")
    prefix = "perceiver."
    if perceiver_state and all(key.startswith(prefix) for key in perceiver_state):
        perceiver_keys = {key[len(prefix):] for key in perceiver_state}
    else:
        perceiver_keys = set(perceiver_state)
    author_keys = {
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
    _require(
        perceiver_keys == author_keys,
        f"Perceiver state is not author-compatible: missing={sorted(author_keys - perceiver_keys)}, "
        f"unexpected={sorted(perceiver_keys - author_keys)}",
    )
    return {
        "ok": True,
        "stage": expected_stage,
        "epoch": int(payload.get("epoch", -1)),
        "global_step": int(payload.get("global_step", -1)),
        "objective": expected_objective,
        "optimizer_state_present": "optimizer_state_dict" in payload,
        "phase1_sha256": metadata.get("phase1_sha256"),
        "init_phase2_sha256": metadata.get("init_phase2_sha256"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--stage", choices=("answer_only", "joint"), required=True)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(json.dumps(audit_loss_final_checkpoint(payload, expected_stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
