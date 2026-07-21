"""Fail-closed audit for the coherent binary-state Phase-II objective."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPECTED_OBJECTIVE = "answer_nll_plus_coherent_binary_state_ce_v3"
EXPECTED_COUNTERFACTUAL_INDEX_VERSION = 3
EXPECTED_GROUP_LRS = {
    "local_continuous": 5e-5,
    "prototype_attention": 1e-4,
    "task_prompt": 5e-5,
}


def audit_loss_objective_checkpoint(
    payload: dict,
    *,
    expected_donor_top_m: int | None = None,
) -> dict:
    metadata = payload.get("reproduction_meta")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint is missing reproduction metadata")
    if metadata.get("objective") != EXPECTED_OBJECTIVE or int(metadata.get("objective_version", 0)) != 3:
        raise ValueError("checkpoint does not use coherent binary state objective v3")
    if int(metadata.get("counterfactual_index_version", 0)) != EXPECTED_COUNTERFACTUAL_INDEX_VERSION:
        raise ValueError("checkpoint was not trained with coherent counterfactual index v3")
    if abs(float(metadata.get("beta_target", -1.0)) - 0.1) > 1e-12:
        raise ValueError("checkpoint beta target is not 0.1")
    if abs(float(metadata.get("beta_warmup_ratio", -1.0)) - 0.1) > 1e-12:
        raise ValueError("checkpoint beta warmup ratio is not 0.1")
    if abs(float(metadata.get("gradient_clip", -1.0)) - 1.0) > 1e-12:
        raise ValueError("checkpoint gradient clipping is not 1.0")
    if metadata.get("fixed_hint_state_gradient") is not False:
        raise ValueError("state loss must not update Fixed/task-prompt parameters")
    if metadata.get("fixed_hint_answer_gradient") is not True:
        raise ValueError("answer loss must train Fixed/task-prompt parameters")
    if metadata.get("fixed_hint_freeze_scope") != "state_loss_only":
        raise ValueError("checkpoint does not declare state-only Fixed/task-prompt freezing")
    if metadata.get("fixed_hint_state_isolation") != "direct_task_prompt_output_stop_gradient":
        raise ValueError("checkpoint does not use the audited direct-prompt stop-gradient path")
    if metadata.get("state_prompt_independent") is not True:
        raise ValueError("state classification must use an independent prompt sequence")
    if metadata.get("state_pairing") != "factual_counterfactual_same_step":
        raise ValueError("factual and counterfactual states were not paired in one step")
    if metadata.get("state_logits_source") != "frozen_lm_head_next_token_two_class":
        raise ValueError("state logits were not selected from the frozen LM head")
    if metadata.get("state_label_policy") != "strict_single_token_numeric_0_1":
        raise ValueError("state labels do not follow the strict numeric 0/1 policy")
    if metadata.get("state_verbalizer_selection") != "prefer_space_prefixed_then_bare_numeric":
        raise ValueError("state verbalizer selection does not follow the tokenizer-adaptive numeric policy")
    if metadata.get("soft_embedding_injection") != "non_inplace_index_copy":
        raise ValueError("soft-token injection is not the documented non-inplace path")
    if "margin" in metadata or "modes" in metadata:
        raise ValueError("checkpoint contains obsolete SLR metadata")

    verbalizers = metadata.get("state_verbalizers", {})
    normal_id = verbalizers.get("normal_id")
    anomalous_id = verbalizers.get("anomalous_id")
    if not isinstance(normal_id, int) or not isinstance(anomalous_id, int) or normal_id == anomalous_id:
        raise ValueError("checkpoint has invalid one-token state verbalizers")
    verbalizer_texts = (
        verbalizers.get("normal_text"),
        verbalizers.get("anomalous_text"),
    )
    if verbalizer_texts not in {(" 0", " 1"), ("0", "1")}:
        raise ValueError("checkpoint does not use numeric 0/1 state verbalizers")
    if not str(verbalizers.get("question", "")).rstrip().endswith("Label:"):
        raise ValueError("checkpoint state prompt does not end at the classification position")

    optimizer_groups = metadata.get("optimizer_groups", {})
    for name, expected_lr in EXPECTED_GROUP_LRS.items():
        group = optimizer_groups.get(name)
        if not group or int(group.get("parameters", 0)) <= 0:
            raise ValueError(f"checkpoint is missing optimizer group {name}")
        if abs(float(group.get("lr", -1.0)) - expected_lr) > 1e-12:
            raise ValueError(f"checkpoint optimizer LR mismatch for {name}")
    policy = metadata.get("counterfactual_policy", {})
    if policy.get("phase_fallback") is not False:
        raise ValueError("checkpoint counterfactual policy used a phase fallback")
    if policy.get("post_patch_dual_source_validation") is not True:
        raise ValueError("checkpoint counterfactual policy skipped dual-source validation")
    if policy.get("donor_selection") != "uniform_top_m_by_window_distance":
        raise ValueError("checkpoint did not use uniform Top-M donor sampling")
    donor_top_m = int(policy.get("donor_top_m", 0))
    if donor_top_m <= 0:
        raise ValueError("checkpoint donor Top-M must be positive")
    if expected_donor_top_m is not None and donor_top_m != expected_donor_top_m:
        raise ValueError("checkpoint donor Top-M does not match the preregistered value")
    if policy.get("donor_sampling_key") != "sha256(seed:anchor_key)":
        raise ValueError("checkpoint donor sampling is not deterministically keyed")
    if policy.get("replacement_across_anchors") is not True:
        raise ValueError("checkpoint donor sampling replacement policy is missing")
    if int(policy.get("donor_sampling_seed", -1)) != int(metadata.get("seed", -2)):
        raise ValueError("checkpoint donor sampling seed does not match training seed")
    if not float(policy.get("gamma_window", 0.0)) > 0.0 or not float(policy.get("gamma_local", 0.0)) > 0.0:
        raise ValueError("checkpoint counterfactual thresholds are not positive")

    return {
        "ok": True,
        "objective": EXPECTED_OBJECTIVE,
        "epoch": int(payload.get("epoch", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "beta_target": metadata["beta_target"],
        "beta_warmup_ratio": metadata["beta_warmup_ratio"],
        "gradient_clip": metadata["gradient_clip"],
        "fixed_hint_state_gradient": metadata["fixed_hint_state_gradient"],
        "fixed_hint_answer_gradient": metadata["fixed_hint_answer_gradient"],
        "fixed_hint_freeze_scope": metadata["fixed_hint_freeze_scope"],
        "fixed_hint_state_isolation": metadata["fixed_hint_state_isolation"],
        "state_verbalizers": {
            "normal_text": verbalizers.get("normal_text"),
            "anomalous_text": verbalizers.get("anomalous_text"),
            "normal_id": normal_id,
            "anomalous_id": anomalous_id,
        },
        "optimizer_groups": optimizer_groups,
        "counterfactual_index_version": metadata["counterfactual_index_version"],
        "donor_top_m": donor_top_m,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-donor-top-m", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = audit_loss_objective_checkpoint(
        payload,
        expected_donor_top_m=args.expected_donor_top_m,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
