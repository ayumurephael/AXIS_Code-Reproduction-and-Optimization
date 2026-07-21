# Branch profile: loss_resesign

`loss_resesign` keeps the original AXIS architecture and replaces the Phase-II auxiliary training objective with coherent factual/counterfactual state supervision.

## Declared changes

Relative to `main`:

- answer NLL remains on the real QA context;
- each valid pair adds two-class state CE on factual and coherent counterfactual contexts;
- the frozen LLM's next-token logits over two single-token verbalizers define the state probability;
- counterfactuals are constructed and audited before training;
- `beta` warms from 0 to 0.2;
- the formal configuration uses parameter-group learning rates specified by the loss design.

The LLM and Phase-I time-series encoder remain frozen. This branch does not enable QK-Norm, continuous bypass or direct task prompt; its architecture variant is `loss_only`.

Because the formal redesign includes parameter-group learning rates, direct comparison with `main` estimates the complete loss-training recipe. To isolate the auxiliary objective from optimizer grouping, run an additional predeclared `beta=0` optimizer control with the same entry point and budget; do not choose whether to include that control after seeing test scores.

## Canonical entry points

- Index: `tools.axis_repro.build_counterfactual_index` and `audit_counterfactual_index`.
- Training: `tools.axis_repro.train_phase2_loss_redesign`.
- Checkpoint audit: `tools.axis_repro.audit_loss_objective_checkpoint`.
- Inference/evaluation: the common protocol in `REPRODUCTION.md` and `EVALUATION_PROTOCOL.md`.

Exact commands are in `LOSS_REDESIGN_REPRODUCTION.md`. Objective rationale is retained in `训练目标(损失函数)改进_new.md`; it is not the runtime command source.