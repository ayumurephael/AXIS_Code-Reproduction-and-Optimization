# Branch profile: architecture_redesign

`architecture_redesign` is the combined redesign branch. It applies the coherent counterfactual state objective used by `loss_resesign` and enables the full architecture variant. It is not a one-factor comparison against `main`.

## Declared changes

Relative to `main`:

1. coherent factual/counterfactual binary state supervision;
2. QK-Norm on prototype cross-attention;
3. continuous encoder-state bypass with gated prototype sidecar;
4. directly learned 30-token task soft prompt;
5. parameter-group learning rates associated with the redesigned paths.

For a clean architecture-effect estimate, compare `--architecture-variant full` with `--architecture-variant loss_only` inside this branch under the same counterfactual index, objective, optimizer groups, seed, budget and validation rule. Comparing this branch directly with `main` estimates the whole redesign bundle.

## Canonical entry points

- Counterfactual index: `tools.axis_repro.build_counterfactual_index`.
- Training: `tools.axis_repro.train_phase2_architecture_redesign` (defaults to `full`).
- Checkpoint audit: `tools.axis_repro.audit_architecture_checkpoint` and `tools.axis_repro.audit_loss_objective_checkpoint`.
- Inference: `tools.axis_repro.run_inference_cli` with the checkpoint's exact architecture metadata.

Formal commands and the 2×2×2 variants are in `ARCHITECTURE_REDESIGN_REPRODUCTION.md`. Design rationale is retained in `AXIS架构改进说明.md`; objective rationale is retained in `训练目标(损失函数)改进_new.md`. Those research notes are not execution instructions.