# Branch profile: architecture_redesign_fixedhint_frozen

`architecture_redesign_fixedhint_frozen` is based on the current `main` branch and carries the coherent counterfactual objective and full architecture bundle from `architecture_redesign`. Its defining correction is strict state-loss isolation for the directly learned task prompt.

## Declared changes

Relative to `main`:

1. coherent factual/counterfactual binary state supervision;
2. QK-Norm on prototype cross-attention;
3. continuous encoder-state bypass with gated prototype sidecar;
4. directly learned 30-token task soft prompt;
5. state-loss stop-gradient at the task prompt, while answer NLL continues to train it;
6. parameter-group learning rates associated with the redesigned paths.

The formal experiment uses `--architecture-variant full`. This branch evaluates the complete redesign bundle with the Fixed/task-prompt gradient contradiction corrected; it is not a one-factor architecture estimate against `main`.

## Canonical entry points

- Counterfactual index: `tools.axis_repro.build_counterfactual_index`.
- Training: `tools.axis_repro.train_phase2_architecture_redesign` (defaults to `full`).
- Checkpoint audit: `tools.axis_repro.audit_architecture_checkpoint` and `tools.axis_repro.audit_loss_objective_checkpoint`.
- Inference: `tools.axis_repro.run_inference_cli` with the checkpoint's exact architecture metadata.

Formal commands are in `ARCHITECTURE_REDESIGN_REPRODUCTION.md`. Design rationale is retained in `AXIS架构改进说明.md`; objective rationale is retained in `训练目标(损失函数)改进_new.md`. Those research notes are design sources, not executable protocol.