# Branch profile: loss_final

`loss_final` is the loss-function-only experiment based on `main`. Its executable model architecture is the author released-code/checkpoint architecture. Relative to `main`, only the declared question-type-routed counterfactual objectives and their required data plumbing may change.

## Architecture invariant

The formal architecture variant is always `loss_only`:

- Local Hint: encoder states are projected and used as queries over the learned prototype bank. There is no continuous residual bypass or gated fusion.
- Fixed Hint: 30 learned query vectors pass through the same prototype cross-attention before insertion into the LLM. They are not 30 directly injected soft-prompt tokens.
- Prototype bank: 1000 prototypes, matching the released author checkpoint.
- QK-Norm: disabled.
- Frozen modules: the Phase-I time-series encoder and backbone LLM.
- Trainable Phase-II state: exactly the author-compatible nine Perceiver tensors (`fix_prompt_embeddings`, prototype mapping, local projection, and four attention projections).

Appendix B.3 reports `K=8`, 1024 prototypes, and a six-layer time-series encoder, whereas the released code/checkpoint uses `K=30`, 1000 prototypes, and eight layers. This branch follows the released implementation because the author checkpoint cannot strictly load into the appendix dimensions.

`tools.axis_repro.train_phase2_loss_final` rejects `full` and every redesign variant. `tools.axis_repro.audit_loss_final_checkpoint` also rejects QK-Norm, residual-bypass, direct-task-prompt, or unexpected Perceiver parameters. Checkpoints produced earlier with `architecture.variant=full` are incompatible and must not initialize a new `loss_final` run.

## Declared objective changes

Relative to the `main` answer NLL, the Joint stage routes auxiliary supervision by question type:

1. multiple choice: factual/counterfactual state cross-entropy;
2. true/false: original-question-conditioned factual/counterfactual cross-entropy;
3. open ended: answer-content QCAR ranking on eligible anomaly-deletion pairs.

Answer NLL updates the complete author Hint Tuner, including the Fixed-Hint query vectors. Auxiliary losses stop-gradient at the processed Fixed-Hint output, so they cannot update `fix_prompt_embeddings` through the Fixed path; they train the evidence path. Because Local and Fixed hints share prototype attention in the author model, the shared prototype/attention parameters remain trainable through Local-Hint evidence, which is intentional.

## Canonical entry points

- Counterfactual index: `tools.axis_repro.build_counterfactual_index`.
- Question-type cache: `tools.axis_repro.build_question_type_supervision`.
- Training: `tools.axis_repro.train_phase2_loss_final`.
- Checkpoint audit: `tools.axis_repro.audit_loss_final_checkpoint`.
- Inference: `tools.axis_repro.run_inference_cli` with `--architecture-variant loss_only` (the default).

Architecture-redesign documents and utilities remain as research history shared with redesign branches, but they are not executable protocol for `loss_final`. Formal comparison must use the same Phase-I checkpoint, split, budget, validation selection, generation protocol, and judge as its answer-only control.
