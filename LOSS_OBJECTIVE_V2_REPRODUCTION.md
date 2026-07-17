# Coherent counterfactual state objective (v2)

This protocol replaces the first answer-head SLR objective. The original answer is supervised only
on the real context. Every valid auxiliary pair contains one original context and one coherent
counterfactual context, and both Local and Window evidence are generated from the same patched full
sequence.

## Objective

For state `z` and a counterfactual state `1-z`:

```text
answer_loss = full-answer teacher-forced NLL on the original QA context
state_loss  = 0.5 * [CE2(original_context, z) + CE2(counterfactual_context, 1-z)]
total_loss  = answer_loss + beta(step) * state_loss
```

`beta(step)` increases linearly from `0` to `0.2` over the first 10% of synchronized updates. The
state classifier selects exactly two one-token verbalizers from the frozen LLM output head. It does
not generate an answer, supervise EOS, use a margin, use an answer-head mask, or sample one of the
old Local/Window intervention modes.

The state path detaches the Fixed/task-prompt embeddings. Thus the auxiliary state loss cannot
modify that branch, while the full-answer loss can still learn the K=30 task soft prompt. The LLM
and Phase-I TS encoder remain frozen.

## Counterfactual construction

- Anomalous anchor: patch only the target interval with its paired `normal_series` interval.
- Normal anchor: retrieve a same-length paired anomalous donor and transplant only
  `time_series - normal_series` into the anchor interval.
- Re-encode every patched full sequence with the frozen Phase-I encoder; never reuse a donor Local
  embedding at an anchor position.
- `tau=0.25`. Window and Local gamma thresholds are P5 of the nonzero labeled-anomaly effects on
  the seeded training split.
- Donors and anomalous deletion pairs must pass Window and Local control-gap constraints.
- Residual transplants must pass a post-patch Window and Local effect audit.
- There is no phase fallback. Invalid pairs are excluded from state CE and are never assigned a
  flipped label on an unchanged input.

## Recommended parameter groups

| Group | Parameters | LR |
|---|---|---:|
| Local/continuous | Local projection, continuous bypass, gate, fusion norm | `5e-5` |
| Prototype attention | prototype mapping, Q/K/V/out projections, QK scale | `2e-5` |
| Task prompt | fixed or direct K-token task prompt, answer loss only | `5e-5` |

AdamW weight decay is `1e-5`; global Perceiver gradient clipping is `1.0`.

## GPU-server commands

```bash
# Build and audit coherent counterfactual index v2 (single GPU indexing stage).
CUDA_VISIBLE_DEVICES=0 python -m tools.axis_repro.build_counterfactual_index \
  --data data/anomaly_llava_training_dataset \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --output experiments/reproduction/coherent_state_v2/counterfactual_index.json \
  --seed 72 --train-ratio 0.95 --tau 0.25 --gamma-percentile 5

python -m tools.axis_repro.audit_counterfactual_index \
  --data data/anomaly_llava_training_dataset \
  --index experiments/reproduction/coherent_state_v2/counterfactual_index.json \
  --output experiments/reproduction/coherent_state_v2/counterfactual_index_audit.json

# Full architecture: v2 loss + QK-Norm + continuous bypass/sidecar + K=30 task prompt.
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_architecture_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/coherent_state_v2/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/architecture_redesign/full_state_v2/phase2 \
  --architecture-variant full --epochs 3 --seed 72 \
  --local-lr 5e-5 --attention-lr 2e-5 --prompt-lr 5e-5 \
  --weight-decay 1e-5 --gradient-clip 1.0 \
  --beta 0.2 --beta-warmup-ratio 0.1

python -m tools.axis_repro.audit_loss_objective_checkpoint \
  --checkpoint experiments/reproduction/architecture_redesign/full_state_v2/phase2/epoch_1.pth
python -m tools.axis_repro.audit_architecture_checkpoint \
  --checkpoint experiments/reproduction/architecture_redesign/full_state_v2/phase2/epoch_1.pth \
  --expected-variant full
```

Formal checkpoint selection remains the complete 1,500-series teacher-forced validation protocol.
After strict minimum-NLL selection, run full284 inference, filter the fixed paper140 manifest, and
run strict DeepSeek-v4-pro Appendix-E G-Eval on the GPU server. Credentials must be injected only
from the protected server-side credential file and must never enter logs or artifacts.
