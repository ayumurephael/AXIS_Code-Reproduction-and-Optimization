# Branch profile: loss_e2e_40epoch_gradient_guard

This branch is the formal AXIS Phase-II gradient-accumulation and local-input normalization experiment. It is not the canonical `main` baseline and it must not resume any checkpoint produced by the failed accumulation=1 lineage.

## Frozen experimental protocol

Both arms start independently from the same Phase-I `pretrain_checkpoint_best.pth`, seed-72 95%/5% split, and fresh seed-72 Phase-II initialization. They share:

- five DDP ranks, one series per rank and microbatch;
- gradient accumulation 26, nominal global batch 130 series;
- 5,700 microsteps and 220 optimizer-attempt windows per epoch;
- 40 epochs, BF16, AdamW, constant LR `1e-4`, weight decay `1e-5`;
- Treatment segmented loss with `alpha=0.40`;
- accumulation → exact global valid-row mean → guard → clip (`1.0`) → AdamW step;
- one checkpoint at every complete epoch and selection over all 40 checkpoints by fixed seed-72 validation global token NLL only.

The last window contains six microsteps per rank (30 series globally). Its loss denominator is the actual number of globally usable QA rows; it is never divided by 26.

## Independent arms

1. `--local-input-norm none`: accumulation-only arm. Run this first.
2. `--local-input-norm rmsnorm`: independent RMSNorm arm. Start again from the same Phase-I checkpoint and seed; do not resume the `none` arm.

RMSNorm is non-affine, uses epsilon `1e-6`, computes in FP32 independently at each local timestep over the last dimension, and is placed immediately before `local_word_proj`. It does not alter the fixed-hint path. Q/K activation metrics are intentionally disabled in this protocol.

## Formal entry points

```bash
# Accumulation-only arm

torchrun --standalone --nproc_per_node=5 \
  -m tools.axis_repro.train_phase2_treatment_40epoch_ddp \
  --phase1 <pretrain_checkpoint_best.pth> \
  --data <training_dataset> \
  --series-split-manifest <seed72_phase2_split.json> \
  --output <accumulation_only_output> \
  --local-input-norm none

# RMSNorm arm: a fresh run, not --resume-from the first arm

torchrun --standalone --nproc_per_node=5 \
  -m tools.axis_repro.train_phase2_treatment_40epoch_ddp \
  --phase1 <same_pretrain_checkpoint_best.pth> \
  --data <same_training_dataset> \
  --series-split-manifest <same_seed72_phase2_split.json> \
  --output <rmsnorm_output> \
  --local-input-norm rmsnorm
```

Complete-epoch resume is allowed only within the same arm and only from accumulation-v2 checkpoints whose source, data, Phase-I hash, optimizer state, RNG state, step counters, guard audit, and normalization policy pass validation.

For each arm, validate all `epoch_01.pth` through `epoch_40.pth` with:

```bash
torchrun --standalone --nproc_per_node=5 \
  -m tools.axis_repro.validate_select_phase2_40epoch \
  --checkpoints <epoch_01.pth ... epoch_40.pth> \
  --data <training_dataset> \
  --series-split-manifest <seed72_phase2_split.json> \
  --output <validation_output>
```

The validator reports global token NLL, conclusion NLL, evidence NLL, the Treatment row objective, and the corresponding checkpoint's training gradient/clipping/skip diagnostics. Only global token NLL selects the best checkpoint; test or Judge results are forbidden for selection.
