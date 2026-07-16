# Loss redesign reproduction

This branch implements the source-likelihood-ratio (SLR) objective described in
the loss-design specification supplied with this repository.

## Fixed protocol

- Base objective: full-context answer NLL.
- SLR weight: `beta=1.0`.
- Sequence margin: `m=ln(2)`.
- Counterfactual contamination threshold: `tau=0.25`.
- Minimum visible source effects: `gamma_window=gamma_local=0` with a strict
  non-zero source change.
- Modes: `local_only`, `window_only`, `full_local`, and `full_window`.
  Each DDP rank samples modes uniformly from an independently seeded RNG.
- An anchor without a valid source-specific counterfactual keeps the answer NLL
  and skips SLR for that row. Residual transplantation is not enabled.
- Split, optimizer, seed, and model selection match the baseline protocol.

## 1. Build the counterfactual index

Run on one GPU. The Phase-I encoder is frozen and each complete current/normal
series is encoded before window slicing.

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.axis_repro.build_counterfactual_index \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/loss_redesign/counterfactual_index.json \
  --seed 72 --train-ratio 0.95 --tau 0.25 \
  --gamma-window 0 --gamma-local 0
```

Do not start training until the emitted Window/Local donor coverage statistics
have been reviewed.

## 2. Multi-GPU smoke and formal training

```bash
export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_loss_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/loss_redesign/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/loss_redesign/smoke \
  --epochs 1 --max-steps 2 --save-every 0 --seed 72

torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_phase2_loss_redesign \
  --phase1 experiments/checkpoints/pretrain_single/pretrain_checkpoint_best.pth \
  --counterfactual-index experiments/reproduction/loss_redesign/counterfactual_index.json \
  --data data/anomaly_llava_training_dataset \
  --output experiments/reproduction/loss_redesign/phase2 \
  --epochs 3 --lr 1e-4 --weight-decay 1e-5 --seed 72 \
  --num-workers 2 --save-every 5000 --beta 1.0
```

## 3. Model selection and Table 1

Strip all three epoch checkpoints, run the same complete 1,500-series
teacher-forced validation inference, and select the minimum mean validation NLL.
Then run per-record full284 inference, filter the fixed paper140 manifest, run
strict DeepSeek-v4-pro G-Eval, audit, and aggregate using the baseline commands
in `readme.md`. Never select an epoch using test or judge scores.

Inject the DeepSeek key only for the judge subprocess:

```bash
python -m tools.axis_repro.run_with_secret \
  --credential-file /protected/path/THU_IE_GPU.md \
  python -m tools.axis_repro.geval_resilient ...
```
