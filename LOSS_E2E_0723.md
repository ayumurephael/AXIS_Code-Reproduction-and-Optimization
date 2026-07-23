# loss_e2e_0723 implementation protocol

This branch implements the confirmed 2026-07-23 continuation experiment
without changing the existing Phase-I to Phase-II reproduction entry points.

## Locked design

- Both arms start from the exact same author
  `axis_qa_by_pretrain_best_accelerate/model_optimizer.pth` model weights.
- The author payload has no optimizer state. Both arms therefore create fresh,
  identical AdamW optimizers with learning rate `1e-4`, weight decay `1e-5`,
  default PyTorch betas and epsilon.
- The training split is the 95% series split with seed 72. The formal paper140
  test manifest remains the fixed seed-42 manifest.
- Exact answers equal to `Error generating answer.` are excluded from both
  objectives. The formal run must find exactly 13 such training QA rows and
  writes their non-answer identifiers to `data_exclusion_manifest.json`.
- Both arms run exactly two complete epochs on three DDP ranks with one series
  per rank and no early stopping. The 25%, 50%, 75%, and 100% checkpoints are
  diagnostics; the 100% endpoint is the primary comparison.
- The control uses the original dynamic fixed hint and global continuation-token
  NLL. The treatment caches fixed hint F0 once at step zero under BF16 autocast,
  stores the injected FP16 tensor in every checkpoint, stops its gradient, and
  uses the segmented objective.
- Segment weight is `alpha=0.40` for the conclusion and `0.60` for the
  explanation. Each segment is token-mean normalized, each QA row is combined,
  and QA rows are then mean normalized globally across all DDP ranks.
- TF conclusion is the True/False label plus the first complete semantic
  statement. Its explanation starts at the next statement; the segments do not
  overlap. A missing second segment uses the available segment alone.
- MC split priority is Explanation/Reasoning marker, blank line, first line,
  first sentence, then whole-answer fallback. OE uses the first sentence and
  merges the second when the first contains fewer than five content tokens.
- Prompt, padding, BOS, intermediate tokenizer special tokens, separator,
  `Answer:`, and split-marker tokens are not part of treatment segments. The
  final appended EOS is supervised by the explanation when it exists and by
  the conclusion otherwise. Fast-tokenizer offsets bind character spans to the
  exact answer tokenization.
- Standalone answer encoding uses `truncation=False`. The formal preflight
  audits all included answers, requires zero over-limit answers, and fails
  closed if its untruncated answer block differs from the baseline full input.
- Sentence boundaries protect common abbreviations such as `e.g.` and `i.e.`
  when they continue a statement. The exclusion manifest also records
  conclusion/explanation character and content-word distributions plus three
  deterministic examples for every segmentation rule.
- Every step records the global L2 norm of the DDP-averaged gradient. Milestone
  checkpoints and the final summary record relative parameter change from the
  common author checkpoint. Gradient clipping is available explicitly but is
  disabled in the locked two-arm protocol unless a calibration run justifies it.

## Formal commands

Run from the repository root in the pinned CUDA environment. Do not run these
commands on a CPU-only host.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_loss_e2e_ddp \
  --checkpoint /path/to/axis_qa_by_pretrain_best_accelerate/model_optimizer.pth \
  --data /path/to/anomaly_llava_training_dataset \
  --output experiments/loss_e2e_0723/control \
  --arm control --epochs 2 --seed 72 --lr 1e-4 --epoch2-lr 3e-5 \
  --weight-decay 1e-5

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_loss_e2e_ddp \
  --checkpoint /path/to/axis_qa_by_pretrain_best_accelerate/model_optimizer.pth \
  --data /path/to/anomaly_llava_training_dataset \
  --output experiments/loss_e2e_0723/treatment \
  --arm treatment --epochs 2 --seed 72 --alpha 0.40 \
  --lr 1e-4 --epoch2-lr 3e-5 --weight-decay 1e-5
```

Use `tools.axis_repro.run_inference_loss_e2e` for checkpoints from this
experiment. It is identical to the pinned inference protocol except that it
restores the treatment checkpoint's cached F0 tensor.

## Confirmed formal recovery schedule

The LR=1e-4 treatment trajectory produced a fail-closed non-finite gradient at
step 9987, after a clean completed epoch-1 checkpoint at step 9500. The
confirmed recovery protocol is identical across both arms:

- epoch 1: AdamW LR `1e-4`;
- epoch 2: AdamW LR `3e-5`;
- treatment resumes only from the complete step-9500 epoch boundary, restoring
  model, optimizer, cumulative metrics and cached F0;
- no mid-epoch batch is skipped and no test score selects this schedule.

The treatment recovery command adds only the audited epoch-boundary checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_loss_e2e_ddp \
  --checkpoint /path/to/author/model_optimizer.pth \
  --data /path/to/anomaly_llava_training_dataset \
  --output experiments/loss_e2e_0723/treatment \
  --arm treatment --epochs 2 --seed 72 --alpha 0.40 \
  --lr 1e-4 --epoch2-lr 3e-5 --weight-decay 1e-5 \
  --resume-from experiments/loss_e2e_0723/treatment/step_9500.pth
```

## Learning-rate calibration

The attention-risk review requires a fixed, non-test calibration before a
revised formal run. Run all four combinations of arm and learning rate with
`--run-purpose calibration --max-steps 200`, keeping every other argument
identical:

```bash
# Repeat for arm={control,treatment} and lr={1e-4,3e-5}.
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nproc_per_node=3 \
  -m tools.axis_repro.train_loss_e2e_ddp \
  --checkpoint /path/to/model_optimizer.pth \
  --data /path/to/anomaly_llava_training_dataset \
  --output experiments/loss_e2e_0723/calibration/${arm}_${lr} \
  --arm ${arm} --run-purpose calibration --max-steps 200 \
  --epochs 2 --seed 72 --alpha 0.40 --lr ${lr} --weight-decay 1e-5
```

The script writes the exact 600-series rank/order manifest. LR selection must
use gradient norms, parameter drift, and calibration loss only—never paper140.
