# Validation protocol

## Status

Preregistered after screening round 1 and before any inference on the
validation split.

## Data isolation

- Validation manifest: `../../data/manifests/validation.json`.
- 72 QA / 36 series, series-disjoint from screening, internal holdout, and
  `paper140`.
- Question counts: 26 MC, 22 OE, and 24 TF.
- The internal holdout and `paper140` remain untouched.

## Locked modes

Controls:

- `base`
- `fixed_role`

The four screening candidates selected by the preregistered lexicographic
rule:

- `pareto_p01_boundary` (A)
- `pareto_p03_task_rule` (C)
- `pareto_p06_abc` (A+B+C)
- `pareto_p07_abd` (A+B+D)

The implementation and exact prompt text are frozen at commit `3481b1a`.
Rendered prompt SHA-256 values are already recorded in
`../screening-round-1/prompt_catalog.md`.

## Execution and audit

- Released author checkpoint only.
- Approved GPU node only; series batching, beam size 5, `--skip-loss`.
- Only `deepseek-v4-pro` Judge.
- Judge prompt, rubric, score-token logprob readout, exact 20-sample fallback,
  and Table-I aggregation remain identical to screening and Baseline.
- Fail closed on missing predictions/scores, unexpected modes, provider/model
  mismatch, invalid prompt hashes, or an unapproved scoring method.

Expected work:

- Predictions: 6 modes x 72 records = 432.
- Judge dimensions per mode:
  `26*2 + 22*3 + 24*2 = 166`.
- Judge scores: 6 x 166 = 996.

## Locked selection and holdout gate

Compute all ten Table-I deltas versus the validation Baseline and rank by:

1. number of nonnegative dimensions;
2. minimum dimension delta;
3. mean of the ten deltas.

A candidate is eligible for internal holdout only if it has at least 8/10
nonnegative dimensions and none of MC Final, OE Final, or TF Final is more
than 0.10 below Baseline. Advance the best two eligible candidates. If fewer
than two qualify, do not touch internal holdout; perform a new outer-loop
synthesis and preregister another development batch instead.

Failure-case analysis is required for each holdout-eligible candidate and any
candidate with a task-Final drop of at least 0.25. No prompt may be revised
after seeing validation results and then carried to holdout without a new
preregistered development/validation cycle.
