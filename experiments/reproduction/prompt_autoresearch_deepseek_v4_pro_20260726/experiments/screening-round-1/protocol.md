# Screening round 1 protocol

## Status and purpose

This is a preregistered **exploratory screening** round. It tests whether the
failure mechanisms identified on the historical `paper140` comparisons can be
reduced on previously unused QA records without tuning on the final reference
set.

## Locked data isolation

- Candidate test collection: 284 QA / 142 series.
- Already exposed final reference set: `paper140`, 140 QA / 70 series.
- Search pool: the remaining 144 QA / 72 series.
- Deterministic series-level split, seed `20260726`:
  - screening: 12 series / 24 QA;
  - validation: 36 series / 72 QA;
  - internal holdout: 24 series / 48 QA.
- A series may occur in exactly one split. `paper140` is not used in screening,
  validation, failure-driven prompt revision, or internal selection.

The split generator optimizes balance jointly over question type and anomaly
label at the series level, then freezes the selected filenames and record IDs
with SHA-256 hashes.

## Fixed execution protocol

- Released author checkpoint only.
- Formal model inference runs on an approved GPU server.
- Series batching, beam size 5, `--skip-loss`.
- Three GPU ranks when three free GPUs are available.
- Only `deepseek-v4-pro` is used as Judge.
- The Judge prompt, rubric, reasoning setting, score-token logprob method,
  20-sample fallback rule, and aggregation code are identical to the locked
  Baseline evaluation.
- Baseline and fixed-role controls are evaluated once and cached per split.

## Prompt factors

All candidates preserve:

- the author's two-block Values / Per-Step Analysis layout;
- all 30 Fixed tokens;
- the released generation boundary;
- the Fixed role rename.

Five factors are screened:

- **A, boundary guard:** use evidence only inside `[start,end)`.
- **B, anomaly calibration:** require unexpected contrast in the complete
  pattern and supporting latent evidence; ordinary extrema or smooth behavior
  are not anomalous by themselves.
- **C, task-conditioned decision rule:** separate MC option verification, TF
  proposition-to-label mapping, and OE diagnostic coverage.
- **D, qualitative numeric guard:** do not routinely convert or quote exact
  values for qualitative questions.
- **E, single-answer concise guard:** do not repeat, revise, or contradict the
  answer.

The twelve locked conditions P00--P11 and their factor matrix are defined in
`../../candidate-matrix.md`. P00 is the existing `fixed_role` control. P01--P11
are implemented as explicit named modes and their exact rendered prompt
templates are saved before inference.

## Predictions

1. B should reduce false positives on normal MC and TF cases.
2. C should most strongly improve TF Correctness/Justification and reduce
   MC post-hoc explanations.
3. D should reduce fabricated numerical evidence, especially in OE.
4. E should reduce TF self-revision, but may reduce OE Completeness if too
   restrictive.
5. P06, P09, P10, and P11 are the most likely Pareto candidates.

## Locked screening analysis

For every condition, compute the ten Table-I metrics and paired differences
against the split Baseline:

`MC Final, MC Correctness, MC Reasoning, OE Final, OE Accuracy,
OE Completeness, OE Relevance, TF Final, TF Correctness, TF Justification`.

The ranking is lexicographic:

1. number of dimensions with delta `>= 0`;
2. minimum of the ten deltas;
3. mean of the ten deltas.

Because 24 QA is deliberately small, screening does not claim final success.
Four candidates advance using the ranking plus a preregistered diversity guard:
if the top four are all supersets of the same factor pattern, replace the
fourth with the best candidate that isolates a different mechanism. Failure
case analysis is performed for every advanced candidate and for any candidate
with a task-Final drop of at least `0.25`.

After 5--10 completed conditions, perform an outer-loop synthesis before
creating any round-2 prompt. New prompts discovered from observed failures are
labelled exploratory and must be re-tested on validation and holdout.

## Final success criterion

Only the single prompt locked after internal holdout may run on `paper140`.
Success requires every one of the ten Table-I metrics to be no lower than the
cached Baseline under the same `deepseek-v4-pro` protocol. Improvement in all
ten is preferred. A prompt that improves averages while lowering any one
dimension is not a successful Pareto result.