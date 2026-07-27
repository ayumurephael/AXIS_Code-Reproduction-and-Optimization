# AXIS Prompt Final Full-284 Evaluation Protocol

## 1. Purpose

This protocol replaces only the former per-case `zero correct→wrong` gate.
It does not reopen prompt design or repeat prompts that already have clear
negative evidence.

The registered implementation contains 69 modes in
`src/models/AXIS/prompt_stage_a.py`: one released Baseline, 65 prompt
conditions, and three released ablation controls. Every registered condition
is included in the evidence inventory. A condition that already failed on a
broader comparable split is excluded using its existing audited result rather
than rerun.

## 2. Paper and dataset scope

- The paper's Table I uses the author-compatible `paper140` path: 70 series,
  140 QA, and Gemini 2.5 Pro.
- The released test directory contains 142 series and 284 QA.
- The user-requested final generalization check in this project uses all 284
  QA (`subset=full`), not `paper140`.
- Paper Table-I numbers are reported only as external context. They are never
  mixed with full-284 scores.

The final full-284 Baseline and every candidate use the same released
checkpoint, inference protocol, Judge model, rubric, and score readout.

## 3. Fixed inference and Judge configuration

- Checkpoint:
  `axis_qa_by_pretrain_best_accelerate/model_optimizer.pth`
- Required checkpoint SHA-256:
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`
- Dataset: released `AXIS_qa_test`, `subset=full`, frozen 284-record manifest
- Inference: series-batched, beam size 5, `max_new_tokens=1000`,
  `skip_loss=true`
- Formal Judge: `deepseek-v4-pro` only
- G-Eval: the existing author-dimension rubric and 1–5 score scale
- Primary readout: complete score-token log-probability expectation
- Registered fallback: exact mean of 20 valid integer-score samples only
- Hardware: authorized remote GPU only for model inference and formal Judge
  audit; no local GPU inference or training

The existing `test_epoch_3_full284` predictions cannot be the Baseline:
they use checkpoint SHA-256
`8d562f8ff22c6709caaa2f7ba8208f3631a652d1932f001f39fa75ff8700fa4b`,
not the released checkpoint above, and they have no matching formal scores.
Therefore the released-checkpoint full-284 Baseline must be generated and
judged anew.

## 4. Wide historical screen

The screen is deliberately permissive enough to retain a plausible candidate
with one small-sample metric shortfall, while using later negative evidence to
avoid rerunning known failures.

A prompt advances to full-284 if all of the following hold:

1. It changes at least one Open-Ended prompt branch relative to Baseline.
2. On at least one audited comparable historical aggregate it has at least
   9/10 Table-I metrics at or above that split's Baseline.
3. Its worst delta on that aggregate is at least `-0.15`.
4. Its mean ten-metric delta is positive.
5. At least one OE metric is strictly above that split's Baseline.
6. It has no later, broader comparable evaluation with fewer than 9/10
   nonnegative metrics or a worst delta below `-0.15`.

This rule necessarily retains any prompt that is 10/10 nonnegative on each of
screening24, validation72, development96, and exposed holdout48. The OE-change
condition implements the user's requirement that literal Baseline OE reuse is
not a comprehensive improvement.

The frozen advancing set is:

| Mode | Existing evidence | Nonnegative | Worst delta | Positive OE deltas | Historical limitation |
|---|---|---:|---:|---|---|
| `lit_r1_08_oe_re2` | screening24 | 9/10 | -0.1429 | Final, Accuracy, Completeness | OE Relevance -0.1429 |
| `lit_r1_10_triplet_re2` | screening24 | 9/10 | -0.1429 | Final, Accuracy, Completeness | OE Relevance -0.1429 |

All other modes are excluded by an existing audited result, exact Baseline OE
reuse, or ablation-control status. They must not be rerun.

## 5. Exact component reuse

Byte- and tensor-identical branches reuse the canonical Baseline response and
its Judge scores:

- `lit_r1_08_oe_re2`: infer and judge only Open-Ended records; MC and TF reuse
  canonical Baseline components.
- `lit_r1_10_triplet_re2`: all three question families change and must be
  inferred and judged.

Reused and newly generated components must have an explicit provenance
manifest. A repeated generation of an unchanged branch must not replace the
canonical Baseline component.

## 6. Final success criterion

For a candidate to be reported as a comprehensive improvement on full-284:

1. all ten Table-I metrics are no lower than the full-284 Baseline within
   numerical tolerance `1e-6`;
2. at least three of the ten metrics are strictly higher than Baseline by more
   than `1e-6`;
3. the candidate actually changes at least one OE prompt branch; and
4. at least one of OE Final, Accuracy, Completeness, or Relevance is strictly
   higher than Baseline by more than `1e-6`.

There is no `correct→wrong` rejection gate. Parsed-decision transitions remain
diagnostics and failure-case evidence only.

## 7. Required audit and reporting

Every reported result must pass a fail-closed audit for:

- 284 unique, non-empty predictions per assembled mode;
- the exact expected question-type/dimension score keys;
- no duplicate or missing prediction/score keys;
- Judge model exactly `deepseek-v4-pro`;
- valid score prompt hashes and registered score methods;
- checkpoint, code commit, dataset manifest, prompt hashes, and component
  provenance.

`prompt_final.md` must contain:

- the complete 69-mode screening inventory and exclusion reasons;
- paper versus full-284 protocol clarification;
- full-284 Baseline and candidate Table-I scores;
- absolute deltas and relative percentages;
- the exact prompt text for every full-284 candidate;
- the final criterion decision;
- for every successful changed question family, a question, gold answer,
  Baseline response, candidate response, and Judge delta;
- explicit disclosure when a successful routed prompt leaves a requested
  question family byte-identical to Baseline, because no improvement example
  may be fabricated for an unchanged branch.

