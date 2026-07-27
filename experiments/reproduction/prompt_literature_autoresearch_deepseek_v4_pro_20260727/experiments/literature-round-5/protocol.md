# Literature Round 5 — conservative decision calibration

## Preregistration status

Frozen before any Round-5 GPU inference, response inspection, or Judge call. This is the final failure-driven cycle under the registered cap of 30 new modes. The prior 96-QA development union and 48-QA exposed holdout are adaptive falsification resources, not untouched test sets. `paper140` remains reserved for exactly one locked candidate.

## Failure mechanism being tested

The exposed-holdout semantic MC route changed two Baseline-correct normal windows into anomalous answers:

- `series_000139:0`: ordinary alternating/random fluctuation was relabeled as an anomalous irregular cluster;
- `series_000141:1`: ordinary negative troughs were relabeled as sudden drops with recoveries.

It also fixed `series_000022:1`, where a truly localized oscillatory anomaly had been missed. The failure is therefore not semantic option binding alone; it is an anomaly-threshold problem. Round 5 separates two conservative mechanisms:

1. **structured-signature guard** — require the option's defining localized/persistent/recovery/boundary signature and explicitly reject raw irregularity as sufficient evidence;
2. **status-then-shape** — decide normal/anomalous status from the learned hints before matching the option text, preventing an anomaly-worded distractor from setting the status.

## Frozen candidates

Existing fallback (does not consume a new-mode slot):

- `lit_r4_01_tf_neg_re2`: exact Baseline MC/OE and positive/non-negated TF; qualitative RE2 only for TF propositions containing an explicit grammatical negation.

Four final new modes:

1. `lit_r5_01_mc_structured_guard`: structured-signature MC; exact Baseline OE/TF.
2. `lit_r5_02_mc_status_then_shape`: status-then-shape MC; exact Baseline OE/TF.
3. `lit_r5_03_joint_structured_tf_neg_re2`: mode 1 MC plus the frozen explicit-negation TF route; exact Baseline OE.
4. `lit_r5_04_joint_status_tf_neg_re2`: mode 2 MC plus the frozen explicit-negation TF route; exact Baseline OE.

No Evidence Contract, hint rename, value reformatting, answer prefill, or OE change is permitted. The released opening, Values → Per-Step Analysis → `Overall Summary Hints` order, 30 Fixed tokens, and question-ending boundary are frozen.

## Execution and component reuse

- Released checkpoint SHA-256: `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- GPU inference: authorized GPUs only, series batching, beam size 5, `--skip-loss`.
- Judge: `deepseek-v4-pro` only; score only unique changed components.
- Use canonical audited Baseline responses/scores for every unchanged record.
- On development96, reuse the already audited explicit-negation TF component from Round 3/4; infer only the two new MC source modes.
- On exposed holdout48, infer the two new MC sources and `lit_r4_01_tf_neg_re2`; assemble all five candidates from exact components.
- Fail closed on missing/duplicate keys, prompt-hash mismatch, unexpected provider/model, or unapproved scoring method.

## Gates and unique lock

A candidate is eligible only if, separately on pooled development96, screening24, validation72, and exposed holdout48:

1. all ten Table-I deltas versus the exact split Baseline are nonnegative;
2. no parsed Baseline-correct MC/TF decision becomes wrong;
3. every routed TF output yields one unambiguous truth decision;
4. provenance and Judge audits pass.

Rank eligible candidates lexicographically by worst delta, mean delta over ten dimensions, wrong→correct count, then fewer changed records. Lock exactly one top candidate without prompt edits. Only that candidate may be assembled and scored once on `paper140`. If no candidate passes, stop under the registered search cap; do not select by average gain.
