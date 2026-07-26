# AXIS Prompt Research Findings

## Research Question

Can a prompt-only intervention on the released AXIS checkpoint improve MC, OE, and TF simultaneously without trading one task family against another?

## Current Understanding

The existing results do not support a universally beneficial long Evidence Contract. The old full Contract shifts quality toward OE and away from MC/TF. Rewriting value scaling and token alignment recovers part of MC/TF, but loses the old Contract's OE benefit. Paired case analysis attributes the trade-off to cross-task instruction overload, missing anomaly calibration, TF negation-to-label failures, MC option anchoring, numeric salience, and inference-prompt distribution shift.

## Key Results

- Fixed role versus Baseline is essentially tied in Macro Final.
- Old full Contract versus fixed role: MC Final -0.3586, OE Final +0.1721, TF Final -0.3000.
- Revised Contract versus fixed role: MC Final -0.3051, OE Final -0.1669, TF Final -0.0524. TF Correctness is an exception: it increases by about 0.0714 while TF Justification decreases by about 0.2381.
- Revised versus old Contract: MC Final +0.0535, OE Final -0.3390, TF Final +0.2476.
- Old Contract changes four MC and eight TF records from a correct extracted decision to an incorrect one.
- Revised OE explicit out-of-window step citations rise to 18.18%, versus 3.64% under fixed role; OE unsupported-number fraction rises by 5.06 percentage points.
- Concrete revised failures include a 148-character truncated answer and an OE answer that invents steps 800, 1003, and 20001.

## Patterns and Insights

- Prompt changes redistribute performance across task families.
- Explicit evidence instructions can increase OE coverage, but lengthy global rules can distract closed-form decisions.
- Correcting a representational inconsistency is not sufficient for Pareto improvement.
- OE gains come from describing the actual transition and preserving uncertainty, not from added length: old-Contract OE response-length change has Pearson correlation -0.13 with Final change.
- TF needs a short proposition-to-label rule; longer reasoning correlates negatively with revised-vs-fixed TF Final change (Pearson -0.44).

## Lessons and Constraints

- Do not optimize repeatedly on `paper140` without a separate selection set.
- Do not equate strict output parsing with answer correctness.
- Do not attribute the revised result to only one clause: multiple clauses changed jointly.
- Do not make routine scale conversion salient for qualitative questions.
- Do not allow a second answer or label revision in TF.
- Do not cite exact steps or values unless they are present inside the stated window.
- Formal evaluation must use only `deepseek-v4-pro`, matching the locked G-Eval protocol.
- All inference and Judge work must run on an approved GPU node; no local GPU or long training.

## Open Questions

- Which exact failure types drive the MC and TF losses?
- Which old-Contract behaviors produced the OE gains?
- Can Pareto-v1 preserve old-Contract OE transition reasoning while avoiding the closed-task and generation failures?
- Which subset of boundary, calibration, task-decomposition, numeric, and
  single-answer guards is sufficient without causing inference-prompt
  distribution shift?

## Locked Search Decision

The user confirmed that success means **all ten Table-I metrics are no lower
than Baseline**, with improvement in all ten preferred. All prompt-only changes,
including question-type-conditioned rules, are allowed. The search uses a
series-disjoint 24/72/48-QA screening/validation/holdout split from the 144
non-`paper140` QA, followed by exactly one final `paper140` run for the locked
winner.

## Optimization Trajectory

Screening round 1 completed 13 inference conditions and 715 formal Judge
dimension scores on the frozen 24-QA development split. No candidate is yet a
strict Pareto success.

- P01 boundary guard: 9/10 nonnegative dimensions, worst -0.286, mean +0.253.
- P03 task rule: 8/10, worst -0.143, mean +0.205.
- P06 A+B+C: 8/10, worst -0.286, mean +0.287.
- P07 A+B+D: 7/10, worst -0.143, mean +0.038.

The four candidates advance unchanged to the frozen 72-QA validation split.
The internal holdout and `paper140` remain untouched.

## Screening Round 1 Mechanistic Findings

- A single boundary guard is the most robust common intervention and causes no
  correct-to-wrong parsed decision flips in screening.
- The current TF task rule fixes negation cases but can suppress a real spike
  or convex anomaly by forcing unnecessary re-diagnosis.
- The anomaly-calibration list is not behaviorally neutral: mentioning peaks,
  trends, extrema, and variability can make those features more salient and
  either create MC false positives or TF false negatives.
- OE questions have different speech acts. A diagnostic question may support
  a conclusion; a methodological “how would you evaluate / what evidence
  would you seek” question requires conditional evidence and methods. Forcing
  every OE answer to start with a diagnostic conclusion loses relevance and
  completeness.
- Shared “shortest explanation” wording improves MC output control but is
  harmful to OE. Any future brevity rule should be closed-task-only.
- Fewer unsupported numbers do not guarantee higher G-Eval scores: P09 has the
  lowest OE unsupported-number fraction but loses every OE record.
## Validation Round 1

The four advanced candidates and the fixed-role control were formally scored
on the series-disjoint 72-QA validation split. No candidate met the locked gate,
so the internal holdout remains untouched.

- P07 (A+B+D) ranked first: 7/10 nonnegative dimensions, worst delta -0.457,
  mean delta +0.128. It improved TF Final by +0.662 but reduced OE Final by
  -0.271.
- Fixed role improved MC Final by +0.232 and TF Final by +0.604, but reduced OE
  Final by -0.191.
- P01 was the safest common rule but achieved only 5/10 nonnegative dimensions.
- P03 and P06 produced explicit TF polarity contradictions: they answered
  `False` while explaining that a negated no-anomaly proposition was true.
- P03/P06 also changed correct MC anomaly decisions to non-anomalous options.
- All 22 validation OE questions are methodological/evidence-seeking. Forcing a
  diagnostic conclusion caused large Accuracy and Completeness losses.

## Outer-loop Cycle 3 Decision

Direction: **pivot** from global factor combinations to question-type routing.
The released Values/Local layout, all 30 Fixed tokens, and released generation
boundary remain fixed. MC receives only closed-option comparison discipline;
TF receives direct proposition-polarity consistency; OE receives speech-act and
evidence-coverage instructions that preserve uncertainty. Screening and
validation are now exposed development data (96 QA / 48 series). The original
48-QA / 24-series holdout remains untouched and confirmatory.
## Development Round 2

Six task-routed candidates were evaluated on the exposed 96-QA development
pool. The formal audit passed for 576 predictions and 1,326 new
`deepseek-v4-pro` scores. None passed the strict 10/10 gate.

- R2-04 ranked first at 7/10, worst -0.381, mean +0.205.
- Its MC deltas were +0.312/+0.294/+0.355 and its TF deltas were
  +0.409/+0.298/+0.576.
- Its OE deltas were Final -0.061, Accuracy -0.381, Completeness -0.060,
  Relevance +0.311.
- Fixed-role MC F0 and stable-selection MC F1 introduced no parsed
  correct-to-wrong MC decisions.
- TF P0 corrected fifteen Baseline errors but changed one real anomaly
  (`series_000132:1`) from correct to incorrect.
- OE-S0's negative “do not claim no further evidence is needed” language
  primed that exact phrase in five responses; Contract+S0 increased it to
  eight.
- The largest OE losses were diagnostic reversal, unsupported/out-of-window
  specifics, and incomplete treatment of boundary/evidence requests—not
  insufficient response length.

## Outer-loop Cycle 4 Decision

Direction: **deepen**. Preserve the proven MC components; use exact Baseline
routes as conservative OE/TF controls; and test short, positive OE obligations
that require qualitative observed evidence without eliciting unsupported
numeric decoding. Component-level artifact reuse is preregistered for prompts
that are byte-identical to already audited GPU/Judge runs.