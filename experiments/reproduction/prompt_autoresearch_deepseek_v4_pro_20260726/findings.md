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

No new search experiments have been run. The diagnostic bootstrap is complete and proposes Pareto-v1 plus an orthogonal, series-level search on the 144 non-paper140 QA.
