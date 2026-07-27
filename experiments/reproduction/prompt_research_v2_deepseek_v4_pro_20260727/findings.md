# Findings

## Baseline facts

1. `paper140` and `full284` are different estimands and must never share one
   row label without an explicit dataset column.
2. The current full-284 Baseline reproduces the historical Baseline response
   exactly on all 140 overlapping records.
3. The non-paper 144 QA are substantially harder for MC and TF, which explains
   most of the later aggregate drop.
4. Probability-weighted score-token readout does not make a reasoning Judge
   deterministic. The Judge can take a different reasoning path before
   emitting a different score distribution for an identical score prompt.

## Established prompt constraints

1. Renaming `Overall Summary Hints` before the 30 trained Fixed tokens changes
   their textual neighborhood and is not behaviorally neutral.
2. Long Evidence Contracts improve some OE dimensions but have repeatedly
   reduced MC and TF performance.
3. Strict output schemas can improve parseability while reducing reasoning or
   answer quality.
4. RE2 is a real perturbation, but its large small-split gains did not
   generalize to all ten metrics on full284.
5. The most common severe Baseline failure is evidence conflict: the answer
   trusts a smooth-looking displayed window and ignores contextual information
   encoded by Per-Step Analysis.

## Current hypothesis

A short, balanced statement of how to use `Per-Step Analysis`, placed after
the unchanged Local and Fixed token blocks, can improve contextual anomaly
recognition without the distribution shift and numeric salience caused by a
full Evidence Contract. Because Table-I scoring is family-specific, the rule
should be tested separately for OE, MC, and TF before exact-component
combination.
