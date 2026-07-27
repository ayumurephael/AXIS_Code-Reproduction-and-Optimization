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

## Round 1 screening findings

1. MC content errors are more responsive to short evidence arbitration than
   to rigid answer formatting: two severe wrong-option cases became correct,
   while the exact trained answer boundary was preserved.
2. OE has two separable failure axes. Evidence-use wording can improve the
   anomaly verdict, yet lower Relevance when the answer omits requested
   methods, boundary comparisons, or multiple anomaly subtypes.
3. A semantic explanation of Fixed tokens is not a harmless rename. Even a
   post-token note reduced all OE dimensions, so role renaming should not be
   reused with the released checkpoint.
4. Whole-statement TF guidance can correct negated normality decisions, but
   exact-prefix control reduced all TF metrics. Output cleanliness and answer
   quality are not interchangeable objectives.

## Round 1 validation and outer-loop findings

1. Both MC rules generalized: all three MC metrics improved on validation72
   and on pooled development96. The balanced content/context rule is stronger
   than the shorter content-to-letter rule.
2. No OE rule generalized across all four dimensions. The evidence router is
   close to neutral and improves Accuracy, but still loses Completeness and
   pooled Relevance.
3. OE failures are bidirectional. The same context instruction can correct a
   false-normal Baseline case and turn a correct anomaly decision into a
   false-normal answer. A textual alignment claim is not a decoder for learned
   Local vectors.
4. The TF whole-statement rule failed through polarity binding: several
   responses begin `False` while the explanation explicitly supports the
   truth of the negative statement.
5. The released checkpoint supports modest, family-specific semantic steering
   for MC. Larger causal reordering of Question, Fixed, Value, and Local should
   be treated as an old-checkpoint compatibility test, not as a clean test of
   the retrained information-flow proposal in `new.md`.
