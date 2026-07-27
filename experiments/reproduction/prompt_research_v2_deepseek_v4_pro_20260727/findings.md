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

## Round 2 `new.md` compatibility findings

1. Even the short P1 compatibility rule is an active decision prior, not a
   neutral explanation of the inputs. It fixes some records and creates both
   false-anomaly and false-normal failures on others.
2. Value-before-Local rows improve human readability and some MC behavior, but
   textual one-to-one alignment is not learned causal alignment. OE Relevance
   falls by `0.6000` on screening24.
3. Question-first and evidence-last are strongly out of distribution for the
   released checkpoint. Their failure does not test the proposed retrained
   Local generator.
4. Cleaner parsing can coexist with worse Table-I scores. P4 raises OE strict
   parse rate to `85.7%` while all four OE metrics fall.

## Round 3 holdout findings

1. The Round-1 MC gains do not generalize to holdout48. Short context rules
   can make ordinary alternating variation sound anomalous and can shorten an
   otherwise complete correct explanation.
2. A development-derived lexical OE router is not a robust content router.
   Words such as “support,” “refute,” and “boundary” do not imply that the
   Baseline omitted those requested parts.
3. When the Baseline is already complete, an extra OE rule creates only
   downside: open `<think>` text, omitted qualifications, and unasked
   speculation.
4. The next OE intervention should control coverage and relevance without
   prescribing anomaly status or asserting a natural-language semantics for
   Local embeddings.

## Round 4–9 and formal full284 findings

### OE coverage and refinement failures

1. Five one-pass OE coverage prompts failed because a generic instruction
   cannot tell whether the Baseline already covers the requested parts.
   Additional coverage language often adds unsupported material or replaces
   a complete direct answer.
2. A two-pass `verbatim_or_add` prompt preserves anomaly status better than
   one-pass evidence reinterpretation, but still deletes boundary, comparison,
   method, and counterevidence details. Its holdout Accuracy gain therefore
   coexists with large Completeness and Relevance losses.
3. The released checkpoint is not a reliable answer-quality selector. In the
   A/B meta-task it often answers the original anomaly question instead of
   selecting a candidate.

### Deterministic repair findings

1. An unclosed `<think>` tag plus a missing Answer boundary identifies a real
   OE failure mode, but structure alone is insufficient: a clean revision can
   preserve a false anomaly verdict or delete useful content.
2. Requiring the Baseline and revision to preserve an explicit no-anomaly
   conclusion protects verdict direction but does not protect completeness.
3. The full284 Round-8 failures had candidate/Baseline word ratios of about
   0.40, 0.58, and 0.48. The single uniformly improved repair retained 0.606.
   A 60% word-retention guard excludes the three deletion failures and selects
   only `series_000111:0`.
4. This threshold is post-hoc because it was designed after inspecting
   full284 Round-8 outcomes. It is an engineering safeguard, not untouched-test
   generalization evidence.

### Final task-family synthesis

1. MC improves when anomaly status is decided before option matching.
   `status-then-shape` raises MC Final/Correctness/Reasoning by
   `+0.0930 / +0.1100 / +0.0532` on full284.
2. TF improves when qualitative RE2 and a consistent verdict rule are routed
   only to propositions with explicit negative cues. TF
   Final/Correctness/Justification rise by
   `+0.1365 / +0.1615 / +0.0989`.
3. OE is safest as a two-pass, deterministic exception handler rather than a
   universal rewrite. The one selected repair raises OE
   Final/Accuracy/Completeness by `+0.0071 / +0.0101 / +0.0101` and leaves
   Relevance unchanged at four decimals.
4. Combining the three disjoint components yields
   `prompt_final_round9_joint`: all ten displayed Table-I metrics are
   non-lower and nine are strictly higher. Two subset routes, MC+OE and TF+OE,
   also pass with six strict gains each.
5. Exact response and score reuse is essential. Unchanged records must not be
   regenerated or re-judged, because both the released checkpoint and
   `deepseek-v4-pro` introduce measurable run variance.

### Paired Judge-repeat limitation

The secondary paired repeat re-judged all 138 changed Baseline/candidate
records in one run. MC and TF remained positive on every component metric.
The single OE repair improved Final and Accuracy, tied Completeness at four
decimals, but lowered Relevance by `-0.0202`. Consequently all three routes
were 9/10 non-lower rather than 10/10 in the repeat.

This does not overwrite the locked primary formal result, but it changes the
strength of the claim. The Round-9 routes are locked-primary engineering
passes, not Judge-repeat-robust passes. Because OE changes only one of 99
answers, a one-point or two-point Judge change on that record moves the OE
mean by about 0.01–0.02. Future confirmation needs either a larger set of
independently improved OE cases or an untouched set with repeated paired
judging.
