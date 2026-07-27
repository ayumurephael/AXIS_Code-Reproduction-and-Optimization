# Findings

## Established before the new experiments

1. Long shared Evidence Contracts are not reliably compositional for the released 7B checkpoint. They compete with the learned Fixed-token representation and change task interpretation.
2. Explicit scale/alignment explanations increase numeric salience and have caused extra rescaling, step hallucinations, and out-of-window claims.
3. OE can gain evidence coverage while MC/TF lose correctness; a higher Judge style/coverage score is not proof of better anomaly decisions.
4. TF failures often involve polarity inversion or a label-rationale contradiction. Longer anti-error rules have also suppressed legitimate anomalies.
5. MC “stability” instructions can freeze an early false positive. Prior routed MC candidates each introduced a correct-to-wrong transition on the exposed 48-QA check.
6. Renaming `Overall Summary Hints` changes the textual neighborhood of 30 learned tokens. Previous minimal routes did not isolate this factor.
7. The old 24/72/48 internal split is fully exposed. It remains useful for falsification and failure analysis, not for claims of untouched generalization.

## Current design principles

- Preserve the checkpoint’s trained prompt scaffold unless a factor is explicitly being tested.
- Use one short rule per task family; do not impose an MC protocol on OE/TF or vice versa.
- Optimize decisions first and formatting second.
- For MC, bind semantic option text before the letter.
- For TF, enforce one coherent polarity without adding a blanket anomaly prior.
- For OE, cover the requested speech act without prescribing a diagnosis.
- Analyze paired correct→wrong and wrong→correct cases, not only aggregate Judge means.
- Reject candidates with hidden family-level regressions even when their ten-metric mean rises.

## Round 1 findings

1. Preserving the old Fixed-token header matters. Four scaffold-preserving MC interventions passed all ten screening dimensions because unchanged OE/TF components were exactly Baseline.
2. MC question re-reading is the strongest new factor: it fixed both tested Baseline MC errors and raised all three MC metrics.
3. Adding semantic binding to RE2 did not improve decisions over RE2 alone and reduced MC Reasoning by 0.250 relative to RE2; one response also invented `-6.40`.
4. Pointwise and semantic-selection rules are weaker than RE2 but remained decision-safe on screening.
5. TF RE2 has a real accuracy/robustness trade-off: four wrong→correct versus one correct→wrong, with the failure caused by reconstructing a nonexistent stable numeric range.
6. Static TF rules over-focus on textual clause verification and can suppress evidence encoded in the learned tokens.
7. OE directness is not diagnostic correctness. Shorter answers with fewer unsupported numbers still denied real boundary anomalies and lost Judge Accuracy/Relevance.
8. OE RE2 raises coverage but repeats question framing, lengthens outputs, and increases scaled-number interpretation errors; the result is higher Completeness but lower Relevance.
9. Decision/report decoupling is too broad for this checkpoint: it changed all task families and lost six dimensions.

## Current questions

- Do MC RE2, pointwise checking, and semantic binding generalize to the exposed 72-QA validation pool without correct→wrong transitions?
- Does the TF RE2 net gain replicate, and can it satisfy a strict zero-regression gate?
- If MC-only routes remain robust, can they pass the exposed 48-QA robustness check that defeated the prior header-renamed routes?

## Result status

Round 1 is complete and audited. Results are exploratory because the split was exposed. No paper140 candidate output has been generated.


## Round 2 findings

1. The strongest validation route reached only 9/10; the screening 10/10 MC RE2 result did not generalize.
2. MC pointwise is the safest decision mechanism so far: zero correct?wrong and one wrong?correct on validation. Its weakness is explanation quality, not option selection.
3. ?Brief reason? is not neutral for this checkpoint. It removes recovery, boundary, and alternative-discrimination evidence that the Reasoning rubric rewards.
4. Exact numeric narration is a repeated failure surface (`1.03`?`10.30`, about `?1.77`?`?21.70`). Qualitative shape/direction/persistence should be tested without explaining scale.
5. RE2 can reconstruct the evidence incorrectly: MC `series_000103:1` changed a spike-plus-drift C into a normal-range A.
6. TF RE2 has a strong net gain (nine TF wrong?correct) but lacks a reliable answer boundary. `series_000109:1` was semantically True yet parsed False because a later ?no? was matched first; its numeric hallucination also lowered Judge quality.
7. A task-specific final verdict is justified for TF, but a broad answer-first protocol remains unjustified for MC/OE.

## Current questions after Round 2

- Can qualitative support lift MC Reasoning to nonnegative while preserving pointwise zero-regression decisions?
- Does salient-event/aftermath matching repair missed compound anomalies without creating false positives?
- Can an explicit final TF verdict retain RE2 gains, and does suppressing exact numbers repair its weakest justification case?
