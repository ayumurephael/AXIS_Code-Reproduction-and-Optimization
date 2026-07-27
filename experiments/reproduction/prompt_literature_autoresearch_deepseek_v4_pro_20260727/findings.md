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

## Pending questions

- Does semantic binding help MC when the original Fixed-token header is preserved?
- Does exact question re-reading help any task type on this checkpoint, or merely add distribution shift?
- Can a very short TF rule fix contradictions without changing otherwise correct decisions?
- Does a direct OE rule improve relevance/accuracy rather than just verbosity/completeness?

## Result status

Round 1 is preregistered but not yet executed. No improvement claim is made.
