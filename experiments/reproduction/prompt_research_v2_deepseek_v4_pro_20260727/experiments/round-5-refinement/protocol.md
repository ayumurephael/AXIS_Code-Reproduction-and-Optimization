# Round 5 protocol: conservative OE draft refinement

## Motivation

The full-284 OE-only RE2 run exposed a specific failure mechanism. Re-reading
the question sometimes replaces a correct Baseline anomaly conclusion with a
false-normal answer, while its intended benefit is mostly better question
coverage. Round 5 separates those two operations.

The released AXIS checkpoint first produces the unchanged Baseline draft. The
same checkpoint then receives the original question, that frozen draft, and
one short revision instruction. No weights, learned hints, value
serialization, decoding parameters, or Baseline draft are changed.

This is an inference-only two-pass prompt protocol. Its extra generation cost
must be reported separately from one-pass Prompt candidates.

## Frozen modes

All modes change OE only. MC and TF use exact Baseline predictions and scores.

1. `v2_r5_01_oe_preserve_cover`

   Revise only where needed to answer every part; preserve the draft's
   anomaly/normal conclusion and avoid unsupported exact details.

2. `v2_r5_02_oe_verbatim_or_add`

   Return a complete draft unchanged; otherwise add only missing requested
   information while preserving verdict and evidence claims.

3. `v2_r5_03_oe_conservative_edit`

   Edit for completeness and relevance only; preserve verdict, cover each
   requested part once, and remove repetition or unsupported details.

4. `v2_r5_04_oe_evidence_repair`

   Preserve verdict but allow the supplied time-series evidence to repair
   unsupported draft details and fill missing requested parts.

5. `v2_r5_05_oe_minimal_facets`

   Make the smallest possible revision and organize it around the conclusion,
   decisive evidence, and only specifically requested qualifications.

## Prompt placement

The author-compatible scaffold, Window, Per-Step Analysis, Overall Summary
Hints, and generation boundary stay unchanged. Only the text under
`### Question` is expanded:

```text
### Original Question
{question}

### Existing Draft Answer
{baseline_response}

### Revision Task
{mode_specific_revision_instruction}
```

## Evaluation

1. screening24 on the exposed search pool;
2. wide screen: at least three of four OE metrics non-lower, worst OE delta at
   least `-0.15`, positive mean OE delta, and at least one strict OE gain;
3. validation72 for every wide-screen survivor;
4. holdout48 only after validation, then pooled exposed-development audit;
5. a final route must still satisfy the user's exact criterion on formal
   paper140/full284: all ten Table-I metrics non-lower, at least three strict
   gains, and a changed OE branch with a strict OE gain.

All formal semantic scores use only `deepseek-v4-pro`.
