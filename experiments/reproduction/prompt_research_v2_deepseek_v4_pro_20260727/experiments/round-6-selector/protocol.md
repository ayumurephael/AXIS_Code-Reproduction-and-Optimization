# Round 6 protocol: conservative OE response selection

## Motivation

Round 5 preserves the Baseline anomaly verdict but can still shorten away
useful evidence. Its closest mode improves OE Final and Accuracy on
validation72 while losing small amounts of Completeness and Relevance.

Round 6 does not generate another rewritten answer. The checkpoint receives
the original question, Candidate A (the frozen Baseline draft), Candidate B
(the frozen Round-5 `verbatim_or_add` revision), and a short selection rule.
It outputs an A/B selection. The evaluated response is then copied exactly
from the selected candidate, and its existing formal Judge component is
reused. An invalid or ambiguous selection defaults to Candidate A.

This is an inference-only multi-pass prompt protocol. Its additional
generation and latency must be reported separately from one-pass Prompts.

## Frozen modes

1. `v2_r6_01_oe_missing_part_selector`

   Select B only when it covers an explicit requested part that A omits,
   without dropping any conclusion, evidence, or qualification.

2. `v2_r6_02_oe_table_i_selector`

   Compare factual accuracy, completeness, and relevance. Select B only when
   it is no worse in all three and clearly better in at least one; ties use A.

3. `v2_r6_03_oe_content_guard_selector`

   Select B only when it preserves every substantive claim from A while
   removing only markup, repetition, or unrelated material.

All modes change OE only. MC and TF use exact Baseline components during
component testing.

## Selection prompt

```text
### Original Question
{question}

### Candidate A — Baseline Draft
{baseline_response}

### Candidate B — Conservative Revision
{round5_response}

### Selection Task
{mode_specific_rule}

Output exactly `Selection: A` or `Selection: B`. Do not write a new answer.
```

The author-compatible Window, Local/Fixed hints, their order, and all 30
Fixed tokens remain unchanged.

## Evaluation

- GPU selector inference on screening24 and validation72;
- no new Judge call for a copied response: Baseline and Round-5 component
  scores are reused exactly;
- wide screening followed by validation and holdout;
- the user's final gate remains all ten Table-I metrics non-lower, at least
  three strict gains, and a changed OE branch with a strict OE gain.

This is the final three-mode outer loop under the registered 30-mode search
budget.
