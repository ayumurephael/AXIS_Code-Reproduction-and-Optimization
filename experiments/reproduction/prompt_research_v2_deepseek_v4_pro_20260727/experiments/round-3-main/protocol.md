# Round 3 main protocol: balanced-evidence OE routing

Frozen before implementation of the new modes or any holdout Judge call for
their responses. This is part of the original prompt-search track; the
`new.md` experiment remains a separate supplement.

## Motivation

Round 1 establishes two different effects:

1. `v2_r1_03_oe_evidence_router` is the closest OE component to Pareto
   neutral. On pooled development96 it improves OE Accuracy by `+0.1224` but
   loses Completeness `-0.0655` and Relevance `-0.0448`.
2. The losses are concentrated in speech acts for which the rule is
   mismatched. A question that asks only for supporting evidence can be
   prematurely closed, while a question asking for several anomaly subtypes
   needs more coverage than the short rule supplies.

An exploratory, question-surface decomposition of the already judged
development records found a principled narrower class: questions that
explicitly request both supporting and counterevidence. For these balanced
questions, “state what the window supports, then address requested evidence”
matches the speech act instead of imposing a new diagnostic prior.

The decomposition is explicitly exploratory because development96 is exposed.
It may generate a candidate, but only the preregistered holdout and later
formal-set results can support advancement.

## Frozen shared intervention

For a matched OE question, insert exactly the already tested Round-1 rule at
the same location:

```text
### Answering Rule
First state what the supplied window and Per-Step Analysis currently support,
then address the requested evidence or boundary qualification. Do not replace
the supplied-window assessment with a generic tutorial.
```

All other prompt text remains byte-identical to Baseline: released opening,
Values → Per-Step Analysis → `Overall Summary Hints`, 30 Fixed tokens,
question-ending position, and released answer boundary. Unmatched records use
the literal Baseline prompt.

## Frozen lexical predicates

All predicates first require `question_type == "open_ended"`.

Definitions are case-insensitive:

- support term:
  `\bsupport(?:s|ed|ing)?\b`;
- counterevidence term:
  `\b(?:refute|refutes|refuted|refuting|challenge|challenges|challenged|challenging)\b`;
- multiplicity exclusion:
  `\bmultiple (?:types?|kinds?)\b`;
- boundary term:
  `\bboundar(?:y|ies)\b|\bedges?\b`;
- assessment term:
  `\b(?:whether|assess|assessment|determine|evaluate)\b`.

The multiplicity exclusion is mechanism-based: Round-1
`series_000043:1` asks the answer to distinguish multiple anomaly types and
loses Completeness under the short rule. It is not a record-ID exception.

## Three OE experiments

### `v2_r3_01_oe_balanced_support`

Match support AND counterevidence AND NOT multiplicity. This is the broad
balanced speech-act route.

### `v2_r3_02_oe_boundary_balanced`

R3-01 AND boundary. This tests whether the existing rule is specifically
useful when the question requests boundary qualification.

### `v2_r3_03_oe_assessment_balanced`

R3-01 AND assessment. This tests whether the route is useful when the
support/challenge request is attached to an explicit assessment.

## Locked existing components in the same outer loop

The original track also continues the two MC candidates that passed
screening24, validation72, and pooled development96:

- `v2_r1_06_mc_context_balanced`;
- `v2_r1_07_mc_content_output`.

They are evaluated unchanged on exposed holdout48. TF remains exact Baseline;
the failed Round-1 TF prompt is not repeated.

## Component reuse and no duplicate work

For R3-matched records, the rendered prompt must be byte-identical to
`v2_r1_03_oe_evidence_router`; therefore its existing GPU response and
deepseek-v4-pro score may be reused only after prompt-hash, response-hash,
record, mode, and rubric-dimension checks. Unmatched records reuse exact
Baseline predictions and scores.

The holdout responses for the Round-1 modes were generated in the earlier
preregistered full search-pool batch but have not been judged. Holdout48 has
two R3-01/R3-02 matches and one R3-03 match. Their outcomes were not inspected
before freezing this protocol.

## Evaluation and gate

1. Reassemble screening24, validation72, and development96 from exact
   components and verify all ten Table-I metrics against their split Baseline.
2. Judge the two unique matched holdout OE responses and the 34 MC candidate
   responses using only `deepseek-v4-pro`.
3. Analyze case-level transitions as diagnostics, not as a rejection gate.
4. Per the user's revised criterion, a route advances when all ten Table-I
   metrics are no lower than Baseline on every required view. At least three
   final-route metrics must be strictly higher.
5. A final comprehensive route must actually change OE responses and improve
   at least one OE metric on the formal evaluation set.
6. Only routes surviving the exposed views may be combined:
   one MC component + one OE component + exact Baseline TF.

No new coverage prompt, Evidence Contract, Fixed rename, answer prefill,
question repetition, or evidence reordering is introduced in this round.
