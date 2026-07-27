# Round 4 main protocol: OE coverage without evidence reinterpretation

Frozen before implementation, GPU generation, or Judge calls for these modes.
This continues the original prompt-search track. Exact previously rejected
prompts are not repeated.

## Motivation from failed cases

Round 1 evidence-use rules changed anomaly decisions bidirectionally. Round 3
showed that a lexical question router did not solve the problem: on holdout,
the selected Baseline answers already covered the requested evidence, while
the extra rule introduced open `<think>` text, omitted qualifications, or
added speculation.

The remaining OE opportunity is instruction coverage and relevance, not a
new anomaly-status prior. Round 4 therefore never claims that Local or Fixed
tokens have a natural-language meaning and never tells the model that a
particular pattern is normal or anomalous.

## Invariants

All five modes:

- change only `open_ended` prompts;
- preserve the released opening exactly;
- preserve Values → `Per-Step Analysis` → `Overall Summary Hints`;
- preserve all 30 Fixed tokens and their original header;
- preserve the question-ending layout and released answer boundary;
- insert one short `### Answering Rule` immediately before `### Question`;
- do not rename roles, interleave values/tokens, reorder evidence, repeat the
  question, add EOS or `Answer:` prefill, or require JSON/XML labels.

MC and TF are exact Baseline in this component experiment.

## Five frozen modes

### `v2_r4_01_oe_requested_parts`

```text
Answer every distinct part of the question. Give the requested conclusion or
assessment first, followed by only the evidence and qualifications needed for
those parts.
```

This is the shortest coverage-only intervention.

### `v2_r4_02_oe_hypothesis_check`

```text
Treat any anomaly description in the question as a hypothesis, not a fact.
State whether the supplied evidence supports or refutes it, then answer every
remaining requested part without unrelated speculation.
```

This targets leading premises such as “observed spikes” without prescribing
the verdict.

### `v2_r4_03_oe_silent_checklist`

```text
Before responding, silently identify the question's requested parts. In the
answer, address each part exactly once with a direct conclusion and concise
supplied-window evidence. Do not add unasked possibilities.
```

The checklist is internal; no checklist markup is requested in the output.

### `v2_r4_04_oe_evidence_scope`

```text
Base the answer only on the supplied evidence. Do not mention tokens or prompt
mechanics, invent exact values, or speculate beyond the window. Directly
answer every requested part.
```

This targets unsupported numbers, token meta-talk, and outside-window
speculation observed in prior failures.

### `v2_r4_05_oe_compact_facets`

```text
Use one compact paragraph: give the conclusion or assessment, the decisive
observed pattern, then any specifically requested boundary, subtype, method,
or counterevidence qualification. Omit categories the question does not ask
for.
```

This supplies an adaptive output order without a rigid universal schema.

## Evaluation and gate

1. Generate all five OE components for the 144 non-paper QA pool on the
   authorized GPU; reuse exact Baseline MC/TF predictions.
2. Form the existing screening24 and validation72 views and judge only changed
   OE responses with `deepseek-v4-pro`.
3. Use the user's wide gate: a candidate can advance if it is plausibly
   capable of all-ten-nonlower after considering small-split Judge noise, but
   candidates with the same OE dimension decreasing on both views are
   rejected.
4. Report screening24, validation72, and pooled development96 separately.
5. Inspect the largest Accuracy, Completeness, and Relevance gains/losses and
   open-`<think>`, unsupported-number, response-length, and requested-subpart
   diagnostics.
6. Perform an outer-loop synthesis after these five experiments.
7. Only a broad survivor may be tested on exposed holdout48. The formal
   paper140 set remains untouched until a complete MC+OE+TF route survives
   all exposed views.
