# Development round 2 protocol

## Status

Preregistered before implementation, prompt rendering, inference, or Judge calls
for any Round-2 candidate.

## Rationale

Validation round 1 showed cross-task semantic interference. MC benefits from a
stable closed-option decision, TF needs direct proposition-polarity
consistency, and OE needs coverage of the question's speech act. Round 2 tests
question-type-conditioned prompt routing while preserving the trained evidence
layout.

## Data and isolation

- Exposed development manifest: `../../data/manifests/development96.json`.
- It is the frozen union of the already exposed screening and validation
  manifests: 96 QA / 48 series, with 34 MC, 29 OE, and 33 TF.
- Record SHA-256:
  `bf1f164c17c976c1fec854944c75e89b4f81f2e3d6ab77f66aa4f2293e381d84`.
- Series SHA-256:
  `d20f2d942421dd2205c121ee5d45352ec8bc1105a28d30250c12c3f9f89d2c2b`.
- The 48-QA / 24-series internal holdout and `paper140` remain untouched.

## Invariants

Every candidate:

- uses the released author checkpoint only;
- retains the released opening, separate Values and Per-Step Analysis blocks,
  `(x * 100):.0f` serialization, all 30 Fixed tokens, and released generation
  boundary;
- renames only `Overall Summary Hints` to
  `Learned Task Guidance/Shared Task-Control Tokens`;
- is evaluated with series batching, three approved GPUs, beam size 5, and
  `--skip-loss`;
- is formally scored only by `deepseek-v4-pro` using the locked G-Eval/Judge
  code and fail-closed audit.

## Exact routed instruction components

The released fixed-role prompt is the common skeleton. A routed `### Task Rule`
section is inserted after `### Contextual Hints` and before `### Question`,
unless the component is `MC-F0` or `OE-C0` as defined below.

### MC-F0 — fixed-role only

No additional text.

### MC-F1 — single stable option

```text
### Task Rule
Compare the complete meaning of every option with the supplied evidence. Select exactly one best-supported option, state that option once, and keep the explanation consistent with it. Do not revise the selected option or introduce a second answer.
```

### TF-P0 — direct proposition polarity

```text
### Task Rule
Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.
```

### TF-P1 — boundary plus direct proposition polarity

```text
### Task Rule
Use only evidence inside the half-open window [{start}, {end}); do not invent or cite a step outside it. Judge the truth of the complete proposition exactly as written, preserving every negation such as "no evidence" and "does not". Answer True when the evidence supports the proposition as written; answer False when any essential clause is contradicted. Before finishing, verify that the label and the first explanatory sentence express the same truth value.
```

### OE-S0 — speech-act coverage

```text
### Task Rule
First determine whether the question asks for a diagnosis, an assessment method, or evidence that would support or challenge an assessment. Address the supplied window before general methods. Cover every requested part: the observed shape and location; what it currently supports or challenges relative to ordinary variation; relevant boundary, persistence, or recovery uncertainty; and only the requested additional evidence or indicators. For a methodological or evidence-seeking question, do not deny its premise merely to force a normal/anomalous verdict, and do not claim that no further evidence is needed unless the question and supplied evidence justify that claim.
```

### OE-C0 — historical old Contract, OE only

Insert the exact historical five-item `EVIDENCE_CONTRACT` before
`### Time Series Data`, with no OE Task Rule. This deliberately tests whether
the previous formal OE gain transfers when the Contract cannot interfere with
MC or TF. Its known inaccurate “same row” wording is retained as a controlled
historical mechanism, not endorsed as a factual description.

### OE-C1 — historical old Contract plus speech-act coverage

Insert the exact `OE-C0` Contract before the data and append the exact `OE-S0`
Task Rule before the question.

## Six candidate routes

| Mode | MC route | TF route | OE route | Purpose |
|---|---|---|---|---|
| `route_r2_01_minimal` | MC-F0 | TF-P0 | OE-S0 | Minimal task routing |
| `route_r2_02_mc_stable` | MC-F1 | TF-P0 | OE-S0 | Isolate stable MC selection |
| `route_r2_03_tf_boundary` | MC-F0 | TF-P1 | OE-S0 | Isolate TF boundary discipline |
| `route_r2_04_oe_old_contract` | MC-F0 | TF-P0 | OE-C0 | Test historical OE mechanism only |
| `route_r2_05_oe_contract_coverage` | MC-F0 | TF-P0 | OE-C1 | Test Contract × OE coverage |
| `route_r2_06_full_routed` | MC-F1 | TF-P1 | OE-C1 | Combine the strongest hypothesized route |

No candidate may be altered after inference begins.

## Pre-data predictions

- MC-F0 is the conservative choice because fixed-role improves all MC metrics
  on the formal test and validation; MC-F1 may improve answer stability but can
  reduce reasoning if it over-constrains generation.
- TF-P0 should directly remove the observed `False` + “no anomaly” polarity
  contradiction. TF-P1 may additionally preserve boundary anomalies, but its
  extra clause can introduce instruction competition.
- OE-S0 should restore Completeness/Relevance on methodological questions
  without forcing diagnostic closure.
- OE-C0 may reproduce the old Contract's formal OE gain, but can fail on the
  exposed development pool because of the false row-alignment claim.
- OE-C1 tests whether coverage rescues that mechanism or merely compounds
  instruction load.

## Work and audit

- New predictions: 6 modes × 96 records = 576.
- Judge dimensions per mode: `34*2 + 29*3 + 33*2 = 221`.
- New Judge scores: 1,326.
- The development Baseline predictions/scores are merged from the already
  audited screening and validation Baseline rows; hashes and record coverage
  must be re-audited. They are not re-judged.
- Fail closed on missing/duplicate keys, unexpected modes, provider/model
  mismatch, invalid prompt hashes, or an unapproved scoring method.

## Locked development gate

Rank by: (1) number of nonnegative Table-I deltas, (2) minimum delta, and
(3) mean of all ten deltas versus the merged development Baseline.

A candidate is eligible for holdout only if **all 10/10 dimensions are
nonnegative**, all three task Finals are nonnegative, and it introduces no
unexplained increase in correct-to-wrong parsed MC/TF decisions. Advance the
best two eligible candidates. If fewer than two qualify, do not touch holdout;
perform another 5–10-experiment outer-loop batch on the exposed development
pool.