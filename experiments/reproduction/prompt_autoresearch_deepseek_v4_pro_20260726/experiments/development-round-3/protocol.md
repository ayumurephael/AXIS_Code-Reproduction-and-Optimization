# Development round 3 protocol

## Status

Preregistered before implementation, prompt rendering, inference, or Judge
calls for any new Round-3 component.

## Objective

Round 2 showed that MC and TF can improve independently, while every new OE
instruction reduced at least one OE metric. Round 3 therefore includes two
conservative Pareto controls that change only MC, two MC+TF routes with exact
Baseline OE, three isolated positive OE rules, and one historical comparison.

## Data and isolation

- Exposed development manifest:
  `../../data/manifests/development96.json` (96 QA / 48 series; 34 MC, 29 OE,
  33 TF).
- Record SHA-256:
  `bf1f164c17c976c1fec854944c75e89b4f81f2e3d6ab77f66aa4f2293e381d84`.
- Series SHA-256:
  `d20f2d942421dd2205c121ee5d45352ec8bc1105a28d30250c12c3f9f89d2c2b`.
- The 48-QA / 24-series holdout and `paper140` remain untouched.

## Component reuse policy

Table-I metrics decompose by question type. A route component whose rendered
model prompt is byte-identical to an already audited development component
will reuse that component's GPU prediction and `deepseek-v4-pro` Judge score
after verifying record coverage and prompt SHA-256. Only the three new OE
components require new inference and Judge calls. This removes meaningless
GPU/Judge nondeterminism between identical prompts and does not reuse a score
across different prompt text, response text, record, or rubric dimension.

Reused sources:

- MC-F0: Round-2 `route_r2_01_minimal` MC rows.
- MC-F1: Round-2 `route_r2_02_mc_stable` MC rows.
- TF-B0 and OE-B0: audited development Baseline rows.
- TF-P0: Round-2 `route_r2_01_minimal` TF rows.
- OE-C0: Round-2 `route_r2_04_oe_old_contract` OE rows.

## New OE rules

### OE-Q1 — positive evidence obligation

```text
### Task Rule
Answer the open-ended question in one coherent paragraph. Include: (1) a direct assessment of this window; (2) two or three observed qualitative features that support the assessment, including shape and persistence or recovery; and (3) the boundary/context limitation or additional evidence requested by the question. State location as beginning, middle, or end and describe magnitude relatively unless an exact value or step is unambiguous in the supplied input.
```

### OE-Q2 — named-event test

```text
### Task Rule
Treat the event or pattern named in the question as the behavior to evaluate, not as a predetermined anomaly label. Compare its abruptness, isolation, persistence or recovery, and boundary position with the rest of the supplied window. Report the evidence supporting your conclusion and the strongest evidence against it or remaining uncertainty.
```

### OE-Q3 — compact qualitative synthesis

```text
### Task Rule
First answer the assessment or analysis requested by the question. Then apply it to this window using qualitative evidence: local contrast, persistence or recovery, and boundary position. State what supports the conclusion and what observation would refute it or require outside-window context. Use beginning, middle, or end for location when exact step alignment is not explicit.
```

All three new OE components retain the exact released opening, original
`Overall Summary Hints` label, Values/Local layout, 30 Fixed tokens, and
generation boundary. They add only the displayed OE Task Rule.

## Eight candidates

| Mode | MC | TF | OE | Purpose |
|---|---|---|---|---|
| `route_r3_01_mc_f0_safe` | MC-F0 | TF-B0 | OE-B0 | Conservative fixed-role MC Pareto control |
| `route_r3_02_mc_f1_safe` | MC-F1 | TF-B0 | OE-B0 | Conservative stable-MC Pareto control |
| `route_r3_03_mc_f0_tf_p0` | MC-F0 | TF-P0 | OE-B0 | Improve MC and TF; preserve OE |
| `route_r3_04_mc_f1_tf_p0` | MC-F1 | TF-P0 | OE-B0 | Stable MC plus TF polarity; preserve OE |
| `route_r3_05_oe_q1` | MC-F0 | TF-P0 | OE-Q1 | Positive evidence coverage |
| `route_r3_06_oe_q2` | MC-F0 | TF-P0 | OE-Q2 | Preserve question-named event without accepting its label |
| `route_r3_07_oe_q3` | MC-F0 | TF-P0 | OE-Q3 | Compact qualitative evidence synthesis |
| `route_r3_08_historical` | MC-F0 | TF-B0 | OE-C0 | Historical paper-favorable composite |

## Pre-data predictions

- R3-01 and R3-02 should pass 10/10 on development because their OE/TF
  components are exactly Baseline and their MC components improved all MC
  metrics with no correct-to-wrong MC flips.
- R3-03/R3-04 should improve both closed task families, but retain the known
  TF `series_000132:1` regression.
- Q1 may improve Completeness but risks instruction load.
- Q2 directly targets `series_000012:0` but may over-trust a misleading
  question premise.
- Q3 is the shortest new OE rule and is expected to be the safest OE
  improvement attempt.
- R3-08 is expected to fail development OE Accuracy but is retained to measure
  split sensitivity of the historically paper-favorable Contract.

## New work and audit

- New inference: three modes × 96 records = 288 GPU predictions. Only their
  87 OE rows are new treatment components.
- New Judge work: 87 OE rows × 3 dimensions = 261
  `deepseek-v4-pro` scores.
- Assemble eight candidate artifacts from audited components, producing
  8 × 96 predictions and 8 × 221 scores, plus Baseline.
- Fail closed on prompt-hash mismatch, record mismatch, duplicate/missing
  component keys, provider/model mismatch, or unapproved scoring method.

## Locked gate

Rank by nonnegative dimension count, minimum delta, then mean delta. A
candidate is holdout-eligible only if all 10 Table-I deltas and all three task
Final deltas are nonnegative and it has no unexplained increase in parsed
correct-to-wrong MC/TF decisions. Advance the best two eligible candidates.
If fewer than two qualify, do not touch holdout and begin another 5–10
experiment development batch.
