# Operational amendment — byte-identical component reuse

Recorded after GPU generation and before aggregate results or candidate ranking.

## Trigger

All 12 routes were generated for all 24 records, producing 288 predictions. Inspection showed that routes whose task-family prompt is byte-identical to Baseline can nevertheless emit different text because repeated CUDA generation is not guaranteed bitwise deterministic. Treating those differences as prompt effects would violate the intended one-factor comparison.

The first all-route Judge process was stopped after 89/660 dimension scores. No aggregate result, ranking, or candidate selection had been computed.

## Frozen correction

Use one canonical GPU prediction per unique `(prompt component, question type, record)`:

- Baseline task components: reuse the previously audited split-Baseline prediction and Judge score.
- MC semantic binding: source `lit_r1_01_mc_semantic_bind`.
- MC pointwise checking: source `lit_r1_02_mc_pointwise`.
- MC RE2: source `lit_r1_03_mc_re2`.
- TF minimal polarity: source `lit_r1_04_tf_minimal`.
- TF clause checking: source `lit_r1_05_tf_clause`.
- TF RE2: source `lit_r1_06_tf_re2`.
- OE direct answer: source `lit_r1_07_oe_direct`.
- OE RE2: source `lit_r1_08_oe_re2`.
- decision/report decoupling: source `lit_r1_11_decoupled` for its three task components.
- MC semantic binding + RE2: source `lit_r1_12_mc_bind_re2`.

The triplet modes reuse the corresponding single-task component because their rendered prompts are byte-identical. Every equivalence is verified by SHA-256 over a fixed representative rendered prompt before assembly.

## Workload and audit

- Raw generated predictions remain archived: 288.
- Unique non-Baseline components to score: 97 predictions / 215 Judge dimensions.
- Assembled candidate matrix: 288 predictions / 660 Judge dimensions.
- The 89 partial scores may be reused only if their key belongs to the canonical 97-component set; all other partial rows are excluded.
- The final assembly must fail closed on prompt-hash mismatch, source-record mismatch, missing dimensions, duplicate keys, provider/model mismatch, or invalid score method.

## Interpretation

This amendment does not change any candidate prompt, response chosen for a unique component, rubric, Judge model, split, ranking rule, or advancement gate. It removes redundant calls and ensures that a task family declared “Base” is exactly the audited Baseline rather than a second stochastic regeneration.
