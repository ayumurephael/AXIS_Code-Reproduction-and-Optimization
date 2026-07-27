# Literature Round 5 — conservative decision calibration

## Outcome

Round 5 completed all frozen GPU inference and `deepseek-v4-pro` Judge work. No candidate passed every preregistered gate. The registered 30-new-mode stopping rule therefore fired, no winner was locked, and `paper140` candidate outputs were not generated.

This is stronger than an average-score rejection: several routes had all ten nonnegative aggregate deltas, but each failed either a held-out split or the zero correct→wrong guard. The released Baseline remains the only verified strict no-regression deployment prompt.

## Frozen execution

- Checkpoint SHA-256: `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- Source commit used by GPU inference: `4d3ace60c036cdb231a4c5aeafeec3cdf030aace`.
- Inference: authorized GPU only, one selected GPU, series batching, beam size 5, `--skip-loss`; no local GPU and no training.
- Judge: `deepseek-v4-pro` only, OpenAI-compatible official endpoint recorded as `api.deepseek.com` in every new score row.
- Development: 192 raw predictions; 83 routed components; 68 newly judged predictions / 136 dimensions (135 top-logprob, 1 exact-20).
- Holdout: 144 raw predictions; 44 routed components; 44 newly judged predictions / 88 dimensions (all top-logprob).
- Audited assembled totals: development 576 predictions / 1326 scores; holdout 288 predictions / 666 scores.
- Exact reuse: unchanged MC/OE/TF records reuse canonical Baseline responses and scores; development TF reuse comes from the already audited Round-4 component.

## Gate matrix

| Mode | development96 | screening24 | validation72 | exposed holdout48 | Eligible |
|---|---:|---:|---:|---:|---:|
| lit_r4_01_tf_neg_re2 | PASS (10/10, c→w=0) | PASS (10/10, c→w=0) | PASS (10/10, c→w=0) | FAIL (10/10, c→w=1) | NO |
| lit_r5_01_mc_structured_guard | FAIL (9/10, c→w=1) | PASS (10/10, c→w=0) | FAIL (7/10, c→w=1) | PASS (10/10, c→w=0) | NO |
| lit_r5_02_mc_status_then_shape | FAIL (10/10, c→w=3) | FAIL (10/10, c→w=1) | FAIL (10/10, c→w=2) | FAIL (10/10, c→w=1) | NO |
| lit_r5_03_joint_structured_tf_neg_re2 | FAIL (9/10, c→w=1) | PASS (10/10, c→w=0) | FAIL (7/10, c→w=1) | FAIL (10/10, c→w=1) | NO |
| lit_r5_04_joint_status_tf_neg_re2 | FAIL (10/10, c→w=3) | FAIL (10/10, c→w=1) | FAIL (10/10, c→w=2) | FAIL (10/10, c→w=2) | NO |

A cell is `PASS` only when all ten split-level Table-I deltas are nonnegative **and** no Baseline-correct parsed MC/TF decision becomes wrong. No row is eligible.

## development96: delta versus exact split Baseline

| Mode | Nonneg. | Worst Δ | Mean Δ | c→w | w→c | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lit_r4_01_tf_neg_re2 | 10/10 | +0.0000 | +0.0756 | 0 | 12 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2609 | +0.3136 | +0.1818 |
| lit_r5_01_mc_structured_guard | 9/10 | -0.0412 | +0.0146 | 1 | 3 | +0.0700 | +0.1176 | -0.0412 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_02_mc_status_then_shape | 10/10 | +0.0000 | +0.0729 | 3 | 4 | +0.2588 | +0.2941 | +0.1765 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_03_joint_structured_tf_neg_re2 | 9/10 | -0.0412 | +0.0903 | 1 | 15 | +0.0700 | +0.1176 | -0.0412 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2609 | +0.3136 | +0.1818 |
| lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.0000 | +0.1486 | 3 | 16 | +0.2588 | +0.2941 | +0.1765 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2609 | +0.3136 | +0.1818 |

## screening24: delta versus exact split Baseline

| Mode | Nonneg. | Worst Δ | Mean Δ | c→w | w→c | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lit_r4_01_tf_neg_re2 | 10/10 | +0.0000 | +0.0867 | 0 | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.3111 | +0.4444 | +0.1111 |
| lit_r5_01_mc_structured_guard | 10/10 | +0.0000 | +0.2725 | 0 | 2 | +0.9750 | +1.1250 | +0.6250 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_02_mc_status_then_shape | 10/10 | +0.0000 | +0.1337 | 1 | 2 | +0.4625 | +0.5000 | +0.3750 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_03_joint_structured_tf_neg_re2 | 10/10 | +0.0000 | +0.3592 | 0 | 6 | +0.9750 | +1.1250 | +0.6250 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.3111 | +0.4444 | +0.1111 |
| lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.0000 | +0.2204 | 1 | 6 | +0.4625 | +0.5000 | +0.3750 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.3111 | +0.4444 | +0.1111 |

## validation72: delta versus exact split Baseline

| Mode | Nonneg. | Worst Δ | Mean Δ | c→w | w→c | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lit_r4_01_tf_neg_re2 | 10/10 | +0.0000 | +0.0715 | 0 | 8 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2421 | +0.2646 | +0.2083 |
| lit_r5_01_mc_structured_guard | 7/10 | -0.2461 | -0.0647 | 1 | 1 | -0.2085 | -0.1923 | -0.2461 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_02_mc_status_then_shape | 10/10 | +0.0000 | +0.0542 | 2 | 2 | +0.1962 | +0.2308 | +0.1154 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_03_joint_structured_tf_neg_re2 | 7/10 | -0.2461 | +0.0068 | 1 | 9 | -0.2085 | -0.1923 | -0.2461 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2421 | +0.2646 | +0.2083 |
| lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.0000 | +0.1257 | 2 | 10 | +0.1962 | +0.2308 | +0.1154 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2421 | +0.2646 | +0.2083 |

## exposed holdout48: delta versus exact split Baseline

| Mode | Nonneg. | Worst Δ | Mean Δ | c→w | w→c | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lit_r4_01_tf_neg_re2 | 10/10 | +0.0000 | +0.0638 | 1 | 4 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2000 | +0.1250 | +0.3125 |
| lit_r5_01_mc_structured_guard | 10/10 | +0.0000 | +0.1665 | 0 | 1 | +0.5471 | +0.5294 | +0.5882 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_02_mc_status_then_shape | 10/10 | +0.0000 | +0.0706 | 1 | 1 | +0.2353 | +0.2353 | +0.2353 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| lit_r5_03_joint_structured_tf_neg_re2 | 10/10 | +0.0000 | +0.2302 | 1 | 5 | +0.5471 | +0.5294 | +0.5882 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2000 | +0.1250 | +0.3125 |
| lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.0000 | +0.1343 | 2 | 5 | +0.2353 | +0.2353 | +0.2353 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.2000 | +0.1250 | +0.3125 |

## Absolute scores — development96

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.8824 | 3.8824 | 3.8824 | 3.2293 | 3.4483 | 2.7932 | 3.4828 | 3.0518 | 3.1470 | 2.9091 |
| lit_r4_01_tf_neg_re2 | 3.8824 | 3.8824 | 3.8824 | 3.2293 | 3.4483 | 2.7932 | 3.4828 | 3.3127 | 3.4606 | 3.0909 |
| lit_r5_01_mc_structured_guard | 3.9524 | 4.0000 | 3.8412 | 3.2293 | 3.4483 | 2.7932 | 3.4828 | 3.0518 | 3.1470 | 2.9091 |
| lit_r5_02_mc_status_then_shape | 4.1412 | 4.1765 | 4.0588 | 3.2293 | 3.4483 | 2.7932 | 3.4828 | 3.0518 | 3.1470 | 2.9091 |
| lit_r5_03_joint_structured_tf_neg_re2 | 3.9524 | 4.0000 | 3.8412 | 3.2293 | 3.4483 | 2.7932 | 3.4828 | 3.3127 | 3.4606 | 3.0909 |
| lit_r5_04_joint_status_tf_neg_re2 | 4.1412 | 4.1765 | 4.0588 | 3.2293 | 3.4483 | 2.7932 | 3.4828 | 3.3127 | 3.4606 | 3.0909 |

## Absolute scores — exposed holdout48

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.8647 | 3.8824 | 3.8235 | 2.8762 | 2.7033 | 2.6002 | 3.4000 | 3.4125 | 3.4375 | 3.3750 |
| lit_r4_01_tf_neg_re2 | 3.8647 | 3.8824 | 3.8235 | 2.8762 | 2.7033 | 2.6002 | 3.4000 | 3.6125 | 3.5625 | 3.6875 |
| lit_r5_01_mc_structured_guard | 4.4118 | 4.4118 | 4.4118 | 2.8762 | 2.7033 | 2.6002 | 3.4000 | 3.4125 | 3.4375 | 3.3750 |
| lit_r5_02_mc_status_then_shape | 4.1000 | 4.1176 | 4.0588 | 2.8762 | 2.7033 | 2.6002 | 3.4000 | 3.4125 | 3.4375 | 3.3750 |
| lit_r5_03_joint_structured_tf_neg_re2 | 4.4118 | 4.4118 | 4.4118 | 2.8762 | 2.7033 | 2.6002 | 3.4000 | 3.6125 | 3.5625 | 3.6875 |
| lit_r5_04_joint_status_tf_neg_re2 | 4.1000 | 4.1176 | 4.0588 | 2.8762 | 2.7033 | 2.6002 | 3.4000 | 3.6125 | 3.5625 | 3.6875 |

## Concrete failure and success cases

### Failure 1 — the structured guard suppresses a true central spike

- Record: `series_000103:1`; gold/base/candidate = `C/C/A`; final Δ = `-2.4000`.
- Question: Within the window spanning steps 41 to 78, which pattern most accurately indicates the presence of an anomaly near the center of the window? A) A steady, gradual decline in the series followed by a quick return to baseline B) Multiple small oscillations with no distinct peaks or persistent changes C) A sudden upward spike immediately succeeded by a sustained upward drift D) Consistent values throughout the window wi…
- Baseline: Answer: C) A sudden upward spike immediately succeeded by a sustained upward drift Explanation: Within the window from step 41 to 78, there is a noticeable sharp upward spike at step 55 (value 3.00), followed by a continued increase in subsequent steps (e.g., 2.52, 0.13, 1.20, etc.). This pattern indicates a local anomaly characterized by a sudden spike and a subsequent period of elevated values, which deviates from…
- Candidate: Answer: A) A steady, gradual decline in the series followed by a quick return to baseline Explanation: The time series window from step 41 to 78 shows a pattern where values decrease gradually over several steps, reaching a low point, and then rapidly rise back toward higher values. This forms a local wide downward spike followed by an upward drift, matching the described anomaly signature. There are no multiple sma…
- Mechanism: the added list of ordinary variation and structured signatures does not merely calibrate a threshold. It redirects attention from the checkpoint's learned local-hint interpretation toward a broad decline/recovery narrative, changing the anomaly shape and the option.

### Failure 2 — status-before-shape creates a premature normality decision

- Record: `series_000083:1`; gold/base/candidate = `C/C/B`; final Δ = `-3.0000`.
- Baseline: Answer: C) A sudden, sharp upward spike occurring locally and abruptly within the window Explanation: The time series window from step 165 to 200 shows mostly negative values with minor fluctuations, but there is a clear and abrupt upward spike at the end of the window, where the value jumps to 0.81. This sharp increase stands out from the preceding negative values and is not part of a sustained trend or oscillation…
- Candidate: Answer: B) An early downward dip that gradually corrects to normal levels Explanation: The time series window from step 165 to 200 shows a pronounced downward dip, with values dropping sharply to around -3.00, followed by a gradual recovery back toward less negative values. This pattern is characterized by a sudden, significant drop and a slow return, indicating a local anomaly rather than ordinary variance or a con…
- Mechanism: forcing an anomaly-status decision before option comparison separates two computations that the finetuned model learned jointly. Once the first stage declares ordinary behavior, the second stage rationalizes a normal/recovery distractor and ignores the correct boundary spike.

### Failure 3 — RE2 fixes polarity but can rewrite the evidence

- Record: `series_000111:1`; gold/base/candidate = `False/False/True`; final Δ = `-1.6000`.
- Question: True or False: No evidence of anomalous behavior is present within the time series window from step 13 to 53.
- Baseline: <think> False. Within the time series window from step 13 to 53, there is clear evidence of anomalous behavior. Initially, the values remain relatively stable and negative, fluctuating slightly around -0.3. However, starting at step 28, a sudden and sharp increase occurs, with values rapidly rising to 0.40, 1.41, 2.44, and then peaking at 262. This is immediately followed by a rapid drop back to negative values. Suc…
- Candidate: Answer: True. The time series window from step 13 to 53 shows a consistent pattern of small fluctuations around negative values, with no sudden spikes, abrupt changes, or irregular fluctuations. There is a brief transition to positive values starting around step 40, but this shift is gradual and does not represent an anomalous event. The overall behavior remains stable and regular throughout the window, indicating n…
- Mechanism: the TF rule often repairs proposition polarity, but it is not a deterministic label normalizer. On this case it also changes the time-series interpretation, calling a sharp excursion gradual and non-anomalous; the forced verdict then becomes internally consistent but wrong.

### Success case — complete-option comparison recovers a localized shake

- Record: `series_000022:1`; gold/base/candidate = `B/D/B`; final Δ = `+3.0000`.
- Baseline: D) A single abrupt spike exactly at the window boundary without further fluctuations Explanation: The time series window from step 21 to 55 shows a gradual decrease in value, followed by a period of relatively stable values. There is no evidence of an abrupt spike or rapid oscillations within the window; instead, the pattern is characterized by a smooth downward trend with minor fluctuations. Therefore, none of the …
- Candidate: Answer: B) Intermittent, rapid oscillations in amplitude localized within the window Explanation: The time series window from step 21 to 55 shows a pattern of alternating small positive and negative values, with no consistent upward or downward trend. Instead, the values fluctuate rapidly between approximately -0.79 and -1.81, indicating a series of small, irregular oscillations. This behavior is characteristic of i…
- The intervention succeeds when the error is true semantic option binding: it maps localized oscillation to the complete option rather than defaulting to a boundary-spike label. This benefit is real, but it is not uniformly separable from the failure modes.

## Failure synthesis

1. Long or multi-clause guidance competes with the checkpoint's finetuned prompt distribution. It changes evidence interpretation, not just response formatting.
2. Normality priors and anomaly-threshold language are unsafe: they prevent false positives on some normal windows but suppress true localized spikes on others.
3. Decomposing status and shape is not modular for this model. The first generated judgment anchors the second and promotes post-hoc rationalization.
4. RE2-style TF rules repair many polarity errors but occasionally alter the underlying evidence narrative; aggregate gains can hide a catastrophic per-case reversal.
5. OE was left byte-identical to Baseline. That is why its four metrics are exactly tied rather than improved; no tested OE rule survived earlier validation.

## Frozen prompt rules

All routes preserve the released opening, Values → Per-Step Analysis → `Overall Summary Hints` order, 30 Fixed tokens and question-ending boundary. Only the following `### Answering Rule` text is conditionally inserted.

### MC structured-signature guard

```text
Compare every option by its complete meaning. Treat alternating signs, ordinary peaks or troughs, isolated large or small values, and irregular-looking fluctuation as normal variability unless the supplied evidence supports the option's specific structured anomaly signature, such as a localized contrast, persistence, recovery, or boundary pattern. Select an anomalous option only when that defining signature is supported; otherwise select the normal option. Explain the decisive qualitative pattern without quoting exact values or step numbers unless asked.
```

### MC status then shape

```text
First decide anomaly status from the supplied Per-Step Analysis and Overall Summary Hints, independently of the option wording. Ordinary variance, alternating signs, isolated highs or lows, and irregular-looking fluctuation are not anomalies by themselves. Then compare only options consistent with that status and choose the one whose complete text best matches the observed shape, persistence, recovery, and boundary behavior. Explain the decisive qualitative evidence without quoting exact values or step numbers unless asked.
```

### Explicit-negation TF RE2

Activated only for the frozen lexical negation cues.

```text
Evaluate the complete proposition exactly as written. Use qualitative shape, direction, persistence, and recovery; do not quote exact values or step numbers unless the question explicitly asks for them. End with exactly "Your answer: True." or "Your answer: False.", and keep the explanation consistent with that verdict.
```

## Final decision

The search completed 30 new modes across at least three distinct mechanistic cycles. Because no candidate passed all four frozen gates, the registered stopping criterion is met. Selecting the largest mean gain or relaxing the correct→wrong guard after seeing holdout would be post-hoc selection. `paper140` therefore remains untouched and there is no claim of a prompt-only Pareto improvement.
