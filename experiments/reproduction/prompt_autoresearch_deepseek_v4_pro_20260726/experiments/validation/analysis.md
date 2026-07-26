# Validation round 1 analysis

## Outcome

No candidate passed the preregistered validation gate. Therefore the internal
holdout and `paper140` remain untouched. The search returns to development and
pivots from one common prompt contract to question-type-conditioned routing.

Gate: at least 8/10 nonnegative Table-I deltas and no task Final more than 0.10
below the split Baseline.

| Rank | Mode | Nonnegative | Worst delta | Mean delta | MC Final Δ | OE Final Δ | TF Final Δ | Eligible? |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `pareto_p07_abd` | 7/10 | -0.457 | +0.128 | +0.081 | -0.271 | +0.662 | No |
| 2 | `fixed_role` | 6/10 | -0.318 | +0.173 | +0.232 | -0.191 | +0.604 | No |
| 3 | `pareto_p01_boundary` | 5/10 | -0.182 | +0.063 | +0.042 | -0.109 | +0.304 | No |
| 4 | `pareto_p03_task_rule` | 4/10 | -0.727 | -0.221 | +0.042 | -0.509 | -0.121 | No |
| 5 | `pareto_p06_abc` | 3/10 | -0.545 | -0.173 | -0.093 | -0.432 | +0.112 | No |

## Formal Table-I scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base` | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| `pareto_p07_abd` | 4.020 | 4.000 | 4.065 | 3.179 | 3.270 | 2.773 | 3.547 | 3.608 | 3.652 | 3.542 |
| `fixed_role` | 4.171 | 4.260 | 3.963 | 3.259 | 3.409 | 2.864 | 3.545 | 3.550 | 3.583 | 3.500 |
| `pareto_p01_boundary` | 3.981 | 4.039 | 3.846 | 3.341 | 3.545 | 3.000 | 3.500 | 3.250 | 3.250 | 3.250 |
| `pareto_p03_task_rule` | 3.981 | 4.038 | 3.846 | 2.941 | 3.000 | 2.364 | 3.545 | 2.825 | 2.875 | 2.750 |
| `pareto_p06_abc` | 3.846 | 3.923 | 3.665 | 3.018 | 3.182 | 2.636 | 3.273 | 3.058 | 3.208 | 2.833 |

## Audit

- Inference: 432/432 predictions, 72 records, six modes, three approved GPUs,
  series batching, beam size 5, and `--skip-loss`.
- Checkpoint SHA-256:
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- Judge: 996/996 dimension scores, model exactly `deepseek-v4-pro`.
- Readout methods: 988 `final_score_top_logprobs` and eight preregistered
  `exact_sample_mean_20` fallbacks.
- Local and remote audits found no missing keys, duplicates, provider/model
  mismatch, invalid prompt hash, or mixed failed-run result.
- An initial TLS run used the wrong system CA path and was stopped before any
  score was accepted. Its log is retained as `artifacts/judge_cert_failed.log`;
  the successful run used the system trust store and a successful authenticated
  probe. TLS verification was never disabled.

## Output behavior

| Mode | MC parsed accuracy | TF parsed accuracy | MC mean chars | OE mean chars | TF mean chars | OE unsupported-number fraction | TF unsupported-number fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base` | 88.46% | 37.50% | 668 | 794 | 608 | 49.73% | 37.75% |
| `fixed_role` | 92.31% | 54.17% | 644 | 745 | 624 | 46.81% | 34.98% |
| `pareto_p01_boundary` | 92.31% | 45.83% | 658 | 758 | 601 | 44.98% | 31.73% |
| `pareto_p03_task_rule` | 88.46% | 58.33% | 644 | 755 | 577 | 37.03% | 24.44% |
| `pareto_p06_abc` | 84.62% | 62.50% | 646 | 723 | 587 | 34.55% | 22.01% |
| `pareto_p07_abd` | 88.46% | 54.17% | 670 | 772 | 637 | 42.16% | 32.85% |

These diagnostics are explanatory only and never replace Table-I Judge scores.
The lower unsupported-number fractions under P03/P06 did not translate to
better OE because those prompts answered the wrong speech act or closed the
assessment prematurely.

## Concrete failure cases

### 1. TF negation-to-label inversion

For `series_000104:0` and `series_000109:1`, the proposition says there is
**no evidence** of anomalous behavior and the gold label is True. P03 and P06
output `False` while their first explanatory claim says there is no anomaly.
The rule “decide the underlying evidence claim first, then map it to True or
False” created a second semantic mapping step. The model classified anomaly
presence, then failed to preserve the proposition's negation. A future rule
must directly compare the proposition as written with the evidence and verify
that the label and explanation have the same truth value.

### 2. Calibration suppresses real anomalies

On `series_000089:1`, the correct MC answer is C, a pronounced upward spike;
on `series_000103:1`, the correct answer is C, an upward spike followed by
sustained drift. P03/P06 change the closed decision away from the correct
anomalous option. The common calibration text lists smooth trends, peaks,
extrema, sign changes, and variability as possible non-anomalies. That list is
not neutral: it primes the model to explain away the very feature it must
recognize. Calibration should not be shared globally; closed-task rules should
compare the complete answer proposition without enumerating anomaly shapes.

### 3. OE speech-act mismatch

All 22 validation OE records are methodological/evidence questions such as
“how would you evaluate” and “what additional evidence would support or
challenge the assessment.” P03 forces every OE answer to start from a
diagnostic conclusion. In failing cases the model denies the question's
premise, omits the requested supporting/challenging factors, or states that no
further evidence is needed. This causes the large P03 losses in OE Accuracy
(-0.727), Completeness (-0.727), and Final (-0.509).

OE needs a speech-act-aware rule: first address the supplied window, then cover
all requested evidence, uncertainty, boundary/persistence/recovery behavior,
and only the requested additional indicators. It must not force a categorical
normal/anomalous closure for a methodological question.

### 4. Short common guards are safer but insufficient

P01 causes no correct-to-wrong parsed closed decisions and reduces unsupported
numbers, but still loses all four OE dimensions and fails the task-Final
gate. P07 strongly improves TF but sacrifices three OE dimensions. The result
supports retaining short boundary discipline where useful while routing the
behavioral instructions by task family.

## Outer-loop synthesis (cycle 3)

Direction: **pivot**.

The first two cycles decomposed the long Contract but still applied most rules
globally. Validation shows the bottleneck is not just contract length; it is
cross-task semantic interference. MC needs stable option comparison, TF needs
direct proposition-polarity consistency, and OE needs coverage of the
question's speech act. The next development batch will therefore keep the
released Values/Local layout, all 30 Fixed tokens, and released generation
boundary, but select a small instruction block by question type.

Because screening and validation have now both informed design, they become a
single 96-QA exposed development pool. The untouched 48-QA/24-series internal
holdout remains the only confirmatory internal set. No candidate may advance
directly from a revised prompt to holdout without first being scored on this
combined development pool under a preregistered gate.