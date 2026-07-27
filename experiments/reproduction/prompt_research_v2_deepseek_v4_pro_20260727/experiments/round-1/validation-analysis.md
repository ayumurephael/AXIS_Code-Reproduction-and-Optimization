# Round 1 validation72 and development96 analysis

## Audit

- Formal Judge: `deepseek-v4-pro` only.
- `validation72`: 72 QA on 36 series, disjoint from `screening24`.
- `development96`: the preregistered pooled view of screening24 and
  validation72. These records are exposed falsification data, not final test
  evidence.
- Unchanged question families reuse the exact canonical Baseline prediction
  and score. Only the selected family is regenerated and re-judged.
- Validation component accounting: 158 changed responses and 398 score
  dimensions. All 398 primary rows completed; ten registered fallbacks also
  completed.

## Validation72

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | All 10 nonlower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | 4.0192 | 4.0769 | 3.8846 | 3.4703 | 3.6750 | 3.0455 | 3.7273 | 2.9158 | 2.9708 | 2.8333 | — |
| OE balanced | = | = | = | 3.2380 (-0.2324) | 3.3091 (-0.3659) | 2.7864 (-0.2591) | 3.6818 (-0.0455) | = | = | = | No |
| OE contrast | = | = | = | 3.0935 (-0.3768) | 3.2273 (-0.4477) | 2.6114 (-0.4340) | 3.5000 (-0.2273) | = | = | = | No |
| OE evidence router | = | = | = | 3.4645 (-0.0058) | 3.7000 (+0.0250) | 2.9591 (-0.0864) | 3.7795 (+0.0523) | = | = | = | No |
| OE direct | = | = | = | 3.1818 (-0.2885) | 3.5000 (-0.1750) | 2.5909 (-0.4545) | 3.5000 (-0.2273) | = | = | = | No |
| MC balanced | 4.1513 (+0.1321) | 4.2385 (+0.1615) | 3.9481 (+0.0635) | = | = | = | = | = | = | = | Yes |
| MC content/output | 4.0513 (+0.0321) | 4.1154 (+0.0385) | 3.9019 (+0.0173) | = | = | = | = | = | = | = | Yes |
| TF whole statement | = | = | = | = | = | = | = | 2.7750 (-0.1408) | 2.8750 (-0.0958) | 2.6250 (-0.2083) | No |

## Pooled development96

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. | All 10 nonlower |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | 3.8942 | 3.9119 | 3.8529 | 3.2701 | 3.3397 | 2.8275 | 3.7052 | 3.0600 | 3.1000 | 3.0000 | — |
| OE balanced | = | = | = | 3.1512 (-0.1189) | 3.2000 (-0.1396) | 2.7000 (-0.1275) | 3.6207 (-0.0845) | = | = | = | No |
| OE contrast | = | = | = | 2.9968 (-0.2732) | 3.1724 (-0.1672) | 2.4638 (-0.3637) | 3.4138 (-0.2914) | = | = | = | No |
| OE evidence router | = | = | = | 3.2765 (+0.0065) | 3.4621 (+0.1224) | 2.7621 (-0.0655) | 3.6603 (-0.0448) | = | = | = | No |
| OE direct | = | = | = | 3.1170 (-0.1531) | 3.4138 (+0.0741) | 2.5362 (-0.2913) | 3.4483 (-0.2569) | = | = | = | No |
| MC balanced | 4.2657 (+0.3715) | 4.3588 (+0.4469) | 4.0485 (+0.1956) | = | = | = | = | = | = | = | Yes |
| MC content/output | 4.0716 (+0.1774) | 4.1471 (+0.2351) | 3.8956 (+0.0427) | = | = | = | = | = | = | = | Yes |
| TF whole statement | = | = | = | = | = | = | = | 3.0060 (-0.0540) | 3.0303 (-0.0697) | 2.9696 (-0.0304) | No |

## Failure-case synthesis

### OE: added context wording can overwrite a correct learned decision

`series_000012:0` asks whether a sharp downward movement is anomalous.
Baseline identifies the step-129 drop and receives
Accuracy/Completeness/Relevance `5/2/4`. The balanced-context candidate says
there is no sharp drop and receives `1/1/1`. Likewise, on
`series_000104:1`, Baseline detects the local downward spike while all four
OE variants deny it. The problem is not merely verbosity: the extra rule
changes the released checkpoint's evidence arbitration and can flip a
previously correct latent decision.

The inverse also occurs. On `series_000140:1`, the candidates recover the
rapid-decline/slow-rise anomaly that Baseline misses. This explains the
screening Accuracy gains, but the correction is not selective: the same
intervention also creates false-normal answers. A universal instruction to
“use Per-Step context” supplies no reliable decoder for the learned vectors.

### OE: verdict correction and requested coverage remain separate

The evidence router is the only OE mode near neutral in aggregate. It improves
Accuracy and Relevance on validation72, but Completeness falls by `0.0864`.
On questions requesting boundary qualifications, alternative explanations,
or methods, candidate answers often stop after a verdict and a local shape
description. Therefore an OE prompt needs both evidence arbitration and an
explicit checklist derived from the actual question; a fixed generic rule
cannot cover the heterogeneous OE requests.

### MC: semantic option comparison is stable but not monotone per case

The balanced MC rule improves all three MC metrics on both validation72 and
development96. It corrects severe cases including `series_000057:1`
(`A` to gold `C`) and `series_000137:1` (`A` to gold `C`). The rule makes the
model compare complete option meanings against both window shape and context.

It can still reduce Judge scores without changing the selected option. On
`series_000075:0` and `series_000025:0`, both Baseline and candidate choose
the gold `A`, but minor factual wording changes lower both dimensions from
`5/5` to `4/4`. This is why finalist judging must use paired repeats rather
than interpreting every one-point case as deterministic.

### TF: explicit negation guidance created verdict/reason contradictions

On `series_000104:0`, `series_000106:0`, and `series_000073:1`, the candidate
starts with `False` but its explanation says there is no anomaly and, in one
case, ends by saying the statement is true. These are polarity-binding
failures induced by the extra whole-statement instruction. The model can
describe the evidence correctly while emitting the opposite Boolean token.
The small screening benefit therefore does not generalize.

## Decision

Only the two MC components survive Round 1 validation. No OE or TF component
is eligible for holdout promotion. Because the project's final success rule
requires a changed and improved OE prompt, Round 1 cannot produce a complete
final prompt; the next outer loop must search for a more selective OE
mechanism while retaining the original checkpoint's learned scaffold.
