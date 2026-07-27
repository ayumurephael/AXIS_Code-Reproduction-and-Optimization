# Literature Round 3 analysis

## Integrity and scope

- Data: exposed `development96`, with pooled results and its screening24 / validation72 constituents reported separately.
- Raw inference: 480/480 predictions on three authorized A100 GPUs.
- New canonical components: 168 predictions / 336 Judge dimensions.
- Final assembly: 768 predictions / 1,768 scores including Baseline.
- Judge: 1,754 `final_score_top_logprobs` and 14 `exact_sample_mean_20` rows after component reuse; every row is `deepseek-v4-pro` from provider `deepseek`.
- The full audit found no missing or duplicate key, invalid prompt hash, model/provider mismatch, or unapproved score method.
- Candidate `paper140` outputs remain ungenerated.

## Pooled result

| Rank | Route | Nonnegative | Worst delta | Mean delta | Closed-task correct→wrong |
|---:|---|---:|---:|---:|---:|
| 1 | `lit_r3_07_joint_semantic_tf_qual` | 10/10 | +0.000 | +0.175 | 1 |
| 2 | `lit_r3_05_tf_re2_qual_verdict` | 10/10 | +0.000 | +0.107 | 1 |
| 3 | `lit_r3_04_tf_re2_verdict` | 10/10 | +0.000 | +0.091 | 1 |
| 4 | `lit_r3_02_mc_semantic_qual` | 10/10 | +0.000 | +0.068 | 0 |
| 5 | `lit_r3_06_joint_pointwise_tf_qual` | 9/10 | −0.029 | +0.138 | 1 |
| 6 | `lit_r3_01_mc_pointwise_qual` | 9/10 | −0.029 | +0.031 | 0 |
| 7 | `lit_r3_03_mc_salient_aftermath` | 7/10 | −0.147 | −0.029 | 1 |

`lit_r3_02_mc_semantic_qual` is the only route that satisfies every preregistered development guard. Its pooled MC Final/Correctness/Reasoning deltas are respectively +0.256/+0.324/+0.097, with exact Baseline OE and TF. It also has nonnegative MC/OE/TF Final deltas and zero closed-task correct→wrong transitions on both screening24 and validation72. It advances to the exposed holdout48 robustness check.

The joint and TF routes do not pass despite their pooled 10/10 aggregate metrics. On screening24, qualitative TF RE2 lowers TF Justification by 0.111 and changes one Baseline-correct answer to wrong. The requested exact final string was obeyed on 0/33 TF records: the checkpoint often placed `Answer: True.` or `Answer: False.` at the beginning instead. This formatting failure is reported independently from semantic parsing; all TF responses still contained a parseable verdict.

## Failure cases

### MC semantic binding plus qualitative support

This route repairs two large Baseline option-selection errors without a decision regression:

- `series_000024:0`: Baseline over-weighted moderate sign-changing variation and selected anomalous oscillations (D). The candidate compared complete option meanings and selected normal stable fluctuation (C), raising Final from 1.300 to 4.700.
- `series_000091:1`: Baseline attended to an isolated positive peak (B) and missed the prolonged broad downward dip (C). The candidate selected C and raised Final from 1.300 to 4.700.

The qualitative guard repairs the earlier numerical-decoding failure, but it does not eliminate every explanation error. In `series_000033:0` the selected answer remains correct B, yet the candidate invents a smooth rise from 2.77 to 4.80 and omits the sharp initial drop; Final falls by 1.000. The aggregate route nevertheless improves all three MC metrics because it avoids both decision regressions and exact-number obligations.

### Pointwise qualitative comparison

Pointwise comparison again has zero correct→wrong transitions, but pooled MC Reasoning remains slightly negative (−0.029). Requiring the strongest missing competitor feature can spend the limited explanation budget on option elimination instead of describing the selected pattern. It is therefore not advanced.

### Salient event plus aftermath

The salience rule over-commits to a single local event and loses all three MC metrics. On `series_000103:1` it changes the correct compound anomaly interpretation to a normal alternative. Explicitly directing salience is not a safe substitute for comparing complete option semantics.

### TF RE2 plus qualitative verdict

The TF component has a large net benefit: pooled TF Final/Correctness/Justification improve by +0.361/+0.379/+0.333 and 14 Baseline-wrong decisions become correct. It also repairs the prior parser and numeric failure on `series_000109:1`: the new response begins `Answer: True.` and no longer invents an endpoint near −21.70.

However, `series_000132:1` remains a real semantic regression. Baseline correctly answers True to the positive claim “local upward spike immediately followed by a sustained increase.” RE2 reconstructs the values as a narrow, stable range and answers False. This is not an extractor artifact. Repeating the question plus a universal TF instruction reallocates attention away from the learned latent evidence on a positive anomaly claim.

Several Judge losses also show that the qualitative rule can omit real transitions. For example, `series_000010:0` remains correctly False but the candidate describes only a smooth rise and omits the later sharp decline and recovery, reducing Final by 1.400.

## Outer-loop synthesis

1. Advance `lit_r3_02_mc_semantic_qual` unchanged; do not add a second MC instruction.
2. Do not apply TF RE2 universally. Route polarity help only to propositions with explicit negative/non-anomaly language; leave positive anomaly claims on the released prompt.
3. Test both the existing RE2 implementation and a new no-reread rule that asks for the model's naturally preferred `Answer: True/False.` prefix.
4. Keep OE exactly Baseline. Earlier directness, RE2, Contract, and positive coverage rules all produced speech-act or diagnostic regressions.
5. Treat lexical routing as an exposed-development hypothesis. It must pass pooled and constituent guards before an exposed holdout run and cannot support an untouched-generalization claim by itself.
