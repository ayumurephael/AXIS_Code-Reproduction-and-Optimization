# Literature Round 1 analysis

## Integrity

- Split: exposed screening24, 24 QA / 12 series.
- Raw inference: 288/288 predictions on three A100 GPUs.
- Unique non-Baseline components: 97 predictions / 215 Judge dimensions.
- Final assembly: 312 predictions / 715 scores including Baseline.
- Judge audit: 715/715 `deepseek-v4-pro`; 715/715 `final_score_top_logprobs`; no duplicates, missing IDs, empty responses, invalid hashes, fallback, or model/provider mismatch.

## Ranking

| Rank | Mode | Nonnegative | Worst Δ | Mean Δ | Changed family |
|---:|---|---:|---:|---:|---|
| 1 | `lit_r1_03_mc_re2` | 10/10 | +0.000 | +0.219 | MC |
| 2 | `lit_r1_12_mc_bind_re2` | 10/10 | +0.000 | +0.186 | MC |
| 3 | `lit_r1_02_mc_pointwise` | 10/10 | +0.000 | +0.144 | MC |
| 4 | `lit_r1_01_mc_semantic_bind` | 10/10 | +0.000 | +0.101 | MC |
| 5 | `lit_r1_06_tf_re2` | 10/10 | +0.000 | +0.033 | TF |
| 6 | `lit_r1_10_triplet_re2` | 9/10 | -0.143 | +0.272 | MC/OE/TF |
| 7 | `lit_r1_08_oe_re2` | 9/10 | -0.143 | +0.020 | OE |
| 8–12 | remaining modes | 3–7/10 | -0.444 to -0.286 | -0.102 to +0.017 | mixed |

## Metric deltas for the useful components

| Mode | MC Final | MC Corr. | MC Rsn. | OE Acc. | OE Comp. | OE Rel. | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MC RE2 | +0.812 | +1.000 | +0.375 | 0 | 0 | 0 | 0 | 0 |
| MC bind + RE2 | +0.738 | +1.000 | +0.125 | 0 | 0 | 0 | 0 | 0 |
| MC pointwise | +0.562 | +0.750 | +0.125 | 0 | 0 | 0 | 0 | 0 |
| MC semantic bind | +0.387 | +0.500 | +0.125 | 0 | 0 | 0 | 0 | 0 |
| TF RE2 | 0 | 0 | 0 | 0 | 0 | 0 | +0.111 | +0.111 |
| OE RE2 | 0 | 0 | 0 | +0.143 | +0.143 | -0.143 | 0 | 0 |

## Failure cases and mechanisms

### MC

MC RE2 corrected `series_000024:0` from D to C by reconsidering whether moderate sign-changing fluctuations were truly erratic. It also corrected `series_000091:1` from a salient isolated positive peak (B) to the longer broad downward dip (C). These are question-comprehension and salience-allocation failures, exactly the mechanism RE2 targets.

All four MC routes had zero correct→wrong transitions. However, semantic binding + RE2 gave the same two corrected decisions as RE2 alone while lowering Reasoning and inventing `-6.40` in one explanation. More instruction is therefore not additive.

### TF

TF RE2 changed four wrong decisions to correct but one correct decision to wrong. On `series_000132:1`, Baseline correctly recognized an upward spike followed by sustained increase. Every added TF wording reconstructed the values as a narrow stable range and answered False. This is not a parser problem: the label and rationale agree with each other but disagree with the learned evidence. The instruction shifts decoding away from the useful Fixed/local latent evidence.

The minimal and clause rules had worse aggregate TF scores. Explicitly checking every clause increased attention to verbal negation while weakening evidence decoding.

### OE

OE RE2 raised Accuracy/Completeness but lowered Relevance. In `series_000008:0`, it converted displayed scaled integers into apparent values such as `-135` and `-244`, then spent text hedging about them. Repeating a long OE question also repeats its hypothetical framing, encouraging verbosity rather than a better diagnosis.

The direct OE rule shortened answers and reduced the unsupported-number fraction, but continued to deny real boundary anomalies such as the early downward spike in `series_000024:1`. It lost OE Final (-0.136), Accuracy (-0.143), and Relevance (-0.286). Output discipline cannot repair a latent anomaly decision.

The decoupled rule increased MC but lowered OE Accuracy/Relevance and every TF dimension. The generic “decide, then report” abstraction is still too much cross-task instruction for this checkpoint.

## Advancement

Advance:

1. `lit_r1_03_mc_re2`;
2. `lit_r1_02_mc_pointwise`;
3. `lit_r1_01_mc_semantic_bind`;
4. `lit_r2_01_mc_tf_re2_safe`, a new component route using MC RE2 + TF RE2 + exact Baseline OE.

The fourth route is an explicit risk review permitted by the screening protocol because TF RE2 had all nonnegative dimensions and a 4:1 wrong→correct/correct→wrong ratio. It receives no waiver in validation: any correct→wrong transition or negative metric rejects it.

Do not advance OE changes, static TF rules, the redundant MC bind+RE2 combination, or broad triplet protocols.
