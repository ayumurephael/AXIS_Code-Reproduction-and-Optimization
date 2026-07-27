# Literature Round 5 — development96

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.000 | +0.149 | +0.259 | +0.294 | +0.176 | +0.000 | +0.000 | +0.000 | +0.000 | +0.261 | +0.314 | +0.182 |
| 2 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.076 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.261 | +0.314 | +0.182 |
| 3 | lit_r5_02_mc_status_then_shape | 10/10 | +0.000 | +0.073 | +0.259 | +0.294 | +0.176 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | lit_r5_03_joint_structured_tf_neg_re2 | 9/10 | -0.041 | +0.090 | +0.070 | +0.118 | -0.041 | +0.000 | +0.000 | +0.000 | +0.000 | +0.261 | +0.314 | +0.182 |
| 5 | lit_r5_01_mc_structured_guard | 9/10 | -0.041 | +0.015 | +0.070 | +0.118 | -0.041 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| lit_r5_04_joint_status_tf_neg_re2 | 4.141 | 4.176 | 4.059 | 3.229 | 3.448 | 2.793 | 3.483 | 3.313 | 3.461 | 3.091 |
| lit_r4_01_tf_neg_re2 | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.313 | 3.461 | 3.091 |
| lit_r5_02_mc_status_then_shape | 4.141 | 4.176 | 4.059 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| lit_r5_03_joint_structured_tf_neg_re2 | 3.952 | 4.000 | 3.841 | 3.229 | 3.448 | 2.793 | 3.483 | 3.313 | 3.461 | 3.091 |
| lit_r5_01_mc_structured_guard | 3.952 | 4.000 | 3.841 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |

## Paired outcome counts

- `lit_r5_04_joint_status_tf_neg_re2`: wins=27, ties=47, losses=22, correct→wrong=3, wrong→correct=16.
- `lit_r4_01_tf_neg_re2`: wins=10, ties=81, losses=5, correct→wrong=0, wrong→correct=12.
- `lit_r5_02_mc_status_then_shape`: wins=17, ties=62, losses=17, correct→wrong=3, wrong→correct=4.
- `lit_r5_03_joint_structured_tf_neg_re2`: wins=27, ties=47, losses=22, correct→wrong=1, wrong→correct=15.
- `lit_r5_01_mc_structured_guard`: wins=17, ties=62, losses=17, correct→wrong=1, wrong→correct=3.
