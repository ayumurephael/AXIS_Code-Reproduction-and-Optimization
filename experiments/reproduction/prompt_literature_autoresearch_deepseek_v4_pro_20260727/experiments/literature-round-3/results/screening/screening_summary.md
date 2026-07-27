# Literature Round 3 screening constituent

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r3_02_mc_semantic_qual | 10/10 | +0.000 | +0.282 | +1.075 | +1.375 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | lit_r3_04_tf_re2_verdict | 10/10 | +0.000 | +0.033 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.111 | +0.111 | +0.111 |
| 3 | lit_r3_07_joint_semantic_tf_qual | 9/10 | -0.111 | +0.320 | +1.075 | +1.375 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.156 | +0.333 | -0.111 |
| 4 | lit_r3_05_tf_re2_qual_verdict | 9/10 | -0.111 | +0.038 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.156 | +0.333 | -0.111 |
| 5 | lit_r3_01_mc_pointwise_qual | 9/10 | -0.125 | +0.026 | +0.137 | +0.250 | -0.125 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 6 | lit_r3_03_mc_salient_aftermath | 9/10 | -0.250 | -0.011 | +0.012 | +0.125 | -0.250 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 7 | lit_r3_06_joint_pointwise_tf_qual | 8/10 | -0.125 | +0.064 | +0.137 | +0.250 | -0.125 | +0.000 | +0.000 | +0.000 | +0.000 | +0.156 | +0.333 | -0.111 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r3_02_mc_semantic_qual | 4.775 | 5.000 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r3_04_tf_re2_verdict | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.444 | 3.444 | 3.444 |
| lit_r3_07_joint_semantic_tf_qual | 4.775 | 5.000 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.489 | 3.667 | 3.222 |
| lit_r3_05_tf_re2_qual_verdict | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.489 | 3.667 | 3.222 |
| lit_r3_01_mc_pointwise_qual | 3.837 | 3.875 | 3.750 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r3_03_mc_salient_aftermath | 3.712 | 3.750 | 3.625 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r3_06_joint_pointwise_tf_qual | 3.837 | 3.875 | 3.750 | 2.536 | 2.571 | 1.857 | 3.286 | 3.489 | 3.667 | 3.222 |

## Paired outcome counts

- `lit_r3_02_mc_semantic_qual`: wins=6, ties=16, losses=2, correct→wrong=0, wrong→correct=2.
- `lit_r3_04_tf_re2_verdict`: wins=5, ties=15, losses=4, correct→wrong=1, wrong→correct=4.
- `lit_r3_07_joint_semantic_tf_qual`: wins=12, ties=7, losses=5, correct→wrong=1, wrong→correct=6.
- `lit_r3_05_tf_re2_qual_verdict`: wins=6, ties=15, losses=3, correct→wrong=1, wrong→correct=4.
- `lit_r3_01_mc_pointwise_qual`: wins=5, ties=16, losses=3, correct→wrong=0, wrong→correct=0.
- `lit_r3_03_mc_salient_aftermath`: wins=2, ties=16, losses=6, correct→wrong=0, wrong→correct=0.
- `lit_r3_06_joint_pointwise_tf_qual`: wins=11, ties=7, losses=6, correct→wrong=1, wrong→correct=4.
