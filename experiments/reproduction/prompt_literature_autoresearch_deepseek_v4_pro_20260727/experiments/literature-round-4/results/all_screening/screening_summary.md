# Literature Round 4 all routes screening

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r4_06_joint_semantic_tf_neg_prefix | 10/10 | +0.000 | +0.400 | +1.075 | +1.375 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.400 | +0.444 | +0.333 |
| 2 | lit_r4_02_joint_semantic_tf_neg_re2 | 10/10 | +0.000 | +0.369 | +1.075 | +1.375 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.311 | +0.444 | +0.111 |
| 3 | lit_r4_05_tf_neg_prefix | 10/10 | +0.000 | +0.118 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.400 | +0.444 | +0.333 |
| 4 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.087 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.311 | +0.444 | +0.111 |
| 5 | lit_r4_04_joint_semantic_tf_nonanomaly_re2 | 9/10 | -0.111 | +0.356 | +1.075 | +1.375 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.289 | +0.556 | -0.111 |
| 6 | lit_r4_03_tf_nonanomaly_re2 | 9/10 | -0.111 | +0.073 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.289 | +0.556 | -0.111 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r4_06_joint_semantic_tf_neg_prefix | 4.775 | 5.000 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.733 | 3.778 | 3.667 |
| lit_r4_02_joint_semantic_tf_neg_re2 | 4.775 | 5.000 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.644 | 3.778 | 3.444 |
| lit_r4_05_tf_neg_prefix | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.733 | 3.778 | 3.667 |
| lit_r4_01_tf_neg_re2 | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.644 | 3.778 | 3.444 |
| lit_r4_04_joint_semantic_tf_nonanomaly_re2 | 4.775 | 5.000 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.622 | 3.889 | 3.222 |
| lit_r4_03_tf_nonanomaly_re2 | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.622 | 3.889 | 3.222 |

## Paired outcome counts

- `lit_r4_06_joint_semantic_tf_neg_prefix`: wins=9, ties=12, losses=3, correct→wrong=0, wrong→correct=6.
- `lit_r4_02_joint_semantic_tf_neg_re2`: wins=9, ties=12, losses=3, correct→wrong=0, wrong→correct=6.
- `lit_r4_05_tf_neg_prefix`: wins=3, ties=20, losses=1, correct→wrong=0, wrong→correct=4.
- `lit_r4_01_tf_neg_re2`: wins=3, ties=20, losses=1, correct→wrong=0, wrong→correct=4.
- `lit_r4_04_joint_semantic_tf_nonanomaly_re2`: wins=11, ties=9, losses=4, correct→wrong=0, wrong→correct=6.
- `lit_r4_03_tf_nonanomaly_re2`: wins=5, ties=17, losses=2, correct→wrong=0, wrong→correct=4.
