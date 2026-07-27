# Literature Round 5 — screening24

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r5_03_joint_structured_tf_neg_re2 | 10/10 | +0.000 | +0.359 | +0.975 | +1.125 | +0.625 | +0.000 | +0.000 | +0.000 | +0.000 | +0.311 | +0.444 | +0.111 |
| 2 | lit_r5_01_mc_structured_guard | 10/10 | +0.000 | +0.272 | +0.975 | +1.125 | +0.625 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 3 | lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.000 | +0.220 | +0.462 | +0.500 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.311 | +0.444 | +0.111 |
| 4 | lit_r5_02_mc_status_then_shape | 10/10 | +0.000 | +0.134 | +0.462 | +0.500 | +0.375 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.087 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.311 | +0.444 | +0.111 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r5_03_joint_structured_tf_neg_re2 | 4.675 | 4.750 | 4.500 | 2.536 | 2.571 | 1.857 | 3.286 | 3.644 | 3.778 | 3.444 |
| lit_r5_01_mc_structured_guard | 4.675 | 4.750 | 4.500 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r5_04_joint_status_tf_neg_re2 | 4.163 | 4.125 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.644 | 3.778 | 3.444 |
| lit_r5_02_mc_status_then_shape | 4.163 | 4.125 | 4.250 | 2.536 | 2.571 | 1.857 | 3.286 | 3.333 | 3.333 | 3.333 |
| lit_r4_01_tf_neg_re2 | 3.700 | 3.625 | 3.875 | 2.536 | 2.571 | 1.857 | 3.286 | 3.644 | 3.778 | 3.444 |

## Paired outcome counts

- `lit_r5_03_joint_structured_tf_neg_re2`: wins=9, ties=12, losses=3, correct→wrong=0, wrong→correct=6.
- `lit_r5_01_mc_structured_guard`: wins=6, ties=16, losses=2, correct→wrong=0, wrong→correct=2.
- `lit_r5_04_joint_status_tf_neg_re2`: wins=7, ties=12, losses=5, correct→wrong=1, wrong→correct=6.
- `lit_r5_02_mc_status_then_shape`: wins=4, ties=16, losses=4, correct→wrong=1, wrong→correct=2.
- `lit_r4_01_tf_neg_re2`: wins=3, ties=20, losses=1, correct→wrong=0, wrong→correct=4.
