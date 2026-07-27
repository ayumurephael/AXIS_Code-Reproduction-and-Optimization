# Literature Round 5 — exposed holdout48

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r5_03_joint_structured_tf_neg_re2 | 10/10 | +0.000 | +0.230 | +0.547 | +0.529 | +0.588 | +0.000 | +0.000 | +0.000 | +0.000 | +0.200 | +0.125 | +0.312 |
| 2 | lit_r5_01_mc_structured_guard | 10/10 | +0.000 | +0.166 | +0.547 | +0.529 | +0.588 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 3 | lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.000 | +0.134 | +0.235 | +0.235 | +0.235 | +0.000 | +0.000 | +0.000 | +0.000 | +0.200 | +0.125 | +0.312 |
| 4 | lit_r5_02_mc_status_then_shape | 10/10 | +0.000 | +0.071 | +0.235 | +0.235 | +0.235 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.064 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.200 | +0.125 | +0.312 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.865 | 3.882 | 3.824 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| lit_r5_03_joint_structured_tf_neg_re2 | 4.412 | 4.412 | 4.412 | 2.876 | 2.703 | 2.600 | 3.400 | 3.613 | 3.563 | 3.687 |
| lit_r5_01_mc_structured_guard | 4.412 | 4.412 | 4.412 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| lit_r5_04_joint_status_tf_neg_re2 | 4.100 | 4.118 | 4.059 | 2.876 | 2.703 | 2.600 | 3.400 | 3.613 | 3.563 | 3.687 |
| lit_r5_02_mc_status_then_shape | 4.100 | 4.118 | 4.059 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| lit_r4_01_tf_neg_re2 | 3.865 | 3.882 | 3.824 | 2.876 | 2.703 | 2.600 | 3.400 | 3.613 | 3.563 | 3.687 |

## Paired outcome counts

- `lit_r5_03_joint_structured_tf_neg_re2`: wins=23, ties=21, losses=4, correct→wrong=1, wrong→correct=5.
- `lit_r5_01_mc_structured_guard`: wins=14, ties=31, losses=3, correct→wrong=0, wrong→correct=1.
- `lit_r5_04_joint_status_tf_neg_re2`: wins=19, ties=21, losses=8, correct→wrong=2, wrong→correct=5.
- `lit_r5_02_mc_status_then_shape`: wins=10, ties=31, losses=7, correct→wrong=1, wrong→correct=1.
- `lit_r4_01_tf_neg_re2`: wins=9, ties=38, losses=1, correct→wrong=1, wrong→correct=4.
