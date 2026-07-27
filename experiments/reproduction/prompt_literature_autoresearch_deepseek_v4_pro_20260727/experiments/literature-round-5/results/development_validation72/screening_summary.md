# Literature Round 5 — validation72

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r5_04_joint_status_tf_neg_re2 | 10/10 | +0.000 | +0.126 | +0.196 | +0.231 | +0.115 | +0.000 | +0.000 | +0.000 | +0.000 | +0.242 | +0.265 | +0.208 |
| 2 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.071 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.242 | +0.265 | +0.208 |
| 3 | lit_r5_02_mc_status_then_shape | 10/10 | +0.000 | +0.054 | +0.196 | +0.231 | +0.115 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | lit_r5_03_joint_structured_tf_neg_re2 | 7/10 | -0.246 | +0.007 | -0.208 | -0.192 | -0.246 | +0.000 | +0.000 | +0.000 | +0.000 | +0.242 | +0.265 | +0.208 |
| 5 | lit_r5_01_mc_structured_guard | 7/10 | -0.246 | -0.065 | -0.208 | -0.192 | -0.246 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r5_04_joint_status_tf_neg_re2 | 4.135 | 4.192 | 4.000 | 3.450 | 3.727 | 3.091 | 3.545 | 3.188 | 3.342 | 2.958 |
| lit_r4_01_tf_neg_re2 | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.188 | 3.342 | 2.958 |
| lit_r5_02_mc_status_then_shape | 4.135 | 4.192 | 4.000 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r5_03_joint_structured_tf_neg_re2 | 3.730 | 3.769 | 3.638 | 3.450 | 3.727 | 3.091 | 3.545 | 3.188 | 3.342 | 2.958 |
| lit_r5_01_mc_structured_guard | 3.730 | 3.769 | 3.638 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |

## Paired outcome counts

- `lit_r5_04_joint_status_tf_neg_re2`: wins=20, ties=35, losses=17, correct→wrong=2, wrong→correct=10.
- `lit_r4_01_tf_neg_re2`: wins=7, ties=61, losses=4, correct→wrong=0, wrong→correct=8.
- `lit_r5_02_mc_status_then_shape`: wins=13, ties=46, losses=13, correct→wrong=2, wrong→correct=2.
- `lit_r5_03_joint_structured_tf_neg_re2`: wins=18, ties=35, losses=19, correct→wrong=1, wrong→correct=9.
- `lit_r5_01_mc_structured_guard`: wins=11, ties=46, losses=15, correct→wrong=1, wrong→correct=1.
