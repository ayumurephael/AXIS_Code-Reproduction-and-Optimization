# Literature Round 4 all routes validation

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r4_04_joint_semantic_tf_nonanomaly_re2 | 10/10 | +0.000 | +0.111 | +0.004 | +0.001 | +0.012 | +0.000 | +0.000 | +0.000 | +0.000 | +0.363 | +0.354 | +0.375 |
| 2 | lit_r4_03_tf_nonanomaly_re2 | 10/10 | +0.000 | +0.109 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.363 | +0.354 | +0.375 |
| 3 | lit_r4_02_joint_semantic_tf_neg_re2 | 10/10 | +0.000 | +0.073 | +0.004 | +0.001 | +0.012 | +0.000 | +0.000 | +0.000 | +0.000 | +0.242 | +0.265 | +0.208 |
| 4 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.071 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.242 | +0.265 | +0.208 |
| 5 | lit_r4_06_joint_semantic_tf_neg_prefix | 10/10 | +0.000 | +0.063 | +0.004 | +0.001 | +0.012 | +0.000 | +0.000 | +0.000 | +0.000 | +0.200 | +0.167 | +0.250 |
| 6 | lit_r4_05_tf_neg_prefix | 10/10 | +0.000 | +0.062 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.200 | +0.167 | +0.250 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r4_04_joint_semantic_tf_nonanomaly_re2 | 3.942 | 3.962 | 3.896 | 3.450 | 3.727 | 3.091 | 3.545 | 3.309 | 3.431 | 3.125 |
| lit_r4_03_tf_nonanomaly_re2 | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.309 | 3.431 | 3.125 |
| lit_r4_02_joint_semantic_tf_neg_re2 | 3.942 | 3.962 | 3.896 | 3.450 | 3.727 | 3.091 | 3.545 | 3.188 | 3.342 | 2.958 |
| lit_r4_01_tf_neg_re2 | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.188 | 3.342 | 2.958 |
| lit_r4_06_joint_semantic_tf_neg_prefix | 3.942 | 3.962 | 3.896 | 3.450 | 3.727 | 3.091 | 3.545 | 3.146 | 3.244 | 3.000 |
| lit_r4_05_tf_neg_prefix | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.146 | 3.244 | 3.000 |

## Paired outcome counts

- `lit_r4_04_joint_semantic_tf_nonanomaly_re2`: wins=23, ties=30, losses=19, correct→wrong=0, wrong→correct=9.
- `lit_r4_03_tf_nonanomaly_re2`: wins=11, ties=56, losses=5, correct→wrong=0, wrong→correct=9.
- `lit_r4_02_joint_semantic_tf_neg_re2`: wins=19, ties=35, losses=18, correct→wrong=0, wrong→correct=8.
- `lit_r4_01_tf_neg_re2`: wins=7, ties=61, losses=4, correct→wrong=0, wrong→correct=8.
- `lit_r4_06_joint_semantic_tf_neg_prefix`: wins=21, ties=35, losses=16, correct→wrong=0, wrong→correct=8.
- `lit_r4_05_tf_neg_prefix`: wins=9, ties=61, losses=2, correct→wrong=0, wrong→correct=8.
