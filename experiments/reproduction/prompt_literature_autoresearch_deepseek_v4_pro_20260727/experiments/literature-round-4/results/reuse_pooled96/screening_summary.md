# Literature Round 4 reuse routers pooled96

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r4_04_joint_semantic_tf_nonanomaly_re2 | 10/10 | +0.000 | +0.167 | +0.256 | +0.324 | +0.097 | +0.000 | +0.000 | +0.000 | +0.000 | +0.342 | +0.409 | +0.242 |
| 2 | lit_r4_02_joint_semantic_tf_neg_re2 | 10/10 | +0.000 | +0.143 | +0.256 | +0.324 | +0.097 | +0.000 | +0.000 | +0.000 | +0.000 | +0.261 | +0.314 | +0.182 |
| 3 | lit_r4_03_tf_nonanomaly_re2 | 10/10 | +0.000 | +0.099 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.342 | +0.409 | +0.242 |
| 4 | lit_r4_01_tf_neg_re2 | 10/10 | +0.000 | +0.076 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.261 | +0.314 | +0.182 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| lit_r4_04_joint_semantic_tf_nonanomaly_re2 | 4.138 | 4.206 | 3.979 | 3.229 | 3.448 | 2.793 | 3.483 | 3.394 | 3.556 | 3.152 |
| lit_r4_02_joint_semantic_tf_neg_re2 | 4.138 | 4.206 | 3.979 | 3.229 | 3.448 | 2.793 | 3.483 | 3.313 | 3.461 | 3.091 |
| lit_r4_03_tf_nonanomaly_re2 | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.394 | 3.556 | 3.152 |
| lit_r4_01_tf_neg_re2 | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.313 | 3.461 | 3.091 |

## Paired outcome counts

- `lit_r4_04_joint_semantic_tf_nonanomaly_re2`: wins=34, ties=39, losses=23, correct→wrong=0, wrong→correct=15.
- `lit_r4_02_joint_semantic_tf_neg_re2`: wins=28, ties=47, losses=21, correct→wrong=0, wrong→correct=14.
- `lit_r4_03_tf_nonanomaly_re2`: wins=16, ties=73, losses=7, correct→wrong=0, wrong→correct=13.
- `lit_r4_01_tf_neg_re2`: wins=10, ties=81, losses=5, correct→wrong=0, wrong→correct=12.
