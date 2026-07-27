# Literature Round 3 validation constituent

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r3_06_joint_pointwise_tf_qual | 10/10 | +0.000 | +0.166 | +0.135 | +0.192 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.437 | +0.396 | +0.500 |
| 2 | lit_r3_07_joint_semantic_tf_qual | 10/10 | +0.000 | +0.135 | +0.004 | +0.001 | +0.012 | +0.000 | +0.000 | +0.000 | +0.000 | +0.437 | +0.396 | +0.500 |
| 3 | lit_r3_05_tf_re2_qual_verdict | 10/10 | +0.000 | +0.133 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.437 | +0.396 | +0.500 |
| 4 | lit_r3_04_tf_re2_verdict | 10/10 | +0.000 | +0.113 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.370 | +0.340 | +0.417 |
| 5 | lit_r3_01_mc_pointwise_qual | 10/10 | +0.000 | +0.033 | +0.135 | +0.192 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 6 | lit_r3_02_mc_semantic_qual | 10/10 | +0.000 | +0.002 | +0.004 | +0.001 | +0.012 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 7 | lit_r3_03_mc_salient_aftermath | 7/10 | -0.115 | -0.035 | -0.115 | -0.115 | -0.115 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r3_06_joint_pointwise_tf_qual | 4.073 | 4.154 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.384 | 3.473 | 3.250 |
| lit_r3_07_joint_semantic_tf_qual | 3.942 | 3.962 | 3.896 | 3.450 | 3.727 | 3.091 | 3.545 | 3.384 | 3.473 | 3.250 |
| lit_r3_05_tf_re2_qual_verdict | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.384 | 3.473 | 3.250 |
| lit_r3_04_tf_re2_verdict | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 3.317 | 3.417 | 3.167 |
| lit_r3_01_mc_pointwise_qual | 4.073 | 4.154 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r3_02_mc_semantic_qual | 3.942 | 3.962 | 3.896 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r3_03_mc_salient_aftermath | 3.823 | 3.846 | 3.769 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |

## Paired outcome counts

- `lit_r3_06_joint_pointwise_tf_qual`: wins=32, ties=22, losses=18, correct→wrong=0, wrong→correct=10.
- `lit_r3_07_joint_semantic_tf_qual`: wins=28, ties=22, losses=22, correct→wrong=0, wrong→correct=10.
- `lit_r3_05_tf_re2_qual_verdict`: wins=16, ties=48, losses=8, correct→wrong=0, wrong→correct=10.
- `lit_r3_04_tf_re2_verdict`: wins=16, ties=48, losses=8, correct→wrong=0, wrong→correct=10.
- `lit_r3_01_mc_pointwise_qual`: wins=16, ties=46, losses=10, correct→wrong=0, wrong→correct=0.
- `lit_r3_02_mc_semantic_qual`: wins=12, ties=46, losses=14, correct→wrong=0, wrong→correct=0.
- `lit_r3_03_mc_salient_aftermath`: wins=12, ties=46, losses=14, correct→wrong=1, wrong→correct=1.
