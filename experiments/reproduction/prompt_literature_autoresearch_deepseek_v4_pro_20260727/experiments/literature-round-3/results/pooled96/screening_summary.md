# Literature Round 3 pooled development96

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r3_07_joint_semantic_tf_qual | 10/10 | +0.000 | +0.175 | +0.256 | +0.324 | +0.097 | +0.000 | +0.000 | +0.000 | +0.000 | +0.361 | +0.379 | +0.333 |
| 2 | lit_r3_05_tf_re2_qual_verdict | 10/10 | +0.000 | +0.107 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.361 | +0.379 | +0.333 |
| 3 | lit_r3_04_tf_re2_verdict | 10/10 | +0.000 | +0.091 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.300 | +0.277 | +0.333 |
| 4 | lit_r3_02_mc_semantic_qual | 10/10 | +0.000 | +0.068 | +0.256 | +0.324 | +0.097 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | lit_r3_06_joint_pointwise_tf_qual | 9/10 | -0.029 | +0.138 | +0.135 | +0.206 | -0.029 | +0.000 | +0.000 | +0.000 | +0.000 | +0.361 | +0.379 | +0.333 |
| 6 | lit_r3_01_mc_pointwise_qual | 9/10 | -0.029 | +0.031 | +0.135 | +0.206 | -0.029 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 7 | lit_r3_03_mc_salient_aftermath | 7/10 | -0.147 | -0.029 | -0.085 | -0.059 | -0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| lit_r3_07_joint_semantic_tf_qual | 4.138 | 4.206 | 3.979 | 3.229 | 3.448 | 2.793 | 3.483 | 3.412 | 3.526 | 3.242 |
| lit_r3_05_tf_re2_qual_verdict | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.412 | 3.526 | 3.242 |
| lit_r3_04_tf_re2_verdict | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.352 | 3.424 | 3.242 |
| lit_r3_02_mc_semantic_qual | 4.138 | 4.206 | 3.979 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| lit_r3_06_joint_pointwise_tf_qual | 4.018 | 4.088 | 3.853 | 3.229 | 3.448 | 2.793 | 3.483 | 3.412 | 3.526 | 3.242 |
| lit_r3_01_mc_pointwise_qual | 4.018 | 4.088 | 3.853 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| lit_r3_03_mc_salient_aftermath | 3.797 | 3.824 | 3.735 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |

## Paired outcome counts

- `lit_r3_07_joint_semantic_tf_qual`: wins=40, ties=29, losses=27, correct→wrong=1, wrong→correct=16.
- `lit_r3_05_tf_re2_qual_verdict`: wins=22, ties=63, losses=11, correct→wrong=1, wrong→correct=14.
- `lit_r3_04_tf_re2_verdict`: wins=21, ties=63, losses=12, correct→wrong=1, wrong→correct=14.
- `lit_r3_02_mc_semantic_qual`: wins=18, ties=62, losses=16, correct→wrong=0, wrong→correct=2.
- `lit_r3_06_joint_pointwise_tf_qual`: wins=43, ties=29, losses=24, correct→wrong=1, wrong→correct=14.
- `lit_r3_01_mc_pointwise_qual`: wins=21, ties=62, losses=13, correct→wrong=0, wrong→correct=0.
- `lit_r3_03_mc_salient_aftermath`: wins=14, ties=62, losses=20, correct→wrong=1, wrong→correct=1.
