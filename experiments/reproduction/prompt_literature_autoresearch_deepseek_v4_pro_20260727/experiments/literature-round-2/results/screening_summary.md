# Literature Round 2 validation results

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r1_01_mc_semantic_bind | 9/10 | -0.077 | +0.010 | +0.058 | +0.115 | -0.077 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | lit_r1_02_mc_pointwise | 9/10 | -0.231 | +0.003 | +0.065 | +0.192 | -0.231 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 3 | lit_r2_01_mc_tf_re2_safe | 8/10 | -0.154 | +0.128 | -0.019 | +0.038 | -0.154 | +0.000 | +0.000 | +0.000 | +0.000 | +0.445 | +0.298 | +0.667 |
| 4 | lit_r1_03_mc_re2 | 8/10 | -0.154 | -0.013 | -0.019 | +0.038 | -0.154 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r1_01_mc_semantic_bind | 3.996 | 4.077 | 3.808 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r1_02_mc_pointwise | 4.004 | 4.154 | 3.654 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| lit_r2_01_mc_tf_re2_safe | 3.919 | 4.000 | 3.731 | 3.450 | 3.727 | 3.091 | 3.545 | 3.392 | 3.375 | 3.417 |
| lit_r1_03_mc_re2 | 3.919 | 4.000 | 3.731 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |

## Paired outcome counts

- `lit_r1_01_mc_semantic_bind`: wins=13, ties=46, losses=13, correct→wrong=1, wrong→correct=1.
- `lit_r1_02_mc_pointwise`: wins=12, ties=46, losses=14, correct→wrong=0, wrong→correct=1.
- `lit_r2_01_mc_tf_re2_safe`: wins=26, ties=22, losses=24, correct→wrong=2, wrong→correct=10.
- `lit_r1_03_mc_re2`: wins=10, ties=46, losses=16, correct→wrong=1, wrong→correct=1.
