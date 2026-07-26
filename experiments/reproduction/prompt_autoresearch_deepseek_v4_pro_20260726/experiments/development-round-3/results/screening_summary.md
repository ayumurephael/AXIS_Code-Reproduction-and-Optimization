# Development round 3 results

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r3_04_mc_f1_tf_p0 | 10/10 | +0.000 | +0.202 | +0.264 | +0.294 | +0.193 | +0.000 | +0.000 | +0.000 | +0.000 | +0.421 | +0.398 | +0.455 |
| 2 | route_r3_03_mc_f0_tf_p0 | 10/10 | +0.000 | +0.192 | +0.229 | +0.265 | +0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.421 | +0.398 | +0.455 |
| 3 | route_r3_02_mc_f1_safe | 10/10 | +0.000 | +0.075 | +0.264 | +0.294 | +0.193 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 4 | route_r3_01_mc_f0_safe | 10/10 | +0.000 | +0.064 | +0.229 | +0.265 | +0.147 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | route_r3_08_historical | 7/10 | -0.381 | +0.045 | +0.229 | +0.265 | +0.147 | -0.061 | -0.381 | -0.060 | +0.311 | +0.000 | +0.000 | +0.000 |
| 6 | route_r3_07_oe_q3 | 7/10 | -0.552 | +0.098 | +0.229 | +0.265 | +0.147 | -0.247 | -0.552 | -0.241 | +0.103 | +0.421 | +0.398 | +0.455 |
| 7 | route_r3_05_oe_q1 | 6/10 | -0.378 | +0.105 | +0.229 | +0.265 | +0.147 | -0.221 | -0.378 | -0.165 | -0.103 | +0.421 | +0.398 | +0.455 |
| 8 | route_r3_06_oe_q2 | 6/10 | -0.586 | +0.071 | +0.229 | +0.265 | +0.147 | -0.312 | -0.586 | -0.276 | -0.034 | +0.421 | +0.398 | +0.455 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_04_mc_f1_tf_p0 | 4.146 | 4.176 | 4.075 | 3.229 | 3.448 | 2.793 | 3.483 | 3.473 | 3.545 | 3.364 |
| route_r3_03_mc_f0_tf_p0 | 4.112 | 4.147 | 4.029 | 3.229 | 3.448 | 2.793 | 3.483 | 3.473 | 3.545 | 3.364 |
| route_r3_02_mc_f1_safe | 4.146 | 4.176 | 4.075 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_01_mc_f0_safe | 4.112 | 4.147 | 4.029 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r3_08_historical | 4.112 | 4.147 | 4.029 | 3.168 | 3.067 | 2.733 | 3.794 | 3.052 | 3.147 | 2.909 |
| route_r3_07_oe_q3 | 4.112 | 4.147 | 4.029 | 2.983 | 2.897 | 2.552 | 3.586 | 3.473 | 3.545 | 3.364 |
| route_r3_05_oe_q1 | 4.112 | 4.147 | 4.029 | 3.008 | 3.071 | 2.628 | 3.379 | 3.473 | 3.545 | 3.364 |
| route_r3_06_oe_q2 | 4.112 | 4.147 | 4.029 | 2.917 | 2.862 | 2.517 | 3.448 | 3.473 | 3.545 | 3.364 |

## Paired outcome counts

- `route_r3_04_mc_f1_tf_p0`: wins=40, ties=29, losses=27, correct→wrong=1, wrong→correct=18.
- `route_r3_03_mc_f0_tf_p0`: wins=43, ties=29, losses=24, correct→wrong=1, wrong→correct=17.
- `route_r3_02_mc_f1_safe`: wins=16, ties=62, losses=18, correct→wrong=0, wrong→correct=3.
- `route_r3_01_mc_f0_safe`: wins=19, ties=62, losses=15, correct→wrong=0, wrong→correct=2.
- `route_r3_08_historical`: wins=36, ties=33, losses=27, correct→wrong=0, wrong→correct=2.
- `route_r3_07_oe_q3`: wins=56, ties=0, losses=40, correct→wrong=1, wrong→correct=17.
- `route_r3_05_oe_q1`: wins=59, ties=0, losses=37, correct→wrong=1, wrong→correct=17.
- `route_r3_06_oe_q2`: wins=55, ties=0, losses=41, correct→wrong=1, wrong→correct=17.
