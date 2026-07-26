# Development round 2 results

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r2_04_oe_old_contract | 7/10 | -0.381 | +0.205 | +0.312 | +0.294 | +0.355 | -0.061 | -0.381 | -0.060 | +0.311 | +0.409 | +0.298 | +0.576 |
| 2 | route_r2_03_tf_boundary | 6/10 | -0.552 | -0.032 | +0.223 | +0.294 | +0.057 | -0.378 | -0.552 | -0.379 | -0.173 | +0.191 | +0.156 | +0.242 |
| 3 | route_r2_01_minimal | 6/10 | -0.621 | +0.011 | +0.229 | +0.265 | +0.147 | -0.460 | -0.621 | -0.517 | -0.207 | +0.421 | +0.398 | +0.455 |
| 4 | route_r2_02_mc_stable | 6/10 | -0.729 | -0.035 | +0.264 | +0.294 | +0.193 | -0.588 | -0.729 | -0.655 | -0.345 | +0.391 | +0.308 | +0.515 |
| 5 | route_r2_06_full_routed | 6/10 | -1.034 | -0.077 | +0.312 | +0.382 | +0.147 | -0.590 | -1.034 | -0.414 | -0.276 | +0.233 | +0.227 | +0.241 |
| 6 | route_r2_05_oe_contract_coverage | 6/10 | -1.069 | -0.073 | +0.256 | +0.265 | +0.235 | -0.643 | -1.069 | -0.517 | -0.291 | +0.337 | +0.282 | +0.420 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.882 | 3.882 | 3.882 | 3.229 | 3.448 | 2.793 | 3.483 | 3.052 | 3.147 | 2.909 |
| route_r2_04_oe_old_contract | 4.195 | 4.176 | 4.237 | 3.168 | 3.067 | 2.733 | 3.794 | 3.461 | 3.445 | 3.485 |
| route_r2_03_tf_boundary | 4.105 | 4.176 | 3.940 | 2.852 | 2.897 | 2.414 | 3.310 | 3.242 | 3.303 | 3.152 |
| route_r2_01_minimal | 4.112 | 4.147 | 4.029 | 2.769 | 2.828 | 2.276 | 3.276 | 3.473 | 3.545 | 3.364 |
| route_r2_02_mc_stable | 4.146 | 4.176 | 4.075 | 2.641 | 2.719 | 2.138 | 3.137 | 3.442 | 3.455 | 3.424 |
| route_r2_06_full_routed | 4.194 | 4.265 | 4.029 | 2.640 | 2.414 | 2.379 | 3.207 | 3.285 | 3.374 | 3.150 |
| route_r2_05_oe_contract_coverage | 4.138 | 4.147 | 4.118 | 2.587 | 2.379 | 2.276 | 3.191 | 3.389 | 3.429 | 3.329 |

## Paired outcome counts

- `route_r2_04_oe_old_contract`: wins=59, ties=0, losses=37, correct→wrong=1, wrong→correct=17.
- `route_r2_03_tf_boundary`: wins=46, ties=0, losses=50, correct→wrong=2, wrong→correct=15.
- `route_r2_01_minimal`: wins=53, ties=0, losses=43, correct→wrong=1, wrong→correct=17.
- `route_r2_02_mc_stable`: wins=49, ties=0, losses=47, correct→wrong=1, wrong→correct=18.
- `route_r2_06_full_routed`: wins=53, ties=0, losses=43, correct→wrong=2, wrong→correct=16.
- `route_r2_05_oe_contract_coverage`: wins=52, ties=0, losses=44, correct→wrong=1, wrong→correct=17.
