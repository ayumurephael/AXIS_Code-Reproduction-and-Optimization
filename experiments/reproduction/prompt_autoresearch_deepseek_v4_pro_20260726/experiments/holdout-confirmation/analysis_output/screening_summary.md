# Internal holdout confirmation results

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | route_r3_02_mc_f1_safe | 7/10 | -0.059 | -0.010 | -0.042 | -0.059 | -0.004 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 2 | route_r3_01_mc_f0_safe | 7/10 | -0.412 | -0.114 | -0.371 | -0.353 | -0.412 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.865 | 3.882 | 3.824 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| route_r3_02_mc_f1_safe | 3.822 | 3.824 | 3.820 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| route_r3_01_mc_f0_safe | 3.494 | 3.529 | 3.412 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |

## Paired outcome counts

- `route_r3_02_mc_f1_safe`: wins=9, ties=31, losses=8, correct→wrong=1, wrong→correct=1.
- `route_r3_01_mc_f0_safe`: wins=7, ties=31, losses=10, correct→wrong=1, wrong→correct=0.
