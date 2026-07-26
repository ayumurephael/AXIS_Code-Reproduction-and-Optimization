# Screening round 1 results

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | pareto_p07_abd | 7/10 | -0.457 | +0.128 | +0.081 | +0.038 | +0.181 | -0.271 | -0.457 | -0.318 | +0.001 | +0.662 | +0.575 | +0.792 |
| 2 | fixed_role | 6/10 | -0.318 | +0.173 | +0.232 | +0.298 | +0.079 | -0.191 | -0.318 | -0.227 | -0.000 | +0.604 | +0.506 | +0.750 |
| 3 | pareto_p01_boundary | 5/10 | -0.182 | +0.063 | +0.042 | +0.077 | -0.038 | -0.109 | -0.182 | -0.091 | -0.045 | +0.304 | +0.173 | +0.500 |
| 4 | pareto_p03_task_rule | 4/10 | -0.727 | -0.221 | +0.042 | +0.077 | -0.038 | -0.509 | -0.727 | -0.727 | +0.000 | -0.121 | -0.202 | +0.000 |
| 5 | pareto_p06_abc | 3/10 | -0.545 | -0.173 | -0.093 | -0.038 | -0.219 | -0.432 | -0.545 | -0.455 | -0.273 | +0.112 | +0.131 | +0.083 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.938 | 3.962 | 3.885 | 3.450 | 3.727 | 3.091 | 3.545 | 2.946 | 3.077 | 2.750 |
| pareto_p07_abd | 4.020 | 4.000 | 4.065 | 3.179 | 3.270 | 2.773 | 3.547 | 3.608 | 3.652 | 3.542 |
| fixed_role | 4.171 | 4.260 | 3.963 | 3.259 | 3.409 | 2.864 | 3.545 | 3.550 | 3.583 | 3.500 |
| pareto_p01_boundary | 3.981 | 4.039 | 3.846 | 3.341 | 3.545 | 3.000 | 3.500 | 3.250 | 3.250 | 3.250 |
| pareto_p03_task_rule | 3.981 | 4.038 | 3.846 | 2.941 | 3.000 | 2.364 | 3.545 | 2.825 | 2.875 | 2.750 |
| pareto_p06_abc | 3.846 | 3.923 | 3.665 | 3.018 | 3.182 | 2.636 | 3.273 | 3.058 | 3.208 | 2.833 |

## Paired outcome counts

- `pareto_p07_abd`: wins=44, ties=0, losses=28, correct→wrong=0, wrong→correct=5.
- `fixed_role`: wins=44, ties=0, losses=28, correct→wrong=1, wrong→correct=7.
- `pareto_p01_boundary`: wins=39, ties=0, losses=33, correct→wrong=0, wrong→correct=3.
- `pareto_p03_task_rule`: wins=33, ties=0, losses=39, correct→wrong=4, wrong→correct=10.
- `pareto_p06_abc`: wins=35, ties=0, losses=37, correct→wrong=4, wrong→correct=10.
