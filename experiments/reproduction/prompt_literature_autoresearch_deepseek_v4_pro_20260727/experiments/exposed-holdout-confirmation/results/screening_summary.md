# Literature-guided exposed holdout robustness

## Delta versus split Baseline

| Rank | Mode | Nonneg. | Worst | Mean | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lit_r4_06_joint_semantic_tf_neg_prefix | 7/10 | -0.118 | +0.058 | -0.076 | -0.059 | -0.118 | +0.000 | +0.000 | +0.000 | +0.000 | +0.275 | +0.250 | +0.313 |
| 2 | lit_r3_02_mc_semantic_qual | 7/10 | -0.118 | -0.025 | -0.076 | -0.059 | -0.118 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

## Absolute scores

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 3.865 | 3.882 | 3.824 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |
| lit_r4_06_joint_semantic_tf_neg_prefix | 3.788 | 3.824 | 3.706 | 2.876 | 2.703 | 2.600 | 3.400 | 3.688 | 3.688 | 3.687 |
| lit_r3_02_mc_semantic_qual | 3.788 | 3.824 | 3.706 | 2.876 | 2.703 | 2.600 | 3.400 | 3.413 | 3.438 | 3.375 |

## Paired outcome counts

- `lit_r4_06_joint_semantic_tf_neg_prefix`: wins=18, ties=21, losses=9, correct→wrong=2, wrong→correct=4.
- `lit_r3_02_mc_semantic_qual`: wins=10, ties=31, losses=7, correct→wrong=2, wrong→correct=1.
