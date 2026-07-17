# Loss-redesign Phase-II results

This directory contains the audited formal run for the redesigned Phase-II
objective on branch `loss_resesign`.

## Protocol

- Seed: 72
- Training: 3 epochs, 3 x A100 40GB, 28,500 optimizer steps
- Runtime: 43,915.72 seconds (12.20 hours)
- Checkpoint selection: strict minimum complete 1,500-series / 3,000-QA
  teacher-forced validation NLL
- Test inference: selected checkpoint on all 284 released QA records, followed
  by fixed-manifest filtering to the paper140 subset
- Judge: strict Appendix-E `deepseek-v4-pro`, maximum 4,096 judge tokens,
  probability scoring when all five final score-token probabilities were
  available, otherwise exactly 20 fallback samples

## Validation selection

| Epoch | Validation rows | Mean teacher-forced NLL |
| ---: | ---: | ---: |
| 1 | 3,000 | 0.9011898744 |
| 2 | 3,000 | 0.8708214486 |
| **3** | **3,000** | **0.8556907533** |

Epoch 3 was selected without consulting test predictions or judge scores.
All three validation outputs passed coverage audits (3,000 rows, 1,500
series, two records per series, no duplicates or empty responses).

## Table 1 metrics

| MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
| -------: | -------: | ------: | -------: | ------: | -------: | ------: | -------: | -------: | ---------: |
| 3.297675 | 3.883722 | 1.930234 | 2.867081 | 2.707734 | 2.561849 | 3.409091 | 2.881430 | 2.945242 | 2.785713 |

The macro average of the three task-type final scores is 3.015396.

## Integrity summary

- full284 inference: 284 predictions / 284 records, audit passed
- paper140 inference: 140 predictions / 140 records, audit passed
- G-Eval: 335 unique dimension scores, audit passed
- Final score-token probability rows: 330
- Exact 20-sample fallback rows: 5
- Duplicate or missing score keys: 0
- Score model mismatch, invalid score, or invalid probability distribution: 0

Machine-readable coverage reports, hashes, predictions, judge scores, and
aggregations are stored alongside this file.
