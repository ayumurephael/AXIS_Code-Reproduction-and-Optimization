# Internal holdout confirmation analysis

## Decision

Neither frozen candidate confirmed. Both fail the preregistered all-ten
nonnegative gate and both introduce one Baseline-correct→candidate-wrong MC
decision. No winner is locked and `paper140` remains untouched.

This is the preregistered stopping condition. The holdout is not reused for
prompt tuning and the numerically less-negative candidate is not promoted.

## Formal results

The split contains 48 QA / 24 series: 17 MC, 15 OE, and 16 TF. Candidate OE
and TF components are exact copies of the split Baseline by construction, so
their seven Table-I deltas are exactly zero.

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Holdout Baseline | 3.864701 | 3.882353 | 3.823513 | 2.876234 | 2.703335 | 2.600190 | 3.400000 | 3.412505 | 3.437510 | 3.374997 |
| R3-02 MC-F1 safe | 3.822412 | 3.823530 | 3.819803 | 2.876234 | 2.703335 | 2.600190 | 3.400000 | 3.412505 | 3.437510 | 3.374997 |
| R3-01 MC-F0 safe | 3.494119 | 3.529412 | 3.411768 | 2.876234 | 2.703335 | 2.600190 | 3.400000 | 3.412505 | 3.437510 | 3.374997 |

### Absolute and relative deltas versus the holdout Baseline

| Mode | Metric | Absolute delta | Relative change |
|---|---|---:|---:|
| R3-02 | MC Final | -0.042289 | -1.094% |
| R3-02 | MC Correctness | -0.058823 | -1.515% |
| R3-02 | MC Reasoning | -0.003710 | -0.097% |
| R3-01 | MC Final | -0.370582 | -9.589% |
| R3-01 | MC Correctness | -0.352940 | -9.091% |
| R3-01 | MC Reasoning | -0.411745 | -10.769% |
| Both | all seven OE/TF metrics | +0.000000 | +0.000% |

R3-02 ranks above R3-01 but reaches only 7/10 nonnegative dimensions; its
worst delta is -0.058823 and its mean ten-metric delta is -0.010482. R3-01
also reaches 7/10, with worst delta -0.411745 and mean delta -0.113527.
Ranking above another failing candidate is not confirmation.

## Paired outcomes

| Mode | Wins | Ties | Losses | Correct→wrong | Wrong→correct |
|---|---:|---:|---:|---:|---:|
| R3-02 | 9 | 31 | 8 | 1 | 1 |
| R3-01 | 7 | 31 | 10 | 1 | 0 |

The 31 ties per candidate are the reused OE/TF records. The MC-only paired
counts therefore expose the complete behavior of the intervention rather than
being diluted by the conservative component reuse.

## Failure cases and mechanisms

### R3-02: stable-selection wording fixes one contradiction but creates a false positive

Positive case `series_000022:1` is a real localized oscillation. Baseline chose
option D and then said that none of the options described an anomaly. R3-02
chose the gold option B and kept the explanation consistent with it, improving
the record Final from 1.000001 to 4.700000. This confirms the intended
mechanism: the stable-selection rule can suppress answer revision and
label/rationale contradiction.

The symmetric failure is `series_000141:1`, a normal window. Baseline correctly
chose A and described gradual, ordinary fluctuation. R3-02 instead anchored on
option D, reinterpreting alternating values as “sharp drops” with immediate
recoveries and hence as isolated anomalies. The record Final fell from
4.700000 to 1.299998 and the parsed decision changed A→D.

The rule controls *commitment* after selection but does not improve the
evidence-to-option decision itself. In an ambiguous normal window it freezes
an early false-positive interpretation. It also slightly shortens otherwise
correct explanations: several unchanged-choice records lose roughly one
reasoning point because useful normality or competitor evidence is omitted.
The one large correction and one large regression almost cancel, while the
smaller reasoning losses make all three aggregate MC dimensions negative.

### R3-01: renaming the learned-token role is not semantics-preserving

On `series_000139:0`, Baseline correctly chose D (normal behavior) and its
explanation was fully consistent. R3-01 chose C (irregular anomalous
fluctuations), but its own explanation said the behavior was normal and ended
with “no evidence of anomalous behavior.” Final fell from 4.999999 to
1.300001 and the parsed decision changed D→C.

This is not a formatting-only failure. The 30 learned fixed tokens were trained
under the released `Overall Summary Hints` header. Relabeling them as
`Learned Task Guidance/Shared Task-Control Tokens` changes their textual
conditioning and can decouple option selection from the following rationale.
The holdout case directly refutes the assumption that a more accurate
human-readable role name is behaviorally neutral for the released checkpoint.

R3-01 has no compensating wrong→correct decision and also lowers reasoning on
several already-correct records. Its development improvement therefore does
not generalize.

## Outer-loop synthesis

Across all rounds, longer semantic contracts and positive OE obligations
change content decisions more often than they improve evidence coverage.
Question-type routing successfully prevents OE/TF collateral damage, but it
does not make the MC intervention distributionally robust. The released model
is strongly co-adapted to its training prompt:

1. Header and role renames can alter how learned latent tokens are decoded.
2. Output-consistency rules can prevent revision but may preserve the wrong
   initial interpretation.
3. Explicit anomaly-language additions prime false positives on normal,
   high-variance windows.
4. Development aggregate gains can be driven by a few corrected catastrophic
   cases and fail to survive a series-disjoint holdout.
5. Exact Baseline reuse is the only intervention that robustly preserves OE
   and TF, but no tested MC text change preserves all three MC dimensions.

The strict all-ten requirement is therefore not met by any tested prompt.
Further prompt-only search would require a new untouched selection resource;
reusing this holdout would invalidate its confirmatory role. Under the locked
protocol, the appropriate result is a negative confirmation, not post-hoc
tuning.

## Integrity and execution audit

- Released checkpoint SHA-256:
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- Manifest SHA-256:
  `cbd4fff8fa79c83bb20a09fcb0e6c84f57f74f65c4a52e6c47dfbe4d4f242d11`.
- Successful inference: 144 predictions, three GPUs, series batching, beam
  size 5, `skip_loss=true`.
- Selective Judge input: 82 answers and 179 dimensions.
- Judge: only `deepseek-v4-pro`; 178 primary probability scores and one
  exact-20 fallback.
- Selective component audit: all three components pass; no fatal Judge log
  lines.
- Assembled audit: 144 predictions, 333 dimension scores, 330 probability
  rows plus three copied exact-20 rows, valid prompt hashes on all rows, no
  missing/duplicate keys, and no provider/model mismatch.
- The first GPU attempt produced zero predictions after an unrelated process
  occupied GPU 0 during model loading. Its log and preflight are retained; the
  successful retry used the unchanged preregistered three-GPU configuration.

## Artifact index

- `artifacts/predictions.jsonl`: successful raw inference.
- `artifacts/run_manifest.json`: successful inference manifest.
- `artifacts/prediction_audit.json`: raw prediction audit.
- `artifacts/inference_three_gpu_failed.log` and
  `artifacts/environment_preflight_failed.txt`: retained zero-output attempt.
- `artifacts/inference_retry.log` and
  `artifacts/environment_preflight_retry.txt`: successful retry evidence.
- `artifacts/judge/judged_predictions.jsonl`: selective Judge input.
- `artifacts/judge/scores.jsonl`: 179 new formal scores.
- `artifacts/judge/audit.json`: three-component Judge audit.
- `artifacts/assembled_predictions.jsonl` and
  `artifacts/assembled_scores.jsonl`: full conservative comparison.
- `artifacts/assembled_audit.json`: final fail-closed audit.
- `analysis_output/screening_summary.json`: exact metrics and ranking.
- `analysis_output/paired_cases.jsonl`: full paired case audit.
