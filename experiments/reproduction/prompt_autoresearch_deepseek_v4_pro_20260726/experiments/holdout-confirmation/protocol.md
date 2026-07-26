# Internal holdout confirmation protocol

## Status

Preregistered after development Round 3 selection and before any holdout
inference, prediction inspection, or Judge call.

## Frozen candidates

No prompt text, checkpoint, generation setting, or selection rule may change:

1. `route_r3_02_mc_f1_safe`: stable-selection MC; exact Baseline OE and TF.
2. `route_r3_01_mc_f0_safe`: fixed-role-only MC; exact Baseline OE and TF.

R3-03/R3-04 are not eligible despite 10/10 aggregate metrics because they
introduce an unresolved Baseline-correct→candidate-wrong decision on a real TF
anomaly (`series_000132:1`). Identifying the mechanism does not justify the
harmful decision.

## Data isolation

- Manifest: `../../data/manifests/holdout.json`.
- Manifest file SHA-256:
  `cbd4fff8fa79c83bb20a09fcb0e6c84f57f74f65c4a52e6c47dfbe4d4f242d11`.
- Frozen size: 48 QA / 24 series; 17 MC, 15 OE, 16 TF.
- No question, label, model response, or Judge result from this split was used
  in prompt design.
- `paper140` remains reserved for the single locked winner.

## Formal execution

- Released author checkpoint only; verify SHA-256
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- Three approved GPUs, series batching, beam size 5, and `--skip-loss`.
- Infer Baseline plus both frozen routes: 3 × 48 = 144 predictions.
- The exact Baseline response is the controlled component for candidate OE/TF.
  Newly score all 48 Baseline rows and the 17 MC rows from each candidate:
  82 Judge inputs and 179 dimensions.
- Judge only with `deepseek-v4-pro`; use final-score token probabilities and
  exact-20 fallback where required.
- Fail closed on missing/duplicate rows, component mismatch, nonmatching
  provider/model, invalid prompt hashes, or unapproved score methods.

## Confirmation and winner lock

A candidate confirms only if all ten Table-I deltas versus the holdout
Baseline are nonnegative, all three task Finals are nonnegative, and parsed MC
has no correct-to-wrong increase. Rank confirming candidates by:

1. number of nonnegative dimensions;
2. minimum delta;
3. mean delta across all ten metrics;
4. MC Final, then MC Correctness, then MC Reasoning;
5. if still tied, prefer the shorter F0 intervention.

Lock the top confirming candidate without further prompt changes and run that
one candidate once on `paper140`. If neither confirms, do not select using
holdout; report confirmation failure and stop rather than tuning on holdout.
