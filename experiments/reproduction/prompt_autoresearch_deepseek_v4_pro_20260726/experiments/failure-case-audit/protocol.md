# Failure-case audit protocol

Status: diagnostic bootstrap; analysis of already completed experiments, not a new prompt trial.

## Comparisons

1. `fixed_role_evidence_contract` minus `fixed_role`
2. `fixed_role_evidence_contract_revised` minus `fixed_role`
3. `fixed_role_evidence_contract_revised` minus `fixed_role_evidence_contract`

## Predefined outputs

- Per-record weighted Final and per-dimension score deltas.
- Largest losses and gains within MC, OE, and TF.
- Answer-decision changes for MC/TF.
- Response-level inspection of representative cases, preserving the same question, answer, and time-series window.
- Failure taxonomy grounded in the response text and Judge dimension changes.

## Selection rule

For each task family and comparison, rank by paired weighted-Final delta. Inspect the largest losses, then retain representative cases only when the response text exposes a concrete mechanism. Also inspect OE gains under the old Contract to distinguish useful evidence elicitation from generic verbosity.

## Interpretation limits

- A case illustrates a mechanism but does not estimate its population frequency by itself.
- Judge explanations are supporting evidence, not ground truth.
- Multiple clauses changed in the revised Contract, so clause-level causality requires future orthogonal experiments.
