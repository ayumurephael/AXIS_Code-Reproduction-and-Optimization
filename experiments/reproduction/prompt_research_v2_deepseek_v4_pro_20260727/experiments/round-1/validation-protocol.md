# Round 1 validation72 protocol

Frozen after screening24 analysis and before validation72 Judge calls.

## Candidates

The seven wide-screen survivors are:

1. `v2_r1_01_oe_context_balanced`
2. `v2_r1_02_oe_context_contrast`
3. `v2_r1_03_oe_evidence_router`
4. `v2_r1_04_oe_context_direct`
5. `v2_r1_06_mc_context_balanced`
6. `v2_r1_07_mc_content_output`
7. `v2_r1_08_tf_context_whole`

The Fixed-role postnote and TF exact-prefix candidates are rejected because
all of their changed-family screening metrics declined. They will not be
scored on validation72.

## Frozen evaluation

- Generate no new answers: reuse the already completed GPU outputs from source
  commit `63e627b`.
- Score only the changed validation72 components with
  `deepseek-v4-pro`, using the same G-Eval rubrics, 4096-token Judge budget,
  probability-weighted 1–5 score readout, and registered exact-20 fallback.
- Reuse canonical Baseline answers and scores for unchanged task families.
- Report validation72 alone and pooled development96
  (screening24 + validation72).
- Treat a component as promising only if all dimensions of its changed family
  are nonlower on pooled development96. OE advancement remains mandatory for
  an eventual comprehensive route.
- Perform a second outer-loop failure-case synthesis before defining any new
  prompt wording or exposed-holdout48 route.
