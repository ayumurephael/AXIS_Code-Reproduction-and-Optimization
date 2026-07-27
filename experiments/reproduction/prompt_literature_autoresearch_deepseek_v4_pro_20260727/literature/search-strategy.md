# Literature search strategy

## Scope

Research question: which inference-only prompt interventions are plausible for a fine-tuned 7B time-series QA model with MC, OE, and TF outputs, under a strict no-regression objective?

Date searched: 2026-07-27.

Sources were restricted to primary papers and official proceedings pages (ACL Anthology, PMLR, OpenReview/ICLR, NeurIPS proceedings, and arXiv author manuscripts when a proceedings version was unavailable).

## Query themes

- time-series LLM prompt-as-prefix and reprogramming;
- multiple-choice option order, label bias, symbol binding, and calibration;
- output-format instructions and correctness;
- prompt lexical sensitivity and robustness;
- question re-reading and repeated-query prompting;
- chain-of-thought scale dependence;
- black-box and small-model prompt optimization.

## Inclusion criteria

Included work had to provide an inference-time prompt or calibration mechanism relevant to at least one AXIS task family, or empirical evidence that such a mechanism can fail. Priority was given to 2021–2025 peer-reviewed work with ablations.

## Exclusion and transfer limits

- Probability calibration was retained as a future method but excluded from Round 1 because the current formal pipeline consumes generated answers rather than calibrated option logits.
- Self-consistency and System-2 multi-pass methods were excluded from the first prompt-only screen because they change inference cost and aggregation semantics.
- CoT was not used: its strongest evidence is on much larger models, while AXIS uses a fine-tuned 7B checkpoint and the current scoring protocol values concise, parseable answers.
- Evidence reordering/interleaving was excluded because the AXIS design note identifies it as requiring training alignment.
- Prompt-optimizer claims on large proprietary models were not transferred uncritically to the 7B checkpoint.

## Reproducibility

The synthesis records direct primary-source URLs/DOIs. Candidate-to-paper mappings are frozen in `candidate-matrix.md`; wording changes after observing results must receive new mode IDs.
