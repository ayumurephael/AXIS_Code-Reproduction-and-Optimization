# Round 2 `new.md` supplemental compatibility experiment

## Scope

This supplement tests whether the five prompt interventions derived from
`new.md` are compatible with the released AXIS checkpoint **without
retraining**. It is not a test of the document's full Phase-II retraining
proposal: under the released causal prefix, the learned Local and Fixed
representations were produced with the original prompt neighborhood.

The frozen protocol, source hash, rendering rules, and promotion gate are in
`protocol.md`. GPU inference used the released checkpoint on the authorized
Port-2225 node. Formal G-Eval used only `deepseek-v4-pro`.

## Screening24 Table-I results

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.4880 | 3.3757 | 3.7500 | 2.6406 | 2.2857 | 2.1427 | 3.6357 | 3.4444 | 3.4444 | 3.4444 |
| P1 short compatibility rule | 3.6250 | 3.6250 | 3.6250 | 2.5000 | 2.4286 | 2.1429 | 3.0000 | 3.2000 | 3.3333 | 3.0000 |
| Δ P1 | +0.1370 | +0.2493 | -0.1250 | -0.1406 | +0.1429 | +0.0002 | -0.6357 | -0.2444 | -0.1111 | -0.4444 |
| P2 aligned Value-before-Local rows | 3.7500 | 3.7500 | 3.7500 | 2.5109 | 2.1429 | 2.4289 | 3.0357 | 3.3778 | 3.3333 | 3.4444 |
| Δ P2 | +0.2620 | +0.3743 | +0.0000 | -0.1298 | -0.1428 | +0.2862 | -0.6000 | -0.0667 | -0.1111 | -0.0000 |
| P3 Question-first | 3.2501 | 3.2501 | 3.2500 | 1.9572 | 1.4286 | 1.7143 | 2.8572 | 2.9592 | 3.0060 | 2.8890 |
| Δ P3 | -0.2379 | -0.1256 | -0.5000 | -0.6835 | -0.8571 | -0.4284 | -0.7785 | -0.4853 | -0.4385 | -0.5554 |
| P4 Fixed-first, evidence-last | 2.9250 | 3.0000 | 2.7500 | 2.1572 | 1.8571 | 1.8571 | 2.8573 | 2.7333 | 2.7778 | 2.6667 |
| Δ P4 | -0.5630 | -0.3757 | -1.0000 | -0.4835 | -0.4286 | -0.2856 | -0.7784 | -0.7111 | -0.6667 | -0.7778 |
| P5 full `new.md` prompt | 3.0375 | 3.0000 | 3.1250 | 2.1643 | 2.1429 | 1.7144 | 2.7143 | 2.9556 | 3.0001 | 2.8889 |
| Δ P5 | -0.4505 | -0.3757 | -0.6250 | -0.4763 | -0.1429 | -0.4283 | -0.9214 | -0.4888 | -0.4444 | -0.5556 |

None of the five modes satisfies all-ten-nonlower. Under the frozen protocol,
none advances, and Local corruption/swap sensitivity is not run because it
was preregistered only for a performance survivor.

## Output behavior

The following rates are diagnostics, not Table-I metrics.

| Mode | MC answer-first / strict | OE answer-first / strict | TF answer-first / strict | Mean OE chars |
|---|---:|---:|---:|---:|
| Baseline | 37.5% / 37.5% | 57.1% / 28.6% | 33.3% / 33.3% | 784.6 |
| P1 | 12.5% / 12.5% | 85.7% / 42.9% | 0.0% / 0.0% | 934.1 |
| P2 | 12.5% / 12.5% | 71.4% / 28.6% | 0.0% / 0.0% | 997.6 |
| P3 | 0.0% / 0.0% | 0.0% / 0.0% | 22.2% / 22.2% | 979.1 |
| P4 | 0.0% / 0.0% | 85.7% / 85.7% | 0.0% / 0.0% | 1179.3 |
| P5 | 62.5% / 62.5% | 42.9% / 42.9% | 11.1% / 11.1% | 1158.0 |

Formatting improvement is not sufficient evidence of answer-quality
improvement. P4, for example, makes OE parsing cleaner while every OE
Table-I dimension falls.

## Failure-case synthesis

### P1: a short semantic rule is still an active decision prior

P1 corrects some decisions, but it also changes the threshold for treating
ordinary variation as anomaly evidence. On `series_000112:1`, Baseline
correctly selects the normal option D; P1 selects A and calls a large value an
abrupt spike. Conversely, real-anomaly examples such as
`series_000132:1` and `series_000058:1` become false-normal. The sentence
about using Local context jointly does not decode the learned Local vectors;
it perturbs how the language model narrates ambiguous displayed values.

### P2: textual one-to-one alignment is not learned causal alignment

P2 has the best MC screening result and improves OE Completeness, but OE
Accuracy and Relevance fall. It corrects `series_000024:0` from D to C and
some TF normality decisions, yet still converts real anomalies to
false-normal. On `series_000083:1`, it selects the correct option C while
describing a negative value as an “upward spike.” The row labels make the
prompt legible to a human, but the released projection layers were not
trained to bind a newly interleaved text/value/token sequence.

### P3 and P4: causal reordering is out of distribution for this checkpoint

Question-first can only provide question-conditioned Local representations
after Phase-II is retrained with that prefix. Moving the question now changes
the LLM context while leaving the learned Local embeddings fixed. P3 produces
polarity contradictions such as `series_000043:0`: it begins `False` and then
argues that the negative proposition is true. P4 additionally produces
corrupted answer starts such as `-edge` and `tsy`, showing an answer-boundary
and prompt-neighborhood mismatch.

### P5: combined shifts compound rather than cancel

P5 changes the opening, role text, causal order, evidence layout, output
protocol, and answer prefill simultaneously. Its OE answers are longer but
less relevant; some duplicate “Final Answer,” discuss token alignment, or
otherwise expose prompt mechanics. It also repeats the false-normal failure
on real anomalies. This is consistent with a released-checkpoint
compatibility failure, not evidence against retraining the model under the
new Phase-II prefix.

## Conclusion

The only locally promising structural signal is P2's MC gain and OE
Completeness gain, but its large OE Relevance loss and accuracy failures make
it unsafe. The main prompt-search track should retain the original trained
scaffold and investigate short task-family protocols. The full
question-conditioned Local/Fixed information-flow proposal in `new.md`
requires retraining before it can be evaluated as intended.
