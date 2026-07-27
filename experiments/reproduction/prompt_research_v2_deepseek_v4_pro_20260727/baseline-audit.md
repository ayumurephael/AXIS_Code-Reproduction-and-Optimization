# Baseline discrepancy audit

## Conclusion

The two reported Baselines are both internally valid, but they summarize
different datasets. The historical values are the author-compatible
`paper140` result; the later values are a newly judged `full284` result.

## Evaluation-set comparison

| View | QA | Series | MC | OE | TF |
|---|---:|---:|---:|---:|---:|
| `paper140` | 140 | 70 | 43 | 55 | 42 |
| non-paper remainder | 144 | 72 | 51 | 44 | 49 |
| `full284` | 284 | 142 | 94 | 99 | 91 |

## Baseline scores

| Baseline view | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Historical `paper140` Judge run | 4.2558 | 4.3256 | 4.0930 | 3.0859 | 2.8365 | 2.7700 | 3.7455 | 3.6421 | 3.6417 | 3.6429 |
| Current full-run outputs, rescored on the same `paper140` records | 4.2269 | 4.2942 | 4.0698 | 3.1106 | 2.9527 | 2.7709 | 3.6909 | 3.6333 | 3.6429 | 3.6190 |
| Current non-paper 144 QA | 3.8824 | 3.8825 | 3.8823 | 3.1962 | 3.1330 | 2.7954 | 3.7375 | 3.1302 | 3.1490 | 3.1020 |
| Current `full284` | 4.0400 | 4.0708 | 3.9681 | 3.1486 | 3.0328 | 2.7818 | 3.7116 | 3.3624 | 3.3769 | 3.3407 |

The MC and TF drop is driven by the non-paper 144 QA. OE happens to be
slightly higher on that remainder, which is why the full284 OE Final is not
lower than the historical paper140 OE Final.

## Exact-overlap audit

For the 140 overlapping records:

- same question: 140/140;
- same gold answer: 140/140;
- same Baseline prompt text and prompt SHA-256: 140/140;
- same generated Baseline response: 140/140;
- same released checkpoint SHA-256:
  `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.

This rules out a changed Baseline prompt, checkpoint, or model response as the
cause of the large `paper140` versus `full284` difference.

## Judge-repeat audit

The two paper140 Judge runs used `deepseek-v4-pro`, the same system
fingerprint, and identical score-prompt hashes for 335/335
record-dimension keys. Nevertheless:

- only 239/335 raw integer ratings match;
- mean absolute difference of the probability-weighted scores is 0.3437;
- the largest per-dimension score difference is about 3.0;
- 328/335 keys used the same registered readout method in both runs.

The aggregate differences are much smaller than the per-record variation
because positive and negative changes partly cancel. Finalists must therefore
be judged in a paired repeated audit; a one-off difference of a few hundredths
must not be treated as a reliable Prompt gain.
