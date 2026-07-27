# Round 1 screening24 outer-loop analysis

This synthesis was written after all 164 `deepseek-v4-pro` component
dimensions completed and before any Round-1 validation72 Judge call.

## Paired Table-I result

Unchanged task families reuse the exact Baseline responses and Judge scores.
Values in parentheses are candidate minus Baseline.

| Mode | MC Final | MC Corr. | MC Rsn. | OE Final | OE Acc. | OE Comp. | OE Rel. | TF Final | TF Corr. | TF Justif. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 3.4880 | 3.3757 | 3.7500 | 2.6406 | 2.2857 | 2.1427 | 3.6357 | 3.4444 | 3.4444 | 3.4444 |
| `v2_r1_01_oe_context_balanced` | = | = | = | 2.8786 (+0.2379) | 2.8571 (+0.5714) | 2.4286 (+0.2859) | 3.4286 (-0.2071) | = | = | = |
| `v2_r1_02_oe_context_contrast` | = | = | = | 2.6929 (+0.0522) | 3.0000 (+0.7143) | 2.0000 (-0.1427) | 3.1429 (-0.4929) | = | = | = |
| `v2_r1_03_oe_evidence_router` | = | = | = | 2.6857 (+0.0450) | 2.7142 (+0.4285) | 2.1429 (+0.0002) | 3.2857 (-0.3500) | = | = | = |
| `v2_r1_04_oe_context_direct` | = | = | = | 2.9132 (+0.2726) | 3.1429 (+0.8571) | 2.3643 (+0.2217) | 3.2857 (-0.3500) | = | = | = |
| `v2_r1_05_oe_fixed_postnote` | = | = | = | 2.1243 (-0.5164) | 1.8572 (-0.4286) | 1.8857 (-0.2570) | 2.7143 (-0.9214) | = | = | = |
| `v2_r1_06_mc_context_balanced` | 4.6375 (+1.1495) | 4.7500 (+1.3743) | 4.3750 (+0.6250) | = | = | = | = | = | = | = |
| `v2_r1_07_mc_content_output` | 4.1375 (+0.6495) | 4.2500 (+0.8743) | 3.8750 (+0.1250) | = | = | = | = | = | = | = |
| `v2_r1_08_tf_context_whole` | = | = | = | = | = | = | = | 3.6221 (+0.1776) | 3.4444 (approximately equal) | 3.8885 (+0.4441) |
| `v2_r1_09_tf_prefix_context` | = | = | = | = | = | = | = | 3.0889 (-0.3556) | 3.0000 (-0.4444) | 3.2222 (-0.2222) |

## Failure-case synthesis

### MC: the short balanced rule repairs content decisions

- `series_000024:0`: Baseline chose D and treated ordinary positive/negative
  variation as erratic anomaly evidence. `mc_context_balanced` chose the gold
  C and explained that the changes were moderate rather than isolated
  deviations. The paired final score rose by 3.70.
- `series_000091:1`: Baseline focused on a single upward point and chose B.
  The candidate selected the gold broad downward dip C and described its
  duration as well as amplitude. The paired final score rose by 3.70.
- The only material loss was `series_000115:0`: both prompts chose the correct
  normal option B, but the candidate's explanation received one lower
  reasoning point. This is a specificity loss, not a decision reversal.

Interpretation: the instruction to judge the option's complete description
against both window shape and Per-Step context reduces the model's tendency
to match one salient point to one option. It is promising but the unusually
large small-screen gain must be tested on validation72.

### OE: anomaly accuracy and question relevance remain in tension

- `series_000024:1`: `oe_context_direct` changed a false "no anomaly" answer
  into a correct downward-boundary-spike assessment. Accuracy rose from about
  1 to 3, but relevance fell from 5 to 3 because the answer did not fully
  address both downward and upward movements and boundary interpretation.
- `series_000043:1`: all short evidence-use variants retained the Baseline's
  false-normal decision for a local drop followed by a sustained lower level.
  The direct candidate also shortened the multi-part evidence discussion;
  completeness and relevance fell.
- `series_000058:0`: the direct rule improved a genuinely normal boundary
  answer by stating the boundary values and giving a direct conclusion;
  accuracy/completeness rose without a relevance loss.

Interpretation: the added rule can redirect the decision, but it does not
reliably decode a subtle latent anomaly. When it prioritizes a direct verdict,
it may omit the question's requested analytical method, boundary comparison,
or multiple anomaly types. The next prompt cycle should separate two axes:
(a) evidence arbitration, and (b) coverage of explicit question subparts.
A universal longer contract would repeat an already falsified failure.

### Fixed-token role explanation is not neutral

`oe_fixed_postnote` reduced every OE dimension. In `series_000083:0`, the
candidate became shorter, omitted requested analytical methods, and scored
lower on accuracy, completeness, and relevance. In `series_000024:1`, it kept
the false-normal decision while relevance fell from 5 to 2.

Interpretation: even when the trained `Overall Summary Hints` prefix is left
in place, explicitly telling the released checkpoint that those soft tokens
are "not sample-specific evidence" changes how it uses a learned internal
representation. This candidate is rejected and the Fixed role wording will
not be repeated without retraining.

### TF: balanced whole-statement use helps, rigid prefixing hurts

- `series_000091:0`: `tf_context_whole` corrected Baseline False to the gold
  True for a normal fluctuating window; paired final rose by 1.60.
- `series_000008:1`: it strengthened a correct normal verdict and raised both
  dimensions from 3 to 5.
- `series_000132:1`: it also changed a correct True anomaly verdict to False,
  over-trusting the smooth-looking values; correctness fell from 4 to 2.
- `tf_prefix_context` reduced all three aggregate TF metrics, including a
  True-to-False failure on `series_000058:1`. The extra prefix requirement
  consumes control capacity without resolving evidence conflict.

Interpretation: preserve-negation/whole-statement guidance is useful, but the
candidate still needs broader validation. Exact answer-prefix instructions
are rejected for this checkpoint.

## Wide-screen decision

To avoid false rejection on only 24 QA, a mode is discarded here only when
all of its changed-family Table-I dimensions decline. Under this deliberately
permissive rule:

- advance: `v2_r1_01`, `v2_r1_02`, `v2_r1_03`, `v2_r1_04`,
  `v2_r1_06`, `v2_r1_07`, and `v2_r1_08`;
- reject: `v2_r1_05_oe_fixed_postnote` and
  `v2_r1_09_tf_prefix_context`.

No screening result is treated as a final improvement claim.
