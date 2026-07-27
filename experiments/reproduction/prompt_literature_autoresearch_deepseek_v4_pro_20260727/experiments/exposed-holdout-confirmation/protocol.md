# Exposed holdout robustness protocol

## Status and interpretation

Preregistered after Literature Round 4 selection and before any new holdout inference, response inspection, or Judge call.

The 48-QA split was inspected by the preceding prompt search. It is therefore an **exposed robustness check**, not an untouched confirmation set. `paper140` remains reserved for one uniquely locked candidate.

## Frozen candidates

No prompt text, checkpoint, generation setting, lexical router, or ranking rule may change:

1. `lit_r3_02_mc_semantic_qual`
   - MC: compare complete option meanings, select by evidence, and justify qualitatively without unnecessary exact values or step numbers.
   - OE and TF: exact Baseline component.
2. `lit_r4_06_joint_semantic_tf_neg_prefix`
   - MC: exact same component as candidate 1.
   - TF with an explicit negative cue (`no`, `not`, `without`, `absence`, `lack`, `neither`, `nor`, `cannot`, `can't`, `doesn't`, `isn't`, `aren't`, `wasn't`, `weren't`): preserve proposition polarity, begin with `Answer: True.` or `Answer: False.`, then give qualitative support.
   - Other TF and every OE record: exact Baseline component.

The broader normality router and universal TF routes are ineligible because they fail a constituent guard or have a known positive-anomaly correct→wrong transition.

## Data and formal execution

- Manifest: `../../../prompt_autoresearch_deepseek_v4_pro_20260726/data/manifests/holdout.json`.
- Frozen size: 48 QA / 24 series; 17 MC, 15 OE, 16 TF.
- Record-id digest: `0f5260b05506f62de0d804b807f2a85a43b6a066c76bc52d652f6308c8d0c392`.
- Series digest: `0d5b7b8bbe38b8d8a7a458392689e62eee2cd080193db455bfffae36baa5586d`.
- Released checkpoint SHA-256: `d22a0ab930d3929e91090923e2046269f1a7cd28eb8de37ee8566c67db1a97a7`.
- Run raw MC semantic and TF-prefix source modes on three authorized GPUs with the released generation settings, series batching, beam size 5, and `--skip-loss`.
- Reuse the already audited exact holdout Baseline predictions and scores.
- Formally score only unique changed components with `deepseek-v4-pro`; assemble exact Baseline components for untouched task/question routes.
- Fail closed on missing or duplicate keys, component/prompt mismatch, unexpected model/provider, missing score, or unapproved scoring method.

## Pass gate and unique lock

A candidate passes only if:

1. all ten Table-I deltas versus the exact holdout Baseline are nonnegative;
2. MC, OE, and TF Final deltas are nonnegative;
3. no parsed MC/TF Baseline-correct decision becomes wrong;
4. every routed TF response has an unambiguous required prefix;
5. all provenance and Judge audits pass.

Rank passers by:

1. nonnegative dimensions;
2. worst delta;
3. mean delta over all ten dimensions;
4. number of closed-task wrong→correct transitions;
5. if still tied, prefer the shorter MC-only candidate.

Lock the unique top passer without further prompt edits and run it once on `paper140`. If neither passes, do not select by mean alone; use at most the remaining four preregistered-mode budget for one failure-driven cycle, otherwise stop under the registered search cap.
