# Research log

## 2026-07-27 — Protocol reset requested by user

Removed `zero correct→wrong` from the selection and final success criteria.
Parsed MC/TF transitions remain diagnostic only.

Verified from the paper and author-compatible code that paper Table I is the
`paper140`/Gemini path, while the released test directory contains 284 QA.
The requested final experiment is frozen as full-284 with a newly generated
released-checkpoint Baseline and `deepseek-v4-pro` for both Baseline and
candidates.

Audited the complete 69-mode implementation registry and historical result
families. Known failures will not be rerun. Exact Baseline OE routes cannot
meet the user's OE-change requirement.

The deliberately wide historical rule advances exactly two unresolved
OE-changing candidates:

- `lit_r1_08_oe_re2`
- `lit_r1_10_triplet_re2`

Both had 9/10 nonnegative metrics on screening24, positive mean improvement,
OE Final/Accuracy/Completeness gains, and a single OE Relevance delta of
-0.1429. Neither received a later broader evaluation, because the previous
per-case gate stopped them.



## 2026-07-27 — Full-284 completion

Generated 568 raw responses on the authorized A100-80GB GPU using the released
checkpoint: 284 Baseline responses and 284 `lit_r1_10_triplet_re2` responses.
The assembled OE-only candidate reuses the identical RE2 OE component and the
exact Baseline MC/TF components.

The `deepseek-v4-pro` raw audit passed with 1,334 unique score keys:
1,322 top-logprob expectations and 12 exact-20 fallbacks. A transient
`RemoteDisconnected` exposed a missing retry classification after 764 scores;
the transport exception was added to the bounded retry set, tested, and the
same score file resumed without repeating completed keys.

Neither candidate passed the frozen final criterion:

- `lit_r1_08_oe_re2`: 7/10 nonnegative, 1 strict improvement.
- `lit_r1_10_triplet_re2`: 5/10 nonnegative, 5 strict improvements.

The triplet RE2 prompt improves all three TF metrics but reduces MC Final,
MC Reasoning, OE Final, OE Accuracy, and OE Completeness. No winner is locked,
and no three-family success examples are claimed.
