# Research log

## 2026-07-28 — Protocol lock

The user confirmed evaluation of Baseline plus all three final Round 9 routes
on paper140. Exact extraction from the already-audited full284 prediction
artifact is authorized, so AXIS inference will not be rerun. The Judge matrix
is locked to DeepSeek v4-pro, Qwen3.5-397B-A17B, and
Qwen3-30B-A3B-Instruct-2507; Gemini is excluded.

Scores will be de-duplicated only for byte-identical per-record G-Eval inputs
within a Judge, then expanded back to four complete 335-dimension Table-I
conditions. Cross-Judge reuse and score pooling are prohibited.

