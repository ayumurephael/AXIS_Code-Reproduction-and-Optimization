#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/axis_project/repos/multiaxis

CONFIG="configs/torch_fixed60_qa600_timercd.json"
LLM_CONFIG="configs/local_qwen25_7b_cuda_remote_timercd_dproj256_523truth600.json"
ENCODER_CKPT="outputs/checkpoints/timercd_global_hint_head_truth600_labelglobal_e10_encoder.pt"
OUT_DIR="outputs/runs/523truth600_eval"
PY="/root/axis/bin/python"

mkdir -p "${OUT_DIR}"

run_eval() {
  local name="$1"
  local interval_source="$2"
  shift 2
  echo "== Running ${name} =="
  "${PY}" -B scripts/run_llm_on_proposals.py \
    --config "${CONFIG}" \
    --llm-config "${LLM_CONFIG}" \
    --split train \
    --limit 600 \
    --interval-source "${interval_source}" \
    --interval-proposer-checkpoint "${ENCODER_CKPT}" \
    --prompt-variant compact_label_json \
    --output "${OUT_DIR}/${name}.jsonl" \
    --report "${OUT_DIR}/${name}_report.json" \
    --raw-log-examples 6 \
    --dump-prompt-examples 2 \
    --question-provider bank \
    "$@"
}

run_eval "truth_all_hints" "truth"
run_eval "proposal_all_hints" "proposal"
run_eval "truth_no_global_hints" "truth" --hide-global-hints
run_eval "truth_no_channel_hints" "truth" --hide-channel-hints
run_eval "truth_no_evidence_card" "truth" --hide-evidence-card

"${PY}" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

out_dir = Path("outputs/runs/523truth600_eval")
names = [
    "truth_all_hints",
    "proposal_all_hints",
    "truth_no_global_hints",
    "truth_no_channel_hints",
    "truth_no_evidence_card",
]
keys = [
    "json_parse_rate",
    "proposal_mean_temporal_iou",
    "proposal_mean_truth_coverage",
    "proposal_close_rate",
    "llm_anomaly_accuracy",
    "llm_answer_label_accuracy",
    "llm_answer_label_count",
    "llm_type_accuracy",
    "llm_root_accuracy",
    "llm_mean_affected_f1",
]
summary = []
for name in names:
    report_path = out_dir / f"{name}_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    row = {"run": name}
    for key in keys:
        row[key] = report.get(key)
    summary.append(row)
(out_dir / "summary_5runs.json").write_text(
    json.dumps({"runs": summary}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({"runs": summary}, ensure_ascii=False, indent=2))
PY
