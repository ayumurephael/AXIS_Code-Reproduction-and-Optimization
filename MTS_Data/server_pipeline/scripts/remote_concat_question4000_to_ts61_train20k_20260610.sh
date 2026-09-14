#!/usr/bin/env bash
set -euo pipefail

WORKROOT="${WORKROOT:-/WORK/fit/zhangchen/axis_project}"
SEARCH_ROOT="${SEARCH_ROOT:-$WORKROOT/data}"
OUT_ROOT="${OUT_ROOT:-$WORKROOT/data/ts61_train_20k_20260610a}"
SUMMARY_TXT="$OUT_ROOT/concat_summary.txt"
QUESTION_OUT="$OUT_ROOT/questions_20000.jsonl"
TEACHER_OUT="$OUT_ROOT/teacher_gpt55.jsonl"

DATASETS=(
  question_4000_ts61_0000_1000_0609a
  question_4000_ts61_1000_2000_0609a
  question_4000_ts61_2000_3000_0609a
  question_4000_ts61_3000_4000_0609a
  question_4000_ts61_4000_5000_0609a
)

find_first_dir() {
  local name="$1"
  find "$SEARCH_ROOT" -type d -name "$name" 2>/dev/null | sort | head -n 1
}

mkdir -p "$OUT_ROOT"
: > "$QUESTION_OUT"
: > "$TEACHER_OUT"
: > "$SUMMARY_TXT"

echo "[concat20k] start $(date --iso-8601=seconds)" | tee -a "$SUMMARY_TXT"
echo "[concat20k] search_root=$SEARCH_ROOT" | tee -a "$SUMMARY_TXT"
echo "[concat20k] out_root=$OUT_ROOT" | tee -a "$SUMMARY_TXT"

total_question_lines=0
total_teacher_lines=0

for dataset in "${DATASETS[@]}"; do
  q_dir="$(find_first_dir "$dataset")"
  t_dir="$(find_first_dir "teacher_${dataset}_v1")"

  if [ -z "$q_dir" ]; then
    echo "[concat20k] missing question dir for $dataset under $SEARCH_ROOT" >&2
    exit 1
  fi
  if [ -z "$t_dir" ]; then
    echo "[concat20k] missing teacher dir for $dataset under $SEARCH_ROOT" >&2
    exit 1
  fi

  q_file="$q_dir/questions_4000.jsonl"
  t_file="$t_dir/teacher_gpt55.jsonl"

  if [ ! -f "$q_file" ]; then
    echo "[concat20k] missing question file: $q_file" >&2
    exit 1
  fi
  if [ ! -f "$t_file" ]; then
    echo "[concat20k] missing teacher file: $t_file" >&2
    exit 1
  fi

  q_lines="$(wc -l < "$q_file")"
  t_lines="$(wc -l < "$t_file")"
  total_question_lines=$((total_question_lines + q_lines))
  total_teacher_lines=$((total_teacher_lines + t_lines))

  echo "[concat20k] append dataset=$dataset q_lines=$q_lines t_lines=$t_lines" | tee -a "$SUMMARY_TXT"
  echo "[concat20k]   q_file=$q_file" | tee -a "$SUMMARY_TXT"
  echo "[concat20k]   t_file=$t_file" | tee -a "$SUMMARY_TXT"

  cat "$q_file" >> "$QUESTION_OUT"
  cat "$t_file" >> "$TEACHER_OUT"
done

merged_question_lines="$(wc -l < "$QUESTION_OUT")"
merged_teacher_lines="$(wc -l < "$TEACHER_OUT")"

echo "[concat20k] merged_question_lines=$merged_question_lines expected=$total_question_lines" | tee -a "$SUMMARY_TXT"
echo "[concat20k] merged_teacher_lines=$merged_teacher_lines expected=$total_teacher_lines" | tee -a "$SUMMARY_TXT"

if [ "$merged_question_lines" -ne "$total_question_lines" ]; then
  echo "[concat20k] merged question line count mismatch" >&2
  exit 1
fi
if [ "$merged_teacher_lines" -ne "$total_teacher_lines" ]; then
  echo "[concat20k] merged teacher line count mismatch" >&2
  exit 1
fi

echo "[concat20k] done $(date --iso-8601=seconds)" | tee -a "$SUMMARY_TXT"
