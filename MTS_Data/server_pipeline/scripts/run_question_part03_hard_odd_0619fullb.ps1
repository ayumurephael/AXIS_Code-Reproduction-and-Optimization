$ErrorActionPreference = 'Stop'

$py = 'C:\Users\31029\AppData\Local\Programs\Python\Python312\python.exe'
$q = 'X:\codex\multiaxis\scripts\mvaxis_generate_bank_questions_from_windows.py'
$dataRoot = 'D:\multiaxis\outputs\ts_data\ts_varfeat_20k_5way_20260617'
$outRoot = 'X:\codex\multiaxis\outputs\question'

$part = 'part_03'
$partTag = $part -replace '_', ''
$windows = Join-Path $dataRoot "$part\windows_8000.jsonl"
$raw = Join-Path $dataRoot "$part\raw_series.jsonl"
$configPath = 'X:\codex\multiaxis\configs\gpt55_question_llm2.json'

foreach ($i in @(1, 3, 5, 7, 9, 11, 13, 15)) {
  $start = $i * 250
  $end = $start + 250
  $suffix = '{0:D4}_{1:D4}' -f $start, $end
  $seed = 72930 + $i

  & $py $q `
    --windows $windows `
    --raw-series $raw `
    --output-dir (Join-Path $outRoot ("question_ts_varfeat20k_{0}_hard_1000_{1}_0619fullb" -f $partTag, $suffix)) `
    --num-questions 1000 `
    --seed $seed `
    --question-seed ($seed + 17) `
    --question-bank hard `
    --source-start-index $start `
    --source-end-index $end `
    --llm-config $configPath `
    --max-retries 5 `
    --retry-sleep 10 `
    --progress-step-percent 1 `
    --resume

  if ($LASTEXITCODE -ne 0) {
    throw "Question generation failed for shard $suffix with exit code $LASTEXITCODE"
  }
}
