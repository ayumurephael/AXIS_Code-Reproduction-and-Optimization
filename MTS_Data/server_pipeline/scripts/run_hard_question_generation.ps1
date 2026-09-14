param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),

    [Parameter(Mandatory = $false)]
    [string]$Windows = $null,

    [Parameter(Mandatory = $false)]
    [string]$RawSeries = "outputs/ts_data/ts_531/raw_series.jsonl",

    [Parameter(Mandatory = $false)]
    [string]$LlmConfig = "configs/gpt55_question_llm.json",

    [Parameter(Mandatory = $false)]
    [string]$OutputDir = "outputs/question_hard_200_0601",

    [Parameter(Mandatory = $false)]
    [int]$NumQuestions = 200,

    [Parameter(Mandatory = $false)]
    [int]$Seed = 601
)

if (-not $Windows) {
    $Windows = Join-Path $ProjectRoot 'outputs\ts_data\ts_531_windows\windows_20000.jsonl'
}

Set-Location $ProjectRoot

$userKey = [Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "User")
if ($userKey) {
    $env:OPENAI_API_KEY = $userKey
}

Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:GIT_HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:GIT_HTTPS_PROXY -ErrorAction SilentlyContinue

& 'C:\Users\31029\AppData\Local\Programs\Python\Python312\python.exe' `
    (Join-Path $ProjectRoot 'scripts\mvaxis_generate_hard_questions_from_windows.py') `
    --windows $Windows `
    --raw-series $RawSeries `
    --llm-config $LlmConfig `
    --output-dir $OutputDir `
    --num-questions $NumQuestions `
    --seed $Seed `
    --max-retries 5 `
    --retry-sleep 10 `
    --progress-step-percent 2.5 `
    --stop-on-error

exit $LASTEXITCODE
