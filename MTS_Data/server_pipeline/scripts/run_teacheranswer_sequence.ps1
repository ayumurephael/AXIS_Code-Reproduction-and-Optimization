param(
    [Parameter(Mandatory = $true)]
    [string]$Part,

    [Parameter(Mandatory = $true)]
    [string]$Config,

    [Parameter(Mandatory = $true)]
    [int[]]$ShardIds,

    [string]$QuestionRoot = 'X:\codex\multiaxis\outputs\question',
    [string]$TeacherRoot = 'E:\multiaxis\teacheranswer',
    [string]$RunSuffix = '0621ta',
    [string]$PythonExe = 'C:\Users\31029\AppData\Local\Programs\Python\Python312\python.exe'
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$teacherScript = Join-Path $scriptRoot 'mvaxis_generate_teacher_answers.py'
$configPath = Join-Path 'X:\codex\multiaxis\configs' $Config

function Get-Bank([string]$PartName, [int]$ShardId) {
    switch ($PartName) {
        'part00' { return 'regular' }
        'part01' { return 'regular' }
        'part02' { if (($ShardId % 2) -eq 0) { return 'regular' } else { return 'hard' } }
        'part03' { return 'hard' }
        'part04' { return 'hard' }
        default { throw "Unsupported part: $PartName" }
    }
}

function Get-QuestionSuffix([string]$Bank) {
    if ($Bank -eq 'regular') { return '0619fulla' }
    if ($Bank -eq 'hard') { return '0619fullb' }
    throw "Unsupported bank: $Bank"
}

foreach ($i in $ShardIds) {
    $start = $i * 250
    $end = $start + 250
    $segment = '{0:D4}_{1:D4}' -f $start, $end
    $bank = Get-Bank $Part $i
    $questionSuffix = Get-QuestionSuffix $bank

    $questionDir = Join-Path $QuestionRoot ("question_ts_varfeat20k_{0}_{1}_1000_{2}_{3}" -f $Part, $bank, $segment, $questionSuffix)
    $questionPath = Join-Path $questionDir 'questions_1000.jsonl'

    if (!(Test-Path -LiteralPath $questionDir)) {
        throw "Missing question shard directory: $questionDir"
    }

    if (!(Test-Path -LiteralPath $questionPath)) {
        throw "Missing question shard file: $questionPath"
    }

    $teacherDir = Join-Path $TeacherRoot ("teacher_ts_varfeat20k_{0}_{1}_1000_{2}_{3}" -f $Part, $bank, $segment, $RunSuffix)
    New-Item -ItemType Directory -Force -Path $teacherDir | Out-Null

    $outputPath = Join-Path $teacherDir 'teacher_gpt55.jsonl'
    $answersPath = Join-Path $teacherDir 'teacher_gpt55.answers.jsonl'
    $summaryPath = Join-Path $teacherDir 'summary.json'

    Write-Host ("[teacheranswer] {0} {1} -> {2}" -f $Part, $segment, $teacherDir)

    & $PythonExe $teacherScript `
        --data $questionPath `
        --llm-config $configPath `
        --output $outputPath `
        --answers-output $answersPath `
        --summary $summaryPath `
        --max-retries 5 `
        --retry-sleep 10 `
        --progress-step-percent 2.5 `
        --resume

    if ($LASTEXITCODE -ne 0) {
        throw "Teacher answer generation failed for $Part $segment with exit code $LASTEXITCODE"
    }
}
