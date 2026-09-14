$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = 'C:\Users\31029\AppData\Local\Programs\Python\Python312\python.exe'
$QuestionRoot = Join-Path $RepoRoot 'outputs\evals\tsdata61_smoke_0606c_rebuilt'
$EvalRoot = Join-Path $RepoRoot 'outputs\evals\baseline_tsdata61_smoke_0606c_20260609a'
$LogRoot = Join-Path $EvalRoot 'logs'

New-Item -ItemType Directory -Force -Path $EvalRoot | Out-Null
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

$env:PYTHONUTF8 = '1'
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:GIT_HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:GIT_HTTPS_PROXY -ErrorAction SilentlyContinue

function Write-Stage {
    param([string]$Message)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$stamp] $Message"
}

function Invoke-PythonLogged {
    param(
        [string]$Name,
        [string[]]$CommandArgs,
        [hashtable]$ExtraEnv = @{}
    )

    $logPath = Join-Path $LogRoot "$Name.log"
    $stdoutPath = Join-Path $LogRoot "$Name.stdout.log"
    $stderrPath = Join-Path $LogRoot "$Name.stderr.log"
    foreach ($key in $ExtraEnv.Keys) {
        Set-Item -Path "Env:$key" -Value ([string]$ExtraEnv[$key])
    }

    try {
        Write-Stage "Running $Name"
        if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force }
        if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force }
        $proc = Start-Process -FilePath $Python `
            -ArgumentList $CommandArgs `
            -NoNewWindow `
            -Wait `
            -PassThru `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath

        $merged = @()
        if (Test-Path $stdoutPath) { $merged += Get-Content $stdoutPath }
        if (Test-Path $stderrPath) { $merged += Get-Content $stderrPath }
        $merged | Set-Content -Path $logPath -Encoding UTF8

        if ($proc.ExitCode -ne 0) {
            throw "Command failed with exit code $($proc.ExitCode)"
        }
        Write-Stage "Finished $Name"
    }
    catch {
        Write-Host "---- tail: $logPath ----"
        if (Test-Path $logPath) {
            Get-Content $logPath -Tail 80
        }
        throw
    }
    finally {
        foreach ($key in $ExtraEnv.Keys) {
            Remove-Item "Env:$key" -ErrorAction SilentlyContinue
        }
    }
}

function Find-LatestResultDir {
    param(
        [string]$BaseDir,
        [string]$RunTag
    )
    $dirs = Get-ChildItem -Path $BaseDir -Directory -Recurse |
        Where-Object { $_.Name -like "results_*_${RunTag}" } |
        Sort-Object LastWriteTime -Descending
    if (-not $dirs) {
        throw "Could not find result directory for run tag $RunTag under $BaseDir"
    }
    return $dirs[0].FullName
}

function Copy-RunArtifacts {
    param(
        [string]$SourceDir,
        [string]$TargetDir
    )
    if (Test-Path $TargetDir) {
        Remove-Item -LiteralPath $TargetDir -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    Copy-Item -Path (Join-Path $SourceDir '*') -Destination $TargetDir -Recurse -Force
}

$providers = @(
    @{ Name = 'gpt55'; Config = 'configs/gpt55_question_llm.json' },
    @{ Name = 'deepseek'; Config = 'configs/deepseek_llm.json' }
)

$splits = @('regular', 'hard')

$baselines = @(
    @{
        Group = 'image'
        Script = 'baseline/ChatTS/src/models/Baselines/api_image_test.py'
        ResultRoot = 'experiments/logs/API_Image'
        Env = @{
            API_IMAGE_TOTAL_BATCHES = '999'
            API_IMAGE_BATCH_SIZE = '5'
            API_IMAGE_MAX_WORKERS = '5'
        }
    },
    @{
        Group = 'anomllm'
        Script = 'baseline/ChatTS/src/models/Baselines/AnomLLM_test.py'
        ResultRoot = 'experiments/logs/AnomLLM'
        Env = @{
            ANOMLLM_TOTAL_BATCHES = '999'
            ANOMLLM_BATCH_SIZE = '5'
            ANOMLLM_MAX_WORKERS = '5'
            ANOMLLM_LOG_EVERY_BATCH = '5'
        }
    },
    @{
        Group = 'llmad'
        Script = 'baseline/ChatTS/src/models/Baselines/LLMAD_test.py'
        ResultRoot = 'experiments/logs/LLMAD_API'
        Env = @{
            LLMAD_TOTAL_BATCHES = '999'
            LLMAD_BATCH_SIZE = '5'
            LLMAD_LOG_EVERY_BATCH = '5'
        }
    }
)

foreach ($split in $splits) {
    $questionPath = Join-Path $QuestionRoot "$split\questions_50_student.jsonl"
    $teacherPath = Join-Path $QuestionRoot "$split\teacher_gpt55.jsonl"
    if ((Test-Path $teacherPath) -and ((Get-Item $teacherPath).Length -gt 0)) {
        Write-Stage "Skipping teacher_${split} because $teacherPath already exists"
        continue
    }
    Invoke-PythonLogged -Name "teacher_${split}" -CommandArgs @(
        (Join-Path $RepoRoot 'scripts\mvaxis_generate_teacher_answers.py'),
        '--data', $questionPath,
        '--llm-config', (Join-Path $RepoRoot 'configs\gpt55_question_llm.json'),
        '--output', $teacherPath,
        '--overwrite',
        '--no-image',
        '--max-retries', '5',
        '--retry-sleep', '10',
        '--progress-step-percent', '10'
    )
}

foreach ($provider in $providers) {
    foreach ($split in $splits) {
        $questionPath = Join-Path $QuestionRoot "$split\questions_50_student.jsonl"
        $teacherPath = Join-Path $QuestionRoot "$split\teacher_gpt55.jsonl"
        $splitRoot = Join-Path $EvalRoot "$($provider.Name)\$split"
        New-Item -ItemType Directory -Force -Path $splitRoot | Out-Null

        foreach ($baseline in $baselines) {
            $runTag = "$($provider.Name)_$split"
            $targetDir = Join-Path $splitRoot $baseline.Group
            if (Test-Path (Join-Path $targetDir 'qwen_raw_answers.jsonl')) {
                Write-Stage "Skipping $($provider.Name)_${split}_$($baseline.Group) because $targetDir already has qwen_raw_answers.jsonl"
                continue
            }
            $extraEnv = @{
                OPENAI_LLM_CONFIG = (Join-Path $RepoRoot $provider.Config)
                MVAXIS_INPUT_JSONL = $questionPath
                MVAXIS_RUN_TAG = $runTag
                MVAXIS_SAMPLE_LIMIT = '50'
            }
            foreach ($key in $baseline.Env.Keys) {
                $extraEnv[$key] = $baseline.Env[$key]
            }

            Invoke-PythonLogged -Name "$($provider.Name)_${split}_$($baseline.Group)" -CommandArgs @(
                (Join-Path $RepoRoot $baseline.Script)
            ) -ExtraEnv $extraEnv

            $resultDir = Find-LatestResultDir -BaseDir (Join-Path $RepoRoot $baseline.ResultRoot) -RunTag $runTag
            Copy-RunArtifacts -SourceDir $resultDir -TargetDir $targetDir
        }

        Invoke-PythonLogged -Name "eval_$($provider.Name)_$split" -CommandArgs @(
            (Join-Path $RepoRoot 'scripts\mvaxis_eval_teacher_aligned_student_runs.py'),
            '--run-root', $splitRoot,
            '--questions', $questionPath,
            '--teacher', $teacherPath,
            '--runs', 'image', 'anomllm', 'llmad'
        )

        Invoke-PythonLogged -Name "open_metrics_$($provider.Name)_$split" -CommandArgs @(
            (Join-Path $RepoRoot 'scripts\mvaxis_compute_metrics_with_open_decision.py'),
            '--root', $splitRoot,
            '--questions', $questionPath,
            '--teacher', $teacherPath,
            '--groups', 'image', 'anomllm', 'llmad'
        )

        Invoke-PythonLogged -Name "review_$($provider.Name)_$split" -CommandArgs @(
            (Join-Path $RepoRoot 'scripts\mvaxis_build_answer_review_html.py'),
            '--root', $splitRoot,
            '--groups', 'image', 'anomllm', 'llmad',
            '--questions', $questionPath,
            '--teacher', $teacherPath
        )

        Invoke-PythonLogged -Name "summary_html_$($provider.Name)_$split" -CommandArgs @(
            (Join-Path $RepoRoot 'scripts\mvaxis_build_eval_summary_html.py'),
            '--summary-json', (Join-Path $splitRoot 'teacher_aligned_summary.json'),
            "--title=TSData61_Smoke_0606c_$($provider.Name)_$split",
            '--output-html', (Join-Path $splitRoot 'one_page_report.html')
        )
    }
}

$indexPath = Join-Path $EvalRoot 'index.html'
$links = @()
foreach ($provider in $providers) {
    foreach ($split in $splits) {
        $rel = "$($provider.Name)/$split/one_page_report.html"
        $reviewRel = "$($provider.Name)/$split/answer_review_index.html"
        $links += "<tr><td>$($provider.Name)</td><td>$split</td><td><a href='$rel'>summary</a></td><td><a href='$reviewRel'>answer review</a></td></tr>"
    }
}
$html = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>TSData61 Smoke 0606c Baseline Eval</title>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }
    table { border-collapse: collapse; width: 100%; background: white; }
    th, td { border: 1px solid #d1d5db; padding: 10px; text-align: left; }
    th { background: #f3f4f6; }
  </style>
</head>
<body>
  <h1>TSData61 Smoke 0606c Baseline Eval</h1>
  <table>
    <thead>
      <tr><th>provider</th><th>split</th><th>summary</th><th>review</th></tr>
    </thead>
    <tbody>
      $($links -join "`n")
    </tbody>
  </table>
</body>
</html>
"@
$html | Set-Content -Path $indexPath -Encoding UTF8

Write-Stage "All runs finished. Index: $indexPath"
