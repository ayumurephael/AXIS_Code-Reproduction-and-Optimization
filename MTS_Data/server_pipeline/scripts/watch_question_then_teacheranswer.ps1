param(
    [Parameter(Mandatory = $true)]
    [string]$QuestionPath,

    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$TeacherOutput,

    [Parameter(Mandatory = $true)]
    [string]$LlmConfig,

    [Parameter(Mandatory = $false)]
    [int]$TargetCount = 1000,

    [Parameter(Mandatory = $false)]
    [int]$PollSeconds = 1800,

    [Parameter(Mandatory = $false)]
    [int]$StableSeconds = 120
)

$ErrorActionPreference = "Stop"

$teacherDir = Split-Path -Parent $TeacherOutput
$watchLog = Join-Path $teacherDir "watcher.log"
$stdoutLog = Join-Path $teacherDir "teacher_stdout.log"
$stderrLog = Join-Path $teacherDir "teacher_stderr.log"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$teacherScript = Join-Path $scriptRoot 'mvaxis_generate_teacher_answers.py'

function Write-Log([string]$Message) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $watchLog -Value "[$stamp] $Message"
}

New-Item -ItemType Directory -Force -Path $teacherDir | Out-Null
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

Write-Log "watcher started"

while ($true) {
    if (-not (Test-Path -LiteralPath $QuestionPath)) {
        Write-Log "question file missing"
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    try {
        $count = (Get-Content -LiteralPath $QuestionPath | Measure-Object -Line).Lines
        $ageSeconds = [int](((Get-Date) - (Get-Item -LiteralPath $QuestionPath).LastWriteTime).TotalSeconds)
        Write-Log "question count=$count age_seconds=$ageSeconds"
        if ($count -ge $TargetCount -and $ageSeconds -ge $StableSeconds) {
            Write-Log "threshold reached; starting teacher answer generation"
            & 'C:\Users\31029\AppData\Local\Programs\Python\Python312\python.exe' `
                $teacherScript `
                --data $QuestionPath `
                --llm-config $LlmConfig `
                --output $TeacherOutput `
                --max-retries 5 `
                --retry-sleep 10 `
                --progress-step-percent 2.5 1>>$stdoutLog 2>>$stderrLog
            $exitCode = $LASTEXITCODE
            Write-Log "teacher answer command exited with code $exitCode"
            exit $exitCode
        }
    } catch {
        Write-Log ("check failed: " + $_.Exception.Message)
    }

    Start-Sleep -Seconds $PollSeconds
}