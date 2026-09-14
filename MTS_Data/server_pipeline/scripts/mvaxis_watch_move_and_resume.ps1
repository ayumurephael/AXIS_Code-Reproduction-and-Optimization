param(
    [string]$SourceRoot = 'X:\multiaxis',
    [string]$SourceOutputRoot = 'X:\multiaxis\outputs\ts_data\ts_varfeat_20k_5way_20260617',
    [string]$DestRoot = 'D:\multiaxis',
    [string]$PythonExe = 'C:\Users\31029\AppData\Local\Programs\Python\Python312\python.exe',
    [string]$WrapperScript = 'X:\codex\multiaxis\scripts\mvaxis_generate_variable_feature_windows.py',
    [int]$PollSeconds = 60,
    [int]$StallPolls = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$opsLogDir = 'D:\PythonProjects\codex\multiaxis\outputs\ops_logs'
New-Item -ItemType Directory -Force -Path $opsLogDir | Out-Null
$runStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $opsLogDir "varfeat_move_resume_$runStamp.log"

$watchParts = @('part_03', 'part_04', 'part_05')
$resumeParts = @('part_00', 'part_01', 'part_02', 'part_03', 'part_04')
$partSeeds = @{
    'part_00' = 617
    'part_01' = 1617
    'part_02' = 2617
    'part_03' = 3617
    'part_04' = 4617
}

function Write-Log {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    $line | Tee-Object -FilePath $logPath -Append
}

function Get-PartState {
    param(
        [string]$OutputRoot,
        [string]$PartName
    )

    $dir = Join-Path $OutputRoot $PartName
    $imagesDir = Join-Path $dir 'images'
    $rawPath = Join-Path $dir 'raw_series.jsonl'
    $summaryPath = Join-Path $dir 'dataset_summary.json'
    $windowsPath = $null
    if (Test-Path $dir) {
        $windowsPath = Get-ChildItem -LiteralPath $dir -Filter 'windows_*.jsonl' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    }
    $stdoutPath = Join-Path $OutputRoot ("_runner_logs\{0}.stdout.log" -f $PartName)
    $stderrPath = Join-Path $OutputRoot ("_runner_logs\{0}.stderr.log" -f $PartName)

    $pngCount = 0
    $latestImageTime = $null
    if (Test-Path $imagesDir) {
        $images = Get-ChildItem -LiteralPath $imagesDir -Filter '*.png' -File -ErrorAction SilentlyContinue
        $pngCount = @($images).Count
        if ($pngCount -gt 0) {
            $latestImageTime = ($images | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime.ToString('s')
        }
    }

    [pscustomobject]@{
        part          = $PartName
        dir_exists    = Test-Path $dir
        raw_lines     = if (Test-Path $rawPath) { (Get-Content -LiteralPath $rawPath | Measure-Object -Line).Lines } else { 0 }
        summary_exists = Test-Path $summaryPath
        windows_exists = $null -ne $windowsPath
        png_count     = $pngCount
        img_mtime     = $latestImageTime
        stdout_mtime  = if (Test-Path $stdoutPath) { (Get-Item $stdoutPath).LastWriteTime.ToString('s') } else { $null }
        stderr_mtime  = if (Test-Path $stderrPath) { (Get-Item $stderrPath).LastWriteTime.ToString('s') } else { $null }
    }
}

function Get-PartProcesses {
    param(
        [string]$OutputRoot,
        [string]$PartName
    )

    $partPath = Join-Path $OutputRoot $PartName
    $currentPid = $PID
    @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessId -ne $currentPid -and
            $_.CommandLine -and
            $_.CommandLine -like "*$partPath*" -and
            ($_.Name -in @('python.exe', 'pwsh.exe', 'powershell.exe'))
        })
}

function Stop-PartProcesses {
    param(
        [string]$OutputRoot,
        [string[]]$Parts
    )

    foreach ($part in $Parts) {
        $procs = Get-PartProcesses -OutputRoot $OutputRoot -PartName $part
        foreach ($proc in $procs) {
            try {
                Write-Log "Stopping process for ${part}: pid=$($proc.ProcessId) name=$($proc.Name)"
                Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            } catch {
                Write-Log "Failed to stop pid=$($proc.ProcessId) for ${part}: $($_.Exception.Message)"
            }
        }
    }
}

function Stop-NotepadOnSourceRoot {
    param([string]$RootPath)

    $procs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -eq 'Notepad.exe' -and
            $_.CommandLine -and
            $_.CommandLine -like "*$RootPath*"
        })
    foreach ($proc in $procs) {
        try {
            Write-Log "Closing Notepad holding source files: pid=$($proc.ProcessId)"
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        } catch {
            Write-Log "Failed to close Notepad pid=$($proc.ProcessId): $($_.Exception.Message)"
        }
    }
}

function Rewrite-DatasetSummaryPaths {
    param(
        [string]$PartDir,
        [string]$OldRoot,
        [string]$NewRoot
    )

    $summaryPath = Join-Path $PartDir 'dataset_summary.json'
    if (-not (Test-Path $summaryPath)) {
        return
    }

    try {
        $payload = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json -Depth 100
        foreach ($field in @('output_dir', 'raw_series_path')) {
            if ($payload.$field -is [string] -and $payload.$field.StartsWith($OldRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                $payload.$field = $NewRoot + $payload.$field.Substring($OldRoot.Length)
            }
        }
        if ($payload.shards) {
            foreach ($shard in $payload.shards) {
                if ($shard.path -is [string] -and $shard.path.StartsWith($OldRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $shard.path = $NewRoot + $shard.path.Substring($OldRoot.Length)
                }
            }
        }
        $payload | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    } catch {
        Write-Log "Failed to rewrite dataset_summary.json for ${PartDir}: $($_.Exception.Message)"
    }
}

function Clear-WindowMetadata {
    param([string]$PartDir)

    Get-ChildItem -LiteralPath $PartDir -Filter 'windows_*.jsonl' -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    $summaryPath = Join-Path $PartDir 'summary.json'
    if (Test-Path $summaryPath) {
        Remove-Item -LiteralPath $summaryPath -Force -ErrorAction SilentlyContinue
    }
}

function Start-ResumeRuns {
    param(
        [string]$OutputRoot,
        [string[]]$Parts
    )

    $resumeLogDir = Join-Path $OutputRoot '_resume_logs'
    New-Item -ItemType Directory -Force -Path $resumeLogDir | Out-Null

    foreach ($part in $Parts) {
        $partDir = Join-Path $OutputRoot $part
        New-Item -ItemType Directory -Force -Path $partDir | Out-Null

        $seed = [int]$partSeeds[$part]
        $partIndex = [int]($part.Substring($part.Length - 2))
        $sampleIdPrefix = 'ts_varfeat20k_{0:D2}' -f $partIndex
        $stdoutPath = Join-Path $resumeLogDir "$part.stdout.log"
        $stderrPath = Join-Path $resumeLogDir "$part.stderr.log"
        if (Test-Path $stdoutPath) { Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue }
        if (Test-Path $stderrPath) { Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue }

        $args = @(
            $WrapperScript,
            '--output-dir', $partDir,
            '--num-samples', '4000',
            '--seq-len-min', '128',
            '--seq-len-max', '512',
            '--num-features-min', '5',
            '--num-features-max', '50',
            '--anomaly-ratio', '1.0',
            '--seed', "$seed",
            '--samples-per-series', '2',
            '--min-window-size', '15',
            '--max-window-size', '60',
            '--window-anomaly-ratio', '0.5',
            '--sample-id-prefix', $sampleIdPrefix,
            '--shard-size', '1000',
            '--progress-step-percent', '1'
        )

        $proc = Start-Process -FilePath $PythonExe -ArgumentList $args -WorkingDirectory $scriptDir -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
        Write-Log "Started resume run for $part pid=$($proc.Id) output_dir=$partDir"
    }
}

Write-Log "Watcher started. source_root=$SourceRoot dest_root=$DestRoot source_output_root=$SourceOutputRoot"

$watchState = @{}
foreach ($part in $watchParts) {
    $watchState[$part] = @{
        last_sig = $null
        stable_polls = 0
        done = $false
    }
}

while ($true) {
    $allDone = $true
    foreach ($part in $watchParts) {
        if ($watchState[$part].done) {
            continue
        }

        $state = Get-PartState -OutputRoot $SourceOutputRoot -PartName $part
        $procs = @(Get-PartProcesses -OutputRoot $SourceOutputRoot -PartName $part)
        $sig = '{0}|{1}|{2}|{3}|{4}' -f $state.png_count, $state.raw_lines, $state.img_mtime, $state.stdout_mtime, $state.stderr_mtime
        if ($sig -eq $watchState[$part].last_sig) {
            $watchState[$part].stable_polls += 1
        } else {
            $watchState[$part].stable_polls = 0
            $watchState[$part].last_sig = $sig
        }

        Write-Log ("watch {0}: png={1} raw={2} proc_count={3} stable_polls={4} img_mtime={5}" -f $part, $state.png_count, $state.raw_lines, $procs.Count, $watchState[$part].stable_polls, $state.img_mtime)

        if (-not $state.dir_exists -and $procs.Count -eq 0) {
            $watchState[$part].done = $true
            Write-Log "watch $part considered done because directory is absent and no matching process exists."
            continue
        }

        if ($procs.Count -eq 0) {
            $watchState[$part].done = $true
            Write-Log "watch $part considered done because no matching process exists."
            continue
        }

        if ($watchState[$part].stable_polls -ge $StallPolls) {
            Write-Log "watch $part stalled for $StallPolls polls; stopping matching processes."
            Stop-PartProcesses -OutputRoot $SourceOutputRoot -Parts @($part)
            Start-Sleep -Seconds 5
            $watchState[$part].done = $true
            continue
        }

        $allDone = $false
    }

    if ($allDone) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-Log "Watch phase complete. Stopping any remaining resume-target processes before move."
Stop-PartProcesses -OutputRoot $SourceOutputRoot -Parts $resumeParts
Stop-NotepadOnSourceRoot -RootPath $SourceRoot
Start-Sleep -Seconds 5

if (Test-Path $DestRoot) {
    throw "Destination already exists: $DestRoot"
}

$destParent = Split-Path -Parent $DestRoot
New-Item -ItemType Directory -Force -Path $destParent | Out-Null
Write-Log "Moving $SourceRoot to $destParent"
Move-Item -LiteralPath $SourceRoot -Destination $destParent
Write-Log "Move complete: $DestRoot"

$relativeOutput = $SourceOutputRoot.Substring($SourceRoot.Length).TrimStart('\')
$destOutputRoot = Join-Path $DestRoot $relativeOutput
Write-Log "Destination output root resolved to $destOutputRoot"

foreach ($part in $resumeParts) {
    $partDir = Join-Path $destOutputRoot $part
    if (-not (Test-Path $partDir)) {
        continue
    }
    Rewrite-DatasetSummaryPaths -PartDir $partDir -OldRoot $SourceRoot -NewRoot $DestRoot
    Clear-WindowMetadata -PartDir $partDir
    Write-Log "Prepared moved part for resume: $partDir"
}

Start-ResumeRuns -OutputRoot $destOutputRoot -Parts $resumeParts
Write-Log "Resume launch complete."
