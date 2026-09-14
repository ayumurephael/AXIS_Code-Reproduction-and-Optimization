param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$ShardPidFile = (Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp') 'upload_chatts_shard1_20260610.pid'),
    [string]$SequentialUploadScript = (Join-Path $PSScriptRoot 'upload_chatts_checkpoint_sequential.ps1'),
    [string]$SequentialUploadLog = (Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp') 'upload_chatts_checkpoint_sequential.log'),
    [string]$MainLog = (Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp') 'continue_chatts_upload_and_resubmit.log'),
    [string]$RemoteHost = 'cln01',
    [string]$RemoteSbatch = '/WORK/fit/zhangchen/axis_project/codex_hold_h800_7d_and_stage_qwen_v2.sbatch'
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -LiteralPath $MainLog -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($MainLog)) | Out-Null
Set-Content -LiteralPath $MainLog -Value '' -Encoding UTF8

Write-Log "Watcher started. project_root=$ProjectRoot"

if (Test-Path -LiteralPath $ShardPidFile) {
    $shardPid = [int](Get-Content -LiteralPath $ShardPidFile)
    Write-Log "Waiting for shard1 process $shardPid"
    while ($true) {
        try {
            $null = Get-Process -Id $shardPid -ErrorAction Stop
            Start-Sleep -Seconds 30
        }
        catch {
            break
        }
    }
    Write-Log "Shard1 process finished."
}
else {
    Write-Log "Shard pid file missing; proceeding immediately."
}

Write-Log "Running sequential uploader."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SequentialUploadScript -ProjectRoot $ProjectRoot -LogPath $SequentialUploadLog

Write-Log "Submitting remote sbatch."
$submitOut = ssh $RemoteHost "/rmprog/slurm/v24.05.1/bin/sbatch $RemoteSbatch"
Write-Log ("sbatch response: " + ($submitOut -join ' '))

Write-Log "Done."
