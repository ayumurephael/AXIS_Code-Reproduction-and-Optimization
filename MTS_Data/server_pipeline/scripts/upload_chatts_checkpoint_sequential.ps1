param(
    [string]$RemoteHost = 'cln01',
    [string]$LocalDir,
    [string]$RemoteDir = '/WORK/fit/zhangchen/axis_project/checkpoints/chatts_legacy14b',
    [string]$LogPath
)

$ErrorActionPreference = 'Stop'

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot
if (-not $LocalDir) {
    $LocalDir = Join-Path $projectRoot 'external_checkpoints\chatts_legacy14b'
}
if (-not $LogPath) {
    $LogPath = Join-Path $projectRoot 'tmp\upload_chatts_checkpoint_sequential.log'
}

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    $line = "[$stamp] $Message"
    Write-Host $line
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path ([System.IO.Path]::GetDirectoryName($LogPath)) | Out-Null
Set-Content -LiteralPath $LogPath -Value '' -Encoding UTF8

$localRoot = (Resolve-Path -LiteralPath $LocalDir).Path
Write-Log "Local root: $localRoot"
Write-Log "Remote target: ${RemoteHost}:$RemoteDir"

ssh $RemoteHost "mkdir -p $RemoteDir" | Out-Null

$files = Get-ChildItem -LiteralPath $localRoot -File | Sort-Object Name
foreach ($file in $files) {
    $remotePath = "$RemoteDir/$($file.Name)"
    $remoteSizeRaw = ssh $RemoteHost "if [ -f '$remotePath' ]; then stat -c%s '$remotePath'; else echo 0; fi"
    $remoteSize = [int64](($remoteSizeRaw | Select-Object -First 1).ToString().Trim())

    if ($remoteSize -eq $file.Length) {
        Write-Log "Skip $($file.Name) ($remoteSize bytes already present)"
        continue
    }

    if ($remoteSize -gt 0 -and $remoteSize -ne $file.Length) {
        Write-Log "Remove partial remote file $($file.Name) ($remoteSize bytes)"
        ssh $RemoteHost "rm -f '$remotePath'" | Out-Null
    }

    Write-Log "Upload $($file.Name) ($($file.Length) bytes)"
    scp $file.FullName "${RemoteHost}:$remotePath"

    $verifyRaw = ssh $RemoteHost "stat -c%s '$remotePath'"
    $verifySize = [int64](($verifyRaw | Select-Object -First 1).ToString().Trim())
    if ($verifySize -ne $file.Length) {
        throw "Size mismatch after upload for $($file.Name): local=$($file.Length) remote=$verifySize"
    }
    Write-Log "Done $($file.Name)"
}

Write-Log "All checkpoint files synchronized."