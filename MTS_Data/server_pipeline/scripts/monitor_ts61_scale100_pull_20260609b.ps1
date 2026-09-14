$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$MultiaxisRoot = Split-Path -Parent $ScriptRoot
$ProjectRoot = Split-Path -Parent $MultiaxisRoot
$RemoteSsh = Join-Path $ProjectRoot "tools\axis_remote\paratera_zw1_ssh.ps1"
$RemoteScp = Join-Path $ProjectRoot "tools\axis_remote\paratera_zw1_scp_from_remote.ps1"

$RemoteRoot = "/tmp/tsdata61_scale100_only_20260609c"
$RemoteArchive = "/tmp/tsdata61_scale100_only_20260609c.tgz"
$LocalBase = Join-Path $MultiaxisRoot "outputs\studentanswer"
$LocalRoot = Join-Path $LocalBase "tsdata61_scale100_only_20260609c_remote"
$LocalArchive = Join-Path $LocalBase "tsdata61_scale100_only_20260609c.tgz"
$LogPath = Join-Path $LocalBase "tsdata61_scale100_only_20260609c_pull_monitor.log"
$SummaryPath = Join-Path $LocalBase "tsdata61_scale100_only_20260609c_eval_summary.txt"

New-Item -ItemType Directory -Force -Path $LocalBase | Out-Null

function Write-Log {
    param([string] $Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $LogPath -Value "[$stamp] $Message"
}

function Write-Summary {
    param([string] $ExtractedRoot)

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add("tsdata61 scale=1.0 pull summary")
    $lines.Add("generated_at=" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    $lines.Add("root=" + $ExtractedRoot)

    foreach ($split in @("regular_smoke_0606c", "hard_smoke_0606c")) {
        $lines.Add("")
        $lines.Add("[" + $split + "]")
        $evalLog = Join-Path $ExtractedRoot ($split + "\eval.log")
        $summaryTxt = Join-Path $ExtractedRoot ($split + "\teacher_aligned_summary.txt")
        if (Test-Path $summaryTxt) {
            $lines.Add("teacher_aligned_summary:")
            $lines.AddRange((Get-Content $summaryTxt))
        }
        if (Test-Path $evalLog) {
            $lines.Add("eval_log:")
            $lines.AddRange((Get-Content $evalLog))
        }
    }

    Set-Content -LiteralPath $SummaryPath -Value $lines
}

Write-Log "monitor started"

while ($true) {
    try {
        $status = & $RemoteSsh -Remote "test -s $RemoteRoot/regular_smoke_0606c/eval.log && test -s $RemoteRoot/hard_smoke_0606c/eval.log && echo ready || echo waiting"
        $status = ($status | Out-String).Trim()
        Write-Log "remote status=$status"
        if ($status -eq "ready") {
            Write-Log "creating remote archive"
            & $RemoteSsh -Remote "tar -C /tmp -czf $RemoteArchive tsdata61_scale100_only_20260609c"
            Write-Log "copying archive to local"
            & $RemoteScp -RemotePath $RemoteArchive -LocalPath $LocalArchive
            if (Test-Path $LocalRoot) {
                Remove-Item -LiteralPath $LocalRoot -Recurse -Force
            }
            New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null
            Write-Log "extracting local archive"
            tar -xzf $LocalArchive -C $LocalRoot
            $ExtractedRoot = Join-Path $LocalRoot "tsdata61_scale100_only_20260609c"
            if (Test-Path $ExtractedRoot) {
                Write-Log "writing summary"
                Write-Summary -ExtractedRoot $ExtractedRoot
            }
            Write-Log "pull complete"
            break
        }
    }
    catch {
        Write-Log ("monitor error: " + $_.Exception.Message)
    }
    Start-Sleep -Seconds 60
}
