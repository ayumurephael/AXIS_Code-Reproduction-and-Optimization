param(
    [switch] $WhatIfOnly
)

$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

$relativeTargets = @(
    "external\AXIS",
    "external\AXIS_src",
    "external\AXIS_repo",
    "external\TSAD_dataset_gen-axis",
    "external\TSAD_dataset_gen_axis_unpacked",
    "external\TimeRCD",
    "external\TimeRCD_src",
    "external\TimeRCD_src2",
    "external\doflow_hint_ablation_qwen_limit50_bestckpt_v1",
    "external\doflow_hint_ablation_qwen_limit50_v1",
    "external\doflow_hint_ablation_qwen_v1",
    "external\multilevel_qa100_compact_text_cache_v1",
    "external\truth_model_hints_compact_text_cache_limit50_v1",
    "external\truth50_compact_ablation_v1",
    "external\AXIS_repo.zip",
    "axis_remote_bundle",
    "axis_remote_minimal",
    "axis_remote_minimal.zip"
)

$filePatterns = @(
    "external\qa_teacher_vs_qwen*.txt",
    "external\qwen_proposal*.json",
    "external\qwen_proposal*.txt"
)

$targets = New-Object System.Collections.Generic.List[string]

foreach ($rel in $relativeTargets) {
    $path = Join-Path $workspace $rel
    if (Test-Path -LiteralPath $path) {
        $targets.Add((Resolve-Path -LiteralPath $path).Path)
    }
}

foreach ($pattern in $filePatterns) {
    Get-ChildItem -Path (Join-Path $workspace $pattern) -File -ErrorAction SilentlyContinue |
        ForEach-Object { $targets.Add($_.FullName) }
}

$targets = $targets | Sort-Object -Unique

foreach ($target in $targets) {
    if (-not $target.StartsWith($workspace, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside workspace: $target"
    }
}

if ($WhatIfOnly) {
    $targets
    return
}

$removed = New-Object System.Collections.Generic.List[string]
$failed = New-Object System.Collections.Generic.List[object]

foreach ($target in $targets) {
    try {
        if (Test-Path -LiteralPath $target) {
            Get-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue |
                ForEach-Object { $_.Attributes = "Normal" }
            Get-ChildItem -LiteralPath $target -Recurse -Force -ErrorAction SilentlyContinue |
                ForEach-Object {
                    try { $_.Attributes = "Normal" } catch {}
                }
            Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
            $removed.Add($target)
        }
    }
    catch {
        $failed.Add([pscustomobject]@{ Path = $target; Error = $_.Exception.Message })
    }
}

[pscustomobject]@{
    RemovedCount = $removed.Count
    FailedCount = $failed.Count
    Removed = $removed
    Failed = $failed
} | ConvertTo-Json -Depth 4
