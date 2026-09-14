$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$secretPath = Join-Path $env:USERPROFILE ".axis_codex\axis_cloud_password.dpapi"

if (-not (Test-Path -LiteralPath $secretPath)) {
    throw "Missing encrypted password file: $secretPath"
}

$payload = (Get-Content -LiteralPath $secretPath -Raw).Trim()
try {
    Add-Type -AssemblyName System.Security | Out-Null
}
catch {
}
$protected = [Convert]::FromBase64String($payload)
$bytes = [Security.Cryptography.ProtectedData]::Unprotect(
    $protected,
    $null,
    [Security.Cryptography.DataProtectionScope]::CurrentUser
)
try {
    [Text.Encoding]::UTF8.GetString($bytes)
}
finally {
    [Array]::Clear($bytes, 0, $bytes.Length)
}
