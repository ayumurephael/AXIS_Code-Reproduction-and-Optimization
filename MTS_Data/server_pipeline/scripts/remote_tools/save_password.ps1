$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$secretDir = Join-Path $env:USERPROFILE ".axis_codex"
$secretPath = Join-Path $secretDir "axis_cloud_password.dpapi"

New-Item -ItemType Directory -Path $secretDir -Force | Out-Null

function ConvertTo-PlainText {
    param([securestring] $Secure)
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try {
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

function Protect-TextWithDpapi {
    param([string] $Text)
    try {
        Add-Type -AssemblyName System.Security | Out-Null
    }
    catch {
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes($Text)
    $protected = [Security.Cryptography.ProtectedData]::Protect(
        $bytes,
        $null,
        [Security.Cryptography.DataProtectionScope]::CurrentUser
    )
    [Convert]::ToBase64String($protected)
}

Write-Host ""
Write-Host "AXIS cloud SSH password setup"
Write-Host "This stores the password encrypted with Windows DPAPI for the current Windows user."
Write-Host "The plaintext password will not be printed."
Write-Host ""

$maxAttempts = 3
for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $secure = Read-Host "Enter the remote SSH password" -AsSecureString
    if ($secure.Length -eq 0) {
        Write-Host "Empty password was not saved."
        continue
    }

    $confirm = Read-Host "Enter the same password again" -AsSecureString
    $plain = ConvertTo-PlainText $secure
    $plainConfirm = ConvertTo-PlainText $confirm

    if ($plain -ne $plainConfirm) {
        Write-Host ""
        Write-Host "Passwords did not match. Try again."
        Write-Host ""
        continue
    }

    try {
        Protect-TextWithDpapi $plain | Set-Content -LiteralPath $secretPath -Encoding ASCII
    }
    catch {
        Write-Host ""
        Write-Host "Failed to save encrypted password:"
        Write-Host $_.Exception.Message
        Write-Host ""
        Read-Host "Press Enter to close this window"
        exit 1
    }
    Write-Host ""
    Write-Host "Saved encrypted password to: $secretPath"
    Write-Host "Codex can now use SSH_ASKPASS for this cloud instance."
    Write-Host ""
    Read-Host "Press Enter to close this window"
    exit 0
}

Write-Host ""
Write-Host "Password was not saved after $maxAttempts failed attempts."
Write-Host ""
Read-Host "Press Enter to close this window"
exit 1
