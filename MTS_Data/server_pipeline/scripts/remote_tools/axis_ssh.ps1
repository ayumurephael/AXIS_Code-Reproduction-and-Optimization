param(
    [string] $Remote,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemoteCommand
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$keyPath = "C:\Users\31029\.ssh\axis_cloud_nopass_ed25519"
$ssh = "C:\Windows\System32\OpenSSH\ssh.exe"

$env:SSH_ASKPASS = Join-Path $scriptRoot "axis_askpass.cmd"
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY = "codex-askpass"

$sshArgs = @(
    "-o", "BatchMode=no",
    "-o", "IdentitiesOnly=yes",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "PreferredAuthentications=publickey,password,keyboard-interactive",
    "-o", "PasswordAuthentication=yes",
    "-o", "KbdInteractiveAuthentication=yes",
    "-i", $keyPath,
    "-p", "2233",
    "-o", "User=root@ackcs-00gjgw1m",
    "ssh.bj8.bz1.paratera.com"
)

if ($Remote) {
    $sshArgs += $Remote
}
elseif ($RemoteCommand.Count -gt 0) {
    $sshArgs += ($RemoteCommand -join " ")
}

& $ssh @sshArgs
exit $LASTEXITCODE
