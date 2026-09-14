param(
    [string] $Remote,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemoteCommand
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Resolve-Path (Join-Path $scriptRoot "..\..\..")
$keyPath = Join-Path $workspaceRoot ".ssh\paratera_zw1_codex_ed25519"
$knownHosts = Join-Path $workspaceRoot ".ssh\known_hosts"
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"

$env:SSH_ASKPASS = Join-Path $scriptRoot "zw1_askpass.cmd"
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY = "codex-askpass"

$sshArgs = @(
    "-o", "BatchMode=no",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=$knownHosts",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "PreferredAuthentications=publickey,password,keyboard-interactive",
    "-o", "PasswordAuthentication=yes",
    "-o", "KbdInteractiveAuthentication=yes",
    "-i", $keyPath,
    "-p", "2222",
    "-o", "User=root@ackcs-00gjgyqg",
    "ssh.zw1.paratera.com"
)

if ($Remote) {
    $sshArgs += $Remote
}
elseif ($RemoteCommand.Count -gt 0) {
    $sshArgs += ($RemoteCommand -join " ")
}

& $ssh @sshArgs
exit $LASTEXITCODE
