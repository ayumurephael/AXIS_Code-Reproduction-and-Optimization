param(
    [Parameter(Mandatory = $true)]
    [string] $RemotePath,

    [Parameter(Mandatory = $true)]
    [string] $LocalPath
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Resolve-Path (Join-Path $scriptRoot "..\..\..")
$keyPath = Join-Path $workspaceRoot ".ssh\paratera_zw1_codex_ed25519"
$knownHosts = Join-Path $workspaceRoot ".ssh\known_hosts"
$scp = "$env:WINDIR\System32\OpenSSH\scp.exe"

$env:SSH_ASKPASS = Join-Path $scriptRoot "zw1_askpass.cmd"
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY = "codex-askpass"

$source = "ssh.zw1.paratera.com:$RemotePath"
$scpArgs = @(
    "-o", "BatchMode=no",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=$knownHosts",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "PreferredAuthentications=publickey,password,keyboard-interactive",
    "-o", "PasswordAuthentication=yes",
    "-o", "KbdInteractiveAuthentication=yes",
    "-i", $keyPath,
    "-P", "2222",
    "-o", "User=root@ackcs-00gjgyqg",
    $source,
    $LocalPath
)

& $scp @scpArgs
exit $LASTEXITCODE
