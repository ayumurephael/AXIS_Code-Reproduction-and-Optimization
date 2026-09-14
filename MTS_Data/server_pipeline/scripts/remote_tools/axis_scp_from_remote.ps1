param(
    [Parameter(Mandatory = $true)]
    [string] $RemotePath,

    [Parameter(Mandatory = $true)]
    [string] $LocalPath
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$keyPath = "C:\Users\31029\.ssh\axis_cloud_nopass_ed25519"
$scp = "C:\Windows\System32\OpenSSH\scp.exe"

$env:SSH_ASKPASS = Join-Path $scriptRoot "axis_askpass.cmd"
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY = "codex-askpass"

$source = "ssh.bj8.bz1.paratera.com:$RemotePath"
$scpArgs = @(
    "-o", "BatchMode=no",
    "-o", "IdentitiesOnly=yes",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "PreferredAuthentications=publickey,password,keyboard-interactive",
    "-o", "PasswordAuthentication=yes",
    "-o", "KbdInteractiveAuthentication=yes",
    "-i", $keyPath,
    "-P", "2233",
    "-o", "User=root@ackcs-00gjgw1m",
    $source,
    $LocalPath
)

& $scp @scpArgs
exit $LASTEXITCODE
