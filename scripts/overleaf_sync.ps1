# Pushes the repo's paper/ folder to the Overleaf project through Overleaf's git bridge.
# The token is read from the DPAPI-encrypted file under C:\Users\aryan\.secrets (see the credentials note), used only for this process's
# Authorization header, and never written to a file, a remote URL or the output.
#
# Usage:  powershell -File scripts/overleaf_sync.ps1 -Message "intro: tighten the contributions"
#         powershell -File scripts/overleaf_sync.ps1 -PullOnly     # bring Overleaf-side edits into paper/ first
param(
    [string]$Message = "update from repository",
    [switch]$PullOnly,
    [string]$ProjectId = "6ac56bc536f86446b37d6e93"
)
$ErrorActionPreference = "Stop"
$repoPaper = Join-Path (Split-Path $PSScriptRoot -Parent) "paper"
$work = "C:\Users\aryan\.claude\jobs\02cd3b07\tmp\overleaf_repo"

$s = Get-Content "C:\Users\aryan\.secrets\overleaf_git_token.dpapi" | ConvertTo-SecureString
$b = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($s)
$tok = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($b)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($b)
$hdr = "Authorization: Basic " + [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("git:$tok"))
$tok = $null
$url = "https://git.overleaf.com/$ProjectId"
function G { & git -c "http.extraheader=$hdr" @args; if ($LASTEXITCODE -ne 0) { throw "git $($args[0]) failed" } }

if (-not (Test-Path "$work\.git")) { G clone $url $work }
Set-Location $work
G pull --ff-only origin main

if ($PullOnly) {
    Get-ChildItem $work -Exclude .git | Copy-Item -Destination $repoPaper -Recurse -Force
    "pulled Overleaf state into $repoPaper"
    exit 0
}

# mirror paper/ into the Overleaf clone (everything except build files)
Get-ChildItem $work -Exclude .git | Remove-Item -Recurse -Force
Get-ChildItem $repoPaper -Exclude "README.md", "build" | Copy-Item -Destination $work -Recurse -Force
& git add -A
& git diff --cached --quiet
if ($LASTEXITCODE -eq 0) { "nothing to push"; exit 0 }
& git -c user.name="Aryan Bhatt" -c user.email="ab1340@rit.edu" commit -q -m $Message
G push origin HEAD:main
"pushed: $Message"
