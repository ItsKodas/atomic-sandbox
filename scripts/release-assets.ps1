<#
  The local release's assets (docs/dev-reference.md "Releasing"): after scripts\package.ps1
  -RequireInstaller, writes dist\SHA256SUMS.txt (the installer and the zip, sha256sum text form, LF)
  and dist\manifest.json (the exe's --write-manifest plus each asset's name, size and sha256), the
  same steps as .github/workflows/release.yml's build job. Signing is the user's, by hand.
  Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\release-assets.ps1 -Version 0.2.0-beta.5
#>
param([Parameter(Mandatory = $true)][string]$Version)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$dist = (Resolve-Path 'dist').Path
$names = @("AtomicSandbox-Setup-$Version.exe", "AtomicSandbox-$Version-windows-x64.zip")
$lines = foreach ($name in $names) {
  $f = Join-Path $dist $name
  if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { throw "dist\$name was not built." }
  '{0}  {1}' -f (Get-FileHash -LiteralPath $f -Algorithm SHA256).Hash.ToLowerInvariant(), $name
}
[System.IO.File]::WriteAllText((Join-Path $dist 'SHA256SUMS.txt'), (($lines -join "`n") + "`n"), (New-Object System.Text.ASCIIEncoding))
Get-Content -LiteralPath (Join-Path $dist 'SHA256SUMS.txt')
$exe = Join-Path $dist "AtomicSandbox-$Version-windows-x64\AtomicSandbox.exe"
$out = Join-Path $dist 'manifest.json'
if (Test-Path -LiteralPath $out) { Remove-Item -LiteralPath $out }
$p = Start-Process -FilePath $exe -ArgumentList @('--write-manifest', "`"$out`"") -Wait -PassThru -NoNewWindow
if ($p.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $out -PathType Leaf)) { throw "--write-manifest failed (exit $($p.ExitCode))" }
$m = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
if ($m.version -cne $Version) { throw "exe says version '$($m.version)'" }
$assets = @(foreach ($name in $names) {
  $f = Get-Item -LiteralPath (Join-Path $dist $name)
  [ordered]@{ name = $name; size = $f.Length; sha256 = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
})
$m | Add-Member -NotePropertyName assets -NotePropertyValue $assets -Force
$json = $m | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($out, $json + "`n", (New-Object System.Text.UTF8Encoding $false))
$back = Get-Content -LiteralPath $out -Raw | ConvertFrom-Json
if (@($back.assets).Count -ne 2 -or -not $back.formats -or -not $back.channel) { throw 'manifest.json lost a field.' }
Get-Content -LiteralPath $out
