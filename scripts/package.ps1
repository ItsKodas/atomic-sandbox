<#
.SYNOPSIS
  Lays out Atomic Sandbox's release from committed files only: the portable folder and zip, the
  installer's staging folder and, when Inno Setup's ISCC.exe is found, the installer.

.DESCRIPTION
  Shared by build.bat (after its cargo build) and the release workflow (installer plan Task 6), so a
  local package equals the CI one. Non-interactive: it never pauses, and exits non-zero on failure.

  Committed files only (spec "Builder-check rulings"): models\, profiles\, scenes\, timings\, the
  preset scenes and docs\images\ come from `git ls-files`, so untracked models (bh3.model,
  nh4+.model, ph3.model), personal profiles and the shared collections (models\collections\) never
  ship. A tracked file with uncommitted changes is packaged as it is on disk, with a warning.

  dist\AtomicSandbox-<ver>-windows-x64\        the portable folder (portable.txt beside the exe:
      AtomicSandbox.exe, portable.txt          its data stays beside the exe)
      README.md, LICENSE.md, docs\images\
      scenes\     the presets (crates\app\presets) and the committed scenes\ library
      models\     the committed models\ (never models\collections\)
      profiles\   the committed profiles\ (the README only: personal profiles never ship)
      timings\    the committed timings\, when there is one
  dist\AtomicSandbox-<ver>-windows-x64.zip     that folder zipped (forward-slash entry names)
  dist\installer-stage\                        what installer\AtomicSandbox.iss packs:
      AtomicSandbox.exe, installed.txt, README.md, LICENSE.md, docs\images\
      bundled\models\*.model, bundled\scenes\*.scene, bundled\timings\
      AtomicSandbox.ico                        the setup's icon (AtomicSandbox.exe --write-icon)
  dist\AtomicSandbox-Setup-<ver>.exe           the installer (ISCC.exe found, -NoInstaller not given)

.PARAMETER Exe
  The built app (default target\release\app.exe), copied in as AtomicSandbox.exe.

.PARAMETER RequireInstaller
  Fail when ISCC.exe isn't found (CI). Without it a missing Inno Setup only skips the installer.

.PARAMETER NoInstaller
  Lay out the staging folder but don't run ISCC.exe.
#>
[CmdletBinding()]
param(
    [string]$Exe = 'target\release\app.exe',
    [switch]$RequireInstaller,
    [switch]$NoInstaller
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2

$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
# git prints -z paths as UTF-8: read them as UTF-8, so non-ASCII names survive
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false

function Fail([string]$message) {
    Write-Host ''
    Write-Host $message
    exit 1
}

function Write-Ascii([string]$path, [string]$text) {
    [System.IO.File]::WriteAllText($path, $text, (New-Object System.Text.ASCIIEncoding))
}

# ---- the version: the root Cargo.toml's  version = "x.y.z"  (the [workspace.package] one) ----
$match = Select-String -LiteralPath (Join-Path $repo 'Cargo.toml') -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
if (-not $match) { Fail 'Could not read the version from the root Cargo.toml.' }
$version = $match.Matches[0].Groups[1].Value

$exePath = if ([System.IO.Path]::IsPathRooted($Exe)) { $Exe } else { Join-Path $repo $Exe }
if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) { Fail "No built app at $exePath (cargo build -p app --release first)." }

# ---- committed files only ----
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail 'git was not found: a release holds committed files only, so packaging needs git and the repository.'
}
$inside = & git rev-parse --is-inside-work-tree
if ($LASTEXITCODE -ne 0 -or $inside -ne 'true') { Fail "$repo is not a git checkout: a release holds committed files only." }

$sources = @('models', 'profiles', 'scenes', 'timings', 'crates/app/presets', 'docs/images', 'README.md', 'LICENSE.md')

# repository-relative paths (forward slashes) of the committed files under $paths
function Get-Committed([string[]]$paths) {
    $out = & git -c core.quotepath=off ls-files -z -- @paths
    if ($LASTEXITCODE -ne 0) { Fail "git ls-files failed for $($paths -join ', ')." }
    $list = @((@($out) -join "`n") -split "`0" | Where-Object { $_ -ne '' })
    # the shared collections are this machine's own training data (M8a ruling P16): never packaged,
    # even if one were ever committed by mistake
    return @($list | Where-Object { -not $_.StartsWith('models/collections/') })
}

$dirty = & git -c core.quotepath=off status --porcelain --untracked-files=no -- @sources
if ($LASTEXITCODE -ne 0) { Fail 'git status failed.' }
if (@($dirty | Where-Object { $_ }).Count -gt 0) {
    Write-Warning "These committed files have uncommitted changes; the package takes them as they are on disk (CI packages the commit):"
    $dirty | ForEach-Object { Write-Warning "  $_" }
}

# copies each committed $files entry (under $prefix, e.g. 'models/') to $destRoot\<the rest of its path>
function Copy-Committed([string[]]$files, [string]$prefix, [string]$destRoot) {
    foreach ($rel in $files) {
        if (-not $rel.StartsWith($prefix)) { continue }
        $src = Join-Path $repo ($rel -replace '/', '\')
        if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
            Write-Warning "$rel is committed but missing on disk: not packaged (CI would package it)."
            continue
        }
        $dest = Join-Path $destRoot ($rel.Substring($prefix.Length) -replace '/', '\')
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        Copy-Item -LiteralPath $src -Destination $dest -Force
    }
}

function Copy-RepoFile([string]$rel, [string]$destDir, [switch]$Optional) {
    $src = Join-Path $repo $rel
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        if ($Optional) { return }
        Fail "$rel is missing."
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $destDir (Split-Path -Leaf $rel)) -Force
}

$models = Get-Committed @('models')
$profiles = Get-Committed @('profiles')
$scenes = Get-Committed @('scenes')
$timings = Get-Committed @('timings')
$presets = @(Get-Committed @('crates/app/presets') | Where-Object { $_ -like '*.scene' })
$images = Get-Committed @('docs/images')
$topFiles = Get-Committed @('README.md', 'LICENSE.md')
if ($topFiles -notcontains 'README.md') { Fail 'README.md is not committed.' }

# ---- clear the previous output ----
$dist = Join-Path $repo 'dist'
$name = "AtomicSandbox-$version-windows-x64"
$portable = Join-Path $dist $name
$stage = Join-Path $dist 'installer-stage'
$zip = Join-Path $dist "$name.zip"
$setup = Join-Path $dist "AtomicSandbox-Setup-$version.exe"
New-Item -ItemType Directory -Force -Path $dist | Out-Null
foreach ($d in @($portable, $stage)) {
    if (Test-Path -LiteralPath $d) {
        try { Remove-Item -LiteralPath $d -Recurse -Force } catch {
            Fail "Could not replace ${d}: close the Atomic Sandbox started from dist\ and run this again ($($_.Exception.Message))."
        }
    }
}
foreach ($f in @($zip, $setup)) {
    if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force }
}

# ---- the portable folder ----
New-Item -ItemType Directory -Force -Path $portable | Out-Null
Copy-Item -LiteralPath $exePath -Destination (Join-Path $portable 'AtomicSandbox.exe') -Force
Write-Ascii (Join-Path $portable 'portable.txt') "Atomic Sandbox $version, portable: its models, profiles, scenes and captures stay beside AtomicSandbox.exe.`r`n"
Copy-RepoFile 'README.md' $portable
Copy-RepoFile 'LICENSE.md' $portable -Optional
Copy-Committed $images 'docs/images/' (Join-Path $portable 'docs\images')
# the presets as files (--scene scenes\h2o.scene), then the committed library (it wins a name clash, as before)
Copy-Committed $presets 'crates/app/presets/' (Join-Path $portable 'scenes')
Copy-Committed $scenes 'scenes/' (Join-Path $portable 'scenes')
Copy-Committed $models 'models/' (Join-Path $portable 'models')
Copy-Committed $profiles 'profiles/' (Join-Path $portable 'profiles')
Copy-Committed $timings 'timings/' (Join-Path $portable 'timings')

# ---- the installer's staging folder ----
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item -LiteralPath $exePath -Destination (Join-Path $stage 'AtomicSandbox.exe') -Force
# crates/app/src/paths.rs: this marker beside the exe is Installed mode
Write-Ascii (Join-Path $stage 'installed.txt') "installed by AtomicSandbox-Setup-$version`r`n"
Copy-RepoFile 'README.md' $stage
Copy-RepoFile 'LICENSE.md' $stage -Optional
Copy-Committed $images 'docs/images/' (Join-Path $stage 'docs\images')
$bundled = Join-Path $stage 'bundled'
Copy-Committed @($models | Where-Object { $_ -like '*.model' }) 'models/' (Join-Path $bundled 'models')
Copy-Committed $presets 'crates/app/presets/' (Join-Path $bundled 'scenes')
Copy-Committed @($scenes | Where-Object { $_ -like '*.scene' }) 'scenes/' (Join-Path $bundled 'scenes')
Copy-Committed $timings 'timings/' (Join-Path $bundled 'timings')

# the setup's icon: the app's own .ico (the one build.rs embeds), from its hidden flag. A windowed
# exe: Start-Process -Wait, or PowerShell wouldn't wait for it.
$ico = Join-Path $stage 'AtomicSandbox.ico'
$p = Start-Process -FilePath $exePath -ArgumentList @('--write-icon', "`"$ico`"") -Wait -PassThru -NoNewWindow
if ($p.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $ico -PathType Leaf)) {
    Fail "AtomicSandbox.exe --write-icon failed (exit $($p.ExitCode)): $ico"
}

# ---- the zip: the portable folder, entries named with forward slashes (a backslash entry name
# unpacks as one long file name outside Windows) ----
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipped = $false
for ($try = 1; $try -le 5 -and -not $zipped; $try++) {
    # a few tries: a virus scan of the just-copied exe can hold it for a moment
    try {
        $archive = [System.IO.Compression.ZipFile]::Open($zip, [System.IO.Compression.ZipArchiveMode]::Create)
        try {
            Get-ChildItem -LiteralPath $portable -Recurse -File | Sort-Object FullName | ForEach-Object {
                $entry = "$name/" + ($_.FullName.Substring($portable.Length + 1) -replace '\\', '/')
                [void][System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $_.FullName, $entry, [System.IO.Compression.CompressionLevel]::Optimal)
            }
        } finally {
            $archive.Dispose()
        }
        $zipped = $true
    } catch {
        if (Test-Path -LiteralPath $zip) { Remove-Item -LiteralPath $zip -Force }
        if ($try -lt 5) { Start-Sleep -Seconds 2 } else { Fail "The folder is ready but zipping it failed: $portable ($($_.Exception.Message))" }
    }
}

Write-Host "Dist:   $portable\AtomicSandbox.exe"
Write-Host "Zip:    $zip"
Write-Host "Stage:  $stage"

# ---- the installer ----
if ($NoInstaller) {
    Write-Host 'Setup:  skipped (-NoInstaller)'
    exit 0
}
$iscc = $null
$found = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($found) {
    $iscc = $found.Source
} else {
    foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
        if (Test-Path -LiteralPath $c -PathType Leaf) { $iscc = $c; break }
    }
}
if (-not $iscc) {
    if ($RequireInstaller) { Fail 'ISCC.exe (Inno Setup 6) was not found, and -RequireInstaller was given.' }
    Write-Host 'Setup:  skipped (Inno Setup 6 not found; install it to build the installer too)'
    exit 0
}
& $iscc /Q "/DAppVersion=$version" "/DStageDir=$stage" "/DIconFile=$ico" "/O$dist" (Join-Path $repo 'installer\AtomicSandbox.iss')
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $setup -PathType Leaf)) {
    Fail "ISCC.exe failed (exit $LASTEXITCODE): the installer was not built."
}
Write-Host "Setup:  $setup"
exit 0
