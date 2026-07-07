<#
.SYNOPSIS
  Add the Owned Shadow Test feature to a Community Shaders checkout, build it, and
  optionally install it as an MO2-ready mod. Idempotent - safe to re-run.

.DESCRIPTION
  Run from "Developer PowerShell for VS" (so cmake/vcpkg are on PATH) with CMake 4.2+.
  See BUILD.md for prerequisites and system setup (long paths, vcpkg, VCPKG_ROOT).

.PARAMETER CSPath
  Path to a cloned skyrim-community-shaders repo (cloned --recursive; branch owned-vsm).

.PARAMETER ModDir
  Optional. If set, installs the built AIO here (becomes an MO2 mod that must win over /
  replace your stock Community Shaders mod).

.PARAMETER Preset
  CMake preset. Default ALL-VS2022 (use ALL for VS 2026 / "Visual Studio 18 2026").

.PARAMETER SkipBuild
  Only copy files + register the feature; do not configure/build.

.EXAMPLE
  .\build.ps1 -CSPath D:\src\skyrim-community-shaders -ModDir "D:\MO2\mods\CS-OwnedVSM" -Preset ALL
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CSPath,
    [string]$ModDir = "",
    [string]$Preset = "ALL-VS2022",
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot   # this repo (ShadowLimitFix): holds src/ and package/

function Info($m) { Write-Host "[owned-vsm] $m" -ForegroundColor Cyan }
function Die($m)  { Write-Host "[owned-vsm] ERROR: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path (Join-Path $CSPath "src\Feature.cpp"))) {
    Die "Not a Community Shaders checkout: '$CSPath' (src\Feature.cpp not found)."
}

# --- 1) copy our feature files in ----------------------------------------------
Info "Copying feature files into $CSPath ..."
Copy-Item (Join-Path $src "src\Features\OwnedShadowTest.h")   (Join-Path $CSPath "src\Features\") -Force
Copy-Item (Join-Path $src "src\Features\OwnedShadowTest.cpp") (Join-Path $CSPath "src\Features\") -Force
$shDest = Join-Path $CSPath "package\Shaders\OwnedShadowTest"
New-Item -ItemType Directory -Force -Path $shDest | Out-Null
Copy-Item (Join-Path $src "package\Shaders\OwnedShadowTest\*") $shDest -Force

# --- 2) register the feature in Feature.cpp (idempotent) -----------------------
$featPath  = Join-Path $CSPath "src\Feature.cpp"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$text      = [System.IO.File]::ReadAllText($featPath)
$marker    = 'OwnedShadowTest::GetSingleton'

if ($text.Contains($marker)) {
    Info "Feature.cpp already registers OwnedShadowTest - skipping patch."
}
else {
    $lines = [System.IO.File]::ReadAllLines($featPath)
    $out   = New-Object System.Collections.Generic.List[string]
    $tab   = [char]9
    $regLine = "$tab$tab" + 'OwnedShadowTest::GetSingleton(),'

    $lastInclude = -1
    for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^\s*#include') { $lastInclude = $i } }

    $listInserted = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $out.Add($lines[$i])
        if ($i -eq $lastInclude) { $out.Add('#include "Features/OwnedShadowTest.h"') }
        if ((-not $listInserted) -and ($lines[$i] -match 'static\s+std::vector<Feature\*>\s+features\s*=\s*\{')) {
            $out.Add($regLine)
            $listInserted = $true
        }
    }
    if (-not $listInserted) {
        Die "Could not find the feature list in Feature.cpp. Add the registration line manually (see INTEGRATION.md section 2)."
    }
    [System.IO.File]::WriteAllLines($featPath, $out, $utf8NoBom)
    Info "Registered OwnedShadowTest in Feature.cpp."
}

if ($SkipBuild) { Info "Files in place (-SkipBuild). Done."; exit 0 }

# --- 3) configure + build ------------------------------------------------------
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Die "cmake not on PATH. Open 'Developer PowerShell for VS' and install CMake 4.2+."
}

Push-Location $CSPath
try {
    Info "Configuring (preset $Preset) - first run bootstraps vcpkg deps, this is slow ..."
    cmake --preset $Preset
    if ($LASTEXITCODE -ne 0) { Die "cmake configure failed." }

    Info "Building (preset $Preset) ..."
    cmake --build --preset $Preset
    if ($LASTEXITCODE -ne 0) { Die "cmake build failed." }

    if ($ModDir -ne "") {
        $bin = Join-Path $CSPath "build\$Preset"
        if (-not (Test-Path (Join-Path $bin "CMakeCache.txt"))) { $bin = Join-Path $CSPath "build\ALL" }
        Info "Installing AIO to mod folder: $ModDir"
        cmake --install $bin --prefix $ModDir
        if ($LASTEXITCODE -ne 0) { Die "cmake install failed." }
        Info "Installed. In MO2, enable this mod and make it win over stock Community Shaders."
    }
    else {
        Info "Build complete. Output under build\$Preset\aio\. Pass -ModDir to install as a mod."
    }
}
finally { Pop-Location }

Info "Done."
