# Build & Package — Owned Shadow Test (M0)

Our feature compiles **into** Community Shaders (we add 3 files + 1 registration line), so the
"mod" is a **Community Shaders build** that replaces/overrides your installed CS. This guide:
prerequisites → system settings → get the code → build → package as a testable mod.

`build.ps1` (next to this file) automates the code-copy + registration + build + deploy. Read
this once so you know what it's doing, then use the script for iteration.

---

## 1. Prerequisites (install once)

- **Visual Studio 2022 Community** (you have it) — via the VS Installer add:
  - Workload: **Desktop development with C++**
  - Individual components: **MSVC v143 build tools**, **Windows 11 SDK**, **C++ CMake tools for Windows**
- **CMake 4.2+** — CS requires it, and VS's bundled CMake is older. Install from
  <https://cmake.org/download/> and make sure its `cmake.exe` is on `PATH` *ahead of* VS's.
  Verify: `cmake --version` → must be ≥ 4.2.
- **vcpkg** — install standalone and set `VCPKG_ROOT`:
  ```pwsh
  git clone https://github.com/microsoft/vcpkg C:\vcpkg
  C:\vcpkg\bootstrap-vcpkg.bat
  setx VCPKG_ROOT C:\vcpkg     # new terminals only; restart your shell after
  ```
  (The exact dependency versions are pinned by `builtin-baseline` in the repo's `vcpkg.json`;
  a recent vcpkg is fine.)
- **Git** (you have it).
- **Skyrim SE/AE + SKSE + a mod manager (MO2)** (you have a dev-ready setup).

## 2. System settings — enable Windows long paths

The CS repo + CommonLibSSE-NG submodule exceed the legacy 260-char path limit. **Two
independent settings** matter — git ignores the OS policy unless its own flag is set:

```pwsh
# 1) Git long paths — REQUIRED even when the OS policy is on (git enforces its own limit).
git config --global core.longpaths true

# 2) OS policy — check first; may already be 1 (no action if so). If 0, set in ADMIN
#    PowerShell and reboot:
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled).LongPathsEnabled
# reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

The earlier clone failing with "Filename too long" is the **git** flag being unset, not the
OS policy — set `core.longpaths=true` and the `--recursive` clone will complete.

## 3. Get the code

Clone Community Shaders **with submodules** (this is where CommonLibSSE-NG comes from):

```pwsh
git clone https://github.com/community-shaders/skyrim-community-shaders.git --recursive
cd skyrim-community-shaders
git checkout shadow-limit-fix          # base our work on the SLF branch
git switch -c owned-vsm                 # our working branch
```

Then add our feature — either run `build.ps1 -CSPath <this-clone>` (does it for you), or manually:
- copy `src/Features/OwnedShadowTest.{h,cpp}` → `<clone>/src/Features/`
- copy `package/Shaders/OwnedShadowTest/` → `<clone>/package/Shaders/`
- register in `<clone>/src/Feature.cpp` (see `INTEGRATION.md §2`): add the include and the
  `OwnedShadowTest::GetSingleton(),` line to `Feature::GetFeatureList()`.

## 4. Build

Open **"Developer PowerShell for VS 2022"** (Start menu) so the toolchain is on PATH, then:

```pwsh
cd <clone>
cmake --preset ALL-VS2022          # first run bootstraps vcpkg deps — long, one-time (~15-40 min)
cmake --build --preset ALL-VS2022  # 'Dev' config is used for fast iteration
```

Output (DLL + shaders) lands under `build/ALL-VS2022/aio/`.

## 5. Package as a testable mod (pick one)

Because this IS Community Shaders, the result must **win over your stock CS mod** (or replace it).

**Option A — auto-deploy on every build (best for iteration).**
Set an env var to your MO2 "Community Shaders" mod folder and enable auto-deploy, then just build:
```pwsh
setx CommunityShadersOutputDir "D:\MO2\mods\Community Shaders"   # restart shell after setx
cmake --preset ALL-VS2022 -DAUTO_PLUGIN_DEPLOYMENT=ON
cmake --build --preset ALL-VS2022
```
Every build now copies the DLL + shaders into that mod folder. Launch the game to test.

**Option B — install a clean mod folder (best for a separate test mod).**
```pwsh
cmake --install build/ALL-VS2022 --prefix "D:\MO2\mods\CommunityShaders-OwnedVSM"
```
That folder is a complete MO2 mod (`SKSE\Plugins\CommunityShaders.dll` + `Shaders\...`). In MO2:
enable it and place it **after / winning over** your stock Community Shaders mod (or disable stock CS).

**Option C — distributable zip.**
```pwsh
cmake --build ./build/ALL-VS2022 --config Release --target AIO_ZIP_PACKAGE   # -> dist/*.7z
```

## 6. Test

See `INTEGRATION.md §4` — enable **Owned Shadow Test** (Lighting category) in the CS menu (`END`),
stand near a shadow-casting local light, and check the live depth-map preview + "casters: N".
`§5` there is the symptom→cause debug runbook.

---

### Notes / gotchas
- The plugin DLL is `CommunityShaders.dll` — same name as stock CS, so your build **conflicts with
  and must override** the stock CS mod. That's expected (it's a CS replacement).
- If configure fails on a missing dependency, confirm `VCPKG_ROOT` is set and you're in the VS
  Developer PowerShell.
- If `cmake --preset` errors that the generator/preset is unknown, you're likely on the wrong
  preset name — use `ALL-VS2022` for VS 2022 (plain `ALL` targets "Visual Studio 18 2026").
