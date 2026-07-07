# Virtual Shadow Maps

**Real shadows for Skyrim's local lights — beyond the engine's shadow budget.**

Skyrim only shadow-casts a handful of local lights at once; the rest illuminate but throw no
shadow. Virtual Shadow Maps is a standalone [SKSE](https://skse.silverlock.org/) plugin that
renders its **own** cube-shadow atlas for the active local lights the engine dropped, and feeds it
to [Community Shaders](https://github.com/doodlum/skyrim-community-shaders)' Light Limit Fix so its
`Lighting.hlsl` samples our shadows. The plugin owns all of its logic — Community Shaders only
exposes a tiny generic menu hook, so nothing of ours lives in CS's DLL.

> This is a work in progress, shared openly. I just want good shadows in Skyrim — if someone else
> takes it further or does it all, that's a win. Contributions and forks welcome.

## Status

Verified working in-game (build 0.9.35):

- **Static local-light shadows** — rooms, furniture, held items; camera-stable.
- **Skinned-character shadows** — the player and NPC bodies, via our own CPU skinning ("Path B")
  using the engine's world-absolute bone matrices.
- **In-menu UI** — the settings panel renders inside the Community Shaders menu.

Known open items (not yet solved — tracked in the design notes):

- An **interior-specific "shadows shift" issue** (observed in the Sleeping Giant Inn) is under
  investigation; the numeric/GPU probe tooling in `VirtualShadowMaps_Diagnostics.cpp` exists to
  chase it.
- **Head/hair** (`BSDynamicTriShape`) skinned meshes are decoded but not yet rendered.
- Performance work (GPU-driven skinning, static/dynamic caching, a paged atlas) is planned, not done.

## How it works

Each frame, on an `IDXGISwapChain::Present` hook, the plugin:

1. Gathers every active local light (not just the engine's shadow-limited subset) and lays out a
   3×2 cube-face "block" per light in a shared depth atlas.
2. Rebuilds a persistent registry of caster geometry from the scene graph + loaded references, and
   CPU-skins skinned casters into a world-absolute posed buffer.
3. Renders all six cube faces per light into the atlas with a minimal depth-only shader.
4. Exposes the atlas + a per-light structured buffer + sampler to Community Shaders' Light Limit
   Fix, which samples them in `Shaders/VirtualShadowMaps/VSM.hlsli`.

Coordinate space is world-absolute throughout (matching how CS treats `ShadowSceneNode` positions);
see `DESIGN.md` for the full as-built design.

## Repository layout

```
plugin/src/
  VirtualShadowMaps.{h,cpp}          core: setup, light collection, skinning, atlas render, menu
  VirtualShadowMaps_Diagnostics.cpp  numeric dumps, GPU/pixel probes, scene census, preview
  VSMConstants.h                     single source for atlas geometry (shared with the shaders)
  VSMInternal.h                      helpers shared across the two translation units
  VSMConfig.{h,cpp}                  TOML config + settings persistence
  Plugin.cpp                         SKSE entry points + the Present / OMSetRenderTargets hooks
dist/Virtual Shadow Maps/Shaders/    shipped CS shader overrides (see License below)
DESIGN.md, VSM_PHASE_*.md            design + roadmap notes
```

`extern/` (CommonLibSSE-NG, vcpkg) and `community-shaders/` are not committed — see Building.

## Building

Requirements: Windows, Visual Studio 2022+ with the C++ workload, CMake 3.21+, and
[vcpkg](https://github.com/microsoft/vcpkg). The plugin targets Skyrim SE/AE via CommonLibSSE-NG.

1. Provide the vendored dependencies under `extern/` (git-ignored):
   - `extern/vcpkg` — a vcpkg checkout
   - `extern/CommonLibSSE-NG` — a [CommonLibSSE-NG](https://github.com/CharmedBaryon/CommonLibSSE-NG) checkout
2. Configure and build the plugin (Release):
   ```sh
   cmake -S plugin -B plugin/build \
     -DCMAKE_TOOLCHAIN_FILE=extern/vcpkg/scripts/buildsystems/vcpkg.cmake
   cmake --build plugin/build --config Release
   ```
   The build deploys `VirtualShadowMaps.dll` into `dist/Virtual Shadow Maps/SKSE/Plugins/`.

The in-menu panel additionally requires a **minimally-hooked `CommunityShaders.dll`** built from a
Community Shaders **v1.7.3** checkout with a ~7-line generic add-on hook in `Menu.cpp` (two exports
+ a callback loop). CS's shaders and DLL must be the same version.

## Installing

The mod is two DLLs plus the shader overrides, as an MO2-ready tree under `dist/Virtual Shadow Maps/`:

```
SKSE/Plugins/VirtualShadowMaps.dll     this plugin
SKSE/Plugins/CommunityShaders.dll      the minimally-hooked CS build
Shaders/...                            CS shader overrides that sample our atlas
```

Runtime tunables live in `Data/SKSE/Plugins/VirtualShadowMaps.toml` (written with documented
defaults on first launch) and are also editable live from the Community Shaders menu.

## Relationship to Community Shaders & License

This project is licensed under **GPL-3.0** (see [`LICENSE`](LICENSE)).

The files under `dist/Virtual Shadow Maps/Shaders/` — notably `Lighting.hlsl` and the Light Limit
Fix `.hlsli` files — are **derived from [Community Shaders](https://github.com/doodlum/skyrim-community-shaders)**,
which is licensed under GPL-3.0, and are redistributed here under the same license with our additions
(the `VSM::` sampling path and a small diagnostic override). All original plugin code in `plugin/`
is likewise GPL-3.0. Community Shaders itself is by doodlum and contributors.

## Contributing

Issues, ideas, and pull requests are welcome — especially on the open interior "shadow shift"
behavior and head/hair rendering. If you just want to run it, grab a release; if you want to hack on
it, `DESIGN.md` and the `VSM_PHASE_*` notes are the map.

## Credits

- [Community Shaders](https://github.com/doodlum/skyrim-community-shaders) (doodlum & contributors)
- [CommonLibSSE-NG](https://github.com/CharmedBaryon/CommonLibSSE-NG)
- The technique takes inspiration from prior local-shadow work in the Skyrim modding community.
