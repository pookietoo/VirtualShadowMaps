# Virtual Shadow Maps

**Real shadows for Skyrim's local lights — beyond the engine's shadow budget.**

Skyrim only shadow-casts a handful of local lights at once; the rest illuminate but throw no
shadow. Virtual Shadow Maps is a standalone [SKSE](https://skse.silverlock.org/) plugin that
renders its **own** cube-shadow atlas for the active local lights the engine dropped, and feeds it
to [Community Shaders](https://github.com/doodlum/skyrim-community-shaders)' Light Limit Fix so its
`Lighting.hlsl` samples our shadows. The plugin owns all of its logic — Community Shaders only
exposes a tiny generic menu hook, so nothing of ours lives in CS's DLL.

## Status

Verified working in-game:

- **Unlimited local-light shadows** — every eligible local light the engine drops casts; there is
  no arbitrary light cap (the atlas and light buffer grow on demand).
- **Static local-light shadows** — rooms, furniture, held items; camera-stable.
- **Skinned-character shadows** — the player and NPC bodies, via our own CPU skinning using the
  engine's world-absolute bone matrices.
- **Variable per-light resolution + soft edges** — each light's cube-face resolution is chosen per
  frame on a power-of-two ladder (floor up to your `iShadowMapResolution`) by on-screen importance,
  and a PCF kernel softens the penumbra.
- **In-menu UI** — the settings panel renders inside the Community Shaders menu.

Opt-in performance modules (default **off**, toggled live in the menu) — static/dynamic shadow
caching (with an optional per-light *incremental* re-bake), and off-screen light culling. Enable them
to trade a little accuracy for speed once you've confirmed the baseline looks right.

Newer modules, **implemented but not yet validated in-game** (treat as experimental): a far-distance
fade that softens a shadow toward the edge of its light's radius, and **colored translucent shadows**
(default off) — glass / alpha-blended surfaces dim and tint the light passing through them instead of
being excluded. The translucent module additionally needs the matching Community Shaders build (see
below).

Only opaque, solid, camera-independent geometry is treated as a hard occluder; billboards,
decals, and engine "does not cast" meshes are excluded, and head/hair skinned meshes don't yet cast.

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
   The build deploys the DLL and copies the `Shaders/` overrides into `dist/Virtual Shadow Maps/` —
   a local, git-ignored mod folder (not published).

The default is a shipping build. Pass `-DVSM_DEV_BUILD=ON` to compile in the development
diagnostics/logging/debug apparatus.

The in-menu panel additionally requires a **minimally-hooked `CommunityShaders.dll`** built from a
Community Shaders **v1.7.3** checkout with a ~7-line generic add-on hook in `Menu.cpp` (two exports
+ a callback loop). CS's shaders and DLL must be the same version.

## Installing

The build produces an MO2-ready tree in `dist/Virtual Shadow Maps/` (local only — not part of this repo):

```
SKSE/Plugins/VirtualShadowMaps.dll     this plugin
SKSE/Plugins/CommunityShaders.dll      the minimally-hooked CS build
Shaders/...                            CS shader overrides that sample our atlas
```

Runtime settings live in `Data/SKSE/Plugins/VirtualShadowMaps.toml` (written with a documented
default on first launch) and are also editable live from the Community Shaders menu. Because this
plugin replaces Skyrim's own local-light shadows, it also honors your existing `SkyrimPrefs.ini`
`[Display]` shadow keys — `iShadowMapResolution` (max per-light resolution), `fShadowBiasScale`,
`fShadowDistance`, and `iBlurDeferredShadowMask` (soft-shadow radius) — so your prior intent carries
over.

The `.toml` also carries optional, **off-by-default** modules, each documented inline in the file:
colored translucent shadows (`translucentShadows`), static-shadow caching (`cacheStaticShadows` /
`incrementalCache`), alpha-tested cutout silhouettes for foliage/grates (`alphaTestedShadows`),
per-shape cast overrides (`forceCast` / `forceNoCast`), and a couple of performance toggles.

## Relationship to Community Shaders & License

This project is licensed under **GPL-3.0** (see [`LICENSE`](LICENSE)).

The files under `Shaders/` — notably `Lighting.hlsl` and the Light Limit
Fix `.hlsli` files — are **derived from [Community Shaders](https://github.com/doodlum/skyrim-community-shaders)**,
which is licensed under GPL-3.0, and are redistributed here under the same license with our additions
(the `VSM::` sampling path and a small diagnostic override). All original plugin code in `plugin/`
is likewise GPL-3.0. Community Shaders itself is by doodlum and contributors.

## Contributing

Issues, ideas, and pull requests are welcome.

## Credits

- [Community Shaders](https://github.com/doodlum/skyrim-community-shaders) (doodlum & contributors)
- [CommonLibSSE-NG](https://github.com/CharmedBaryon/CommonLibSSE-NG)
- The technique takes inspiration from prior local-shadow work in the Skyrim modding community.
