# Virtual Shadow Maps

**Real shadows for Skyrim's local lights.**

Skyrim only allows shadows from a handful of local lights at once. Virtual Shadow Maps is a 
standalone [SKSE](https://skse.silverlock.org/) plugin that
renders its **own** cube-shadow atlas for the active local lights the engine dropped, and feeds it
to [Community Shaders](https://github.com/doodlum/skyrim-community-shaders)' Light Limit Fix so its
`Lighting.hlsl` samples our shadows. The plugin owns all of its shadow logic in its own DLL; Community
Shaders carries only a small generic menu hook and the Light Limit Fix coupling described
[below](#changes-to-community-shaders) — never VSM's shadow generation.

> **Requires Community Shaders with Light Limit Fix — this is not a standalone shadow renderer.**
> Virtual Shadow Maps *generates* a shadow atlas and hands it to Community Shaders' Light Limit Fix (LLF),
> whose `Lighting.hlsl` samples it per local light and whose clustered (froxel) light system it plugs into.
> Without CS + LLF the plugin does nothing. Because the integration is at the shader + light-buffer level and
> **VSM aligns its per-light data to LLF's own light ordering**, a **compatible CS / LLF build is required** —
> the shaders and the `CommunityShaders.dll` must match. See
> [Light Limit Fix coupling](#relationship-to-community-shaders--license) for what an incompatible build breaks.

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

Opt-in performance modules (default **off**, configured in the `.toml`) — static/dynamic shadow
caching (with an optional per-light *incremental* re-bake), and off-screen light culling. Enable them
to trade a little accuracy for speed once you've confirmed the baseline looks right.

Newer modules, **implemented but not yet validated in-game** (treat as experimental): **colored
translucent shadows** (default off) — glass / alpha-blended surfaces dim and tint the light passing
through them instead of being excluded. The translucent module additionally needs the matching
Community Shaders build (see below).

Only opaque, solid, camera-independent geometry is treated as a hard occluder; billboards, decals, and
engine "does not cast" meshes are excluded. Head/hair skinned meshes **do** cast (as solid silhouettes);
alpha-tested cutout transparency for hair cards isn't wired for skinned meshes yet.

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

Full integration additionally requires a companion **`CommunityShaders.dll`** built from a compatible
Community Shaders **v1.7.3** checkout carrying VSM's small, clearly-marked edits (see
[Changes to Community Shaders](#changes-to-community-shaders) below). CS's shaders and DLL must be the
same version.

## Installing

The build produces an MO2-ready tree in `dist/Virtual Shadow Maps/` (local only — not part of this repo):

```
SKSE/Plugins/VirtualShadowMaps.dll     this plugin
SKSE/Plugins/CommunityShaders.dll      the minimally-hooked CS build
Shaders/...                            CS shader overrides that sample our atlas
```

Runtime settings live in `Data/SKSE/Plugins/VirtualShadowMaps.toml` (optional — sensible defaults are
used if it is absent). Because this
plugin replaces Skyrim's own local-light shadows, it also honors your existing `SkyrimPrefs.ini`
`[Display]` shadow keys — `iShadowMapResolution` (max per-light resolution), `fShadowBiasScale`,
`fShadowDistance`, and `iBlurDeferredShadowMask` (soft-shadow radius) — so your prior intent carries
over.

The `.toml` also carries optional, **off-by-default** modules, each documented inline in the file:
colored translucent shadows (`translucentShadows`), static-shadow caching (`cacheStaticShadows` /
`incrementalCache`), off-screen light culling (`cullCasters`), and per-shape cast overrides
(`forceCast` / `forceNoCast`). Strictly-better optimizations and correctness features — the world-space
spatial caster index, empty-light-pass culling, and alpha-tested cutout silhouettes for foliage/grates — are
**always on** (no toggle); they change nothing visible and are bisectable through version history if needed.

## Relationship to Community Shaders

### Light Limit Fix coupling

The integration with Community Shaders' **Light Limit Fix (LLF)** is deliberately tight, and it is a hard
runtime dependency — VSM is a *companion* to LLF, not a replacement for it:

- **Shader-level integration.** VSM binds its shadow atlas and per-light data into LLF's lighting prepass
  (the depth atlas at `t110`, our per-light shadow buffer at `t111`, sampler `s7`, and an optional
  transmittance atlas at `t112`), and LLF's `Lighting.hlsl` calls our `VSM::GetLocalShadow` once per local
  light per pixel. LLF's clustered (froxel) light grid is what bounds how many lights each pixel evaluates.
- **Light-index alignment.** To keep the per-pixel sampling affordable as the light count grows, each of LLF's
  lights **carries VSM's shadow-buffer index** (a small field VSM fills, keyed on the shared `BSLight`), so the
  shader looks a light's shadow parameters up **directly by that index** rather than searching our buffer per
  pixel (which would be O(lights²) per pixel — untenable for the "unlimited lights" goal). This spans both of
  LLF's light stores (its per-draw *strict* lights and its clustered/froxel lights).
- **Consequence — version coupling.** VSM is therefore coupled to LLF's light collection: VSM's companion
  `CommunityShaders.dll` and shaders are modified to carry and read that index. A Community Shaders / Light Limit
  Fix update that changes **how LLF gathers, filters, or orders its lights** requires a matching VSM update.
  Running mismatched builds can **misalign the light lookup and produce wrong or missing local-light shadows**
  (and, for the translucent module, requires the `CommunityShaders.dll` that binds `t112`, or local lights can go
  dark). Always run the CS shaders and DLL that ship with (or are documented as compatible with) your VSM build.

### Changes to Community Shaders

VSM ships a companion `CommunityShaders.dll` and a few shader overrides built from a compatible CS
checkout. Every change is small, self-contained, and marked in-source (`// VSM`) so it is easy to review
and re-apply across CS updates. There are three kinds:

- **A generic add-on menu hook** (`Menu.cpp`, `Menu.h`, `FeatureListRenderer.cpp`) — two exports
  (`CS_GetImGui`, `CS_RegisterExternalMenu`) plus a small registry that lets *any* external SKSE plugin
  list its own panel in the CS feature list under a named category, sharing CS's ImGui context. VSM
  registers under **Lighting**, beside Light Limit Fix. This is feature-agnostic — it contains nothing
  about shadows — and it is the piece we would most like to upstream: a supported add-on API would let
  plugins like VSM integrate without forking CS at all.
- **A per-light shadow index for Light Limit Fix** (`LightLimitFix.cpp`/`.h`, `Common.hlsli`) — one
  spare field on LLF's light struct that VSM fills, keyed on the shared `BSLight`, so the shader can look
  a light's shadow record up in O(1) instead of searching per pixel (see *Light Limit Fix coupling*
  above). It degrades gracefully: with VSM absent the field reads a sentinel and CS behaves normally.
- **Local-light shadow sampling in `Lighting.hlsl`** — the lighting shader calls `VSM::GetLocalShadow`
  for each local light and skips the engine's own (now-unused) local shadow mask, so our atlas supplies
  the shadow the engine dropped.

**Merge intent.** Our own logic stays in `VirtualShadowMaps.dll`, never in CS. The menu hook is the part
we would like to see land in Community Shaders as a general extension point. The Light Limit Fix and
shader changes are the coupling VSM needs; we would contribute them as an **optional, VSM-aware
integration** rather than a hard change to CS's defaults, and we keep every edit minimal and `// VSM`-tagged
to make that merge as painless as possible.

## License

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
