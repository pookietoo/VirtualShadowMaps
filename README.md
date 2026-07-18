# Virtual Shadow Maps

**Real shadows for Skyrim's local lights.**

> [!CAUTION]
> Experimental. No external testing, no coverage across hardware nor mods. Many features are
> implemented and not validated. Expect bugs, visual glitches, and crashes. Use at your own risk.

Skyrim only allows shadows from a handful of local lights at once. Virtual Shadow Maps is an
[SKSE](https://skse.silverlock.org/) plugin that renders its **own** cube-shadow atlas
for the active local lights the engine dropped, and feeds it to
[Community Shaders](https://github.com/doodlum/skyrim-community-shaders) so its
`Lighting.hlsl` samples our shadows. It is **not standalone**: it depends **entirely** on Community
Shaders and does nothing without it — all of its output is produced inside Community Shaders'
rendering (see [Installing](#installing) and
[Relationship to Community Shaders](#relationship-to-community-shaders)).

> [!NOTE]
> This is a separate, **unofficial** mod. It is **not affiliated with, endorsed by, or part of**
> Community Shaders or its authors. We would love for it to become part of the
> Community Shaders project one day, but there is no association today.

## Status

Verified working in-game:

- **Unlimited local-light shadows** — every eligible local light the engine drops casts; there is
  no arbitrary light cap (the atlas and light buffer grow on demand).
- **Static local-light shadows** — rooms, furniture, held items; camera-stable.
- **Skinned-character shadows** — the player and NPC bodies, via our own CPU skinning using the
  engine's world-absolute bone matrices.
- **Variable per-light resolution + soft edges** — each light's cube-face resolution is chosen per
  frame on a power-of-two ladder (up to your `iShadowMapResolution`) by on-screen importance, and a
  PCF kernel softens the penumbra.

Implemented but not yet validated in-game:

- **Colored translucent shadows** — glass / alpha-blended surfaces dim and tint the light passing
  through them instead of being excluded (needs the matching Community Shaders build).
- **Moving-light smoothing** — swinging lanterns/torches are de-stepped so their shadows glide
  rather than jerk with the animation.

Only opaque, solid, camera-independent geometry is treated as a hard occluder; billboards, decals,
and engine "does not cast" meshes are excluded. Head/hair skinned meshes cast as solid silhouettes;
alpha-tested cutout transparency for hair cards is not wired for skinned meshes yet.

## Installing

Requires **Community Shaders**. Virtual Shadow Maps ships a companion
`CommunityShaders.dll` and shader overrides, so it must match the Community Shaders version it targets
(currently **v1.7.3**); mixing versions can break local-light shadows.

Install with a mod manager (Mod Organizer 2). The mod provides:

    SKSE/Plugins/VirtualShadowMaps.dll     this plugin
    SKSE/Plugins/CommunityShaders.dll      the companion CS build (replaces stock CS)
    Shaders/...                            CS shader overrides that sample our atlas

No configuration is required. It honors your existing `SkyrimPrefs.ini` `[Display]` shadow keys —
`iShadowMapResolution`, `fShadowBiasScale`, `fShadowDistance`, and `iBlurDeferredShadowMask` — so your
prior shadow settings carry over.

## Relationship to Community Shaders

Virtual Shadow Maps is a companion to Community Shaders' built-in **Light Limit Fix (LLF)** feature,
not a replacement, and depends on Community Shaders at runtime. Our own logic stays in
`VirtualShadowMaps.dll`; the companion
`CommunityShaders.dll` and shader overrides carry only small, `// VSM`-tagged edits, of three kinds:

- **A generic add-on menu hook** (`Menu.cpp`, `Menu.h`, `FeatureListRenderer.cpp`) — two exports plus
  a small registry that let an external plugin list a panel in the CS feature list under a category.
  VSM registers under **Lighting**. It contains nothing about shadows.
- **A per-light shadow index for Light Limit Fix** (`LightLimitFix.cpp`/`.h`, `Common.hlsli`) — one
  field on LLF's light struct that VSM fills, keyed on the shared `BSLight`, so the shader looks a
  light's shadow record up by index instead of searching per pixel. With VSM absent it reads a
  sentinel and CS is unaffected.
- **Local-light shadow sampling in `Lighting.hlsl`** — the shader calls `VSM::GetLocalShadow` per
  local light and skips the engine's own local shadow mask.

Because VSM aligns its per-light data to LLF's lights, the CS shaders and `CommunityShaders.dll`
must match the VSM build. A Community Shaders update that changes how LLF gathers or orders its lights
needs a matching VSM update; a mismatched build can produce wrong or missing local-light shadows.

The menu hook is feature-agnostic and could be upstreamed as a general extension point. The Light
Limit Fix and shader changes are VSM-specific and would be offered as an optional, VSM-aware path
rather than a change to Community Shaders' defaults.

## License

GPL-3.0 (see [`LICENSE`](LICENSE)).

The files under `Shaders/` — notably `Lighting.hlsl` and the Light Limit Fix `.hlsli` files — are
derived from [Community Shaders](https://github.com/doodlum/skyrim-community-shaders) (GPL-3.0) and
are redistributed here under the same license with our additions. All plugin code in `plugin/` is
GPL-3.0. Community Shaders is by doodlum and contributors.

## Credits

- **PookieToo** — author ([github.com/pookietoo](https://github.com/pookietoo))
- [Claude](https://claude.ai) (Anthropic) — development assistance
- [Community Shaders](https://github.com/doodlum/skyrim-community-shaders) (doodlum & contributors)
- [CommonLibSSE-NG](https://github.com/CharmedBaryon/CommonLibSSE-NG)
