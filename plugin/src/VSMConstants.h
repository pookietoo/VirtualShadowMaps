#pragma once

#include <cstddef>
#include <cstdint>

// ============================================================================
// Single source of truth for the shadow-atlas geometry and fixed capacities.
//
// IMPORTANT: these are the COMPILE-TIME FALLBACK / capacity values. The live atlas is sized at RUNTIME
// from iShadowMapResolution (rtFaceRes/rtAtlasW/rtAtlasH), and both the CS-side shader (VSM.hlsli) and the
// plugin's own preview shader use the RUNTIME dims (via the b13 cbuffer / dimension-agnostic sampling) —
// NOT these constants. (The old GPU compute probe that baked these in was removed for exactly that reason.)
// kMaxLights is only the INITIAL light-buffer/atlas capacity (both grow on demand — NO hard cap on lights);
// kLightsPerRow factors ONLY into the initial/fallback atlas-size math below — NOT a runtime packing
// stride (the runtime packer targets kAtlasBlocksWide); the k*Res/k*Atlas* here are only initial sizing.
// ============================================================================
namespace vsm
{
	// ---- Shadow atlas layout ----
	// Every active light gets a 3x2 cube-face "block"; PackAtlas shelf-packs the variable-size
	// blocks and the atlas texture grows on demand (NOT a fixed grid). Per-face resolution is
	// variable (AssignFaceRes ladder) and lights are uncapped — both already implemented; the
	// constants below are only compile-time fallback / initial sizing.
	inline constexpr int kFaceRes       = 256;             // one cube face, in texels
	inline constexpr int kMaxLights     = 32;             // INITIAL light-buffer + atlas capacity (BOTH grow on demand via EnsureLightBuffer/PackAtlas — NOT a hard cap on shadow lights)
	inline constexpr int kLightsPerRow  = 4;              // ONLY factors into the fallback kAtlasW/kAtlasH below (runtime packer uses kAtlasBlocksWide)
	inline constexpr int kCubeFacesWide = 3;              // a light's cube-face block is 3 faces wide (+X -X +Y)...
	inline constexpr int kCubeFacesTall = 2;              // ...and 2 faces tall (-Y +Z -Z) — one 3x2 block per light
	inline constexpr int kBlockW        = kFaceRes * kCubeFacesWide;   // 768: three faces wide
	inline constexpr int kBlockH        = kFaceRes * kCubeFacesTall;   // 512: two faces tall
	inline constexpr int kAtlasW       = kLightsPerRow * kBlockW;                                       // 3072
	inline constexpr int kAtlasH       = ((kMaxLights + kLightsPerRow - 1) / kLightsPerRow) * kBlockH;  // 4096

	// ---- Variable per-light resolution (Step 1; notes/VSM_VARIABLE_RESOLUTION.md) ----
	// Each active light's cube-face resolution is chosen PER FRAME on a power-of-two ladder from
	// kFloorFaceRes up to the user's iShadowMapResolution. The TOP rung is the exact ini value (may be
	// non-pow2); every rung below is a power of two. Detail follows on-screen importance: a big/near light
	// gets a high rung, a small/distant one drops to the floor — so many lights stay affordable in VRAM.
	inline constexpr int   kFloorFaceRes    = 128;    // smallest per-light cube-face resolution = the atlas page/quantum
	// Hard D3D11 2D-texture dimension limit (D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION). The packed atlas must never
	// exceed this in either axis or CreateTexture2D fails and ALL shadows vanish; PackAtlas clamps to it and drops
	// the least-important overflow lights (they render unshadowed — fail-safe) rather than growing past it.
	inline constexpr int   kMaxAtlasDim     = 16384;
	// Ceiling on a single light's cube-face resolution rung. A light's block is kCubeFacesWide (3) faces across,
	// so the top rung MUST satisfy rung*3 <= kMaxAtlasDim or even ONE light can't fit -> = kMaxAtlasDim/3 = 5461.
	// (Was 8192, which makes a 24576-wide block that overflows the texture limit; see ComputeAtlasDims.)
	inline constexpr int   kMaxFaceResCeiling = kMaxAtlasDim / kCubeFacesWide;  // 5461
	inline constexpr float kLevelHysteresis = 0.25f;  // hold a light's rung until desired res moves >25% (anti-flicker)
	inline constexpr float kTanHalfFovV     = 0.577f; // ~60deg vertical-FOV proxy for the screen-size metric (approx; k absorbs error)
	inline constexpr int   kAtlasBlocksWide = 4;      // packer target width = this many MAX-size light blocks across
	inline constexpr int   kMaxPCFRadius    = 4;      // clamp for the runtime soft-shadow PCF half-width (from iBlurDeferredShadowMask); caps per-pixel tap cost

	// ---- Fixed capacities / cadence ----
	inline constexpr std::uint32_t kRebuildInterval    = 30;    // frames between scene-graph re-traversals
	inline constexpr std::uint32_t kConsumerGraceFrames = 60;   // N3: skip the atlas render after this many frames with no LLF fetch (no consumer -> no point rendering); generous so startup/hitches never wrongly skip
	inline constexpr std::uint64_t kSlotEvictFrames    = 90;    // free a light's persistent atlas slot after this many frames unseen (LRU)
	inline constexpr float         kLightMoveEps       = 0.5f;  // world-units a collected light must move to invalidate the P2 static cache
	// Moving-light shadow smoothing (aesthetics). Swinging lanterns/torches update their position in
	// discrete Havok/animation steps; rendering each step faithfully looks jerky. Low-pass the cube
	// CENTER we generate + sample from toward the light's true position so the shadow motion is smooth.
	// The shadow lags the light slightly; the bright hotspot (driven by the game's own light position,
	// not smoothed) has soft falloff that hides the small desync.
	inline constexpr float         kLightSmoothAlpha    = 0.30f;  // per-frame lerp toward the true position (~tuned @60fps). 1 = off/snap; lower = smoother + more lag.
	inline constexpr float         kLightSmoothSnapDist = 160.0f; // world-units: a per-frame jump larger than this is a TELEPORT (cell load / reposition) -> snap, don't glide across the room.
	inline constexpr std::uint64_t kLightGraceFrames   = 30;    // keep a light shadowed this many frames after it last appeared (anti-flicker debounce)
	inline constexpr int           kMaxRebakesPerFrame = 2;     // P4: max MOVED lights whose static cache re-bakes per frame (new lights bypass this) — amortizes re-bake cost
	inline constexpr float         kTranslucentCoverage = 0.5f; // A5: default per-glass opacity when the material alpha can't be read (0 = clear .. 1 = opaque tint)
	inline constexpr std::size_t   kMaxCasters         = 8192;  // registry hard cap (huge-cell / runaway guard)
	inline constexpr int           kMaxSceneGraphDepth = 64;    // recursion-depth guard for RebuildRegistry's scene-graph traversal (cycle guard)

	// ---- Skinned detached-shadow guard ----
	inline constexpr float kDetachMargin = 120.0f;  // skinned detached-shadow guard: slack over the mesh radius

	// ---- Shadow near/far planes ----
	// far = the light's radius (computed in CollectLights). near = far * kNearPlaneFraction, floored at
	// kNearPlaneEpsilon world units for depth-projection stability. Both were a runtime 'NearFrac' slider;
	// now fixed — reverse-Z tolerates a small near plane without the precision loss that motivated tuning it,
	// so there is nothing for a user to adjust.
	inline constexpr float kNearPlaneFraction = 0.01f;   // near = far * this
	inline constexpr float kNearPlaneEpsilon  = 0.1f;    // absolute world-unit floor on the near plane


	// ---- CS <-> plugin resource-binding CONTRACT ----
	// Community Shaders' LightLimitFix::Prepass fetches our resources (see VSM_GetShadowResources in
	// Plugin.cpp) and binds them into these shader slots so the game's lighting shader can sample our
	// shadows. These slot numbers are the agreed interface with LLF — they MUST match what LLF binds.
	// RenderDepth nulls kShadowAtlasSRVSlot before writing the atlas as its DSV (same resource -> read/write
	// hazard); the other two are listed for completeness / documentation of the contract.
	inline constexpr unsigned int kShadowAtlasSRVSlot = 110;  // t110: our shadow-atlas depth SRV
	inline constexpr unsigned int kLightBufferSRVSlot = 111;  // t111: our per-light LightRecord buffer SRV
	inline constexpr unsigned int kShadowSamplerSlot  = 7;    // s7  : our atlas sampler

	// ---- CS <-> plugin ABI handshake (VSM_GetABIVersion) ----
	// VSM and the forked Community Shaders share three binary layouts with NO runtime check: the 32-byte t111
	// LightRecord, the 32-byte b13 VSMParams cbuffer, and CS's Light.vsmShadowIndex field/slot contract. A
	// mismatched plugin/CS pair silently renders WRONG shadows. The plugin exports this id (VSM_GetABIVersion);
	// a VSM-aware CS compares it to the value it was built against and refuses to bind our resources on mismatch
	// (falling back to no VSM shadows — safe) instead of trusting a drifted layout. BUMP whenever ANY of those
	// shared layouts changes. High byte / low bytes are just a readable major.minor; the value is opaque to CS.
	inline constexpr std::uint32_t kABIVersion = 0x00010000u;  // v1.0 — 32B record + 32B b13 + vsmShadowIndex@Light
}
