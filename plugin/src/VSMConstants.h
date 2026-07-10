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
// kLightsPerRow is the atlas grid stride; the k*Res/k*Atlas* here are only the initial sizing.
// ============================================================================
namespace vsm
{
	// ---- Shadow atlas layout ----
	// Every active light gets a 3x2 cube-face "block"; blocks tile the atlas in a grid.
	// Correctness-first: modest per-face resolution + a light cap. Both scale up once the
	// culling/paging work (Phase 2) lands.
	inline constexpr int kFaceRes      = 256;              // one cube face, in texels
	inline constexpr int kMaxLights    = 32;              // INITIAL light-buffer + atlas capacity (BOTH grow on demand via EnsureLightBuffer/PackAtlas — NOT a hard cap on shadow lights)
	inline constexpr int kLightsPerRow = 4;               // light-blocks per atlas row
	inline constexpr int kBlockW       = kFaceRes * 3;    // 768: three faces (+X -X +Y) wide
	inline constexpr int kBlockH       = kFaceRes * 2;    // 512: two faces (-Y +Z -Z) tall
	inline constexpr int kAtlasW       = kLightsPerRow * kBlockW;                                       // 3072
	inline constexpr int kAtlasH       = ((kMaxLights + kLightsPerRow - 1) / kLightsPerRow) * kBlockH;  // 4096

	// ---- Variable per-light resolution (Step 1; notes/VSM_VARIABLE_RESOLUTION.md) ----
	// Each active light's cube-face resolution is chosen PER FRAME on a power-of-two ladder from
	// kFloorFaceRes up to the user's iShadowMapResolution. The TOP rung is the exact ini value (may be
	// non-pow2); every rung below is a power of two. Detail follows on-screen importance: a big/near light
	// gets a high rung, a small/distant one drops to the floor — so many lights stay affordable in VRAM.
	inline constexpr int   kFloorFaceRes    = 128;    // smallest per-light cube-face resolution = the atlas page/quantum
	inline constexpr float kQualityFactor   = 1.0f;   // k: target shadow texels per screen pixel (menu-configurable later)
	inline constexpr float kLevelHysteresis = 0.25f;  // hold a light's rung until desired res moves >25% (anti-flicker)
	inline constexpr float kTanHalfFovV     = 0.577f; // ~60deg vertical-FOV proxy for the screen-size metric (approx; k absorbs error)
	inline constexpr int   kAtlasBlocksWide = 4;      // packer target width = this many MAX-size light blocks across
	inline constexpr int   kMaxPCFRadius    = 4;      // clamp for the runtime soft-shadow PCF half-width (from iBlurDeferredShadowMask); caps per-pixel tap cost

	// ---- Debug preview linearizer near/far ----
	// Used ONLY by the grayscale atlas-preview shader, to map perspective depth to a
	// readable gradient. These are NOT the per-light shadow planes (those come from each
	// light's radius in CollectLights).
	inline constexpr float kPreviewNear = 8.0f;
	inline constexpr float kPreviewFar  = 8192.0f;

	// ---- Fixed capacities / cadence ----
	inline constexpr std::uint32_t kRebuildInterval    = 30;    // frames between scene-graph re-traversals
	inline constexpr std::uint64_t kSlotEvictFrames    = 90;    // free a light's persistent atlas slot after this many frames unseen (LRU)
	inline constexpr float         kLightMoveEps       = 0.5f;  // world-units a collected light must move to invalidate the P2 static cache
	inline constexpr std::uint64_t kLightGraceFrames   = 30;    // keep a light shadowed this many frames after it last appeared (anti-flicker debounce)
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

	// ---- Debug-overlay-only params ----
	// The b13 gBiasWorld / gMatchThresh cbuffer fields feed ONLY diagnostic overlay modes 15/16. The REAL
	// shadow path uses a calculated receiver-plane bias and a threshold-free nearest-light match, so these
	// are no longer user knobs — they're fixed values the debug viz reads.
	inline constexpr float kDebugModeBias        = 3.0f;  // mode 15 (linPix-linOcc > this) viz
	inline constexpr float kDebugModeMatchThresh = 5.0f;  // mode 16 (nearest-light dist < this) viz

	// (Former kMaxShadowMagnification / spatial fixture cull removed; billboards are now excluded at registry build instead.)

	// ---- CS <-> plugin resource-binding CONTRACT ----
	// Community Shaders' LightLimitFix::Prepass fetches our resources (see VSM_GetShadowResources in
	// Plugin.cpp) and binds them into these shader slots so the game's lighting shader can sample our
	// shadows. These slot numbers are the agreed interface with LLF — they MUST match what LLF binds.
	// RenderDepth nulls kShadowAtlasSRVSlot before writing the atlas as its DSV (same resource -> read/write
	// hazard); the other two are listed for completeness / documentation of the contract.
	inline constexpr unsigned int kShadowAtlasSRVSlot = 110;  // t110: our shadow-atlas depth SRV
	inline constexpr unsigned int kLightBufferSRVSlot = 111;  // t111: our per-light LightRecord buffer SRV
	inline constexpr unsigned int kShadowSamplerSlot  = 7;    // s7  : our atlas sampler
}
