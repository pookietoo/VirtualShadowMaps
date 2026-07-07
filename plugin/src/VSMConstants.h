#pragma once

#include <cstddef>
#include <cstdint>

// ============================================================================
// Single source of truth for the shadow-atlas geometry and fixed capacities.
//
// IMPORTANT: the HLSL side (Shaders/VirtualShadowMaps/VSM.hlsli, which ships as a
// Community Shaders override) keeps a hand-mirrored copy of the atlas constants
// that MUST stay equal to these. The plugin's OWN embedded shaders (the depth
// preview linearizer and the GPU shadow-math probe) are generated from these
// values at runtime (see MakeAtlasConstantsHLSL in VirtualShadowMaps.cpp), so
// they can never drift.
// ============================================================================
namespace vsm
{
	// ---- Shadow atlas layout ----
	// Every active light gets a 3x2 cube-face "block"; blocks tile the atlas in a grid.
	// Correctness-first: modest per-face resolution + a light cap. Both scale up once the
	// culling/paging work (Phase 2) lands.
	inline constexpr int kFaceRes      = 256;              // one cube face, in texels
	inline constexpr int kMaxLights    = 32;              // atlas capacity, in light-blocks
	inline constexpr int kLightsPerRow = 4;               // light-blocks per atlas row
	inline constexpr int kBlockW       = kFaceRes * 3;    // 768: three faces (+X -X +Y) wide
	inline constexpr int kBlockH       = kFaceRes * 2;    // 512: two faces (-Y +Z -Z) tall
	inline constexpr int kAtlasW       = kLightsPerRow * kBlockW;                                       // 3072
	inline constexpr int kAtlasH       = ((kMaxLights + kLightsPerRow - 1) / kLightsPerRow) * kBlockH;  // 4096

	// ---- Debug preview linearizer near/far ----
	// Used ONLY by the grayscale atlas-preview shader, to map perspective depth to a
	// readable gradient. These are NOT the per-light shadow planes (those come from each
	// light's radius in CollectLights).
	inline constexpr float kPreviewNear = 8.0f;
	inline constexpr float kPreviewFar  = 8192.0f;

	// ---- Fixed capacities / cadence ----
	inline constexpr std::uint32_t kRebuildInterval = 30;    // frames between scene-graph re-traversals
	inline constexpr std::size_t   kMaxCasters      = 8192;  // registry hard cap (huge-cell / runaway guard)
}
