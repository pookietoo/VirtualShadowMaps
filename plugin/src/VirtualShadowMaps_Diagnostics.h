#pragma once

// ============================================================================
// VsmDiagnostics — the diagnostics apparatus extracted off the VirtualShadowMaps
// god-object. All the PURE-diagnostic work lives here: the full numeric shadow-math
// dump, the Path-B skinned-geometry probe, the scene-graph census, caster inspection,
// the world-triangle geometry sidecar dump, and the debug depth-preview resolve.
//
// A VsmDiagnostics holds a reference to its owning VirtualShadowMaps (m_core) and
// reaches the render-side state through it. VirtualShadowMaps declares
// `friend class VsmDiagnostics;`, so these methods touch its private members, and
// owns a `VsmDiagnostics diag{ *this };` member. The method bodies are defined in
// VirtualShadowMaps_Diagnostics.cpp (the same translation unit as before); each one
// opens with a block of `auto& x = m_core.x;` aliases so the (unchanged) body reads
// the core state through m_core with byte-identical behaviour.
//
// NOTE: the render-path helpers ComputeVertAABB, GetFreshWorldAABB, NodeOwningRef are
// VirtualShadowMaps members DEFINED in the core unit (VirtualShadowMaps.cpp) — they are
// called every frame by the render path, which must exist in a deploy build where this
// diagnostics unit is dropped — so they are NOT declared here.
// ============================================================================

#include <DirectXMath.h>

#include <cstdint>
#include <string>

// RE::FormID resolves via the force-included PCH (same as VirtualShadowMaps.h).
namespace RE
{
	class NiAVObject;
	class BSGeometry;
}

class VirtualShadowMaps;

// Internal diagnostics helper owned by VirtualShadowMaps. Members/methods are public
// by design: it is reached only through VirtualShadowMaps::diag, and its owner
// (DrawMenu) reads a few of these state fields directly (diag.dbgHaveDump, etc.).
class VsmDiagnostics
{
public:
	explicit VsmDiagnostics(VirtualShadowMaps& a_core) : m_core(a_core) {}

	// --- pure-diagnostic methods (moved off VirtualShadowMaps) ---
	void        DumpDiagnosticLog();    // full numeric shadow-math dump (+ atlas/id/geom sidecars)
	void        DumpSkinnedGeometry();  // Path B probe: what skinned-mesh buffers/data exist at Present
	void        DumpGeometry();         // every caster's REAL world-space triangles -> geom.bin sidecar
	void        DumpFocusedCaster(int a_idx);  // full render+projection detail for one caster (flashed/isolated)
	void        DumpSceneCensus(RE::NiAVObject* a_root, const char* a_label);  // SEH-guarded scene-graph census
	void        DumpPlayerDiag(RE::NiAVObject* a_p3d);                          // player geometry + registry membership
	void        PlayerWalk(RE::NiAVObject* a_o, int a_depth, int& a_tris, int& a_inReg, int& a_other);  // recursive helper
	void        InspectCaster();        // read back a caster's raw vertex/index bytes for diagnostics
	void        ComputeCasterBounds();  // AABB of collected caster world positions (debug)
	void        ResolvePreview();       // linearize the depth map to a grayscale debug texture
	std::string RefIdentity(RE::FormID a_refID);  // dump-only: full identity + relationship of a TESObjectREFR

	// --- diagnostic state (used only by the methods above + the menu) ---
	// vertex/index readback of one caster (InspectCaster -> menu)
	bool              dbgHaveDump   = false;
	std::uint32_t     dbgStride     = 0;
	std::uint32_t     dbgIndexCount = 0;
	bool              dbgFullPrec   = false;
	DirectX::XMFLOAT3 dbgV[4]       = {};  // first 4 decoded local vertex positions (float3)
	std::uint32_t     dbgIdx[6]     = {};  // first 6 indices (16-bit, widened)

	// AABB of collected caster world positions (ComputeCasterBounds -> menu)
	DirectX::XMFLOAT3 dbgCasterMin{};
	DirectX::XMFLOAT3 dbgCasterMax{};

	// preview normalization range: world units mapped white->black (menu slider -> ResolvePreview)
	float previewRange = 2000.0f;

	VirtualShadowMaps& m_core;  // the god-object this diagnostics apparatus reports on
};
