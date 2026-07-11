#pragma once

// ============================================================================
// Internal helpers shared between the core render path (VirtualShadowMaps.cpp)
// and the diagnostics translation unit (VirtualShadowMaps_Diagnostics.cpp).
// These are implementation details, not part of the public class in
// VirtualShadowMaps.h. Relies on PCH.h (force-included) for RE:: types + ComPtr.
// ============================================================================

#include <DirectXMath.h>
#include <d3d11.h>

namespace vsm::internal
{
	// Human-readable build label shown in the menu + diagnostic dump, so it's obvious which
	// plugin DLL is actually running. Bump the description each build. (The SKSE plugin version
	// number is separate — see Plugin.h / CMake project VERSION.)
	inline constexpr char kBuildTag[] =
	    "0.9.94 PERF modules (both code-complete, compile clean, UNTESTED in-game): O4 cullEmptyLightPasses (config "
	    "toggle, default OFF) — skip a light's whole 6-face caster pass when no caster lies within its radius; "
	    "O9 lazy transmittance atlas — allocate the A5 RGBA atlas only when translucentShadows is on "
	    "(EnsureTransmittanceAtlas creates it on-demand incl. the runtime A5 toggle so t112 is never unbound; freed on "
	    "A5-off), saving ~1 atlas of VRAM in the default case. No behaviour change with the toggles off. Prior "
	    "0.9.93 FIX skinned-head explosion, ROOT CAUSE FOUND + validated (supersedes the 0.9.91/0.9.92 index theories). "
	    "TWO real bugs, both now fixed: (1) SPACE MISMATCH — for a BSDynamicTriShape, dynamicData AND every skin partition's "
	    "buffData are ONE shared, shape-GLOBAL vertex buffer (dataSize/16 == sp->vertexCount == UNIQUE count; partitions SHARE "
	    "seam verts so sum p.vertices > unique; PROVEN by dump: p0.buffData[0:N] == p1.buffData[0:N]). We read POSITION at the "
	    "global index vertexMap[lv] but WEIGHTS/bone-slots at partition-LOCAL lv -> for a non-identity map (partitions 1+) that "
	    "paired a vertex's position with a DIFFERENT vertex's bones -> scrambled skin, 100% correlated with numPartitions>1"
	    "(identity map hid it on partition 0). FIX: read BOTH position and weights at the SAME global index gv=vertexMap[lv]. "
	    "(2) RACE — we read dynamicData WITHOUT its BSSpinLock while the engine's async facegen/morph writer updates it; live "
	    "reads got torn positions (offline dumps read settled data, so they always looked coherent -> the offline-vs-live "
	    "discrepancy). FIX: take dr.lock around the read in SkinAllCasters. VALIDATION (full scene dump, offline): our bone "
	    "palette == boneWorldTransforms o skinToBone for 63/88 meshes; our posed bound matches the engine's worldBound for "
	    "EVERY dynamic mesh (0/43 exceed) incl. MaleHeadChild 13.1u vs engine 11.9u. Diagnostics added this saga: skin.bin "
	    "(full per-caster skinning) + scene.bin (the ENTIRE scene graph: every node + geometry buffers/props/skin). UNTESTED "
	    "in-game (user could not test). Prior "
	    "0.9.92 FIX multi-partition head explosion (follow-on to 0.9.91) — 0.9.91 fixed SINGLE-partition dynamic meshes (hair, "
	    "mouth) but MULTI-partition heads (MaleHeadChild n=2, adult FemaleHeadNord n=3) still scattered; explosion correlated "
	    "EXACTLY with numPartitions>1. ROOT: dynamicData is CONCATENATED in partition/DRAW order (dataSize/16 == sum of p.vertices, "
	    "seam verts duplicated), NOT shape-vertex order — so indexing it by vertexMap[lv] (shape order) misindexes once >1 "
	    "partition; it coincides only for a single partition (vertexMap identity), which is why single-partition worked. FIX: index "
	    "the bind position by a RUNNING draw-order counter (per-partition base + lv), vertexMap drops out of the dynamic path. "
	    "UNTESTED — verify: re-dump geom.bin, the n=2/n=3 heads' posed extent should collapse to head-size (~15u) like the n=1 hair did. "
	    "Prior 0.9.91 FIX exploding head/hair shadows (BSDynamicTriShape) — ROOT CAUSE: dynamicData is CPU-MORPHED but PRE-SKINNING "
	    "MODEL space, NOT already-skinned. The old code multiplied it by geom->world, but a skinned mesh is placed by the BONE "
	    "PALETTE, not the node transform -> adult heads ~10x oversized, CHILD heads scattered ~2380u (child skeleton scale lives "
	    "in the bones we skipped; MaleHeadChild cast 20.8% of the atlas). FIX: VSM_PosedDynamicCaster now SKINS dynamicData through "
	    "the same bone palette as the body path (VSM_SkinOneCaster) — bind position from dynamicData[vertexMap[lv]] (shape-global), "
	    "weights/bone-slots from the partition stream at local lv (VA_SKINNING only; position is a split dynamic stream), output "
	    "world-absolute, NO geom->world. Grounded in CommonLibSSE NiSkinInstance + SSE GPU-skin architecture + NifSkope. UNTESTED — "
	    "verify: re-dump geom.bin, MaleHeadChild posed extent should collapse to head-size (~15u) not ~2380u. "
	    "Prior 0.9.89 AUTO engine 'don't cast' flag filter — mirror the shader/object flags the vanilla shadow pass OMITS so we stop "
	    "casting shadows the game itself never casts: kNotVisible (NiAVObject; prime suspect for the reported invisible-mesh-casting-"
	    "on-characters), kNonProjectiveShadows, kBillboard shader flag, kWireframe, kLODObjects/kLODLandscape/kHDLODObjects, "
	    "kProjectedUV + kWeaponBlood (folded into the decal reject), and BSWaterShaderProperty (water). Dump logs notVisible / "
	    "engineFlag reject counts. UNTESTED. "
	    "Prior 0.9.88 config [classification] pattern override — user forceCast/forceNoCast string lists in VirtualShadowMaps.toml let a "
	    "user force specific meshes to cast (beating every built-in exclusion) or never cast, matched per-shape against the model "
	    "NIF path (substring, e.g. '\\effects\\') and shape name (whole-token: 'marker' hits 'EditorMarker' not 'Market'). forceCast "
	    "wins conflicts. Empty lists (default) = 0.9.87 behavior exactly. Dump logs forceNoCast-rejected/forceCast-forced counts. "
	    "UNTESTED — user deferred testing (backlog). Research-grounded (NIF kCastShadows is the engine's own per-mesh cast bit; we "
	    "already gate on it).\n"
	    "Prior 0.9.87 FIX black local lights (regression surfaced testing 0.9.86: 'the entire light around the fire is a shadow'). ROOT: the A5 shader sampled the t112 transmittance atlas and MULTIPLIED every local light by it UNCONDITIONALLY — even with A5 off — so when t112 isn't bound to our (white) atlas (i.e. the running CommunityShaders.dll doesn't bind t112), it reads BLACK and zeroes the light. FIX: gate the t112 sample on a new gTranslucentOn flag (b13 row1.x, repurposes the dead gMatchThresh -> NO cbuffer-size change, NO CS rebuild needed) that the plugin sets = 1 ONLY when config.translucentShadows is on. With A5 OFF (default) the transmittance stays white and local lights NEVER touch t112 -> baseline is robust regardless of which CommunityShaders.dll is loaded. A5 ON still REQUIRES the matching CS build that binds t112. Also: module toggle states are now logged on settings change. The offline verifier's synthShadow self-tests confirm the depth/shadow math + the 0.9.86 refactor are sound. TEST: with A5 OFF the fire (and all local lights) should read correctly again; A1 fade + P2/P3/P4 unchanged.";

	// Depth convention for the shadow atlas. TRUE = reverse-Z (near->1, far->0), which is Skyrim's own
	// main-view convention and gives uniform float-depth precision across the light's range. This flag is the
	// SINGLE SOURCE OF TRUTH for the atlas-GENERATION side and now drives ALL of it consistently:
	//   - the cube PROJECTION (BuildCubeMatrices: post-multiplies a z-flip when set),
	//   - the depth CLEAR value (0.0 = far),
	//   - the depth-test DepthFunc (GREATER = keep nearest).
	// The atlas CONSUMERS must mirror this same convention by hand (there is no shared compile unit):
	//   - the CS-side sampler  Shaders/VirtualShadowMaps/VSM.hlsli :: LinearizeCubeDepth,
	//   - the offline verifier tools/shadow_truth.py :: _lin.
	// If you ever flip this, flip those two to match (both are commented pointing back here).
	inline constexpr bool kReverseZ = true;

	// NiTransform (column-vector R*s + t) -> DirectXMath row-vector (v*M) matrix.
	inline DirectX::XMMATRIX NiTransformToXM(const RE::NiTransform& t)
	{
		const auto& e = t.rotate.entry;  // float entry[3][3]
		const float s = t.scale;
		return DirectX::XMMatrixSet(
		    s * e[0][0], s * e[1][0], s * e[2][0], 0.0f,
		    s * e[0][1], s * e[1][1], s * e[2][1], 0.0f,
		    s * e[0][2], s * e[1][2], s * e[2][2], 0.0f,
		    t.translate.x, t.translate.y, t.translate.z, 1.0f);
	}

	// Save/restore the graphics-stage state our draw/preview passes disturb.
	struct GraphicsStateGuard
	{
		ID3D11DeviceContext* ctx;
		ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]{};
		ID3D11DepthStencilView* dsv = nullptr;
		D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{};
		UINT numViewports = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
		ComPtr<ID3D11RasterizerState> rs;
		ComPtr<ID3D11DepthStencilState> ds;
		UINT stencilRef = 0;
		ComPtr<ID3D11InputLayout> il;
		D3D11_PRIMITIVE_TOPOLOGY topo{};
		ComPtr<ID3D11VertexShader> vs;
		ComPtr<ID3D11PixelShader> ps;
		ID3D11Buffer* vsCB0 = nullptr;

		explicit GraphicsStateGuard(ID3D11DeviceContext* c) : ctx(c)
		{
			ctx->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, &dsv);
			ctx->RSGetViewports(&numViewports, viewports);
			ctx->RSGetState(&rs);
			ctx->OMGetDepthStencilState(&ds, &stencilRef);
			ctx->IAGetInputLayout(&il);
			ctx->IAGetPrimitiveTopology(&topo);
			ctx->VSGetShader(&vs, nullptr, nullptr);
			ctx->PSGetShader(&ps, nullptr, nullptr);
			ctx->VSGetConstantBuffers(0, 1, &vsCB0);
		}
		~GraphicsStateGuard()
		{
			ctx->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, dsv);
			for (auto* rtv : rtvs)
				if (rtv) rtv->Release();
			if (dsv) dsv->Release();
			ctx->RSSetViewports(numViewports, viewports);
			ctx->RSSetState(rs.Get());
			ctx->OMSetDepthStencilState(ds.Get(), stencilRef);
			ctx->IASetInputLayout(il.Get());
			ctx->IASetPrimitiveTopology(topo);
			ctx->VSSetShader(vs.Get(), nullptr, 0);
			ctx->PSSetShader(ps.Get(), nullptr, 0);
			ctx->VSSetConstantBuffers(0, 1, &vsCB0);
			if (vsCB0) vsCB0->Release();
		}
	};

	// (Former ProbeIn / ProbeOut / ProbeCBData compute-probe I/O structs removed; superseded by the pixel probe below.)

	// Real-shader pixel probe readback (MUST match VSM.hlsli::VSMPixelProbe — 10 float4 = 160 bytes).
	struct PixelProbe
	{
		DirectX::XMFLOAT4 pixel;       // xy = SV_Position, z = written(1 => shader wrote this), w = mode
		DirectX::XMFLOAT4 P;           // input.WorldPosition (camera-relative) as the REAL shader received it
		DirectX::XMFLOAT4 camAdjust;   // REAL FrameBuffer::CameraPosAdjust.xyz — resolves the last assumption
		DirectX::XMFLOAT4 W;           // SampleP(P).xyz, w = sampleSpace
		DirectX::XMFLOAT4 lightPosWS;  // light as LLF passed it
		DirectX::XMFLOAT4 absLight;    // match key
		DirectX::XMFLOAT4 matched;     // x = matched index, y = face, z = inBounds, w = inFront
		DirectX::XMFLOAT4 ndc;         // xyz = ndc, w = clip.w
		DirectX::XMFLOAT4 uv_occ;      // xy = atlasUV, z = occluder, w = reason(0 eval,1 mode,2 no-match,3 behind,4 oob)
		DirectX::XMFLOAT4 result;      // x = linPix, y = linOcc, z = diff, w = shadow (0 = shadowed)
	};
}
