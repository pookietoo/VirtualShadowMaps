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
	    "0.9.81 VARIABLE per-light RESOLUTION + SOFT SHADOWS (the coupled milestone). Each active light's cube-face resolution is now chosen per-frame on a power-of-two ladder from a 128-texel floor up to iShadowMapResolution (top rung = the exact ini value; pow2 rungs below), driven by the light's ON-SCREEN size (radius/dist), snapped UP, with 25% hysteresis to stop rung flicker. Variable-size 3x2 blocks are shelf-packed into a grow-only atlas (per-light atlasX/atlasY/faceRes descriptor from 0.9.78); BuildCubeMatrices takes per-light faceRes for the guard-band FOV. SOFT SHADOWS: GetLocalShadow now does 5x5 PCF — a FIXED texel kernel x the variable faceRes = ~constant SCREEN-space penumbra, so distant lights are soft+cheap and near lights sharp+detailed. Also SHRINKS atlas VRAM vs uniform-max (most lights drop to the floor). SHADER CHANGED (PCF + per-light addressing) -> FORCE CS RECOMPILE. KNOWN-INCOMPLETE (do not trust for these): diagnostics/dump per-face scans + tools/shadow_truth.py still assume one uniform rtFaceRes -> WRONG for variable-res lights (gameplay is correct; the offline verifier is not, pending Step 4). Config still constants (k=1, floor=128, PCF radius=2; SkyrimPrefs/menu wiring = Step 3). Allocator is a shelf packer, not the buddy/quadtree (per-light 3:2 block isn't square; swappable later w/o shader change). PCSS contact-hardening deferred. Prior 0.9.79 = cap removal (kept).";

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
