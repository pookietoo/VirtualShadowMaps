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
	    "0.9.82 PERFORMANCE modules P1/P2/P3 (each an INDEPENDENT toggle; validate one at a time). Also Step 3 (SkyrimPrefs config: fShadowBiasScale->bias, iBlurDeferredShadowMask->runtime PCF radius, fShadowDistance; k menu slider) + Step 4 (diagnostics + tools/shadow_truth.py now variable-res-correct) landed. P1 (always on): caster STATIC/DYNAMIC classification (base Static/StaticCollection & not-skinned & no-anim-controller = cacheable) + STABLE per-light atlas slots (persistent; replaces the every-frame repack) -> shader UNCHANGED. Also FIXED a latent Step-1 bug: RenderDepth viewport used rtFaceRes (max) not the light's faceRes (corrupted the atlas for variable-res lights; worked only when all lights were near/max). P2 cacheStaticShadows (default OFF): static-cache depth texture + one whole-atlas CopyResource + dynamic overlay -> LLF samples the same single t110 atlas, shader UNCHANGED, no CS-fork change. COARSE invalidation (whole cache re-bakes on light move/appear or cell change; holds when stationary). P3 cullCasters (default OFF): drop lights behind the camera / beyond fShadowDistance (conservative broad-phase). SHADER CHANGED since 0.9.81? NO new shader change vs 0.9.81's PCF, but the 0.9.81 PCF+per-light shader is required -> FORCE CS RECOMPILE if coming from <=0.9.80. VALIDATE: baseline (all toggles off) == 0.9.81; then flip Cache static shadows -> shadows same but re-bake log rare when still; then flip Cull off-screen lights -> off-screen lights lose shadows, on-screen unchanged. P4 (per-light invalidation + scheduler) DEFERRED to a testable session (delicate caching-core refactor). Snapshot branch wip/vsm-variable-res-perf-base.";

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
