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
	    "0.9.35 per-light jitter detector: BSLight::dynamic + per-frame position delta (find the moving shadow light)";

	// Skyrim's main view is reverse-Z. Flip if the dumped map looks inverted; drives the depth
	// clear value + DepthFunc (single source of truth for the atlas-generation pass).
	inline constexpr bool kReverseZ = false;

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

	// ---- GPU shadow-math probe I/O (see VirtualShadowMaps.h). One thread per probe point runs the
	// EXACT VSM::GetLocalShadow logic against the live atlas + light buffer and writes every
	// intermediate out. These structs are sized in SetupResources and filled/read in the dump; they
	// MUST stay byte-identical to the kProbeCS HLSL structs so the probe proves the real path. ----
	struct ProbeIn   // must match kProbeCS ProbeIn
	{
		DirectX::XMFLOAT4 P;           // pixel/surface position, camera-relative (== shader input.WorldPosition)
		DirectX::XMFLOAT4 lightPosWS;  // the light as LLF passes it (camera-relative to the FrameBuffer eye)
	};
	struct ProbeOut  // must match kProbeCS ProbeOut
	{
		DirectX::XMFLOAT4 W;           // xyz = sample-space position; w = sampleSpace used
		DirectX::XMFLOAT4 absLight;    // xyz = match key (lightPosWS + matchEye); w = nearest buffer-light distance
		DirectX::XMFLOAT4 ndc;         // xyz = ndc; w = clip.w
		DirectX::XMFLOAT4 uv_occ;      // xy = atlasUV; z = occluder sampled; w = matched light index (-1 none)
		DirectX::XMFLOAT4 result;      // x = linPix; y = linOcc; z = diff; w = shadowed? (1=shadow)
		DirectX::XMFLOAT4 face_flags;  // x = face; y = inBounds; z = inFront; w = matched?(1/0)
	};
	struct alignas(16) ProbeCBData  // must match kProbeCS cbuffer ProbeCB
	{
		DirectX::XMFLOAT4 camAdjust;   // xyz used as FrameBuffer::CameraPosAdjust (we feed altEye — exact in 3rd person)
		DirectX::XMFLOAT4 altEye;      // xyz render eye (posAdjust.getEye)
		float    bias;        int compareMode; int matchEye; int sampleSpace;
		float    matchThresh; int numProbes;   int pad0;     int pad1;
	};

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
