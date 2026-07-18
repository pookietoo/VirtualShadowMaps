#pragma once

// ============================================================================
// Internal helpers used by the core render path (VirtualShadowMaps.cpp).
// These are implementation details, not part of the public class in
// VirtualShadowMaps.h. Relies on PCH.h (force-included) for RE:: types + ComPtr.
// ============================================================================

#include <DirectXMath.h>
#include <d3d11.h>

namespace vsm::internal
{
	// Human-readable build label shown in the (dev) menu, so it's obvious which plugin DLL is
	// actually running. Bump the description each build. (The SKSE plugin version number is
	// separate — see Plugin.h / CMake project VERSION.)
	inline constexpr char kBuildTag[] =
	    "0.9.103 - A5 O(lights²)->O(1) BSLight* light dedup on the 0.9.102 compatibility-hardening base; per-phase profiler kept dev-only.";

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

	// Save/restore the graphics-stage state our draw/preview passes disturb. Because our passes run at Present
	// time (before the game's original Present chain — where ENB does its post-processing), any state we leave
	// dirty is inherited by ENB and the next engine frame; this guard must therefore cover EVERY stage the
	// passes touch. The set below matches what RenderDepth / RenderCasterPass / the A6 alpha path / the A5
	// translucent path actually mutate: OM (RTV/DSV, blend, depth-stencil), RS (state, viewports), IA (layout,
	// topology, vertex/index buffers), VS (shader, CB0) and PS (shader, sampler s0, SRV t0, CB0).
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
		// Added coverage (previously leaked into ENB / the next frame):
		ID3D11BlendState* blend = nullptr;
		FLOAT             blendFactor[4]{};
		UINT              sampleMask = 0xFFFFFFFFu;
		ID3D11Buffer*     vb0 = nullptr;                 // IA vertex buffer slot 0
		UINT              vbStride = 0, vbOffset = 0;
		ID3D11Buffer*     ib = nullptr;                  // IA index buffer
		DXGI_FORMAT       ibFormat = DXGI_FORMAT_UNKNOWN;
		UINT              ibOffset = 0;
		ID3D11SamplerState*       psSamp0 = nullptr;     // PS sampler s0 (A6 alpha diffuse sampler)
		ID3D11ShaderResourceView* psSrv0  = nullptr;     // PS SRV t0     (A6 diffuse SRV)
		ID3D11Buffer*             psCB0   = nullptr;     // PS constant buffer b0 (A5 translucent pass)

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
			ctx->OMGetBlendState(&blend, blendFactor, &sampleMask);
			ctx->IAGetVertexBuffers(0, 1, &vb0, &vbStride, &vbOffset);
			ctx->IAGetIndexBuffer(&ib, &ibFormat, &ibOffset);
			ctx->PSGetSamplers(0, 1, &psSamp0);
			ctx->PSGetShaderResources(0, 1, &psSrv0);
			ctx->PSGetConstantBuffers(0, 1, &psCB0);
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
			ctx->OMSetBlendState(blend, blendFactor, sampleMask);
			if (blend) blend->Release();
			ctx->IASetVertexBuffers(0, 1, &vb0, &vbStride, &vbOffset);
			if (vb0) vb0->Release();
			ctx->IASetIndexBuffer(ib, ibFormat, ibOffset);
			if (ib) ib->Release();
			ctx->PSSetSamplers(0, 1, &psSamp0);
			if (psSamp0) psSamp0->Release();
			ctx->PSSetShaderResources(0, 1, &psSrv0);
			if (psSrv0) psSrv0->Release();
			ctx->PSSetConstantBuffers(0, 1, &psCB0);
			if (psCB0) psCB0->Release();
		}
	};

}
