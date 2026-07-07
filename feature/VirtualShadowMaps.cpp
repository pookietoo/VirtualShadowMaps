#include "VirtualShadowMaps.h"

#include "Globals.h"  // globals::d3d::{device,context}, globals::game::smState

#include <DirectXMath.h>
#include <d3dcompiler.h>
#pragma comment(lib, "d3dcompiler.lib")

// RE types, ImGui, logger, json come from the CS PCH.

using namespace DirectX;

namespace
{
	// Skyrim's main view is reverse-Z. Flip if the preview looks inverted; drives the
	// depth clear value + DepthFunc (single source of truth).
	constexpr bool kReverseZ = false;

	// Embedded depth-only VS (model position -> light clip). Row-vector: mul(v, M).
	constexpr char kDepthVS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; };
float4 main(float3 pos : POSITION) : SV_Position { return mul(float4(pos, 1.0), WorldViewProj); }
)";

	struct alignas(16) PerDrawCB
	{
		XMFLOAT4X4 WorldViewProj;
	};

	// NiTransform (column-vector R*s + t) -> DirectXMath row-vector (v*M) matrix.
	XMMATRIX NiTransformToXM(const RE::NiTransform& t)
	{
		const auto& e = t.rotate.entry;
		const float s = t.scale;
		return XMMatrixSet(
		    s * e[0][0], s * e[1][0], s * e[2][0], 0.0f,
		    s * e[0][1], s * e[1][1], s * e[2][1], 0.0f,
		    s * e[0][2], s * e[1][2], s * e[2][2], 0.0f,
		    t.translate.x, t.translate.y, t.translate.z, 1.0f);
	}

	// Save/restore the graphics-stage state our draw pass disturbs.
	struct GraphicsStateGuard
	{
		ID3D11DeviceContext* ctx;
		ID3D11RenderTargetView* rtvs[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT]{};
		ID3D11DepthStencilView* dsv = nullptr;
		D3D11_VIEWPORT viewports[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]{};
		UINT numViewports = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
		winrt::com_ptr<ID3D11RasterizerState> rs;
		winrt::com_ptr<ID3D11DepthStencilState> ds;
		UINT stencilRef = 0;
		winrt::com_ptr<ID3D11InputLayout> il;
		D3D11_PRIMITIVE_TOPOLOGY topo{};
		winrt::com_ptr<ID3D11VertexShader> vs;
		winrt::com_ptr<ID3D11PixelShader> ps;
		ID3D11Buffer* vsCB0 = nullptr;

		explicit GraphicsStateGuard(ID3D11DeviceContext* c) : ctx(c)
		{
			ctx->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, &dsv);
			ctx->RSGetViewports(&numViewports, viewports);
			ctx->RSGetState(rs.put());
			ctx->OMGetDepthStencilState(ds.put(), &stencilRef);
			ctx->IAGetInputLayout(il.put());
			ctx->IAGetPrimitiveTopology(&topo);
			ctx->VSGetShader(vs.put(), nullptr, nullptr);
			ctx->PSGetShader(ps.put(), nullptr, nullptr);
			ctx->VSGetConstantBuffers(0, 1, &vsCB0);
		}
		~GraphicsStateGuard()
		{
			ctx->OMSetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, rtvs, dsv);
			for (auto* rtv : rtvs)
				if (rtv) rtv->Release();
			if (dsv) dsv->Release();
			ctx->RSSetViewports(numViewports, viewports);
			ctx->RSSetState(rs.get());
			ctx->OMSetDepthStencilState(ds.get(), stencilRef);
			ctx->IASetInputLayout(il.get());
			ctx->IASetPrimitiveTopology(topo);
			ctx->VSSetShader(vs.get(), nullptr, 0);
			ctx->PSSetShader(ps.get(), nullptr, 0);
			ctx->VSSetConstantBuffers(0, 1, &vsCB0);
			if (vsCB0) vsCB0->Release();
		}
	};
}

void VirtualShadowMaps::SetupResources()
{
	auto* device = globals::d3d::device;

	D3D11_TEXTURE2D_DESC td{};
	td.Width = td.Height = static_cast<UINT>(settings.Resolution);
	td.MipLevels = 1;
	td.ArraySize = 1;
	td.Format = DXGI_FORMAT_R32_TYPELESS;
	td.SampleDesc.Count = 1;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	depthMap = std::make_unique<Texture2D>(td, "VirtualShadowMaps::DepthMap");

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
	dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
	dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	depthMap->CreateDSV(dsvDesc);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = 1;
	depthMap->CreateSRV(srvDesc);

	perDrawCB = std::make_unique<ConstantBuffer>(ConstantBufferDesc<PerDrawCB>(), "VirtualShadowMaps::PerDrawCB");

	winrt::com_ptr<ID3DBlob> vsBlob, err;
	HRESULT hr = D3DCompile(kDepthVS, sizeof(kDepthVS) - 1, "DepthVS", nullptr, nullptr, "main", "vs_5_0", 0, 0, vsBlob.put(), err.put());
	if (FAILED(hr)) {
		logger::error("VSM: depth VS compile failed: {}", err ? (const char*)err->GetBufferPointer() : "");
		return;
	}
	device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, depthVS.put());

	auto makeLayout = [&](DXGI_FORMAT fmt, winrt::com_ptr<ID3D11InputLayout>& out) {
		D3D11_INPUT_ELEMENT_DESC ie{ "POSITION", 0, fmt, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };
		device->CreateInputLayout(&ie, 1, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), out.put());
	};
	makeLayout(DXGI_FORMAT_R32G32B32_FLOAT, ilFull);
	makeLayout(DXGI_FORMAT_R16G16B16A16_FLOAT, ilPacked);

	D3D11_DEPTH_STENCIL_DESC dsd{};
	dsd.DepthEnable = TRUE;
	dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsd.DepthFunc = kReverseZ ? D3D11_COMPARISON_GREATER : D3D11_COMPARISON_LESS;
	device->CreateDepthStencilState(&dsd, depthState.put());

	D3D11_RASTERIZER_DESC rd{};
	rd.FillMode = D3D11_FILL_SOLID;
	rd.CullMode = D3D11_CULL_BACK;
	rd.FrontCounterClockwise = FALSE;
	rd.DepthClipEnable = TRUE;
	rd.DepthBias = 100;
	rd.SlopeScaledDepthBias = 1.5f;
	device->CreateRasterizerState(&rd, rasterState.put());

	resourcesReady = true;
}

bool VirtualShadowMaps::PickTestLight(RE::ShadowSceneNode* a_ssn)
{
	if (!a_ssn)
		return false;
	auto& rt = a_ssn->GetRuntimeData();
	for (auto* sl : rt.shadowLightsAccum) {
		if (!sl)
			continue;
		if (reinterpret_cast<void*>(sl) == reinterpret_cast<void*>(rt.sunShadowDirLight))
			continue;
		auto& descriptors = sl->GetRuntimeData().shadowmapDescriptors;
		if (descriptors.empty())
			continue;
		XMMATRIX m = XMLoadFloat4x4(reinterpret_cast<const XMFLOAT4X4*>(&descriptors.front().lightTransform));
		XMStoreFloat4x4(&lightViewProj, XMMatrixTranspose(m));
		return true;
	}
	return false;
}

void VirtualShadowMaps::CollectCasters(RE::NiAVObject* a_obj, float a_radius)
{
	if (!a_obj)
		return;
	if (auto* tri = a_obj->AsTriShape()) {
		auto& grt = tri->GetGeometryRuntimeData();
		auto* rd = grt.rendererData;
		if (rd && rd->vertexBuffer && rd->indexBuffer && tri->GetTrishapeRuntimeData().triangleCount) {
			if (a_radius <= 0.0f || a_obj->worldBound.radius > 0.0f) {
				CasterEntry e{};
				e.vertexBuffer  = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
				e.indexBuffer   = reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer);
				e.vertexStride  = rd->vertexDesc.GetSize();
				e.fullPrecision = rd->vertexDesc.HasFlag(RE::BSGraphics::Vertex::VF_FULLPREC);
				e.indexCount    = static_cast<uint32_t>(tri->GetTrishapeRuntimeData().triangleCount) * 3u;
				XMStoreFloat4x4(&e.world, NiTransformToXM(a_obj->world));
				registry.push_back(e);
			}
		}
	}
	if (auto* node = a_obj->AsNode()) {
		for (auto& child : node->GetChildren())
			CollectCasters(child.get(), a_radius);
	}
}

void VirtualShadowMaps::RenderDepth()
{
	if (registry.empty() || !depthMap || !depthMap->dsv)
		return;

	auto* context = globals::d3d::context;
	GraphicsStateGuard guard(context);

	ID3D11RenderTargetView* noRTV = nullptr;
	context->OMSetRenderTargets(1, &noRTV, depthMap->dsv.get());
	context->ClearDepthStencilView(depthMap->dsv.get(), D3D11_CLEAR_DEPTH, kReverseZ ? 0.0f : 1.0f, 0);

	D3D11_VIEWPORT vp{ 0, 0, (float)settings.Resolution, (float)settings.Resolution, 0, 1 };
	context->RSSetViewports(1, &vp);
	context->RSSetState(rasterState.get());
	context->OMSetDepthStencilState(depthState.get(), 0);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->VSSetShader(depthVS.get(), nullptr, 0);
	context->PSSetShader(nullptr, nullptr, 0);

	XMMATRIX lightVP = XMLoadFloat4x4(&lightViewProj);
	auto* cb = perDrawCB->CB();
	context->VSSetConstantBuffers(0, 1, &cb);

	for (const auto& e : registry) {
		PerDrawCB data{};
		XMStoreFloat4x4(&data.WorldViewProj, XMMatrixMultiply(XMLoadFloat4x4(&e.world), lightVP));
		perDrawCB->Update(data);
		context->IASetInputLayout(e.fullPrecision ? ilFull.get() : ilPacked.get());
		UINT stride = e.vertexStride, offset = 0;
		context->IASetVertexBuffers(0, 1, &e.vertexBuffer, &stride, &offset);
		context->IASetIndexBuffer(e.indexBuffer, DXGI_FORMAT_R16_UINT, 0);
		context->DrawIndexed(e.indexCount, 0, 0);
	}
}

void VirtualShadowMaps::Prepass()
{
	registry.clear();
	haveTestLight = false;
	if (!settings.Enabled || !resourcesReady)
		return;

	auto* ssn = globals::game::smState->shadowSceneNode[0];
	haveTestLight = PickTestLight(ssn);
	if (!haveTestLight)
		return;

	CollectCasters(static_cast<RE::NiAVObject*>(ssn), settings.CasterRadius);
	RenderDepth();
}

void VirtualShadowMaps::DrawSettings()
{
	ImGui::Checkbox("Enabled", &settings.Enabled);
	ImGui::SliderInt("Resolution (restart to apply)", &settings.Resolution, 512, 4096);
	ImGui::SliderFloat("Preview scale", &settings.PreviewScale, 0.05f, 1.0f, "%.2f");
	ImGui::Text("Active: %s   |   casters: %d", haveTestLight ? "yes" : "no", (int)registry.size());

	// Live depth-map preview (the M0 validation): the scene silhouetted from the
	// borrowed light's point of view (R32 depth in the red channel).
	if (depthMap && depthMap->srv) {
		float sz = settings.Resolution * settings.PreviewScale;
		ImGui::Image(depthMap->srv.get(), ImVec2(sz, sz));
	}
}

void VirtualShadowMaps::LoadSettings(json& o_json) { settings = o_json; }
void VirtualShadowMaps::SaveSettings(json& o_json) { o_json = settings; }

void VirtualShadowMaps::Reset()
{
	registry.clear();
	haveTestLight = false;
}
