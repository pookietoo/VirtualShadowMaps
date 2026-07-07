#include "VirtualShadowMaps.h"

#include <imgui.h>  // shared with CS's context via the add-on hook

#include <DirectXPackedVector.h>  // XMConvertHalfToFloat (bone-weight halfs)

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

// Path B (skinned characters): read skin partitions + the engine bone-matrix palette, CPU-skin, render.
#include "RE/B/BSGeometry.h"
#include "RE/B/BSDynamicTriShape.h"     // CPU-skinned posed verts (dynamicData)
#include "RE/N/NiSkinInstance.h"        // GPU-skinned: skinData/skinPartition/bones/boneMatrices
#include "RE/N/NiSkinData.h"            // inverse-bind (boneData[b].skinToBone)
#include "RE/N/NiSkinPartition.h"       // bind-pose buffers + bone indices/weights per partition

using namespace DirectX;

namespace
{
	// Bump this every build so the menu confirms the plugin DLL actually updated. (Shader
	// changes are separate — the shader debug tint below confirms those.)
	constexpr char kBuildTag[] = "0.9.35 per-light jitter detector: BSLight::dynamic + per-frame position delta (find the moving shadow light)";

	// Skyrim's main view is reverse-Z. Flip if the dumped map looks inverted; drives
	// the depth clear value + DepthFunc (single source of truth).
	constexpr bool kReverseZ = false;

	// Light perspective near/far (must match kNear/kFar in the linearize PS below).
	constexpr float kNear = 8.0f;
	constexpr float kFar  = 8192.0f;

	// Re-traverse the scene graph to rebuild the persistent registry at most this often
	// (frames). Between rebuilds each live geometry's buffers/transform are re-read, so
	// this interval only bounds how quickly streamed-in/out geometry is picked up.
	constexpr uint32_t kRebuildInterval = 30;

	// Embedded depth-only VS (position -> light clip). Row-vector: mul(v, M).
	constexpr char kDepthVS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; };
float4 main(float3 pos : POSITION) : SV_Position { return mul(float4(pos, 1.0), WorldViewProj); }
)";

	// Fullscreen triangle (no vertex buffer) + linearize PS: turns the raw perspective
	// depth map into a readable grayscale (near = white, far = black) for the debug preview.
	constexpr char kFullscreenVS[] = R"(
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VSOut main(uint id : SV_VertexID) {
	VSOut o;
	o.uv  = float2((id << 1) & 2, id & 2);
	o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
	return o;
}
)";
	constexpr char kLinearizePS[] = R"(
Texture2D DepthTex : register(t0);
SamplerState Samp  : register(s0);
cbuffer PreviewCB : register(b0) { float PreviewRange; float3 _pad; };
static const float kNear = 8.0;
static const float kFar  = 8192.0;
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
	float d = DepthTex.Sample(Samp, uv).r;
	float z = kNear * kFar / max(kFar - d * (kFar - kNear), 1e-4);  // perspective -> view-space Z (world units)
	float g = saturate(1.0 - z / max(PreviewRange, 1.0));           // near = white, PreviewRange+ units = black
	return float4(g, g, g, 1);
}
)";

	struct alignas(16) PerDrawCB
	{
		XMFLOAT4X4 WorldViewProj;
	};

	// ---- GPU shadow-math probe (see VirtualShadowMaps.h). One thread per probe point runs the
	// EXACT VSM::GetLocalShadow logic against the live atlas + light buffer and writes every
	// intermediate out. Kept byte-identical to VSM.hlsli so it proves the real path, not a copy. ----
	struct ProbeIn   // must match kProbeCS ProbeIn
	{
		XMFLOAT4 P;           // pixel/surface position, camera-relative (== shader input.WorldPosition)
		XMFLOAT4 lightPosWS;  // the light as LLF passes it (camera-relative to the FrameBuffer eye)
	};
	struct ProbeOut  // must match kProbeCS ProbeOut
	{
		XMFLOAT4 W;           // xyz = sample-space position; w = sampleSpace used
		XMFLOAT4 absLight;    // xyz = match key (lightPosWS + matchEye); w = nearest buffer-light distance
		XMFLOAT4 ndc;         // xyz = ndc; w = clip.w
		XMFLOAT4 uv_occ;      // xy = atlasUV; z = occluder sampled; w = matched light index (-1 none)
		XMFLOAT4 result;      // x = linPix; y = linOcc; z = diff; w = shadowed? (1=shadow)
		XMFLOAT4 face_flags;  // x = face; y = inBounds; z = inFront; w = matched?(1/0)
	};
	struct alignas(16) ProbeCBData  // must match kProbeCS cbuffer ProbeCB
	{
		XMFLOAT4 camAdjust;   // xyz used as FrameBuffer::CameraPosAdjust (we feed altEye — exact in 3rd person)
		XMFLOAT4 altEye;      // xyz render eye (posAdjust.getEye)
		float    bias;        int compareMode; int matchEye; int sampleSpace;
		float    matchThresh; int numProbes;   int pad0;     int pad1;
	};

	// Real-shader pixel probe readback (MUST match VSM.hlsli::VSMPixelProbe — 10 float4 = 160 bytes).
	struct PixelProbe
	{
		XMFLOAT4 pixel;       // xy = SV_Position, z = written(1 => shader wrote this), w = mode
		XMFLOAT4 P;           // input.WorldPosition (camera-relative) as the REAL shader received it
		XMFLOAT4 camAdjust;   // REAL FrameBuffer::CameraPosAdjust.xyz — resolves the last assumption
		XMFLOAT4 W;           // SampleP(P).xyz, w = sampleSpace
		XMFLOAT4 lightPosWS;  // light as LLF passed it
		XMFLOAT4 absLight;    // match key
		XMFLOAT4 matched;     // x = matched index, y = face, z = inBounds, w = inFront
		XMFLOAT4 ndc;         // xyz = ndc, w = clip.w
		XMFLOAT4 uv_occ;      // xy = atlasUV, z = occluder, w = reason(0 eval,1 mode,2 no-match,3 behind,4 oob)
		XMFLOAT4 result;      // x = linPix, y = linOcc, z = diff, w = shadow (0 = shadowed)
	};

	// Compute shader: replicates VSM.hlsli::GetLocalShadow verbatim (constants inlined). Inputs are
	// the SAME resources CS binds at t110/t111/s7; here bound at t0/t1/s0 for our own dispatch.
	constexpr char kProbeCS[] = R"(
struct LightRecord { float4 positionWS; float farPlane; float nearPlane; uint atlasCol; uint atlasRow; row_major float4x4 cubeVP[6]; };
struct ProbeIn  { float4 P; float4 lightPosWS; };
struct ProbeOut { float4 W; float4 absLight; float4 ndc; float4 uv_occ; float4 result; float4 face_flags; };
cbuffer ProbeCB : register(b0) {
	float4 gCamAdjust; float4 gAltEye;
	float gBias; int gCompareMode; int gMatchEye; int gSampleSpace;
	float gMatchThresh; int gNumProbes; int _p0; int _p1;
};
Texture2D<float>              Atlas  : register(t0);
StructuredBuffer<LightRecord> Lights : register(t1);
StructuredBuffer<ProbeIn>     Probes : register(t2);
SamplerState                  Samp   : register(s0);
RWStructuredBuffer<ProbeOut>  Out    : register(u0);
static const int kFaceRes=256, kMaxLights=32, kBlockW=768, kBlockH=512, kAtlasW=3072, kAtlasH=4096;
int FaceFromDir(float3 v){ float3 a=abs(v); if(a.x>=a.y&&a.x>=a.z) return v.x>=0?0:1; if(a.y>=a.z) return v.y>=0?2:3; return v.z>=0?4:5; }
float Lin(float d,float n,float f){ return n*f/max(f-d*(f-n),1e-4); }
[numthreads(64,1,1)]
void main(uint3 id: SV_DispatchThreadID){
	uint i=id.x; if(i>=(uint)gNumProbes) return;
	ProbeIn pin=Probes[i];
	float3 P=pin.P.xyz, lightPosWS=pin.lightPosWS.xyz;
	float3 W = (gSampleSpace==1)?P:((gSampleSpace==2)?(P+gAltEye.xyz):(P+gCamAdjust.xyz));
	float3 matchEye=(gMatchEye==1)?gAltEye.xyz:gCamAdjust.xyz;
	float3 absLight=lightPosWS+matchEye;
	int best=-1; float bestD=1e30;
	[loop] for(uint k=0;k<(uint)kMaxLights;k++){ float3 rp=Lights[k].positionWS.xyz; if(dot(rp,rp)<1) continue; float d=distance(rp,absLight); if(d<bestD){bestD=d;best=(int)k;} }
	int matched=-1, face=-1, inb=0, front=0, shadow=0;
	float clipW=0, occ=1, linPix=0, linOcc=0; float3 ndc=float3(9,9,9); float2 auv=float2(0,0);
	[loop] for(uint k2=0;k2<(uint)kMaxLights;k2++){
		float3 rp=Lights[k2].positionWS.xyz; if(dot(rp,rp)<1) continue;
		if(distance(rp,absLight)>=gMatchThresh) continue;
		matched=(int)k2; LightRecord L=Lights[k2];
		face=FaceFromDir(W-L.positionWS.xyz);
		float4 clip=mul(float4(W,1),L.cubeVP[face]); clipW=clip.w;
		front=clipW>1e-4?1:0; if(!front) break;
		ndc=clip.xyz/clip.w;
		inb=(ndc.x>=-1&&ndc.x<=1&&ndc.y>=-1&&ndc.y<=1&&ndc.z>=0&&ndc.z<=1)?1:0;
		float2 fuv=ndc.xy*float2(0.5,-0.5)+0.5;
		float2 tile=float2(face%3,face/3);
		float2 px=float2(L.atlasCol*kBlockW,L.atlasRow*kBlockH)+(tile+fuv)*kFaceRes;
		auv=px/float2(kAtlasW,kAtlasH);
		occ=Atlas.SampleLevel(Samp,auv,0);
		linPix=Lin(ndc.z,L.nearPlane,L.farPlane); linOcc=Lin(occ,L.nearPlane,L.farPlane);
		// out-of-bounds -> lit, exactly as VSM.hlsli::GetLocalShadow early-returns 1.0
		shadow = inb ? ((gCompareMode==1)?((ndc.z-occ>0)?1:0):(((linPix-linOcc)>gBias)?1:0)) : 0;
		break;
	}
	ProbeOut o;
	o.W=float4(W,(float)gSampleSpace);
	o.absLight=float4(absLight,bestD);
	o.ndc=float4(ndc,clipW);
	o.uv_occ=float4(auv,occ,(float)matched);
	o.result=float4(linPix,linOcc,linPix-linOcc,(float)shadow);
	o.face_flags=float4((float)face,(float)inb,(float)front,(float)(matched>=0?1:0));
	Out[i]=o;
}
)";

	// NiTransform (column-vector R*s + t) -> DirectXMath row-vector (v*M) matrix.
	XMMATRIX NiTransformToXM(const RE::NiTransform& t)
	{
		const auto& e = t.rotate.entry;  // float entry[3][3]
		const float s = t.scale;
		return XMMatrixSet(
		    s * e[0][0], s * e[1][0], s * e[2][0], 0.0f,
		    s * e[0][1], s * e[1][1], s * e[2][1], 0.0f,
		    s * e[0][2], s * e[1][2], s * e[2][2], 0.0f,
		    t.translate.x, t.translate.y, t.translate.z, 1.0f);
	}

	// Scene-graph census: counts what our RebuildRegistry traversal reaches, by kind. triShapes are
	// what we capture; otherLeaves are non-node, non-BSTriShape leaves (geometry types we SKIP —
	// e.g. BSDynamicTriShape/BSSubIndexTriShape/particles). If otherLeaves >> triShapes, we're
	// missing most geometry because we only handle AsTriShape().
	struct SceneCensus { int nodes = 0, triShapes = 0, otherLeaves = 0, maxDepth = 0, lodSkipped = 0; const char* sampleOther = nullptr; };
	void CensusWalk(RE::NiAVObject* o, int depth, SceneCensus& c)
	{
		if (!o || depth > 128)
			return;
		if (depth > c.maxDepth)
			c.maxDepth = depth;
		if (const char* nm = o->name.c_str(); nm && std::strcmp(nm, "ObjectLODRoot") == 0) {
			++c.lodSkipped;
			return;
		}
		if (o->AsTriShape())
			++c.triShapes;
		if (auto* node = o->AsNode()) {
			++c.nodes;
			for (auto& ch : node->GetChildren())
				CensusWalk(ch.get(), depth + 1, c);
		} else if (!o->AsTriShape()) {
			++c.otherLeaves;  // geometry/effect leaf we don't currently register
			if (!c.sampleOther && o->GetRTTI())
				c.sampleOther = o->GetRTTI()->name;  // RTTI type so we can see WHAT we're missing
		}
	}

	// Extract 6 world-space frustum planes from a row-vector view-proj (clip = pos * M).
	// Plane vectors come from the COLUMNS of M (Gribb-Hartmann, row-vector form), D3D [0,1] Z.
	void ExtractFrustumPlanes(const XMFLOAT4X4& m, XMFLOAT4 outPlanes[6])
	{
		const XMVECTOR c0 = XMVectorSet(m._11, m._21, m._31, m._41);
		const XMVECTOR c1 = XMVectorSet(m._12, m._22, m._32, m._42);
		const XMVECTOR c2 = XMVectorSet(m._13, m._23, m._33, m._43);
		const XMVECTOR c3 = XMVectorSet(m._14, m._24, m._34, m._44);
		XMVECTOR p[6] = {
			XMVectorAdd(c3, c0),       // left
			XMVectorSubtract(c3, c0),  // right
			XMVectorAdd(c3, c1),       // bottom
			XMVectorSubtract(c3, c1),  // top
			c2,                        // near (z >= 0)
			XMVectorSubtract(c3, c2),  // far  (z <= w)
		};
		for (int i = 0; i < 6; ++i) {
			const XMVECTOR len = XMVector3Length(p[i]);
			if (XMVectorGetX(len) > 1e-6f)
				p[i] = XMVectorDivide(p[i], len);
			XMStoreFloat4(&outPlanes[i], p[i]);
		}
	}

	// Sphere vs frustum: false if the sphere lies fully outside any plane.
	bool SphereInFrustum(const XMFLOAT4 planes[6], const RE::NiPoint3& c, float r)
	{
		for (int i = 0; i < 6; ++i) {
			const float dist = planes[i].x * c.x + planes[i].y * c.y + planes[i].z * c.z + planes[i].w;
			if (dist < -r)
				return false;
		}
		return true;
	}

	// Save/restore the graphics-stage state our draw pass disturbs.
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
}

void VirtualShadowMaps::OnD3DReady(ID3D11Device* a_device, ID3D11DeviceContext* a_context)
{
	device = a_device;
	context = a_context;
	SetupResources();
}

void VirtualShadowMaps::SetupResources()
{
	if (!device)
		return;

	// owned shadow atlas: typeless R32 -> D32 DSV + R32F SRV (6 cube faces in a 3x2 grid)
	D3D11_TEXTURE2D_DESC td{};
	td.Width  = static_cast<UINT>(kAtlasW);
	td.Height = static_cast<UINT>(kAtlasH);
	td.MipLevels = 1;
	td.ArraySize = 1;
	td.Format = DXGI_FORMAT_R32_TYPELESS;
	td.SampleDesc.Count = 1;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	if (FAILED(device->CreateTexture2D(&td, nullptr, &depthTex))) {
		logger::error("VSM: depth texture creation failed");
		return;
	}

	D3D11_DEPTH_STENCIL_VIEW_DESC dsvDesc{};
	dsvDesc.Format = DXGI_FORMAT_D32_FLOAT;
	dsvDesc.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	device->CreateDepthStencilView(depthTex.Get(), &dsvDesc, &dsv);

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
	srvDesc.Format = DXGI_FORMAT_R32_FLOAT;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
	srvDesc.Texture2D.MipLevels = 1;
	device->CreateShaderResourceView(depthTex.Get(), &srvDesc, &srv);

	// per-draw CB
	D3D11_BUFFER_DESC cbDesc{};
	cbDesc.ByteWidth = sizeof(PerDrawCB);
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	device->CreateBuffer(&cbDesc, nullptr, &perDrawCB);

	// compile the embedded depth VS -> shader + input layouts
	ComPtr<ID3DBlob> vsBlob, err;
	HRESULT hr = D3DCompile(kDepthVS, sizeof(kDepthVS) - 1, "DepthVS", nullptr, nullptr,
	    "main", "vs_5_0", 0, 0, &vsBlob, &err);
	if (FAILED(hr)) {
		logger::error("VSM: depth VS compile failed: {}",
		    err ? static_cast<const char*>(err->GetBufferPointer()) : "");
		return;
	}
	device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &depthVS);

	// position is always R32G32B32_FLOAT at offset 0 (see CollectCasters)
	const D3D11_INPUT_ELEMENT_DESC ie{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };
	device->CreateInputLayout(&ie, 1, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &ilFull);

	D3D11_DEPTH_STENCIL_DESC dsd{};
	dsd.DepthEnable = TRUE;
	dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsd.DepthFunc = kReverseZ ? D3D11_COMPARISON_GREATER : D3D11_COMPARISON_LESS;
	device->CreateDepthStencilState(&dsd, &depthState);

	D3D11_RASTERIZER_DESC rd{};
	rd.FillMode = D3D11_FILL_SOLID;
	rd.CullMode = D3D11_CULL_BACK;
	rd.FrontCounterClockwise = FALSE;
	rd.DepthClipEnable = TRUE;
	rd.DepthBias = 100;
	rd.SlopeScaledDepthBias = 1.5f;
	device->CreateRasterizerState(&rd, &rasterState);

	// linear-depth debug view: RGBA8 color target + fullscreen resolve shaders + point sampler
	D3D11_TEXTURE2D_DESC pd = td;  // same dimensions as the depth map
	pd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	pd.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	if (SUCCEEDED(device->CreateTexture2D(&pd, nullptr, &previewTex))) {
		device->CreateRenderTargetView(previewTex.Get(), nullptr, &previewRTV);
		device->CreateShaderResourceView(previewTex.Get(), nullptr, &previewSRV);
	}

	ComPtr<ID3DBlob> fsBlob, psBlob, e2;
	if (SUCCEEDED(D3DCompile(kFullscreenVS, sizeof(kFullscreenVS) - 1, "FullscreenVS", nullptr, nullptr,
	        "main", "vs_5_0", 0, 0, &fsBlob, &e2)))
		device->CreateVertexShader(fsBlob->GetBufferPointer(), fsBlob->GetBufferSize(), nullptr, &fullscreenVS);
	if (SUCCEEDED(D3DCompile(kLinearizePS, sizeof(kLinearizePS) - 1, "LinearizePS", nullptr, nullptr,
	        "main", "ps_5_0", 0, 0, &psBlob, &e2)))
		device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &linearizePS);

	D3D11_SAMPLER_DESC samp{};
	samp.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	samp.AddressU = samp.AddressV = samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	device->CreateSamplerState(&samp, &pointSampler);

	D3D11_BUFFER_DESC pcb{};
	pcb.ByteWidth = 16;
	pcb.Usage = D3D11_USAGE_DYNAMIC;
	pcb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	pcb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	device->CreateBuffer(&pcb, nullptr, &previewCB);

	// per-light structured buffer (mirrors lightRecords) for the LLF shader
	D3D11_BUFFER_DESC lb{};
	lb.ByteWidth = sizeof(LightRecord) * kMaxLights;
	lb.Usage = D3D11_USAGE_DYNAMIC;
	lb.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	lb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	lb.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	lb.StructureByteStride = sizeof(LightRecord);
	if (SUCCEEDED(device->CreateBuffer(&lb, nullptr, &lightBuffer))) {
		D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
		sd.Format = DXGI_FORMAT_UNKNOWN;
		sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		sd.Buffer.FirstElement = 0;
		sd.Buffer.NumElements = kMaxLights;
		device->CreateShaderResourceView(lightBuffer.Get(), &sd, &lightBufferSRV);
	}

	// live debug tuning cbuffer (b13): bias / mode / vizScale / sampleSpace / matchThresh /
	// compareMode / altEye (see UpdateDebugCB). 48 bytes = three float4 rows.
	D3D11_BUFFER_DESC dcb{};
	dcb.ByteWidth      = 48;
	dcb.Usage          = D3D11_USAGE_DYNAMIC;
	dcb.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
	dcb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	device->CreateBuffer(&dcb, nullptr, &debugCB);

	// ---- GPU shadow-math probe resources (compute pass + in/out structured buffers) ----
	{
		ComPtr<ID3DBlob> csBlob, csErr;
		HRESULT hrc = D3DCompile(kProbeCS, sizeof(kProbeCS) - 1, "ProbeCS", nullptr, nullptr,
		    "main", "cs_5_0", 0, 0, &csBlob, &csErr);
		if (FAILED(hrc)) {
			logger::error("VSM: probe CS compile failed: {}",
			    csErr ? static_cast<const char*>(csErr->GetBufferPointer()) : "");
		} else {
			device->CreateComputeShader(csBlob->GetBufferPointer(), csBlob->GetBufferSize(), nullptr, &probeCS);

			// probe input: dynamic structured buffer (t2)
			D3D11_BUFFER_DESC ib{};
			ib.ByteWidth = sizeof(ProbeIn) * kMaxProbes;
			ib.Usage = D3D11_USAGE_DYNAMIC;
			ib.BindFlags = D3D11_BIND_SHADER_RESOURCE;
			ib.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			ib.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			ib.StructureByteStride = sizeof(ProbeIn);
			if (SUCCEEDED(device->CreateBuffer(&ib, nullptr, &probeInBuf))) {
				D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
				sd.Format = DXGI_FORMAT_UNKNOWN;
				sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
				sd.Buffer.NumElements = kMaxProbes;
				device->CreateShaderResourceView(probeInBuf.Get(), &sd, &probeInSRV);
			}

			// probe output: default structured buffer + UAV (u0), plus a staging copy for readback
			D3D11_BUFFER_DESC ob{};
			ob.ByteWidth = sizeof(ProbeOut) * kMaxProbes;
			ob.Usage = D3D11_USAGE_DEFAULT;
			ob.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
			ob.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			ob.StructureByteStride = sizeof(ProbeOut);
			if (SUCCEEDED(device->CreateBuffer(&ob, nullptr, &probeOutBuf))) {
				D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
				ud.Format = DXGI_FORMAT_UNKNOWN;
				ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
				ud.Buffer.NumElements = kMaxProbes;
				device->CreateUnorderedAccessView(probeOutBuf.Get(), &ud, &probeOutUAV);
			}
			D3D11_BUFFER_DESC stg{};
			stg.ByteWidth = sizeof(ProbeOut) * kMaxProbes;
			stg.Usage = D3D11_USAGE_STAGING;
			stg.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			stg.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
			stg.StructureByteStride = sizeof(ProbeOut);
			device->CreateBuffer(&stg, nullptr, &probeOutStaging);

			// probe cbuffer (b0)
			D3D11_BUFFER_DESC pcbd{};
			pcbd.ByteWidth = sizeof(ProbeCBData);
			pcbd.Usage = D3D11_USAGE_DYNAMIC;
			pcbd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
			pcbd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
			device->CreateBuffer(&pcbd, nullptr, &probeCB);
		}
	}

	// ---- Real-shader pixel probe (u7): 1-element structured buffer the game's Lighting.hlsl writes,
	// bound by our OMSetRenderTargets hook only while armed; plus a staging copy for readback. ----
	{
		D3D11_BUFFER_DESC pb{};
		pb.ByteWidth = sizeof(PixelProbe);
		pb.Usage = D3D11_USAGE_DEFAULT;
		pb.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
		pb.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		pb.StructureByteStride = sizeof(PixelProbe);
		if (SUCCEEDED(device->CreateBuffer(&pb, nullptr, &pixelProbeBuf))) {
			D3D11_UNORDERED_ACCESS_VIEW_DESC ud{};
			ud.Format = DXGI_FORMAT_UNKNOWN;
			ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
			ud.Buffer.NumElements = 1;
			device->CreateUnorderedAccessView(pixelProbeBuf.Get(), &ud, &pixelProbeUAV);
		}
		D3D11_BUFFER_DESC ps{};
		ps.ByteWidth = sizeof(PixelProbe);
		ps.Usage = D3D11_USAGE_STAGING;
		ps.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ps.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		ps.StructureByteStride = sizeof(PixelProbe);
		device->CreateBuffer(&ps, nullptr, &pixelProbeStaging);
	}

	registry.reserve(1024);  // avoid per-frame reallocation churn

	resourcesReady = true;
	logger::info("VSM: resources ready ({}x{} shadow atlas + linear preview)", kAtlasW, kAtlasH);
}

void VirtualShadowMaps::CollectLights(RE::ShadowSceneNode* a_ssn)
{
	lightRecords.clear();
	lightDynamic.clear();
	haveTestLight    = false;
	activeLightCount = 0;
	lightsAmbient    = 0;
	lightsNonShadow  = 0;
	if (!a_ssn)
		return;
	auto& rt = a_ssn->GetRuntimeData();
	const auto& camPos = rt.cameraPos;
	dbgCam = { camPos.x, camPos.y, camPos.z };
	sceneCameraPos = { camPos.x, camPos.y, camPos.z };  // reliable atlas eye (see RenderDepth/UpdateDebugCB)

	// Gather ALL active local lights (not the engine's shadow-limited subset) — the point
	// of the mod is to shadow lights the engine dropped. Cap at kMaxLights (correctness-
	// first; the cap lifts once culling/paging land). Each light gets a cube "block" laid
	// out in a kLightsPerRow grid in the atlas.
	float nearestDistSq = FLT_MAX;
	int   nearestIdx    = -1;

	// Add one scene-graph light to the buffer, deduped by world position (a light can appear
	// in both arrays). Returns false once the atlas is full so the caller stops. We collect
	// BOTH activeLights AND activeShadowLights — matching LLF's light set exactly — so
	// shadow-casting lights (braziers etc., which live in activeShadowLights) also get cubes.
	// The sun and origin-positioned lights are skipped.
	auto addLight = [&](RE::BSLight* bl) -> bool {
		if (lightRecords.size() >= static_cast<size_t>(kMaxLights))
			return false;  // atlas full — cap reached (capacity limit, not a filter)
		if (!bl || bl == rt.sunLight)
			return true;
		// Only genuine shadow-casters — skip ambient/fill lights (the engine's own flags). This is the
		// fix for "shadows from nowhere": we were shadowing every illuminating light, not just casters.
		if (bl->ambientLight) {
			++lightsAmbient;
			return true;
		}
		if (!bl->IsShadowLight()) {
			++lightsNonShadow;
			return true;
		}
		RE::NiLight* niLight = bl->light.get();  // scene-graph light; worldTranslate reads 0
		if (!niLight)
			return true;
		const RE::NiPoint3& p = niLight->world.translate;
		if (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f)
			return true;
		for (const auto& rec : lightRecords)  // dedup: same light already added via the other array
			if (rec.positionWS.x == p.x && rec.positionWS.y == p.y && rec.positionWS.z == p.z)
				return true;

		const float    radius = niLight->GetLightRuntimeData().radius.x;  // this light's actual reach
		const uint32_t idx    = static_cast<uint32_t>(lightRecords.size());
		LightRecord r;
		r.positionWS = { p.x, p.y, p.z, 1.0f };
		r.farPlane   = (std::max)(radius * dbgFarScale, 1.0f);          // far  = light radius * FarScale
		r.nearPlane  = (std::max)(r.farPlane * dbgNearFrac, 0.1f);      // near = far * NearFrac
		r.atlasCol   = idx % static_cast<uint32_t>(kLightsPerRow);
		r.atlasRow   = idx / static_cast<uint32_t>(kLightsPerRow);
		BuildCubeMatrices(XMVectorSet(p.x, p.y, p.z, 1.0f), r.nearPlane, r.farPlane, r.cubeVP);
		lightRecords.push_back(r);
		lightDynamic.push_back(bl->dynamic ? 1 : 0);  // parallel to lightRecords: is this a moving/animated light?

		const float dx = p.x - camPos.x, dy = p.y - camPos.y, dz = p.z - camPos.z;
		const float d2 = dx * dx + dy * dy + dz * dz;
		if (d2 < nearestDistSq) {
			nearestDistSq = d2;
			nearestIdx    = static_cast<int>(idx);
		}
		return true;
	};

	for (auto& lightPtr : rt.activeLights) {  // BSTArray<NiPointer<BSLight>> — all active local lights
		if (!addLight(lightPtr.get()))
			break;
	}
	for (auto& lightPtr : rt.activeShadowLights) {  // shadow-casters the engine tracks separately
		if (!addLight(lightPtr.get()))
			break;
	}
	activeLightCount = static_cast<int>(lightRecords.size());

	// Per-frame light movement (jitter detector): compare each light to last frame's SAME-position light
	// (nearest match, robust to reordering). A moving/flickering shadow light throws a sweeping shadow.
	lightMove.assign(lightRecords.size(), -1.0f);  // -1 = no prior match (new light)
	for (size_t i = 0; i < lightRecords.size(); ++i) {
		const auto& p = lightRecords[i].positionWS;
		float best = FLT_MAX;
		for (const auto& q : prevLightPos) {
			const float d = std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z));
			best = (std::min)(best, d);
		}
		if (!prevLightPos.empty())
			lightMove[i] = best;  // distance to the closest light last frame = how far this light moved
	}
	prevLightPos.resize(lightRecords.size());
	for (size_t i = 0; i < lightRecords.size(); ++i)
		prevLightPos[i] = { lightRecords[i].positionWS.x, lightRecords[i].positionWS.y, lightRecords[i].positionWS.z };
	haveTestLight    = !lightRecords.empty();

	// Diagnostics: highlight the selected light (or nearest if none selected).
	const int sel = (lightSelect >= 0 && lightSelect < activeLightCount) ? lightSelect : nearestIdx;
	if (sel >= 0) {
		dbgEye = { lightRecords[sel].positionWS.x, lightRecords[sel].positionWS.y, lightRecords[sel].positionWS.z };
		const float dx = dbgEye.x - camPos.x, dy = dbgEye.y - camPos.y, dz = dbgEye.z - camPos.z;
		dbgLightDist = std::sqrt(dx * dx + dy * dy + dz * dz);
	}
}

void VirtualShadowMaps::UploadLightBuffer()
{
	if (!lightBuffer)
		return;
	D3D11_MAPPED_SUBRESOURCE m{};
	if (SUCCEEDED(context->Map(lightBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		const size_t n = (std::min)(lightRecords.size(), static_cast<size_t>(kMaxLights));
		if (n)
			std::memcpy(m.pData, lightRecords.data(), n * sizeof(LightRecord));
		// Zero unused slots so the shader's position match can't hit stale data
		// (real lights are never at the origin — see CollectLights).
		if (n < static_cast<size_t>(kMaxLights))
			std::memset(static_cast<uint8_t*>(m.pData) + n * sizeof(LightRecord), 0,
			    (static_cast<size_t>(kMaxLights) - n) * sizeof(LightRecord));
		context->Unmap(lightBuffer.Get(), 0);
	}
}

// Mirror the live tuning sliders into the shader's b13 cbuffer (must match VSM.hlsli's VSMDebug).
// Layout is three float4 rows (48 bytes); scalar members are laid out to match the HLSL offsets.
void VirtualShadowMaps::UpdateDebugCB()
{
	if (!debugCB)
		return;

	// altEye = the render eye the atlas was rasterized around. Use the RELIABLE ShadowSceneNode
	// cameraPos (set in CollectLights), NOT posAdjust.getEye() which intermittently returns garbage
	// (0,0,1) and mis-places the atlas. Space 2 samples with P + altEye for comparison.
	const RE::NiPoint3 eye{ sceneCameraPos.x, sceneCameraPos.y, sceneCameraPos.z };

	struct
	{
		float biasWorld;    // 0  shadow bias in world units (linear-distance compare)
		int   mode;         // 4  0 shadow / 1 off / >=2 RGB diagnostics
		float vizScale;     // 8  atlas-depth view scale
		int   sampleSpace;  // 12 0 absP / 1 P / 2 P+altEye
		float matchThresh;  // 16 light-match distance (world units)
		int   compareMode;  // 20 0 linearized / 1 raw ndc.z
		int   matchEye;     // 24 0 CameraPosAdjust / 1 altEye (posAdjust.getEye)
		float dbgB;         // 28 spare
		float altEyeX;      // 32 render eye (posAdjust)
		float altEyeY;      // 36
		float altEyeZ;      // 40
		int   probeArmedCB; // 44 1 -> Lighting.hlsl writes the centre-pixel probe to u7
	} cb{ dbgBiasWorld, dbgMode, dbgVizScale, dbgSampleSpace,
	      dbgMatchThresh, dbgCompareMode, dbgMatchEye, 0.0f,
	      eye.x, eye.y, eye.z, probeArmed ? 1 : 0 };

	D3D11_MAPPED_SUBRESOURCE m{};
	if (SUCCEEDED(context->Map(debugCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		std::memcpy(m.pData, &cb, sizeof(cb));
		context->Unmap(debugCB.Get(), 0);
	}
}

// Six 90-degree perspective faces of a cube centred on the light. Face order (+X,-X,+Y,-Y,
// +Z,-Z) and the up vectors below MUST match VSM.hlsli's face selection + sampling.
void VirtualShadowMaps::BuildCubeMatrices(FXMVECTOR a_eye, float a_near, float a_far, XMFLOAT4X4 a_outVP[6])
{
	// Side faces (+X,-X,+Y,-Y) use world-Z up; the top/bottom faces (+Z,-Z) look along Z,
	// so they use world-Y up instead.
	const XMVECTOR dirs[6] = {
		XMVectorSet(1, 0, 0, 0), XMVectorSet(-1, 0, 0, 0),
		XMVectorSet(0, 1, 0, 0), XMVectorSet(0, -1, 0, 0),
		XMVectorSet(0, 0, 1, 0), XMVectorSet(0, 0, -1, 0),
	};
	const XMVECTOR ups[6] = {
		XMVectorSet(0, 0, 1, 0), XMVectorSet(0, 0, 1, 0),
		XMVectorSet(0, 0, 1, 0), XMVectorSet(0, 0, 1, 0),
		XMVectorSet(0, 1, 0, 0), XMVectorSet(0, 1, 0, 0),
	};
	const XMMATRIX proj = XMMatrixPerspectiveFovLH(XMConvertToRadians(90.0f), 1.0f, a_near, a_far);
	for (int f = 0; f < 6; ++f) {
		const XMMATRIX view = XMMatrixLookToLH(a_eye, dirs[f], ups[f]);
		XMStoreFloat4x4(&a_outVP[f], XMMatrixMultiply(view, proj));
	}
}

// Traverse the scene graph and collect all opaque casters into the persistent registry.
// Light-independent: no frustum cull here (that runs per-frame in RenderDepth, against
// each geometry's current bound). We store an NiPointer to keep the geometry alive; its
// buffers/transform are re-read every frame. GPU readback confirmed POSITION is float3
// at offset 0 with stride == VertexDesc::GetSize() (see [[reading-skyrim-geometry]]).
// SEH-guarded entry point. A render-thread walk of the live scene graph can transiently
// hit torn-down / streaming geometry (LOD, animated water) whose child pointers are
// momentarily garbage — an access violation there would crash the game. On a fault we
// discard the partial result and keep going; the periodic rebuild retries shortly.
void VirtualShadowMaps::RebuildRegistrySafe(RE::NiAVObject* a_obj)
{
	__try {
		registrySeen.clear();
		rejNoRD = rejNoVB = rejNoIB = rejZeroTris = rejDup = 0;
		RebuildRegistry(a_obj, 0);   // ShadowSceneNode (sky/LOD skipped) — mostly sky in interiors
		RebuildFromReferences();     // player + loaded-cell references: room shell, clutter, NPCs, player
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		registry.clear();
		registrySeen.clear();
		logger::warn("VSM: registry rebuild faulted on streaming geometry; skipped this rebuild");
	}
}

// Gather geometry from the loaded references (the room shell, furniture, NPCs are references, NOT
// under the ShadowSceneNode as BSTriShapes) and the player's own 3D. Each reference's Get3D() is a
// normal node tree of BSTriShapes we can capture. Deduped against everything already registered.
void VirtualShadowMaps::RebuildFromReferences()
{
	auto* pc = RE::PlayerCharacter::GetSingleton();
	if (!pc)
		return;
	if (auto* p3d = pc->Get3D())      // the player character itself
		RebuildRegistry(p3d, 0);
	if (auto* cell = pc->GetParentCell()) {
		cell->ForEachReference([this](RE::TESObjectREFR* a_ref) {
			if (a_ref) {
				if (auto* r3d = a_ref->Get3D())
					RebuildRegistry(r3d, 0);
			}
			return RE::BSContainer::ForEachResult::kContinue;
		});
	}
}

// SEH-guarded scene census (like RebuildRegistrySafe — a live scene-graph walk can hit torn-down
// geometry). Logs what our traversal reaches from `a_root`, total and per direct child, so we can
// see whether the room geometry is even under this root and whether we're missing geometry types.
void VirtualShadowMaps::DumpSceneCensus(RE::NiAVObject* a_root, const char* a_label)
{
	__try {
		SceneCensus total{};
		CensusWalk(a_root, 0, total);
		logger::info("scene census ({}): triShapes={} otherLeaves={} nodes={} maxDepth={} lodSkipped={}  (triShapes ~= registry; otherLeaves>>triShapes => we skip most geometry types)",
		    a_label, total.triShapes, total.otherLeaves, total.nodes, total.maxDepth, total.lodSkipped);
		if (auto* rn = a_root ? a_root->AsNode() : nullptr) {
			int idx = 0;
			for (auto& ch : rn->GetChildren()) {
				if (!ch)
					continue;
				SceneCensus cc{};
				CensusWalk(ch.get(), 0, cc);
				logger::info("  {} child[{}] '{}': triShapes={} otherLeaves={} nodes={} maxDepth={} otherType='{}'",
				    a_label, idx, ch->name.c_str(), cc.triShapes, cc.otherLeaves, cc.nodes, cc.maxDepth,
				    cc.sampleOther ? cc.sampleOther : "-");
				++idx;
			}
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		logger::warn("VSM census: faulted walking the scene graph");
	}
}

// Count the player's own triShapes and how many are in OUR caster registry (is the player captured?).
void VirtualShadowMaps::PlayerWalk(RE::NiAVObject* a_o, int a_depth, int& a_tris, int& a_inReg, int& a_other)
{
	if (!a_o || a_depth > 128)
		return;
	if (auto* tri = a_o->AsTriShape()) {
		++a_tris;
		bool inReg = false;
		for (const auto& e : registry)
			if (e.geom.get() == static_cast<RE::BSGeometry*>(tri)) { inReg = true; ++a_inReg; break; }
		// Log WHY each mesh is / isn't captured, so we can handle skinned body meshes precisely.
		auto* rd = tri->GetGeometryRuntimeData().rendererData;
		const uint32_t triCount = tri->GetTrishapeRuntimeData().triangleCount;
		logger::info("    mesh '{}' type={} vb={} ib={} tris={} stride={} inReg={}",
		    a_o->name.c_str(), tri->GetRTTI() ? tri->GetRTTI()->name : "?",
		    rd && rd->vertexBuffer ? 1 : 0, rd && rd->indexBuffer ? 1 : 0, triCount,
		    rd ? rd->vertexDesc.GetSize() : 0, inReg ? 1 : 0);
	} else if (!a_o->AsNode()) {
		++a_other;
	}
	if (auto* node = a_o->AsNode())
		for (auto& ch : node->GetChildren())
			PlayerWalk(ch.get(), a_depth + 1, a_tris, a_inReg, a_other);
}

void VirtualShadowMaps::DumpPlayerDiag(RE::NiAVObject* a_p3d)
{
	__try {
		int tris = 0, inReg = 0, other = 0;
		PlayerWalk(a_p3d, 0, tris, inReg, other);
		logger::info("PLAYER geometry: triShapes={} inOurRegistry={} otherLeaves={}  (inRegistry==0 => player not captured => casts no shadow; only ONE of several occluder classes)",
		    tris, inReg, other);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		logger::warn("VSM player diag: faulted");
	}
}

void VirtualShadowMaps::RebuildRegistry(RE::NiAVObject* a_obj, int a_depth)
{
	if (!a_obj || a_depth > 64)  // depth cap guards against scene-graph cycles
		return;
	if (registry.size() >= kMaxCasters)  // hard cap (huge cell / runaway guard)
		return;

	// Skip subtrees that are NOT valid local-shadow casters:
	//  - "Sky": the skybox (AtmosphereDome/Galaxy/Constellations/celestial dome) is CAMERA-ATTACHED
	//    (its world.translate.xy == cameraPos.xy), so capturing it makes shadows slide with the camera.
	//  - "Weather"/"LODRoot"/"ObjectLODRoot": weather effects and distant LOD, no useful local shadow.
	if (const char* nm = a_obj->name.c_str(); nm) {
		if (std::strcmp(nm, "ObjectLODRoot") == 0 || std::strcmp(nm, "Sky") == 0 ||
		    std::strcmp(nm, "Weather") == 0 || std::strcmp(nm, "LODRoot") == 0)
			return;
	}

	if (auto* tri = a_obj->AsTriShape()) {
		auto& grt = tri->GetGeometryRuntimeData();
		auto* rd = grt.rendererData;
		const uint32_t triCount = tri->GetTrishapeRuntimeData().triangleCount;
		if (!rd) {
			// No renderer buffers = skinned/dynamic geometry (the character body/head/armor). Our
			// custom depth VS can't draw it, but the ENGINE path (BSUtilityShader) can — it draws from
			// the BSGeometry + its shaderProperty and skins internally. Capture it as engine-only so the
			// engine path has characters to draw; the custom-VS path skips engineOnly (no buffers).
			if (grt.shaderProperty && registrySeen.insert(static_cast<RE::BSGeometry*>(tri)).second) {
				CasterEntry e;
				e.geom       = RE::NiPointer<RE::BSGeometry>(tri);
				e.engineOnly = true;  // vertexStride/indexCount stay 0
				registry.push_back(std::move(e));
			} else {
				++rejNoRD;
			}
		}
		else if (!rd->vertexBuffer)                                             ++rejNoVB;
		else if (!rd->indexBuffer)                                              ++rejNoIB;
		else if (!triCount)                                                     ++rejZeroTris;
		else if (!registrySeen.insert(static_cast<RE::BSGeometry*>(tri)).second) ++rejDup;
		else {
			CasterEntry e;
			e.geom         = RE::NiPointer<RE::BSGeometry>(tri);  // add-ref: keep alive between rebuilds
			e.vertexStride = rd->vertexDesc.GetSize();
			e.indexCount   = triCount * 3u;
			registry.push_back(std::move(e));
		}
	}
	if (auto* node = a_obj->AsNode()) {
		for (auto& child : node->GetChildren())
			RebuildRegistry(child.get(), a_depth + 1);
	}
}

namespace
{
	constexpr std::uint32_t kMaxSkinVerts = 262144;  // fixed CPU-skin buffers (no realloc during fill)
	constexpr std::uint32_t kMaxSkinIdx   = 524288;
}

// SEH-isolated (POD-only locals) so __try is legal: skin ONE caster's clean BSTriShape partitions into
// pre-allocated arrays, bounds-checked. A torn-down streaming partition faults here and is caught, leaving
// partial-but-safe results instead of a CTD. posed = Σ weightᵢ · boneMat3x4[remap[idxᵢ]] · bindPos, and the
// engine matrices output WORLD-absolute (verified 0.9.30). Only clean float3 layouts (posOff/skinOff+12
// fit the stride); BSDynamicTriShape split/dynamic streams (head/hair) are handled later.
static void VSM_SkinOneCaster(RE::BSGeometry* a_g, DirectX::XMFLOAT3* a_posed, std::uint32_t* a_idx,
    std::uint32_t a_maxV, std::uint32_t a_maxI, std::uint32_t& a_vCount, std::uint32_t& a_iCount)
{
	using namespace DirectX::PackedVector;
	using V = RE::BSGraphics::Vertex;
	__try {
		auto* skin = a_g->GetGeometryRuntimeData().skinInstance.get();
		if (!skin || !skin->boneMatrices || !skin->skinPartition || skin->numMatrices == 0)
			return;
		auto* sp = skin->skinPartition.get();
		const float* boneMats = reinterpret_cast<const float*>(skin->boneMatrices);
		const int matStrideF = static_cast<int>(skin->allocatedSize / skin->numMatrices) / 4;  // 12 floats = 3x4
		if (matStrideF < 12)
			return;
		for (std::uint32_t pi = 0; pi < sp->numPartitions; ++pi) {
			auto& p = sp->partitions[pi];
			auto* bd = p.buffData;
			if (!bd || !bd->rawVertexData || !p.triList || !p.bones)
				continue;
			auto& vd = p.vertexDesc;
			const std::uint32_t stride  = vd.GetSize();
			const std::uint32_t posOff  = vd.GetAttributeOffset(V::VA_POSITION);
			const std::uint32_t skinOff = vd.GetAttributeOffset(V::VA_SKINNING);
			if (posOff + 12 > stride || skinOff + 12 > stride)
				continue;  // not the clean float3 layout (head/hair dynamic stream) — skip for now
			const std::uint8_t* rv = reinterpret_cast<const std::uint8_t*>(bd->rawVertexData);
			const std::uint16_t* remap = p.bones;
			const std::uint32_t partBase = a_vCount;  // where this partition's verts start in the global buffer
			for (std::uint16_t v = 0; v < p.vertices; ++v) {
				if (a_vCount >= a_maxV)
					return;
				const std::uint8_t* vb = rv + static_cast<size_t>(v) * stride;
				const float* bp = reinterpret_cast<const float*>(vb + posOff);
				const std::uint16_t* w = reinterpret_cast<const std::uint16_t*>(vb + skinOff);
				const std::uint8_t* ib = reinterpret_cast<const std::uint8_t*>(vb + skinOff + 8);
				float wx = 0.0f, wy = 0.0f, wz = 0.0f;
				for (int b = 0; b < 4; ++b) {
					const float wt = XMConvertHalfToFloat(w[b]);
					if (wt <= 0.0f)
						continue;
					const float* M = boneMats + static_cast<size_t>(remap[ib[b]]) * matStrideF;  // 3x4 row-major
					wx += wt * (M[0] * bp[0] + M[1] * bp[1] + M[2] * bp[2] + M[3]);
					wy += wt * (M[4] * bp[0] + M[5] * bp[1] + M[6] * bp[2] + M[7]);
					wz += wt * (M[8] * bp[0] + M[9] * bp[1] + M[10] * bp[2] + M[11]);
				}
				a_posed[a_vCount++] = { wx, wy, wz };
			}
			const std::uint16_t* tl = p.triList;
			const std::uint32_t nidx = static_cast<std::uint32_t>(p.triangles) * 3u;
			for (std::uint32_t k = 0; k < nidx; ++k) {
				if (a_iCount >= a_maxI)
					return;
				a_idx[a_iCount++] = partBase + tl[k];
			}
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		// partial results already written; the caller's range reflects a_iCount at return.
	}
}

// Path B: CPU-skin every engineOnly BSTriShape caster into the shared world-absolute posed buffer, once
// per frame (skin-once, rasterize into every light face). Buffers are fixed-size so the SEH helper writes
// via raw pointers without reallocation. GPU upload only the used prefix.
void VirtualShadowMaps::SkinAllCasters()
{
	skinnedRanges.clear();
	if (skinPosed.size() < kMaxSkinVerts)
		skinPosed.resize(kMaxSkinVerts);
	if (skinIndices.size() < kMaxSkinIdx)
		skinIndices.resize(kMaxSkinIdx);

	std::uint32_t vCount = 0, iCount = 0;
	for (const auto& e : registry) {
		if (!e.engineOnly)
			continue;
		RE::BSGeometry* g = e.geom.get();
		if (!g)
			continue;
		const std::uint32_t i0 = iCount;
		VSM_SkinOneCaster(g, skinPosed.data(), skinIndices.data(), kMaxSkinVerts, kMaxSkinIdx, vCount, iCount);
		if (iCount > i0)
			skinnedRanges.push_back({ i0, iCount - i0, g->worldBound.center, g->worldBound.radius });
	}

	if (vCount == 0 || !device)
		return;
	const std::uint32_t vbBytes = vCount * static_cast<std::uint32_t>(sizeof(DirectX::XMFLOAT3));
	const std::uint32_t ibBytes = iCount * static_cast<std::uint32_t>(sizeof(std::uint32_t));
	if (!skinnedVB || skinnedVBCap < vbBytes) {
		skinnedVB.Reset();
		D3D11_BUFFER_DESC bd{};
		bd.ByteWidth = vbBytes + 65536;
		bd.Usage = D3D11_USAGE_DYNAMIC;
		bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		device->CreateBuffer(&bd, nullptr, &skinnedVB);
		skinnedVBCap = bd.ByteWidth;
	}
	if (!skinnedIB || skinnedIBCap < ibBytes) {
		skinnedIB.Reset();
		D3D11_BUFFER_DESC bd{};
		bd.ByteWidth = ibBytes + 65536;
		bd.Usage = D3D11_USAGE_DYNAMIC;
		bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		device->CreateBuffer(&bd, nullptr, &skinnedIB);
		skinnedIBCap = bd.ByteWidth;
	}
	D3D11_MAPPED_SUBRESOURCE m{};
	if (skinnedVB && SUCCEEDED(context->Map(skinnedVB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		std::memcpy(m.pData, skinPosed.data(), vbBytes);
		context->Unmap(skinnedVB.Get(), 0);
	}
	if (skinnedIB && SUCCEEDED(context->Map(skinnedIB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		std::memcpy(m.pData, skinIndices.data(), ibBytes);
		context->Unmap(skinnedIB.Get(), 0);
	}
}

void VirtualShadowMaps::RenderDepth()
{
	visibleCasters = 0;
	if (!dsv)
		return;

	GraphicsStateGuard guard(context);

	ID3D11RenderTargetView* noRTV = nullptr;
	context->OMSetRenderTargets(1, &noRTV, dsv.Get());
	context->ClearDepthStencilView(dsv.Get(), D3D11_CLEAR_DEPTH, kReverseZ ? 0.0f : 1.0f, 0);

	// No local shadow light picked (e.g. outdoors in daylight — the sun is a directional
	// source handled by M5, not here) or nothing loaded: leave the map cleared so the
	// preview shows black instead of freezing on a stale frame.
	if (!haveTestLight || registry.empty())
		return;

	// Shared pipeline state for all 6 faces.
	context->RSSetState(rasterState.Get());
	context->OMSetDepthStencilState(depthState.Get(), 0);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->VSSetShader(depthVS.Get(), nullptr, 0);
	context->PSSetShader(nullptr, nullptr, 0);
	context->IASetInputLayout(ilFull.Get());  // position is always R32G32B32_FLOAT
	auto* cb = perDrawCB.Get();
	context->VSSetConstantBuffers(0, 1, &cb);

	// Skyrim rebases renderable geometry (geom->world) to CAMERA-RELATIVE space, but our light
	// cubes are built from ABSOLUTE light positions and the shader samples with absolute world
	// positions. Add the scene camera offset so casters render in the SAME absolute space:
	//   clip = modelPos * World * T(cameraPos) * cubeVP.
	const bool toAbs = dbgCastersAbsolute;
	// Offset casters into absolute world by the RELIABLE scene camera (ShadowSceneNode::cameraPos),
	// so the atlas shares one absolute frame with the absolute light cubes and the absP sampling.
	// NOT posAdjust.getEye() — that returns garbage (0,0,1) intermittently and mis-places the atlas.
	const RE::NiPoint3 eye{ sceneCameraPos.x, sceneCameraPos.y, sceneCameraPos.z };
	const XMMATRIX camT = toAbs ? XMMatrixTranslation(eye.x, eye.y, eye.z) : XMMatrixIdentity();

	// Path B: skin all characters ONCE this frame into a world-absolute posed buffer (skin-once,
	// rasterize into every light face below). Cheap no-op if there are no skinned casters.
	SkinAllCasters();

	// Every light = 6 cube faces rendered into its atlas block. NOTE (M2/M4): rendering every
	// light's full cube every frame is the dominant cost -- becomes GPU-driven indirect +
	// static/dynamic caching once the base is correct.
	for (const auto& lr : lightRecords) {
		const int blockX = static_cast<int>(lr.atlasCol) * kBlockW;
		const int blockY = static_cast<int>(lr.atlasRow) * kBlockH;

		for (int f = 0; f < 6; ++f) {
			const int col = f % 3, row = f / 3;
			const D3D11_VIEWPORT vp{ (float)(blockX + col * kFaceRes), (float)(blockY + row * kFaceRes),
				(float)kFaceRes, (float)kFaceRes, 0.0f, 1.0f };
			context->RSSetViewports(1, &vp);
			const XMMATRIX faceVP = XMLoadFloat4x4(&lr.cubeVP[f]);
			const XMMATRIX drawVP = XMMatrixMultiply(camT, faceVP);  // T(cameraPos) * cubeVP (casters -> absolute)
			XMFLOAT4 planes[6];
			ExtractFrustumPlanes(lr.cubeVP[f], planes);  // absolute-space frustum (matches drawVP output)

			for (size_t i = 0; i < registry.size(); ++i) {
				if (isolateCaster > 0 && static_cast<int>(i) != isolateCaster - 1)
					continue;  // debug: render only one mesh
				RE::BSGeometry* geom = registry[i].geom.get();
				if (!geom)
					continue;
				auto* rd = geom->GetGeometryRuntimeData().rendererData;
				if (!rd || !rd->vertexBuffer || !rd->indexBuffer)
					continue;  // streamed out / buffers freed -> skip safely

				const auto& wb = geom->worldBound;
				RE::NiPoint3 cc = wb.center;
				if (toAbs) { cc.x += eye.x; cc.y += eye.y; cc.z += eye.z; }  // cull in absolute space
				if (frustumCull && wb.radius > 0.0f && !SphereInFrustum(planes, cc, wb.radius))
					continue;

				const XMMATRIX wvp = XMMatrixMultiply(NiTransformToXM(geom->world), drawVP);
				D3D11_MAPPED_SUBRESOURCE mapped{};
				if (SUCCEEDED(context->Map(perDrawCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped))) {
					XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(mapped.pData), wvp);
					context->Unmap(perDrawCB.Get(), 0);
				}
				auto* vb = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
				UINT stride = registry[i].vertexStride, offset = 0;
				context->IASetVertexBuffers(0, 1, &vb, &stride, &offset);
				context->IASetIndexBuffer(reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer), DXGI_FORMAT_R16_UINT, 0);
				context->DrawIndexed(registry[i].indexCount, 0, 0);
				++visibleCasters;
			}

			// --- Path B: skinned characters (world-absolute posed buffer -> same cube VP, world=identity) ---
			if (skinnedVB && skinnedIB && !skinnedRanges.empty()) {
				D3D11_MAPPED_SUBRESOURCE sm{};
				if (SUCCEEDED(context->Map(perDrawCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &sm))) {
					XMStoreFloat4x4(reinterpret_cast<XMFLOAT4X4*>(sm.pData), drawVP);  // posed verts ARE world -> WVP = cube VP
					context->Unmap(perDrawCB.Get(), 0);
				}
				ID3D11Buffer* svb = skinnedVB.Get();
				UINT sstride = static_cast<UINT>(sizeof(XMFLOAT3)), soff = 0;
				context->IASetVertexBuffers(0, 1, &svb, &sstride, &soff);
				context->IASetIndexBuffer(skinnedIB.Get(), DXGI_FORMAT_R32_UINT, 0);
				for (const auto& r : skinnedRanges) {
					if (frustumCull && r.radius > 0.0f && !SphereInFrustum(planes, r.center, r.radius))
						continue;
					context->DrawIndexed(r.indexCount, r.ibStart, 0);
					++visibleCasters;
				}
			}
		}
	}
	// engine state restored by ~GraphicsStateGuard
}

void VirtualShadowMaps::ResolvePreview()
{
	if (!previewRTV || !fullscreenVS || !linearizePS || !srv)
		return;

	GraphicsStateGuard guard(context);

	ID3D11RenderTargetView* rtv = previewRTV.Get();
	context->OMSetRenderTargets(1, &rtv, nullptr);
	D3D11_VIEWPORT vp{ 0, 0, (float)kAtlasW, (float)kAtlasH, 0, 1 };
	context->RSSetViewports(1, &vp);
	context->OMSetDepthStencilState(nullptr, 0);
	context->RSSetState(nullptr);
	context->IASetInputLayout(nullptr);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->VSSetShader(fullscreenVS.Get(), nullptr, 0);
	context->PSSetShader(linearizePS.Get(), nullptr, 0);

	ID3D11ShaderResourceView* depthIn = srv.Get();
	ID3D11SamplerState*       samp    = pointSampler.Get();
	context->PSSetShaderResources(0, 1, &depthIn);
	context->PSSetSamplers(0, 1, &samp);
	if (previewCB) {
		D3D11_MAPPED_SUBRESOURCE m{};
		if (SUCCEEDED(context->Map(previewCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
			*reinterpret_cast<float*>(m.pData) = previewRange;
			context->Unmap(previewCB.Get(), 0);
		}
		ID3D11Buffer* pcbb = previewCB.Get();
		context->PSSetConstantBuffers(0, 1, &pcbb);
	}
	context->Draw(3, 0);

	// unbind our depth SRV from the PS input slot
	ID3D11ShaderResourceView* nullSRV = nullptr;
	context->PSSetShaderResources(0, 1, &nullSRV);
	// engine state restored by ~GraphicsStateGuard
}

void VirtualShadowMaps::InspectCaster()
{
	dbgHaveDump = false;
	if (registry.empty() || !device || !context)
		return;
	const size_t sel = (isolateCaster > 0 && isolateCaster <= static_cast<int>(registry.size())) ? isolateCaster - 1 : 0;
	RE::BSGeometry* geom = registry[sel].geom.get();
	if (!geom)
		return;
	auto* rd = geom->GetGeometryRuntimeData().rendererData;
	if (!rd || !rd->vertexBuffer || !rd->indexBuffer || registry[sel].vertexStride == 0)
		return;
	dbgStride     = registry[sel].vertexStride;
	dbgFullPrec   = rd->vertexDesc.HasFlag(RE::BSGraphics::Vertex::VF_FULLPREC);
	dbgIndexCount = registry[sel].indexCount;
	auto* vbuf = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
	auto* ibuf = reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer);

	// Copy the head of a GPU buffer into a CPU-readable staging buffer.
	auto readback = [&](ID3D11Buffer* src, UINT wanted, std::vector<uint8_t>& out) -> bool {
		D3D11_BUFFER_DESC bd{};
		src->GetDesc(&bd);
		const UINT n = (std::min)(wanted, bd.ByteWidth);
		if (n == 0)
			return false;
		D3D11_BUFFER_DESC sd{};
		sd.ByteWidth = n;
		sd.Usage = D3D11_USAGE_STAGING;
		sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ComPtr<ID3D11Buffer> staging;
		if (FAILED(device->CreateBuffer(&sd, nullptr, &staging)))
			return false;
		D3D11_BOX box{ 0, 0, 0, n, 1, 1 };
		context->CopySubresourceRegion(staging.Get(), 0, 0, 0, 0, src, 0, &box);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m)))
			return false;
		out.assign(static_cast<const uint8_t*>(m.pData), static_cast<const uint8_t*>(m.pData) + n);
		context->Unmap(staging.Get(), 0);
		return true;
	};

	std::vector<uint8_t> vb;
	if (readback(vbuf, dbgStride * 4 + 16, vb)) {
		for (int i = 0; i < 4; ++i) {
			const size_t off = static_cast<size_t>(i) * dbgStride;
			if (off + 12 > vb.size()) {
				dbgV[i] = {};
				continue;
			}
			const float* f = reinterpret_cast<const float*>(vb.data() + off);  // position: float3 @ offset 0
			dbgV[i] = { f[0], f[1], f[2] };
		}
	}

	std::vector<uint8_t> ib;
	if (readback(ibuf, 12, ib)) {
		const uint16_t* ix = reinterpret_cast<const uint16_t*>(ib.data());
		for (int i = 0; i < 6 && (i * 2 + 2) <= static_cast<int>(ib.size()); ++i)
			dbgIdx[i] = ix[i];
	}

	dbgHaveDump = true;
}

void VirtualShadowMaps::RenderFrame()
{
	if (!resourcesReady || !context)
		return;
	++frameIndex;  // per-tick counter for the CS handshake diagnostic
	if (!enabled) {
		registry.clear();      // release held geometry refs while the feature is off
		registryDirty = true;  // force a fresh rebuild when re-enabled
		return;
	}

	const bool menuOpen = menuVisible;
	menuVisible = false;  // DrawMenu re-sets this next time the menu is drawn

	auto& smState = RE::BSShaderManager::State::GetSingleton();  // returns a reference
	auto* ssn = smState.shadowSceneNode[0];

	// Persistent registry: re-traverse the scene graph only when dirty or on the periodic
	// refresh; otherwise reuse the cached geometry list (buffers/transforms are re-read
	// per frame in RenderDepth, so this just catches geometry streaming in/out).
	if (registryDirty || ++framesSinceRebuild >= kRebuildInterval) {
		registry.clear();
		RebuildRegistrySafe(static_cast<RE::NiAVObject*>(ssn));
		registryDirty = false;
		framesSinceRebuild = 0;
	}

	CollectLights(ssn);      // gather all active local lights -> lightRecords + assign atlas blocks
	UploadLightBuffer();     // mirror lightRecords into the GPU buffer for the LLF shader (phase 3)
	UpdateDebugCB();         // push live tuning sliders (bias/near/far/mode) to the shader
	RenderDepth();           // render every light's cube into the atlas (clears if no lights)

	if (dbgLogRequested) {   // on-demand numeric dump (menu button) — after lights are collected
		dbgLogRequested = false;
		DumpDiagnosticLog();
		dumpConfirmFrames = 180;  // ~3s of "dump written" confirmation in the menu
	}

	// Debug-only work — skipped entirely unless the Debug section is open and on screen.
	if (showDebug && menuOpen) {
		ComputeCasterBounds();
		if (dbgDumpRequested) {
			dbgDumpRequested = false;
			InspectCaster();
		}
		ResolvePreview();
	}
}

// AABB of collected caster world positions (debug sanity-check of the transforms).
void VirtualShadowMaps::ComputeCasterBounds()
{
	if (registry.empty()) {
		dbgCasterMin = dbgCasterMax = {};
		return;
	}
	dbgCasterMin = { FLT_MAX, FLT_MAX, FLT_MAX };
	dbgCasterMax = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (const auto& e : registry) {
		RE::BSGeometry* geom = e.geom.get();
		if (!geom)
			continue;
		const auto& t = geom->world.translate;
		dbgCasterMin.x = (std::min)(dbgCasterMin.x, t.x);
		dbgCasterMin.y = (std::min)(dbgCasterMin.y, t.y);
		dbgCasterMin.z = (std::min)(dbgCasterMin.z, t.z);
		dbgCasterMax.x = (std::max)(dbgCasterMax.x, t.x);
		dbgCasterMax.y = (std::max)(dbgCasterMax.y, t.y);
		dbgCasterMax.z = (std::max)(dbgCasterMax.z, t.z);
	}
}

// Dump the coordinate-space vectors that decide the mode-0 light match, so we can verify the
// math from real numbers instead of RGB. The match works iff the eye LLF used to make the shader
// Skinned-geometry probe (Path B foundation). For every engineOnly (skinned) caster, log what data is
// ACTUALLY reachable at our Present-time tick — so we build the compute-skinning path from real values,
// not assumptions. CPU-skinned (BSDynamicTriShape) exposes posed verts in dynamicData; GPU-skinned
// (BSTriShape + skinInstance) exposes bind pose + bones/weights in the skin partition. SEH-guarded.
void VirtualShadowMaps::DumpSkinnedGeometry()
{
	using V = RE::BSGraphics::Vertex;
	__try {
		logger::info("--- skinned-geometry probe: FULL Path B input set (buffers, formats, bones, indices) ---");
		int n = 0, cpuSkin = 0, dynValid = 0, gpuSkin = 0, boneMatsValid = 0, partVBValid = 0;
		bool didContent = false;     // one-shot: raw bytes of the first BSTriShape skinned caster (body math)
		bool didDynContent = false;  // one-shot: raw dynamicData bytes of the first BSDynamicTriShape (head/hair format)
		int  spaceWorld = 0, spaceCamRel = 0, spaceModel = 0, spaceOther = 0;  // per-caster skinning-space classification
		for (const auto& e : registry) {
			if (!e.engineOnly)
				continue;
			RE::BSGeometry* g = e.geom.get();
			if (!g)
				continue;
			auto& grd  = g->GetGeometryRuntimeData();
			auto* skin = grd.skinInstance.get();
			const char* type = g->GetRTTI() ? g->GetRTTI()->name : "?";
			const auto& t = g->world.translate;
			auto& vd = grd.vertexDesc;
			const std::uint32_t stride = vd.GetSize();

			// --- CPU-skinned (already-posed verts) ---
			void* dynData = nullptr;
			std::uint32_t dynSize = 0;
			if (auto* dts = g->AsDynamicTriShape()) {
				auto& dr = dts->GetDynamicTrishapeRuntimeData();
				dynData = dr.dynamicData;
				dynSize = dr.dataSize;
				++cpuSkin;
				if (dynData && dynSize)
					++dynValid;
				if (!didDynContent && dynData && dynSize >= 24) {
					didDynContent = true;
					const std::uint8_t* d = reinterpret_cast<const std::uint8_t*>(dynData);
					const std::uint32_t st = vd.GetSize();
					logger::info("  skinDYN '{}' dynData first24B: {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x}",
					    g->name.c_str(), d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15],d[16],d[17],d[18],d[19],d[20],d[21],d[22],d[23]);
					logger::info("  skinDYN '{}' geomVd[stride={} posOff={} skinOff={} flags={:#x} fullprec={}] dynVerts={} (dynSize={})",
					    g->name.c_str(), st, static_cast<int>(vd.GetAttributeOffset(V::VA_POSITION)), static_cast<int>(vd.GetAttributeOffset(V::VA_SKINNING)),
					    static_cast<std::uint32_t>(vd.GetFlags()), vd.HasFlag(V::VF_FULLPREC) ? 1 : 0, st ? dynSize / st : 0, dynSize);
				}
			}

			// --- skin instance: the engine's PRECOMPUTED palette + bind-pose partition ---
			int mats = 0, sdBones = 0, pN = 0, pV = 0;
			std::uint32_t allocSz = 0;
			void* boneMats = nullptr; void* boneWT = nullptr; void* bonesArr = nullptr; void* boneData = nullptr;
			int p0v = 0, p0tri = 0, p0nb = 0, p0bpv = 0, p0stride = 0, p0posOff = 0, p0skinOff = 0;
			void *p0vb = nullptr, *p0ib = nullptr, *p0rawV = nullptr, *p0rawI = nullptr, *p0triList = nullptr, *p0remap = nullptr;
			if (skin) {
				++gpuSkin;
				mats     = static_cast<int>(skin->numMatrices);
				allocSz  = skin->allocatedSize;
				boneMats = skin->boneMatrices;
				boneWT   = skin->boneWorldTransforms;
				bonesArr = skin->bones;
				if (boneMats)
					++boneMatsValid;
				if (auto* sd = skin->skinData.get()) {
					sdBones  = static_cast<int>(sd->bones);
					boneData = sd->boneData;
				}
				if (auto* sp = skin->skinPartition.get()) {
					pN = static_cast<int>(sp->numPartitions);
					pV = static_cast<int>(sp->vertexCount);
					if (sp->numPartitions > 0) {
						auto& p0 = sp->partitions[0];
						p0v = p0.vertices; p0tri = p0.triangles; p0nb = p0.numBones; p0bpv = p0.bonesPerVertex;
						p0stride  = static_cast<int>(p0.vertexDesc.GetSize());
						p0posOff  = static_cast<int>(p0.vertexDesc.GetAttributeOffset(V::VA_POSITION));
						p0skinOff = static_cast<int>(p0.vertexDesc.GetAttributeOffset(V::VA_SKINNING));
						p0triList = p0.triList;
						p0remap   = p0.bones;  // partition-local -> global bone index
						if (auto* bd = p0.buffData) {
							p0vb = bd->vertexBuffer; p0ib = bd->indexBuffer;
							p0rawV = bd->rawVertexData; p0rawI = bd->rawIndexData;
							if (p0vb)
								++partVBValid;
						}
					}
				}
			}

			// One-shot: dump the COMPLETE ingredient set to skin vertex 0 BY HAND and read off the output
			// space (compare the skinned result to worldBound.center=world vs cameraPos vs geom world). This
			// answers the one thing the math needs that pointers alone can't: what space boneMatrix*bindPos is in.
			if (!didContent && boneMats && mats > 0 && p0rawV && p0stride > 0) {
				didContent = true;
				const std::uint8_t* rv = reinterpret_cast<const std::uint8_t*>(p0rawV);
				const float* bm = reinterpret_cast<const float*>(boneMats);
				const int matStrideB = mats > 0 ? static_cast<int>(allocSz) / mats : 0;  // 64=4x4, 48=3x4
				const std::uint8_t* pos0  = rv + p0posOff;
				const std::uint8_t* skin0 = rv + p0skinOff;
				const auto& wc = g->worldBound.center;
				const auto& gt = g->world.translate;
				logger::info("  skinMATH '{}' matStrideBytes={} (64=>4x4,48=>3x4) numMats={} allocSz={} bindposFullprec={}",
				    g->name.c_str(), matStrideB, mats, allocSz, vd.HasFlag(V::VF_FULLPREC) ? 1 : 0);
				logger::info("  skinMATH '{}' vert0 pos @off{} 12B: {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} (asF3 {:.2f} {:.2f} {:.2f})",
				    g->name.c_str(), p0posOff, pos0[0],pos0[1],pos0[2],pos0[3],pos0[4],pos0[5],pos0[6],pos0[7],pos0[8],pos0[9],pos0[10],pos0[11],
				    reinterpret_cast<const float*>(pos0)[0], reinterpret_cast<const float*>(pos0)[1], reinterpret_cast<const float*>(pos0)[2]);
				logger::info("  skinMATH '{}' vert0 SKIN @off{} 16B: {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} {:02x}{:02x}{:02x}{:02x} (weights+indices packing)",
				    g->name.c_str(), p0skinOff, skin0[0],skin0[1],skin0[2],skin0[3],skin0[4],skin0[5],skin0[6],skin0[7],skin0[8],skin0[9],skin0[10],skin0[11],skin0[12],skin0[13],skin0[14],skin0[15]);
				// first 4 bone matrices (translation column, both conventions) so vertex 0's bones are covered
				for (int k = 0; k < 4 && k < mats; ++k) {
					const float* m = bm + (matStrideB / 4) * k;
					logger::info("  skinMATH '{}' boneMat[{}] 16f: [{:.3f} {:.3f} {:.3f} {:.3f}][{:.3f} {:.3f} {:.3f} {:.3f}][{:.3f} {:.3f} {:.3f} {:.3f}][{:.3f} {:.3f} {:.3f} {:.3f}]",
					    g->name.c_str(), k, m[0],m[1],m[2],m[3],m[4],m[5],m[6],m[7],m[8],m[9],m[10],m[11],m[12],m[13],m[14],m[15]);
				}
				if (p0remap) {
					const std::uint16_t* rm = reinterpret_cast<const std::uint16_t*>(p0remap);
					logger::info("  skinMATH '{}' partBoneRemap[0..7]: {} {} {} {} {} {} {} {}  (partition-local -> global)",
					    g->name.c_str(), rm[0],rm[1],rm[2],rm[3],rm[4],rm[5],rm[6],rm[7]);
				}
				logger::info("  skinMATH '{}' REFERENCE: worldBound.center=({:.1f} {:.1f} {:.1f}) cameraPos=({:.1f} {:.1f} {:.1f}) geomWorld.t=({:.1f} {:.1f} {:.1f})",
				    g->name.c_str(), wc.x, wc.y, wc.z, sceneCameraPos.x, sceneCameraPos.y, sceneCameraPos.z, gt.x, gt.y, gt.z);
			}

			{
				logger::info("  skinA '{}' type={} world=({:.0f} {:.0f} {:.0f}) vd[stride={} flags={:#x} posOff={} skinOff={} skinned={} fullprec={}] dyn[data={} size={} verts={}]",
				    g->name.c_str(), type, t.x, t.y, t.z, stride, static_cast<std::uint32_t>(vd.GetFlags()),
				    static_cast<int>(vd.GetAttributeOffset(V::VA_POSITION)), static_cast<int>(vd.GetAttributeOffset(V::VA_SKINNING)),
				    vd.HasFlag(V::VF_SKINNED) ? 1 : 0, vd.HasFlag(V::VF_FULLPREC) ? 1 : 0,
				    dynData ? 1 : 0, dynSize, stride ? dynSize / stride : 0);
				logger::info("  skinB '{}' inst[mats={} boneMats={} boneWT={} bones={}] data[bones={} boneData={}] part[n={} vcount={} p0(v={} tri={} nb={} bpv={} vb={} ib={} rawV={} rawI={} triList={} stride={} posOff={} skinOff={})]",
				    g->name.c_str(), mats, boneMats ? 1 : 0, boneWT ? 1 : 0, bonesArr ? 1 : 0, sdBones, boneData ? 1 : 0,
				    pN, pV, p0v, p0tri, p0nb, p0bpv, p0vb ? 1 : 0, p0ib ? 1 : 0, p0rawV ? 1 : 0, p0rawI ? 1 : 0,
				    p0triList ? 1 : 0, p0stride, p0posOff, p0skinOff);

				// SKINNING-SPACE CHECK (the key test for "moves with the player"): does bone matrix 0's origin
				// land at this caster's WORLD bound (=world-absolute, correct/fixed), or at world-minus-camera
				// (=camera-relative, drifts as you move), or near the origin (=model space)?
				if (boneMats && mats > 0) {
					const float* M = reinterpret_cast<const float*>(boneMats);
					const int msf = static_cast<int>(allocSz / static_cast<std::uint32_t>(mats)) / 4;
					if (msf >= 12) {
						const float bx = M[3], by = M[7], bz = M[11];
						const auto& wc = g->worldBound.center;
						const float ew  = std::sqrt((bx - wc.x) * (bx - wc.x) + (by - wc.y) * (by - wc.y) + (bz - wc.z) * (bz - wc.z));
						const float cx = wc.x - sceneCameraPos.x, cy = wc.y - sceneCameraPos.y, cz = wc.z - sceneCameraPos.z;
						const float ecr = std::sqrt((bx - cx) * (bx - cx) + (by - cy) * (by - cy) + (bz - cz) * (bz - cz));
						const float em  = std::sqrt(bx * bx + by * by + bz * bz);
						const char* cls = (ew < 150.0f) ? "WORLD" : (ecr < 150.0f) ? "CAMERA-REL(BUG)" : (em < 150.0f) ? "MODEL(BUG)" : "OTHER(BUG)";
						if (ew < 150.0f)       ++spaceWorld;
						else if (ecr < 150.0f) ++spaceCamRel;
						else if (em < 150.0f)  ++spaceModel;
						else                   ++spaceOther;
						logger::info("  skinSPACE '{}' boneMat0.t=({:.0f} {:.0f} {:.0f}) wbC=({:.0f} {:.0f} {:.0f}) errWorld={:.0f} errCamRel={:.0f} errModel={:.0f} => {}",
						    g->name.c_str(), bx, by, bz, wc.x, wc.y, wc.z, ew, ecr, em, cls);
					}
				}
			}
			++n;
		}
		logger::info("skinned probe SUMMARY: {} engineOnly | CPU-skinned={} (posed dynData valid={}) | GPU-skinned={} (engine boneMatrices valid={}, partition VB valid={})",
		    n, cpuSkin, dynValid, gpuSkin, boneMatsValid, partVBValid);
		logger::info("skinning SPACE SUMMARY: WORLD(correct)={} CAMERA-REL(drifts w/ player)={} MODEL={} OTHER={}  (any non-WORLD = the moving/detached shadows)",
		    spaceWorld, spaceCamRel, spaceModel, spaceOther);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		logger::warn("VSM skinned-geometry probe faulted");
	}
}

// light camera-relative (posAdjust.getEye == altEye) equals the eye the shader adds back. Our
// buffer stores rp = niLight->world.translate (absolute); LLF stores lightPosWS = rp - altEye.
// Full numeric proof of the shadow-sample math. For a selected light + the caster nearest it, this
// replicates the EXACT shader path (VSM.hlsli::GetLocalShadow) on the CPU against real surface
// vertices, and reads the atlas back so we can compare linPix vs linOcc per point. A vertex that
// was rasterized into the atlas MUST self-shadow to near-equality (linPix ≈ linOcc, diff ≈ 0). Any
// large divergence localizes the coordinate-space / projection bug with hard numbers — no visuals.
void VirtualShadowMaps::DumpDiagnosticLog()
{
	if (!device || !context) {
		logger::warn("VSM dump: no device/context");
		return;
	}

	RE::NiPoint3 camPos{};
	auto& smState = RE::BSShaderManager::State::GetSingleton();
	if (auto* ssn = smState.shadowSceneNode[0])
		camPos = ssn->GetRuntimeData().cameraPos;

	RE::NiPoint3 posAdj{};
	if (auto* rss = RE::BSGraphics::RendererShadowState::GetSingleton())
		posAdj = rss->GetRuntimeData().posAdjust.getEye();
	// The atlas eye is now ShadowSceneNode::cameraPos (reliable), NOT posAdjust (garbage-prone).
	// Everything below (probe camAdjust, matchEye, round-trip) uses this eye to mirror the atlas.
	const RE::NiPoint3 altEye = camPos;

	// CPU mirrors of the shader helpers (MUST match VSM.hlsli exactly).
	auto faceFromDir = [](float x, float y, float z) -> int {
		const float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
		if (ax >= ay && ax >= az) return x >= 0 ? 0 : 1;
		if (ay >= az)             return y >= 0 ? 2 : 3;
		return z >= 0 ? 4 : 5;
	};
	auto lin = [](float d, float n, float f) { return n * f / (std::max)(f - d * (f - n), 1e-4f); };
	static const char* kFaceName[6] = { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

	logger::info("==== VSM shadow-math dump (build {}) ====", kBuildTag);
	logger::info("atlas eye = cameraPos (ssn) = {:12.3f} {:12.3f} {:12.3f}   <- now the atlas offset (reliable)", camPos.x, camPos.y, camPos.z);
	logger::info("posAdjust.getEye()         = {:12.3f} {:12.3f} {:12.3f}   <- NO LONGER USED (was garbage-prone)", posAdj.x, posAdj.y, posAdj.z);
	logger::info("cameraPos - posAdjust      = {:12.3f} {:12.3f} {:12.3f}   (nonzero here = posAdjust was stale; proves why we switched)",
	    camPos.x - posAdj.x, camPos.y - posAdj.y, camPos.z - posAdj.z);

	// THE value we kept needing the probe for: read the REAL FrameBuffer::CameraPosAdjust straight
	// from the engine per-frame cbuffer (same address CS uses, RelocationID SE=524768/AE=411384).
	// Layout = 10x Matrix (640B) then float4 CameraPosAdjust @640, ... DynamicResolutionParams2 @704.
	// Now in EVERY dump — no probe, no arming. This directly confirms whether cameraPos == the shader eye.
	{
		static REL::Relocation<ID3D11Buffer**> perFrameReloc{ REL::RelocationID(524768, 411384) };
		ID3D11Buffer* pf = perFrameReloc.get() ? *perFrameReloc.get() : nullptr;
		if (pf) {
			D3D11_BUFFER_DESC bd{};
			pf->GetDesc(&bd);
			D3D11_BUFFER_DESC sd{};
			sd.ByteWidth = bd.ByteWidth;
			sd.Usage = D3D11_USAGE_STAGING;
			sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			ComPtr<ID3D11Buffer> st;
			if (bd.ByteWidth >= 720 && SUCCEEDED(device->CreateBuffer(&sd, nullptr, &st))) {
				context->CopyResource(st.Get(), pf);
				D3D11_MAPPED_SUBRESOURCE m{};
				if (SUCCEEDED(context->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) {
					const float* f = reinterpret_cast<const float*>(m.pData);
					const float cx = f[160], cy = f[161], cz = f[162];   // CameraPosAdjust  @ byte 640
					const float d2x = f[176], d2y = f[177];              // DynResParams2    @ byte 704
					logger::info("REAL FrameBuffer::CameraPosAdjust = {:12.3f} {:12.3f} {:12.3f}   (engine cbuffer — the shader's actual eye)", cx, cy, cz);
					logger::info("  >>> CameraPosAdjust - cameraPos(atlas eye) = {:.3f} {:.3f} {:.3f}   (~0 => atlas eye EXACT; nonzero => residual offset = shadows biased) <<<",
					    cx - camPos.x, cy - camPos.y, cz - camPos.z);
					logger::info("  DynResParams2 = {:.6f} {:.6f}  => render ~{}x{}, probe centre px ~({:.0f},{:.0f})",
					    d2x, d2y, d2x > 0 ? static_cast<int>(1.0f / d2x) : 0, d2y > 0 ? static_cast<int>(1.0f / d2y) : 0,
					    d2x > 0 ? 0.5f / d2x : 0.0f, d2y > 0 ? 0.5f / d2y : 0.0f);
					logger::info("  CameraPreviousPosAdjust = {:.3f} {:.3f} {:.3f}   FrameParams = {:.2f} {:.2f} {:.2f} {:.2f}  (FrameParams.y!=0 => first-person light path)",
					    f[164], f[165], f[166], f[168], f[169], f[170], f[171]);
					context->Unmap(st.Get(), 0);
				}
			} else {
				logger::warn("VSM dump: perFrame cbuffer unavailable or too small ({} B)", bd.ByteWidth);
			}
		} else {
			logger::warn("VSM dump: engine perFrame cbuffer pointer not resolved");
		}
	}

	// Camera POSE — position + orientation basis (columns of the camera node's world rotation =
	// local axes in world space). The reported bands are camera-tied, so logging WHERE the camera is
	// and WHICH WAY it looks lets us correlate shadow behaviour across captures. Skyrim's look
	// direction is typically the +Y column; all three axes are logged so it is unambiguous.
	if (auto* pcam = RE::PlayerCamera::GetSingleton(); pcam && pcam->cameraRoot) {
		const auto& ct = pcam->cameraRoot->world.translate;
		const auto& cr = pcam->cameraRoot->world.rotate;
		logger::info("camera pos   = {:12.3f} {:12.3f} {:12.3f}   (cameraRoot; cf. cameraPos above)", ct.x, ct.y, ct.z);
		logger::info("camera axesW  X(right)={:7.4f} {:7.4f} {:7.4f}  Y(fwd)={:7.4f} {:7.4f} {:7.4f}  Z(up)={:7.4f} {:7.4f} {:7.4f}",
		    cr.entry[0][0], cr.entry[1][0], cr.entry[2][0],
		    cr.entry[0][1], cr.entry[1][1], cr.entry[2][1],
		    cr.entry[0][2], cr.entry[1][2], cr.entry[2][2]);
	} else {
		logger::info("camera pose: PlayerCamera/cameraRoot unavailable");
	}
	logger::info("MODE={}  (0=REAL shadow; 1=off/lit; >=2=RGB debug — the shader casts NO shadow unless MODE==0)  enabled={}",
	    dbgMode, enabled ? 1 : 0);
	logger::info("bias(world)={:.2f} matchThresh={:.2f} FarScale={:.2f} NearFrac={:.3f} sampleSpace={} compareMode={} matchEye={}",
	    dbgBiasWorld, dbgMatchThresh, dbgFarScale, dbgNearFrac, dbgSampleSpace, dbgCompareMode, dbgMatchEye);
	logger::info("atlas {}x{}  kFaceRes={} kBlockW={} kBlockH={}  lights={} registry={} visibleCasters={}",
	    kAtlasW, kAtlasH, kFaceRes, kBlockW, kBlockH, lightRecords.size(), registry.size(), visibleCasters);
	logger::info("light filter: shadow-casters(added)={} ambient(skipped)={} nonShadow(skipped)={}  (only IsShadowLight()==true & !ambient cast shadows)",
	    lightRecords.size(), lightsAmbient, lightsNonShadow);
	logger::info("dump #{}  frame={}  CS resource-fetch: count={} lastFrame={} (frame-lastFrame={} -> 0/1 = CS binding our atlas NOW; large/stale = CS<->plugin handshake broken)",
	    ++dumpOrdinal, frameIndex, resourceFetchCount, lastResourceFetchFrame, frameIndex - lastResourceFetchFrame);
	logger::info("atlas cfg: reverseZ={} cubeFOV=90deg depthBias=100 slopeScaledBias=1.5 (atlas generation is standard-Z; see SetupResources)", kReverseZ);
	if (auto* ssn = smState.shadowSceneNode[0]) {
		auto& rt = ssn->GetRuntimeData();
		logger::info("light-set provenance: engine activeLights={} activeShadowLights={} -> we collected={} (cap {}); large drop => dedup/origin-skip/cap",
		    rt.activeLights.size(), rt.activeShadowLights.size(), lightRecords.size(), static_cast<int>(kMaxLights));
	}

	if (lightRecords.empty() || registry.empty()) {
		logger::info("(no lights or no casters this frame — nothing to prove)");
		logger::info("=========================================");
		return;
	}

	// ---- Registry health: valid buffers, triangle-count spread, absolute-space caster AABB. ----
	{
		int valid = 0; uint32_t triMin = 0xFFFFFFFFu, triMax = 0; uint64_t triSum = 0;
		XMFLOAT3 mn{ FLT_MAX, FLT_MAX, FLT_MAX }, mx{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
		for (const auto& e : registry) {
			RE::BSGeometry* g = e.geom.get();
			if (!g) continue;
			auto* grd = g->GetGeometryRuntimeData().rendererData;
			if (grd && grd->vertexBuffer && grd->indexBuffer) ++valid;
			const uint32_t tri = e.indexCount / 3u;
			triMin = (std::min)(triMin, tri); triMax = (std::max)(triMax, tri); triSum += tri;
			const auto& c = g->worldBound.center;  // already world-absolute (geom world.translate is world space)
			mn.x = (std::min)(mn.x, c.x); mn.y = (std::min)(mn.y, c.y); mn.z = (std::min)(mn.z, c.z);
			mx.x = (std::max)(mx.x, c.x); mx.y = (std::max)(mx.y, c.y); mx.z = (std::max)(mx.z, c.z);
		}
		const size_t n = registry.size();
		logger::info("registry: {} casters ({} with live buffers)  tris/caster min={} avg={} max={}  absAABB=[{:.0f} {:.0f} {:.0f}]..[{:.0f} {:.0f} {:.0f}]",
		    n, valid, triMin == 0xFFFFFFFFu ? 0 : triMin, n ? static_cast<uint32_t>(triSum / n) : 0, triMax,
		    mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
	}

	logger::info("capture rejections (BSTriShapes reached but NOT registered): noRendererData={} noVertexBuffer={} noIndexBuffer={} zeroTris={} duplicate={}",
	    rejNoRD, rejNoVB, rejNoIB, rejZeroTris, rejDup);

	// ---- Skinned-geometry probe (Path B): what buffers/data are ACTUALLY available at Present time for
	// the skinned casters, so we build the compute-skinning path from real data, not assumptions. ----
	DumpSkinnedGeometry();

	// ---- Scene-graph census: is the room geometry even under our traversal root, and are we
	// missing geometry types? (triShapes ~= registry means the root is genuinely this sparse.) ----
	DumpSceneCensus(static_cast<RE::NiAVObject*>(smState.shadowSceneNode[0]), "ssn");

	// ---- Player character (a controlled, known occluder): absolute position, bound, whether its
	// geometry is captured as a caster, and its distance to every collected light. This is ONE of
	// several occluder classes (walls/pillars are separate — checked via the registry list below). ----
	if (auto* pc = RE::PlayerCharacter::GetSingleton()) {
		if (auto* p3d = pc->Get3D()) {
			const auto& t = p3d->world.translate;   // world-absolute
			const auto& wb = p3d->worldBound;
			const float px = t.x, py = t.y, pz = t.z;
			logger::info("PLAYER abs pos={:.1f} {:.1f} {:.1f}  bound.c(abs)={:.1f} {:.1f} {:.1f} r={:.1f}",
			    px, py, pz, wb.center.x, wb.center.y, wb.center.z, wb.radius);
			DumpPlayerDiag(p3d);
			logger::info("PLAYER->light distances (for a controlled light-rig scene):");
			for (size_t i = 0; i < lightRecords.size(); ++i) {
				const auto& r = lightRecords[i];
				const float d = std::sqrt((r.positionWS.x - px) * (r.positionWS.x - px) +
				                          (r.positionWS.y - py) * (r.positionWS.y - py) +
				                          (r.positionWS.z - pz) * (r.positionWS.z - pz));
				logger::info("  L{:02d} at {:8.0f} {:8.0f} {:8.0f}  dist={:7.0f}  far={:7.0f} {}",
				    static_cast<int>(i), r.positionWS.x, r.positionWS.y, r.positionWS.z, d, r.farPlane,
				    d < r.farPlane ? "(player IN range)" : "(player beyond far)");
			}
		} else {
			logger::info("PLAYER: Get3D() null (not loaded)");
		}
	}

	// ---- REAL light-set / match audit: read the lights the SHADER actually shades (LLF's own
	// StrictLights cbuffer @b3 and clustered lights @t35, still bound from this frame's draws) and
	// check whether each lands within matchThresh of a light in OUR shadow buffer. If the room's
	// lights don't match our buffer, GetLocalShadow returns lit -> no shadow, regardless of atlas. ----
	{
		// absLight the shader reconstructs = light.positionWS + CameraPosAdjust (== cameraPos here).
		auto matchOurs = [&](float ax, float ay, float az, float& outD) -> int {
			int best = -1; float bd = 1e30f;
			for (size_t i = 0; i < lightRecords.size(); ++i) {
				const auto& r = lightRecords[i];
				const float d = std::sqrt((r.positionWS.x - ax) * (r.positionWS.x - ax) +
				                          (r.positionWS.y - ay) * (r.positionWS.y - ay) +
				                          (r.positionWS.z - az) * (r.positionWS.z - az));
				if (d < bd) { bd = d; best = static_cast<int>(i); }
			}
			outD = bd;
			return best;
		};
		auto readBufBytes = [&](ID3D11Resource* res, UINT wanted, std::vector<uint8_t>& out) -> bool {
			if (!res) return false;
			ComPtr<ID3D11Buffer> src;
			if (FAILED(res->QueryInterface(IID_PPV_ARGS(&src)))) return false;
			D3D11_BUFFER_DESC bd{}; src->GetDesc(&bd);
			const UINT n = (std::min)(wanted, bd.ByteWidth);
			if (!n) return false;
			D3D11_BUFFER_DESC sd{}; sd.ByteWidth = n; sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			ComPtr<ID3D11Buffer> st;
			if (FAILED(device->CreateBuffer(&sd, nullptr, &st))) return false;
			D3D11_BOX box{ 0, 0, 0, n, 1, 1 };
			context->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, src.Get(), 0, &box);
			D3D11_MAPPED_SUBRESOURCE m{};
			if (FAILED(context->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return false;
			out.assign(static_cast<const uint8_t*>(m.pData), static_cast<const uint8_t*>(m.pData) + n);
			context->Unmap(st.Get(), 0);
			return true;
		};

		// LLF Light struct = 80 bytes; positionWS (float4) at offset 32.
		auto auditLight = [&](const uint8_t* base, const char* tag, int idx) {
			const float* pw = reinterpret_cast<const float*>(base + 32);
			const float ax = pw[0] + camPos.x, ay = pw[1] + camPos.y, az = pw[2] + camPos.z;  // -> absolute
			const float* col = reinterpret_cast<const float*>(base + 0);
			const float radius = *reinterpret_cast<const float*>(base + 16);
			float d = 0; const int m = matchOurs(ax, ay, az, d);
			logger::info("  {}[{}] posWS={:8.1f} {:8.1f} {:8.1f} abs={:8.1f} {:8.1f} {:8.1f} r={:.0f} col=({:.2f} {:.2f} {:.2f}) -> ourBuf L{} dist={:.2f} {}",
			    tag, idx, pw[0], pw[1], pw[2], ax, ay, az, radius, col[0], col[1], col[2],
			    m, d, d < dbgMatchThresh ? "MATCH" : "NO-MATCH");
		};

		// StrictLights cbuffer @ b3: NumStrictLights (uint@0), Light StrictLights[15] starting @ byte 16.
		ID3D11Buffer* b3 = nullptr;
		context->PSGetConstantBuffers(3, 1, &b3);
		if (b3) {
			std::vector<uint8_t> buf;
			if (readBufBytes(b3, 16 + 80 * 15, buf) && buf.size() >= 16) {
				const uint32_t numStrict = *reinterpret_cast<const uint32_t*>(buf.data());
				logger::info("LLF StrictLights (b3): NumStrictLights={}  (these are the room's lights the shader shades)", numStrict);
				const int n = (std::min)(static_cast<int>(numStrict), 15);
				for (int i = 0; i < n && static_cast<size_t>(16 + i * 80 + 80) <= buf.size(); ++i)
					auditLight(buf.data() + 16 + i * 80, "strict", i);
			} else {
				logger::warn("LLF audit: b3 read failed/too small");
			}
			b3->Release();
		} else {
			logger::info("LLF audit: no cbuffer bound at b3 (can't read StrictLights)");
		}

		// Clustered lights @ t35 (StructuredBuffer<Light>): read the WHOLE buffer, log EVERY light.
		ID3D11ShaderResourceView* t35 = nullptr;
		context->PSGetShaderResources(35, 1, &t35);
		if (t35) {
			ComPtr<ID3D11Resource> res; t35->GetResource(&res);
			std::vector<uint8_t> buf;
			if (readBufBytes(res.Get(), 0xFFFFFFFFu, buf)) {  // wanted huge -> clamped to full buffer size
				const int n = static_cast<int>(buf.size() / 80);
				logger::info("LLF clustered lights (t35): {} lights (ALL)", n);
				for (int i = 0; i < n; ++i)
					auditLight(buf.data() + i * 80, "clust", i);
			} else {
				logger::info("LLF audit: t35 bound but read failed (maybe not a structured buffer this frame)");
			}
			t35->Release();
		} else {
			logger::info("LLF audit: no SRV bound at t35 (can't read clustered lights)");
		}
	}

	// ---- Camera proximity: how many captured casters are actually near the player (camera-relative
	// center magnitude = distance from camera). If ~0 are within a few hundred units, we're capturing
	// distant geometry but NOT the room the player is standing in. ----
	{
		int n300 = 0, n600 = 0, n1000 = 0;
		for (const auto& e : registry) {
			RE::BSGeometry* g = e.geom.get();
			if (!g) continue;
			const auto& c = g->worldBound.center;  // world-absolute
			const float d = std::sqrt((c.x - camPos.x) * (c.x - camPos.x) +
			                          (c.y - camPos.y) * (c.y - camPos.y) +
			                          (c.z - camPos.z) * (c.z - camPos.z));  // distance from camera
			if (d < 300.0f) ++n300;
			if (d < 600.0f) ++n600;
			if (d < 1000.0f) ++n1000;
		}
		logger::info("casters near camera: <300u={}  <600u={}  <1000u={}  (of {})", n300, n600, n1000, registry.size());
	}

	// ---- Registry contents: every captured caster with its NAME, so we can see WHAT geometry we
	// have (room shell vs clutter) and where it sits relative to the player/lights. ----
	logger::info("--- registry casters ({} total; abs center / radius / tris / stride / type / name) ---", registry.size());
	for (size_t i = 0; i < registry.size(); ++i) {
		RE::BSGeometry* g = registry[i].geom.get();
		if (!g) continue;
		const auto& c = g->worldBound.center;
		logger::info("  c{:03d} abs={:8.0f} {:8.0f} {:8.0f} r={:6.0f} tris={:5} stride={} type={} '{}'",
		    static_cast<int>(i), c.x, c.y, c.z, g->worldBound.radius,
		    registry[i].indexCount / 3u, registry[i].vertexStride,
		    g->GetRTTI() ? g->GetRTTI()->name : "?", g->name.c_str());
	}

	// ---- Read the whole depth atlas back to the CPU (R32 typeless -> float). ----
	std::vector<float> atlas;
	UINT atlasRowFloats = 0;
	bool haveAtlas = false;
	if (depthTex) {
		D3D11_TEXTURE2D_DESC td{};
		depthTex->GetDesc(&td);
		D3D11_TEXTURE2D_DESC sd = td;
		sd.Usage = D3D11_USAGE_STAGING;
		sd.BindFlags = 0;
		sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		sd.MiscFlags = 0;
		ComPtr<ID3D11Texture2D> staging;
		if (SUCCEEDED(device->CreateTexture2D(&sd, nullptr, &staging))) {
			context->CopyResource(staging.Get(), depthTex.Get());
			D3D11_MAPPED_SUBRESOURCE m{};
			if (SUCCEEDED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &m))) {
				atlasRowFloats = m.RowPitch / static_cast<UINT>(sizeof(float));
				atlas.resize(static_cast<size_t>(atlasRowFloats) * td.Height);
				std::memcpy(atlas.data(), m.pData, static_cast<size_t>(m.RowPitch) * td.Height);
				context->Unmap(staging.Get(), 0);
				haveAtlas = true;
			}
		}
	}
	if (!haveAtlas)
		logger::warn("VSM dump: atlas readback FAILED — occluder values below are placeholder 1.0");
	auto atlasAtPx = [&](int x, int y) -> float {
		if (!haveAtlas) return 1.0f;
		x = std::clamp(x, 0, kAtlasW - 1);
		y = std::clamp(y, 0, kAtlasH - 1);
		return atlas[static_cast<size_t>(y) * atlasRowFloats + x];
	};
	auto atlasAtUV = [&](float u, float v) -> float {
		return atlasAtPx(static_cast<int>(std::floor(u * kAtlasW)), static_cast<int>(std::floor(v * kAtlasH)));
	};

	// Copy the head of a GPU vertex buffer into CPU memory (for probe surface points).
	auto readVerts = [&](ID3D11Buffer* src, UINT wanted, std::vector<uint8_t>& out) -> bool {
		if (!src) return false;
		D3D11_BUFFER_DESC bd{};
		src->GetDesc(&bd);
		const UINT n = (std::min)(wanted, bd.ByteWidth);
		if (!n) return false;
		D3D11_BUFFER_DESC sdb{};
		sdb.ByteWidth = n;
		sdb.Usage = D3D11_USAGE_STAGING;
		sdb.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ComPtr<ID3D11Buffer> st;
		if (FAILED(device->CreateBuffer(&sdb, nullptr, &st))) return false;
		D3D11_BOX box{ 0, 0, 0, n, 1, 1 };
		context->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, src, 0, &box);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (FAILED(context->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) return false;
		out.assign(static_cast<const uint8_t*>(m.pData), static_cast<const uint8_t*>(m.pData) + n);
		context->Unmap(st.Get(), 0);
		return true;
	};

	// Round-trip one face of a light: back-project the stored occluder texel nearest the face
	// centre to world, forward-project it, and return the linear diff. diff≈0 && reFace==f => the
	// render matrix and the sample path are consistent for that face. Returns false if empty.
	auto roundTrip = [&](const LightRecord& Lr, int f, float& diff, int& reFace) -> bool {
		const int tileX = f % 3, tileY = f / 3;
		const int px0 = static_cast<int>(Lr.atlasCol) * kBlockW + tileX * kFaceRes;
		const int py0 = static_cast<int>(Lr.atlasRow) * kBlockH + tileY * kFaceRes;
		int bx = -1, by = -1; float bd = 1.0f, bDist = 1e30f;
		for (int y = 0; y < kFaceRes; y += 4)
			for (int x = 0; x < kFaceRes; x += 4) {
				const float d = atlasAtPx(px0 + x, py0 + y);
				if (d >= 0.9990f) continue;
				const float cd = static_cast<float>((x - kFaceRes / 2) * (x - kFaceRes / 2) + (y - kFaceRes / 2) * (y - kFaceRes / 2));
				if (cd < bDist) { bDist = cd; bx = x; by = y; bd = d; }
			}
		if (bx < 0) return false;
		const float fu = (bx + 0.5f) / kFaceRes, fv = (by + 0.5f) / kFaceRes;
		const XMMATRIX inv = XMMatrixInverse(nullptr, XMLoadFloat4x4(&Lr.cubeVP[f]));
		XMFLOAT4 W4; XMStoreFloat4(&W4, XMVector4Transform(XMVectorSet(2.0f * fu - 1.0f, 1.0f - 2.0f * fv, bd, 1.0f), inv));
		if (std::fabs(W4.w) < 1e-6f) return false;
		const XMFLOAT3 Wp{ W4.x / W4.w, W4.y / W4.w, W4.z / W4.w };
		reFace = faceFromDir(Wp.x - Lr.positionWS.x, Wp.y - Lr.positionWS.y, Wp.z - Lr.positionWS.z);
		XMFLOAT4 C; XMStoreFloat4(&C, XMVector4Transform(XMVectorSet(Wp.x, Wp.y, Wp.z, 1.0f), XMLoadFloat4x4(&Lr.cubeVP[reFace])));
		const float nx = C.x / C.w, ny = C.y / C.w;
		const float su = (Lr.atlasCol * kBlockW + ((reFace % 3) + (nx * 0.5f + 0.5f)) * kFaceRes) / static_cast<float>(kAtlasW);
		const float sv = (Lr.atlasRow * kBlockH + ((reFace / 3) + (ny * -0.5f + 0.5f)) * kFaceRes) / static_cast<float>(kAtlasH);
		diff = lin(bd, Lr.nearPlane, Lr.farPlane) - lin(atlasAtUV(su, sv), Lr.nearPlane, Lr.farPlane);
		return true;
	};

	// ================= PER-LIGHT SWEEP (all lights, one dump) =================
	// For every light: population % per cube face (is the map rendered?), round-trip diff per face
	// (is render↔sample consistent?), and probe inputs (this light's nearest caster, first 2 verts)
	// queued for one GPU dispatch below. pop% == 0 everywhere => nothing rendered into that block;
	// large rtDiff or reFace mismatch (marked '*') => a projection inconsistency for that face.
	struct ProbeTag { int light, caster, vtx; };
	std::vector<ProbeTag> tags;
	tags.reserve(kMaxProbes);

	D3D11_MAPPED_SUBRESOURCE mi{};
	const bool haveProbeMap = probeInBuf && SUCCEEDED(context->Map(probeInBuf.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mi));
	ProbeIn* pin = haveProbeMap ? static_cast<ProbeIn*>(mi.pData) : nullptr;
	int pc = 0;

	logger::info("--- per-light sweep ({} lights): pop% per face | rtDiff per face (~0 = consistent; '*' = reFace mismatch) ---",
	    lightRecords.size());
	for (size_t li = 0; li < lightRecords.size(); ++li) {
		const LightRecord& L = lightRecords[li];
		const float dCam = std::sqrt((L.positionWS.x - camPos.x) * (L.positionWS.x - camPos.x) +
		                             (L.positionWS.y - camPos.y) * (L.positionWS.y - camPos.y) +
		                             (L.positionWS.z - camPos.z) * (L.positionWS.z - camPos.z));
		const int   ldyn  = (li < lightDynamic.size()) ? lightDynamic[li] : -1;
		const float lmove = (li < lightMove.size()) ? lightMove[li] : -1.0f;
		logger::info("L{:02d} posAbs={:9.1f} {:9.1f} {:9.1f} near={:6.2f} far={:8.2f} cell=({},{}) dCam={:.0f} dyn={} moveThisFrame={:.2f}u lightPosWS={:9.1f} {:9.1f} {:9.1f}",
		    static_cast<int>(li), L.positionWS.x, L.positionWS.y, L.positionWS.z, L.nearPlane, L.farPlane,
		    L.atlasCol, L.atlasRow, dCam, ldyn, lmove, L.positionWS.x - altEye.x, L.positionWS.y - altEye.y, L.positionWS.z - altEye.z);

		const float NaNf = std::numeric_limits<float>::quiet_NaN();
		float pop[6]; float rtd[6]; int rtf[6]; float nod[6];  // nod = nearest-occluder linear distance
		for (int f = 0; f < 6; ++f) {
			const int tileX = f % 3, tileY = f / 3;
			const int px0 = static_cast<int>(L.atlasCol) * kBlockW + tileX * kFaceRes;
			const int py0 = static_cast<int>(L.atlasRow) * kBlockH + tileY * kFaceRes;
			int popc = 0, total = 0; float mn = 1.0f;
			for (int y = 0; y < kFaceRes; y += 2)
				for (int x = 0; x < kFaceRes; x += 2) {
					++total; const float dd = atlasAtPx(px0 + x, py0 + y);
					if (dd < 0.9999f) ++popc;
					if (dd < mn) mn = dd;
				}
			pop[f] = 100.0f * popc / total;
			nod[f] = (mn < 0.9999f) ? lin(mn, L.nearPlane, L.farPlane) : NaNf;
			float d; int rf;
			if (roundTrip(L, f, d, rf)) { rtd[f] = d; rtf[f] = rf; }
			else { rtd[f] = NaNf; rtf[f] = f; }
		}
		logger::info("     pop%  +X={:5.0f} -X={:5.0f} +Y={:5.0f} -Y={:5.0f} +Z={:5.0f} -Z={:5.0f}",
		    pop[0], pop[1], pop[2], pop[3], pop[4], pop[5]);
		logger::info("     nearOcc +X={:7.1f} -X={:7.1f} +Y={:7.1f} -Y={:7.1f} +Z={:7.1f} -Z={:7.1f}  (linear dist to closest occluder per face)",
		    nod[0], nod[1], nod[2], nod[3], nod[4], nod[5]);
		logger::info("     rtDiff +X={:7.1f}{} -X={:7.1f}{} +Y={:7.1f}{} -Y={:7.1f}{} +Z={:7.1f}{} -Z={:7.1f}{}",
		    rtd[0], rtf[0] == 0 ? " " : "*", rtd[1], rtf[1] == 1 ? " " : "*", rtd[2], rtf[2] == 2 ? " " : "*",
		    rtd[3], rtf[3] == 3 ? " " : "*", rtd[4], rtf[4] == 4 ? " " : "*", rtd[5], rtf[5] == 5 ? " " : "*");

		// Caster coverage vs this light's frustum: of casters within range (far plane), how many
		// project in-bounds vs get clipped beyond far / before near / behind. High beyondFar => the
		// far plane (radius x FarScale) is too small and nearby geometry casts/receives no shadow.
		{
			const float radius = L.farPlane / (std::max)(dbgFarScale, 1e-3f);
			int inRad = 0, inB = 0, beyF = 0, tooN = 0, behind = 0;
			for (const auto& e : registry) {
				RE::BSGeometry* g = e.geom.get();
				if (!g) continue;
				const auto& c = g->worldBound.center;  // world-absolute
				const float br = g->worldBound.radius;
				const float dx = c.x - L.positionWS.x, dy = c.y - L.positionWS.y, dz = c.z - L.positionWS.z;
				const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
				if (dist - br > L.farPlane) continue;  // wholly beyond the light's reach
				++inRad;
				const int face = faceFromDir(dx, dy, dz);
				XMFLOAT4 C; XMStoreFloat4(&C, XMVector4Transform(XMVectorSet(c.x, c.y, c.z, 1.0f), XMLoadFloat4x4(&L.cubeVP[face])));
				if (C.w <= 1e-4f) { ++behind; continue; }
				const float nz = C.z / C.w;
				if (nz > 1.0f) ++beyF; else if (nz < 0.0f) ++tooN; else ++inB;
			}
			logger::info("     casters: radius={:.0f} inRadius={} inBounds={} beyondFar={} tooNear={} behind={}",
			    radius, inRad, inB, beyF, tooN, behind);
		}

		// Synthetic shadow test: take a REAL occluder from this light's atlas, push a point 100u
		// BEHIND it along the light ray, and run the shadow decision. It MUST be SHADOW if the math
		// is sound — this is independent of scene coverage. 'lit' here => the shadow LOGIC is broken.
		{
			int tf = -1;
			for (int f = 0; f < 6; ++f) if (!std::isnan(nod[f])) { tf = f; break; }
			if (tf >= 0) {
				const int tileX = tf % 3, tileY = tf / 3;
				const int px0 = static_cast<int>(L.atlasCol) * kBlockW + tileX * kFaceRes;
				const int py0 = static_cast<int>(L.atlasRow) * kBlockH + tileY * kFaceRes;
				int bx = -1, by = -1; float bd = 1.0f, bDist = 1e30f;
				for (int y = 0; y < kFaceRes; y += 4)
					for (int x = 0; x < kFaceRes; x += 4) {
						const float d = atlasAtPx(px0 + x, py0 + y);
						if (d >= 0.9990f) continue;
						const float cd = static_cast<float>((x - kFaceRes / 2) * (x - kFaceRes / 2) + (y - kFaceRes / 2) * (y - kFaceRes / 2));
						if (cd < bDist) { bDist = cd; bx = x; by = y; bd = d; }
					}
				if (bx >= 0) {
					const float fu = (bx + 0.5f) / kFaceRes, fv = (by + 0.5f) / kFaceRes;
					const XMMATRIX invM = XMMatrixInverse(nullptr, XMLoadFloat4x4(&L.cubeVP[tf]));
					XMFLOAT4 W4; XMStoreFloat4(&W4, XMVector4Transform(XMVectorSet(2.0f * fu - 1.0f, 1.0f - 2.0f * fv, bd, 1.0f), invM));
					if (std::fabs(W4.w) > 1e-6f) {
						const XMFLOAT3 Wp{ W4.x / W4.w, W4.y / W4.w, W4.z / W4.w };
						const float ox = Wp.x - L.positionWS.x, oy = Wp.y - L.positionWS.y, oz = Wp.z - L.positionWS.z;
						const float dOcc = std::sqrt(ox * ox + oy * oy + oz * oz);
						const float s = dOcc > 1e-3f ? (dOcc + 100.0f) / dOcc : 1.0f;  // 100u past the occluder
						const XMFLOAT3 Wt{ L.positionWS.x + ox * s, L.positionWS.y + oy * s, L.positionWS.z + oz * s };
						const int f2 = faceFromDir(Wt.x - L.positionWS.x, Wt.y - L.positionWS.y, Wt.z - L.positionWS.z);
						XMFLOAT4 C; XMStoreFloat4(&C, XMVector4Transform(XMVectorSet(Wt.x, Wt.y, Wt.z, 1.0f), XMLoadFloat4x4(&L.cubeVP[f2])));
						if (C.w > 1e-4f) {
							const float nz = C.z / C.w, nx = C.x / C.w, ny = C.y / C.w;
							const float su = (L.atlasCol * kBlockW + ((f2 % 3) + (nx * 0.5f + 0.5f)) * kFaceRes) / static_cast<float>(kAtlasW);
							const float sv = (L.atlasRow * kBlockH + ((f2 / 3) + (ny * -0.5f + 0.5f)) * kFaceRes) / static_cast<float>(kAtlasH);
							const float occ = atlasAtUV(su, sv);
							const float lp = lin(nz, L.nearPlane, L.farPlane), lo = lin(occ, L.nearPlane, L.farPlane);
							const bool shadow = (lp - lo) > dbgBiasWorld;
							logger::info("     synthShadow: pt 100u behind occluder(face {} dist {:.0f}) linPix={:.1f} linOcc={:.1f} diff={:.1f} occ={:.4f} -> {}",
							    tf, dOcc, lp, lo, lp - lo, occ, shadow ? "SHADOW (logic OK)" : "lit (LOGIC BROKEN)");
						}
					}
				}
			}
		}

		// queue a caster WITHIN this light's range (nearest such) for the GPU probe — testing a caster
		// beyond the far plane just reports out-of-bounds and teaches nothing. Fall back to nearest.
		if (pin && pc < kMaxProbes) {
			int ci = -1; float best = FLT_MAX; int ciAny = -1; float bestAny = FLT_MAX;
			for (size_t k = 0; k < registry.size(); ++k) {
				RE::BSGeometry* g = registry[k].geom.get();
				if (!g) continue;
				const auto& c = g->worldBound.center;  // world-absolute
				const float br = g->worldBound.radius;
				const float d2 = (c.x - L.positionWS.x) * (c.x - L.positionWS.x) +
				                 (c.y - L.positionWS.y) * (c.y - L.positionWS.y) +
				                 (c.z - L.positionWS.z) * (c.z - L.positionWS.z);
				if (d2 < bestAny) { bestAny = d2; ciAny = static_cast<int>(k); }
				const float dist = std::sqrt(d2);
				if (dist - br > L.farPlane) continue;  // out of the light's reach
				if (d2 < best) { best = d2; ci = static_cast<int>(k); }
			}
			if (ci < 0) ci = ciAny;  // no in-range caster; test the nearest anyway
			if (ci >= 0) {
				RE::BSGeometry* g = registry[ci].geom.get();
				auto* grd = g ? g->GetGeometryRuntimeData().rendererData : nullptr;
				const UINT stride = registry[ci].vertexStride;
				std::vector<uint8_t> vbuf;
				if (grd && grd->vertexBuffer && stride &&
				    readVerts(reinterpret_cast<ID3D11Buffer*>(grd->vertexBuffer), stride * 2u + 16u, vbuf)) {
					const XMMATRIX world = NiTransformToXM(g->world);
					const int nv = (std::min)(2, static_cast<int>(vbuf.size() / stride));
					for (int v = 0; v < nv && pc < kMaxProbes; ++v) {
						const float* fp = reinterpret_cast<const float*>(vbuf.data() + static_cast<size_t>(v) * stride);
						XMFLOAT3 cr;  // world-absolute vertex (geom world is world-space)
						XMStoreFloat3(&cr, XMVector3TransformCoord(XMVectorSet(fp[0], fp[1], fp[2], 1.0f), world));
						// P mimics the shader's camera-relative input.WorldPosition, so W = P + camAdjust = world.
						pin[pc].P          = { cr.x - altEye.x, cr.y - altEye.y, cr.z - altEye.z, 1.0f };
						pin[pc].lightPosWS = { L.positionWS.x - altEye.x, L.positionWS.y - altEye.y, L.positionWS.z - altEye.z, 0.0f };
						tags.push_back({ static_cast<int>(li), ci, v });
						++pc;
					}
				}
			}
		}
	}
	if (haveProbeMap)
		context->Unmap(probeInBuf.Get(), 0);

	// ================= GPU PROBE (all queued points, one dispatch) =================
	// Runs the EXACT VSM::GetLocalShadow on the GPU against the live atlas + light buffer. matched
	// should equal the queued light index; '(MISMATCH)' means the shader picked a different light.
	if (probeCS && haveProbeMap && pc > 0 && probeInSRV && probeOutBuf && probeOutUAV && probeOutStaging && probeCB && lightBufferSRV && srv) {
		D3D11_MAPPED_SUBRESOURCE mc{};
		if (SUCCEEDED(context->Map(probeCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mc))) {
			ProbeCBData cb{};
			cb.camAdjust   = { altEye.x, altEye.y, altEye.z, 0.0f };
			cb.altEye      = { altEye.x, altEye.y, altEye.z, 0.0f };
			cb.bias        = dbgBiasWorld; cb.compareMode = dbgCompareMode; cb.matchEye = dbgMatchEye; cb.sampleSpace = dbgSampleSpace;
			cb.matchThresh = dbgMatchThresh; cb.numProbes = pc; cb.pad0 = 0; cb.pad1 = 0;
			std::memcpy(mc.pData, &cb, sizeof(cb));
			context->Unmap(probeCB.Get(), 0);
		}
		ID3D11ShaderResourceView*  csSRVs[3] = { srv.Get(), lightBufferSRV.Get(), probeInSRV.Get() };
		ID3D11SamplerState*        csSamp    = pointSampler.Get();
		ID3D11UnorderedAccessView* csUAV     = probeOutUAV.Get();
		ID3D11Buffer*              csCB      = probeCB.Get();
		context->CSSetShader(probeCS.Get(), nullptr, 0);
		context->CSSetConstantBuffers(0, 1, &csCB);
		context->CSSetShaderResources(0, 3, csSRVs);
		context->CSSetSamplers(0, 1, &csSamp);
		context->CSSetUnorderedAccessViews(0, 1, &csUAV, nullptr);
		context->Dispatch((pc + 63) / 64, 1, 1);
		ID3D11ShaderResourceView*  nullSRV[3] = {};
		ID3D11UnorderedAccessView* nullUAV    = nullptr;
		context->CSSetShaderResources(0, 3, nullSRV);
		context->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
		context->CSSetShader(nullptr, nullptr, 0);

		context->CopyResource(probeOutStaging.Get(), probeOutBuf.Get());
		D3D11_MAPPED_SUBRESOURCE mo{};
		if (SUCCEEDED(context->Map(probeOutStaging.Get(), 0, D3D11_MAP_READ, 0, &mo))) {
			const auto* po = static_cast<const ProbeOut*>(mo.pData);
			logger::info("--- GPU probe (real VSM math, live atlas; {} pts = each light's nearest caster x2 verts; camAdjust=altEye) ---", pc);
			int nShadow = 0, nLit = 0, nInB = 0, nEmptyOcc = 0;
			for (int i = 0; i < pc; ++i) {
				const auto& o = po[i];
				const auto& t = tags[i];
				const int matched = static_cast<int>(o.uv_occ.w);
				const bool sh = static_cast<int>(o.result.w) != 0;
				sh ? ++nShadow : ++nLit;
				if (static_cast<int>(o.face_flags.y)) ++nInB;         // in-bounds
				if (o.uv_occ.z > 0.9999f) ++nEmptyOcc;                // sampled an empty atlas texel
				logger::info("  L{:02d} c{} v{} W=({:8.1f} {:8.1f} {:8.1f}) matched={}{} face={} inB={} front={} ndcZ={:.4f} occ={:.4f} linPix={:7.1f} linOcc={:7.1f} diff={:7.1f} -> {}",
				    t.light, t.caster, t.vtx, o.W.x, o.W.y, o.W.z,
				    matched, matched == t.light ? "" : "(MISMATCH)",
				    static_cast<int>(o.face_flags.x), static_cast<int>(o.face_flags.y), static_cast<int>(o.face_flags.z),
				    o.ndc.z, o.uv_occ.z, o.result.x, o.result.y, o.result.z, sh ? "SHADOW" : "lit");
			}
			logger::info("  probe tally: {} SHADOW / {} lit of {}  (inBounds={}, sampledEmptyTexel={})  <- 0 SHADOW + MODE 0 means the shader is shadowing NOTHING",
			    nShadow, nLit, pc, nInB, nEmptyOcc);
			context->Unmap(probeOutStaging.Get(), 0);
		} else {
			logger::warn("VSM dump: GPU probe readback map failed");
		}
	} else {
		logger::warn("VSM dump: GPU probe skipped (resources/points unavailable, pc={})", pc);
	}

	// ================= REAL-SHADER PIXEL PROBE (centre pixel — the actual Lighting.hlsl path) =================
	if (probeArmed && pixelProbeBuf && pixelProbeStaging) {
		context->CopyResource(pixelProbeStaging.Get(), pixelProbeBuf.Get());
		D3D11_MAPPED_SUBRESOURCE mp{};
		if (SUCCEEDED(context->Map(pixelProbeStaging.Get(), 0, D3D11_MAP_READ, 0, &mp))) {
			const auto& p = *static_cast<const PixelProbe*>(mp.pData);
			if (p.pixel.z < 0.5f) {
				logger::info("REAL-SHADER PROBE: armed but NOT written (written flag=0). Means the u7 UAV never reached");
				logger::info("  Lighting.hlsl, OR the centre pixel is not lit by any point light. Aim the crosshair at a");
				logger::info("  shadow band on a point-lit surface. If it stays 0 everywhere, the OMSetRenderTargets hook");
				logger::info("  isn't binding u7 into the lighting pass (shader/hook mismatch).");
			} else {
				static const char* kReason[5] = { "eval(shadow computed)", "mode!=0", "no-match", "behind-light", "out-of-bounds" };
				const int rz = (int)p.uv_occ.w; const int reason = (rz < 0 || rz > 4) ? 0 : rz;
				logger::info("REAL-SHADER PROBE  centre pixel=({:.0f},{:.0f})  mode={}  reason={}  -> {}",
				    p.pixel.x, p.pixel.y, (int)p.pixel.w, kReason[reason], p.result.w < 0.5f ? "SHADOW" : "lit");
				logger::info("  P (real WorldPos, cam-rel) = {:9.1f} {:9.1f} {:9.1f}", p.P.x, p.P.y, p.P.z);
				logger::info("  CameraPosAdjust (REAL GPU) = {:9.1f} {:9.1f} {:9.1f}   altEye (C++) = {:9.1f} {:9.1f} {:9.1f}",
				    p.camAdjust.x, p.camAdjust.y, p.camAdjust.z, altEye.x, altEye.y, altEye.z);
				logger::info("  >>> CameraPosAdjust - altEye = {:9.3f} {:9.3f} {:9.3f}   (NON-ZERO = our atlas eye != shader eye = the bug) <<<",
				    p.camAdjust.x - altEye.x, p.camAdjust.y - altEye.y, p.camAdjust.z - altEye.z);
				logger::info("  W (sampleSpace={}) = {:9.1f} {:9.1f} {:9.1f}   absP (P + realCamAdj) = {:9.1f} {:9.1f} {:9.1f}",
				    (int)p.W.w, p.W.x, p.W.y, p.W.z, p.P.x + p.camAdjust.x, p.P.y + p.camAdjust.y, p.P.z + p.camAdjust.z);
				logger::info("  lightPosWS = {:9.1f} {:9.1f} {:9.1f}   absLight = {:9.1f} {:9.1f} {:9.1f}",
				    p.lightPosWS.x, p.lightPosWS.y, p.lightPosWS.z, p.absLight.x, p.absLight.y, p.absLight.z);
				logger::info("  matched={} face={} inB={} front={}  ndc=({:.4f} {:.4f} {:.5f}) clipW={:.2f}  atlasUV=({:.4f} {:.4f}) occ={:.5f}",
				    (int)p.matched.x, (int)p.matched.y, (int)p.matched.z, (int)p.matched.w,
				    p.ndc.x, p.ndc.y, p.ndc.z, p.ndc.w, p.uv_occ.x, p.uv_occ.y, p.uv_occ.z);
				logger::info("  linPix={:.2f} linOcc={:.2f} diff={:.2f}", p.result.x, p.result.y, p.result.z);
			}
			context->Unmap(pixelProbeStaging.Get(), 0);
		} else {
			logger::warn("VSM dump: pixel-probe readback map failed");
		}
	} else if (!probeArmed) {
		logger::info("(real-shader pixel probe not armed — tick 'Arm real-shader pixel probe', aim crosshair at the band, then Dump)");
	}

	logger::info("=========================================");
}

void VirtualShadowMaps::DrawMenu()
{
	// Drawn inside the Community Shaders menu via the CS add-on hook (shared ImGui context).
	menuVisible = true;

	if (!ImGui::CollapsingHeader("Virtual Shadow Maps", ImGuiTreeNodeFlags_DefaultOpen)) {
		showDebug = false;  // panel collapsed -> no debug work
		return;
	}

	ImGui::TextDisabled("build %s", kBuildTag);
	ImGui::Checkbox("Enabled##VSM", &enabled);
	ImGui::SameLine();
	ImGui::TextDisabled("light: %s | casters: %d/%d", haveTestLight ? "yes" : "no", visibleCasters, (int)registry.size());
	ImGui::Checkbox("Frustum cull##VSM", &frustumCull);
	ImGui::SameLine();
	ImGui::Checkbox("Casters absolute (+cam)##VSM", &dbgCastersAbsolute);

	if (ImGui::Button("Dump diagnostic log##VSM"))
		dbgLogRequested = true;  // writes cameraPos/altEye/light vectors to VirtualShadowMaps.log
	ImGui::SameLine();
	if (dumpConfirmFrames > 0) {
		--dumpConfirmFrames;  // counts down each menu draw; set to 180 when the dump actually ran
		ImGui::TextColored(ImVec4(0.30f, 1.00f, 0.30f, 1.00f), "dump written to VirtualShadowMaps.log");
	} else if (dbgLogRequested) {
		ImGui::TextDisabled("dumping...");  // clicked; RenderFrame writes it next frame (needs Enabled)
	} else {
		ImGui::TextDisabled("-> VirtualShadowMaps.log");
	}
	ImGui::Checkbox("Arm real-shader pixel probe (crosshair)##VSM", &probeArmed);
	ImGui::SameLine();
	ImGui::TextDisabled(probeArmed ? "binds u7 during lighting; aim at the band, then Dump" : "off (default)");

	// ---- Live shadow tuning: updates the shader immediately via the b13 cbuffer, no rebuild ----
	ImGui::Separator();
	ImGui::Text("Shadow tuning (live)");

	// Dimension A: which world-space we PROJECT/MATCH in. The atlas is absolute, so absP is the
	// fix; 1/2 are controls. Re-tests every mode below without a rebuild.
	const char* spaces[] = { "0 absP (P + CameraPosAdjust)", "1 P (camera-relative)", "2 P + altEye (C++ render eye)" };
	ImGui::Combo("Sample space##VSM", &dbgSampleSpace, spaces, IM_ARRAYSIZE(spaces));
	// Dimension B: how mode 0 compares depths (isolates the linearization math).
	const char* cmp[] = { "0 linearized distance", "1 raw ndc.z" };
	ImGui::Combo("Compare (mode 0)##VSM", &dbgCompareMode, cmp, IM_ARRAYSIZE(cmp));
	// Dimension C: which eye recovers the shader light's absolute position for the buffer match.
	// LLF's positionWS is relative to posAdjust.getEye() (altEye), not CameraPosAdjust — if they
	// differ the CameraPosAdjust match drifts with the camera. See mode 22 for the delta.
	const char* meye[] = { "0 CameraPosAdjust", "1 altEye (posAdjust.getEye)" };
	ImGui::Combo("Match eye (mode 0)##VSM", &dbgMatchEye, meye, IM_ARRAYSIZE(meye));

	const char* modes[] = {
		"0 Shadow (real)",
		"1 Off (fully lit)",
		"2 absP RGB (control: should stay GLUED)",
		"3 P RGB (control: moves w/ camera)",
		"4 Nearest light pos RGB (const/region)",
		"5 Light->pixel direction RGB",
		"6 Cube face (6 flat colors)",
		"7 In FRONT of light? (green=yes)",
		"8 ndc.xy face coords (RG)",
		"9 Pixel depth ndc.z (radial gray)",
		"10 Inside face bounds? (green=yes)",
		"11 atlasUV we sample (RG)",
		"12 Raw atlas depth / occluder",
		"13 Occluder distance",
		"14 Pixel distance from light",
		"15 Shadow decision (RED=shadow)",
		"16 Light matched? (green=yes)",
		"17 Matched light index (color)",
		"18 Space delta absP-sample (BLACK=match)",
		"19 Atlas populated here? (green=depth<1)",
		"20 Raw-depth shadow decision (RED=shadow)",
		"21 CameraPosAdjust sanity (RGB)",
		"22 Eye delta CamPosAdj-altEye (BLACK=equal)",
	};
	ImGui::Combo("Mode##VSM", &dbgMode, modes, IM_ARRAYSIZE(modes));
	ImGui::TextWrapped("Modes 2+ paint a pipeline stage as full-screen RGB (mode 0 = real shadow). Pan the "
	                   "camera: patterns GLUED to surfaces are correct; ones that TRACK the camera are the "
	                   "broken stage. With Sample space = absP, expect: 2 glued, 6 flat face regions, "
	                   "7/10 green near lights, 16 green near lights, 18 BLACK. Flip to space 1 to see the "
	                   "old broken behavior for comparison.");
	ImGui::Text("far = radius x FarScale ; near = far x NearFrac ; shadow if pixel-occluder > Bias");
	ImGui::SliderFloat("FarScale (x radius)##VSM", &dbgFarScale, 0.25f, 4.0f, "%.2f");
	ImGui::SliderFloat("NearFrac (x far)##VSM", &dbgNearFrac, 0.001f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Bias (world units)##VSM", &dbgBiasWorld, 0.0f, 50.0f, "%.1f");
	ImGui::SliderFloat("Match threshold (world units)##VSM", &dbgMatchThresh, 0.5f, 200.0f, "%.1f", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Depth-view scale (debug)##VSM", &dbgVizScale, 100.0f, 8192.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
	ImGui::Separator();

	// Everything below is a temporary M0 validation view. When this section is closed,
	// RenderFrame does no debug-only GPU work at all.
	showDebug = ImGui::CollapsingHeader("Debug / test view##VSM");
	if (!showDebug)
		return;

	ImGui::Text("active lights: %d   |   chosen dist: %.0f", activeLightCount, dbgLightDist);
	ImGui::SliderInt("Light (-1 = nearest)##VSM", &lightSelect, -1, activeLightCount > 0 ? activeLightCount - 1 : 0);
	ImGui::TextDisabled("preview = 6 cube faces (3x2): +X -X +Y / -Y +Z -Z");
	ImGui::SliderFloat("Depth range##VSM", &previewRange, 2.0f, 6000.0f, "%.0f units", ImGuiSliderFlags_Logarithmic);
	ImGui::SliderFloat("Preview scale##VSM", &previewScale, 0.05f, 1.0f, "%.2f");
	const int casterCount = static_cast<int>(registry.size());
	ImGui::SliderInt("Isolate caster##VSM", &isolateCaster, 0, casterCount, isolateCaster == 0 ? "all" : "%d");
	isolateCaster = (std::min)(isolateCaster, casterCount);

	ImGui::Text("light  : %8.0f %8.0f %8.0f", dbgEye.x, dbgEye.y, dbgEye.z);
	ImGui::Text("camera : %8.0f %8.0f %8.0f", dbgCam.x, dbgCam.y, dbgCam.z);
	ImGui::Text("aabb   : [%.0f %.0f %.0f]..[%.0f %.0f %.0f]",
	    dbgCasterMin.x, dbgCasterMin.y, dbgCasterMin.z, dbgCasterMax.x, dbgCasterMax.y, dbgCasterMax.z);

	if (ImGui::Button("Inspect caster (raw bytes)##VSM"))
		dbgDumpRequested = true;
	if (dbgHaveDump) {
		ImGui::Text("stride=%u fullPrec=%d idxCount=%u", dbgStride, (int)dbgFullPrec, dbgIndexCount);
		for (int i = 0; i < 4; ++i)
			ImGui::Text("  v%d: %9.2f %9.2f %9.2f", i, dbgV[i].x, dbgV[i].y, dbgV[i].z);
		ImGui::Text("  idx: %u %u %u %u %u %u", dbgIdx[0], dbgIdx[1], dbgIdx[2], dbgIdx[3], dbgIdx[4], dbgIdx[5]);
	}

	// Live linear-depth preview: the light's 6 cube faces (near = white, far = black).
	if (previewSRV) {
		ImGui::Image((ImTextureID)previewSRV.Get(), ImVec2(kAtlasW * previewScale, kAtlasH * previewScale));
	}
}
