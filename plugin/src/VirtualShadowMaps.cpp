#include "VirtualShadowMaps.h"
#include "VSMInternal.h"  // shared helpers: NiTransformToXM, GraphicsStateGuard, probe structs, kBuildTag, kReverseZ

#include <imgui.h>  // shared with CS's context via the add-on hook

#include <DirectXPackedVector.h>  // XMConvertHalfToFloat (bone-weight halfs)

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <string>
#include <vector>

// Path B (skinned characters): read skin partitions + the engine bone-matrix palette, CPU-skin, render.
#include "RE/B/BSGeometry.h"
#include "RE/B/BSDynamicTriShape.h"     // CPU-skinned posed verts (dynamicData)
#include "RE/N/NiSkinInstance.h"        // GPU-skinned: skinData/skinPartition/bones/boneMatrices
#include "RE/N/NiSkinData.h"            // inverse-bind (boneData[b].skinToBone)
#include "RE/N/NiSkinPartition.h"       // bind-pose buffers + bone indices/weights per partition

using namespace DirectX;
using namespace vsm;            // atlas geometry + capacities (VSMConstants.h)
using namespace vsm::internal;  // shared helpers (VSMInternal.h)

namespace
{

	// Emit the shared atlas constants as an HLSL declaration line, so the embedded preview
	// and probe shaders below are generated from the SAME values as the C++ code (vsm::) and
	// can never silently diverge. Prepended to each generated shader source at compile time.
	std::string MakeAtlasConstantsHLSL()
	{
		return std::format(
		    "static const int kFaceRes={}, kMaxLights={}, kBlockW={}, kBlockH={}, kAtlasW={}, kAtlasH={};\n",
		    kFaceRes, kMaxLights, kBlockW, kBlockH, kAtlasW, kAtlasH);
	}

	// Same idea for the debug preview linearizer's near/far planes (vsm::kPreviewNear/kPreviewFar).
	std::string MakePreviewConstantsHLSL()
	{
		return std::format("static const float kNear={:.1f}, kFar={:.1f};\n", kPreviewNear, kPreviewFar);
	}

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
	// kNear/kFar are prepended at compile time from vsm::kPreviewNear/kPreviewFar
	// (see MakePreviewConstantsHLSL / SetupResources) so they can't drift from the C++ side.
	constexpr char kLinearizePS[] = R"(
Texture2D DepthTex : register(t0);
SamplerState Samp  : register(s0);
cbuffer PreviewCB : register(b0) { float PreviewRange; float3 _pad; };
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

	// Preview linearizer cbuffer (b0 of the debug preview pass): one float + padding to 16 bytes.
	struct alignas(16) PreviewCBData
	{
		float previewRange;
		float pad[3];
	};

	// Live tuning cbuffer shared with the game's Lighting.hlsl at b13 (VSM.hlsli::VSMDebug).
	// Exactly three float4 rows; the field order/types below MUST match VSM.hlsli or the shader
	// reads garbage. Populated by UpdateDebugCB, sized via sizeof here and in SetupResources.
	struct alignas(16) VSMDebugCB
	{
		float biasWorld;   int mode;        float vizScale;  int sampleSpace;  // row 0
		float matchThresh; int compareMode; int matchEye;    float spare;      // row 1
		float altEyeX;     float altEyeY;   float altEyeZ;   int probeArmed;   // row 2
	};
	static_assert(sizeof(VSMDebugCB) == 48, "VSMDebugCB must stay three float4 rows to match VSM.hlsli::VSMDebug");


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
// kFaceRes/kMaxLights/kBlockW/kBlockH/kAtlasW/kAtlasH are prepended at compile time from vsm:: (see MakeAtlasConstantsHLSL).
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
	const std::string linearizeSrc = MakePreviewConstantsHLSL() + kLinearizePS;
	if (SUCCEEDED(D3DCompile(linearizeSrc.data(), linearizeSrc.size(), "LinearizePS", nullptr, nullptr,
	        "main", "ps_5_0", 0, 0, &psBlob, &e2)))
		device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &linearizePS);

	D3D11_SAMPLER_DESC samp{};
	samp.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
	samp.AddressU = samp.AddressV = samp.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
	device->CreateSamplerState(&samp, &pointSampler);

	D3D11_BUFFER_DESC pcb{};
	pcb.ByteWidth = sizeof(PreviewCBData);
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
	// compareMode / altEye (see UpdateDebugCB). VSMDebugCB = three float4 rows.
	D3D11_BUFFER_DESC dcb{};
	dcb.ByteWidth      = sizeof(VSMDebugCB);
	dcb.Usage          = D3D11_USAGE_DYNAMIC;
	dcb.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
	dcb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	device->CreateBuffer(&dcb, nullptr, &debugCB);

	// ---- GPU shadow-math probe resources (compute pass + in/out structured buffers) ----
	{
		ComPtr<ID3DBlob> csBlob, csErr;
		const std::string probeSrc = MakeAtlasConstantsHLSL() + kProbeCS;
		HRESULT hrc = D3DCompile(probeSrc.data(), probeSrc.size(), "ProbeCS", nullptr, nullptr,
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

	// ---- Real-shader pixel probe (u8): 1-element structured buffer the game's Lighting.hlsl writes,
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

// Mirror the live tuning sliders into the shader's b13 cbuffer. Field order/types are fixed by
// VSMDebugCB (which a static_assert pins to VSM.hlsli's VSMDebug layout).
void VirtualShadowMaps::UpdateDebugCB()
{
	if (!debugCB)
		return;

	// altEye = the render eye the atlas was rasterized around. Use the RELIABLE ShadowSceneNode
	// cameraPos (set in CollectLights), NOT posAdjust.getEye() which intermittently returns garbage
	// (0,0,1) and mis-places the atlas. Space 2 samples with P + altEye for comparison.
	const RE::NiPoint3 eye{ sceneCameraPos.x, sceneCameraPos.y, sceneCameraPos.z };

	const VSMDebugCB cb{
		dbgBiasWorld, dbgMode, dbgVizScale, dbgSampleSpace,   // row 0
		dbgMatchThresh, dbgCompareMode, dbgMatchEye, 0.0f,    // row 1 (spare = 0)
		eye.x, eye.y, eye.z, probeArmed ? 1 : 0,             // row 2 (probeArmed -> write pixel probe to u8)
	};

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
	ImGui::TextDisabled(probeArmed ? "binds u8 during lighting; aim at the band, then Dump" : "off (default)");

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
