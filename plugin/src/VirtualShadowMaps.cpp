#include "VirtualShadowMaps.h"
#include "VSMInternal.h"  // shared helpers: NiTransformToXM, GraphicsStateGuard, probe structs, kBuildTag, kReverseZ
#include "VSMConfig.h"    // persisted runtime tunables (VirtualShadowMaps.toml)

#include <imgui.h>  // shared with CS's context via the add-on hook

#include <DirectXPackedVector.h>  // XMConvertHalfToFloat (bone-weight halfs)

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

// Path B (skinned characters): read skin partitions + the engine bone-matrix palette, CPU-skin, render.
#include "RE/B/BSGeometry.h"
#include "RE/F/FormTypes.h"            // FormType::Static/StaticCollection — static/dynamic caster classification
#include "RE/B/BSDynamicTriShape.h"     // CPU-skinned posed verts (dynamicData)
#include "RE/N/NiSkinInstance.h"        // GPU-skinned: skinData/skinPartition/bones/boneMatrices
#include "RE/N/NiSkinData.h"            // inverse-bind (boneData[b].skinToBone)
#include "RE/N/NiSkinPartition.h"       // bind-pose buffers + bone indices/weights per partition
#include "RE/N/NiAlphaProperty.h"       // GetAlphaBlending()/GetAlphaTesting() + alphaThreshold (A6 cutout clip)
#include "RE/B/BSLightingShaderProperty.h"     // A6: material access for the alpha-tested cutout's diffuse texture
#include "RE/B/BSLightingShaderMaterialBase.h" // A6: diffuseTexture
#include "RE/N/NiSourceTexture.h"              // A6: rendererTexture->resourceView (the diffuse SRV)
#include "RE/N/NiBillboardNode.h"       // camera-facing effect planes (smoke/vapor/fire) — excluded as casters
#include "RE/T/TESObjectREFR.h"         // light reference -> base object (identity + authored LIGH data)
#include "RE/T/TESObjectLIGH.h"         // OBJ_LIGH: nearDistance, flags (shadow type), color, radius
#include "RE/T/TESModel.h"              // model NIF path of a caster's base object (config pattern override)
#include "RE/B/BSShaderProperty.h"      // EShaderPropertyFlag::kCastShadows — mirror the engine's cast flag
#include "RE/B/BSEffectShaderProperty.h"// effect-shader geometry (emitters/FX) — not an opaque caster
#include "RE/B/BSWaterShaderProperty.h" // water shader geometry — reflector/receiver, never a shadow caster
#include "RE/B/BSEffectShaderMaterial.h"// A5: effect-glass baseColor (NiColorA tint + alpha) for translucent shadow tint
#include "RE/P/PlayerCharacter.h"       // player 3D root (viewer-attached light rejection)
#include "RE/P/PlayerCamera.h"          // camera root (viewer-attached light rejection)

using namespace DirectX;
using namespace vsm;            // atlas geometry + capacities (VSMConstants.h)
using namespace vsm::internal;  // shared helpers (VSMInternal.h)

namespace
{

	// ---- File-local named constants (values used only in this translation unit) ----
	inline constexpr int          kMaxParentWalkDepth  = 128;    // parent-chain walk cap (cycle guard), IsSelfOrDescendantOf
	// (RebuildRegistry's child-recursion cap is vsm::kMaxSceneGraphDepth, in VSMConstants.h.)
	inline constexpr size_t       kRegistryReserve     = 1024;   // initial caster-registry capacity (avoid per-frame realloc churn)
	inline constexpr float        kExplosionClampFactor = 4.0f;  // per-vertex clamp radius = bound.radius * factor + base
	inline constexpr float        kExplosionClampBase  = 150.0f; // ...world-unit floor added to that clamp radius
	inline constexpr unsigned int kFlashHalfPeriodFrames = 20;   // caster-flash on/off half-cycle, in frames (~1/3 s @ 60 fps)
	inline constexpr int          kDumpConfirmFrames   = 180;    // ~3 s of "dump written" confirmation in the menu, in frames
	inline constexpr unsigned int kBufferHeadroomBytes = 65536;  // spare bytes over exact size when growing skinned VB/IB
	inline constexpr std::int64_t kBytesPerMB          = 1024 * 1024;  // atlas VRAM log divisor
	inline constexpr float        kCameraStillThreshold = 1.0f;  // camera moved <= this (world units) => treat as still
	inline constexpr float        kRideMatchTolerance  = 0.5f;   // per-frame delta within this fraction of the camera delta => rides camera
	inline constexpr size_t       kLocalAABBCacheCap   = 8192;   // per-geometry local-AABB cache cap (huge-cell guard); used by GetFreshWorldAABB

	// Preview linearizer near/far planes (vsm::kPreviewNear/kPreviewFar) as an HLSL declaration.
	std::string MakePreviewConstantsHLSL()
	{
		return std::format("static const float kNear={:.1f}, kFar={:.1f};\n", kPreviewNear, kPreviewFar);
	}

	// Embedded depth-only VS (position -> light clip). Row-vector: mul(v, M).
	constexpr char kDepthVS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; uint CasterId; uint3 _pad; };
float4 main(float3 pos : POSITION) : SV_Position { return mul(float4(pos, 1.0), WorldViewProj); }
)";

	// Occluder ID pixel shader: writes the per-draw caster id (+1; 0 = empty after clear) to the R32_UINT
	// id atlas alongside depth. The depth test keeps the nearest fragment, so the surviving id texel is
	// exactly the closest occluder — the dump reads it at a pixel's atlas UV to name the caster EXACTLY.
	constexpr char kIdPS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; uint CasterId; uint3 _pad; };
uint main(float4 pos : SV_Position) : SV_Target { return CasterId; }
)";

	// A5 transmittance PS: outputs the per-caster colored transmittance (from PerDrawCB, offset 80) into the RGBA
	// transmittance atlas under a MULTIPLICATIVE blend, so overlapping glass multiplies (order-independent). Reuses
	// the depth VS (position -> light clip). 1 = clear (no attenuation); < 1 dims/tints that channel of the light.
	constexpr char kTransPS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; uint CasterId; uint3 _pad; float4 Transmittance; };
float4 main(float4 pos : SV_Position) : SV_Target { return float4(Transmittance.rgb, 1.0); }
)";

	// A6: alpha-TESTED cutout depth pass. VS passes the caster's UV through; PS samples the diffuse alpha and
	// clip()s below the material threshold so foliage / grates / chain / hair cast their punched-through
	// SILHOUETTE instead of a solid quad. Still writes CasterId to the id atlas (like kIdPS). Used whenever the
	// caster was fully wired (diffuse SRV + a UV attribute); else it renders via the solid kDepthVS/kIdPS path
	// (fail-safe). Always on — no config gate.
	constexpr char kAlphaVS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; uint CasterId; float AlphaThreshold; uint2 _pad; float4 Transmittance; };
struct VOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
VOut main(float3 pos : POSITION, float2 uv : TEXCOORD0) { VOut o; o.pos = mul(float4(pos, 1.0), WorldViewProj); o.uv = uv; return o; }
)";
	constexpr char kAlphaPS[] = R"(
cbuffer PerDrawCB : register(b0) { row_major float4x4 WorldViewProj; uint CasterId; float AlphaThreshold; uint2 _pad; float4 Transmittance; };
Texture2D    DiffuseTex  : register(t0);
SamplerState DiffuseSamp : register(s0);
uint main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
	float a = DiffuseTex.Sample(DiffuseSamp, uv).a;
	clip(a - AlphaThreshold);   // discard fragments below the material's alpha-test threshold -> silhouette
	return CasterId;
}
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
static const float kDivEps = 1e-4;  // near-zero divide guard (mirrors VSM::kDivEps)
float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
	float d = DepthTex.Sample(Samp, uv).r;
	float z = kNear * kFar / max(kNear + d * (kFar - kNear), kDivEps);  // REVERSE-Z perspective -> view-space Z (world units); d=1 near, d=0 far
	float g = saturate(1.0 - z / max(PreviewRange, 1.0));           // near = white, PreviewRange+ units = black
	return float4(g, g, g, 1);
}
)";

	struct alignas(16) PerDrawCB
	{
		XMFLOAT4X4 WorldViewProj;
		uint32_t   CasterId = 0;          // registry index + 1 for the id-atlas (0 = empty); ignored by depth VS
		float      AlphaThreshold = 0.0f; // A6: diffuse-alpha clip threshold for the alpha-tested cutout PS (ignored by opaque/id/trans)
		uint32_t   _pad[2]  = {};
		XMFLOAT4   Transmittance = { 1, 1, 1, 1 };  // A5: per-channel transmittance for the glass pass (unused by the depth/id passes)
	};

	// Preview linearizer cbuffer (b0 of the debug preview pass): one float + padding to 16 bytes.
	struct alignas(16) PreviewCBData
	{
		float previewRange;
		float pad[3];
	};

	// Live tuning cbuffer shared with the game's Lighting.hlsl at b13 (VSM.hlsli::VSMDebug).
	// Exactly four float4 rows; the field order/types below MUST match VSM.hlsli or the shader
	// reads garbage. Populated by UpdateDebugCB, sized via sizeof here and in SetupResources.
	struct alignas(16) VSMDebugCB
	{
		float biasScale;    int mode;        float vizScale;  int sampleSpace;  // row 0 (biasScale = fShadowBiasScale x calc bias)
		float translucentOn; int compareMode; int matchEye;   float pcfRadius;  // row 1 (translucentOn = A5 t112 live? was the dead matchThresh; pcfRadius from iBlurDeferredShadowMask)
		float altEyeX;     float altEyeY;   float altEyeZ;   int probeArmed;   // row 2
		float probePixelX; float probePixelY; float atlasH;  float atlasW;     // row 3 (ppad0/ppad1 -> runtime atlasH/atlasW)
	};
	static_assert(sizeof(VSMDebugCB) == 64, "VSMDebugCB must stay four float4 rows to match VSM.hlsli::VSMDebug");

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

	// True if a_node is a_root or sits somewhere beneath it in the scene graph (walk the parent
	// chain). Used to spot lights attached to the player/camera: such a light rides the viewer, so
	// its shadow cube swings around and paints shadows from an invisible source as you move. A real
	// placed scene light (wall sconce, hearth) parents under the cell/room, never under the viewer,
	// so this never trips on static lights. Depth-capped against cycles.
	bool IsSelfOrDescendantOf(RE::NiAVObject* a_node, RE::NiAVObject* a_root)
	{
		if (!a_node || !a_root)
			return false;
		RE::NiAVObject* n = a_node;
		for (int guard = 0; n && guard < kMaxParentWalkDepth; ++guard, n = n->parent) {
			if (n == a_root)
				return true;
		}
		return false;
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
	LoadConfig();       // apply persisted tunables before the first frame
	SetupResources();
}

// Pull the persisted defaults from VirtualShadowMaps.toml into the live settings at startup.
void VirtualShadowMaps::LoadConfig()
{
	auto& cfg = vsm::GetConfig();
	cfg.Load();
	enabled = cfg.enabled;
}

// When a persisted tuning control changed and the user has released it (nothing active), mirror
// the live settings back into the config and write the file. Called at the end of DrawMenu.
void VirtualShadowMaps::PersistSettingsIfChanged()
{
	if (!settingsDirty || ImGui::IsAnyItemActive())
		return;
	auto& cfg = vsm::GetConfig();
	cfg.enabled = enabled;
	cfg.Save();
	settingsDirty = false;
}

void VirtualShadowMaps::ComputeAtlasDims()
{
	int ini = 0;
	if (auto* col = RE::INIPrefSettingCollection::GetSingleton())
		if (auto* s = col->GetSetting("iShadowMapResolution:Display"))
			ini = s->data.i;   // 'i' prefix = signed-int setting
	// The user's iShadowMapResolution is the TOP rung of the per-light resolution ladder (its exact value,
	// may be non-pow2). The floor rung = min(kFloorFaceRes, max). Per-light faceRes is chosen each frame
	// between these bounds by AssignFaceRes; this only sets the BOUNDS + the packer/diagnostics reference.
	int fr = (ini >= vsm::kFloorFaceRes) ? ini : kFaceRes;
	if (fr < vsm::kFloorFaceRes) fr = vsm::kFloorFaceRes;
	if (fr > 8192) fr = 8192;
	rtMaxFaceRes   = fr;
	rtFloorFaceRes = (std::min)(static_cast<int>(vsm::kFloorFaceRes), rtMaxFaceRes);
	rtFaceRes = rtMaxFaceRes;      // reference = the MAX rung (packer target-width unit + diagnostics)
	rtBlockW  = rtMaxFaceRes * 3;  // a MAX-size light block (3x2 faces)
	rtBlockH  = rtMaxFaceRes * 2;
	rtAtlasW  = rtFloorFaceRes * 3;  // seed: one floor block; PackAtlas grows the atlas to fit the active lights
	rtAtlasH  = rtFloorFaceRes * 2;
	// Honor the user's other SkyrimPrefs shadow keys — we ARE the local shadow system now, so preserve intent.
	if (auto* col = RE::INIPrefSettingCollection::GetSingleton()) {
		if (auto* s = col->GetSetting("fShadowBiasScale:Display"))        rtBiasScale      = s->data.f;  // may be <0 (STEP: everything shadowed)
		if (auto* s = col->GetSetting("iBlurDeferredShadowMask:Display")) rtPCFRadius      = std::clamp(s->data.i, 0, vsm::kMaxPCFRadius);
		if (auto* s = col->GetSetting("fShadowDistance:Display"))         rtShadowDistance = s->data.f;
	}
}

// (Re)create the per-light structured buffer + SRV at `cap` elements. Called once at setup and by
// EnsureLightBuffer when the active-light count outgrows the current capacity (grow-only). CS re-fetches
// GetLightBufferSRV() each Prepass, so replacing the buffer/SRV mid-session is safe (same as the atlas).
bool VirtualShadowMaps::CreateLightBufferResources(int cap)
{
	if (!device || cap < 1) return false;
	lightBuffer.Reset();
	lightBufferSRV.Reset();
	D3D11_BUFFER_DESC lb{};
	lb.ByteWidth           = static_cast<UINT>(sizeof(LightRecord)) * static_cast<UINT>(cap);
	lb.Usage               = D3D11_USAGE_DYNAMIC;
	lb.BindFlags           = D3D11_BIND_SHADER_RESOURCE;
	lb.CPUAccessFlags      = D3D11_CPU_ACCESS_WRITE;
	lb.MiscFlags           = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	lb.StructureByteStride = sizeof(LightRecord);
	if (FAILED(device->CreateBuffer(&lb, nullptr, &lightBuffer))) {
		logger::error("VSM: light buffer creation failed (cap {})", cap);
		return false;
	}
	D3D11_SHADER_RESOURCE_VIEW_DESC sd{};
	sd.Format              = DXGI_FORMAT_UNKNOWN;
	sd.ViewDimension       = D3D11_SRV_DIMENSION_BUFFER;
	sd.Buffer.FirstElement = 0;
	sd.Buffer.NumElements  = static_cast<UINT>(cap);
	device->CreateShaderResourceView(lightBuffer.Get(), &sd, &lightBufferSRV);
	lightBufCapacity = cap;
	return true;
}

// Grow the per-light buffer to hold at least nLights (grow-only, geometric doubling). Called each frame
// after CollectLights, before UploadLightBuffer. NO cap on lights: the buffer is tiny (sizeof(LightRecord)
// per light) so the real cost is the atlas VRAM, which per-light variable resolution keeps in check.
void VirtualShadowMaps::EnsureLightBuffer(int nLights)
{
	if (nLights <= lightBufCapacity) return;
	int cap = lightBufCapacity > 0 ? lightBufCapacity : 1;
	while (cap < nLights) cap *= 2;
	if (!CreateLightBufferResources(cap))
		logger::error("VSM: light buffer grow to {} for {} lights FAILED", cap, nLights);
}

// (Re)create the atlas + id-atlas textures and their views at rtAtlasW x rtAtlasH. Called by SetupResources
// and by PackAtlas when the packed active-light blocks need a bigger atlas. Returns false on allocation failure.
bool VirtualShadowMaps::CreateAtlasResources()
{
	if (!device) return false;
	depthTex.Reset(); dsv.Reset(); srv.Reset();
	idTex.Reset(); idRTV.Reset(); idSRV.Reset();

	D3D11_TEXTURE2D_DESC td{};
	td.Width  = static_cast<UINT>(rtAtlasW);
	td.Height = static_cast<UINT>(rtAtlasH);
	td.MipLevels = 1;
	td.ArraySize = 1;
	td.Format = DXGI_FORMAT_R32_TYPELESS;
	td.SampleDesc.Count = 1;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
	if (FAILED(device->CreateTexture2D(&td, nullptr, &depthTex))) {
		logger::error("VSM: depth texture creation failed ({}x{})", rtAtlasW, rtAtlasH);
		return false;
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

	// Occluder-id atlas (R32_UINT), same size/viewports as depth. Rasterized alongside depth so a texel
	// records which caster (registryIndex+1) wrote the nearest fragment. Diagnostic; failure is non-fatal.
	{
		D3D11_TEXTURE2D_DESC idt = td;
		idt.Format = DXGI_FORMAT_R32_UINT;
		idt.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
		if (SUCCEEDED(device->CreateTexture2D(&idt, nullptr, &idTex))) {
			D3D11_RENDER_TARGET_VIEW_DESC rtv{};
			rtv.Format = DXGI_FORMAT_R32_UINT;
			rtv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
			device->CreateRenderTargetView(idTex.Get(), &rtv, &idRTV);
			D3D11_SHADER_RESOURCE_VIEW_DESC isr{};
			isr.Format = DXGI_FORMAT_R32_UINT;
			isr.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
			isr.Texture2D.MipLevels = 1;
			device->CreateShaderResourceView(idTex.Get(), &isr, &idSRV);
		}
	}
	// A5: colored transmittance atlas (RGBA8), SAME size/layout as the depth atlas. Glass/alpha casters render here
	// under a multiplicative blend; LLF samples it at t112. Cleared white each frame, so it's a no-op when A5 is off.
	// O9 (lazy): allocated ONLY when translucentShadows is on — the default (A5 off) case skips a full W*H*4 atlas.
	// (transTex is only ever USED when A5 is on; the shader gates the t112 sample on gTranslucentOn.)
	transTex.Reset(); transRTV.Reset(); transSRV.Reset();
	if (vsm::GetConfig().translucentShadows)
		EnsureTransmittanceAtlas();
	// A5: read-only DSV of the opaque depth atlas — the glass pass depth-tests (GREATER) against it WITHOUT writing,
	// so glass occluded by opaque geometry doesn't tint, while overlapping glass panes all still contribute.
	dsvReadOnly.Reset();
	{
		D3D11_DEPTH_STENCIL_VIEW_DESC ro{};
		ro.Format        = DXGI_FORMAT_D32_FLOAT;
		ro.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
		ro.Flags         = D3D11_DSV_READ_ONLY_DEPTH;
		device->CreateDepthStencilView(depthTex.Get(), &ro, &dsvReadOnly);
	}
	// P2: keep the static-cache texture (if it exists) sized with the live atlas; recreating it drops the cache.
	if (staticDepthTex)
		CreateStaticCacheResources();
	staticCacheValid = false;   // the live atlas was (re)created -> any cached static depth is gone; re-bake
	return true;
}

// O9 (lazy transmittance): create the A5 RGBA8 transmittance atlas at the current atlas size, only if it doesn't
// already exist. Called from CreateAtlasResources when translucentShadows is on, and on-demand when A5 is toggled
// on at runtime (so the shader never samples an unbound t112, which would sample black and zero the local lights). Idempotent.
void VirtualShadowMaps::EnsureTransmittanceAtlas()
{
	if (transTex || !device || rtAtlasW <= 0 || rtAtlasH <= 0)
		return;
	D3D11_TEXTURE2D_DESC tt{};
	tt.Width              = static_cast<UINT>(rtAtlasW);
	tt.Height             = static_cast<UINT>(rtAtlasH);
	tt.MipLevels          = 1;
	tt.ArraySize          = 1;
	tt.Format             = DXGI_FORMAT_R8G8B8A8_UNORM;
	tt.SampleDesc.Count   = 1;
	tt.Usage              = D3D11_USAGE_DEFAULT;
	tt.BindFlags          = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
	if (SUCCEEDED(device->CreateTexture2D(&tt, nullptr, &transTex))) {
		device->CreateRenderTargetView(transTex.Get(), nullptr, &transRTV);
		device->CreateShaderResourceView(transTex.Get(), nullptr, &transSRV);
	} else {
		logger::warn("VSM: transmittance atlas creation failed ({}x{}); A5 colored shadows unavailable", rtAtlasW, rtAtlasH);
	}
}

// (Re)create the P2 static-cache depth texture + DSV at the current atlas size. Same D32_TYPELESS desc as the
// live atlas so CopyResource(depthTex <- staticDepthTex) is legal. No SRV (it is never sampled — LLF only ever
// sees the single live atlas). Failure is non-fatal: RenderDepth falls back to the non-cached single pass.
bool VirtualShadowMaps::CreateStaticCacheResources()
{
	if (!device) return false;
	staticDepthTex.Reset(); staticDsv.Reset();
	D3D11_TEXTURE2D_DESC td{};
	td.Width  = static_cast<UINT>(rtAtlasW);
	td.Height = static_cast<UINT>(rtAtlasH);
	td.MipLevels = 1; td.ArraySize = 1;
	td.Format = DXGI_FORMAT_R32_TYPELESS;
	td.SampleDesc.Count = 1;
	td.Usage = D3D11_USAGE_DEFAULT;
	td.BindFlags = D3D11_BIND_DEPTH_STENCIL;   // depth target only; never bound as an SRV
	if (FAILED(device->CreateTexture2D(&td, nullptr, &staticDepthTex))) {
		logger::error("VSM: static-cache texture creation failed ({}x{})", rtAtlasW, rtAtlasH);
		return false;
	}
	D3D11_DEPTH_STENCIL_VIEW_DESC dv{};
	dv.Format = DXGI_FORMAT_D32_FLOAT;
	dv.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
	device->CreateDepthStencilView(staticDepthTex.Get(), &dv, &staticDsv);
	// A recreated static-cache texture is UNINITIALIZED -> force a full re-bake for BOTH cache paths: P2 (whole,
	// via staticCacheValid) and P4 (per-light). Clearing bakedPose alone makes every light classify as "new" and
	// re-bake next frame; we deliberately do NOT reset lastStaticCasterCount (that would re-invalidate a SECOND
	// time the frame after, a redundant full bake).
	staticCacheValid      = false;
	bakedPose.clear();
	dirtyLights.clear();
	lastBakedRegistrySize = -1;
	return staticDsv != nullptr;
}

// Lazily create the static cache the first time the P2 module is enabled. Returns false if unavailable.
bool VirtualShadowMaps::EnsureStaticCache()
{
	if (staticDsv) return true;
	return CreateStaticCacheResources();
}

void VirtualShadowMaps::SetupResources()
{
	if (!device)
		return;
	ComputeAtlasDims();                       // faceRes from iShadowMapResolution (no cap)
	if (!CreateAtlasResources()) {            // atlas seeded at one floor block; PackAtlas grows it to fit the active lights
		logger::error("VSM: initial atlas creation failed");
		return;
	}

	// per-draw CB
	D3D11_BUFFER_DESC cbDesc{};
	cbDesc.ByteWidth = sizeof(PerDrawCB);
	cbDesc.Usage = D3D11_USAGE_DYNAMIC;
	cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
	cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	device->CreateBuffer(&cbDesc, nullptr, &perDrawCB);

	// P1a: a D3D11.1 context lets us offset-bind sub-ranges of one big cbuffer (VSSetConstantBuffers1), so the
	// solid caster draws can share a single pre-filled buffer instead of Map-ing per draw. Requires the
	// ConstantBufferOffsetting feature (to bind 16-constant windows) — without it, we drop context1 and keep the
	// per-draw Map path. Null context1 => immediate path everywhere (drawBatchActive stays false).
	if (context && SUCCEEDED(context->QueryInterface(IID_PPV_ARGS(&context1)))) {
		D3D11_FEATURE_DATA_D3D11_OPTIONS opts{};
		if (FAILED(device->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &opts, sizeof(opts))) ||
		    !opts.ConstantBufferOffsetting || !opts.ConstantBufferPartialUpdate) {
			context1.Reset();  // can't offset-bind cbuffer windows -> fall back to the immediate per-draw path
			logger::info("VSM P1a: constant-buffer offsetting unsupported; using the per-draw cbuffer path");
		}
	}

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

	// occluder-id PS (optional; only used when idRTV exists)
	{
		ComPtr<ID3DBlob> idBlob, idErr;
		if (SUCCEEDED(D3DCompile(kIdPS, sizeof(kIdPS) - 1, "IdPS", nullptr, nullptr, "main", "ps_5_0", 0, 0, &idBlob, &idErr)))
			device->CreatePixelShader(idBlob->GetBufferPointer(), idBlob->GetBufferSize(), nullptr, &idPS);
		else
			logger::warn("VSM: id PS compile failed: {}", idErr ? static_cast<const char*>(idErr->GetBufferPointer()) : "");
	}

	// A5 transmittance PS (only used when translucentShadows is on): outputs the per-caster transmittance color.
	{
		ComPtr<ID3DBlob> tBlob, tErr;
		if (SUCCEEDED(D3DCompile(kTransPS, sizeof(kTransPS) - 1, "TransPS", nullptr, nullptr, "main", "ps_5_0", 0, 0, &tBlob, &tErr)))
			device->CreatePixelShader(tBlob->GetBufferPointer(), tBlob->GetBufferSize(), nullptr, &transPS);
		else
			logger::warn("VSM: transmittance PS compile failed: {}", tErr ? static_cast<const char*>(tErr->GetBufferPointer()) : "");
	}

	// A6 alpha-tested cutout depth pass (opt-in). VS passes UV; PS clips on diffuse alpha. Keep the VS bytecode
	// so GetAlphaLayout can CreateInputLayout per UV offset on demand. All failures are non-fatal (feature just
	// stays inert -> casters render solid).
	{
		ComPtr<ID3DBlob> aErr;
		if (SUCCEEDED(D3DCompile(kAlphaVS, sizeof(kAlphaVS) - 1, "AlphaVS", nullptr, nullptr, "main", "vs_5_0", 0, 0, &alphaVSBytecode, &aErr)))
			device->CreateVertexShader(alphaVSBytecode->GetBufferPointer(), alphaVSBytecode->GetBufferSize(), nullptr, &alphaVS);
		else
			logger::warn("VSM: alpha VS compile failed: {}", aErr ? static_cast<const char*>(aErr->GetBufferPointer()) : "");
		ComPtr<ID3DBlob> apBlob, apErr;
		if (SUCCEEDED(D3DCompile(kAlphaPS, sizeof(kAlphaPS) - 1, "AlphaPS", nullptr, nullptr, "main", "ps_5_0", 0, 0, &apBlob, &apErr)))
			device->CreatePixelShader(apBlob->GetBufferPointer(), apBlob->GetBufferSize(), nullptr, &alphaPS);
		else
			logger::warn("VSM: alpha PS compile failed: {}", apErr ? static_cast<const char*>(apErr->GetBufferPointer()) : "");
		D3D11_SAMPLER_DESC sd{};
		sd.Filter   = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
		sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
		sd.MaxLOD   = D3D11_FLOAT32_MAX;
		device->CreateSamplerState(&sd, &alphaSampler);
	}

	// position is always R32G32B32_FLOAT at offset 0 (see RebuildRegistry)
	const D3D11_INPUT_ELEMENT_DESC ie{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 };
	device->CreateInputLayout(&ie, 1, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &ilFull);

	D3D11_DEPTH_STENCIL_DESC dsd{};
	dsd.DepthEnable = TRUE;
	dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
	dsd.DepthFunc = kReverseZ ? D3D11_COMPARISON_GREATER : D3D11_COMPARISON_LESS;
	device->CreateDepthStencilState(&dsd, &depthState);

	D3D11_RASTERIZER_DESC rasterDesc{};
	rasterDesc.FillMode = D3D11_FILL_SOLID;
	rasterDesc.CullMode = D3D11_CULL_BACK;
	rasterDesc.FrontCounterClockwise = FALSE;
	rasterDesc.DepthClipEnable = TRUE;
	// No rasterizer depth bias: the atlas is D32_FLOAT, on which the integer DepthBias is scaled by
	// 2^(exponent(max_z)-23) and evaluates to ~0.19 world units near z=1 — far too small to matter, and
	// depth-magnitude-dependent (unusable). Shadow bias is instead CALCULATED per-pixel in VSM.hlsli
	// (receiver-plane / texel-footprint x slope), which is exact and self-scaling. See the audit.
	rasterDesc.DepthBias = 0;
	rasterDesc.SlopeScaledDepthBias = 0.0f;
	device->CreateRasterizerState(&rasterDesc, &rasterState);
	// Same state but no back-face culling, bound for two-sided casters (planes, foliage cards) so their
	// away-facing side still writes depth — otherwise a two-sided plane pointing away from the light casts
	// no atlas shadow even though the geometry occludes.
	rasterDesc.CullMode = D3D11_CULL_NONE;
	device->CreateRasterizerState(&rasterDesc, &rasterStateNoCull);

	// P4: depth-clear state — DepthFunc ALWAYS + write. A viewport-confined fullscreen triangle (fullscreenVS emits
	// z = 0 = reverse-Z far) then resets exactly ONE light's atlas block to far before its static casters re-render
	// (BakeOneLightStatic), so incremental re-bakes don't disturb the other lights' cached blocks.
	{
		D3D11_DEPTH_STENCIL_DESC dc{};
		dc.DepthEnable    = TRUE;
		dc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
		dc.DepthFunc      = D3D11_COMPARISON_ALWAYS;
		device->CreateDepthStencilState(&dc, &depthClearState);
	}
	// A5: glass depth-tests vs the opaque atlas but does NOT write depth (bound with a read-only DSV), so glass
	// occluded by opaque geometry is rejected while overlapping panes all still contribute. Same reverse-Z sense.
	{
		D3D11_DEPTH_STENCIL_DESC dro{};
		dro.DepthEnable    = TRUE;
		dro.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
		dro.DepthFunc      = kReverseZ ? D3D11_COMPARISON_GREATER : D3D11_COMPARISON_LESS;
		device->CreateDepthStencilState(&dro, &depthReadOnlyState);
	}
	// A5: multiplicative, ORDER-INDEPENDENT transmittance accumulation — dst = src*DEST_COLOR + dst*ZERO = src*dst.
	// Overlapping glass multiplies (commutative), so no per-frame sorting of translucent casters is needed.
	{
		D3D11_BLEND_DESC bd{};
		bd.RenderTarget[0].BlendEnable           = TRUE;
		bd.RenderTarget[0].SrcBlend              = D3D11_BLEND_DEST_COLOR;
		bd.RenderTarget[0].DestBlend             = D3D11_BLEND_ZERO;
		bd.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
		bd.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ZERO;
		bd.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_ONE;
		bd.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
		bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
		device->CreateBlendState(&bd, &multiplyBlend);
	}

	// linear-depth debug view: RGBA8 color target + fullscreen resolve shaders + point sampler
	D3D11_TEXTURE2D_DESC pd{};   // same dimensions as the depth atlas (rt-sized)
	pd.Width = static_cast<UINT>(rtAtlasW);
	pd.Height = static_cast<UINT>(rtAtlasH);
	pd.MipLevels = 1;
	pd.ArraySize = 1;
	pd.SampleDesc.Count = 1;
	pd.Usage = D3D11_USAGE_DEFAULT;
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

	// per-light structured buffer (mirrors lightRecords) for the LLF shader. Sized for kMaxLights INITIALLY,
	// but EnsureLightBuffer grows it on demand each frame — there is NO hard cap on shadow-casting lights.
	CreateLightBufferResources(kMaxLights);

	// live debug tuning cbuffer (b13): bias / mode / vizScale / sampleSpace / matchThresh /
	// compareMode / altEye (see UpdateDebugCB). VSMDebugCB = four float4 rows.
	D3D11_BUFFER_DESC dcb{};
	dcb.ByteWidth      = sizeof(VSMDebugCB);
	dcb.Usage          = D3D11_USAGE_DYNAMIC;
	dcb.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
	dcb.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	device->CreateBuffer(&dcb, nullptr, &debugCB);


	registry.reserve(kRegistryReserve);  // avoid per-frame reallocation churn

	resourcesReady = true;
}

// Snap a desired face resolution UP to the pow2 ladder [floor .. max]. The TOP rung is the EXACT max (the
// user's iShadowMapResolution, may be non-pow2); every rung below is a power of two. Round the top DOWN
// below max (never exceed the user's budget): if `desired` lands above the largest pow2 <= max but below
// max, use max. See notes/VSM_VARIABLE_RESOLUTION.md.
static int SnapFaceResToLadder(int a_desired, int a_floorRes, int a_maxRes)
{
	if (a_maxRes <= a_floorRes) return a_maxRes;      // degenerate: single rung = max
	if (a_desired <= a_floorRes) return a_floorRes;
	if (a_desired >= a_maxRes)   return a_maxRes;
	int r = a_floorRes;
	while (r < a_desired) {
		const int next = r * 2;
		if (next > a_maxRes) return a_maxRes;         // no pow2 rung >= desired fits under max -> top rung = max
		r = next;
	}
	return r;                                          // smallest pow2 rung >= desired, still <= max
}

// Pick a light's cube-face resolution rung from its ON-SCREEN size (angular metric radius/dist), snapped
// UP to the ladder, with hysteresis: hold the previous rung until the desired resolution moves more than
// kLevelHysteresis, so the level doesn't flicker at a boundary as the camera moves.
int VirtualShadowMaps::AssignFaceRes(float a_radius, const RE::NiPoint3& a_lightPos, const RE::NiPoint3& a_camPos, const void* a_lightId)
{
	const float dx = a_lightPos.x - a_camPos.x, dy = a_lightPos.y - a_camPos.y, dz = a_lightPos.z - a_camPos.z;
	float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
	if (dist < 1.0f) dist = 1.0f;
	// Projected shadow diameter in screen pixels ~ (radius/dist) * (screenH / tanHalfFov); aim for ~k shadow
	// texels per screen pixel -> desiredFaceRes = k * that. k, tanHalfFov are approximate; the ladder snap
	// + hysteresis absorb the imprecision.
	const float desiredF = vsm::GetConfig().qualityFactor * (a_radius / dist) * (screenH / vsm::kTanHalfFovV);
	const int   desired  = (desiredF < 1.0f) ? 1 : static_cast<int>(desiredF);
	int level = SnapFaceResToLadder(desired, rtFloorFaceRes, rtMaxFaceRes);
	if (const auto it = prevLightFaceRes.find(a_lightId); it != prevLightFaceRes.end()) {
		const float prev = static_cast<float>(it->second);
		if (desired <= prev * (1.0f + vsm::kLevelHysteresis) && desired >= prev * (1.0f - vsm::kLevelHysteresis))
			level = it->second;                        // within the band -> hold last frame's rung
		level = std::clamp(level, rtFloorFaceRes, rtMaxFaceRes);  // re-validate against current bounds
	}
	prevLightFaceRes[a_lightId] = level;
	return level;
}

// Shelf-pack every active light's 3x2 cube-face block (sized by its per-light faceRes) into the atlas and
// grow the atlas to fit (grow-only). Sets each LightRecord's atlasX/atlasY (pixel origin). Blocks vary in
// size (variable resolution), so this replaces the old uniform grid. Tallest-first shelf packer; the
// descriptor scheme (arbitrary per-light origin) lets a paged/buddy allocator replace it later without any
// shader change (Phase 2). Target width = kAtlasBlocksWide MAX-size blocks.
// STABLE per-light atlas allocation (P1). Each light keeps the SAME atlas slot across frames while its
// resolution tier is unchanged (so P2 can cache its static depth); slots are (re)allocated only on first
// appearance or a tier change, LRU-freed when unseen, and recycled by exact block size. This REPLACES the
// old every-frame repack (which moved every light's slot each frame and would defeat caching). The shader
// is unchanged — it still reads the per-light {atlasX, atlasY, faceRes} descriptor.
void VirtualShadowMaps::PackAtlas()
{
	++atlasFrame;
	const int targetW = vsm::kAtlasBlocksWide * (rtMaxFaceRes * vsm::kCubeFacesWide);
	auto freeKey = [](int w, int h) {
		return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(w)) << 32) | static_cast<std::uint32_t>(h);
	};
	auto releaseSlot = [&](const AtlasSlot& s) { freeSlots[freeKey(s.w, s.h)].emplace_back(s.x, s.y); };
	// Allocate a NEW slot: reuse a freed same-size slot (no fragmentation for same-size churn), else shelf-bump.
	auto allocSlot = [&](int bw, int bh, int fr) -> AtlasSlot {
		auto& fl = freeSlots[freeKey(bw, bh)];
		if (!fl.empty()) {
			const auto [fx, fy] = fl.back(); fl.pop_back();
			return { fx, fy, bw, bh, fr, atlasFrame };
		}
		if (atlasBumpX > 0 && atlasBumpX + bw > targetW) { atlasBumpX = 0; atlasBumpY += atlasBumpRowH; atlasBumpRowH = 0; }  // wrap shelf
		AtlasSlot s{ atlasBumpX, atlasBumpY, bw, bh, fr, atlasFrame };
		atlasBumpX += bw;
		if (bh > atlasBumpRowH) atlasBumpRowH = bh;
		return s;
	};

	// 1. Give each active light a STABLE slot: reuse if the tier is unchanged, else (re)allocate.
	for (size_t i = 0; i < lightRecords.size(); ++i) {
		const void* id = (i < lightPtrs.size()) ? lightPtrs[i] : nullptr;
		const int   fr = static_cast<int>(lightRecords[i].positionWS.w);
		const int   bw = fr * vsm::kCubeFacesWide, bh = fr * vsm::kCubeFacesTall;
		auto it = lightSlots.find(id);
		if (it != lightSlots.end() && it->second.faceRes == fr) {
			it->second.lastUsed = atlasFrame;                                   // reuse -> slot STABLE (cache survives)
		} else {
			if (it != lightSlots.end()) { releaseSlot(it->second); lightSlots.erase(it); }  // tier changed -> free old block
			it = lightSlots.emplace(id, allocSlot(bw, bh, fr)).first;
			staticCacheValid = false;   // P2: a new/moved slot -> any baked static for this light is now in the wrong block
			dirtyLights.insert(id);     // P4: this light's block moved -> only ITS static cache must re-bake (not the whole atlas)
			bakedPose.erase(id);        // P4: the new block is uncleared -> treat as "new" so it bakes immediately (not budget-deferred)
		}
		lightRecords[i].atlasX = static_cast<float>(it->second.x);
		lightRecords[i].atlasY = static_cast<float>(it->second.y);
	}
	// 2. LRU: release slots not used for a while so their space recycles (bounds the atlas over cell churn).
	for (auto it = lightSlots.begin(); it != lightSlots.end();) {
		if (atlasFrame - it->second.lastUsed > vsm::kSlotEvictFrames) { releaseSlot(it->second); it = lightSlots.erase(it); }
		else ++it;
	}
	// 3. Grow the atlas to the shelf high-water mark (grow-only; slot pixel origins stay valid across a grow).
	const int needW = (std::max)((atlasBumpY > 0) ? targetW : atlasBumpX, rtFloorFaceRes * 3);
	const int needH = (std::max)(atlasBumpY + atlasBumpRowH, rtFloorFaceRes * 2);
	if (needW > rtAtlasW || needH > rtAtlasH) {
		rtAtlasW = (needW > rtAtlasW) ? needW : rtAtlasW;
		rtAtlasH = (needH > rtAtlasH) ? needH : rtAtlasH;
		if (!CreateAtlasResources())
			logger::error("VSM: atlas grow to {}x{} FAILED (out of VRAM?)", rtAtlasW, rtAtlasH);
	}
}

// Restore one held light's cube VPs + position from its baked-pose snapshot, so the shader samples its cached
// block at the EXACT pose it was baked with (positionWS.w = faceRes is left untouched — a tier change already
// invalidated the block). Shared by both cache paths (P4 incremental and P2 whole-cache).
void VirtualShadowMaps::PoseFreezeLight(size_t a_i, const BakedPose& a_bp)
{
	for (int f = 0; f < 6; ++f)
		lightRecords[a_i].cubeVP[f] = a_bp.cubeVP[f];
	lightRecords[a_i].positionWS.x = a_bp.pos.x;
	lightRecords[a_i].positionWS.y = a_bp.pos.y;
	lightRecords[a_i].positionWS.z = a_bp.pos.z;
}

// Finalize the P2 static-cache invalidation and, if the cache HOLDS this frame, FREEZE each light's pose to
// the snapshot the cache was baked with — so the shader samples the held atlas with the EXACT baking pose. A
// sub-threshold light jitter (fire flicker, moveThisFrame < kLightMoveEps) would otherwise smear the cached
// static shadow (a pulsing shadow from the fire). Runs after PackAtlas, before UploadLightBuffer.
void VirtualShadowMaps::UpdateStaticCacheState()
{
	auto& cfg = vsm::GetConfig();
	if (!cfg.cacheStaticShadows)
		return;

	if (cfg.incrementalCache) {
		// ---- P4: per-light dirty tracking + budgeted re-bake selection (consumed by RenderDepth) ----
		bakeThisFrame.clear();
		// Prune cache state for lights no longer collected: a light can vanish (cell change, cull, out of grace)
		// while still dirty-and-budget-deferred; without this, bakedPose/dirtyLights would grow unbounded across cells.
		{
			const std::unordered_set<const void*> present(lightPtrs.begin(), lightPtrs.end());
			for (auto it = bakedPose.begin(); it != bakedPose.end();)
				it = present.count(it->first) ? std::next(it) : bakedPose.erase(it);
			for (auto it = dirtyLights.begin(); it != dirtyLights.end();)
				it = present.count(*it) ? std::next(it) : dirtyLights.erase(it);
		}
		// A change in the STATIC-caster set (cell change / streaming) invalidates every cached block at once.
		int staticCount = 0;
		for (const auto& c : registry)
			if (c.isStatic && !c.isTranslucent) ++staticCount;
		if (staticCount != lastStaticCasterCount) {
			bakedPose.clear();
			dirtyLights.clear();
			for (const void* id : lightPtrs) dirtyLights.insert(id);
			lastStaticCasterCount = staticCount;
		}
		// Classify each collected light: NEW / slot-changed (no cached block -> bakedPose erased in PackAtlas) vs
		// MOVED-in-place (> kLightMoveEps from its baked pose). PackAtlas already flagged slot changes into dirtyLights.
		std::vector<size_t> newBakes, movedBakes;
		for (size_t i = 0; i < lightRecords.size(); ++i) {
			const void* id = (i < lightPtrs.size()) ? lightPtrs[i] : nullptr;
			const auto bp = bakedPose.find(id);
			if (bp == bakedPose.end()) { dirtyLights.insert(id); newBakes.push_back(i); continue; }
			const float dx = lightRecords[i].positionWS.x - bp->second.pos.x;
			const float dy = lightRecords[i].positionWS.y - bp->second.pos.y;
			const float dz = lightRecords[i].positionWS.z - bp->second.pos.z;
			if (std::sqrt(dx * dx + dy * dy + dz * dz) > vsm::kLightMoveEps) dirtyLights.insert(id);
			if (dirtyLights.count(id)) movedBakes.push_back(i);
		}
		// Budget: NEW / slot-changed lights bake immediately (a fresh block is uncleared -> must fill it now, never
		// show garbage); MOVED-in-place lights are capped at kMaxRebakesPerFrame, biggest faceRes (nearest) first,
		// so their re-bakes amortize across frames. The rest stay dirty and re-bake on a later frame.
		std::sort(movedBakes.begin(), movedBakes.end(), [&](size_t a, size_t b) {
			return lightRecords[a].positionWS.w > lightRecords[b].positionWS.w;
		});
		bakeThisFrame = newBakes;
		for (size_t k = 0; k < movedBakes.size() && static_cast<int>(k) < vsm::kMaxRebakesPerFrame; ++k)
			bakeThisFrame.push_back(movedBakes[k]);
		const std::unordered_set<size_t> bakeSet(bakeThisFrame.begin(), bakeThisFrame.end());
		// Pose-freeze every light that KEEPS its cached block this frame (not baking now) to the pose it was baked
		// with, so its cached static depth stays aligned. Lights baking this frame keep their CURRENT pose so the
		// re-bake captures the new pose (BakeOneLightStatic refreshes bakedPose).
		for (size_t i = 0; i < lightRecords.size(); ++i) {
			if (bakeSet.count(i)) continue;
			const void* id = (i < lightPtrs.size()) ? lightPtrs[i] : nullptr;
			const auto it = bakedPose.find(id);
			if (it != bakedPose.end()) PoseFreezeLight(i, it->second);
		}
		return;
	}

	// ---- P2 (whole-cache) path (unchanged): one global validity bit, whole-atlas rebake, whole pose-freeze ----
	if (lastBakedRegistrySize >= 0 && static_cast<int>(registry.size()) != lastBakedRegistrySize)
		staticCacheValid = false;   // caster-count change ~ cell change
	if (!staticCacheValid)
		return;                     // will re-bake in RenderDepth with the CURRENT poses -> nothing to freeze
	for (size_t i = 0; i < lightRecords.size(); ++i) {
		const void* id = (i < lightPtrs.size()) ? lightPtrs[i] : nullptr;
		const auto it = bakedPose.find(id);
		if (it != bakedPose.end()) PoseFreezeLight(i, it->second);
	}
}

void VirtualShadowMaps::CollectLights(RE::ShadowSceneNode* a_ssn)
{
	lightRecords.clear();
	lightNames.clear();
	lightOwnerRef.clear();
	lightNodeRef.clear();
	lightMeta.clear();
	lightPtrs.clear();
	lightDynamic.clear();
	haveTestLight    = false;
	activeLightCount = 0;
	lightsAmbient    = 0;
	lightsNonShadow  = 0;
	lightsCulled     = 0;
	lightsViewerAttached = 0;
	lightsHidden         = 0;
	if (!a_ssn)
		return;
	auto& rt = a_ssn->GetRuntimeData();
	const auto& camPos = rt.cameraPos;
	dbgCam = { camPos.x, camPos.y, camPos.z };
	sceneCameraPos = { camPos.x, camPos.y, camPos.z };  // reliable atlas eye (see RenderDepth/UpdateDebugCB)

	// How far the camera itself moved since last frame — the yardstick for "does this shadow ride the
	// camera?" A light (or caster) whose per-frame movement matches this is locked to the viewer.
	cameraMovedThisFrame = havePrevSceneCam ? std::sqrt(
	    (camPos.x - prevSceneCameraPos.x) * (camPos.x - prevSceneCameraPos.x) +
	    (camPos.y - prevSceneCameraPos.y) * (camPos.y - prevSceneCameraPos.y) +
	    (camPos.z - prevSceneCameraPos.z) * (camPos.z - prevSceneCameraPos.z)) : 0.0f;
	prevSceneCameraPos = { camPos.x, camPos.y, camPos.z };
	havePrevSceneCam   = true;

	// Roots used to reject lights that ride the viewer (torch/eye lights with no visible source):
	// the player's own 3D and the camera node. A light beneath either paints shadows that swing from
	// nowhere as you move; a placed scene light never parents under these. (See addLight below.)
	RE::NiAVObject* playerRoot = nullptr;
	RE::NiAVObject* cameraRoot = nullptr;
	if (auto* pc = RE::PlayerCharacter::GetSingleton())
		playerRoot = pc->Get3D();
	if (auto* pcam = RE::PlayerCamera::GetSingleton())
		cameraRoot = pcam->cameraRoot.get();
	// P3 (cullCasters): camera forward = Y row of the camera rotation, for the behind-camera light cull.
	RE::NiPoint3 camFwd{ 0.0f, 1.0f, 0.0f };
	if (cameraRoot) {
		const auto& cr = cameraRoot->world.rotate;
		camFwd = { cr.entry[0][1], cr.entry[1][1], cr.entry[2][1] };
	}

	// Gather ALL active local lights (not the engine's shadow-limited subset) — the point
	// of the mod is to shadow lights the engine dropped. NO cap: each light later gets a
	// variable-resolution 3x2 cube-face block shelf-packed into the atlas by PackAtlas (both
	// the atlas texture and the per-light GPU buffer grow to fit — EnsureLightBuffer).
	float nearestDistSq = FLT_MAX;
	int   nearestIdx    = -1;

	// Add one scene-graph light to the buffer, deduped by world position (a light can appear
	// in both arrays). ALWAYS returns true (no light cap; the bool is a vestige of the old capped design). We collect
	// BOTH activeLights AND activeShadowLights — matching LLF's light set exactly — so
	// shadow-casting lights (braziers etc., which live in activeShadowLights) also get cubes.
	// The sun and origin-positioned lights are skipped.
	auto addLight = [&](RE::BSLight* bl) -> bool {
		// NO cap on shadow lights (the project's core design goal). The per-light GPU buffer AND the atlas
		// both grow to fit every collected light (EnsureLightBuffer / PackAtlas). VRAM is the only limit,
		// and per-light variable resolution keeps it small (see notes/VSM_VARIABLE_RESOLUTION.md).
		if (!bl || bl == rt.sunLight)
			return true;
		// Only genuine shadow-casters — skip ambient/fill lights (the engine's own flags). Shadowing every
		// illuminating light rather than just casters produces "shadows from nowhere".
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
		// Skip lights the game isn't really presenting as placed scene lights:
		//  - hidden/app-culled (kHidden): a disabled/off light shouldn't cast.
		//  - attached to the player or camera (torch / eye light): it rides the viewer, so its cube
		//    swings around and casts shadows from an invisible source that track the camera. We test
		//    both the NiLight and the BSLight's attachment node against the viewer roots.
		if (niLight->GetAppCulled()) {
			++lightsHidden;
			return true;
		}
		if (IsSelfOrDescendantOf(niLight, playerRoot) || IsSelfOrDescendantOf(niLight, cameraRoot) ||
		    IsSelfOrDescendantOf(bl->objectNode.get(), playerRoot) || IsSelfOrDescendantOf(bl->objectNode.get(), cameraRoot)) {
			++lightsViewerAttached;
			return true;
		}
		const RE::NiPoint3& p = niLight->world.translate;
		if (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f)
			return true;
		for (const auto& rec : lightRecords)  // dedup: same light already added via the other array
			if (rec.positionWS.x == p.x && rec.positionWS.y == p.y && rec.positionWS.z == p.z)
				return true;

		const float    radius = niLight->GetLightRuntimeData().radius.x;  // this light's actual reach
		// P3 broad-phase (cullCasters): drop lights that cast NO visible shadow — entirely behind the camera or
		// entirely beyond fShadowDistance. Conservative (radius slack) so a light that could reach the view is
		// never dropped. Culled = no atlas slot + no render + no sample.
		if (vsm::GetConfig().cullCasters) {
			const float dx = p.x - camPos.x, dy = p.y - camPos.y, dz = p.z - camPos.z;
			const float fwd = dx * camFwd.x + dy * camFwd.y + dz * camFwd.z;          // signed distance along camera forward
			if (fwd + radius < 0.0f) { ++lightsCulled; return true; }                 // whole reach behind the camera
			const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
			if (dist - radius > rtShadowDistance) { ++lightsCulled; return true; }    // whole reach beyond fShadowDistance
		}
		const uint32_t idx    = static_cast<uint32_t>(lightRecords.size());
		// Per-light cube-face resolution from the on-screen size metric (variable resolution, ladder rung).
		const int faceRes = AssignFaceRes(radius, p, camPos, static_cast<const void*>(bl));
		// Aesthetics (Q2): smooth the cube CENTER for moving lights so stepped Havok/animation motion
		// (swinging lanterns/torches) glides instead of jerking. New lights + teleports SNAP; small
		// per-frame steps glide toward the true position. Feeds BOTH positionWS (what the shader samples)
		// and BuildCubeMatrices (what we generate) so generation and sampling stay consistent.
		DirectX::XMFLOAT3 sp = { p.x, p.y, p.z };
		{
			const auto sit = smoothLightById.find(static_cast<const void*>(bl));
			if (sit != smoothLightById.end()) {
				const DirectX::XMFLOAT3& q = sit->second;
				const float d = std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z));
				if (d <= vsm::kLightSmoothSnapDist) {  // small step -> glide; big jump (teleport) -> snap (keep sp = raw p)
					const float a = lightSmoothAlpha;
					sp = { q.x + (p.x - q.x) * a, q.y + (p.y - q.y) * a, q.z + (p.z - q.z) * a };
				}
			}
			smoothLightById[static_cast<const void*>(bl)] = sp;
		}
		LightRecord r;
		r.positionWS = { sp.x, sp.y, sp.z, static_cast<float>(faceRes) };  // w = this light's chosen faceRes rung
		// far = the light's own radius: the shader's illumination cutoff is exactly d==radius
		// (intensityFactor==1 => continue), so no shadow can exist past it. Not a tunable — a fact.
		r.farPlane   = (std::max)(radius, 1.0f);
		// near = far * kNearPlaneFraction, floored at kNearPlaneEpsilon world units. Calculated, not tuned
		// (reverse-Z tolerates a small near plane). See VSMConstants.h.
		r.nearPlane  = (std::max)(r.farPlane * vsm::kNearPlaneFraction, vsm::kNearPlaneEpsilon);
		// atlasX/atlasY (this block's pixel origin) are assigned by PackAtlas once every light's rung is known.
		BuildCubeMatrices(XMVectorSet(sp.x, sp.y, sp.z, 1.0f), r.nearPlane, r.farPlane, faceRes, r.cubeVP);
		lightRecords.push_back(r);
		lightNames.push_back(niLight->name.c_str() ? niLight->name.c_str() : "");  // identify the camera light in the dump
		lightOwnerRef.push_back(niLight->GetUserData() ? niLight->GetUserData()->GetFormID() : 0);  // for the ref cull
		lightNodeRef.push_back(NodeOwningRef(niLight));  // the ref the light's node hangs under (may differ from GetUserData)
		lightPtrs.push_back(static_cast<const void*>(bl));  // stable identity for true per-frame movement
		lightDynamic.push_back(bl->dynamic ? 1 : 0);  // parallel to lightRecords: is this a moving/animated light?

		// Authored + engine light metadata via the light's reference -> base LIGH record. Gives the log the
		// designer's own identity / shadow-type / Near-Clip / radius / color instead of only our recompute.
		LightMeta lm;
		const auto& lrt = niLight->GetLightRuntimeData();
		lm.radiusX = lrt.radius.x; lm.radiusY = lrt.radius.y; lm.radiusZ = lrt.radius.z;
		if (RE::TESObjectREFR* ref = niLight->GetUserData()) {
			lm.formID = ref->GetFormID();
			if (RE::TESObjectLIGH* base = skyrim_cast<RE::TESObjectLIGH*>(ref->GetBaseObject())) {
				if (const char* eid = base->GetFormEditorID()) lm.editorID = eid;
				lm.authoredNear = base->data.nearDistance;
				lm.colR = base->data.color.red / 255.0f; lm.colG = base->data.color.green / 255.0f; lm.colB = base->data.color.blue / 255.0f;
				const auto ff = base->data.flags;
				std::string ts;
				if (ff.any(RE::TES_LIGHT_FLAGS::kOmniShadow)) ts += "OmniShadow ";
				if (ff.any(RE::TES_LIGHT_FLAGS::kHemiShadow)) ts += "HemiShadow ";
				if (ff.any(RE::TES_LIGHT_FLAGS::kSpotShadow)) ts += "SpotShadow ";
				if (ff.any(RE::TES_LIGHT_FLAGS::kSpotlight))  ts += "Spot ";
				if (ff.any(RE::TES_LIGHT_FLAGS::kNegative))   ts += "Negative ";
				lm.typeStr = ts.empty() ? "Omni(noShadowFlag)" : ts;
			}
		}
		lightMeta.push_back(lm);

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

	// --- Collection DEBOUNCE (anti-flicker) ---
	// The engine rotates which lights are IsShadowLight()/activeShadowLights each RENDER frame — even when the
	// game is paused (console) — so our collected set churns 5<->8 and shadows blink on/off. Keep each light
	// shadowed for a GRACE window: refresh the cache for lights collected this frame, then RE-ADD any recently
	// seen light that dropped out this frame (using its cached record). This stabilizes the set to the UNION of
	// recently-active shadow lights — which is exactly the mod's goal ("shadow the lights the engine dropped").
	++collectFrame;
	{
		std::unordered_set<const void*> present;
		present.reserve(lightRecords.size() * 2u + 1u);
		for (size_t i = 0; i < lightRecords.size(); ++i) {
			present.insert(lightPtrs[i]);
			recentLights[lightPtrs[i]] = { lightRecords[i], collectFrame };
		}
		for (auto it = recentLights.begin(); it != recentLights.end();) {
			if (collectFrame - it->second.lastSeen > vsm::kLightGraceFrames) {
				it = recentLights.erase(it);                       // gone too long -> forget
			} else {
				if (!present.count(it->first)) {                   // seen recently but missing now -> re-add (bridge the rotation)
					lightRecords.push_back(it->second.rec);
					lightPtrs.push_back(it->first);
					lightNames.emplace_back();
					lightOwnerRef.push_back(0);
					lightNodeRef.push_back(0);
					lightMeta.emplace_back();
					lightDynamic.push_back(0);
				}
				++it;
			}
		}
	}
	activeLightCount = static_cast<int>(lightRecords.size());

	// Per-frame light movement, BY IDENTITY (BSLight*): the true distance THIS light moved since last
	// frame. If lightMove[i] ~= cameraMovedThisFrame (and the camera actually moved), that light rides
	// the camera — the "shadow that moves with the camera". Nearest-neighbour matching hid this by
	// snapping a moving light onto a static neighbour; identity keying gives the real delta.
	lightMove.assign(lightRecords.size(), -1.0f);  // -1 = no prior frame for this light (new)
	for (size_t i = 0; i < lightRecords.size(); ++i) {
		const auto it = prevLightById.find(lightPtrs[i]);
		if (it != prevLightById.end()) {
			const auto& q = it->second;
			const auto& p = lightRecords[i].positionWS;
			lightMove[i] = std::sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y) + (p.z - q.z) * (p.z - q.z));
		}
	}
	prevLightById.clear();
	for (size_t i = 0; i < lightRecords.size(); ++i)
		prevLightById[lightPtrs[i]] = { lightRecords[i].positionWS.x, lightRecords[i].positionWS.y, lightRecords[i].positionWS.z };
	// Prune smoothed-position state for lights that vanished this frame (mirror prevLightById's lifetime),
	// so a light that returns far away later SNAPS via the teleport guard rather than gliding from a stale pos.
	if (!smoothLightById.empty()) {
		const std::unordered_set<const void*> seen(lightPtrs.begin(), lightPtrs.end());
		for (auto it = smoothLightById.begin(); it != smoothLightById.end();)
			it = seen.count(it->first) ? std::next(it) : smoothLightById.erase(it);
	}
	// P2: a collected light MOVED (mv > eps) or a NEW light appeared (mv == -1) -> the cached static depth,
	// baked in each light's cube space, is stale. Invalidate so RenderDepth re-bakes. Rare when standing still
	// (Skyrim lights are mostly static + the light set is stable), so caching holds there.
	for (const float mv : lightMove)
		if (mv < 0.0f || mv > vsm::kLightMoveEps) { staticCacheValid = false; break; }
	haveTestLight    = !lightRecords.empty();

	// Diagnostics: highlight the selected light (or nearest if none selected).
	const int sel = (lightSelect >= 0 && lightSelect < activeLightCount) ? lightSelect : nearestIdx;
	if (sel >= 0) {
		dbgEye = { lightRecords[sel].positionWS.x, lightRecords[sel].positionWS.y, lightRecords[sel].positionWS.z };
		const float dx = dbgEye.x - camPos.x, dy = dbgEye.y - camPos.y, dz = dbgEye.z - camPos.z;
		dbgLightDist = std::sqrt(dx * dx + dy * dy + dz * dz);
	}
}

// Per-frame: measure each caster's world.translate (render-origin) movement BY IDENTITY (BSGeometry*) and
// flag any that move WITH the camera. A caster drawn in camera-relative space has its origin move by ~ the
// camera's own delta each frame; a correct world-space caster stays put. This is the caster-side twin
// of the light-movement check — it pinpoints the "shadow that rides the camera" now that lights are
// ruled out. Cheap (one small map rebuilt per frame); results surface in the diagnostic dump.
void VirtualShadowMaps::TrackCasterMotion()
{
	castersRidingCam   = 0;
	casterRideMaxDelta = 0.0f;
	casterRideName.clear();
	const float camMove = cameraMovedThisFrame;

	std::unordered_map<const void*, RE::NiPoint3> cur;
	cur.reserve(registry.size());
	for (const auto& e : registry) {
		RE::BSGeometry* g = e.geom.get();
		if (!g)
			continue;
		// Track the actual RENDER transform origin (geom->world.translate) — that's what places the occluder
		// in the atlas, and (see [[skyrim-world-space-not-camera-relative]]) it is GAME-ABSOLUTE for static
		// ShadowSceneNode geometry. This diagnostic VERIFIES that invariant: a correct caster's origin stays
		// put frame-to-frame, so any caster whose origin moves ~by the camera delta would be a space
		// regression (its shadow riding the camera). We diff the render origin, not worldBound.center,
		// because the render origin is exactly what the draw uses.
		const RE::NiPoint3 c = g->world.translate;
		const void*        key = g;
		cur[key] = c;
		if (camMove <= kCameraStillThreshold)
			continue;  // camera essentially still this frame — can't separate riders from statics
		const auto it = prevCasterCenter.find(key);
		if (it == prevCasterCenter.end())
			continue;
		const auto& q = it->second;
		const float d = std::sqrt((c.x - q.x) * (c.x - q.x) + (c.y - q.y) * (c.y - q.y) + (c.z - q.z) * (c.z - q.z));
		// Rides the camera iff it moved by roughly the camera's own delta (not ~0 like a static, and
		// not some unrelated animation amount).
		if (d > camMove * kRideMatchTolerance && std::fabs(d - camMove) < camMove * kRideMatchTolerance) {
			++castersRidingCam;
			if (d > casterRideMaxDelta) {
				casterRideMaxDelta = d;
				casterRideName     = g->name.c_str() ? g->name.c_str() : "";
			}
		}
	}
	prevCasterCenter.swap(cur);
}

void VirtualShadowMaps::UploadLightBuffer()
{
	if (!lightBuffer)
		return;
	D3D11_MAPPED_SUBRESOURCE m{};
	if (SUCCEEDED(context->Map(lightBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		const size_t cap = static_cast<size_t>(lightBufCapacity);
		const size_t n   = (std::min)(lightRecords.size(), cap);  // EnsureLightBuffer already grew cap >= count
		if (n)
			std::memcpy(m.pData, lightRecords.data(), n * sizeof(LightRecord));
		// Zero unused slots so the shader's nearest-light match (GetDimensions loop) can't hit stale data
		// (real lights are never at the origin — see CollectLights).
		if (n < cap)
			std::memset(static_cast<uint8_t*>(m.pData) + n * sizeof(LightRecord), 0,
			    (cap - n) * sizeof(LightRecord));
		// LLF light-index alignment (Option c): publish BSLight* -> t111 slot ATOMICALLY with this upload, so the
		// map always describes exactly the buffer LLF binds (LLF consumes both one frame later — the render tick is
		// the Present hook, LLF's Prepass reads the previous Present's output; see docs/ARCHITECTURE §6.1). Only the
		// n uploaded slots are mapped; everything else resolves to kNoShadowIndex (LLF renders it unshadowed).
		lightIndexByPtr.clear();
		lightIndexByPtr.reserve(n);
		for (size_t i = 0; i < n; ++i)
			if (i < lightPtrs.size() && lightPtrs[i])
				lightIndexByPtr.emplace(lightPtrs[i], static_cast<std::uint32_t>(i));
		context->Unmap(lightBuffer.Get(), 0);
	}
}

// LLF light-index alignment (Option c): BSLight* -> its ShadowLights (t111) slot, or kNoShadowIndex if this light
// has no shadow record this frame. O(1) hash lookup. Called by LLF (VSM_GetLightShadowIndex export) while building
// its light buffers; reads the map published by the previous frame's UploadLightBuffer (consistent with the buffer
// LLF has bound this frame). No lock: the map is written only in UploadLightBuffer (Present tick) and read during
// the next frame's LLF passes, which do not overlap.
std::uint32_t VirtualShadowMaps::GetLightShadowIndex(const void* a_bsLight) const
{
	const auto it = lightIndexByPtr.find(a_bsLight);
	return it != lightIndexByPtr.end() ? it->second : kNoShadowIndex;
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

	// A5 gate: the shader samples/multiplies the t112 transmittance ONLY when this is 1 — so with the module OFF
	// (default), local lights never depend on t112 being bound (an unbound t112 samples black and would zero them).
	const float translucentOn = vsm::GetConfig().translucentShadows ? 1.0f : 0.0f;
	const VSMDebugCB cb{
		rtBiasScale, dbgMode, dbgVizScale, dbgSampleSpace,   // row 0 (biasScale = fShadowBiasScale multiplier on the calc bias)
		translucentOn, dbgCompareMode, dbgMatchEye, (float)rtPCFRadius,  // row 1 (translucentOn = A5 t112 live?; pcfRadius = PCF half-width from iBlurDeferredShadowMask)
		eye.x, eye.y, eye.z, probeArmed ? 1 : 0,             // row 2 (probeArmed -> write pixel probe to u8)
		probeFracX * screenW, probeFracY * screenH, (float)rtAtlasH, (float)rtAtlasW,  // row 3 (ppad0/1 -> atlasH/atlasW)
	};

	D3D11_MAPPED_SUBRESOURCE m{};
	if (SUCCEEDED(context->Map(debugCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		std::memcpy(m.pData, &cb, sizeof(cb));
		context->Unmap(debugCB.Get(), 0);
	}
}

// Six 90-degree perspective faces of a cube centred on the light. Face order (+X,-X,+Y,-Y,
// +Z,-Z) and the up vectors below MUST match VSM.hlsli's face selection + sampling.
void VirtualShadowMaps::BuildCubeMatrices(FXMVECTOR a_eye, float a_near, float a_far, int a_faceRes, XMFLOAT4X4 a_outVP[6])
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
	// Cube-face guard band: render each face at slightly MORE than 90 degrees so neighbouring faces
	// overlap by ~one texel. A receiver whose direction lands right on a face seam then still finds
	// valid rasterized depth on the picked face instead of the wrong neighbour, removing seam bars.
	// The shader samples with these exact matrices, so it stays self-consistent (no shader change).
	const float fovR = 2.0f * std::atan(static_cast<float>(a_faceRes) / (static_cast<float>(a_faceRes) - 2.0f));  // guard band (~1 texel overlap), per this light's face resolution
	XMMATRIX proj = XMMatrixPerspectiveFovLH(fovR, 1.0f, a_near, a_far);   // standard-Z: near->0, far->1
	if (vsm::internal::kReverseZ) {
		// Reverse-Z (Skyrim's convention): remap NDC z -> 1 - z, so near->1, far->0 for uniform float-depth
		// precision. Post-multiply the standard projection by a z-flip (row-major v*M): (x,y,z,w)->(x,y,w-z,w).
		// The clip range stays [0,w], so near/far CLIPPING is unchanged — only the stored depth is reversed.
		// Pairs with clear=0.0 + DepthFunc=GREATER (SetupResources) and LinearizeCubeDepth's reverse-Z branch.
		proj = XMMatrixMultiply(proj, XMMatrixSet(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));
	}
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
		rejectedNoRenderData = rejectedNoVertexBuffer = rejectedNoIndexBuffer = rejectedZeroTriangles = rejectedDuplicate = 0;
		rejectedHidden = rejectedTransparent = rejectedNoCast = rejectedDecal = rejectedBillboard = 0;
		rejectedUserNoCast = userForcedCast = rejectedNotVisible = rejectedEngineFlag = 0;
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
		RebuildRegistry(p3d, 0, pc->GetFormID());
	if (auto* cell = pc->GetParentCell()) {
		cell->ForEachReference([this](RE::TESObjectREFR* a_ref) {
			if (a_ref) {
				if (auto* r3d = a_ref->Get3D())
					RebuildRegistry(r3d, 0, a_ref->GetFormID());  // tag every caster with its owning ref
			}
			return RE::BSContainer::ForEachResult::kContinue;
		});
	}
}

// Classify a caster STATIC (safe to cache its depth once — P2) vs DYNAMIC (re-render each frame). Err toward
// DYNAMIC: a mis-cached mover shows a frozen shadow, so the asymmetry must favour "don't cache". Recipe from
// the R5 research (notes/VSM_PERFORMANCE_PLAN.md); runs once per registry rebuild, cached on CasterEntry.
static bool IsCasterStatic(RE::BSGeometry* a_g, RE::FormID a_ownerRef)
{
	if (!a_g)
		return false;
	// 0. Skinned geometry deforms every frame -> DYNAMIC.
	if (a_g->AsDynamicTriShape())                    return false;  // head/hair/face morph (BSDynamicTriShape)
	if (a_g->GetGeometryRuntimeData().skinInstance)  return false;  // bone-skinned (actors, cloth)
	// 1. An animation controller on the node or ANY ancestor drives the transform in place (MovableStatic
	//    flames, swinging chains, spinning wheels) -> DYNAMIC. Cheapest catch for the animated-MSTT case.
	for (RE::NiAVObject* n = a_g; n; n = n->parent)
		if (n->GetControllers())                     return false;
	// 2. Base FormType via the owning reference: only worldspace STATIC / StaticCollection is safe to cache.
	if (a_ownerRef) {
		if (auto* ref = RE::TESForm::LookupByID<RE::TESObjectREFR>(a_ownerRef))
			if (auto* base = ref->GetBaseObject()) {
				const RE::FormType ft = base->GetFormType();
				if (ft == RE::FormType::Static || ft == RE::FormType::StaticCollection)
					return true;
			}
	}
	// 3. Default: DYNAMIC (actors/doors/activators/furniture/trees/loose havok clutter/ownerRef==0).
	return false;
}

// A5: per-glass transmittance from the material. coverage = opacity (0 clear .. 1 fully tinted), tint = the color the
// light picks up passing through. Effect-shader glass exposes both via baseColor (NiColorA); lighting-shader glass
// falls back to a default dim with a white tint (reading BSLightingShaderMaterialBase's own color is a follow-up).
static void ReadTranslucentMaterial(RE::BSShaderProperty* a_prop, float& a_coverage,
    float& a_tintR, float& a_tintG, float& a_tintB)
{
	a_coverage = vsm::kTranslucentCoverage;
	a_tintR = a_tintG = a_tintB = 1.0f;
	if (!a_prop)
		return;
	if (auto* efx = netimmerse_cast<RE::BSEffectShaderProperty*>(a_prop)) {
		if (auto* mat = efx->GetMaterial()) {
			const RE::NiColorA& bc = mat->baseColor;                 // material rgb tint + alpha coverage
			a_tintR    = std::clamp(bc.red, 0.0f, 1.0f);
			a_tintG    = std::clamp(bc.green, 0.0f, 1.0f);
			a_tintB    = std::clamp(bc.blue, 0.0f, 1.0f);
			a_coverage = std::clamp(bc.alpha, 0.0f, 1.0f);
		}
	}
}

// -----------------------------------------------------------------------------------------------
// Config pattern override (config.forceCast / forceNoCast) — per-shape shadow-caster whitelist/blacklist.
// The NIF itself carries the engine's cast intent (kCastShadows etc., handled below); these let a USER
// force specific meshes to cast (beating every built-in exclusion) or never cast, by matching the
// caster's model NIF PATH (substring, e.g. '\effects\') and shape NAME (WHOLE-TOKEN, so 'marker' hits
// 'EditorMarker' but not 'Market'/'Moonlight'). Research found no other mod does runtime pattern-based
// shadow classification — this is the user escape hatch for the cases flags miss.
// -----------------------------------------------------------------------------------------------

// Lowercase + normalize '/'->'\' so path patterns match regardless of separator/case.
static std::string VsmNormalize(std::string_view a_s)
{
	std::string o;
	o.reserve(a_s.size());
	for (char c : a_s)
		o.push_back(c == '/' ? '\\' : static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
	return o;
}

// Split a shape name into lowercased alphanumeric tokens, breaking on non-alnum AND camelCase
// (lower->upper) boundaries. 'EditorMarker' -> {editor, marker}; 'Plane03' -> {plane03}.
static void VsmTokenizeName(std::string_view a_name, std::vector<std::string>& a_out)
{
	std::string cur;
	auto flush = [&] {
		if (!cur.empty()) {
			a_out.push_back(cur);
			cur.clear();
		}
	};
	for (size_t i = 0; i < a_name.size(); ++i) {
		const unsigned char c = static_cast<unsigned char>(a_name[i]);
		if (!std::isalnum(c)) {
			flush();
			continue;
		}
		if (i > 0 && std::isupper(c) && std::islower(static_cast<unsigned char>(a_name[i - 1])))
			flush();  // camelCase boundary
		cur.push_back(static_cast<char>(std::tolower(c)));
	}
	flush();
}

// A pattern hits if it's a substring of the normalized path OR equals a whole name token.
static bool VsmPatternHits(const std::vector<std::string>& a_pats, const std::string& a_pathLower,
    const std::vector<std::string>& a_nameToks)
{
	for (const auto& p : a_pats) {
		if (p.empty())
			continue;
		const std::string pl = VsmNormalize(p);
		if (!a_pathLower.empty() && a_pathLower.find(pl) != std::string::npos)
			return true;
		for (const auto& t : a_nameToks)
			if (t == pl)
				return true;
	}
	return false;
}

// The model NIF path of a caster's owning reference (per-ref; all shapes in the NIF share it).
static std::string VsmModelPath(RE::FormID a_owner)
{
	if (!a_owner)
		return {};
	if (auto* ref = RE::TESForm::LookupByID<RE::TESObjectREFR>(a_owner))
		if (auto* base = ref->GetBaseObject())
			if (auto* m = skyrim_cast<RE::TESModel*>(base))
				if (const char* mp = m->GetModel())
					return mp;
	return {};
}

// Per-shape verdict from the user patterns: +1 force-cast, -1 force-no-cast, 0 no override.
// Short-circuits to 0 (skips the model-path lookup) when both lists are empty, so it's free when unused.
static int VsmPatternVerdict(RE::FormID a_owner, const char* a_shapeName)
{
	const auto& cfg = vsm::GetConfig();
	if (cfg.forceCast.empty() && cfg.forceNoCast.empty())
		return 0;
	std::vector<std::string> toks;
	if (a_shapeName)
		VsmTokenizeName(a_shapeName, toks);
	const std::string pathLower = VsmNormalize(VsmModelPath(a_owner));
	if (VsmPatternHits(cfg.forceCast, pathLower, toks))
		return 1;  // force-cast wins over force-no-cast on a conflict
	if (VsmPatternHits(cfg.forceNoCast, pathLower, toks))
		return -1;
	return 0;
}

void VirtualShadowMaps::RebuildRegistry(RE::NiAVObject* a_obj, int a_depth, RE::FormID a_owner, bool a_underBillboard)
{
	if (!a_obj || a_depth > vsm::kMaxSceneGraphDepth)  // depth cap guards against scene-graph cycles
		return;
	if (registry.size() >= kMaxCasters)  // hard cap (huge cell / runaway guard)
		return;

	// Skip anything the GAME is not drawing — casting its shadow produces "shadows from nothing"
	// disconnected from any visible geometry. The single biggest offender in interiors is editor
	// markers (XMarker / MarkerX / Marker_Idle / EditorMarker): they carry real vertex buffers but are
	// hidden in-game (a flat idle marker sits exactly where an NPC stands -> a big rectangle stuck to
	// the character). kHidden (== GetAppCulled) is the engine's own "not shown" flag — disabled refs,
	// markers, script-hidden nodes; it's exactly what the "Show Markers" toggle flips. It's a
	// PERSISTENT authored/scripted state, NOT per-frame view culling, so it won't drop an enabled
	// caster that's merely off-screen. Match on this flag only — never on node name (a real actor or
	// item could legitimately contain "marker"). Skip the whole subtree.
	if (a_obj->GetAppCulled()) {
		++rejectedHidden;
		return;
	}

	// Skip subtrees that are NOT valid local-shadow casters:
	//  - "Sky": the skybox (AtmosphereDome/Galaxy/Constellations/celestial dome) is CAMERA-ATTACHED
	//    (its world.translate.xy == cameraPos.xy), so capturing it makes shadows slide with the camera.
	//  - "Weather"/"LODRoot"/"ObjectLODRoot": weather effects and distant LOD, no useful local shadow.
	if (const char* nm = a_obj->name.c_str(); nm) {
		if (std::strcmp(nm, "ObjectLODRoot") == 0 || std::strcmp(nm, "Sky") == 0 ||
		    std::strcmp(nm, "Weather") == 0 || std::strcmp(nm, "LODRoot") == 0)
			return;
	}

	// A NiBillboardNode turns to face the camera every frame. Any geometry beneath one is a camera-facing
	// effect plane (smoke/vapor/fire/glow); it must NOT cast a hard shadow: (1) as it re-orients each frame
	// its shadow would RIDE THE CAMERA, and (2) it is a translucent effect. Detect it during descent and
	// propagate to all descendants. (An opaque-material billboard has no alpha/effect/decal/no-cast flag to
	// catch it — being a billboard is the only reliable disqualifier.)
	const bool underBillboard = a_underBillboard || (netimmerse_cast<RE::NiBillboardNode*>(a_obj) != nullptr);

	if (auto* tri = a_obj->AsTriShape()) {
		auto& geoRt = tri->GetGeometryRuntimeData();
		auto* rd = geoRt.rendererData;
		const uint32_t triCount = tri->GetTrishapeRuntimeData().triangleCount;
		// Shadow casters must be OPAQUE. Skip alpha-BLENDED geometry (blood decals, glass, glow,
		// liquids) and effect-shader geometry (emitters / weapon FX): it's transparent or additive
		// in-game — invisible as a solid — yet would cast a solid shadow. Much of it is skinned to
		// weapons/characters, so those shadows move with no visible source (the "moving invisible
		// geometry" seen in both 1st and 3rd person). Alpha-TESTED cutouts (foliage, hair cards) keep
		// GetAlphaTesting() but NOT GetAlphaBlending(), so they still cast — correct.
		const bool transparent =
		    (geoRt.alphaProperty && geoRt.alphaProperty->GetAlphaBlending()) ||
		    (geoRt.shaderProperty && netimmerse_cast<RE::BSEffectShaderProperty*>(geoRt.shaderProperty.get()) != nullptr);
		// The engine's OWN per-mesh flag: a shader property with CastShadows OFF is not a shadow
		// caster (fire/light planes, glow billboards, effect meshes). Mirror it exactly so we cast
		// only what the game itself casts — no names, no heuristics.
		const bool noCast = geoRt.shaderProperty &&
		    !geoRt.shaderProperty->flags.any(RE::BSShaderProperty::EShaderPropertyFlag::kCastShadows);
		// A DECAL is a flat texture projected onto a surface (scorch marks, blood, dirt, fire-glow
		// pools) — no volume, never a shadow caster. Skip it. Universally correct regardless of the
		// CastShadows bit.
		const bool decal = geoRt.shaderProperty &&
		    (geoRt.shaderProperty->flags.any(RE::BSShaderProperty::EShaderPropertyFlag::kDecal) ||
		     geoRt.shaderProperty->flags.any(RE::BSShaderProperty::EShaderPropertyFlag::kDynamicDecal) ||
		     geoRt.shaderProperty->flags.any(RE::BSShaderProperty::EShaderPropertyFlag::kProjectedUV) ||   // projected-UV decal
		     geoRt.shaderProperty->flags.any(RE::BSShaderProperty::EShaderPropertyFlag::kWeaponBlood));     // weapon-blood overlay
		// Two-sided material: must rasterize with CULL_NONE so the away-facing side still casts (else a
		// plane pointing away from the light writes no atlas depth).
		const bool twoSided = geoRt.shaderProperty &&
		    geoRt.shaderProperty->flags.any(RE::BSShaderProperty::EShaderPropertyFlag::kTwoSided);
		// --- Additional engine "don't cast" signals (research-derived; see notes/VSM_SHADOW_CLASSIFICATION_
		// RESEARCH.md). The vanilla shadow pass also omits geometry carrying these shader flags / of the water
		// type / flagged not-visible, so we mirror it and stop casting shadows the game itself never casts.
		using SPF = RE::BSShaderProperty::EShaderPropertyFlag;
		auto spHas = [sp = geoRt.shaderProperty.get()](SPF f) { return sp && sp->flags.any(f); };
		const bool engineNoCastFlag =
		    spHas(SPF::kNonProjectiveShadows) ||  // explicitly excluded from projected shadow maps
		    spHas(SPF::kBillboard)            ||  // billboard shader (backup to the NiBillboardNode check)
		    spHas(SPF::kWireframe)            ||  // wireframe (particles)
		    spHas(SPF::kLODObjects)           ||  // object-LOD proxy geometry
		    spHas(SPF::kLODLandscape)         ||  // landscape-LOD geometry
		    spHas(SPF::kHDLODObjects);            // HD object-LOD geometry
		// Water is its own shader type (reflector/receiver) — never a shadow caster.
		const bool isWater = geoRt.shaderProperty &&
		    netimmerse_cast<RE::BSWaterShaderProperty*>(geoRt.shaderProperty.get()) != nullptr;
		// Invisible in-game via the runtime NiAVObject flag (distinct from kHidden/app-cull, skipped at the top
		// of RebuildRegistry): the engine doesn't draw OR cast it. Catches an invisible mesh (hidden this way
		// but with kCastShadows still on) that would otherwise cast a shadow on characters.
		const bool notVisible = a_obj->GetFlags().any(RE::NiAVObject::Flag::kNotVisible);
		// Config pattern override (per-shape). Checked FIRST: forceCast beats every built-in exclusion below,
		// forceNoCast beats casting. Free (no path lookup) unless the user set patterns.
		const int patternVerdict = VsmPatternVerdict(a_owner, a_obj->name.c_str());
		if (patternVerdict < 0) {
			++rejectedUserNoCast;                            // user forceNoCast: never cast this mesh
		} else if (patternVerdict > 0) {
			// User forceCast: capture as an OPAQUE caster regardless of billboard/alpha/effect/decal/no-cast
			// flags. Uses the buffer path when the shape has renderer buffers, else the engine-only path.
			if (registrySeen.insert(static_cast<RE::BSGeometry*>(tri)).second) {
				CasterEntry e;
				e.geom     = RE::NiPointer<RE::BSGeometry>(tri);
				e.twoSided = twoSided;
				e.ownerRef = a_owner;
				if (rd && rd->vertexBuffer && rd->indexBuffer && triCount) {
					e.vertexStride = rd->vertexDesc.GetSize();
					e.indexCount   = triCount * 3u;
					e.isStatic     = IsCasterStatic(tri, a_owner);
				} else {
					e.engineOnly = true;                     // no buffers -> engine draw path (skinned/dynamic)
				}
				registry.push_back(std::move(e));
				++userForcedCast;
			} else {
				++rejectedDuplicate;
			}
		} else if (underBillboard) {
			++rejectedBillboard;
		} else if (notVisible) {
			++rejectedNotVisible;                            // engine flag: invisible in-game -> don't cast
		} else if (transparent) {
			// A5: instead of discarding glass/alpha geometry, capture it as a TRANSLUCENT caster — it renders into
			// the transmittance atlas to DIM/TINT the light rather than block it. v1 captures clean static-ish glass
			// (has renderer buffers); skinned/effect translucent (no rd) stays deferred. Off => rejected as before.
			if (vsm::GetConfig().translucentShadows && rd && rd->vertexBuffer && rd->indexBuffer && triCount &&
			    registrySeen.insert(static_cast<RE::BSGeometry*>(tri)).second) {
				CasterEntry e;
				e.geom          = RE::NiPointer<RE::BSGeometry>(tri);
				e.vertexStride  = rd->vertexDesc.GetSize();
				e.indexCount    = triCount * 3u;
				e.twoSided      = twoSided;
				e.ownerRef      = a_owner;
				e.isStatic      = IsCasterStatic(tri, a_owner);
				e.isTranslucent = true;
				float coverage = vsm::kTranslucentCoverage, tintR = 1.0f, tintG = 1.0f, tintB = 1.0f;
				ReadTranslucentMaterial(geoRt.shaderProperty.get(), coverage, tintR, tintG, tintB);
				e.transR = 1.0f - coverage * (1.0f - tintR);  // per-channel transmittance (1 = clear)
				e.transG = 1.0f - coverage * (1.0f - tintG);
				e.transB = 1.0f - coverage * (1.0f - tintB);
				registry.push_back(std::move(e));
			} else {
				++rejectedTransparent;
			}
		} else if (decal) {
			++rejectedDecal;
		} else if (noCast) {
			++rejectedNoCast;
		} else if (engineNoCastFlag || isWater) {
			++rejectedEngineFlag;                            // nonProjective/billboard/wireframe/LOD/water -> engine omits
		} else if (!rd) {
			// No renderer buffers = skinned/dynamic geometry (the character body/head/armor). Our
			// custom depth VS can't draw it, but the ENGINE path (BSUtilityShader) can — it draws from
			// the BSGeometry + its shaderProperty and skins internally. Capture it as engine-only so the
			// engine path has characters to draw; the custom-VS path skips engineOnly (no buffers).
			if (geoRt.shaderProperty && registrySeen.insert(static_cast<RE::BSGeometry*>(tri)).second) {
				CasterEntry e;
				e.geom       = RE::NiPointer<RE::BSGeometry>(tri);
				e.engineOnly = true;  // vertexStride/indexCount stay 0
				e.twoSided   = twoSided;
				e.ownerRef   = a_owner;
				registry.push_back(std::move(e));
			} else {
				++rejectedNoRenderData;
			}
		}
		else if (!rd->vertexBuffer)                                             ++rejectedNoVertexBuffer;
		else if (!rd->indexBuffer)                                              ++rejectedNoIndexBuffer;
		else if (!triCount)                                                     ++rejectedZeroTriangles;
		else if (!registrySeen.insert(static_cast<RE::BSGeometry*>(tri)).second) ++rejectedDuplicate;
		else {
			CasterEntry e;
			e.geom         = RE::NiPointer<RE::BSGeometry>(tri);  // add-ref: keep alive between rebuilds
			e.vertexStride = rd->vertexDesc.GetSize();
			e.indexCount   = triCount * 3u;
			e.twoSided     = twoSided;
			e.ownerRef     = a_owner;
			e.isStatic     = IsCasterStatic(tri, a_owner);       // R5: cache-safe worldspace statics
			// A6 (always on): alpha-TESTED cutout (foliage/grate/chain) — capture the diffuse SRV + threshold + UV
			// offset so the depth pass clip()s the silhouette. Any missing piece leaves alphaTested=false -> solid quad.
			if (geoRt.alphaProperty && geoRt.alphaProperty->GetAlphaTesting() &&
			    !geoRt.alphaProperty->GetAlphaBlending() && rd->vertexDesc.HasFlag(RE::BSGraphics::Vertex::VF_UV)) {
				const std::uint32_t uvOff = rd->vertexDesc.GetAttributeOffset(RE::BSGraphics::Vertex::VA_TEXCOORD0);
				ID3D11ShaderResourceView* diffSrv = nullptr;
				if (auto* lp = netimmerse_cast<RE::BSLightingShaderProperty*>(geoRt.shaderProperty.get()))
					if (auto* mat = static_cast<RE::BSLightingShaderMaterialBase*>(lp->material))
						if (auto* dt = mat->diffuseTexture.get())
							if (auto* rt = dt->rendererTexture)
								diffSrv = rt->resourceView;
				if (diffSrv && uvOff + 4u <= e.vertexStride) {  // valid UV (half2 = 4 bytes) inside the stride
					e.alphaTested    = true;
					e.diffuseSRV     = diffSrv;
					e.alphaThreshold = static_cast<float>(geoRt.alphaProperty->alphaThreshold) / 255.0f;
					e.uvUVOffset     = uvOff;
				}
			}
			registry.push_back(std::move(e));
		}
	}
	if (auto* node = a_obj->AsNode()) {
		for (auto& child : node->GetChildren())
			RebuildRegistry(child.get(), a_depth + 1, a_owner, underBillboard);
	}
}

namespace
{
	constexpr std::uint32_t kMaxSkinVerts = 262144;  // fixed CPU-skin buffers (no realloc during fill)
	constexpr std::uint32_t kMaxSkinIdx   = 524288;
}

// SEH-isolated (POD-only locals) so __try is legal: CPU-skin ONE engineOnly caster — BSTriShape (body/armor)
// OR BSDynamicTriShape (head/hair/face) — into pre-allocated arrays, bounds-checked. A torn-down streaming
// partition faults here and is caught, leaving partial-but-safe results instead of a CTD.
//
// Skin-buffer layout (verified via triangle edge-length checks):
//  - The skin vertex buffer is ONE shape-GLOBAL buffer of N = sp->vertexCount unique verts; all partitions share
//    it. `vertexMap` AND `triList` entries are GLOBAL vertex ids in [0,N) (triList.max()==vertexMap.max()
//    and > p.vertices in every partition). The inline bone-slot bytes are DIRECT global palette indices
//    (max slot == numMats-1), so NiSkinPartition::Partition::bones is VESTIGIAL in this layout.
//  - Therefore: skin each GLOBAL vertex ONCE into [base, base+N), and emit each partition's triList directly
//    against the caster's SINGLE vertex base. A per-partition `partBase + triList[k]` would treat the already-
//    global triList as partition-local, so on partitions >=2 the triangle indices run PAST the caster's block
//    into the NEXT caster's verts in the shared skinPosed buffer. Single-partition meshes are unaffected
//    (partBase==0). Vertex-RADIUS checks are blind to this (positions coherent, connectivity wrong).
//  - Position source is the ONLY body/dynamic difference: dynamicData[g] (CPU-morphed model space) for a dynamic
//    shape, else the buffer's VA_POSITION[g]. Output is WORLD-absolute (palette folds skin->world; no geom->world).
static void VSM_SkinCaster(RE::BSGeometry* a_g, DirectX::XMFLOAT3* a_posed, std::uint32_t* a_idx,
    std::uint32_t a_maxV, std::uint32_t a_maxI, std::uint32_t& a_vCount, std::uint32_t& a_iCount)
{
	using namespace DirectX::PackedVector;
	using V = RE::BSGraphics::Vertex;
	__try {
		auto* skin = a_g->GetGeometryRuntimeData().skinInstance.get();
		if (!skin || !skin->boneMatrices || !skin->skinPartition || skin->numMatrices == 0)
			return;
		auto* sp = skin->skinPartition.get();
		const std::uint32_t N = sp->vertexCount;   // unique global vertex count == shared-buffer length
		if (N == 0 || sp->numPartitions == 0)
			return;
		const float* boneMats = reinterpret_cast<const float*>(skin->boneMatrices);
		const std::uint16_t numMats = static_cast<std::uint16_t>(skin->numMatrices);  // palette size (bounds guard)
		const int matStrideF = static_cast<int>(skin->allocatedSize / skin->numMatrices) / static_cast<int>(sizeof(float));  // -> float count; 12 = 3x4
		if (matStrideF < 12)
			return;

		// Position source. A BSDynamicTriShape supplies CPU-morphed model-space positions in dynamicData
		// (float4, stride 16, indexed by global vertex id); its buffer VA_POSITION is a split/empty stream and
		// must NOT be read. A rigid BSTriShape uses the buffer's own VA_POSITION.
		const std::uint8_t* dyn = nullptr;
		bool isDynShape = false;
		if (auto* dts = a_g->AsDynamicTriShape()) {
			isDynShape = true;
			auto& dr = dts->GetDynamicTrishapeRuntimeData();
			if (dr.dynamicData && dr.dataSize / 16u == N)
				dyn = reinterpret_cast<const std::uint8_t*>(dr.dynamicData);
		}
		if (isDynShape && !dyn)
			return;  // dynamic shape whose dynamicData is absent/mismatched -> skip (buffer position is unusable)

		// Shared global buffer + layout (identical across a shape's partitions -> read layout from partition 0).
		auto& p0 = sp->partitions[0];
		auto* bd0 = p0.buffData;
		if (!bd0 || !bd0->rawVertexData)
			return;
		const std::uint8_t* rv = reinterpret_cast<const std::uint8_t*>(bd0->rawVertexData);
		auto& vd = p0.vertexDesc;
		const std::uint32_t stride  = vd.GetSize();
		const std::uint32_t posOff  = vd.GetAttributeOffset(V::VA_POSITION);
		const std::uint32_t skinOff = vd.GetAttributeOffset(V::VA_SKINNING);
		const size_t kSkinBlockBytes = 4 * sizeof(std::uint16_t) + 4 * sizeof(std::uint8_t);  // 8 + 4 = 12
		if (skinOff + kSkinBlockBytes > stride)
			return;
		if (!dyn && posOff + sizeof(DirectX::XMFLOAT3) > stride)
			return;  // rigid path needs a valid float3 VA_POSITION inside the stride

		// Skin only the vertex range actually REFERENCED by triangles. sp->vertexCount can EXCEED the real /
		// referenced vertex count on some props (seen on Fish: vertexCount=54 but only 51 verts referenced), so
		// skinning the full [0,N) would touch unreferenced entries that may be garbage — polluting the caster
		// bounds (and risking a genuine out-of-bounds buffer read when vertexCount > the real buffer). maxRef =
		// highest triangle index -> skin exactly [0, Nskin).
		std::uint32_t maxRef = 0;
		bool anyTri = false;
		for (std::uint32_t pi = 0; pi < sp->numPartitions; ++pi) {
			auto& p = sp->partitions[pi];
			if (!p.triList)
				continue;
			const std::uint32_t nidx = static_cast<std::uint32_t>(p.triangles) * 3u;
			for (std::uint32_t k = 0; k < nidx; ++k) {
				const std::uint16_t gv = p.triList[k];
				if (gv < N) { anyTri = true; if (gv > maxRef) maxRef = gv; }
			}
		}
		if (!anyTri)
			return;                             // no usable triangles -> nothing to render
		const std::uint32_t Nskin = maxRef + 1;   // referenced range [0,Nskin) <= [0,N)

		const std::uint32_t base = a_vCount;   // this caster's SINGLE vertex-block start (constant across partitions)
		if (base + Nskin > a_maxV)
			return;                             // wouldn't fit -> skip whole caster (never partial/torn)

		// (1) Skin each referenced GLOBAL vertex ONCE into [base, base+Nskin).
		for (std::uint32_t g = 0; g < Nskin; ++g) {
			const std::uint8_t* vb = rv + static_cast<size_t>(g) * stride;
			const std::uint16_t* w  = reinterpret_cast<const std::uint16_t*>(vb + skinOff);
			const std::uint8_t*  ib = reinterpret_cast<const std::uint8_t*>(vb + skinOff + 4 * sizeof(std::uint16_t));  // bone slots follow the 4 weight halfs
			float px, py, pz;
			if (dyn) {
				const float* lp = reinterpret_cast<const float*>(dyn + static_cast<size_t>(g) * 16u);
				px = lp[0]; py = lp[1]; pz = lp[2];
			} else {
				const float* bp = reinterpret_cast<const float*>(vb + posOff);
				px = bp[0]; py = bp[1]; pz = bp[2];
			}
			float wx = 0.0f, wy = 0.0f, wz = 0.0f;
			for (int b = 0; b < 4; ++b) {
				const float wt = XMConvertHalfToFloat(w[b]);
				if (wt <= 0.0f)
					continue;
				const std::uint16_t gi = ib[b];   // DIRECT global palette index (p.bones remap dropped)
				if (gi >= numMats)
					continue;
				const float* M = boneMats + static_cast<size_t>(gi) * matStrideF;  // 3x4 row-major, skin->world
				wx += wt * (M[0] * px + M[1] * py + M[2] * pz + M[3]);
				wy += wt * (M[4] * px + M[5] * py + M[6] * pz + M[7]);
				wz += wt * (M[8] * px + M[9] * py + M[10] * pz + M[11]);
			}
			a_posed[base + g] = { wx, wy, wz };
		}
		a_vCount = base + Nskin;

		// (2) Triangles: triList entries are GLOBAL vertex ids -> single per-caster base, iterate ALL partitions.
		for (std::uint32_t pi = 0; pi < sp->numPartitions; ++pi) {
			auto& p = sp->partitions[pi];
			if (!p.triList)
				continue;
			const std::uint16_t* tl = p.triList;
			const std::uint32_t nidx = static_cast<std::uint32_t>(p.triangles) * 3u;
			for (std::uint32_t k = 0; k < nidx; ++k) {
				if (a_iCount >= a_maxI)
					return;
				const std::uint16_t gv = tl[k];
				if (gv >= Nskin)
					continue;   // guard: keep the index inside this caster's skinned [base, base+Nskin) block
				a_idx[a_iCount++] = base + gv;
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

	skinnedRendered = 0;
	skinnedDetached = 0;
	skinnedMaxDetachDist = 0.0f;
	skinnedClampedVerts = 0;
	std::uint32_t vCount = 0, iCount = 0;
	for (size_t regIdx = 0; regIdx < registry.size(); ++regIdx) {
		const auto& e = registry[regIdx];
		if (!e.engineOnly)
			continue;
		RE::BSGeometry* g = e.geom.get();
		if (!g)
			continue;
		const std::uint32_t v0 = vCount;
		const std::uint32_t i0 = iCount;

		// One unified skin path: VSM_SkinCaster handles BOTH the rigid BSTriShape (bind pose +
		// palette) and the BSDynamicTriShape (position from dynamicData) — they differ only in position source.
		// For a dynamic shape the engine writes dynamicData (facegen/expression/morph) ASYNC under this spinlock;
		// read it unlocked and we race the writer -> torn positions. So take the lock around the dynamic read.
		if (auto* dts = g->AsDynamicTriShape(); dts && dts->GetDynamicTrishapeRuntimeData().dynamicData) {
			auto& dr = dts->GetDynamicTrishapeRuntimeData();
			dr.lock.Lock();
			VSM_SkinCaster(g, skinPosed.data(), skinIndices.data(), kMaxSkinVerts, kMaxSkinIdx, vCount, iCount);
			dr.lock.Unlock();
		} else {
			VSM_SkinCaster(g, skinPosed.data(), skinIndices.data(), kMaxSkinVerts, kMaxSkinIdx, vCount, iCount);
		}
		if (iCount <= i0)
			continue;  // nothing produced this caster

		// Posed-vertex sanity — measured SELF-REFERENTIALLY against the posed centroid, NEVER against
		// worldBound.CENTER. That center is the geometry NODE's bound and goes STALE when the skeleton
		// moves the mesh away from it: a displaced/ragdolled NPC can pose cleanly far off its own stale
		// bound. The game draws the mesh from the SAME boneMatrices we pose from, so a COHERENT pose is
		// valid WHEREVER its centroid lands — only a SCATTERED pose is a real skinning explosion. So
		// reference the centroid (position, self-consistent) and the bound RADIUS (size, reliable), and
		// never the bound CENTER (position, can be stale).
		double cx = 0.0, cy = 0.0, cz = 0.0;
		const std::uint32_t n = vCount - v0;
		for (std::uint32_t v = v0; v < vCount; ++v) {
			cx += skinPosed[v].x; cy += skinPosed[v].y; cz += skinPosed[v].z;
		}
		if (n) { cx /= n; cy /= n; cz /= n; }
		const RE::NiPoint3 ctr{ static_cast<float>(cx), static_cast<float>(cy), static_cast<float>(cz) };
		const float R = g->worldBound.radius;

		// Per-vertex explosion clamp: snap any single vertex flung far from the CENTROID back onto it,
		// collapsing its triangles to ~zero area (a stray bone index would otherwise smear a giant sliver).
		// Also track the pose SPREAD (furthest surviving vertex from the centroid).
		const float clampR  = R * kExplosionClampFactor + kExplosionClampBase;
		const float clampR2 = clampR * clampR;
		float spread2 = 0.0f;
		for (std::uint32_t v = v0; v < vCount; ++v) {
			const float ex = skinPosed[v].x - ctr.x, ey = skinPosed[v].y - ctr.y, ez = skinPosed[v].z - ctr.z;
			const float d2 = ex * ex + ey * ey + ez * ez;
			if (d2 > clampR2) { skinPosed[v] = { ctr.x, ctr.y, ctr.z }; ++skinnedClampedVerts; }
			else if (d2 > spread2) { spread2 = d2; }
		}
		const float spread = std::sqrt(spread2);

		// Detached-shadow guard: a coherent pose fits inside the mesh's own bounding sphere, so its spread
		// from the centroid is <= the bound RADIUS. Drop the caster only if the spread exceeds radius by
		// more than the fixed slack (vsm::kDetachMargin) — that means a genuinely SCATTERED pose (a skeleton
		// resolved to garbage / the wrong space), not merely a displaced one. Replaces the old
		// centroid-vs-worldBound.center check, which false-dropped valid shadows whenever the mesh's bound
		// went stale relative to its skeleton.
		const float limit = R + vsm::kDetachMargin;
		if (spread > limit) {
			++skinnedDetached;
			if (spread > skinnedMaxDetachDist) {
				skinnedMaxDetachDist = spread;
				skinnedMaxDetachName = g->name.c_str();
			}
			vCount = v0;  // rewind the fill cursors — discard this caster's verts + indices
			iCount = i0;
			continue;
		}
		++skinnedRendered;
		// Range center/radius for the per-light frustum cull: use the POSED centroid (self-consistent with
		// the verts we just wrote), not the possibly-stale worldBound.center.
		skinnedRanges.push_back({ i0, iCount - i0, ctr, (std::max)(R, spread), static_cast<int>(regIdx) });
	}

	if (vCount == 0 || !device)
		return;
	const std::uint32_t vbBytes = vCount * static_cast<std::uint32_t>(sizeof(DirectX::XMFLOAT3));
	const std::uint32_t ibBytes = iCount * static_cast<std::uint32_t>(sizeof(std::uint32_t));
	if (!skinnedVB || skinnedVBCap < vbBytes) {
		skinnedVB.Reset();
		D3D11_BUFFER_DESC bd{};
		bd.ByteWidth = vbBytes + kBufferHeadroomBytes;
		bd.Usage = D3D11_USAGE_DYNAMIC;
		bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		device->CreateBuffer(&bd, nullptr, &skinnedVB);
		skinnedVBCap = bd.ByteWidth;
	}
	if (!skinnedIB || skinnedIBCap < ibBytes) {
		skinnedIB.Reset();
		D3D11_BUFFER_DESC bd{};
		bd.ByteWidth = ibBytes + kBufferHeadroomBytes;
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

// Per-caster frustum-cull bounding sphere for this frame, from each caster's CURRENT world AABB.
// GetFreshWorldAABB = the cached LOCAL box transformed by the live geom->world — the SAME transform
// RenderDepth draws with — so the sphere lives in the DRAW's space (absolute world) and can't drift from
// the drawn geometry the way the engine's separately-maintained worldBound.center can.
void VirtualShadowMaps::BuildCasterBounds()
{
	casterWorldSphere.assign(registry.size(), DirectX::XMFLOAT4{ 0, 0, 0, -1.0f });  // w < 0 = no sphere -> don't cull
	for (size_t i = 0; i < registry.size(); ++i) {
		RE::BSGeometry* g = registry[i].geom.get();
		if (!g) continue;
		DirectX::XMFLOAT3 worldMin, worldMax;
		if (!GetFreshWorldAABB(g, registry[i].vertexStride, worldMin, worldMax)) continue;  // skinned/no-buffer -> leave w<0

		// Bounding sphere (AABB centre + half-diagonal), in the draw's absolute space.
		const float hx = 0.5f * (worldMax.x - worldMin.x), hy = 0.5f * (worldMax.y - worldMin.y), hz = 0.5f * (worldMax.z - worldMin.z);
		casterWorldSphere[i] = { 0.5f * (worldMin.x + worldMax.x), 0.5f * (worldMin.y + worldMax.y), 0.5f * (worldMin.z + worldMax.z),
		    std::sqrt(hx * hx + hy * hy + hz * hz) };
	}

	// O10 (always on): (re)build the world-space caster grid from the fresh spheres so each light's cube cull
	// queries only nearby casters (render cull O(#lights), not O(lights x casters)). Draw set is identical to the
	// exhaustive scan (harness-verified) — a strictly-better optimization, so there is no way to turn it off.
	BuildCasterGrid();
}

// O10 tuning. The query half-extent is ADDITIVE (1.02*farPlane + kCasterGridRadiusK*maxCasterRadius), not a flat
// multiple of the radius: SphereInFrustum is a PLANE test, so at a cube face's far-plane corner the two 45deg-tilted
// side planes admit a passing sphere whose CENTER sits up to ~1.414x its radius laterally beyond the true frustum.
// A flat radius multiple drops those casters (verified: a naive 1.10x cube mismatched the exhaustive scan by ~0.2%
// on every scenario); the additive radius term covers them and keeps the candidate set a conservative superset. The
// 1.02 factor covers the ~90.45deg guard-band FOV whose own far corner sits at 1.008*farPlane per axis.
static constexpr float kCasterGridFovGuard = 1.02f;
static constexpr float kCasterGridRadiusK  = 2.0f;   // > 0.414 (the 1.414x lateral push minus the radius itself); generous.

static int VSM_FloorDiv(float v, float cell) { return static_cast<int>(std::floor(v / cell)); }

// Bin every caster with a valid world sphere into casterGrid, keyed by the cells its sphere-AABB spans, and record
// the largest radius (sizes the query bound). O(#casters); rebuilt each frame (the harness timed build+query
// together and still measured the O(#lights) win). NOTE: a future refinement builds a STATIC grid once per cell and
// only re-bins dynamics; this per-frame rebuild is the simplest form that matches the verified algorithm.
void VirtualShadowMaps::BuildCasterGrid()
{
	casterGrid.clear();
	if (casterQuerySeen.size() < registry.size())
		casterQuerySeen.assign(registry.size(), 0);  // grow the stamp-dedup array with the registry
	const float cell = casterGrid.cell;
	for (std::uint32_t i = 0; i < casterWorldSphere.size(); ++i) {
		const auto& s = casterWorldSphere[i];
		if (s.w < 0.0f)
			continue;  // no sphere (skinned/engineOnly) -> not gridded; those draw via skinnedRanges
		if (s.w > casterGrid.maxRadius)
			casterGrid.maxRadius = s.w;
		const int x0 = VSM_FloorDiv(s.x - s.w, cell), x1 = VSM_FloorDiv(s.x + s.w, cell);
		const int y0 = VSM_FloorDiv(s.y - s.w, cell), y1 = VSM_FloorDiv(s.y + s.w, cell);
		const int z0 = VSM_FloorDiv(s.z - s.w, cell), z1 = VSM_FloorDiv(s.z + s.w, cell);
		for (int x = x0; x <= x1; ++x)
			for (int y = y0; y <= y1; ++y)
				for (int z = z0; z <= z1; ++z)
					casterGrid.cells[CasterGrid::Key(x, y, z)].push_back(i);
	}
}

// Gather the registry indices of casters near a light: the cells its query cube (see kCasterGridFovGuard) overlaps,
// stamp-deduped into a_out. A conservative superset of every caster that could pass any of the light's 6 face
// frustums, so the caller's per-face SphereInFrustum refine reproduces the exhaustive scan's drawn set exactly.
void VirtualShadowMaps::QueryCasterGrid(const LightRecord& a_light, std::vector<std::uint32_t>& a_out)
{
	a_out.clear();
	if (++casterQueryStamp == 0) {  // stamp wrapped (2^32 queries) -> reset so stale marks can't false-match
		std::fill(casterQuerySeen.begin(), casterQuerySeen.end(), 0u);
		casterQueryStamp = 1;
	}
	const float cell = casterGrid.cell;
	const float qr   = a_light.farPlane * kCasterGridFovGuard + kCasterGridRadiusK * casterGrid.maxRadius;
	const int x0 = VSM_FloorDiv(a_light.positionWS.x - qr, cell), x1 = VSM_FloorDiv(a_light.positionWS.x + qr, cell);
	const int y0 = VSM_FloorDiv(a_light.positionWS.y - qr, cell), y1 = VSM_FloorDiv(a_light.positionWS.y + qr, cell);
	const int z0 = VSM_FloorDiv(a_light.positionWS.z - qr, cell), z1 = VSM_FloorDiv(a_light.positionWS.z + qr, cell);
	for (int x = x0; x <= x1; ++x)
		for (int y = y0; y <= y1; ++y)
			for (int z = z0; z <= z1; ++z) {
				auto it = casterGrid.cells.find(CasterGrid::Key(x, y, z));
				if (it == casterGrid.cells.end())
					continue;
				for (std::uint32_t idx : it->second)
					if (casterQuerySeen[idx] != casterQueryStamp) {
						casterQuerySeen[idx] = casterQueryStamp;
						a_out.push_back(idx);
					}
			}
}

// Walk a scene-graph node's parent chain and return the FIRST ancestor carrying a TESObjectREFR (the ref the
// node actually hangs under). For a light this catches an ATTACHED light whose GetUserData ref differs from
// the mesh's ref — exactly the case where the ref-cull would silently miss. Render-path helper (called from
// CollectLights every frame), so it lives in the core unit, not the diagnostics one.
RE::FormID VirtualShadowMaps::NodeOwningRef(RE::NiAVObject* a_node)
{
	for (RE::NiAVObject* n = a_node; n; n = n->parent) {
		if (auto* r = n->GetUserData()) return r->GetFormID();
	}
	return 0;
}

// CURRENT world AABB, correct for moving/animated geometry: cache the invariant LOCAL box (computed once) and
// transform its 8 corners by the geometry's CURRENT world matrix every call. Transforming the box (not the
// verts) is conservative — the result contains the true rotated mesh. Cheap: 8 points per caster, no per-frame
// vertex reads; the LOCAL box is read back and cached once per mesh. Render-path helper (BuildCasterBounds).
bool VirtualShadowMaps::GetFreshWorldAABB(RE::BSGeometry* a_geom, std::uint32_t a_stride,
    DirectX::XMFLOAT3& a_worldMin, DirectX::XMFLOAT3& a_worldMax)
{
	if (!a_geom) return false;
	DirectX::XMFLOAT3 lmn, lmx;
	auto it = localAABBCache.find(a_geom);
	if (it != localAABBCache.end()) {
		lmn = it->second.first; lmx = it->second.second;
	} else {
		DirectX::XMFLOAT3 wmn0, wmx0;
		if (!ComputeVertAABB(a_geom, a_stride, lmn, lmx, wmn0, wmx0)) return false;
		if (localAABBCache.size() > kLocalAABBCacheCap) localAABBCache.clear();
		localAABBCache.emplace(a_geom, std::make_pair(lmn, lmx));
	}
	const DirectX::XMMATRIX W = vsm::internal::NiTransformToXM(a_geom->world);  // CURRENT transform this frame
	bool first = true;
	for (int c = 0; c < 8; ++c) {
		DirectX::XMVECTOR p = DirectX::XMVectorSet((c & 1) ? lmx.x : lmn.x, (c & 2) ? lmx.y : lmn.y, (c & 4) ? lmx.z : lmn.z, 1.0f);
		DirectX::XMFLOAT3 pw; DirectX::XMStoreFloat3(&pw, DirectX::XMVector3TransformCoord(p, W));
		if (first) { a_worldMin = a_worldMax = pw; first = false; }
		else {
			a_worldMin.x = (std::min)(a_worldMin.x, pw.x); a_worldMin.y = (std::min)(a_worldMin.y, pw.y); a_worldMin.z = (std::min)(a_worldMin.z, pw.z);
			a_worldMax.x = (std::max)(a_worldMax.x, pw.x); a_worldMax.y = (std::max)(a_worldMax.y, pw.y); a_worldMax.z = (std::max)(a_worldMax.z, pw.z);
		}
	}
	return true;
}

// Read a caster's WHOLE vertex buffer once (via a reused staging copy) and return its LOCAL and WORLD
// vertex AABB. The local AABB reveals a mesh's intrinsic shape (a flat plane has one ~0 axis, which a
// bounding SPHERE can never show); the world AABB is a tight box occluder. Called by GetFreshWorldAABB
// (render path) and the diagnostics registry dump. Returns false if the buffer can't be read.
bool VirtualShadowMaps::ComputeVertAABB(RE::BSGeometry* a_geom, std::uint32_t a_stride,
    DirectX::XMFLOAT3& a_localMin, DirectX::XMFLOAT3& a_localMax,
    DirectX::XMFLOAT3& a_worldMin, DirectX::XMFLOAT3& a_worldMax)
{
	using namespace DirectX;
	if (!a_geom || !a_stride || !device || !context) return false;
	auto* rd = a_geom->GetGeometryRuntimeData().rendererData;
	if (!rd || !rd->vertexBuffer) return false;
	auto* src = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
	D3D11_BUFFER_DESC bd{}; src->GetDesc(&bd);
	const UINT want = bd.ByteWidth;
	if (want < a_stride) return false;
	if (!aabbStaging || aabbStagingSize < want) {                 // grow the reused staging buffer
		aabbStaging.Reset();
		D3D11_BUFFER_DESC sd{}; sd.ByteWidth = want; sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		if (FAILED(device->CreateBuffer(&sd, nullptr, &aabbStaging))) { aabbStagingSize = 0; return false; }
		aabbStagingSize = want;
	}
	D3D11_BOX box{ 0, 0, 0, want, 1, 1 };
	context->CopySubresourceRegion(aabbStaging.Get(), 0, 0, 0, 0, src, 0, &box);
	D3D11_MAPPED_SUBRESOURCE m{};
	if (FAILED(context->Map(aabbStaging.Get(), 0, D3D11_MAP_READ, 0, &m))) return false;
	const XMMATRIX world = NiTransformToXM(a_geom->world);
	XMFLOAT3 lmn{ FLT_MAX, FLT_MAX, FLT_MAX }, lmx{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	XMFLOAT3 wmn{ FLT_MAX, FLT_MAX, FLT_MAX }, wmx{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	const UINT nv = want / a_stride;
	const auto* base = static_cast<const uint8_t*>(m.pData);
	for (UINT v = 0; v < nv; ++v) {
		const float* f = reinterpret_cast<const float*>(base + static_cast<size_t>(v) * a_stride);
		lmn.x = (std::min)(lmn.x, f[0]); lmn.y = (std::min)(lmn.y, f[1]); lmn.z = (std::min)(lmn.z, f[2]);
		lmx.x = (std::max)(lmx.x, f[0]); lmx.y = (std::max)(lmx.y, f[1]); lmx.z = (std::max)(lmx.z, f[2]);
		XMFLOAT3 wp; XMStoreFloat3(&wp, XMVector3TransformCoord(XMVectorSet(f[0], f[1], f[2], 1.0f), world));
		wmn.x = (std::min)(wmn.x, wp.x); wmn.y = (std::min)(wmn.y, wp.y); wmn.z = (std::min)(wmn.z, wp.z);
		wmx.x = (std::max)(wmx.x, wp.x); wmx.y = (std::max)(wmx.y, wp.y); wmx.z = (std::max)(wmx.z, wp.z);
	}
	context->Unmap(aabbStaging.Get(), 0);
	a_localMin = lmn; a_localMax = lmx; a_worldMin = wmn; a_worldMax = wmx;
	return true;
}

// One static caster into the current cube face. Extracted from RenderDepth's face loop; the per-iteration
// values (caster index i, faceVP, frustum planes, blinkOff) are parameters, everything else is direct
// member access. Early-returns replace the loop's continue checks.
// One place for the per-face atlas viewport + face view-proj + frustum-plane math, shared by RenderCasterPass
// (opaque depth), BakeOneLightStatic (P4 static bake) and RenderTranslucentPass (A5). See the header.
void VirtualShadowMaps::ForEachLightFace(const LightRecord& a_light,
    const std::function<void(const DirectX::XMMATRIX&, const DirectX::XMFLOAT4*)>& a_body)
{
	const int blockX = static_cast<int>(a_light.atlasX);
	const int blockY = static_cast<int>(a_light.atlasY);
	const int lfr    = static_cast<int>(a_light.positionWS.w);   // this light's variable-resolution cube-face size
	for (int f = 0; f < 6; ++f) {
		const int col = f % vsm::kCubeFacesWide, row = f / vsm::kCubeFacesWide;   // position in the 3x2 face block
		const D3D11_VIEWPORT vp{ static_cast<float>(blockX + col * lfr), static_cast<float>(blockY + row * lfr),
			static_cast<float>(lfr), static_cast<float>(lfr), 0.0f, 1.0f };
		context->RSSetViewports(1, &vp);
		curFaceViewport = vp;  // P1a: a deferred (batched) draw restores this face's viewport at submit time
		XMFLOAT4 planes[6];
		ExtractFrustumPlanes(a_light.cubeVP[f], planes);
		a_body(XMLoadFloat4x4(&a_light.cubeVP[f]), planes);
	}
}

// Shared tail of every buffer-per-caster draw (see header): push the per-draw cbuffer, bind buffers, draw.
// A6: create/reuse the alpha-tested input layout (POSITION@0 R32G32B32_FLOAT + TEXCOORD@uvOffset R16G16_FLOAT)
// for a given UV byte offset. Returns null on failure -> caller falls back to the solid path (fail-safe).
ID3D11InputLayout* VirtualShadowMaps::GetAlphaLayout(std::uint32_t a_uvOffset)
{
	if (!alphaVSBytecode || !device)
		return nullptr;
	if (auto it = alphaLayouts.find(a_uvOffset); it != alphaLayouts.end())
		return it->second.Get();
	const D3D11_INPUT_ELEMENT_DESC ie[2] = {
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,          D3D11_INPUT_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R16G16_FLOAT,    0, a_uvOffset, D3D11_INPUT_PER_VERTEX_DATA, 0 },
	};
	ComPtr<ID3D11InputLayout> layout;
	if (FAILED(device->CreateInputLayout(ie, 2, alphaVSBytecode->GetBufferPointer(), alphaVSBytecode->GetBufferSize(), &layout)))
		layout.Reset();  // cache the null too -> don't retry a bad offset every frame
	alphaLayouts[a_uvOffset] = layout;
	return layout.Get();
}

void VirtualShadowMaps::EmitCasterDraw(size_t a_i, ID3D11Buffer* a_vb, ID3D11Buffer* a_ib,
    const DirectX::XMMATRIX& a_wvp, const DirectX::XMFLOAT4& a_transmittance)
{
	D3D11_MAPPED_SUBRESOURCE m{};
	if (SUCCEEDED(context->Map(perDrawCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) {
		auto* pd = reinterpret_cast<PerDrawCB*>(m.pData);
		XMStoreFloat4x4(&pd->WorldViewProj, a_wvp);
		pd->CasterId       = static_cast<uint32_t>(a_i) + 1u;   // id-atlas: registry index + 1 (0 = empty)
		pd->AlphaThreshold = registry[a_i].alphaThreshold;      // A6 cutout clip threshold (ignored by opaque/id/trans PS)
		pd->Transmittance  = a_transmittance;                   // A5 glass tint; ignored by the depth/id shaders
		context->Unmap(perDrawCB.Get(), 0);
	}
	UINT stride = registry[a_i].vertexStride, offset = 0;
	context->IASetVertexBuffers(0, 1, &a_vb, &stride, &offset);
	context->IASetIndexBuffer(a_ib, DXGI_FORMAT_R16_UINT, 0);
	context->DrawIndexed(registry[a_i].indexCount, 0, 0);
}

void VirtualShadowMaps::DrawStaticCaster(size_t i, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6], bool blinkOff)
{
	if (isolateCaster > 0 && static_cast<int>(i) != isolateCaster - 1)
		return;  // debug: render only one mesh
	if (blinkOff && static_cast<int>(i) == flashCaster)
		return;  // flash: this caster is on the "off" blink phase
	if (registry[i].isTranslucent)
		return;  // A5: glass/alpha casters render into the transmittance atlas (RenderTranslucentPass), never the opaque depth atlas
	RE::BSGeometry* geom = registry[i].geom.get();
	if (!geom)
		return;
	auto* rd = geom->GetGeometryRuntimeData().rendererData;
	if (!rd || !rd->vertexBuffer || !rd->indexBuffer)
		return;  // streamed out / buffers freed -> skip safely

	// Frustum-cull sphere: from casterWorldSphere (GetFreshWorldAABB = the SAME geom->world we draw
	// with), NOT geom->worldBound.center — that is a separately-maintained value that can drift from
	// the render transform (it went stale for skinned casters), which is the space ambiguity we're
	// removing. This sphere is in the draw's absolute space.
	const DirectX::XMFLOAT4& cs = (i < casterWorldSphere.size()) ? casterWorldSphere[i]
	                                                             : DirectX::XMFLOAT4{ 0, 0, 0, -1.0f };
	if (cs.w >= 0.0f && !SphereInFrustum(planes, RE::NiPoint3{ cs.x, cs.y, cs.z }, cs.w))
		return;

	// A6: alpha-tested cutout? It needs a per-draw pipeline switch (UV/clip shaders + diffuse SRV), so it can't
	// batch — those stay on the immediate path. Every prerequisite is re-checked; any miss keeps the solid path.
	ID3D11InputLayout* alphaLayout = nullptr;
	const bool useAlpha = registry[i].alphaTested && registry[i].diffuseSRV &&
	                      alphaVS && alphaPS && (alphaLayout = GetAlphaLayout(registry[i].uvUVOffset)) != nullptr;

	const DirectX::XMMATRIX wvp = XMMatrixMultiply(NiTransformToXM(geom->world), faceVP);

	// P1a: SOLID caster -> defer into the batch (one shared cbuffer, offset-bound at submit). Depth-only draws are
	// order-independent, so this is output-identical. Raster state (twoSided) + viewport are recorded per entry.
	if (drawBatchActive && !useAlpha) {
		BatchedDraw& bd = drawBatch.emplace_back();
		bd.vb         = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
		bd.ib         = reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer);
		bd.indexCount = registry[i].indexCount;
		bd.stride     = registry[i].vertexStride;
		bd.twoSided   = registry[i].twoSided;
		bd.vp         = curFaceViewport;
		XMStoreFloat4x4(&bd.wvp, wvp);
		bd.casterId       = static_cast<std::uint32_t>(i) + 1u;
		bd.alphaThreshold = registry[i].alphaThreshold;
		++visibleCasters;
		return;
	}

	// Immediate path (alpha-tested casters, or no D3D11.1 context). Two-sided casters draw with CULL_NONE so their
	// away-facing side still writes depth.
	context->RSSetState(registry[i].twoSided ? rasterStateNoCull.Get() : rasterState.Get());
	if (useAlpha) {
		context->VSSetShader(alphaVS.Get(), nullptr, 0);
		context->PSSetShader(alphaPS.Get(), nullptr, 0);
		context->IASetInputLayout(alphaLayout);
		context->RSSetState(rasterStateNoCull.Get());  // cutouts (foliage) are effectively two-sided
		ID3D11ShaderResourceView* diffSrv = registry[i].diffuseSRV;
		context->PSSetShaderResources(0, 1, &diffSrv);
		context->PSSetSamplers(0, 1, alphaSampler.GetAddressOf());
	}
	EmitCasterDraw(i, reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer), reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer),
	    wvp, XMFLOAT4{ 1.0f, 1.0f, 1.0f, 1.0f });  // opaque: transmittance unused
	if (useAlpha) {  // restore the solid pipeline for the next caster
		context->VSSetShader(depthVS.Get(), nullptr, 0);
		context->PSSetShader(idPS.Get(), nullptr, 0);
		context->IASetInputLayout(ilFull.Get());
		ID3D11ShaderResourceView* nullSRV = nullptr;
		context->PSSetShaderResources(0, 1, &nullSRV);
	}
	++visibleCasters;
}

// P1a: flush every deferred solid caster draw. Fill ONE dynamic cbuffer (one Map) with all per-draw constants at
// 256-byte slots, then submit each draw binding just its 16-constant window via VSSetConstantBuffers1 — replacing
// the per-draw Map(perDrawCB). The solid pipeline (depthVS + idPS/null PS + ilFull) is already bound by the caller;
// we only restore each entry's viewport + raster state (deduped) before its draw. Depth-only => order-independent.
void VirtualShadowMaps::SubmitDrawBatch()
{
	if (drawBatch.empty() || !context1)
		return;

	// Grow the slot buffer to fit this frame's draw count (round up to reduce reallocation churn).
	const UINT need = static_cast<UINT>(drawBatch.size());
	if (perDrawCBArrayCount < need) {
		const UINT cap = (need + 255u) & ~255u;  // round up to a multiple of 256 slots
		D3D11_BUFFER_DESC bd{};
		bd.ByteWidth      = cap * kPerDrawSlotBytes;
		bd.Usage          = D3D11_USAGE_DYNAMIC;
		bd.BindFlags      = D3D11_BIND_CONSTANT_BUFFER;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		perDrawCBArray.Reset();
		if (FAILED(device->CreateBuffer(&bd, nullptr, &perDrawCBArray))) {
			perDrawCBArrayCount = 0;
			return;  // allocation failed -> skip the batch this frame (solids simply don't draw; recovers next frame)
		}
		perDrawCBArrayCount = cap;
	}

	// One Map: write every draw's PerDrawCB into its 256-byte slot.
	D3D11_MAPPED_SUBRESOURCE m{};
	if (FAILED(context1->Map(perDrawCBArray.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m)))
		return;
	auto* base = static_cast<std::uint8_t*>(m.pData);
	for (size_t k = 0; k < drawBatch.size(); ++k) {
		auto* pd = reinterpret_cast<PerDrawCB*>(base + k * kPerDrawSlotBytes);
		pd->WorldViewProj  = drawBatch[k].wvp;
		pd->CasterId       = drawBatch[k].casterId;
		pd->AlphaThreshold = drawBatch[k].alphaThreshold;
		pd->Transmittance  = { 1.0f, 1.0f, 1.0f, 1.0f };  // solid opaque: unused by depth/id shaders
	}
	context1->Unmap(perDrawCBArray.Get(), 0);

	// Submit. Dedup viewport + raster-state changes; bind each draw's 16-constant window to VS (and PS, for the id
	// atlas on dump frames). curVp starts invalid so the first entry always sets its viewport.
	ID3D11Buffer* arr = perDrawCBArray.Get();
	const UINT zeroOffset = 0;
	D3D11_VIEWPORT curVp{ -1, -1, -1, -1, -1, -1 };
	int curTwoSided = -1;  // -1 = unset -> first draw sets raster state
	for (size_t k = 0; k < drawBatch.size(); ++k) {
		const BatchedDraw& d = drawBatch[k];
		if (d.vp.TopLeftX != curVp.TopLeftX || d.vp.TopLeftY != curVp.TopLeftY ||
		    d.vp.Width != curVp.Width || d.vp.Height != curVp.Height) {
			context->RSSetViewports(1, &d.vp);
			curVp = d.vp;
		}
		if (static_cast<int>(d.twoSided) != curTwoSided) {
			context->RSSetState(d.twoSided ? rasterStateNoCull.Get() : rasterState.Get());
			curTwoSided = static_cast<int>(d.twoSided);
		}
		const UINT first = static_cast<UINT>(k) * kPerDrawSlotConstants;  // 16-constant units
		const UINT num   = kPerDrawSlotConstants;
		context1->VSSetConstantBuffers1(0, 1, &arr, &first, &num);
		context1->PSSetConstantBuffers1(0, 1, &arr, &first, &num);  // idPS reads CasterId (b0) on dump frames
		context->IASetVertexBuffers(0, 1, &d.vb, &d.stride, &zeroOffset);
		context->IASetIndexBuffer(d.ib, DXGI_FORMAT_R16_UINT, 0);
		context->DrawIndexed(d.indexCount, 0, 0);
	}
	drawBatch.clear();
}

// One skinned range into the current cube face. Extracted verbatim from RenderDepth's face loop; the
// shared skinned VB/IB are bound once per face by the caller, only the per-range body lives here.
void VirtualShadowMaps::DrawSkinnedRange(const SkinnedRange& r, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6], bool blinkOff)
{
	if (blinkOff && r.registryIndex == flashCaster)
		return;  // flash: this skinned caster is on the "off" blink phase
	if (isolateCaster > 0 && r.registryIndex != isolateCaster - 1)
		return;  // debug: render only one mesh (skinned)
	// Skinned casters are only frustum-culled (r.center = the posed centroid, draw-consistent).
	if (r.radius > 0.0f && !SphereInFrustum(planes, r.center, r.radius))
		return;
	// posed skinned verts ARE world -> WVP = cube VP (shared); update CasterId per range for the id-atlas.
	D3D11_MAPPED_SUBRESOURCE sm{};
	if (SUCCEEDED(context->Map(perDrawCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &sm))) {
		auto* pd = reinterpret_cast<PerDrawCB*>(sm.pData);
		XMStoreFloat4x4(&pd->WorldViewProj, faceVP);  // posed verts already absolute -> straight through the cube VP
		pd->CasterId = static_cast<uint32_t>(r.registryIndex) + 1u;
		context->Unmap(perDrawCB.Get(), 0);
	}
	context->DrawIndexed(r.indexCount, r.ibStart, 0);
	++visibleCasters;
}

// Draw casters into the CURRENTLY-BOUND dsv/viewport, looping all lights x 6 cube faces at each light's OWN
// faceRes (variable resolution). filter: 0=all, 1=static-only, 2=dynamic-only. drawSkinned adds the
// CPU-skinned character ranges (always dynamic). Pipeline state (VS/PS/IL/CB/raster/depth) must be set by the
// caller. SPACE: DRAW and CULL are ONE absolute-world space (cubeVP from absolute light pos; static
// geom->world is game-absolute; skinned verts posed to absolute; cull sphere from the SAME geom->world).
void VirtualShadowMaps::RenderCasterPass(int a_filter, bool a_drawSkinned)
{
	const bool blinkOff = (flashCaster >= 0) && (((frameIndex / kFlashHalfPeriodFrames) % 2u) == 1u);
	// P1a: defer SOLID caster draws into drawBatch this pass, submitted in one batch below (when a D3D11.1 context
	// is available). Skinned/glass/alpha draws stay immediate and interleave harmlessly (depth is order-independent).
	drawBatchActive = (context1 != nullptr);
	drawBatch.clear();
	// O4 (always on): skip a light's whole 6-face pass when NO caster this pass would draw lies within the light's
	// radius — a CPU sphere-vs-radius pre-check over exactly what the pass draws (opaque buffer casters matching the
	// filter via casterWorldSphere; skinned ranges when a_drawSkinned). Conservative: culls only when the pass is empty.
	// O10 (always on): each light iterates only NEARBY casters via the world-space grid (cand) instead of the whole
	// registry. `cand == nullptr` means "iterate all of registry" (fallback when the grid is empty). The candidate
	// set is a conservative superset, so DrawStaticCaster's per-face SphereInFrustum refine yields the identical set.
	const bool useGrid = !casterGrid.cells.empty();
	for (const auto& light : lightRecords) {
		const std::vector<std::uint32_t>* cand = nullptr;
		if (useGrid) {
			QueryCasterGrid(light, casterQueryScratch);
			cand = &casterQueryScratch;
		}
		// Run `body(i)` over the candidate registry indices (grid) or the whole registry (brute), identically.
		auto forEachCaster = [&](auto&& body) {
			if (cand) { for (std::uint32_t i : *cand) body(static_cast<size_t>(i)); }
			else      { for (size_t i = 0; i < registry.size(); ++i) body(i); }
		};
		{  // O4 (always on): skip this light's whole 6-face pass if no caster it would draw is within radius
			const float lx = light.positionWS.x, ly = light.positionWS.y, lz = light.positionWS.z;
			const float lr = light.farPlane;  // far plane == light radius
			bool any = false;
			forEachCaster([&](size_t i) {
				if (any) return;
				if (registry[i].isTranslucent) return;                   // glass never draws in the opaque passes
				if (a_filter == 1 && !registry[i].isStatic) return;
				if (a_filter == 2 && registry[i].isStatic) return;
				const auto& s = casterWorldSphere[i];
				if (s.w < 0.0f) return;                                   // no sphere (skinned/engineOnly) -> checked below
				const float dx = s.x - lx, dy = s.y - ly, dz = s.z - lz;
				if (std::sqrt(dx * dx + dy * dy + dz * dz) - s.w <= lr) any = true;
			});
			if (!any && a_drawSkinned)
				for (const auto& r : skinnedRanges) {
					const float dx = r.center.x - lx, dy = r.center.y - ly, dz = r.center.z - lz;
					if (std::sqrt(dx * dx + dy * dy + dz * dz) - r.radius <= lr) { any = true; break; }
				}
			if (!any)
				continue;  // no caster reaches this light -> skip all 6 faces (no viewport/matrix/clear/draw)
		}
		ForEachLightFace(light, [&](const XMMATRIX& faceVP, const XMFLOAT4* planes) {
			forEachCaster([&](size_t i) {
				if (a_filter == 1 && !registry[i].isStatic) return;     // static-only pass
				if (a_filter == 2 &&  registry[i].isStatic) return;     // dynamic-only pass
				DrawStaticCaster(i, faceVP, planes, blinkOff);
			});
			// --- Path B: skinned characters (world-absolute posed buffer -> same cube VP, world=identity) ---
			// These stay IMMEDIATE and run here with b0 still bound to perDrawCB (the batch rebinds b0 only in the
			// flush below, AFTER this whole light loop) — so the skinned Map(perDrawCB) draws are unaffected.
			if (a_drawSkinned && skinnedVB && skinnedIB && !skinnedRanges.empty()) {
				ID3D11Buffer* svb = skinnedVB.Get();
				UINT sstride = static_cast<UINT>(sizeof(XMFLOAT3)), soff = 0;
				context->IASetVertexBuffers(0, 1, &svb, &sstride, &soff);
				context->IASetIndexBuffer(skinnedIB.Get(), DXGI_FORMAT_R32_UINT, 0);
				for (const auto& r : skinnedRanges)
					DrawSkinnedRange(r, faceVP, planes, blinkOff);
			}
		});
	}
	// P1a: flush the deferred solid casters (one cbuffer fill + offset-bound submit). No-op if batching is off.
	if (drawBatchActive) {
		SubmitDrawBatch();
		drawBatchActive = false;
	}
}

// P4: clear ONE light's atlas block to far-Z and re-render only ITS static casters into the static cache. The caller
// has bound staticDsv (depth-only). A viewport-confined fullscreen triangle (fullscreenVS emits z=0 = reverse-Z far)
// resets just this light's 3x2 block WITHOUT a whole-atlas clear; its static casters then draw GREATER over it, and
// bakedPose is refreshed so future cache-HOLD frames sample this block at the pose it was baked with.
void VirtualShadowMaps::BakeOneLightStatic(size_t a_lightIdx)
{
	if (a_lightIdx >= lightRecords.size())
		return;
	const LightRecord& light = lightRecords[a_lightIdx];
	const int blockX = static_cast<int>(light.atlasX);
	const int blockY = static_cast<int>(light.atlasY);
	const int lfr    = static_cast<int>(light.positionWS.w);

	// 1. Clear this block to far-Z. fullscreenVS emits z=0, and reverse-Z is committed, so 0 == far (asserted
	//    below — if kReverseZ were ever flipped this clear would wrongly reset the block to NEAR). The block
	//    viewport clips the fullscreen triangle to exactly this light's 3x2 region.
	static_assert(kReverseZ, "BakeOneLightStatic clears via fullscreenVS z=0; that is 'far' only under reverse-Z");
	const D3D11_VIEWPORT bvp{ static_cast<float>(blockX), static_cast<float>(blockY),
		static_cast<float>(lfr * vsm::kCubeFacesWide), static_cast<float>(lfr * vsm::kCubeFacesTall), 0.0f, 1.0f };
	context->RSSetViewports(1, &bvp);
	context->OMSetDepthStencilState(depthClearState.Get(), 0);   // ALWAYS + write
	context->RSSetState(rasterState.Get());
	context->VSSetShader(fullscreenVS.Get(), nullptr, 0);
	context->PSSetShader(nullptr, nullptr, 0);
	context->IASetInputLayout(nullptr);
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->Draw(3, 0);

	// 2. Render this light's STATIC opaque casters into its 6 faces (depth-only, GREATER over the cleared block).
	context->OMSetDepthStencilState(depthState.Get(), 0);
	context->VSSetShader(depthVS.Get(), nullptr, 0);
	context->IASetInputLayout(ilFull.Get());
	auto* cb = perDrawCB.Get();
	context->VSSetConstantBuffers(0, 1, &cb);
	ForEachLightFace(light, [&](const XMMATRIX& faceVP, const XMFLOAT4* planes) {
		for (size_t i = 0; i < registry.size(); ++i) {
			if (!registry[i].isStatic || registry[i].isTranslucent)
				continue;  // static opaque casters only (DrawStaticCaster also skips translucent)
			DrawStaticCaster(i, faceVP, planes, false);
		}
	});

	// 3. Refresh this light's baked pose so subsequent HOLD frames pose-freeze to it; clear its dirty flag.
	BakedPose bp;
	for (int f = 0; f < 6; ++f) bp.cubeVP[f] = light.cubeVP[f];
	bp.pos = { light.positionWS.x, light.positionWS.y, light.positionWS.z };
	const void* id = (a_lightIdx < lightPtrs.size()) ? lightPtrs[a_lightIdx] : nullptr;
	bakedPose[id] = bp;
	dirtyLights.erase(id);
}

void VirtualShadowMaps::RenderDepth()
{
	visibleCasters = 0;
	if (!dsv)
		return;

	GraphicsStateGuard guard(context);

	// Occluder-id atlas is only needed on a DUMP frame; normal frames stay depth-only. It renders in the
	// DYNAMIC/live pass only (static-cached ids aren't tracked — a diagnostic nuance under P2 caching).
	const bool wantId = idRTV && idPS && dbgLogRequested;
	ID3D11RenderTargetView* rtvBind = wantId ? idRTV.Get() : nullptr;

	// Unbind the atlas SRV (t110) before we write it as DSV — read/write hazard; LLF re-binds it next Prepass.
	ID3D11ShaderResourceView* nullSRV[1] = { nullptr };
	context->PSSetShaderResources(vsm::kShadowAtlasSRVSlot, 1, nullSRV);
	context->CSSetShaderResources(vsm::kShadowAtlasSRVSlot, 1, nullSRV);

	// A5: reset the transmittance atlas to WHITE every frame (glass renders over it below only when the module is on).
	// Always-white keeps the always-bound t112 a no-op multiply when A5 is off. One cheap RGBA clear.
	if (transRTV) {
		const float white[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
		context->ClearRenderTargetView(transRTV.Get(), white);
	}

	// Shared depth-only pipeline state (both P2 passes + the non-cached path).
	auto setPipeline = [&](bool withId) {
		context->RSSetState(rasterState.Get());
		context->OMSetDepthStencilState(depthState.Get(), 0);
		context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		context->VSSetShader(depthVS.Get(), nullptr, 0);
		context->PSSetShader(withId ? idPS.Get() : nullptr, nullptr, 0);
		context->IASetInputLayout(ilFull.Get());  // position is always R32G32B32_FLOAT
		auto* cb = perDrawCB.Get();
		context->VSSetConstantBuffers(0, 1, &cb);
		if (withId) context->PSSetConstantBuffers(0, 1, &cb);
	};
	const float clearZ = kReverseZ ? 0.0f : 1.0f;
	const float zeros[4] = { 0, 0, 0, 0 };
	const bool  cache    = vsm::GetConfig().cacheStaticShadows && EnsureStaticCache();

	if (!cache) {
		// --- P2 OFF: single pass. Clear the live atlas, draw ALL casters + skinned (baseline behaviour). ---
		context->OMSetRenderTargets(1, &rtvBind, dsv.Get());
		context->ClearDepthStencilView(dsv.Get(), D3D11_CLEAR_DEPTH, clearZ, 0);
		if (wantId) context->ClearRenderTargetView(idRTV.Get(), zeros);
		if (!haveTestLight || registry.empty())
			return;
		setPipeline(wantId);
		RenderCasterPass(0, true);
		RenderTranslucentPass();  // A5: glass over the completed opaque atlas (depth-tested vs it)
		return;
	}

	// --- P2 ON: static/dynamic caching. (Invalidation + pose-freeze were finalized in UpdateStaticCacheState.) ---
	if (!haveTestLight || registry.empty()) {
		context->OMSetRenderTargets(1, &rtvBind, dsv.Get());
		context->ClearDepthStencilView(dsv.Get(), D3D11_CLEAR_DEPTH, clearZ, 0);   // don't freeze the preview
		return;
	}

	// PASS 1: bake STATIC casters into the static cache.
	if (vsm::GetConfig().incrementalCache) {
		// P4 incremental: bake ONLY the scheduled dirty blocks (bakeThisFrame); every other block persists in the
		// static cache from earlier frames. No whole-atlas clear — BakeOneLightStatic clears each block itself.
		if (!bakeThisFrame.empty()) {
			ID3D11RenderTargetView* noRtv = nullptr;
			context->OMSetRenderTargets(0, &noRtv, staticDsv.Get());
			for (size_t idx : bakeThisFrame)
				BakeOneLightStatic(idx);
		}
	} else if (!staticCacheValid) {
		// P2 whole-cache: re-bake ALL static casters at once when the single validity bit is clear.
		ID3D11RenderTargetView* noRtv = nullptr;
		context->OMSetRenderTargets(0, &noRtv, staticDsv.Get());     // depth-only, no id
		context->ClearDepthStencilView(staticDsv.Get(), D3D11_CLEAR_DEPTH, clearZ, 0);
		setPipeline(false);
		RenderCasterPass(1, false);                                  // static casters only, no skinned
		staticCacheValid      = true;
		lastBakedRegistrySize = static_cast<int>(registry.size());
		// Snapshot the poses this bake used, so future cache-HOLD frames sample the atlas with the SAME pose
		// (UpdateStaticCacheState restores these) — otherwise a sub-threshold light jitter smears the shadow.
		bakedPose.clear();
		for (size_t i = 0; i < lightRecords.size(); ++i) {
			BakedPose bp;
			for (int f = 0; f < 6; ++f) bp.cubeVP[f] = lightRecords[i].cubeVP[f];
			bp.pos = { lightRecords[i].positionWS.x, lightRecords[i].positionWS.y, lightRecords[i].positionWS.z };
			bakedPose[(i < lightPtrs.size()) ? lightPtrs[i] : nullptr] = bp;
		}
	}

	// COMPOSITE: live atlas <- cached static depth (single whole-resource copy; legal for depth). Unbind the
	// static DSV first so the source isn't bound during the copy.
	ID3D11RenderTargetView* noRtv2 = nullptr;
	context->OMSetRenderTargets(0, &noRtv2, nullptr);
	context->CopyResource(depthTex.Get(), staticDepthTex.Get());

	// PASS 2: DYNAMIC casters + skinned over the copied static (NO clear -> keep the static baseline; reverse-Z
	// GREATER depth-test composes dynamics in front of static).
	context->OMSetRenderTargets(1, &rtvBind, dsv.Get());
	if (wantId) context->ClearRenderTargetView(idRTV.Get(), zeros);
	setPipeline(wantId);
	RenderCasterPass(2, true);
	RenderTranslucentPass();  // A5: glass over the completed opaque (static + dynamic) atlas
	// engine state restored by ~GraphicsStateGuard (note: it does NOT save blend — RenderTranslucentPass restores it)
}

// A5: render one glass caster into the transmittance atlas — same WVP as the opaque path, but the PS emits the
// caster's per-channel transmittance under the multiplicative blend. Frustum-culled like the opaque casters; the
// caller has already bound the transmittance RTV + read-only opaque DSV + multiply blend + two-sided raster.
void VirtualShadowMaps::DrawTranslucentCaster(size_t i, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6])
{
	if (i >= registry.size() || !registry[i].isTranslucent)
		return;
	RE::BSGeometry* geom = registry[i].geom.get();
	if (!geom)
		return;
	auto* rd = geom->GetGeometryRuntimeData().rendererData;
	if (!rd || !rd->vertexBuffer || !rd->indexBuffer)
		return;
	const DirectX::XMFLOAT4& cs = (i < casterWorldSphere.size()) ? casterWorldSphere[i]
	                                                             : DirectX::XMFLOAT4{ 0, 0, 0, -1.0f };
	if (cs.w >= 0.0f && !SphereInFrustum(planes, RE::NiPoint3{ cs.x, cs.y, cs.z }, cs.w))
		return;
	EmitCasterDraw(i, reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer), reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer),
	    XMMatrixMultiply(NiTransformToXM(geom->world), faceVP),
	    XMFLOAT4{ registry[i].transR, registry[i].transG, registry[i].transB, 1.0f });  // per-channel light multiplier
	// Not counted in visibleCasters — that tallies OPAQUE caster draws for the menu readout.
}

// A5: render glass/alpha casters into the RGBA transmittance atlas. Runs AFTER the opaque atlas is complete so it can
// depth-test (read-only) against it — glass hidden behind opaque geometry doesn't tint. The multiplicative blend makes
// overlapping panes accumulate ORDER-INDEPENDENTLY (no sorting). The atlas was already cleared white in RenderDepth,
// so with no glass (or the module off) it stays a no-op multiply by 1.
void VirtualShadowMaps::RenderTranslucentPass()
{
	if (!vsm::GetConfig().translucentShadows || !transRTV || !transPS || !dsvReadOnly)
		return;
	if (!haveTestLight || registry.empty())
		return;
	bool hasTranslucent = false;
	for (const auto& c : registry)
		if (c.isTranslucent) { hasTranslucent = true; break; }
	if (!hasTranslucent)
		return;  // no glass -> transmittance atlas stays white (a no-op multiply)

	context->OMSetRenderTargets(1, transRTV.GetAddressOf(), dsvReadOnly.Get());
	context->OMSetBlendState(multiplyBlend.Get(), nullptr, 0xFFFFFFFF);
	context->OMSetDepthStencilState(depthReadOnlyState.Get(), 0);   // GREATER, depth-write OFF — test vs opaque atlas
	context->RSSetState(rasterStateNoCull.Get());                   // glass casts from both faces
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	context->VSSetShader(depthVS.Get(), nullptr, 0);
	context->PSSetShader(transPS.Get(), nullptr, 0);
	context->IASetInputLayout(ilFull.Get());
	auto* cb = perDrawCB.Get();
	context->VSSetConstantBuffers(0, 1, &cb);
	context->PSSetConstantBuffers(0, 1, &cb);

	for (const auto& light : lightRecords)
		ForEachLightFace(light, [&](const XMMATRIX& faceVP, const XMFLOAT4* planes) {
			for (size_t i = 0; i < registry.size(); ++i)
				DrawTranslucentCaster(i, faceVP, planes);
		});

	// Restore default (opaque) blend — GraphicsStateGuard does NOT save blend state, so the engine's next draws would
	// otherwise inherit our multiplicative blend. RTV / depth / raster / VS / PS / viewport are restored by ~guard.
	context->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
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

	menuVisible = false;  // DrawMenu re-sets this next time the menu is drawn

	// Delayed-dump countdown. Runs every frame (menu open or not); once it hits zero we request the
	// dump for THIS frame — so CollectLights below records real camera movement (the menu having been
	// closed by the user to walk/turn) instead of the zero an immediate menu-click always sees.
	if (dumpCountdownFrames > 0 && --dumpCountdownFrames == 0)
		dbgLogRequested = true;

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
	PackAtlas();                    // assign each light's variable-res block origin + grow the atlas to fit
	UpdateStaticCacheState();       // P2: finalize static-cache invalidation + freeze poses if the cache holds
	EnsureLightBuffer(activeLightCount);  // grow the per-light GPU buffer to fit ALL active lights — no cap
	BuildCasterBounds();     // per-caster frustum spheres, in the draw's absolute space
	UploadLightBuffer();     // mirror lightRecords into the GPU buffer for the LLF shader
	UpdateDebugCB();         // push live tuning sliders (bias/near/far/mode) + runtime atlas dims to the shader
	// Path B: skin all characters ONCE this frame into a world-absolute posed buffer (skin-once,
	// rasterize into every light face below). Cheap no-op if there are no skinned casters.
	SkinAllCasters();        // CPU-skin engineOnly casters -> world-absolute posed buffer (consumed by RenderDepth)
	RenderDepth();           // render every light's cube into the atlas (clears if no lights)

}

void VirtualShadowMaps::DrawMenu()
{
	// Drawn in the Community Shaders feature list under the "Lighting" category via the CS
	// add-on hook (shared ImGui context) — invoked only while our entry is selected.
	menuVisible = true;

	// Deployment build: VSM runs automatically; there are no user-facing settings here
	// (any advanced overrides live in VirtualShadowMaps.toml). Attribution only.
	ImGui::SeparatorText("Virtual Shadow Maps");
	ImGui::TextDisabled("Made by PookieToo");
}
