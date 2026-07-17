#pragma once

// ============================================================================
// VirtualShadowMaps — a standalone SKSE plugin.
//
// Each frame (driven by an IDXGISwapChain::Present hook) it renders a cube-shadow
// atlas for every active local light — statics from a persistent scene-graph registry,
// plus CPU-skinned characters (Path B) — and exposes the atlas + per-light buffer to
// Community Shaders' LightLimitFix, whose Lighting.hlsl samples them (VSM.hlsli) so local
// lights beyond the engine's shadow budget still cast shadows.
//
// The settings UI (DrawMenu) draws into the Community Shaders menu via CS's add-on hook
// (CS_RegisterExternalMenu / CS_GetImGui); the plugin's logic never lives in CS's DLL, and
// the DLL shares CS's ImGui context.
// ============================================================================

#include "VSMConstants.h"  // vsm:: atlas geometry + capacities (single source of truth)

#include <DirectXMath.h>
#include <d3d11.h>
#include <d3d11_1.h>   // ID3D11DeviceContext1 for P1a batched-draw offset-bound cbuffer submit
#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace RE
{
	class ShadowSceneNode;
	class NiAVObject;
	class BSGeometry;
}

class VirtualShadowMaps
{
public:
	static VirtualShadowMaps* GetSingleton()
	{
		static VirtualShadowMaps singleton;
		return &singleton;
	}

	void OnD3DReady(ID3D11Device* a_device, ID3D11DeviceContext* a_context);
	void SetRenderSize(std::uint32_t, std::uint32_t a_h) { screenH = static_cast<float>(a_h); }
	void SetSwapChain(IDXGISwapChain* a_sc) { swapChain = a_sc; }  // not owned; used to refresh screenH on a resolution change
	void RenderFrame();  // per-frame, from the Present hook
	void DrawMenu();     // ImGui, called from the CS menu via our registered callback

	// Exposed to Plugin.cpp's C exports so LLF's shader can sample our shadows.
	ID3D11ShaderResourceView* GetAtlasSRV() const { return srv.Get(); }
	ID3D11ShaderResourceView* GetLightBufferSRV() const { return lightBufferSRV.Get(); }
	ID3D11ShaderResourceView* GetTransmittanceSRV() const { return transSRV.Get(); }  // A5: colored transmittance atlas (t112)
	ID3D11SamplerState*       GetPointSampler() const { return pointSampler.Get(); }
	ID3D11Buffer*             GetParamsCB() const { return paramsCB.Get(); }
	int  GetShadowLightCount() const { return static_cast<int>(lightRecords.size()); }
	bool IsEnabled() const { return enabled; }
	// N3: Community Shaders' LightLimitFix::Prepass calls VSM_GetShadowResources every frame it samples our atlas.
	// It stamps this flag so RenderFrame can skip the whole atlas render when nothing is consuming it (LLF off /
	// absent) — otherwise we'd render + hold a full atlas for zero output. Set from the CS render thread, read+reset
	// on our Present tick; atomic so it's safe regardless of which thread CS calls the export on.
	void NotifyResourcesFetched() { resourceFetched.store(true, std::memory_order_relaxed); }
	// LLF light-index alignment (Option c): map a BSLight* to its slot in the ShadowLights (t111) buffer, so
	// LLF's Lighting.hlsl can index our shadow record directly instead of an O(lights²)/pixel position search.
	// Returns kNoShadowIndex when this light has no shadow record this frame (LLF renders it unshadowed — the
	// fail-safe). The map is (re)built atomically with UploadLightBuffer, so it always describes the buffer LLF
	// binds (both consumed one frame later; see docs/ARCHITECTURE §6.1). See [[llf-light-index-coupling]].
	static constexpr std::uint32_t kNoShadowIndex = 0xFFFFFFFFu;
	std::uint32_t GetLightShadowIndex(const void* a_bsLight) const;

private:
	VirtualShadowMaps() = default;

	void SetupResources();
	void CollectLights(RE::ShadowSceneNode* a_ssn);    // gather all active local lights -> lightRecords
	void UploadLightBuffer();                          // mirror lightRecords into the GPU structured buffer
	void UpdateParamsCB();                              // push the live tuning sliders to the shader cbuffer
	void BuildCubeMatrices(DirectX::FXMVECTOR a_eye, float a_near, float a_far, int a_faceRes, DirectX::XMFLOAT4X4 a_outVP[6]);  // 6 cube-face view-projs (per-light near/far; faceRes for the guard-band FOV)
	void RebuildRegistrySafe(RE::NiAVObject* a_obj);            // SEH-guarded entry (traversal can hit bad geometry)
	void RebuildRegistry(RE::NiAVObject* a_obj, int a_depth, RE::FormID a_owner = 0, bool a_underBillboard = false);  // recursive; skips LOD/sky + caps depth; tags casters with owning ref; a_underBillboard = descended through a NiBillboardNode (camera-facing effect -> excluded)
	void RebuildFromReferences();                              // player 3D + loaded-cell references (room/NPCs/clutter)
	void RenderDepth();
	void SkinAllCasters();           // Path B: CPU-skin engineOnly BSTriShape casters -> world-absolute posed buffer
	// Compute a caster's LOCAL and WORLD vertex AABB by reading its whole vertex buffer (reused staging
	// copy). Diagnostic-only; lets the log carry real geometry (flatness / true silhouette) instead of
	// just a bounding sphere. Returns false if the buffer isn't readable.
	bool ComputeVertAABB(RE::BSGeometry* a_geom, std::uint32_t a_stride,
	    DirectX::XMFLOAT3& a_localMin, DirectX::XMFLOAT3& a_localMax,
	    DirectX::XMFLOAT3& a_worldMin, DirectX::XMFLOAT3& a_worldMax);
	void LoadConfig();                 // pull persisted tunables from VirtualShadowMaps.toml at startup
	void PersistSettingsIfChanged();   // write tuning changes back to the config file (once, on control release)

	// A caster references its live geometry via NiPointer (so it can't be freed under us).
	// Buffers, world transform and bound are re-read from `geom` every frame — this keeps
	// dynamic objects correct and avoids dangling GPU pointers. Only the stable stride /
	// index count are cached.
	struct CasterEntry
	{
		RE::NiPointer<RE::BSGeometry> geom;
		uint32_t                      vertexStride = 0;      // 0 for engineOnly (no renderer buffers)
		uint32_t                      indexCount   = 0;      // 0 for engineOnly
		bool                          engineOnly   = false;  // skinned/dynamic: no rendererData; only the
		                                                     // engine BSUtilityShader path can draw it (custom VS can't)
		bool                          twoSided     = false;  // NiAlphaProperty/material two-sided: rasterize with
		                                                     // CULL_NONE so the away-face still casts (else a plane
		                                                     // facing away from the light casts no atlas shadow)
		bool                          isStatic     = false;  // R5 classification: STATIC = never moves (base Static/
		                                                     // StaticCollection, not skinned, no anim controller) ->
		                                                     // safe to CACHE once (P2). Else DYNAMIC (err this way).
		bool                          isTranslucent = false; // A5: glass / alpha-blended caster. Rendered into the
		                                                     // TRANSMITTANCE atlas (dims/tints the light), never the
		                                                     // opaque depth atlas. Skipped by every depth pass.
		float                         transR = 1.0f, transG = 1.0f, transB = 1.0f;  // A5 per-channel transmittance
		                                                     // (1 = clear) = 1 - coverage*(1 - materialTint); what fraction of each light channel passes
		// A6 alpha-tested cutout (always on): when alphaTested, the depth pass samples diffuseSRV's
		// alpha at the vertex UV (byte offset uvByteOffset, R16G16_FLOAT) and clip()s below alphaThreshold -> silhouette.
		// diffuseSRV is a RAW pointer OWNED BY THE GAME (valid while `geom` is alive, which we hold via NiPointer);
		// null / not-alphaTested -> the caster renders as a solid quad (fail-safe).
		bool                          alphaTested   = false;
		ID3D11ShaderResourceView*     diffuseSRV    = nullptr;
		float                         alphaThreshold = 0.0f;   // NiAlphaProperty::alphaThreshold / 255
		std::uint32_t                 uvByteOffset    = 0;        // VA_TEXCOORD0 byte offset within the vertex stride
	};

	// One record per shadow-casting light (CPU-side). cubeVP holds the 6 cube-face view-projs used to GENERATE
	// the atlas (ForEachLightFace + P2 pose-freeze); it is NOT uploaded to the shader — cube-direct sampling
	// derives the face from the light->receiver direction, so the GPU needs only the 32-byte prefix (GPULightRecord).
	struct LightRecord
	{
		DirectX::XMFLOAT4   positionWS = {};    // xyz = light pos; w = faceRes (per-light cube-face resolution, texels)
		float               farPlane   = 0.0f;  // offset 16
		float               nearPlane  = 0.0f;  // 20
		float               atlasX     = 0.0f;  // 24 — pixel origin X of this light's 3x2 block in the atlas
		float               atlasY     = 0.0f;  // 28 — pixel origin Y (end of the 32-byte uploaded prefix)
		DirectX::XMFLOAT4X4 cubeVP[6]  = {};     // 32+ — world -> face-clip per cube face; CPU-ONLY (atlas generation), never uploaded
		float faceRes() const { return positionWS.w; }  // per-light cube-face resolution (texels); non-virtual -> no layout change
	};

	// The GPU-uploaded per-light record (t111): exactly the 32-byte prefix the LLF shader samples. MUST match
	// VSM.hlsli::LightRecord. cubeVP is deliberately absent — cube-direct sampling doesn't read it, so dropping it
	// from the upload shrinks the t111 buffer ~13x (416 -> 32 B/light) and the per-frame Map/copy along with it.
	struct GPULightRecord
	{
		DirectX::XMFLOAT4 positionWS;
		float             farPlane, nearPlane, atlasX, atlasY;
	};

	// cached engine D3D (not owned)
	ID3D11Device*        device = nullptr;
	ID3D11DeviceContext* context = nullptr;
	IDXGISwapChain*      swapChain = nullptr;  // not owned; only queried (GetDesc) to keep screenH current on a resolution change

	// Runtime atlas dimensions, derived from the game's iShadowMapResolution ini at startup (compile-time
	// vsm::kFaceRes etc. are the fallback). Lets the per-cube-face resolution track the user's shadow-quality
	// setting; uploaded to the shader via the b13 cbuffer (gFaceRes/gAtlasH) so sampling matches the texture
	// we actually allocate. The atlas is packed with per-light variable-resolution blocks (PackAtlas) and
	// grows on demand — no fixed kMaxLights x 6 layout; each light's faceRes rung tracks its on-screen size
	// up to iShadowMapResolution (uncapped lights, variable resolution — both shipped).
	int rtFaceRes = vsm::kFaceRes, rtBlockW = vsm::kBlockW, rtBlockH = vsm::kBlockH, rtAtlasW = vsm::kAtlasW, rtAtlasH = vsm::kAtlasH;
	// Variable-resolution ladder bounds, from iShadowMapResolution (ComputeAtlasDims). rtMaxFaceRes = top rung
	// (exact ini value); rtFloorFaceRes = smallest rung (min(kFloorFaceRes, max)). Per-light faceRes rides in
	// LightRecord.positionWS.w; rtFaceRes/rtBlockW/H stay = the MAX (reference for the packer + diagnostics).
	int rtMaxFaceRes = vsm::kFaceRes, rtFloorFaceRes = vsm::kFloorFaceRes;
	// SkyrimPrefs shadow keys we honor (read once in ComputeAtlasDims; we ARE the local shadow system now, so
	// preserve the user's intent). Fed to the shader via b13 (bias/blur) or the level metric.
	float rtBiasScale      = 1.0f;    // fShadowBiasScale -> multiplier on the calculated per-pixel bias (may be <0)
	int   rtPCFRadius      = 2;       // iBlurDeferredShadowMask -> soft-shadow PCF kernel half-width in texels (0 = hard)
	float rtShadowDistance = 8000.0f; // fShadowDistance -> camera-distance shadow reach (feeds the parked fade/cull, #10)
	void ComputeAtlasDims();        // read iShadowMapResolution -> ladder bounds (max/floor) + block dims (once, before setup)
	bool CreateAtlasResources();    // (re)create depth+id atlas textures/views at rtAtlasW x rtAtlasH
	void EnsureTransmittanceAtlas();  // O9: lazily create the A5 RGBA transmittance atlas (only when translucentShadows on)
	// Per-frame variable-resolution: pick each light's cube-face rung on the pow2 ladder from the on-screen
	// size metric (radius/dist), with hysteresis; then pack the per-light 3x2 blocks into the atlas + size it.
	int  AssignFaceRes(float a_radius, const RE::NiPoint3& a_lightPos, const RE::NiPoint3& a_camPos, const void* a_lightId);
	void PackAtlas();               // assign each light's atlasX/atlasY + grow the atlas to fit (shelf packer)
	bool CreateStaticCacheResources();  // (re)create staticDepthTex + staticDsv at rtAtlasW x rtAtlasH (P2)
	bool EnsureStaticCache();           // lazily create the static cache when cacheStaticShadows is on; false if unavailable
	void RenderCasterPass(int a_filter, bool a_drawSkinned);  // draw casters into the BOUND dsv/viewport; filter 0=all 1=static 2=dynamic
	void BakeOneLightStatic(size_t a_lightIdx);   // P4: clear + re-render ONE light's static block into the static cache (staticDsv bound by caller)
	void RenderTranslucentPass();                 // A5: render glass/alpha casters into the transmittance atlas (multiplicative, depth-tested vs opaque)
	// Iterate one light's 6 cube faces: bind each face's atlas viewport and hand the body the face's view-proj +
	// world-space frustum planes. The block/face/viewport/plane math lives ONLY here, shared by every atlas pass
	// (opaque depth, P4 static bake, A5 translucent). planes points to a temporary 6-element array valid for the call.
	void ForEachLightFace(const LightRecord& a_light,
	    const std::function<void(const DirectX::XMMATRIX& a_faceVP, const DirectX::XMFLOAT4* a_planes)>& a_body);
	// Shared tail of every caster draw: push the per-draw cbuffer (WVP, id, transmittance), bind the caster's
	// vertex/index buffers, issue the indexed draw. Caller has set the pass pipeline + per-caster raster and
	// validated the buffers; a_transmittance is used only by the A5 pass (ignored by the depth/id shaders).
	void EmitCasterDraw(size_t a_i, ID3D11Buffer* a_vb, ID3D11Buffer* a_ib,
	    const DirectX::XMMATRIX& a_wvp, const DirectX::XMFLOAT4& a_transmittance);
	bool CreateLightBufferResources(int cap);  // (re)create the per-light structured buffer + SRV at cap elements
	void EnsureLightBuffer(int nLights);        // grow the per-light buffer to fit ALL active lights (no cap)
	int  lightBufCapacity = vsm::kMaxLights;    // current element capacity of lightBuffer (grows on demand)

	// our resources
	ComPtr<ID3D11Texture2D>          depthTex;
	ComPtr<ID3D11DepthStencilView>   dsv;
	ComPtr<ID3D11ShaderResourceView> srv;
	// A5 colored translucent shadows (config.translucentShadows): an RGBA transmittance atlas, SAME layout/size as the
	// depth atlas. Glass/alpha casters render into it with a MULTIPLICATIVE, order-independent blend (dst=dst*src);
	// LLF samples it at t112 and multiplies the local light. ALWAYS created + cleared white each frame, so when A5 is
	// OFF the shader path is a no-op multiply by white. dsvReadOnly lets the glass depth-test vs the opaque atlas.
	ComPtr<ID3D11Texture2D>          transTex;
	ComPtr<ID3D11RenderTargetView>   transRTV;
	ComPtr<ID3D11ShaderResourceView> transSRV;
	ComPtr<ID3D11DepthStencilView>   dsvReadOnly;         // read-only DSV of depthTex — glass tests vs the opaque atlas, no depth write
	ComPtr<ID3D11BlendState>         multiplyBlend;       // dst = dst * src (order-independent transmittance accumulation)
	ComPtr<ID3D11DepthStencilState>  depthReadOnlyState;  // GREATER, depth-write OFF (glass tests vs opaque; layered glass all contributes)
	ComPtr<ID3D11PixelShader>        transPS;             // outputs the per-caster transmittance color from PerDrawCB
	ComPtr<ID3D11DepthStencilState>  depthClearState;     // P4: ALWAYS + depth-write, for the per-block far-Z clear (fullscreen triangle)
	// P2 static/dynamic caching (config.cacheStaticShadows): a STATIC-only depth cache (ours, NOT sampled by
	// LLF). Each frame: bake static casters into it ONLY when invalidated, then live atlas = CopyResource(cache)
	// + DYNAMIC casters rendered over it -> LLF samples the SAME single t110 atlas (shader unchanged, no CS fork).
	ComPtr<ID3D11Texture2D>          staticDepthTex;   // static-only depth cache (D32), same size as depthTex
	ComPtr<ID3D11DepthStencilView>   staticDsv;
	bool staticCacheValid      = false;   // false -> re-bake static casters into staticDepthTex next frame
	int  lastBakedRegistrySize = -1;      // registry size at last bake (a size change ~ cell change -> invalidate)
	// P2 pose-freeze: while the cache HOLDS, the shader must sample it with the EXACT pose it was baked with,
	// or a sub-threshold light jitter (fire flicker) smears the cached static shadow. Snapshot each light's
	// baked cube VP + position; on hold frames restore them into lightRecords before upload/render.
	struct BakedPose { DirectX::XMFLOAT4X4 cubeVP[6]; DirectX::XMFLOAT3 pos; };
	std::unordered_map<const void*, BakedPose> bakedPose;   // BSLight* -> pose the static cache was baked with
	// P4 incremental cache (config.incrementalCache, requires cacheStaticShadows): per-light dirty set. A light is
	// dirty on first appearance, slot/tier change (PackAtlas), movement > kLightMoveEps, or a static-caster-set change.
	// Only dirty lights re-bake — new ones immediately, moved ones budgeted at kMaxRebakesPerFrame — while every clean
	// light keeps its cached block (bakedPose presence = "has a valid baked static block"). Replaces P2's whole-cache rebuild.
	std::unordered_set<const void*> dirtyLights;            // BSLight* needing a static re-bake
	int lastStaticCasterCount = -1;                         // static-caster count at last incremental sync (a change ~ cell change -> invalidate all)
	std::vector<std::size_t> bakeThisFrame;                 // P4: light indices selected to re-bake this frame (chosen in UpdateStaticCacheState, consumed by RenderDepth)
	void UpdateStaticCacheState();        // after PackAtlas, before upload: finalize invalidation + freeze poses if holding
	void PoseFreezeLight(size_t a_i, const BakedPose& a_bp);  // restore one held light's cube VPs + pos from its baked snapshot (keeps faceRes)
	ComPtr<ID3D11VertexShader>       depthVS;
	ComPtr<ID3D11InputLayout>        ilFull;    // POSITION R32G32B32_FLOAT (always; see RebuildRegistry)
	// A6 alpha-tested cutout depth pass (always on). VS passes UV, PS clips on diffuse alpha.
	ComPtr<ID3D11VertexShader>       alphaVS;
	ComPtr<ID3D11PixelShader>        alphaPS;
	ComPtr<ID3D11SamplerState>       alphaSampler;   // wrap+linear for the diffuse alpha sample
	ComPtr<ID3DBlob>                 alphaVSBytecode; // kept to CreateInputLayout per UV-offset on demand
	std::unordered_map<std::uint32_t, ComPtr<ID3D11InputLayout>> alphaLayouts;  // POSITION@0 + TEXCOORD@uvOffset (R16G16_FLOAT), cached
	ID3D11InputLayout* GetAlphaLayout(std::uint32_t a_uvOffset);  // create/reuse the alpha input layout for a UV byte offset
	ComPtr<ID3D11DepthStencilState>  depthState;
	ComPtr<ID3D11RasterizerState>    rasterState;
	ComPtr<ID3D11RasterizerState>    rasterStateNoCull;  // CULL_NONE for two-sided casters (both faces cast)
	ComPtr<ID3D11Buffer>             perDrawCB;

	// --- P1a batched draw submit (always on when the hardware exposes an ID3D11DeviceContext1) -----------
	// Every SOLID opaque caster draw used to Map(perDrawCB, WRITE_DISCARD) its own 96-byte cbuffer (1556x/frame
	// in the 7-light test scene). Instead we DEFER those draws into drawBatch, fill ONE large cbuffer in a single
	// Map, then submit each draw with a D3D11.1 offset-bound 16-constant window (VSSetConstantBuffers1). Depth-only
	// rendering is order-independent, so deferring is output-identical; skinned / glass / alpha-tested casters keep
	// the immediate path. Falls back to the per-draw Map path when context1 is null (no D3D11.1).
	static constexpr UINT kPerDrawSlotConstants = 16;                 // 16 float4 constants = 256 B; the VSSetConstantBuffers1 alignment unit
	static constexpr UINT kPerDrawSlotBytes     = kPerDrawSlotConstants * 16;  // 256 B per slot (PerDrawCB is 96 B; rest is pad)
	struct BatchedDraw
	{
		ID3D11Buffer*      vb = nullptr;   // caster vertex buffer (owned by the game; valid this frame)
		ID3D11Buffer*      ib = nullptr;   // caster index buffer
		UINT               indexCount = 0;
		UINT               stride = 0;
		bool               twoSided = false;
		D3D11_VIEWPORT     vp{};           // the atlas face viewport this draw targets (restored at submit)
		DirectX::XMFLOAT4X4 wvp{};         // world * cube-face view-proj
		std::uint32_t      casterId = 0;   // registry index + 1 (id atlas)
		float              alphaThreshold = 0.0f;
	};
	ComPtr<ID3D11DeviceContext1> context1;              // QI'd from context; null => immediate per-draw path
	ComPtr<ID3D11Buffer>         perDrawCBArray;         // large dynamic cbuffer of kPerDrawSlotBytes slots (grown on demand)
	UINT                         perDrawCBArrayCount = 0;// slot capacity of perDrawCBArray
	std::vector<BatchedDraw>     drawBatch;              // deferred solid opaque draws for the current RenderCasterPass
	bool                         drawBatchActive = false;// true while RenderCasterPass defers solids into drawBatch
	D3D11_VIEWPORT               curFaceViewport{};      // set by ForEachLightFace; captured into each batched draw
	void SubmitDrawBatch();      // flush drawBatch: one cbuffer fill + offset-bound submit of every deferred draw

	ComPtr<ID3D11VertexShader>       fullscreenVS;  // fullscreen-triangle VS for the P4 per-block far-Z clear (BakeOneLightStatic)
	ComPtr<ID3D11SamplerState>       pointSampler;  // shadow-atlas sampler; CS binds it at s7 (VSM_GetShadowResources contract)

	// --- Path B: skinned characters. Skinned casters are CPU-skinned once per frame into a shared
	// world-absolute posed vertex buffer (bind pose x engine bone matrices), then rasterized with the
	// same depth VS as statics (world=identity). One range per skinned caster for frustum-culled draws.
	struct SkinnedRange
	{
		std::uint32_t ibStart       = 0;
		std::uint32_t indexCount    = 0;
		RE::NiPoint3  center;
		float         radius        = 0.0f;
		int           registryIndex = -1;
	};
	std::vector<SkinnedRange>       skinnedRanges;

	// Per-face draw bodies extracted from RenderDepth's cube-face loop (declared here, after LightRecord /
	// SkinnedRange are defined, because /permissive- requires the nested types to be visible). Both access
	// members directly; only the per-iteration values are parameters.
	bool CulledBySphere(size_t i, const DirectX::XMFLOAT4 planes[6]) const;  // frustum-cull caster i by its casterWorldSphere; true = outside -> skip
	void DrawStaticCaster(size_t i, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6]);
	void DrawSkinnedRange(const SkinnedRange& r, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6]);
	void DrawTranslucentCaster(size_t i, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6]);  // A5: one glass caster -> transmittance atlas

	std::vector<DirectX::XMFLOAT3>  skinPosed;    // CPU scratch: world-absolute posed positions
	std::vector<std::uint32_t>      skinIndices;  // CPU scratch: 32-bit indices into skinPosed
	ComPtr<ID3D11Buffer>            skinnedVB, skinnedIB;
	std::uint32_t                   skinnedVBCap = 0, skinnedIBCap = 0;

	// Detached-shadow guard (see SkinAllCasters): a posed caster whose vertex SPREAD from its own centroid
	// exceeds worldBound.radius + vsm::kDetachMargin is dropped — a genuinely scattered pose (palette resolved
	// to garbage / wrong space), which would paint a ghost. Self-referential: a coherent pose is kept wherever
	// it lands, so a displaced/ragdolled NPC whose mesh bound went stale is NO LONGER false-dropped.

	std::vector<CasterEntry> registry;             // persistent; rebuilt only when geometry streams in/out
	std::unordered_set<RE::BSGeometry*> registrySeen;  // dedup across all capture sources this rebuild
	bool                     registryDirty      = true;
	uint32_t                 framesSinceRebuild = 0;
	ComPtr<ID3D11Buffer>     aabbStaging;              // reused staging buffer for per-caster vertex-AABB readback
	std::uint32_t            aabbStagingSize    = 0;
	// LOCAL AABB cache (stable — never changes for a mesh). The frustum cull needs each caster's CURRENT
	// world box; for animated/moving geometry (a MoveableStatic flame, a billboard) that differs from any
	// single frozen world box, so we cache the invariant LOCAL box and transform it by the CURRENT world
	// matrix every frame (GetFreshWorldAABB).
	std::unordered_map<RE::BSGeometry*, std::pair<DirectX::XMFLOAT3, DirectX::XMFLOAT3>> localAABBCache;
	bool GetFreshWorldAABB(RE::BSGeometry* a_geom, std::uint32_t a_stride,
	    DirectX::XMFLOAT3& a_worldMin, DirectX::XMFLOAT3& a_worldMax);

	// Per-caster bounding sphere for the FRUSTUM cull, parallel to `registry` (xyz = center, w = radius;
	// w < 0 = no sphere -> don't cull). Derived from GetFreshWorldAABB = the SAME geom->world transform
	// RenderDepth draws with, so the cull sphere is in the DRAW's absolute space and can never
	// drift from the drawn geometry the way the engine's worldBound.center can (it went stale for skinned).
	std::vector<DirectX::XMFLOAT4>              casterWorldSphere;

	// Fill casterWorldSphere per-frame from each caster's CURRENT world AABB (GetFreshWorldAABB) — one space
	// with the draw (absolute world via geom->world), so no worldBound.center anywhere in RenderDepth's cull.
	void BuildCasterBounds();

	// --- O10 (always on): world-space uniform caster grid -------------------------------------------------
	// Each light's 6-face cull would otherwise plane-test EVERY caster (O(lights x casters)); the harness
	// measured that at 3.48 s/frame at scale. Binning caster spheres into a coarse world grid lets a light
	// query only the cells its bounding cube overlaps -> O(#lights x local density). Built each frame from
	// casterWorldSphere in BuildCasterBounds; queried per light in RenderCasterPass. The candidate set is a
	// conservative SUPERSET, so the per-face SphereInFrustum refine then yields the IDENTICAL drawn shadow set
	// as the exhaustive scan (verified offline in testing/) — strictly better, so there is no config gate.
	struct CasterGrid
	{
		float cell      = 512.0f;   // world units per cell (~median light radius; a coarse bucket, not tight)
		float maxRadius = 0.0f;     // largest binned caster radius -> sizes the conservative query bound
		std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> cells;  // cell key -> registry indices
		void clear() { maxRadius = 0.0f; cells.clear(); }
		// Pack a signed cell coord into 21 bits/axis (biased); +/-2^20 cells * 512u covers any Skyrim coordinate.
		static std::uint64_t Key(int x, int y, int z)
		{
			auto u = [](int v) { return static_cast<std::uint64_t>(static_cast<std::uint32_t>(v + (1 << 20))); };
			return (u(x) << 42) | (u(y) << 21) | u(z);
		}
	};
	CasterGrid                    casterGrid;             // rebuilt each frame (O10 always on)
	std::vector<std::uint32_t>    casterQueryScratch;     // reused per-light candidate list (no per-light alloc)
	std::vector<std::uint32_t>    casterQuerySeen;        // stamp-dedup, parallel to registry
	std::uint32_t                 casterQueryStamp = 0;   // bumped per query; seen[i]==stamp => already a candidate
	void BuildCasterGrid();                                      // bin casterWorldSphere into casterGrid
	void QueryCasterGrid(const LightRecord& a_light, std::vector<std::uint32_t>& a_out);  // nearby registry indices

	std::vector<LightRecord>         lightRecords;   // all shadowed lights this frame (NO cap; buffer grows to fit)
	ComPtr<ID3D11Buffer>             lightBuffer;    // GPU structured copy for the LLF shader
	ComPtr<ID3D11ShaderResourceView> lightBufferSRV;

	// Atlas geometry constants (kFaceRes/kMaxLights/kBlockW/kAtlasW/...) live in VSMConstants.h as the
	// C++-side capacity + fallback sizing. VSM.hlsli does NOT share them: the shader samples using the
	// RUNTIME atlas dims (gFaceRes/gAtlasW/gAtlasH from the b13 cbuffer). The only genuine shared contract
	// with the shader is the LightRecord LAYOUT (above).

	// settings (driven by DrawMenu; persisted defaults loaded from VirtualShadowMaps.toml — see VSMConfig)
	bool  enabled       = false;
	bool  settingsDirty = false;   // a persisted tuning control changed this frame -> save on release
	// NOTE: casters are NEVER offset by cameraPos. geom->world.translate under the ShadowSceneNode is ALREADY
	// game-absolute (confirmed vs CS: LightLimitFix passes niLight->world.translate as the absolute light pos
	// and SUBTRACTS the eye to get camera-relative positionWS), and skinned verts are posed to absolute world.
	// The light cubes + shader sampling are absolute too, so no-offset geometry aligns exactly. Adding
	// cameraPos would assume camera-relative geometry and shift casters (~1300u) out of place.

	// Reliable render eye = ShadowSceneNode::cameraPos (set each frame in CollectLights). Offset the
	// atlas by THIS, NOT RendererShadowState::posAdjust.getEye() — the latter intermittently returns
	// garbage (0,0,1) that mis-places the whole atlas by the camera position (dump #3).
	DirectX::XMFLOAT3 sceneCameraPos{};

	// Render-target height (from the swapchain), fed to the on-screen size metric for per-light face-res
	// selection. DynamicResolutionParams2 is a scale (1,1) here, so screen-derived pixels come from C++.
	float screenH = 1080.0f;

	// Per-light near/far are CALCULATED from the light's own radius (CollectLights): far = radius,
	// near = far * vsm::kNearPlaneFraction (floored at kNearPlaneEpsilon). The shadow test uses a
	// CALCULATED receiver-plane bias in the shader. None of these are user knobs. (Former FarScale /
	// NearFrac / BiasWorld / fixture-ratio sliders removed — see VSMConstants.h.)
	ComPtr<ID3D11Buffer> paramsCB;  // 32-byte cbuffer @ b13, two float4 rows (see UpdateParamsCB / VSM.hlsli::VSMParams)

	// Real-shader pixel probe (u8): written by VSM.hlsli for the centre pixel when armed.

	int               activeLightCount = 0;    // valid shadow-casting lights this frame; sizes the per-light GPU buffer (EnsureLightBuffer)
	std::vector<float>             lightMove;       // per shadow-light: world-units moved since last frame (BY IDENTITY); drives P2 static-cache invalidation
	// Identity-keyed previous light positions (BSLight* as opaque key) so each light reports its TRUE
	// per-frame delta into lightMove — nearest-neighbour matching hid that (it snapped a moving light
	// onto a static neighbour). lightMove feeds the P2 static-cache invalidation (see kLightMoveEps).
	std::vector<const void*>       lightPtrs;      // parallel to lightRecords: the BSLight* identity
	// LLF light-index alignment (Option c): BSLight* -> its slot in ShadowLights (t111). Rebuilt in
	// UploadLightBuffer (atomic with the GPU upload) from lightPtrs. Read via GetLightShadowIndex by LLF.
	// The mutex guards the write (UploadLightBuffer, our Present tick) against the read (GetLightShadowIndex,
	// called from CS's LLF Prepass): today they run on the same thread a frame apart, but the export is a public
	// cross-DLL entry point, so a lock makes the read safe even if a future CS path ever calls it off-thread.
	std::unordered_map<const void*, std::uint32_t> lightIndexByPtr;
	mutable std::mutex                             lightIndexMutex;
	std::unordered_map<const void*, DirectX::XMFLOAT3> prevLightById;  // BSLight* -> last-frame world pos
	std::unordered_map<const void*, DirectX::XMFLOAT3> smoothLightById;  // BSLight* -> SMOOTHED cube center (aesthetics: de-step swinging lights; see vsm::kLightSmoothAlpha)
	float lightSmoothAlpha = vsm::kLightSmoothAlpha;  // live moving-light smoothing strength (DEV slider tunes it; deploy = the constant)
	std::unordered_map<const void*, int>               prevLightFaceRes;  // BSLight* -> last-frame chosen face rung (hysteresis)
	// STABLE atlas allocation (P1, performance): a light keeps the SAME atlas slot across frames so P2 can
	// cache its static depth. Slots are allocated on a light's first appearance, reused while its resolution
	// tier is unchanged, and LRU-freed when unseen. Freed slots recycle by exact size (freeSlots); a new size
	// shelf-bumps the cursor. Replaces the old every-frame repack (which moved slots and defeated caching).
	struct AtlasSlot { int x = 0, y = 0, w = 0, h = 0, faceRes = 0; std::uint64_t lastUsed = 0; };
	std::unordered_map<const void*, AtlasSlot>                          lightSlots;   // BSLight* -> persistent atlas slot
	std::unordered_map<std::uint64_t, std::vector<std::pair<int, int>>> freeSlots;    // (w<<32|h) -> freed (x,y) origins, reused same-size
	int           atlasBumpX = 0, atlasBumpY = 0, atlasBumpRowH = 0;    // shelf-bump cursor for NEW slots (pixels)
	std::uint64_t atlasFrame = 0;                                        // per-PackAtlas tick, drives LRU eviction
	// Collection DEBOUNCE (anti-flicker): the engine rotates which lights are IsShadowLight()/activeShadowLights
	// every RENDER frame (even when paused), churning our collected set -> shadows blink. Keep a light shadowed
	// for a GRACE window after it last appeared, re-adding recently-seen lights that dropped out this frame.
	struct CachedLight { LightRecord rec; std::uint64_t lastSeen; };
	std::unordered_map<const void*, CachedLight> recentLights;   // BSLight* -> last collected record + last-seen frame
	std::uint64_t collectFrame = 0;                              // per-CollectLights tick, drives the grace window


	bool resourcesReady = false;
	bool haveLights  = false;

	// N3: consumer gate. resourceFetched is set by NotifyResourcesFetched (CS Prepass) and read+reset each
	// RenderFrame; framesSinceResourceFetch counts frames with no fetch, so we skip the render once nothing has
	// sampled our atlas for kConsumerGraceFrames (LLF off/absent). The grace keeps us rendering through startup
	// and any transient hitch, and any single fetch resumes full rendering on the next frame.
	std::atomic<bool> resourceFetched{ false };
	std::uint32_t     framesSinceResourceFetch = 0;
	// N4: last player parent cell (opaque pointer identity). A change (coc / door / fast-travel) forces an
	// immediate registry rebuild so stale casters from the previous cell stop casting at once.
	const void*       lastPlayerCell = nullptr;

};
