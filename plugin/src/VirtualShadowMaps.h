#pragma once

// ============================================================================
// VirtualShadowMaps — standalone SKSE plugin, owned entirely by us.
//
// Each frame (driven by an IDXGISwapChain::Present hook) it renders our OWN cube-shadow
// atlas for every active local light — statics from a persistent scene-graph registry,
// plus CPU-skinned characters (Path B) — and exposes the atlas + per-light buffer to
// Community Shaders' LightLimitFix, whose Lighting.hlsl samples them (VSM.hlsli) so local
// lights the engine's shadow budget dropped still cast shadows.
//
// The settings UI (DrawMenu) draws INTO the Community Shaders menu via CS's add-on hook
// (CS_RegisterExternalMenu / CS_GetImGui) — our logic never lives in CS's DLL; CS only
// exposes a tiny generic hook, and our DLL shares CS's ImGui context.
// ============================================================================

#include "VSMBuildConfig.h"  // VSM_DIAGNOSTICS dev/deploy toggle + VSM_LOG macro
#include "VSMConstants.h"  // vsm:: atlas geometry + capacities (single source of truth)
#if VSM_DIAGNOSTICS
	#include "VirtualShadowMaps_Diagnostics.h"  // VsmDiagnostics — the extracted diagnostics apparatus (diag member below)
#endif

#include <DirectXMath.h>
#include <d3d11.h>
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
	void SetRenderSize(std::uint32_t a_w, std::uint32_t a_h) { screenW = static_cast<float>(a_w); screenH = static_cast<float>(a_h); }
	void RenderFrame();  // per-frame, from the Present hook
	void DrawMenu();     // ImGui, called from the CS menu via our registered callback

	// Exposed to Plugin.cpp's C exports so LLF's shader can sample our shadows.
	ID3D11ShaderResourceView* GetAtlasSRV() const { return srv.Get(); }
	ID3D11ShaderResourceView* GetLightBufferSRV() const { return lightBufferSRV.Get(); }
	ID3D11SamplerState*       GetPointSampler() const { return pointSampler.Get(); }
	ID3D11Buffer*             GetDebugCB() const { return debugCB.Get(); }
	int  GetShadowLightCount() const { return static_cast<int>(lightRecords.size()); }
	bool IsEnabled() const { return enabled; }

	// Real-shader pixel probe: our OMSetRenderTargets hook (Plugin.cpp) binds this UAV at u8 during
	// the lighting draws ONLY while armed, so the real Lighting.hlsl can write its per-pixel state.
	// (u8 because Lighting.hlsl's pixel shader already outputs render targets at u0..u7.)
#if VSM_DIAGNOSTICS
	ID3D11UnorderedAccessView* GetPixelProbeUAV() const { return pixelProbeUAV.Get(); }
	bool IsProbeArmed() const { return probeArmed; }
#endif

	// Called from VSM_GetShadowResources each time CS pulls our resources to bind — lets the dump
	// verify the CS<->plugin handshake is live (fetch happening on the current frame).
	void NoteResourceFetch() { ++resourceFetchCount; lastResourceFetchFrame = frameIndex; }

private:
	VirtualShadowMaps() = default;

	// The diagnostics apparatus (dumps / probes / census / preview) lives in its own class,
	// off this god-object. It reaches our private render-state through m_core, so it's a friend;
	// we own it by value and hand it a reference to ourselves. See VirtualShadowMaps_Diagnostics.*.
#if VSM_DIAGNOSTICS
	friend class VsmDiagnostics;
	VsmDiagnostics diag{ *this };
#endif

	void SetupResources();
	void CollectLights(RE::ShadowSceneNode* a_ssn);    // gather all active local lights -> lightRecords
	void UploadLightBuffer();                          // mirror lightRecords into the GPU structured buffer
	void UpdateDebugCB();                              // push the live tuning sliders to the shader cbuffer
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
	static RE::FormID NodeOwningRef(RE::NiAVObject* a_node);  // walk parent chain -> the ref this node hangs under
	void LoadConfig();                 // pull persisted tunables from VirtualShadowMaps.toml at startup
	void PersistSettingsIfChanged();   // write tuning changes back to the config file (once, on control release)
	void TrackCasterMotion();          // per-frame: flag casters whose worldBound moves WITH the camera (camera-relative bug)

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
		RE::FormID                    ownerRef     = 0;      // the TESObjectREFR this mesh belongs to. A shadow
		                                                     // light's OWN housing (fire-pit rim/flame plane) shares
		                                                     // the light's ref -> skip it (see RenderDepth ref cull).
		bool                          isStatic     = false;  // R5 classification: STATIC = never moves (base Static/
		                                                     // StaticCollection, not skinned, no anim controller) ->
		                                                     // safe to CACHE once (P2). Else DYNAMIC (err this way).
	};

	// One record per shadow-casting light, uploaded to a GPU buffer so the LLF shader
	// can locate + sample the light's cube in the atlas. Must match VSM.hlsli.
	// The 6 face view-projs are included so the shader samples with the EXACT matrices we
	// rendered with (avoids re-deriving the cube projection and getting it subtly wrong).
	struct LightRecord
	{
		DirectX::XMFLOAT4   positionWS = {};    // xyz = light pos; w = faceRes (per-light cube-face resolution, texels)
		float               farPlane   = 0.0f;  // offset 16
		float               nearPlane  = 0.0f;  // 20
		float               atlasX     = 0.0f;  // 24 — pixel origin X of this light's 3x2 block in the atlas
		float               atlasY     = 0.0f;  // 28 — pixel origin Y (cubeVP stays at 32; size/layout unchanged)
		DirectX::XMFLOAT4X4 cubeVP[6]  = {};     // 32 — world -> face-clip, per cube face (MUST match VSM.hlsli)
		float faceRes() const { return positionWS.w; }  // per-light cube-face resolution (texels); non-virtual -> no layout change
	};

	// cached engine D3D (not owned)
	ID3D11Device*        device = nullptr;
	ID3D11DeviceContext* context = nullptr;

	// Runtime atlas dimensions, derived from the game's iShadowMapResolution ini at startup (compile-time
	// vsm::kFaceRes etc. are the fallback). Lets the per-cube-face resolution track the user's shadow-quality
	// setting; uploaded to the shader via the b13 cbuffer (gFaceRes/gAtlasH) so sampling matches the texture
	// we actually allocate. NOTE: the dense atlas holds kMaxLights x 6 faces, so faceRes is bounded for now
	// (VRAM); the uncapped, full-ini-resolution path arrives with active-light / virtual sizing.
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
	// Per-frame variable-resolution: pick each light's cube-face rung on the pow2 ladder from the on-screen
	// size metric (radius/dist), with hysteresis; then pack the per-light 3x2 blocks into the atlas + size it.
	int  AssignFaceRes(float a_radius, const RE::NiPoint3& a_lightPos, const RE::NiPoint3& a_camPos, const void* a_lightId);
	void PackAtlas();               // assign each light's atlasX/atlasY + grow the atlas to fit (shelf packer)
	bool CreateStaticCacheResources();  // (re)create staticDepthTex + staticDsv at rtAtlasW x rtAtlasH (P2)
	bool EnsureStaticCache();           // lazily create the static cache when cacheStaticShadows is on; false if unavailable
	void RenderCasterPass(int a_filter, bool a_drawSkinned);  // draw casters into the BOUND dsv/viewport; filter 0=all 1=static 2=dynamic
	bool CreateLightBufferResources(int cap);  // (re)create the per-light structured buffer + SRV at cap elements
	void EnsureLightBuffer(int nLights);        // grow the per-light buffer to fit ALL active lights (no cap)
	int  lightBufCapacity = vsm::kMaxLights;    // current element capacity of lightBuffer (grows on demand)

	// our resources
	ComPtr<ID3D11Texture2D>          depthTex;
	ComPtr<ID3D11DepthStencilView>   dsv;
	ComPtr<ID3D11ShaderResourceView> srv;
	// P2 static/dynamic caching (config.cacheStaticShadows): a STATIC-only depth cache (ours, NOT sampled by
	// LLF). Each frame: bake static casters into it ONLY when invalidated, then live atlas = CopyResource(cache)
	// + DYNAMIC casters rendered over it -> LLF samples the SAME single t110 atlas (shader unchanged, no CS fork).
	ComPtr<ID3D11Texture2D>          staticDepthTex;   // static-only depth cache (D32), same size as depthTex
	ComPtr<ID3D11DepthStencilView>   staticDsv;
	bool staticCacheValid      = false;   // false -> re-bake static casters into staticDepthTex next frame
	int  lastBakedRegistrySize = -1;      // registry size at last bake (a size change ~ cell change -> invalidate)
	ComPtr<ID3D11VertexShader>       depthVS;
	ComPtr<ID3D11InputLayout>        ilFull;    // POSITION R32G32B32_FLOAT (always; see RebuildRegistry)
	// Occluder-id atlas: R32_UINT target rasterized alongside depth. Each draw writes (registryIndex+1);
	// the depth test keeps the nearest, so a texel names the EXACT closest caster (0 = empty). Diagnostic.
	ComPtr<ID3D11Texture2D>          idTex;
	ComPtr<ID3D11RenderTargetView>   idRTV;
	ComPtr<ID3D11ShaderResourceView> idSRV;
	ComPtr<ID3D11PixelShader>        idPS;
	ComPtr<ID3D11DepthStencilState>  depthState;
	ComPtr<ID3D11RasterizerState>    rasterState;
	ComPtr<ID3D11RasterizerState>    rasterStateNoCull;  // CULL_NONE for two-sided casters (both faces cast)
	ComPtr<ID3D11Buffer>             perDrawCB;

	// linear-depth debug view (fullscreen resolve of depthTex -> grayscale)
	ComPtr<ID3D11Texture2D>          previewTex;
	ComPtr<ID3D11RenderTargetView>   previewRTV;
	ComPtr<ID3D11ShaderResourceView> previewSRV;
	ComPtr<ID3D11VertexShader>       fullscreenVS;
	ComPtr<ID3D11PixelShader>        linearizePS;
	ComPtr<ID3D11SamplerState>       pointSampler;
	ComPtr<ID3D11Buffer>             previewCB;  // preview normalization range

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
	void DrawStaticCaster(size_t i, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6], bool blinkOff);
	void DrawSkinnedRange(const SkinnedRange& r, DirectX::FXMMATRIX faceVP, const DirectX::XMFLOAT4 planes[6], bool blinkOff);

	std::vector<DirectX::XMFLOAT3>  skinPosed;    // CPU scratch: world-absolute posed positions
	std::vector<std::uint32_t>      skinIndices;  // CPU scratch: 32-bit indices into skinPosed
	ComPtr<ID3D11Buffer>            skinnedVB, skinnedIB;
	std::uint32_t                   skinnedVBCap = 0, skinnedIBCap = 0;

	// Detached-shadow guard (see SkinAllCasters): a posed caster whose vertex SPREAD from its own centroid
	// exceeds worldBound.radius + vsm::kDetachMargin is dropped — a genuinely scattered pose (palette resolved
	// to garbage / wrong space), which would paint a ghost. Self-referential: a coherent pose is kept wherever
	// it lands, so a displaced/ragdolled NPC whose mesh bound went stale is NO LONGER false-dropped.
	int                    skinnedRendered      = 0;   // skinned casters posed coherently and rendered this frame
	int                    skinnedDetached      = 0;   // skinned casters dropped as off-bound (would be detached)
	float                  skinnedMaxDetachDist = 0.0f;// worst measured centroid-vs-bound offset among the dropped
	std::string            skinnedMaxDetachName;       // name of that worst offender (for the dump)
	int                    skinnedClampedVerts  = 0;   // posed verts snapped back to bound (skinning explosions killed)

	std::vector<CasterEntry> registry;             // persistent; rebuilt only when geometry streams in/out
	std::unordered_set<RE::BSGeometry*> registrySeen;  // dedup across all capture sources this rebuild
	// Capture-rejection tally (reset each rebuild): why a BSTriShape we reached was NOT registered.
	int rejectedNoRenderData = 0, rejectedNoVertexBuffer = 0, rejectedNoIndexBuffer = 0, rejectedZeroTriangles = 0, rejectedDuplicate = 0;
	// Subtrees skipped because the GAME isn't drawing them: hidden / app-culled (kHidden) nodes —
	// e.g. editor markers (XMarker/Marker_Idle/EditorMarker...). Casting these = "shadows from nothing".
	int rejectedHidden = 0;
	// Non-opaque geometry skipped: alpha-blended (blood/glass/glow/liquid) or effect-shader (emitters/
	// FX). Transparent in-game but would cast solid shadows; much is skinned -> "moving invisible" casters.
	int rejectedTransparent = 0;
	// Geometry the engine's own shader flag says is NOT a shadow caster (kCastShadows off): fire/light
	// planes, glow billboards. We mirror the flag so we cast exactly what the game casts.
	int rejectedNoCast = 0;
	// Decal geometry (kDecal/kDynamicDecal): flat projected textures (scorch/blood/glow) — no volume,
	// never a shadow caster.
	int rejectedDecal = 0;
	// Billboard geometry (under a NiBillboardNode): camera-facing effect planes (smoke/vapor/fire/glow).
	// They (1) always turn to face the player, so any hard shadow they cast RIDES THE CAMERA, and (2) are
	// translucent effects that should not throw a solid shadow. This was the moving 'Plane03' vapor-smoke
	// bar — opaque-material + CastShadows=on, so no material flag caught it; being a billboard is what does.
	int rejectedBillboard = 0;
	bool                     registryDirty      = true;
	uint32_t                 framesSinceRebuild = 0;
	int                      visibleCasters     = 0;  // drawn this frame (after frustum cull, summed over faces)
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
	// RenderDepth draws with, so the cull sphere is provably in the DRAW's absolute space and can never
	// drift from the drawn geometry the way the engine's worldBound.center can (it went stale for skinned).
	std::vector<DirectX::XMFLOAT4>              casterWorldSphere;

	// Fill casterWorldSphere per-frame from each caster's CURRENT world AABB (GetFreshWorldAABB) — one space
	// with the draw (absolute world via geom->world), so no worldBound.center anywhere in RenderDepth's cull.
	void BuildCasterBounds();
	std::vector<LightRecord>         lightRecords;   // all shadowed lights this frame (NO cap; buffer grows to fit)
	std::vector<std::string>         lightNames;     // parallel to lightRecords: NiLight name (identify the camera light)
	std::vector<RE::FormID>          lightOwnerRef;  // parallel to lightRecords: the light's TESObjectREFR (GetUserData) — drives the ref-cull
	std::vector<RE::FormID>          lightNodeRef;   // parallel: the ref the light's NiNode actually hangs under (attached-light case)
	// Authored + engine light metadata, parallel to lightRecords. Read from the NiLight's reference ->
	// base TESObjectLIGH so the log carries the level designer's own values (identity, shadow type, the
	// authored Near Clip, radius, color) instead of only what we recompute.
	struct LightMeta
	{
		uint32_t    formID = 0;
		std::string editorID;
		std::string typeStr;                       // decoded shadow-type flags (Omni/Hemi/Spot/None/Neg...)
		float authoredNear = 0.0f;                 // OBJ_LIGH::nearDistance (the CK "Near Clip")
		float radiusX = 0.0f, radiusY = 0.0f, radiusZ = 0.0f;
		float colR = 0.0f, colG = 0.0f, colB = 0.0f;
	};
	std::vector<LightMeta>           lightMeta;      // parallel to lightRecords
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
	// The light cubes + shader sampling are absolute too, so no-offset geometry aligns exactly. A former
	// 'add cameraPos' toggle assumed camera-relative geometry — wrong, it shifted casters ~1300u off — removed.

	// Reliable render eye = ShadowSceneNode::cameraPos (set each frame in CollectLights). Offset the
	// atlas by THIS, NOT RendererShadowState::posAdjust.getEye() — the latter intermittently returns
	// garbage (0,0,1) that mis-places the whole atlas by the camera position (dump #3).
	DirectX::XMFLOAT3 sceneCameraPos{};

	// Movable pixel-probe target as a fraction of the screen (default centre). A magenta on-screen
	// marker shows it; the user drags it onto a camera-tied artifact instead of aiming the camera.
	float probeFracX = 0.5f, probeFracY = 0.5f;

	// Render-target size (from the swapchain), used to aim the pixel probe at the screen centre.
	// DynamicResolutionParams2 turned out to be a scale (1,1) here, not 1/size, so we feed the target
	// pixel from C++ instead of deriving it in the shader.
	float screenW = 1920.0f, screenH = 1080.0f;

	// debug view — RenderFrame does the debug-only GPU work (preview resolve, AABB scan,
	// inspection) ONLY when the Debug section is open AND the menu is actually on screen.
	bool  showDebug    = false;
	bool  menuVisible  = false;  // set by DrawMenu each frame the menu is drawn
	float previewScale = 0.25f;  // atlas-preview overlay size, as a fraction of the screen

	// debug-view controls
	int   lightSelect   = -1;     // -1 = nearest local light to camera; else index into active lights
	// (previewRange moved to VsmDiagnostics — read only by ResolvePreview + the menu preview slider)
	int   isolateCaster = -1;     // -1 = render all; else 0-based registry index (matches flashCaster)
	// Flash selector: -1 = off; else registry index whose shadow BLINKS (we skip drawing it into the
	// atlas every other ~1/3 s so its shadow winks on/off in the world). Lets the user cycle casters
	// until the WRONG shadow blinks, then read its name — identifies the culprit without a crosshair.
	int   flashCaster = -1;

	// Per-light near/far are CALCULATED from the light's own radius (CollectLights): far = radius,
	// near = far * vsm::kNearPlaneFraction (floored at kNearPlaneEpsilon). The shadow test uses a
	// CALCULATED receiver-plane bias in the shader. None of these are user knobs. (Former FarScale /
	// NearFrac / BiasWorld / fixture-ratio sliders removed — see VSMConstants.h.)
	int   dbgMode      = 0;        // 0 = shadow  1 = off(lit)  >=2 RGB diagnostic overlays
	float dbgVizScale  = 2000.0f;  // grayscale scale for the atlas-depth view (debug only)

	// Two live diagnostic DIMENSIONS that re-test every mode without a rebuild. The atlas is
	// rasterized in ABSOLUTE world space (geom->world + absolute-light cubeVP), so absP is the
	// physically-correct sample space; 1/2 are controls to prove it. See VSM.hlsli::SampleP.
	int   dbgSampleSpace = 0;      // 0 = absP (P + CameraPosAdjust)  1 = P (cam-rel)  2 = P + altEye(C++ render eye)
	int   dbgCompareMode = 0;      // 0 = linearized-distance compare  1 = raw ndc.z compare (isolates linearize)
	// The shader light passed to GetLocalShadow (LLF's positionWS) is camera-relative to
	// posAdjust.getEye() (== altEye), NOT to FrameBuffer::CameraPosAdjust. If those two eyes
	// differ, matching the shader light to our absolute buffer with CameraPosAdjust drifts with
	// the camera. Default to altEye (the provably-correct eye); toggle back to compare.
	// Match with the shader's OWN same-frame FrameBuffer::CameraPosAdjust (0), exactly as LLF builds
	// light.positionWS. Using our altEye (1) is 1 frame stale in the shader's b13 cbuffer, so on fast
	// camera motion the stale match key can miss the nearest-light match and the shadow blinks off
	// (flicker). 0 = robust.
	int   dbgMatchEye    = 0;      // 0 = CameraPosAdjust (default, same-frame)  1 = altEye (stale)
	bool  probeArmed     = false;  // menu: arm the real-shader pixel probe (writes centre pixel to u8)
	// gBiasWorld / gMatchThresh in this cbuffer are debug overlay only (modes 15/16); the real path uses a
	// calculated receiver-plane bias and a threshold-free nearest-light match.
	ComPtr<ID3D11Buffer> debugCB;  // 64-byte cbuffer @ b13, four float4 rows (see UpdateDebugCB / VSM.hlsli::VSMDebug)

	// (Former GPU compute-shader shadow-math probe removed — superseded by the real-shader pixel probe below + tools/shadow_truth.py.)

	// Real-shader pixel probe (u8): written by VSM.hlsli for the centre pixel when armed.
#if VSM_DIAGNOSTICS
	ComPtr<ID3D11Buffer>              pixelProbeBuf;
	ComPtr<ID3D11UnorderedAccessView> pixelProbeUAV;
	ComPtr<ID3D11Buffer>              pixelProbeStaging;
#endif

	// diagnostics (shown in the menu to sanity-check world-space coordinates)
	int               activeLightCount = 0;    // valid local lights this frame (shadow-casters, after filtering)
	int               lightsAmbient    = 0;    // active lights skipped this frame: ambient/fill (never shadow)
	int               lightsNonShadow  = 0;    // active lights skipped this frame: not flagged shadow-casters
	int               lightsCulled     = 0;    // P3: lights dropped this frame (behind camera / beyond fShadowDistance)
	int               lightsViewerAttached = 0;// active lights skipped this frame: attached to player/camera (ride the viewer)
	int               lightsHidden         = 0;// active lights skipped this frame: hidden / app-culled (kHidden)
	std::vector<uint8_t>           lightDynamic;   // per shadow-light: BSLight::dynamic (a moving/animated light?)
	std::vector<float>             lightMove;       // per shadow-light: units moved since last frame (BY IDENTITY; ~cameraMoved => tracks camera)
	// Identity-keyed previous light positions (BSLight* as opaque key) so a light that MOVES WITH the
	// camera reports its true per-frame delta — nearest-neighbour matching hid that (it snapped to a
	// static neighbour). Compared against cameraMovedThisFrame to spot a shadow that rides the camera.
	std::vector<const void*>       lightPtrs;      // parallel to lightRecords: the BSLight* identity
	std::unordered_map<const void*, DirectX::XMFLOAT3> prevLightById;  // BSLight* -> last-frame world pos
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
	DirectX::XMFLOAT3              prevSceneCameraPos{};
	bool                           havePrevSceneCam    = false;
	float                          cameraMovedThisFrame = 0.0f;  // |cameraPos - prevCameraPos| this frame (world units)

	// Camera-relative-caster detector (the "shadow that rides the camera" once lights were ruled out).
	// Track each caster's worldBound.center BY IDENTITY (BSGeometry*); a caster drawn in camera-relative
	// space has its center move by ~ -cameraDelta each frame, i.e. |delta| ~= cameraMovedThisFrame,
	// whereas a correct world-space caster stays put. Captured mid-motion via the delayed dump.
	std::unordered_map<const void*, RE::NiPoint3> prevCasterCenter;  // BSGeometry* -> last-frame world bound centre
	int                            castersRidingCam  = 0;      // casters whose centre moved ~= the camera this frame
	float                          casterRideMaxDelta = 0.0f;  // worst rider's centre movement (world units)
	std::string                    casterRideName;             // worst rider's name (identify the camera-relative caster)
	float             dbgLightDist     = 0.0f;  // camera -> chosen light distance
	DirectX::XMFLOAT3 dbgEye{};        // picked light world position
	DirectX::XMFLOAT3 dbgCam{};        // camera world position
	// (dbgCasterMin/dbgCasterMax moved to VsmDiagnostics — written by ComputeCasterBounds, read by the menu)

	// vertex/index readback of one caster (triggered by a menu button)
	bool              dbgDumpRequested = false;
	bool              dbgLogRequested  = false;  // menu button -> DumpDiagnosticLog() next frame
	int               dumpConfirmFrames = 0;     // >0 -> menu shows a "dump written" confirmation, counts down
	// Delayed dump: arm from the menu, close it, then WALK/TURN — the dump fires mid-motion so
	// cameraMovedThisFrame + per-light moveThisFrame are captured while the camera is actually moving
	// (the menu pauses input, so an immediate dump always reads zero camera movement).
	int               dumpDelaySeconds  = 5;      // menu slider: how long to wait before the delayed dump
	int               dumpCountdownFrames = 0;    // >0 -> counting down to a delayed dump (fires at 0)
	// (dbgHaveDump/dbgStride/dbgIndexCount/dbgFullPrec/dbgV/dbgIdx moved to VsmDiagnostics —
	//  written by InspectCaster, read by the menu's "Inspect caster (raw bytes)" panel)

	bool resourcesReady = false;
	bool haveTestLight  = false;

	// CS<->plugin handshake + frame accounting (diagnostics; see DumpDiagnosticLog).
	uint32_t frameIndex             = 0;  // ++ every RenderFrame tick
	uint32_t dumpOrdinal            = 0;  // ++ every diagnostic dump (orders multiple dumps in one log)
	uint32_t resourceFetchCount     = 0;  // times CS pulled our resources via VSM_GetShadowResources
	uint32_t lastResourceFetchFrame = 0;  // frameIndex at the most recent fetch
};
