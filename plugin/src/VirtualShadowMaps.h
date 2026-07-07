#pragma once

// ============================================================================
// VirtualShadowMaps — standalone SKSE plugin (M0), owned entirely by us.
//
// Renders our OWN depth map for one local shadow light from a minimal
// scene-graph registry, driven by an IDXGISwapChain::Present hook. Its settings
// UI (DrawMenu) is drawn INTO the Community Shaders menu via CS's add-on hook
// (CS_RegisterExternalMenu / CS_GetImGui) — our logic never lives in CS's DLL;
// CS only exposes a tiny generic hook, and our DLL shares CS's ImGui context.
// ============================================================================

#include <DirectXMath.h>
#include <d3d11.h>
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
	void RenderFrame();  // per-frame, from the Present hook
	void DrawMenu();     // ImGui, called from the CS menu via our registered callback

	// Exposed to Plugin.cpp's C exports so LLF's shader can sample our shadows (phase 3).
	ID3D11ShaderResourceView* GetAtlasSRV() const { return srv.Get(); }
	ID3D11ShaderResourceView* GetLightBufferSRV() const { return lightBufferSRV.Get(); }
	ID3D11SamplerState*       GetPointSampler() const { return pointSampler.Get(); }
	ID3D11Buffer*             GetDebugCB() const { return debugCB.Get(); }
	int  GetShadowLightCount() const { return static_cast<int>(lightRecords.size()); }
	bool IsEnabled() const { return enabled; }

	// Real-shader pixel probe: our OMSetRenderTargets hook (Plugin.cpp) binds this UAV at u7 during
	// the lighting draws ONLY while armed, so the real Lighting.hlsl can write its per-pixel state.
	ID3D11UnorderedAccessView* GetPixelProbeUAV() const { return pixelProbeUAV.Get(); }
	bool IsProbeArmed() const { return probeArmed; }

	// Called from VSM_GetShadowResources each time CS pulls our resources to bind — lets the dump
	// verify the CS<->plugin handshake is live (fetch happening on the current frame).
	void NoteResourceFetch() { ++resourceFetchCount; lastResourceFetchFrame = frameIndex; }

private:
	VirtualShadowMaps() = default;

	void SetupResources();
	void CollectLights(RE::ShadowSceneNode* a_ssn);    // gather all active local lights -> lightRecords
	void UploadLightBuffer();                          // mirror lightRecords into the GPU structured buffer
	void UpdateDebugCB();                              // push the live tuning sliders to the shader cbuffer
	void BuildCubeMatrices(DirectX::FXMVECTOR a_eye, float a_near, float a_far, DirectX::XMFLOAT4X4 a_outVP[6]);  // 6 cube-face view-projs (per-light near/far)
	void RebuildRegistrySafe(RE::NiAVObject* a_obj);            // SEH-guarded entry (traversal can hit bad geometry)
	void RebuildRegistry(RE::NiAVObject* a_obj, int a_depth);  // recursive; skips LOD/sky + caps depth
	void RebuildFromReferences();                              // player 3D + loaded-cell references (room/NPCs/clutter)
	void RenderDepth();
	void DumpSkinnedGeometry();      // Path B probe: what skinned-mesh buffers/data exist at Present
	void SkinAllCasters();           // Path B: CPU-skin engineOnly BSTriShape casters -> world-absolute posed buffer
	void ResolvePreview();       // linearize the depth map to a grayscale debug texture
	void InspectCaster();        // read back a caster's raw vertex/index bytes for diagnostics
	void ComputeCasterBounds();  // AABB of caster positions (debug)
	void DumpDiagnosticLog();    // write the coordinate-space vectors to the log for numeric analysis
	void DumpSceneCensus(RE::NiAVObject* a_root, const char* a_label);  // SEH-guarded scene-graph census
	void DumpPlayerDiag(RE::NiAVObject* a_p3d);                          // player geometry + registry membership
	void PlayerWalk(RE::NiAVObject* a_o, int a_depth, int& a_tris, int& a_inReg, int& a_other);  // recursive helper

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
	};

	// One record per shadow-casting light, uploaded to a GPU buffer so the LLF shader
	// (phase 3) can locate + sample the light's cube in the atlas. Must match VSM.hlsli.
	// The 6 face view-projs are included so the shader samples with the EXACT matrices we
	// rendered with (avoids re-deriving the cube projection and getting it subtly wrong).
	struct LightRecord
	{
		DirectX::XMFLOAT4   positionWS = {};    // xyz = light pos (float4 avoids HLSL float3 packing ambiguity)
		float               farPlane   = 0.0f;  // offset 16
		float               nearPlane  = 0.0f;  // 20
		uint32_t            atlasCol   = 0;      // 24
		uint32_t            atlasRow   = 0;      // 28
		DirectX::XMFLOAT4X4 cubeVP[6]  = {};     // 32 — world -> face-clip, per cube face (MUST match VSM.hlsli)
	};

	// cached engine D3D (not owned)
	ID3D11Device*        device = nullptr;
	ID3D11DeviceContext* context = nullptr;

	// our resources
	ComPtr<ID3D11Texture2D>          depthTex;
	ComPtr<ID3D11DepthStencilView>   dsv;
	ComPtr<ID3D11ShaderResourceView> srv;
	ComPtr<ID3D11VertexShader>       depthVS;
	ComPtr<ID3D11InputLayout>        ilFull;    // POSITION R32G32B32_FLOAT (always; see CollectCasters)
	ComPtr<ID3D11DepthStencilState>  depthState;
	ComPtr<ID3D11RasterizerState>    rasterState;
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
	struct SkinnedRange { std::uint32_t ibStart = 0, indexCount = 0; RE::NiPoint3 center; float radius = 0.0f; };
	std::vector<SkinnedRange>       skinnedRanges;
	std::vector<DirectX::XMFLOAT3>  skinPosed;    // CPU scratch: world-absolute posed positions
	std::vector<std::uint32_t>      skinIndices;  // CPU scratch: 32-bit indices into skinPosed
	ComPtr<ID3D11Buffer>            skinnedVB, skinnedIB;
	std::uint32_t                   skinnedVBCap = 0, skinnedIBCap = 0;

	std::vector<CasterEntry> registry;             // persistent; rebuilt only when geometry streams in/out
	std::unordered_set<RE::BSGeometry*> registrySeen;  // dedup across all capture sources this rebuild
	static constexpr size_t  kMaxCasters        = 8192;  // hard cap so a huge cell can't runaway
	// Capture-rejection tally (reset each rebuild): why a BSTriShape we reached was NOT registered.
	int rejNoRD = 0, rejNoVB = 0, rejNoIB = 0, rejZeroTris = 0, rejDup = 0;
	bool                     registryDirty      = true;
	uint32_t                 framesSinceRebuild = 0;
	int                      visibleCasters     = 0;  // drawn this frame (after cull, summed over faces)
	std::vector<LightRecord>         lightRecords;   // all shadowed lights this frame (<= kMaxLights)
	ComPtr<ID3D11Buffer>             lightBuffer;    // GPU structured copy for the LLF shader
	ComPtr<ID3D11ShaderResourceView> lightBufferSRV;

	// Every active light = 6 cube faces = one 3x2 "block". Blocks tile the atlas in a grid.
	// (Correctness-first: modest per-face res + a light cap; both scale up after the perf
	// work in M2/M4.)
	static constexpr int kFaceRes      = 256;
	static constexpr int kMaxLights    = 32;
	static constexpr int kLightsPerRow = 4;
	static constexpr int kBlockW       = kFaceRes * 3;
	static constexpr int kBlockH       = kFaceRes * 2;
	static constexpr int kAtlasW       = kLightsPerRow * kBlockW;
	static constexpr int kAtlasH       = ((kMaxLights + kLightsPerRow - 1) / kLightsPerRow) * kBlockH;

	// settings (driven by DrawMenu)
	bool  enabled      = false;
	bool  frustumCull  = true;    // cull casters to the light frustum (leaf-level)
	// Do NOT offset casters: geom->world.translate under the ShadowSceneNode is ALREADY game-absolute
	// (confirmed vs CS: LightLimitFix passes niLight->world.translate as the absolute light pos and
	// SUBTRACTS the eye to get camera-relative positionWS). Adding cameraPos (the old =true default)
	// shifted casters ~1300u off the lights -> empty atlases -> no shadows. Light cubes are absolute
	// and the shader samples absP (P + CameraPosAdjust = world), so no-offset geometry aligns exactly.
	bool  dbgCastersAbsolute = false;

	// Reliable render eye = ShadowSceneNode::cameraPos (set each frame in CollectLights). Offset the
	// atlas by THIS, NOT RendererShadowState::posAdjust.getEye() — the latter intermittently returns
	// garbage (0,0,1) that mis-places the whole atlas by the camera position (dump #3).
	DirectX::XMFLOAT3 sceneCameraPos{};

	// debug view — RenderFrame does the debug-only GPU work (preview resolve, AABB scan,
	// inspection) ONLY when the Debug section is open AND the menu is actually on screen.
	bool  showDebug    = false;
	bool  menuVisible  = false;  // set by DrawMenu each frame the menu is drawn
	float previewScale = 0.25f;

	// debug-view controls
	int   lightSelect   = -1;     // -1 = nearest local light to camera; else index into active lights
	float previewRange  = 2000.0f;// world units mapped white->black in the preview
	int   isolateCaster = 0;      // 0 = render all; N = render only caster N-1 (debug)

	// Tuning COEFFICIENTS for the per-light shadow calculation. Each light's near/far are
	// derived from its own radius (read in CollectLights); these sliders only scale/shift that
	// derivation, so their meaning is explicit:
	//     farPlane  = light.radius * FarScale
	//     nearPlane = farPlane     * NearFrac
	//     shadowed  = (pixelDist - occluderDist) > BiasWorld     // all in WORLD units
	// Live via debugCB @ b13 (no rebuild). Mode/VizScale are debug-view only.
	float dbgFarScale  = 1.0f;     // far  = light radius * this
	float dbgNearFrac  = 0.01f;    // near = far * this (precision knob; smaller = tighter)
	float dbgBiasWorld = 3.0f;     // shadow bias in WORLD UNITS (linear-distance compare)
	int   dbgMode      = 0;        // 0 = shadow  1 = off(lit)  >=2 RGB diagnostic overlays
	float dbgVizScale  = 2000.0f;  // grayscale scale for the atlas-depth view (debug only)

	// Two live diagnostic DIMENSIONS that re-test every mode without a rebuild. The atlas is
	// rasterized in ABSOLUTE world space (geom->world + absolute-light cubeVP), so absP is the
	// physically-correct sample space; 1/2 are controls to prove it. See VSM.hlsli::SampleP.
	int   dbgSampleSpace = 0;      // 0 = absP (P + CameraPosAdjust)  1 = P (cam-rel)  2 = P + altEye(C++ render eye)
	float dbgMatchThresh = 5.0f;   // light-match distance (world units): buffer light vs shader light
	int   dbgCompareMode = 0;      // 0 = linearized-distance compare  1 = raw ndc.z compare (isolates linearize)
	// The shader light passed to GetLocalShadow (LLF's positionWS) is camera-relative to
	// posAdjust.getEye() (== altEye), NOT to FrameBuffer::CameraPosAdjust. If those two eyes
	// differ, matching the shader light to our absolute buffer with CameraPosAdjust drifts with
	// the camera. Default to altEye (the provably-correct eye); toggle back to compare.
	// Match with the shader's OWN same-frame FrameBuffer::CameraPosAdjust (0), exactly as LLF builds
	// light.positionWS. Using our altEye (1) is 1 frame stale in the shader's b13 cbuffer, so on fast
	// camera motion the match drifts past matchThresh and the shadow blinks off (flicker). 0 = robust.
	int   dbgMatchEye    = 0;      // 0 = CameraPosAdjust (default, same-frame)  1 = altEye (stale)
	bool  probeArmed     = false;  // menu: arm the real-shader pixel probe (writes centre pixel to u7)
	ComPtr<ID3D11Buffer> debugCB;  // 48-byte cbuffer @ b13 (see UpdateDebugCB / VSM.hlsli::VSMDebug)

	// GPU shadow-math PROBE. Runs the EXACT VSM sample logic on the GPU against the LIVE atlas +
	// light buffer for a handful of probe points, and reads every intermediate back to the log.
	// This is the only way to prove what the shader really computes (bound resources, real sampler,
	// GPU precision) — CPU replication can't. Driven by the diagnostic-dump button.
	static constexpr int kMaxProbes = 128;  // covers every light (nearest caster, a few verts each) in one dispatch
	ComPtr<ID3D11ComputeShader>       probeCS;
	ComPtr<ID3D11Buffer>              probeInBuf;    // StructuredBuffer<ProbeIn>  (t2)
	ComPtr<ID3D11ShaderResourceView>  probeInSRV;
	ComPtr<ID3D11Buffer>              probeOutBuf;   // RWStructuredBuffer<ProbeOut> (u0)
	ComPtr<ID3D11UnorderedAccessView> probeOutUAV;
	ComPtr<ID3D11Buffer>              probeOutStaging;
	ComPtr<ID3D11Buffer>              probeCB;       // ProbeCB (b0)

	// Real-shader pixel probe (u7): written by VSM.hlsli for the centre pixel when armed.
	ComPtr<ID3D11Buffer>              pixelProbeBuf;
	ComPtr<ID3D11UnorderedAccessView> pixelProbeUAV;
	ComPtr<ID3D11Buffer>              pixelProbeStaging;

	// diagnostics (shown in the menu to sanity-check world-space coordinates)
	int               activeLightCount = 0;    // valid local lights this frame (shadow-casters, after filtering)
	int               lightsAmbient    = 0;    // active lights skipped this frame: ambient/fill (never shadow)
	int               lightsNonShadow  = 0;    // active lights skipped this frame: not flagged shadow-casters
	std::vector<uint8_t>           lightDynamic;   // per shadow-light: BSLight::dynamic (a moving/animated light?)
	std::vector<DirectX::XMFLOAT3> prevLightPos;   // last frame's shadow-light positions (to detect jitter/movement)
	std::vector<float>             lightMove;       // per shadow-light: units moved since last frame (jitter = culprit)
	float             dbgLightDist     = 0.0f;  // camera -> chosen light distance
	DirectX::XMFLOAT3 dbgEye{};        // picked light world position
	DirectX::XMFLOAT3 dbgCam{};        // camera world position
	DirectX::XMFLOAT3 dbgCasterMin{};  // AABB of collected caster world positions
	DirectX::XMFLOAT3 dbgCasterMax{};

	// vertex/index readback of one caster (triggered by a menu button)
	bool              dbgDumpRequested = false;
	bool              dbgLogRequested  = false;  // menu button -> DumpDiagnosticLog() next frame
	int               dumpConfirmFrames = 0;     // >0 -> menu shows a "dump written" confirmation, counts down
	bool              dbgHaveDump      = false;
	uint32_t          dbgStride        = 0;
	uint32_t          dbgIndexCount    = 0;
	bool              dbgFullPrec      = false;
	DirectX::XMFLOAT3 dbgV[4]          = {};  // first 4 decoded local vertex positions (float3)
	uint32_t          dbgIdx[6]        = {};  // first 6 indices (16-bit)

	bool resourcesReady = false;
	bool haveTestLight  = false;

	// CS<->plugin handshake + frame accounting (diagnostics; see DumpDiagnosticLog).
	uint32_t frameIndex             = 0;  // ++ every RenderFrame tick
	uint32_t dumpOrdinal            = 0;  // ++ every diagnostic dump (orders multiple dumps in one log)
	uint32_t resourceFetchCount     = 0;  // times CS pulled our resources via VSM_GetShadowResources
	uint32_t lastResourceFetchFrame = 0;  // frameIndex at the most recent fetch
};
