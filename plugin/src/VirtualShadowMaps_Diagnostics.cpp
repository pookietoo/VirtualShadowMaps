// ============================================================================
// VirtualShadowMaps — diagnostics translation unit.
//
// All the numeric dumps, GPU/pixel probes, scene-graph census, caster inspection,
// and the debug depth-preview resolve live here, split out of the core render path
// (VirtualShadowMaps.cpp) so that file stays focused on setup + light collection +
// skinning + the atlas render. The pure-diagnostic entry points are methods of the
// extracted VsmDiagnostics class (declared in VirtualShadowMaps_Diagnostics.h); each
// reaches the core render-state through m_core (aliased locally at the top of every
// method so the bodies stay byte-identical). The render-path helpers ComputeVertAABB /
// GetFreshWorldAABB / NodeOwningRef were MOVED to the core unit (VirtualShadowMaps.cpp) because
// the per-frame render path calls them and this unit is dropped from a deploy build; only
// ComputeVertAABB is still called from here (via m_core). Shared helpers come from VSMInternal.h.
// ============================================================================

#include "VirtualShadowMaps.h"
#include "VirtualShadowMaps_Diagnostics.h"
#include "VSMInternal.h"

#include <DirectXPackedVector.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <string>
#include <vector>

// Skinned-geometry probe reads skin partitions + the engine bone palette (Path B diagnostics).
#include "RE/B/BSGeometry.h"
#include "RE/B/BSDynamicTriShape.h"
#include "RE/N/NiSkinInstance.h"
#include "RE/N/NiSkinData.h"
#include "RE/N/NiSkinPartition.h"
#include "RE/N/NiAlphaProperty.h"      // alpha blend/test flags for the focused-caster dump
#include "RE/B/BSShaderProperty.h"     // shader type / material type for the focused-caster dump
#include "RE/B/BSLightingShaderProperty.h"  // emissive color/mult (glow) per caster
#include "RE/T/TESObjectREFR.h"        // ref identity (base object, linked ref, cell) for the object dump
#include "RE/T/TESModel.h"             // mesh/NIF path of a ref's base object
#include "RE/T/TESObjectCELL.h"        // parent cell formID
#include "RE/N/NiAVObject.h"           // parent-chain walk for a light's owning ref
#include "RE/N/NiNode.h"               // GetChildren() for the full scene-graph walk
#include "RE/B/BSShaderManager.h"      // shadowSceneNode[] roots (full scene dump)
#include "RE/B/BSEffectShaderProperty.h"  // effect-shader detection (full scene dump)
#include "RE/P/PlayerCharacter.h"      // player 3D + parent cell references (full scene dump)
#include "RE/E/ExtraEnableStateParent.h"  // enable-parent link between separate refs

using namespace DirectX;
using namespace vsm;
using namespace vsm::internal;

namespace
{
	// ---- Reverse-Z depth atlas ----
	// The atlas stores reverse-Z depth: an empty/far texel reads 0.0, an occluder reads (0, 1] with
	// near = 1.0. A texel only counts as "populated" (a real occluder was rasterized) when its depth
	// exceeds this epsilon; the same gate rejects empty texels when hunting for the nearest occluder.
	inline constexpr float kEmptyDepthEps = 0.001f;

	// ---- Homogeneous-w guards ----
	// A clip-space w above kClipWFrontEps means the point is in front of the light plane; a |w| below
	// kClipWEps means the (inverse-)projection is degenerate and the point is unusable.
	inline constexpr float kClipWFrontEps = 1e-4f;
	inline constexpr float kClipWEps      = 1e-6f;
	// Divide-by-zero floor for the reverse-Z linearizer's denominator.
	inline constexpr float kLinDenomFloor = 1e-4f;

	// ---- Misc sentinels ----
	inline constexpr float kNdcSentinel = 9.0f;    // ndc coordinate for a point behind the light plane
	inline constexpr float kBigDistance = 1e30f;   // stand-in for +infinity when minimizing a distance

	// ---- Scene-graph walk / caches ----
	inline constexpr int         kSceneWalkMaxDepth  = 128;    // recursion cap for the SEH-guarded graph walks
	inline constexpr float       kSkinSpaceTolerance = 150.0f; // world-unit tolerance for the skinning-space classifier

	// ---- Atlas face-sampling strides (texels between samples) ----
	inline constexpr int kFaceScanStride = 4;  // coarse round-trip / synthetic-shadow passes
	inline constexpr int kPopScanStride  = 2;  // population / nearest-occluder scan

	// ---- Debug vertex/index probe counts ----
	inline constexpr int kInspectVerts   = 4;   // leading verts decoded by InspectCaster (== dbgV[])
	inline constexpr int kInspectIndices = 6;   // leading indices decoded by InspectCaster (== dbgIdx[])
	inline constexpr int kFocusVertsRead = 12;  // leading verts DumpFocusedCaster reads back from the VB
	inline constexpr int kFocusVertsLog  = 8;   // leading verts DumpFocusedCaster projects + logs

	// Cube-face index from a direction vector: the axis of largest magnitude picks the face
	// (+X -X +Y -Y +Z -Z = 0..5). Mirrors VSM.hlsli's face selection exactly.
	int faceFromDir(float x, float y, float z)
	{
		const float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z);
		if (ax >= ay && ax >= az) return x >= 0 ? 0 : 1;
		if (ay >= az)             return y >= 0 ? 2 : 3;
		return z >= 0 ? 4 : 5;
	}
	inline constexpr const char* kFaceName[6] = { "+X", "-X", "+Y", "-Y", "+Z", "-Z" };

	// Scene-graph census: counts what our RebuildRegistry traversal reaches, by kind. triShapes are
	// what we capture; otherLeaves are non-node, non-BSTriShape leaves (geometry types we SKIP —
	// e.g. BSDynamicTriShape/BSSubIndexTriShape/particles). If otherLeaves >> triShapes, we're
	// missing most geometry because we only handle AsTriShape().
	struct SceneCensus { int nodes = 0, triShapes = 0, otherLeaves = 0, maxDepth = 0, lodSkipped = 0; const char* sampleOther = nullptr; };
	void CensusWalk(RE::NiAVObject* o, int depth, SceneCensus& c)
	{
		if (!o || depth > kSceneWalkMaxDepth)
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
}

// SEH-guarded scene census (like RebuildRegistrySafe — a live scene-graph walk can hit torn-down
// geometry). Logs what our traversal reaches from `a_root`, total and per direct child, so we can
// see whether the room geometry is even under this root and whether we're missing geometry types.
void VsmDiagnostics::DumpSceneCensus(RE::NiAVObject* a_root, const char* a_label)
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
void VsmDiagnostics::PlayerWalk(RE::NiAVObject* a_o, int a_depth, int& a_tris, int& a_inReg, int& a_other)
{
	auto& registry = m_core.registry;  // core render-state, reached through the owning VirtualShadowMaps
	if (!a_o || a_depth > kSceneWalkMaxDepth)
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

void VsmDiagnostics::DumpPlayerDiag(RE::NiAVObject* a_p3d)
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

void VsmDiagnostics::ResolvePreview()
{
	// core render-state, reached through the owning VirtualShadowMaps (previewRange is our own member)
	auto& context      = m_core.context;
	auto& previewRTV   = m_core.previewRTV;
	auto& fullscreenVS = m_core.fullscreenVS;
	auto& linearizePS  = m_core.linearizePS;
	auto& srv          = m_core.srv;
	auto& pointSampler = m_core.pointSampler;
	auto& previewCB    = m_core.previewCB;
	auto& rtAtlasW     = m_core.rtAtlasW;
	auto& rtAtlasH     = m_core.rtAtlasH;

	if (!previewRTV || !fullscreenVS || !linearizePS || !srv)
		return;

	GraphicsStateGuard guard(context);

	ID3D11RenderTargetView* rtv = previewRTV.Get();
	context->OMSetRenderTargets(1, &rtv, nullptr);
	D3D11_VIEWPORT vp{ 0, 0, (float)rtAtlasW, (float)rtAtlasH, 0, 1 };
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

void VsmDiagnostics::InspectCaster()
{
	// core render-state, reached through the owning VirtualShadowMaps (dbg* are our own members)
	auto& registry      = m_core.registry;
	auto& device        = m_core.device;
	auto& context       = m_core.context;
	auto& isolateCaster = m_core.isolateCaster;

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
	if (readback(vbuf, dbgStride * kInspectVerts + 16, vb)) {
		for (int i = 0; i < kInspectVerts; ++i) {
			const size_t off = static_cast<size_t>(i) * dbgStride;
			if (off + 3 * sizeof(float) > vb.size()) {  // need a full float3 position
				dbgV[i] = {};
				continue;
			}
			const float* f = reinterpret_cast<const float*>(vb.data() + off);  // position: float3 @ offset 0
			dbgV[i] = { f[0], f[1], f[2] };
		}
	}

	std::vector<uint8_t> ib;
	if (readback(ibuf, kInspectIndices * sizeof(uint16_t), ib)) {
		const uint16_t* ix = reinterpret_cast<const uint16_t*>(ib.data());
		for (int i = 0; i < kInspectIndices && (i + 1) * sizeof(uint16_t) <= ib.size(); ++i)
			dbgIdx[i] = ix[i];
	}

	dbgHaveDump = true;
}

// AABB of collected caster world positions (debug sanity-check of the transforms).
void VsmDiagnostics::ComputeCasterBounds()
{
	auto& registry = m_core.registry;  // core render-state (dbgCasterMin/Max are our own members)
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

// Skinned-geometry probe (Path B foundation). For every engineOnly (skinned) caster, log what data is
// ACTUALLY reachable at our Present-time tick — so we build the compute-skinning path from real values,
// not assumptions. CPU-skinned (BSDynamicTriShape) exposes posed verts in dynamicData; GPU-skinned
// (BSTriShape + skinInstance) exposes bind pose + bones/weights in the skin partition. SEH-guarded.
void VsmDiagnostics::DumpSkinnedGeometry()
{
	// core render-state, reached through the owning VirtualShadowMaps
	auto& registry             = m_core.registry;
	auto& sceneCameraPos       = m_core.sceneCameraPos;
	const float detachMargin   = vsm::kDetachMargin;  // fixed guard slack (was m_core.detachMargin; now a constant)
	auto& skinnedRendered      = m_core.skinnedRendered;
	auto& skinnedDetached      = m_core.skinnedDetached;
	auto& skinnedMaxDetachDist = m_core.skinnedMaxDetachDist;
	auto& skinnedMaxDetachName = m_core.skinnedMaxDetachName;
	auto& skinnedClampedVerts  = m_core.skinnedClampedVerts;

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

				// SKINNING-SPACE CHECK (diagnostic only): does bone matrix 0's origin land at this caster's
				// WORLD bound (=world-absolute), world-minus-camera (=camera-relative, drifts), or the origin
				// (=model space)? CAVEAT: bone 0 is NOT the mesh center — it's whatever bone leads the palette
				// (often the root), so a legitimately displaced/ragdolled NPC reads OTHER here as a FALSE
				// POSITIVE. The authoritative test is the self-referential spread guard in SkinAllCasters; this
				// classifier is only a coarse hint. (Confirmed: a coherent female NPC flagged OTHER by bone0.)
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
						const char* cls = (ew < kSkinSpaceTolerance) ? "WORLD" : (ecr < kSkinSpaceTolerance) ? "CAMERA-REL(BUG)" : (em < kSkinSpaceTolerance) ? "MODEL(BUG)" : "OTHER(BUG)";
						if (ew < kSkinSpaceTolerance)       ++spaceWorld;
						else if (ecr < kSkinSpaceTolerance) ++spaceCamRel;
						else if (em < kSkinSpaceTolerance)  ++spaceModel;
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
		// REAL posed-vertex guard result from this frame's SkinAllCasters (measured, not the bone0 heuristic
		// above, which false-flags a displaced-but-coherent NPC): how many skinned casters posed COHERENTLY
		// and cast, vs how many were dropped for a SCATTERED pose (spread > radius + detachMargin = a real
		// skinning explosion / wrong-space palette). worst = largest pose spread from its own centroid.
		// Both the guard and the clamp are self-referential (centroid + radius), NOT worldBound.center —
		// so a valid shadow is no longer dropped when the mesh's bound goes stale relative to its skeleton.
		logger::info("skinned guard: rendered(coherent)={} dropped(scattered)={} worstSpread={:.0f}u '{}' clampedVerts(exploded->centroid)={}  (drop threshold = radius + detachMargin={:.0f}u)",
		    skinnedRendered, skinnedDetached, skinnedMaxDetachDist, skinnedMaxDetachName.empty() ? "-" : skinnedMaxDetachName.c_str(), skinnedClampedVerts, detachMargin);
	} __except (EXCEPTION_EXECUTE_HANDLER) {
		logger::warn("VSM skinned-geometry probe faulted");
	}
}

// GEOMETRY DUMP — write every static caster's REAL world-space triangles (positions + indices) to a
// binary sidecar. A bounding box over-occludes (a box isn't a mesh); triangles give the EXACT silhouette,
// so the offline "expected shadow" is a real ray-vs-triangle test, not ray-vs-box. Per-record layout:
//   u32 registryIndex, u32 vcount, u32 tcount, f32[vcount*3] worldPos, u32[tcount*3] indices.
void VsmDiagnostics::DumpGeometry()
{
	// core render-state, reached through the owning VirtualShadowMaps
	auto& registry      = m_core.registry;
	auto& device        = m_core.device;
	auto& context       = m_core.context;
	auto& skinnedRanges = m_core.skinnedRanges;
	auto& skinIndices   = m_core.skinIndices;
	auto& skinPosed     = m_core.skinPosed;

	using namespace DirectX;
	if (registry.empty() || !device || !context)
		return;
	auto dir = logger::log_directory();
	if (!dir)
		return;
	std::ofstream gf(*dir / "VirtualShadowMaps_geom.bin", std::ios::binary | std::ios::trunc);
	if (!gf) {
		logger::warn("VSM geom dump: cannot open geom.bin");
		return;
	}

	ComPtr<ID3D11Buffer> vstg;
	UINT vstgSz = 0;
	ComPtr<ID3D11Buffer> istg;
	UINT istgSz = 0;
	std::vector<uint8_t> vraw;
	std::vector<uint16_t> iraw;
	std::vector<float> wpos;
	uint32_t nC = 0;
	uint64_t totV = 0, totT = 0;

	for (size_t i = 0; i < registry.size(); ++i) {
		RE::BSGeometry* g = registry[i].geom.get();
		if (!g)
			continue;
		auto* rd = g->GetGeometryRuntimeData().rendererData;
		const UINT stride = registry[i].vertexStride;
		const UINT icount = registry[i].indexCount;
		if (!rd || !rd->vertexBuffer || !rd->indexBuffer || stride == 0 || icount == 0)
			continue;  // skinned/streamed -> handled in the skinned-range pass below

		auto* vb = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
		D3D11_BUFFER_DESC vbd{};
		vb->GetDesc(&vbd);
		const UINT vbytes = vbd.ByteWidth;
		if (vbytes < stride)
			continue;
		if (!vstg || vstgSz < vbytes) {
			vstg.Reset();
			D3D11_BUFFER_DESC sd{};
			sd.ByteWidth = vbytes;
			sd.Usage = D3D11_USAGE_STAGING;
			sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			if (FAILED(device->CreateBuffer(&sd, nullptr, &vstg))) {
				vstgSz = 0;
				continue;
			}
			vstgSz = vbytes;
		}
		{
			D3D11_BOX b{ 0, 0, 0, vbytes, 1, 1 };
			context->CopySubresourceRegion(vstg.Get(), 0, 0, 0, 0, vb, 0, &b);
		}
		D3D11_MAPPED_SUBRESOURCE vm{};
		if (FAILED(context->Map(vstg.Get(), 0, D3D11_MAP_READ, 0, &vm)))
			continue;
		vraw.assign(static_cast<const uint8_t*>(vm.pData), static_cast<const uint8_t*>(vm.pData) + vbytes);
		context->Unmap(vstg.Get(), 0);
		const UINT vcount = vbytes / stride;

		auto* ib = reinterpret_cast<ID3D11Buffer*>(rd->indexBuffer);
		const UINT ibytes = icount * static_cast<UINT>(sizeof(uint16_t));
		if (!istg || istgSz < ibytes) {
			istg.Reset();
			D3D11_BUFFER_DESC sd{};
			sd.ByteWidth = ibytes;
			sd.Usage = D3D11_USAGE_STAGING;
			sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			if (FAILED(device->CreateBuffer(&sd, nullptr, &istg))) {
				istgSz = 0;
				continue;
			}
			istgSz = ibytes;
		}
		{
			D3D11_BOX b{ 0, 0, 0, ibytes, 1, 1 };
			context->CopySubresourceRegion(istg.Get(), 0, 0, 0, 0, ib, 0, &b);
		}
		D3D11_MAPPED_SUBRESOURCE im{};
		if (FAILED(context->Map(istg.Get(), 0, D3D11_MAP_READ, 0, &im)))
			continue;
		iraw.assign(reinterpret_cast<const uint16_t*>(im.pData), reinterpret_cast<const uint16_t*>(im.pData) + icount);
		context->Unmap(istg.Get(), 0);

		const XMMATRIX world = NiTransformToXM(g->world);  // exact matrix RenderDepth rasterizes with
		wpos.resize(static_cast<size_t>(vcount) * 3);
		for (UINT v = 0; v < vcount; ++v) {
			const float* f = reinterpret_cast<const float*>(vraw.data() + static_cast<size_t>(v) * stride);
			XMFLOAT3 wp;
			XMStoreFloat3(&wp, XMVector3TransformCoord(XMVectorSet(f[0], f[1], f[2], 1.0f), world));
			wpos[v * 3 + 0] = wp.x;
			wpos[v * 3 + 1] = wp.y;
			wpos[v * 3 + 2] = wp.z;
		}
		const uint32_t idx = static_cast<uint32_t>(i), vc = vcount, tc = icount / 3u;
		gf.write(reinterpret_cast<const char*>(&idx), sizeof(uint32_t));
		gf.write(reinterpret_cast<const char*>(&vc), sizeof(uint32_t));
		gf.write(reinterpret_cast<const char*>(&tc), sizeof(uint32_t));
		gf.write(reinterpret_cast<const char*>(wpos.data()), static_cast<std::streamsize>(vcount) * 3 * sizeof(float));
		for (UINT k = 0; k < icount; ++k) {
			const uint32_t ii = iraw[k];
			gf.write(reinterpret_cast<const char*>(&ii), sizeof(uint32_t));
		}
		++nC;
		totV += vcount;
		totT += tc;
	}

	// SKINNED casters (player/NPCs): posed WORLD verts already computed in skinPosed/skinIndices. Write
	// each range with verts inline in index order (indices become trivial 0..n) — same record format.
	uint32_t nSkin = 0;
	for (const auto& r : skinnedRanges) {
		if (r.registryIndex < 0 || r.indexCount == 0)
			continue;
		if (static_cast<size_t>(r.ibStart) + r.indexCount > skinIndices.size())
			continue;
		const uint32_t idx = static_cast<uint32_t>(r.registryIndex), vc = r.indexCount, tc = r.indexCount / 3u;
		gf.write(reinterpret_cast<const char*>(&idx), sizeof(uint32_t));
		gf.write(reinterpret_cast<const char*>(&vc), sizeof(uint32_t));
		gf.write(reinterpret_cast<const char*>(&tc), sizeof(uint32_t));
		bool ok = true;
		for (uint32_t k = 0; k < r.indexCount; ++k) {
			const uint32_t gi = skinIndices[r.ibStart + k];
			if (gi >= skinPosed.size()) {
				ok = false;
				break;
			}
			const auto& p = skinPosed[gi];
			gf.write(reinterpret_cast<const char*>(&p.x), sizeof(float));
			gf.write(reinterpret_cast<const char*>(&p.y), sizeof(float));
			gf.write(reinterpret_cast<const char*>(&p.z), sizeof(float));
		}
		if (!ok)
			continue;
		for (uint32_t k = 0; k < r.indexCount; ++k)
			gf.write(reinterpret_cast<const char*>(&k), sizeof(uint32_t));  // sequential
		++nC;
		++nSkin;
		totV += vc;
		totT += tc;
	}
	logger::info("=== GEOM DUMP: VirtualShadowMaps_geom.bin  casters={} (skinned={}) totalVerts={} totalTris={}  (per-record LE: u32 registryIndex, u32 vcount, u32 tcount, f32[vcount*3] worldPos, u32[tcount*3] indices; skinned verts inline in index order) ===",
	    nC, nSkin, totV, totT);
}

// COMPLETE SKINNING STATE DUMP — for every caster with a skin instance, write EVERYTHING the CPU/GPU holds
// that determines its posed vertices, to a binary sidecar (skin.bin), so the full skinning can be reproduced
// and diagnosed OFFLINE with zero further captures: the whole bone-matrix PALETTE, the whole dynamicData
// (morphed model-space positions), and per partition the bones remap, vertexMap, triList, and per-vertex raw
// weights+bone-slots+buffData position. All the referenced buffers (skin->boneMatrices, dynamicData,
// partition buffData->rawVertexData, bones/vertexMap/triList) are CPU-readable at Present (the skinned probe
// already reads them directly). Also logs a readable per-caster summary + the Σ p.vertices==dataSize/16 check.
void VsmDiagnostics::DumpSkinningFull()
{
	using namespace DirectX;
	using V = RE::BSGraphics::Vertex;
	auto& registry = m_core.registry;
	if (registry.empty())
		return;
	auto dir = logger::log_directory();
	if (!dir)
		return;
	std::ofstream sf(*dir / "VirtualShadowMaps_skin.bin", std::ios::binary | std::ios::trunc);
	if (!sf) {
		logger::warn("VSM skin dump: cannot open skin.bin");
		return;
	}
	auto wr = [&sf](const void* p, size_t n) { sf.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n)); };
	auto wu32 = [&](std::uint32_t v) { wr(&v, 4); };
	auto wu16 = [&](std::uint16_t v) { wr(&v, 2); };

	logger::info("=== FULL SKINNING DUMP -> VirtualShadowMaps_skin.bin  (per record LE: u32 MAGIC='SKIN', u32 regIdx, u32 isDynamic, "
	             "char[64] name, f32[16] worldMat, f32[4] boundC+r, u32 numMats,u32 matFloats, f32[numMats*matFloats] palette, "
	             "u32 dynVerts, f32[dynVerts*3] dynamicData, u32 numParts, [u16 verts,tris,numBones,strips; u32 stride,skinOff,posOff,"
	             "vdFlags,fullprec; u16[numBones] bones; u16[verts] vertexMap; u16[tris*3] triList; per-vert(u16[4] wHalf,u8[4] slot,"
	             "f32[3] bufPos)]xnumParts) ===");

	std::uint32_t nRec = 0;
	for (std::uint32_t ri = 0; ri < registry.size(); ++ri) {
		RE::BSGeometry* g = registry[ri].geom.get();
		if (!g)
			continue;
		auto* skin = g->GetGeometryRuntimeData().skinInstance.get();
		if (!skin || !skin->skinPartition || !skin->boneMatrices || skin->numMatrices == 0)
			continue;  // not a skinned caster
		{
			auto* sp = skin->skinPartition.get();
			auto* dts = g->AsDynamicTriShape();
			const bool isDyn = dts != nullptr;
			const std::uint8_t* dyn = isDyn ? reinterpret_cast<const std::uint8_t*>(dts->GetDynamicTrishapeRuntimeData().dynamicData) : nullptr;
			const std::uint32_t dynSize = isDyn ? dts->GetDynamicTrishapeRuntimeData().dataSize : 0u;
			const std::uint32_t dynVerts = (dyn && dynSize >= 16) ? dynSize / 16u : 0u;
			const std::uint32_t numMats = skin->numMatrices;
			const std::uint32_t matFloats = (skin->allocatedSize / numMats) / static_cast<std::uint32_t>(sizeof(float));
			const float* pal = reinterpret_cast<const float*>(skin->boneMatrices);

			// --- record header ---
			wu32(0x4E494B53);  // 'SKIN' LE
			wu32(ri);
			wu32(isDyn ? 1u : 0u);
			char nm[64]{};
			std::strncpy(nm, g->name.c_str() ? g->name.c_str() : "", sizeof(nm) - 1);
			wr(nm, sizeof(nm));
			XMFLOAT4X4 wm; XMStoreFloat4x4(&wm, NiTransformToXM(g->world));
			wr(&wm, sizeof(wm));
			const float bc[4]{ g->worldBound.center.x, g->worldBound.center.y, g->worldBound.center.z, g->worldBound.radius };
			wr(bc, sizeof(bc));
			// --- full bone palette ---
			wu32(numMats); wu32(matFloats);
			wr(pal, static_cast<size_t>(numMats) * matFloats * sizeof(float));
			// --- full dynamicData (morphed model-space positions) ---
			wu32(dynVerts);
			for (std::uint32_t v = 0; v < dynVerts; ++v) {
				const float* lp = reinterpret_cast<const float*>(dyn + static_cast<size_t>(v) * 16u);
				const float xyz[3]{ lp[0], lp[1], lp[2] };
				wr(xyz, sizeof(xyz));
			}
			// --- per-partition full state ---
			wu32(sp->numPartitions);
			std::uint32_t sumVerts = 0;
			for (std::uint32_t pi = 0; pi < sp->numPartitions; ++pi) {
				auto& p = sp->partitions[pi];
				auto& vd = p.vertexDesc;
				const std::uint32_t stride = vd.GetSize();
				const std::uint32_t skinOff = vd.GetAttributeOffset(V::VA_SKINNING);
				const std::uint32_t posOff  = vd.GetAttributeOffset(V::VA_POSITION);
				sumVerts += p.vertices;
				wu16(p.vertices); wu16(p.triangles); wu16(p.numBones); wu16(p.strips);
				wu32(stride); wu32(skinOff); wu32(posOff);
				wu32(static_cast<std::uint32_t>(vd.GetFlags())); wu32(vd.HasFlag(V::VF_FULLPREC) ? 1u : 0u);
				for (std::uint16_t b = 0; b < p.numBones; ++b) wu16(p.bones ? p.bones[b] : 0xFFFFu);
				for (std::uint16_t lv = 0; lv < p.vertices; ++lv) wu16(p.vertexMap ? p.vertexMap[lv] : 0xFFFFu);
				for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(p.triangles) * 3u; ++k) wu16(p.triList ? p.triList[k] : 0u);
				const std::uint8_t* rv = p.buffData ? reinterpret_cast<const std::uint8_t*>(p.buffData->rawVertexData) : nullptr;
				for (std::uint16_t lv = 0; lv < p.vertices; ++lv) {
					std::uint16_t wh[4]{}; std::uint8_t slot[4]{}; float bp[3]{};
					if (rv && stride) {
						const std::uint8_t* vb = rv + static_cast<size_t>(lv) * stride;
						if (skinOff + 12 <= stride) {
							std::memcpy(wh, vb + skinOff, 8);
							std::memcpy(slot, vb + skinOff + 8, 4);
						}
						if (posOff + 12 <= stride)
							std::memcpy(bp, vb + posOff, 12);
					}
					wr(wh, sizeof(wh)); wr(slot, sizeof(slot)); wr(bp, sizeof(bp));
				}
			}
			// --- readable summary + the layout sanity check ---
			logger::info("  skinFULL[{}] '{}' dyn={} parts={} dynVerts={} sumPartVerts={} spVertexCount={} {} numMats={} matFloats={}",
			    ri, nm, isDyn ? 1 : 0, sp->numPartitions, dynVerts, sumVerts, sp->vertexCount,
			    (dynVerts == sumVerts) ? "[sum==dynVerts OK]" : "[MISMATCH!]", numMats, matFloats);
			for (std::uint32_t pi = 0; pi < sp->numPartitions; ++pi) {
				auto& p = sp->partitions[pi];
				std::string bones;
				for (std::uint16_t b = 0; b < p.numBones && b < 32; ++b)
					bones += std::format("{} ", p.bones ? p.bones[b] : 0xFFFF);
				logger::info("      p{} verts={} tris={} numBones={} bones[0..{}]={} {}",
				    pi, p.vertices, p.triangles, p.numBones, (std::min<int>)(p.numBones, 32) - 1, bones,
				    (p.numBones > 0 && p.bones && p.bones[0] == 0 && (p.numBones < 2 || p.bones[1] == 1)) ? "(looks identity-ish)" : "(NON-identity remap)");
			}
			++nRec;
		}
	}
	logger::info("=== FULL SKINNING DUMP: wrote {} skinned casters to skin.bin ===", nRec);
}

// COMPLETE SCENE DUMP — serialize LITERALLY EVERYTHING reachable in the render scene graph to a binary
// sidecar (scene.bin): every node (pre-order) with its type/name/flags/local+world transform/bound, and for
// every geometry the STAGED GPU vertex+index buffers, vertexDesc, dynamicData, shader/alpha properties, and
// the FULL skin state (bone palette, per-bone world transforms, NiSkinData inverse-binds, and every partition
// with bones/vertexMap/triList/p.weights/p.bonePalette + per-vertex embedded weights/slots/pos). Roots: all
// four shadowSceneNode[] + player 3D + every loaded-cell reference's 3D. Deduped by address, depth-capped.
// This is the "there is finite data, dump ALL of it" pass — one capture reconstructs the entire scene offline.
void VsmDiagnostics::DumpSceneFull()
{
	using namespace DirectX;
	using V = RE::BSGraphics::Vertex;
	auto& device  = m_core.device;
	auto& context = m_core.context;
	auto dir = logger::log_directory();
	if (!dir || !device || !context)
		return;
	std::ofstream sf(*dir / "VirtualShadowMaps_scene.bin", std::ios::binary | std::ios::trunc);
	if (!sf) {
		logger::warn("VSM scene dump: cannot open scene.bin");
		return;
	}

	ComPtr<ID3D11Buffer> stg;
	UINT stgSz = 0;
	std::vector<std::uint8_t> raw;
	std::unordered_set<const void*> visited;
	std::uint64_t nNodes = 0, nGeom = 0, nBytes = 0;

	auto wr   = [&](const void* p, size_t n) { sf.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n)); nBytes += n; };
	auto wu64 = [&](std::uint64_t v) { wr(&v, 8); };
	auto wu32 = [&](std::uint32_t v) { wr(&v, 4); };
	auto wu16 = [&](std::uint16_t v) { wr(&v, 2); };
	auto wu8  = [&](std::uint8_t v) { wr(&v, 1); };
	auto wf   = [&](float v) { wr(&v, 4); };
	auto wstr = [&](const char* s) { const std::uint16_t n = s ? static_cast<std::uint16_t>(strnlen(s, 4096)) : 0; wu16(n); if (n) wr(s, n); };
	auto wxform = [&](const RE::NiTransform& t) {
		const auto& e = t.rotate.entry;
		float f[13] = { e[0][0], e[0][1], e[0][2], e[1][0], e[1][1], e[1][2], e[2][0], e[2][1], e[2][2],
			t.translate.x, t.translate.y, t.translate.z, t.scale };
		wr(f, sizeof(f));
	};
	// Stage a GPU buffer into `raw`; returns byte count (0 on failure/none).
	auto stage = [&](RE::ID3D11Buffer* rebuf) -> UINT {
		if (!rebuf)
			return 0;
		auto* buf = reinterpret_cast<ID3D11Buffer*>(rebuf);
		D3D11_BUFFER_DESC bd{};
		buf->GetDesc(&bd);
		const UINT n = bd.ByteWidth;
		if (!n)
			return 0;
		if (!stg || stgSz < n) {
			stg.Reset();
			D3D11_BUFFER_DESC sd{};
			sd.ByteWidth = n;
			sd.Usage = D3D11_USAGE_STAGING;
			sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
			if (FAILED(device->CreateBuffer(&sd, nullptr, &stg))) { stgSz = 0; return 0; }
			stgSz = n;
		}
		D3D11_BOX b{ 0, 0, 0, n, 1, 1 };
		context->CopySubresourceRegion(stg.Get(), 0, 0, 0, 0, buf, 0, &b);
		D3D11_MAPPED_SUBRESOURCE m{};
		if (FAILED(context->Map(stg.Get(), 0, D3D11_MAP_READ, 0, &m)))
			return 0;
		raw.assign(static_cast<const std::uint8_t*>(m.pData), static_cast<const std::uint8_t*>(m.pData) + n);
		context->Unmap(stg.Get(), 0);
		return n;
	};

	std::function<void(RE::NiAVObject*, std::uint32_t)> ser = [&](RE::NiAVObject* o, std::uint32_t depth) {
		if (!o || depth > 128)
			return;
		if (!visited.insert(o).second)
			return;
		++nNodes;
		// --- generic node header ---
		wu32(0x444F4E53);  // 'SNOD'
		wu64(reinterpret_cast<std::uint64_t>(o));
		wu32(depth);
		wu32(o->GetFlags().underlying());
		wstr(o->GetRTTI() ? o->GetRTTI()->name : "?");
		wstr(o->name.c_str());
		wxform(o->local);
		wxform(o->world);
		wf(o->worldBound.center.x); wf(o->worldBound.center.y); wf(o->worldBound.center.z); wf(o->worldBound.radius);

		auto* geom = netimmerse_cast<RE::BSGeometry*>(o);
		auto* node = o->AsNode();
		wu8(geom ? 2 : (node ? 1 : 0));
		wu8(geom ? 1 : 0);
		if (geom) {
			++nGeom;
			auto& grd = geom->GetGeometryRuntimeData();
			auto* rd  = grd.rendererData;
			auto& vd  = grd.vertexDesc;
			wu32(vd.GetSize());
			wu32(static_cast<std::uint32_t>(vd.GetFlags()));
			wu32(vd.GetAttributeOffset(V::VA_POSITION));
			wu32(vd.GetAttributeOffset(V::VA_SKINNING));
			wu32(vd.HasFlag(V::VF_FULLPREC) ? 1u : 0u);
			// staged GPU vertex + index buffers
			const UINT vbN = rd ? stage(rd->vertexBuffer) : 0;
			wu32(vbN); if (vbN) wr(raw.data(), vbN);
			const UINT ibN = rd ? stage(rd->indexBuffer) : 0;
			wu32(ibN); if (ibN) wr(raw.data(), ibN);
			const std::uint32_t triCount = geom->AsTriShape() ? geom->AsTriShape()->GetTrishapeRuntimeData().triangleCount : 0u;
			wu32(triCount);
			// dynamicData (morphed model-space positions)
			auto* dts = geom->AsDynamicTriShape();
			const std::uint8_t* dyn = dts ? reinterpret_cast<const std::uint8_t*>(dts->GetDynamicTrishapeRuntimeData().dynamicData) : nullptr;
			const std::uint32_t dynSize = dts ? dts->GetDynamicTrishapeRuntimeData().dataSize : 0u;
			const std::uint32_t dynVerts = (dyn && dynSize >= 16) ? dynSize / 16u : 0u;
			wu8(dts ? 1 : 0);
			wu32(dynVerts);
			if (dyn) wr(dyn, static_cast<size_t>(dynVerts) * 16u);
			// properties (shader + alpha)
			auto* sp = grd.shaderProperty.get();
			wu8(sp ? 1 : 0);
			wu64(sp ? sp->flags.underlying() : 0ull);
			wu8(sp && netimmerse_cast<RE::BSEffectShaderProperty*>(sp) ? 1 : 0);
			wu8(sp && netimmerse_cast<RE::BSLightingShaderProperty*>(sp) ? 1 : 0);
			auto* ap = grd.alphaProperty.get();
			wu8(ap ? 1 : 0);
			wu8(ap && ap->GetAlphaBlending() ? 1 : 0);
			wu8(ap && ap->GetAlphaTesting() ? 1 : 0);
			wu8(ap ? ap->alphaThreshold : 0);
			// FULL skin state
			auto* skin = grd.skinInstance.get();
			wu8(skin && skin->skinPartition && skin->boneMatrices && skin->numMatrices ? 1 : 0);
			if (skin && skin->skinPartition && skin->boneMatrices && skin->numMatrices) {
				const std::uint32_t numMats = skin->numMatrices;
				const std::uint32_t matFloats = (skin->allocatedSize / numMats) / static_cast<std::uint32_t>(sizeof(float));
				wu32(numMats); wu32(matFloats);
				wr(skin->boneMatrices, static_cast<size_t>(numMats) * matFloats * sizeof(float));  // GPU palette
				// per-bone WORLD transforms (the actual pose)
				wu8(skin->boneWorldTransforms ? 1 : 0);
				if (skin->boneWorldTransforms)
					for (std::uint32_t b = 0; b < numMats; ++b) {
						const RE::NiTransform* bt = skin->boneWorldTransforms[b];
						wu8(bt ? 1 : 0);
						if (bt) wxform(*bt);
					}
				// NiSkinData inverse-binds (skinToBone) + bounds
				auto* sd = skin->skinData.get();
				wu8(sd ? 1 : 0);
				if (sd) {
					wu32(sd->bones);
					wxform(sd->rootParentToSkin);
					if (sd->boneData)
						for (std::uint32_t b = 0; b < sd->bones; ++b) {
							wxform(sd->boneData[b].skinToBone);
							wf(sd->boneData[b].bound.center.x); wf(sd->boneData[b].bound.center.y);
							wf(sd->boneData[b].bound.center.z); wf(sd->boneData[b].bound.radius);
						}
				}
				// partitions — every array
				auto* spt = skin->skinPartition.get();
				wu32(spt->numPartitions);
				for (std::uint32_t pi = 0; pi < spt->numPartitions; ++pi) {
					auto& p = spt->partitions[pi];
					auto& pvd = p.vertexDesc;
					const std::uint32_t pstride = pvd.GetSize();
					const std::uint32_t pskin = pvd.GetAttributeOffset(V::VA_SKINNING);
					const std::uint32_t ppos  = pvd.GetAttributeOffset(V::VA_POSITION);
					wu16(p.vertices); wu16(p.triangles); wu16(p.numBones); wu16(p.strips); wu16(p.bonesPerVertex);
					wu32(pstride); wu32(pskin); wu32(ppos); wu32(static_cast<std::uint32_t>(pvd.GetFlags()));
					for (std::uint16_t b = 0; b < p.numBones; ++b) wu16(p.bones ? p.bones[b] : 0xFFFFu);
					for (std::uint16_t lv = 0; lv < p.vertices; ++lv) wu16(p.vertexMap ? p.vertexMap[lv] : 0xFFFFu);
					for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(p.triangles) * 3u; ++k) wu16(p.triList ? p.triList[k] : 0u);
					const std::uint32_t bpv = p.bonesPerVertex;
					// canonical partition weights (float) + bone palette (byte), vertices*bonesPerVertex each
					wu8(p.weights ? 1 : 0);
					if (p.weights) wr(p.weights, static_cast<size_t>(p.vertices) * bpv * sizeof(float));
					wu8(p.bonePalette ? 1 : 0);
					if (p.bonePalette) wr(p.bonePalette, static_cast<size_t>(p.vertices) * bpv);
					// per-vertex embedded skinning (from the partition vertex buffer) + position
					const std::uint8_t* rv = p.buffData ? reinterpret_cast<const std::uint8_t*>(p.buffData->rawVertexData) : nullptr;
					wu8(rv ? 1 : 0);
					if (rv)
						for (std::uint16_t lv = 0; lv < p.vertices; ++lv) {
							std::uint16_t wh[4]{}; std::uint8_t slot[4]{}; float bp[3]{};
							const std::uint8_t* vb = rv + static_cast<size_t>(lv) * pstride;
							if (pskin + 12 <= pstride) { std::memcpy(wh, vb + pskin, 8); std::memcpy(slot, vb + pskin + 8, 4); }
							if (ppos + 12 <= pstride) std::memcpy(bp, vb + ppos, 12);
							wr(wh, 8); wr(slot, 4); wr(bp, 12);
						}
				}
			}
		}
		// --- children (pre-order) ---
		if (node) {
			auto& ch = node->GetChildren();
			wu32(static_cast<std::uint32_t>(ch.size()));
			for (auto& c : ch)
				ser(c.get(), depth + 1);
		} else {
			wu32(0);
		}
	};

	auto& sm = RE::BSShaderManager::State::GetSingleton();
	for (int i = 0; i < 4; ++i)
		if (sm.shadowSceneNode[i])
			ser(reinterpret_cast<RE::NiAVObject*>(sm.shadowSceneNode[i]), 0);
	if (auto* pc = RE::PlayerCharacter::GetSingleton()) {
		if (auto* p3 = pc->Get3D())
			ser(p3, 0);
		if (auto* cell = pc->GetParentCell())
			cell->ForEachReference([&](RE::TESObjectREFR* r) {
				if (r)
					if (auto* r3 = r->Get3D())
						ser(r3, 0);
				return RE::BSContainer::ForEachResult::kContinue;
			});
	}
	logger::info("=== FULL SCENE DUMP -> VirtualShadowMaps_scene.bin: nodes={} geometries={} bytes={} ({:.1f} MB) ===",
	    nNodes, nGeom, nBytes, static_cast<double>(nBytes) / (1024.0 * 1024.0));
}

// Full identity + relationship of a TESObjectREFR: base object (formID/editorID/type), the mesh/NIF path,
// its linked ref + enable-state parent (how a SEPARATE ref links to the light), and its parent cell. This is
// the complete object record so the log alone answers "what is this object, and is it the light's own housing".
std::string VsmDiagnostics::RefIdentity(RE::FormID a_refID)
{
	if (!a_refID) return "ref=0";
	auto* ref = RE::TESForm::LookupByID<RE::TESObjectREFR>(a_refID);
	if (!ref) return std::format("ref=0x{:08X}(not-found)", a_refID);
	auto* base = ref->GetBaseObject();
	const RE::FormID baseID   = base ? base->GetFormID() : 0;
	const char*      baseEID  = (base && base->GetFormEditorID()) ? base->GetFormEditorID() : "";
	const int        baseType = base ? static_cast<int>(base->GetFormType()) : -1;
	const char*      model    = "";
	if (auto* m = skyrim_cast<RE::TESModel*>(base)) { if (m->GetModel()) model = m->GetModel(); }
	RE::FormID linked = 0;
	if (auto* lr = ref->GetLinkedRef(nullptr)) linked = lr->GetFormID();
	RE::FormID enableParent = 0;
	if (auto xp = ref->extraList.GetByType<RE::ExtraEnableStateParent>()) {
		if (auto p = xp->parent.get()) enableParent = p->GetFormID();
	}
	const RE::FormID cellID = ref->GetParentCell() ? ref->GetParentCell()->GetFormID() : 0;
	const RE::NiPoint3 rp = ref->GetPosition();
	return std::format("ref=0x{:08X} base=0x{:08X}('{}' type={}) model='{}' linkedRef=0x{:08X} enableParent=0x{:08X} cell=0x{:08X} refPos=({:.0f} {:.0f} {:.0f})",
	    a_refID, baseID, baseEID, baseType, model, linked, enableParent, cellID, rp.x, rp.y, rp.z);
}

// Full render + projection detail for ONE caster (the flashed/isolated one). Answers "where does this
// mesh actually get rasterized into the atlas?" — its transform, its local->world verts, their world
// AABB, and where each vert projects in the nearest light's cube (face / ndc / atlas UV / in-bounds).
// If the world verts don't sit at the mesh's bound, the transform/vertex read is wrong; if they sit
// right but project to the wrong atlas spot, the cube math is wrong. Maximal on purpose (log-first).
void VsmDiagnostics::DumpFocusedCaster(int a_idx)
{
	// core render-state, reached through the owning VirtualShadowMaps
	auto& registry     = m_core.registry;
	auto& device       = m_core.device;
	auto& context      = m_core.context;
	auto& lightRecords = m_core.lightRecords;
	auto& rtAtlasW     = m_core.rtAtlasW;
	auto& rtAtlasH     = m_core.rtAtlasH;

	if (a_idx < 0 || a_idx >= static_cast<int>(registry.size()) || !device || !context)
		return;
	RE::BSGeometry* g = registry[a_idx].geom.get();
	if (!g)
		return;

	const auto& tr = g->world;
	const auto& c  = g->worldBound;
	logger::info("=== FOCUSED CASTER #{} '{}' type={} engineOnly={} ===", a_idx, g->name.c_str(),
	    g->GetRTTI() ? g->GetRTTI()->name : "?", registry[a_idx].engineOnly ? 1 : 0);
	logger::info("  world.translate=({:.2f} {:.2f} {:.2f}) scale={:.4f}  bound.center=({:.2f} {:.2f} {:.2f}) r={:.2f}",
	    tr.translate.x, tr.translate.y, tr.translate.z, tr.scale, c.center.x, c.center.y, c.center.z, c.radius);
	logger::info("  world.rotate = [{:.4f} {:.4f} {:.4f}][{:.4f} {:.4f} {:.4f}][{:.4f} {:.4f} {:.4f}]",
	    tr.rotate.entry[0][0], tr.rotate.entry[0][1], tr.rotate.entry[0][2],
	    tr.rotate.entry[1][0], tr.rotate.entry[1][1], tr.rotate.entry[1][2],
	    tr.rotate.entry[2][0], tr.rotate.entry[2][1], tr.rotate.entry[2][2]);

	// Material: is this something that SHOULD cast a solid shadow, or an effect/decal plane we ought to
	// have filtered? Shader RTTI type + material feature + alpha flags decide it. A decal/effect/refract
	// plane casting a solid shadow = "shadow with no visible source in the wrong place".
	{
		auto& grd = g->GetGeometryRuntimeData();
		auto* sp  = grd.shaderProperty.get();
		auto* ap  = grd.alphaProperty.get();
		const char* spType = (sp && sp->GetRTTI()) ? sp->GetRTTI()->name : "none";
		const int   matType = sp ? static_cast<int>(sp->GetMaterialType()) : -1;
		const bool  isEffect = sp && netimmerse_cast<RE::BSEffectShaderProperty*>(sp) != nullptr;
		logger::info("  MATERIAL: shader='{}' effectShader={} matType={}  alpha[prop={} blend={} test={}]  (effectShader/blend => should NOT cast solid; decal/effect plane is the classic bad caster)",
		    spType, isEffect ? 1 : 0, matType, ap ? 1 : 0, (ap && ap->GetAlphaBlending()) ? 1 : 0, (ap && ap->GetAlphaTesting()) ? 1 : 0);
	}

	auto* rd = g->GetGeometryRuntimeData().rendererData;
	const UINT stride = registry[a_idx].vertexStride;
	if (!rd || !rd->vertexBuffer || !stride) {
		logger::info("  (no renderer vertex buffer / skinned — world verts not read here; see skinned probe)");
		return;
	}
	// read the head of the vertex buffer
	std::vector<uint8_t> vb;
	{
		auto* src = reinterpret_cast<ID3D11Buffer*>(rd->vertexBuffer);
		D3D11_BUFFER_DESC bd{}; src->GetDesc(&bd);
		const UINT want = (std::min)(stride * static_cast<UINT>(kFocusVertsRead), bd.ByteWidth);
		D3D11_BUFFER_DESC sd{}; sd.ByteWidth = want; sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		ComPtr<ID3D11Buffer> st;
		if (want && SUCCEEDED(device->CreateBuffer(&sd, nullptr, &st))) {
			D3D11_BOX box{ 0, 0, 0, want, 1, 1 };
			context->CopySubresourceRegion(st.Get(), 0, 0, 0, 0, src, 0, &box);
			D3D11_MAPPED_SUBRESOURCE m{};
			if (SUCCEEDED(context->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) {
				vb.assign(static_cast<const uint8_t*>(m.pData), static_cast<const uint8_t*>(m.pData) + want);
				context->Unmap(st.Get(), 0);
			}
		}
	}
	if (vb.empty()) {
		logger::info("  (vertex readback failed)");
		return;
	}
	const XMMATRIX world = NiTransformToXM(g->world);  // same matrix RenderDepth rasterizes with
	const int nv = (std::min)(kFocusVertsLog, static_cast<int>(vb.size() / stride));

	// nearest collected light to this caster (the one whose cube it most likely lands in)
	int li = -1; float lbest = FLT_MAX;
	for (size_t k = 0; k < lightRecords.size(); ++k) {
		const auto& L = lightRecords[k];
		const float d = std::sqrt((L.positionWS.x - c.center.x) * (L.positionWS.x - c.center.x) +
		                          (L.positionWS.y - c.center.y) * (L.positionWS.y - c.center.y) +
		                          (L.positionWS.z - c.center.z) * (L.positionWS.z - c.center.z));
		if (d < lbest) { lbest = d; li = static_cast<int>(k); }
	}
	logger::info("  nearest light L{} at ({:.1f} {:.1f} {:.1f}) dist={:.1f} far={:.1f}  ({} verts of {} tri):",
	    li, li >= 0 ? lightRecords[li].positionWS.x : 0, li >= 0 ? lightRecords[li].positionWS.y : 0,
	    li >= 0 ? lightRecords[li].positionWS.z : 0, lbest, li >= 0 ? lightRecords[li].farPlane : 0, nv, registry[a_idx].indexCount / 3u);

	XMFLOAT3 mn{ FLT_MAX, FLT_MAX, FLT_MAX }, mx{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
	for (int v = 0; v < nv; ++v) {
		const float* f = reinterpret_cast<const float*>(vb.data() + static_cast<size_t>(v) * stride);
		XMFLOAT3 wp; XMStoreFloat3(&wp, XMVector3TransformCoord(XMVectorSet(f[0], f[1], f[2], 1.0f), world));
		mn.x = (std::min)(mn.x, wp.x); mn.y = (std::min)(mn.y, wp.y); mn.z = (std::min)(mn.z, wp.z);
		mx.x = (std::max)(mx.x, wp.x); mx.y = (std::max)(mx.y, wp.y); mx.z = (std::max)(mx.z, wp.z);
		if (li >= 0) {
			const auto& L = lightRecords[li];
			const int face = faceFromDir(wp.x - L.positionWS.x, wp.y - L.positionWS.y, wp.z - L.positionWS.z);
			XMFLOAT4 clip; XMStoreFloat4(&clip, XMVector4Transform(XMVectorSet(wp.x, wp.y, wp.z, 1.0f), XMLoadFloat4x4(&L.cubeVP[face])));
			const bool front = clip.w > kClipWFrontEps;
			const float nx = front ? clip.x / clip.w : kNdcSentinel, ny = front ? clip.y / clip.w : kNdcSentinel, nz = front ? clip.z / clip.w : kNdcSentinel;
			const bool inb = (nx >= -1 && nx <= 1 && ny >= -1 && ny <= 1 && nz >= 0 && nz <= 1);
			const float au = (L.atlasX + ((face % 3) + (nx * 0.5f + 0.5f)) * L.positionWS.w) / static_cast<float>(rtAtlasW);
			const float av = (L.atlasY + ((face / 3) + (ny * -0.5f + 0.5f)) * L.positionWS.w) / static_cast<float>(rtAtlasH);
			logger::info("    v{}: local=({:.1f} {:.1f} {:.1f}) -> world=({:.1f} {:.1f} {:.1f})  face={} ndc=({:.3f} {:.3f} {:.4f}) atlasUV=({:.4f} {:.4f}) front={} inB={}",
			    v, f[0], f[1], f[2], wp.x, wp.y, wp.z, kFaceName[face], nx, ny, nz, au, av, front ? 1 : 0, inb ? 1 : 0);
		} else {
			logger::info("    v{}: local=({:.1f} {:.1f} {:.1f}) -> world=({:.1f} {:.1f} {:.1f})", v, f[0], f[1], f[2], wp.x, wp.y, wp.z);
		}
	}
	logger::info("  world-vert AABB=[{:.1f} {:.1f} {:.1f}]..[{:.1f} {:.1f} {:.1f}]  (should straddle bound.center; if far off => transform/vertex read is wrong)",
	    mn.x, mn.y, mn.z, mx.x, mx.y, mx.z);
}

void VsmDiagnostics::DumpDiagnosticLog()
{
	// core render-state, reached through the owning VirtualShadowMaps. Aliased here so the (unchanged)
	// dump body reads the god-object's members with byte-identical behaviour. ComputeVertAABB stays a
	// VirtualShadowMaps member (render-path helper) and is called below as m_core.ComputeVertAABB(...).
	auto& device                 = m_core.device;
	auto& context                = m_core.context;
	auto& enabled                = m_core.enabled;
	auto& dbgMode                = m_core.dbgMode;
	auto& dbgSampleSpace         = m_core.dbgSampleSpace;
	auto& dbgCompareMode         = m_core.dbgCompareMode;
	auto& dbgMatchEye            = m_core.dbgMatchEye;
	auto& registry               = m_core.registry;
	auto& flashCaster            = m_core.flashCaster;
	auto& isolateCaster          = m_core.isolateCaster;
	auto& lightRecords           = m_core.lightRecords;
	auto& sceneCameraPos         = m_core.sceneCameraPos;
	auto& rtAtlasW               = m_core.rtAtlasW;
	auto& rtAtlasH               = m_core.rtAtlasH;
	auto& rtFaceRes              = m_core.rtFaceRes;
	auto& rtBlockW               = m_core.rtBlockW;
	auto& rtBlockH               = m_core.rtBlockH;
	auto& visibleCasters         = m_core.visibleCasters;
	auto& lightsAmbient          = m_core.lightsAmbient;
	auto& lightsNonShadow        = m_core.lightsNonShadow;
	auto& lightsCulled           = m_core.lightsCulled;
	auto& lightsViewerAttached   = m_core.lightsViewerAttached;
	auto& lightsHidden           = m_core.lightsHidden;
	auto& dumpOrdinal            = m_core.dumpOrdinal;
	auto& frameIndex             = m_core.frameIndex;
	auto& resourceFetchCount     = m_core.resourceFetchCount;
	auto& lastResourceFetchFrame = m_core.lastResourceFetchFrame;
	auto& rejectedNoRenderData   = m_core.rejectedNoRenderData;
	auto& rejectedNoVertexBuffer = m_core.rejectedNoVertexBuffer;
	auto& rejectedNoIndexBuffer  = m_core.rejectedNoIndexBuffer;
	auto& rejectedZeroTriangles  = m_core.rejectedZeroTriangles;
	auto& rejectedDuplicate      = m_core.rejectedDuplicate;
	auto& rejectedNoCast         = m_core.rejectedNoCast;
	auto& rejectedDecal          = m_core.rejectedDecal;
	auto& rejectedHidden         = m_core.rejectedHidden;
	auto& rejectedTransparent    = m_core.rejectedTransparent;
	auto& rejectedBillboard      = m_core.rejectedBillboard;
	auto& rejectedUserNoCast     = m_core.rejectedUserNoCast;
	auto& userForcedCast         = m_core.userForcedCast;
	auto& rejectedNotVisible     = m_core.rejectedNotVisible;
	auto& rejectedEngineFlag     = m_core.rejectedEngineFlag;
	auto& depthTex               = m_core.depthTex;
	auto& idTex                  = m_core.idTex;
	auto& lightDynamic           = m_core.lightDynamic;
	auto& lightMove              = m_core.lightMove;
	auto& lightNames             = m_core.lightNames;
	auto& lightOwnerRef          = m_core.lightOwnerRef;
	auto& lightNodeRef           = m_core.lightNodeRef;
	auto& lightMeta              = m_core.lightMeta;
	auto& cameraMovedThisFrame   = m_core.cameraMovedThisFrame;
	auto& castersRidingCam       = m_core.castersRidingCam;
	auto& casterRideMaxDelta     = m_core.casterRideMaxDelta;
	auto& casterRideName         = m_core.casterRideName;
	auto& screenW                = m_core.screenW;
	auto& screenH                = m_core.screenH;
	auto& probeArmed             = m_core.probeArmed;
	auto& pixelProbeBuf          = m_core.pixelProbeBuf;
	auto& pixelProbeStaging      = m_core.pixelProbeStaging;

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

	// CPU mirror of the shader linearizer (MUST match VSM.hlsli exactly); faceFromDir/kFaceName are at file scope.
	auto lin = [](float d, float n, float f) { return n * f / (std::max)(n + d * (f - n), kLinDenomFloor); };  // REVERSE-Z (matches kReverseZ): d=1 near, d=0 far

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
	{
		auto casterName = [&](int idx) -> const char* {
			if (idx < 0 || idx >= static_cast<int>(registry.size()) || !registry[idx].geom.get())
				return "-";
			const char* nm = registry[idx].geom.get()->name.c_str();
			return nm ? nm : "?";
		};
		logger::info("SELECTORS: flashCaster={} '{}'{}  isolateCaster={} '{}'  (flash blinks a caster's shadow; -1/0 = off)",
		    flashCaster, casterName(flashCaster), (flashCaster >= 0 && flashCaster < static_cast<int>(registry.size()) && registry[flashCaster].engineOnly) ? " [skinned]" : "",
		    isolateCaster, isolateCaster > 0 ? casterName(isolateCaster - 1) : "all");
	}
	// Full render+projection detail for whichever caster is flashed/isolated (log-first: get it all at once).
	if (flashCaster >= 0)
		DumpFocusedCaster(flashCaster);
	if (isolateCaster > 0 && isolateCaster - 1 != flashCaster)
		DumpFocusedCaster(isolateCaster - 1);
	logger::info("near = far*{:.3f} (calc, floored {:.2f}); bias = calculated receiver-plane (in shader). sampleSpace={} compareMode={} matchEye={}",
	    vsm::kNearPlaneFraction, vsm::kNearPlaneEpsilon, dbgSampleSpace, dbgCompareMode, dbgMatchEye);
	logger::info("fixture cull: REMOVED. Solid geometry (fire pit / fuel / lamp) casts naturally; billboards & translucent effects are excluded at registry build (see BILLBOARD rejection above). No spatial 'light-inside-object' heuristic.");
	// caster space: casters rasterize in ABSOLUTE world (geom->world game-absolute; skinned posed to
	// absolute; NO cameraPos offset — the old dbgCastersAbsolute toggle was removed). So geom.bin world
	// verts are already absolute; the offline GT uses them directly. sceneCameraPos logged for reference.
	logger::info("SPACE: casters=ABSOLUTE (no cameraPos offset) sceneCameraPos=({:.1f} {:.1f} {:.1f})",
	    sceneCameraPos.x, sceneCameraPos.y, sceneCameraPos.z);
	// The s7 shadow sampler is bound by LLF, not us. Point vs bilinear changes the sampled edge texel.
	{
		ComPtr<ID3D11SamplerState> s7; context->PSGetSamplers(7, 1, &s7);
		if (s7) {
			D3D11_SAMPLER_DESC sdsc{}; s7->GetDesc(&sdsc);
			logger::info("SAMPLER s7 (shadow atlas): filter={} addressU={} addressV={} compFunc={}  (our replay point-samples; if filter!=POINT the live edge is softened)",
			    static_cast<int>(sdsc.Filter), static_cast<int>(sdsc.AddressU), static_cast<int>(sdsc.AddressV), static_cast<int>(sdsc.ComparisonFunc));
		} else {
			logger::info("SAMPLER s7: none bound at dump time (LLF binds it during lighting; may be unbound at Present)");
		}
	}
	logger::info("atlas {}x{}  rtFaceRes={} rtBlockW={} rtBlockH={}  lights={} registry={} visibleCasters={}",
	    rtAtlasW, rtAtlasH, rtFaceRes, rtBlockW, rtBlockH, lightRecords.size(), registry.size(), visibleCasters);
	logger::info("light filter: shadow-casters(added)={} ambient(skipped)={} nonShadow(skipped)={} viewerAttached(skipped)={} hidden(skipped)={} culled-P3(behind/far)={}  (only IsShadowLight & !ambient & !viewer-attached & !hidden cast)",
	    lightRecords.size(), lightsAmbient, lightsNonShadow, lightsViewerAttached, lightsHidden, lightsCulled);
	logger::info("dump #{}  frame={}  CS resource-fetch: count={} lastFrame={} (frame-lastFrame={} -> 0/1 = CS binding our atlas NOW; large/stale = CS<->plugin handshake broken)",
	    ++dumpOrdinal, frameIndex, resourceFetchCount, lastResourceFetchFrame, frameIndex - lastResourceFetchFrame);
	logger::info("atlas cfg: reverseZ={} (near->1, far->0, empty=0) cubeFOV=guard-band rasterDepthBias=0 slopeScaledBias=0 (shadow bias is computed in-shader on LINEAR distances)", kReverseZ);
	if (auto* ssn = smState.shadowSceneNode[0]) {
		auto& rt = ssn->GetRuntimeData();
		logger::info("light-set provenance: engine activeLights={} activeShadowLights={} -> we collected={} (NO cap; light-buffer capacity={}); a large drop = the !IsShadowLight()/ambient/viewer-attached/dedup/origin-skip filters, NOT a cap",
		    rt.activeLights.size(), rt.activeShadowLights.size(), lightRecords.size(), m_core.lightBufCapacity);
	}

	if (lightRecords.empty() || registry.empty()) {
		logger::info("(no lights or no casters this frame — nothing to prove)");
		logger::info("=========================================");
		return;
	}

	// ---- Registry health: valid buffers, triangle-count spread, absolute-space caster AABB. ----
	{
		int valid = 0, nStatic = 0; uint32_t triMin = 0xFFFFFFFFu, triMax = 0; uint64_t triSum = 0;
		XMFLOAT3 mn{ FLT_MAX, FLT_MAX, FLT_MAX }, mx{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
		for (const auto& e : registry) {
			RE::BSGeometry* g = e.geom.get();
			if (!g) continue;
			auto* grd = g->GetGeometryRuntimeData().rendererData;
			if (grd && grd->vertexBuffer && grd->indexBuffer) ++valid;
			if (e.isStatic) ++nStatic;
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
		logger::info("caster class (R5): STATIC(cacheable)={} DYNAMIC(re-render)={}  |  atlas: {} stable light slots (P1 persistent allocation)",
		    nStatic, static_cast<int>(n) - nStatic, m_core.lightSlots.size());
	}

	logger::info("capture rejections (BSTriShapes reached but NOT registered): noRendererData={} noVertexBuffer={} noIndexBuffer={} zeroTris={} duplicate={}",
	    rejectedNoRenderData, rejectedNoVertexBuffer, rejectedNoIndexBuffer, rejectedZeroTriangles, rejectedDuplicate);
	logger::info("skipped as NOT-DRAWN-by-game: kCastShadows-OFF(engine flag: fire/light planes, glow)={} decal(kDecal|kDynamicDecal)={}  (decals=flat projected textures, never occluders)",
	    rejectedNoCast, rejectedDecal);
	logger::info("skipped as NOT-DRAWN-by-game: hidden/appCulled(kHidden)={} transparent/effect(alphaBlend|effectShader)={}  (hidden=editor markers etc.=shadows-from-nothing; transparent=blood/glass/glow/FX, often skinned=the moving invisible casters)",
	    rejectedHidden, rejectedTransparent);
	logger::info("skipped as BILLBOARD (under NiBillboardNode: camera-facing smoke/vapor/fire/glow planes)={}  (their hard shadow would ride the camera + they are translucent effects; the moving 'Plane03' vapor-smoke bar)",
	    rejectedBillboard);
	logger::info("config [classification] pattern override: forceNoCast-rejected={} forceCast-forced={}  (0/0 unless the user set forceCast/forceNoCast in the .toml; forceCast beats every built-in exclusion)",
	    rejectedUserNoCast, userForcedCast);
	logger::info("skipped as ENGINE-doesn't-cast (research flags): notVisible(NiAVObject kNotVisible)={} engineFlag(nonProjective|billboard|wireframe|LOD|water)={}  (mirror what vanilla omits from the shadow pass; notVisible = prime suspect for invisible-mesh-casting-on-characters)",
	    rejectedNotVisible, rejectedEngineFlag);

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
			int best = -1; float bd = kBigDistance;
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
			    m, d, d < vsm::kDebugModeMatchThresh ? "MATCH" : "NO-MATCH");
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

		// DirectionalAmbient: PerGeometry (b2) packoffset c11 (byte 176), row_major float3x4. Ambient
		// at a surface with normal N = mul(M, float4(N,1)). Lets offline visibility decide if a computed
		// shadow is washed out by ambient (cosmetic; NOT needed to compute the shadow itself).
		ID3D11Buffer* b2 = nullptr;
		context->PSGetConstantBuffers(2, 1, &b2);
		if (b2) {
			std::vector<uint8_t> ab;
			if (readBufBytes(b2, 176 + 48, ab) && ab.size() >= 176 + 48) {
				const float* a = reinterpret_cast<const float*>(ab.data() + 176);
				logger::info("DirectionalAmbient (b2 c11 float3x4): [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f} {:.3f}]  (ambient@N = M*float4(N,1))",
				    a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11]);
			} else {
				logger::info("DirectionalAmbient: b2 read failed/too small");
			}
			b2->Release();
		} else {
			logger::info("DirectionalAmbient: no cbuffer bound at b2 at dump time");
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
	logger::info("--- registry casters ({} total; abs center / radius / tris / stride / MATERIAL(shader cast/blend/test) / type / name) ---", registry.size());
	for (size_t i = 0; i < registry.size(); ++i) {
		RE::BSGeometry* g = registry[i].geom.get();
		if (!g) continue;
		const auto& c = g->worldBound.center;
		const auto& wt = g->world.translate;  // the ACTUAL render-transform origin (what places the occluder)
		// MATERIAL: the engine's own flags for this mesh, so a spurious caster is identifiable straight
		// from the log (no isolate step). cast = kCastShadows (should be 1 for every registered caster
		// now we filter on it); blend/test = alpha props; shader = the BSShaderProperty RTTI.
		using SPF = RE::BSShaderProperty::EShaderPropertyFlag;
		auto grt = g->GetGeometryRuntimeData();
		auto* sp = grt.shaderProperty.get();
		auto* ap = grt.alphaProperty.get();
		// Log the ENTIRE shader flag word (all 64 bits = SLSF1 low + SLSF2 high) as hex, so ANY flag is
		// recoverable from the log without another dump. Decode the shadow-relevant ones inline.
		const std::uint64_t flg = sp ? sp->flags.underlying() : 0ull;
		const int  cast  = sp ? (sp->flags.any(SPF::kCastShadows) ? 1 : 0) : -1;
		const int  decal = sp ? ((sp->flags.any(SPF::kDecal) || sp->flags.any(SPF::kDynamicDecal)) ? 1 : 0) : 0;
		const int  twoSd = sp ? (sp->flags.any(SPF::kTwoSided) ? 1 : 0) : 0;
		const int  blend = ap ? (ap->GetAlphaBlending() ? 1 : 0) : 0;
		const int  test  = ap ? (ap->GetAlphaTesting() ? 1 : 0) : 0;
		const char* shdr = sp && sp->GetRTTI() ? sp->GetRTTI()->name : "-";
		// Cheap intrinsic fields: vertexCount (a flat quad is ~4), world scale (a unit quad scaled up = a
		// bar), scene-graph kShadowCaster (a 2nd authoritative cast signal), material Feature (glow/cloud/
		// eye/tree = non-solid). All decode the "is this really solid geometry" question.
		auto* tri = g->AsTriShape();
		const int   vcnt  = tri ? static_cast<int>(tri->GetTrishapeRuntimeData().vertexCount) : -1;
		const float wscale = g->world.scale;
		const int   sceneCast = g->GetFlags().any(RE::NiAVObject::Flag::kShadowCaster) ? 1 : 0;
		const int   feat = (sp && sp->GetBaseMaterial()) ? static_cast<int>(sp->GetBaseMaterial()->GetFeature()) : -99;
		// Owning-ref identity (base editorID) so the log names the OBJECT each caster belongs to — that is how
		// we prove the fire-pit / flame plane share the hearth light's ref (the ref cull's target). Dump-only lookup.
		const char* refBase = "";
		if (auto* rr = RE::TESForm::LookupByID<RE::TESObjectREFR>(registry[i].ownerRef))
			if (auto* base = rr->GetBaseObject())
				if (const char* e = base->GetFormEditorID()) refBase = e;
		logger::info("  c{:03d} abs={:8.0f} {:8.0f} {:8.0f} r={:6.0f} worldT={:8.0f} {:8.0f} {:8.0f} tris={:5} vcount={:5} stride={} scale={:.3f} cast={} sceneCast={} decal={} twoSided={} blend={} test={} feat={} ref=0x{:08X}({}) flags=0x{:016X} shdr={} type={} '{}'",
		    static_cast<int>(i), c.x, c.y, c.z, g->worldBound.radius, wt.x, wt.y, wt.z,
		    registry[i].indexCount / 3u, vcnt, registry[i].vertexStride, wscale, cast, sceneCast, decal, twoSd, blend, test, feat, registry[i].ownerRef, refBase, flg, shdr,
		    g->GetRTTI() ? g->GetRTTI()->name : "?", g->name.c_str());
		// FULL object identity + relationship + material extras (the complete record, so the log alone answers
		// "what object is this, is it the light's own housing, does it glow"): owning-ref chain (base/type/
		// model/linkedRef/enableParent/cell), alpha-test threshold, emissive glow, distance to nearest light.
		{
			const int   alphaTh = ap ? static_cast<int>(ap->alphaThreshold) : -1;
			float       emitMax = -1.0f;
			if (auto* lsp = skyrim_cast<RE::BSLightingShaderProperty*>(sp)) {
				float cmax = 0.0f;
				if (lsp->emissiveColor) cmax = (std::max)({ lsp->emissiveColor->red, lsp->emissiveColor->green, lsp->emissiveColor->blue });
				emitMax = lsp->emissiveMult * cmax;
			}
			float dNear = 1e9f;
			for (const auto& Lr : lightRecords) {
				const float dx = c.x - Lr.positionWS.x, dy = c.y - Lr.positionWS.y, dz = c.z - Lr.positionWS.z;
				dNear = (std::min)(dNear, std::sqrt(dx * dx + dy * dy + dz * dz));
			}
			logger::info("       c{:03d} {}  alphaThresh={} emissive={:.2f} distNearestLight={:.0f}",
			    static_cast<int>(i), RefIdentity(registry[i].ownerRef), alphaTh, emitMax, dNear);
		}
		// Real geometry: local + world vertex AABB (via readback). Local extents reveal a FLAT plane (one
		// ~0 axis) that a sphere hides; the world AABB is a tight box occluder for the true-shadow calc.
		DirectX::XMFLOAT3 lmn, lmx, wmn, wmx;
		if (m_core.ComputeVertAABB(g, registry[i].vertexStride, lmn, lmx, wmn, wmx)) {
			logger::info("       c{:03d} localAABB=[{:.1f} {:.1f} {:.1f}]..[{:.1f} {:.1f} {:.1f}] worldAABB=[{:.1f} {:.1f} {:.1f}]..[{:.1f} {:.1f} {:.1f}]",
			    static_cast<int>(i), lmn.x, lmn.y, lmn.z, lmx.x, lmx.y, lmx.z, wmn.x, wmn.y, wmn.z, wmx.x, wmx.y, wmx.z);
		}
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
		logger::warn("VSM dump: atlas readback FAILED — occluder values below are placeholder 0.0 (reverse-Z empty/far)");

	// ============ FULL ATLAS DUMP (the complete occluder field — replaces per-point probing) ============
	// The whole depth atlas IS every occluder for every light. Persist it (+ the id atlas) as raw binary
	// sidecars next to the log; offline, replaying GetLocalShadow against this atlas yields EVERY shadow
	// for ANY receiver at any resolution — no sampling. (12.6M floats can't live in text; binary it is.)
	if (haveAtlas) {
		if (auto dir = logger::log_directory()) {
			const auto apath = *dir / "VirtualShadowMaps_atlas.bin";
			std::ofstream af(apath, std::ios::binary | std::ios::trunc);
			if (af) af.write(reinterpret_cast<const char*>(atlas.data()), static_cast<std::streamsize>(atlas.size() * sizeof(float)));
			std::string idInfo = "id=none(not-populated)";
			if (idTex) {
				D3D11_TEXTURE2D_DESC itd{}; idTex->GetDesc(&itd);
				D3D11_TEXTURE2D_DESC isd = itd; isd.Usage = D3D11_USAGE_STAGING; isd.BindFlags = 0; isd.CPUAccessFlags = D3D11_CPU_ACCESS_READ; isd.MiscFlags = 0;
				ComPtr<ID3D11Texture2D> idStg;
				if (SUCCEEDED(device->CreateTexture2D(&isd, nullptr, &idStg))) {
					context->CopyResource(idStg.Get(), idTex.Get());
					D3D11_MAPPED_SUBRESOURCE im{};
					if (SUCCEEDED(context->Map(idStg.Get(), 0, D3D11_MAP_READ, 0, &im))) {
						const UINT idRowUints = im.RowPitch / static_cast<UINT>(sizeof(uint32_t));
						std::ofstream idf(*dir / "VirtualShadowMaps_id.bin", std::ios::binary | std::ios::trunc);
						if (idf) idf.write(reinterpret_cast<const char*>(im.pData), static_cast<std::streamsize>(static_cast<size_t>(im.RowPitch) * itd.Height));
						context->Unmap(idStg.Get(), 0);
						idInfo = std::format("id=VirtualShadowMaps_id.bin(R32_UINT,registryIndex+1;0=empty) idRowUints={}", idRowUints);
					}
				}
			}
			logger::info("=== FULL ATLAS DUMP written to: {} ===", dir->string());
			logger::info("=== FULL ATLAS DUMP: depth=VirtualShadowMaps_atlas.bin (R32 float, REVERSE-Z: near=1.0 far=0.0 empty=0.0) width={} height={} rowFloats={}  {}  blockW={} blockH={} faceRes={}  (offline: for any receiver W, match light -> face=argmax|W-Lpos| -> project via that light's cubeVP -> atlasUV -> sample this atlas -> linearize(near,far) -> compare(bias) = EVERY shadow, no probe) ===",
			    rtAtlasW, rtAtlasH, atlasRowFloats, idInfo, rtBlockW, rtBlockH, rtFaceRes);
		}
	}
	auto atlasAtPx = [&](int x, int y) -> float {
		if (!haveAtlas) return 0.0f;  // reverse-Z: no atlas => empty/far = 0.0 (not an occluder)
		x = std::clamp(x, 0, rtAtlasW - 1);
		y = std::clamp(y, 0, rtAtlasH - 1);
		return atlas[static_cast<size_t>(y) * atlasRowFloats + x];
	};
	auto atlasAtUV = [&](float u, float v) -> float {
		return atlasAtPx(static_cast<int>(std::floor(u * rtAtlasW)), static_cast<int>(std::floor(v * rtAtlasH)));
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
	auto roundTrip = [&](const VirtualShadowMaps::LightRecord& Lr, int f, float& diff, int& reFace) -> bool {
		const int lfr = static_cast<int>(Lr.positionWS.w);  // this light's per-face resolution (variable res)
		const int tileX = f % 3, tileY = f / 3;
		const int px0 = static_cast<int>(Lr.atlasX) + tileX * lfr;
		const int py0 = static_cast<int>(Lr.atlasY) + tileY * lfr;
		int bx = -1, by = -1; float bd = 0.0f, bDist = kBigDistance;  // bd init = reverse-Z far (0.0)
		for (int y = 0; y < lfr; y += kFaceScanStride)
			for (int x = 0; x < lfr; x += kFaceScanStride) {
				const float d = atlasAtPx(px0 + x, py0 + y);
				if (d <= kEmptyDepthEps) continue;  // reverse-Z: empty/far texel = 0.0
				const float cd = static_cast<float>((x - lfr / 2) * (x - lfr / 2) + (y - lfr / 2) * (y - lfr / 2));
				if (cd < bDist) { bDist = cd; bx = x; by = y; bd = d; }
			}
		if (bx < 0) return false;
		const float fu = (bx + 0.5f) / lfr, fv = (by + 0.5f) / lfr;
		const XMMATRIX inv = XMMatrixInverse(nullptr, XMLoadFloat4x4(&Lr.cubeVP[f]));
		XMFLOAT4 W4; XMStoreFloat4(&W4, XMVector4Transform(XMVectorSet(2.0f * fu - 1.0f, 1.0f - 2.0f * fv, bd, 1.0f), inv));
		if (std::fabs(W4.w) < kClipWEps) return false;
		const XMFLOAT3 Wp{ W4.x / W4.w, W4.y / W4.w, W4.z / W4.w };
		reFace = faceFromDir(Wp.x - Lr.positionWS.x, Wp.y - Lr.positionWS.y, Wp.z - Lr.positionWS.z);
		XMFLOAT4 C; XMStoreFloat4(&C, XMVector4Transform(XMVectorSet(Wp.x, Wp.y, Wp.z, 1.0f), XMLoadFloat4x4(&Lr.cubeVP[reFace])));
		const float nx = C.x / C.w, ny = C.y / C.w;
		const float su = (Lr.atlasX + ((reFace % 3) + (nx * 0.5f + 0.5f)) * lfr) / static_cast<float>(rtAtlasW);
		const float sv = (Lr.atlasY + ((reFace / 3) + (ny * -0.5f + 0.5f)) * lfr) / static_cast<float>(rtAtlasH);
		diff = lin(bd, Lr.nearPlane, Lr.farPlane) - lin(atlasAtUV(su, sv), Lr.nearPlane, Lr.farPlane);
		return true;
	};

	// ================= PER-LIGHT SWEEP (all lights, one dump) =================
	// For every light: population % per cube face (is the map rendered?), round-trip diff per face
	// (is render↔sample consistent?). pop% == 0 everywhere => nothing rendered into that block;
	// large rtDiff or reFace mismatch (marked '*') => a projection inconsistency for that face.

	logger::info("--- per-light sweep ({} lights): pop% per face | rtDiff per face (~0 = consistent; '*' = reFace mismatch) ---",
	    lightRecords.size());
	logger::info("camera moved this frame = {:.2f}u  (compare to each light's moveThisFrame below: moveThisFrame ~= this => that light RIDES THE CAMERA = the shadow that moves with the camera)",
	    cameraMovedThisFrame);
	logger::info("CASTERS riding the camera this frame = {} (worst {:.1f}u '{}')  (a caster whose bound moved ~= the camera => it's drawn in CAMERA-RELATIVE space => its shadow slides with the camera; needs camMove>1u i.e. a moving/delayed dump)",
	    castersRidingCam, casterRideMaxDelta, casterRideName.empty() ? "-" : casterRideName.c_str());
	for (size_t li = 0; li < lightRecords.size(); ++li) {
		const VirtualShadowMaps::LightRecord& L = lightRecords[li];
		const int lfr = static_cast<int>(L.positionWS.w);  // this light's per-face resolution (variable res)
		const float dCam = std::sqrt((L.positionWS.x - camPos.x) * (L.positionWS.x - camPos.x) +
		                             (L.positionWS.y - camPos.y) * (L.positionWS.y - camPos.y) +
		                             (L.positionWS.z - camPos.z) * (L.positionWS.z - camPos.z));
		const int   ldyn  = (li < lightDynamic.size()) ? lightDynamic[li] : -1;
		const float lmove = (li < lightMove.size()) ? lightMove[li] : -1.0f;
		const char* lname = (li < lightNames.size() && !lightNames[li].empty()) ? lightNames[li].c_str() : "?";
		// Mode-23 tint colour for this light index (matches VSM.hlsli::ShadowTint) — so a shadow the user
		// reports as this colour on screen maps straight back to THIS light.
		auto fracf = [](float v) { return v - std::floor(v); };
		const float t23r = fracf(li * 0.1237f + 0.11f), t23g = fracf(li * 0.3391f + 0.11f), t23b = fracf(li * 0.5541f + 0.11f);
		const RE::FormID lref = (li < lightOwnerRef.size()) ? lightOwnerRef[li] : 0;
		const char* lrefBase = "";
		if (auto* rr = RE::TESForm::LookupByID<RE::TESObjectREFR>(lref))
			if (auto* base = rr->GetBaseObject())
				if (const char* e = base->GetFormEditorID()) lrefBase = e;
		logger::info("L{:02d} '{}' ownerRef=0x{:08X}({}) posAbs={:9.1f} {:9.1f} {:9.1f} near={:6.2f} far={:8.2f} atlasOrigin=({:.0f} {:.0f}) faceRes={:.0f} dCam={:.0f} dyn={} moveThisFrame={:.2f}u tint23=RGB({:.2f} {:.2f} {:.2f}) lightPosWS={:9.1f} {:9.1f} {:9.1f}",
		    static_cast<int>(li), lname, lref, lrefBase, L.positionWS.x, L.positionWS.y, L.positionWS.z, L.nearPlane, L.farPlane,
		    L.atlasX, L.atlasY, L.positionWS.w, dCam, ldyn, lmove, t23r, t23g, t23b, L.positionWS.x - altEye.x, L.positionWS.y - altEye.y, L.positionWS.z - altEye.z);
		// FULL light-object identity: the GetUserData ref chain + the ref the light's NODE hangs under (they
		// differ for an ATTACHED light) — so the log alone proves whether this light and the fire-pit mesh
		// are the same object / share a parent, which is what the ref-cull relies on.
		logger::info("       L{:02d} owner {}  |  nodeHangsUnderRef=0x{:08X}{}",
		    static_cast<int>(li), RefIdentity(lref),
		    (li < lightNodeRef.size()) ? lightNodeRef[li] : 0,
		    (li < lightNodeRef.size() && lightNodeRef[li] != lref) ? "  <== differs from GetUserData ref!" : "");
		// Authored + engine light metadata (identity/type/near/color/radius) so the log names the light and
		// carries the game's own "should this cast + how close does it start" data. Parallel to lightRecords.
		if (li < lightMeta.size()) {
			const auto& lm = lightMeta[li];
			logger::info("     L{:02d} identity formID=0x{:08X} editorID='{}' type={} authoredNear={:.2f} radiusXYZ=({:.1f} {:.1f} {:.1f}) colorRGB=({:.2f} {:.2f} {:.2f})",
			    static_cast<int>(li), lm.formID, lm.editorID.c_str(), lm.typeStr.c_str(), lm.authoredNear,
			    lm.radiusX, lm.radiusY, lm.radiusZ, lm.colR, lm.colG, lm.colB);
		}
		// The six cube view-proj matrices (world-absolute -> face clip; row-vector v*M, matches VSM.hlsli).
		// With these + a caster's world verts you can project any point to any face and reproduce ndc/depth
		// exactly. 16 floats per face, row-major.
		for (int f = 0; f < 6; ++f) {
			const auto& m = L.cubeVP[f].m;
			logger::info("     L{:02d} cubeVP[{}] = {: .5g} {: .5g} {: .5g} {: .5g}  {: .5g} {: .5g} {: .5g} {: .5g}  {: .5g} {: .5g} {: .5g} {: .5g}  {: .5g} {: .5g} {: .5g} {: .5g}",
			    static_cast<int>(li), f, m[0][0], m[0][1], m[0][2], m[0][3], m[1][0], m[1][1], m[1][2], m[1][3],
			    m[2][0], m[2][1], m[2][2], m[2][3], m[3][0], m[3][1], m[3][2], m[3][3]);
		}

		const float NaNf = std::numeric_limits<float>::quiet_NaN();
		float pop[6]; float rtd[6]; int rtf[6]; float nod[6];  // nod = nearest-occluder linear distance
		for (int f = 0; f < 6; ++f) {
			const int tileX = f % 3, tileY = f / 3;
			const int px0 = static_cast<int>(L.atlasX) + tileX * lfr;
			const int py0 = static_cast<int>(L.atlasY) + tileY * lfr;
			int popc = 0, total = 0; float mx = 0.0f;  // reverse-Z: track MAX depth = nearest occluder
			for (int y = 0; y < lfr; y += kPopScanStride)
				for (int x = 0; x < lfr; x += kPopScanStride) {
					++total; const float dd = atlasAtPx(px0 + x, py0 + y);
					if (dd > kEmptyDepthEps) ++popc;  // populated = a real occluder was rasterized
					if (dd > mx) mx = dd;
				}
			pop[f] = 100.0f * popc / total;
			nod[f] = (mx > kEmptyDepthEps) ? lin(mx, L.nearPlane, L.farPlane) : NaNf;
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
		// far plane (== the light radius) is too small and nearby geometry casts/receives no shadow.
		{
			const float radius = L.farPlane;  // far == radius (no FarScale anymore)
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
				if (C.w <= kClipWFrontEps) { ++behind; continue; }
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
				const int px0 = static_cast<int>(L.atlasX) + tileX * lfr;
				const int py0 = static_cast<int>(L.atlasY) + tileY * lfr;
				int bx = -1, by = -1; float bd = 0.0f, bDist = kBigDistance;  // bd init = reverse-Z far (0.0)
				for (int y = 0; y < lfr; y += kFaceScanStride)
					for (int x = 0; x < lfr; x += kFaceScanStride) {
						const float d = atlasAtPx(px0 + x, py0 + y);
						if (d <= kEmptyDepthEps) continue;  // reverse-Z: skip empty/far texels, pick a real occluder
						const float cd = static_cast<float>((x - lfr / 2) * (x - lfr / 2) + (y - lfr / 2) * (y - lfr / 2));
						if (cd < bDist) { bDist = cd; bx = x; by = y; bd = d; }
					}
				if (bx >= 0) {
					const float fu = (bx + 0.5f) / lfr, fv = (by + 0.5f) / lfr;
					const XMMATRIX invM = XMMatrixInverse(nullptr, XMLoadFloat4x4(&L.cubeVP[tf]));
					XMFLOAT4 W4; XMStoreFloat4(&W4, XMVector4Transform(XMVectorSet(2.0f * fu - 1.0f, 1.0f - 2.0f * fv, bd, 1.0f), invM));
					if (std::fabs(W4.w) > kClipWEps) {
						const XMFLOAT3 Wp{ W4.x / W4.w, W4.y / W4.w, W4.z / W4.w };
						const float ox = Wp.x - L.positionWS.x, oy = Wp.y - L.positionWS.y, oz = Wp.z - L.positionWS.z;
						const float dOcc = std::sqrt(ox * ox + oy * oy + oz * oz);
						const float s = dOcc > 1e-3f ? (dOcc + 100.0f) / dOcc : 1.0f;  // 100u past the occluder
						const XMFLOAT3 Wt{ L.positionWS.x + ox * s, L.positionWS.y + oy * s, L.positionWS.z + oz * s };
						const int f2 = faceFromDir(Wt.x - L.positionWS.x, Wt.y - L.positionWS.y, Wt.z - L.positionWS.z);
						XMFLOAT4 C; XMStoreFloat4(&C, XMVector4Transform(XMVectorSet(Wt.x, Wt.y, Wt.z, 1.0f), XMLoadFloat4x4(&L.cubeVP[f2])));
						if (C.w > kClipWFrontEps) {
							const float nz = C.z / C.w, nx = C.x / C.w, ny = C.y / C.w;
							const float su = (L.atlasX + ((f2 % 3) + (nx * 0.5f + 0.5f)) * lfr) / static_cast<float>(rtAtlasW);
							const float sv = (L.atlasY + ((f2 / 3) + (ny * -0.5f + 0.5f)) * lfr) / static_cast<float>(rtAtlasH);
							const float occ = atlasAtUV(su, sv);
							const float lp = lin(nz, L.nearPlane, L.farPlane), lo = lin(occ, L.nearPlane, L.farPlane);
							const bool shadow = (lp - lo) > vsm::kDebugModeBias;
							logger::info("     synthShadow: pt 100u behind occluder(face {} dist {:.0f}) linPix={:.1f} linOcc={:.1f} diff={:.1f} occ={:.4f} -> {}",
							    tf, dOcc, lp, lo, lp - lo, occ, shadow ? "SHADOW (logic OK)" : "lit (LOGIC BROKEN)");
						}
					}
				}
			}
		}

	}

	// ================= REAL-SHADER PIXEL PROBE (centre pixel — the actual Lighting.hlsl path) =================
	if (probeArmed && pixelProbeBuf && pixelProbeStaging) {
		logger::info("probe target pixel (from C++) = ({:.0f},{:.0f})  render size={:.0f}x{:.0f}  (if the probe's 'centre pixel' below != this, CS is running a STALE shader — recompile needed)",
		    screenW * 0.5f, screenH * 0.5f, screenW, screenH);
		context->CopyResource(pixelProbeStaging.Get(), pixelProbeBuf.Get());
		D3D11_MAPPED_SUBRESOURCE mp{};
		if (SUCCEEDED(context->Map(pixelProbeStaging.Get(), 0, D3D11_MAP_READ, 0, &mp))) {
			const auto& p = *static_cast<const PixelProbe*>(mp.pData);
			if (p.pixel.z < 0.5f) {
				logger::info("REAL-SHADER PROBE: armed but NOT written (written flag=0). Means the u8 UAV never reached");
				logger::info("  Lighting.hlsl, OR the centre pixel is not lit by any point light. Aim the crosshair at a");
				logger::info("  shadow band on a point-lit surface. If it stays 0 everywhere, the OMSetRenderTargets hook");
				logger::info("  isn't binding u8 into the lighting pass (shader/hook mismatch).");
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

				// ---- Identify WHICH CASTER cast the shadow at the crosshair. Back-project the sampled
				// occluder (same face-UV as the pixel, but at the occluder's depth) through the matched
				// light's inverse cube VP to a world point, then name the nearest registry casters. This
				// is what turns "aim at the bad shadow + dump" into "the culprit is caster 'X'". ----
				const int pm = (int)p.matched.x, pf = (int)p.matched.y;
				if (p.result.w < 0.5f && pm >= 0 && pm < static_cast<int>(lightRecords.size()) && pf >= 0 && pf < 6 && p.uv_occ.z > kEmptyDepthEps) {  // reverse-Z: occluder depth > eps
					const XMMATRIX inv = XMMatrixInverse(nullptr, XMLoadFloat4x4(&lightRecords[pm].cubeVP[pf]));
					XMFLOAT4 occ4;
					XMStoreFloat4(&occ4, XMVector4Transform(XMVectorSet(p.ndc.x, p.ndc.y, p.uv_occ.z, 1.0f), inv));
					if (std::fabs(occ4.w) > kClipWEps) {
						const XMFLOAT3 occW{ occ4.x / occ4.w, occ4.y / occ4.w, occ4.z / occ4.w };
						logger::info("  >>> CROSSHAIR SHADOW: occluder world ~= ({:.1f} {:.1f} {:.1f}) (via light {} face {}). Nearest casters:", occW.x, occW.y, occW.z, pm, pf);
						// nearest 3 registry casters to the occluder point (by bound centre)
						int bi[3] = { -1, -1, -1 };
						float bd[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
						for (size_t k = 0; k < registry.size(); ++k) {
							RE::BSGeometry* g = registry[k].geom.get();
							if (!g) continue;
							const auto& c = g->worldBound.center;
							const float d = std::sqrt((c.x - occW.x) * (c.x - occW.x) + (c.y - occW.y) * (c.y - occW.y) + (c.z - occW.z) * (c.z - occW.z));
							for (int s = 0; s < 3; ++s) {
								if (d < bd[s]) { for (int t = 2; t > s; --t) { bd[t] = bd[t-1]; bi[t] = bi[t-1]; } bd[s] = d; bi[s] = static_cast<int>(k); break; }
							}
						}
						for (int s = 0; s < 3; ++s) {
							if (bi[s] < 0) continue;
							RE::BSGeometry* g = registry[bi[s]].geom.get();
							const float rad = g ? g->worldBound.radius : 0.0f;
							logger::info("       caster #{} '{}' dist={:.1f} (r={:.0f}{}){}",
							    bi[s], (g && g->name.c_str()) ? g->name.c_str() : "?", bd[s], rad,
							    bd[s] <= rad ? " INSIDE-bound=LIKELY" : "", registry[bi[s]].engineOnly ? " [skinned]" : "");
						}
						// EXACT occluder from the id-atlas: the caster id written at the sampled atlas texel is
						// definitively the caster that cast this shadow (no "nearest by centre" guessing).
						if (idTex) {
							int px = static_cast<int>(p.uv_occ.x * rtAtlasW), py = static_cast<int>(p.uv_occ.y * rtAtlasH);
							px = px < 0 ? 0 : (px >= rtAtlasW ? rtAtlasW - 1 : px);
							py = py < 0 ? 0 : (py >= rtAtlasH ? rtAtlasH - 1 : py);
							D3D11_TEXTURE2D_DESC sd{}; sd.Width = 1; sd.Height = 1; sd.MipLevels = 1; sd.ArraySize = 1;
							sd.Format = DXGI_FORMAT_R32_UINT; sd.SampleDesc.Count = 1; sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
							ComPtr<ID3D11Texture2D> idStage;
							if (SUCCEEDED(device->CreateTexture2D(&sd, nullptr, &idStage))) {
								D3D11_BOX box{ static_cast<UINT>(px), static_cast<UINT>(py), 0, static_cast<UINT>(px) + 1, static_cast<UINT>(py) + 1, 1 };
								context->CopySubresourceRegion(idStage.Get(), 0, 0, 0, 0, idTex.Get(), 0, &box);
								D3D11_MAPPED_SUBRESOURCE im{};
								if (SUCCEEDED(context->Map(idStage.Get(), 0, D3D11_MAP_READ, 0, &im))) {
									const uint32_t raw = *reinterpret_cast<const uint32_t*>(im.pData);
									context->Unmap(idStage.Get(), 0);
									if (raw == 0) {
										logger::info("  >>> CROSSHAIR OCCLUDER (id-atlas): EMPTY texel — no caster wrote here; the 'shadow' is a match/bounds artifact, not a real occluder.");
									} else {
										const int occId = static_cast<int>(raw) - 1;
										RE::BSGeometry* og = (occId >= 0 && occId < static_cast<int>(registry.size())) ? registry[occId].geom.get() : nullptr;
										const XMFLOAT3 absP{ p.P.x + p.camAdjust.x, p.P.y + p.camAdjust.y, p.P.z + p.camAdjust.z };
										const auto& Lp = lightRecords[pm].positionWS;
										const float rdist = std::sqrt((absP.x - Lp.x) * (absP.x - Lp.x) + (absP.y - Lp.y) * (absP.y - Lp.y) + (absP.z - Lp.z) * (absP.z - Lp.z));
										const float odist = std::sqrt((occW.x - Lp.x) * (occW.x - Lp.x) + (occW.y - Lp.y) * (occW.y - Lp.y) + (occW.z - Lp.z) * (occW.z - Lp.z));
										logger::info("  >>> CROSSHAIR OCCLUDER (id-atlas, EXACT): caster #{} '{}'  receiverDist={:.1f} occluderDist={:.1f} magnification={:.1f}x  (this is THE caster that wrote the shadow texel)",
										    occId, (og && og->name.c_str()) ? og->name.c_str() : "?", rdist, odist, odist > 1e-3f ? rdist / odist : 0.0f);
									}
								}
							}
						}
					}
				}
			}
			context->Unmap(pixelProbeStaging.Get(), 0);
		} else {
			logger::warn("VSM dump: pixel-probe readback map failed");
		}
	} else if (!probeArmed) {
		logger::info("(real-shader pixel probe not armed — tick 'Arm real-shader pixel probe', aim crosshair at the band, then Dump)");
	}

	logger::info("=========================================");

	// Log EVERYTHING for offline ground-truth: real world-space triangles of every caster (exact silhouettes,
	// no box over-occlusion), then the actual shadow map from the real shader.
	DumpGeometry();
	DumpSkinningFull();  // COMPLETE skinning state (palette + dynamicData + per-partition bones/maps/weights/slots) -> skin.bin
	DumpSceneFull();     // EVERYTHING in the scene graph (all nodes + all geometry buffers/props/skin) -> scene.bin
}
