// ============================================================================
// VirtualShadowMaps — diagnostics translation unit.
//
// All the numeric dumps, GPU/pixel probes, scene-graph census, caster inspection,
// and the debug depth-preview resolve live here, split out of the core render path
// (VirtualShadowMaps.cpp) so that file stays focused on setup + light collection +
// skinning + the atlas render. These are methods of the same VirtualShadowMaps class
// (declared in VirtualShadowMaps.h); shared helpers come from VSMInternal.h.
//
// This is a debugging apparatus built to chase the open "shadows shift in the Inn"
// investigation — kept intact, just relocated.
// ============================================================================

#include "VirtualShadowMaps.h"
#include "VSMInternal.h"

#include <DirectXPackedVector.h>

#include <algorithm>
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

using namespace DirectX;
using namespace vsm;
using namespace vsm::internal;

namespace
{
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
}

// ---- The diagnostic methods below are appended verbatim from the original single file. ----

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

// Dump the coordinate-space vectors + a full numeric proof of the shadow-sample math. We verify the
// mode-0 light match from real numbers instead of RGB: the match works iff the eye LLF used to make
// the shader light camera-relative (posAdjust.getEye == altEye) equals the eye the shader adds back.
// Our buffer stores rp = niLight->world.translate (absolute); LLF stores lightPosWS = rp - altEye.
// For a selected light + the caster nearest it, this replicates the EXACT shader path
// (VSM.hlsli::GetLocalShadow) on the CPU against real surface vertices, and reads the atlas back so we
// can compare linPix vs linOcc per point. A vertex that was rasterized into the atlas MUST self-shadow
// to near-equality (linPix ≈ linOcc, diff ≈ 0). Any large divergence localizes the coordinate-space /
// projection bug with hard numbers — no visuals.
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
	    shadowBiasWorld, dbgMatchThresh, shadowFarScale, shadowNearFrac, dbgSampleSpace, dbgCompareMode, dbgMatchEye);
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
	    rejectedNoRenderData, rejectedNoVertexBuffer, rejectedNoIndexBuffer, rejectedZeroTriangles, rejectedDuplicate);

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
			const float radius = L.farPlane / (std::max)(shadowFarScale, 1e-3f);
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
							const bool shadow = (lp - lo) > shadowBiasWorld;
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
			cb.bias        = shadowBiasWorld; cb.compareMode = dbgCompareMode; cb.matchEye = dbgMatchEye; cb.sampleSpace = dbgSampleSpace;
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
