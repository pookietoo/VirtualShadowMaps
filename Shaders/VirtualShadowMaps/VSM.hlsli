#ifndef VSM_HLSLI
#define VSM_HLSLI

// Virtual Shadow Maps — samples the plugin's cube-shadow atlas for a local light.
// Bound by LightLimitFix::Prepass (t110/t111/s7 via VirtualShadowMaps.dll). The layout and
// cube-face order below MUST match VirtualShadowMaps.h (LightRecord / atlas) and
// VirtualShadowMaps.cpp::BuildCubeMatrices. This file ships with CS but is part of the plugin.
namespace VSM
{
	// The ShadowLights buffer is DYNAMIC (grows with the active shadow-light count — there is NO fixed cap).
	// The light loops below get the real element count from ShadowLights.GetDimensions(), never a compile-time
	// size. Atlas DIMENSIONS are RUNTIME too (gAtlasW/gAtlasH, b13 cbuffer) from iShadowMapResolution — never
	// hard-code them here; the old compile-time kFaceRes/kBlockW/kAtlasW/kAtlasH mis-sampled the atlas at res != 256.

	// Named tunable/sentinel constants (values unchanged; documented for the hot path).
	// Namespace scope so every function below shares one definition.
	static const float kNdotLFloor   = 0.05;  // min N·L in the bias calc — clamps grazing slope so tan(incidence) stays finite
	static const float kEmptySlotMin = 1.0;   // dot(rp,rp) below this = empty ShadowLights slot -> skip
	static const float kDivEps       = 1e-4;  // near-zero divide / behind-light guard
	static const float kNormalOffsetK = 2.0;  // A3: normal-offset bias strength (texel footprints of receiver shift at grazing)
	static const float kCubeFaceNdcExtent = 2.0;  // a cube face spans [-1,1] in NDC = 2 units; (this / faceRes) = NDC per texel
	static const float kBiasTexelScale    = 2.0;  // receiver-plane depth bias expressed in cube-face texel footprints
	static const uint  kAtlasBlockCols = 3;    // a light's 6 faces pack as a 3-wide x 2-tall block: tile = (face % this, face / this)
	static const int   kMaxPCFRadius = 4;      // clamp cap for the runtime soft-shadow PCF half-width (gPCFRadius, from
	                                           // iBlurDeferredShadowMask). Fixed texels x per-light faceRes = constant screen penumbra.

	struct LightRecord   // 32 B — MUST match VirtualShadowMaps.h::GPULightRecord (the uploaded t111 stride)
	{
		float4             positionWS;  // xyz = light pos; w = per-light cube-face resolution (faceRes, texels)
		float              farPlane;
		float              nearPlane;
		float              atlasX;      // pixel origin X of this light's 3x2 cube-face block in the atlas
		float              atlasY;      // pixel origin Y — record ends here; cube-direct sampling needs no cubeVP matrix
	};

	Texture2D<float>              ShadowAtlas   : register(t110);
	StructuredBuffer<LightRecord> ShadowLights  : register(t111);
	SamplerState                  ShadowSampler : register(s7);
	// A5 colored translucent shadows: per-texel transmittance of glass / alpha-blended casters in front of the
	// nearest opaque occluder. 1 = clear (no effect). Bound at t112 by LightLimitFix::Prepass (VirtualShadowMaps.dll).
	// ALWAYS bound and cleared white by the plugin, so when the A5 module is OFF this sample is a no-op multiply by 1.
	Texture2D<float4>             ShadowTransAtlas : register(t112);

	// Runtime params bound at b13 by LightLimitFix::Prepass. 32-byte layout (two float4 rows);
	// MUST match VirtualShadowMaps.cpp::VSMParamsCB (static_assert) and the bench.
	cbuffer VSMParams : register(b13)
	{
		float gBiasScale;      // 0  fShadowBiasScale multiplier on the calculated receiver-plane bias
		float gTranslucentOn;  // 4  A5: 1 = sample+tint via t112, 0 = skip (local light never depends on t112)
		float gPCFRadius;      // 8  soft-shadow PCF kernel half-width in texels (iBlurDeferredShadowMask)
		float gAtlasW;         // 12 runtime atlas width
		float gAtlasH;         // 16 runtime atlas height
		float _pad0, _pad1, _pad2;  // 20,24,28 -> 32 B
	};

	// Perspective depth -> world-space distance from the light. The atlas is REVERSE-Z (near->1, far->0),
	// matching the plugin's kReverseZ=true (VSMInternal.h) — projection z-flip + clear 0.0 + DepthFunc GREATER.
	// Reverse-Z inverse: d=1 -> lnear, d=0 -> lfar  (standard-Z would be lfar - d*(lfar-lnear) in the denom).
	// MUST stay in sync with VSMInternal.h::kReverseZ and tools/shadow_truth.py::_lin.
	float LinearizeCubeDepth(float d, float lnear, float lfar)
	{
		return lnear * lfar / max(lnear + d * (lfar - lnear), kDivEps);
	}

	// Real per-light shadow. LLF stamped this light's ShadowLights slot into Light.vsmShadowIndex and
	// passes it in (Lighting.hlsl). O(1) direct index; sentinel/OOB reads a zeroed record -> lit.
	float GetLocalShadow(uint vsmIndex, float3 P, float2 pixelPos, float3 worldNormal, out float3 transmittance)
	{
		// SINGLE EXIT (one return): early returns here get inlined into Lighting.hlsl's per-light
		// [loop], and multi-exit control flow inside that 20-feature shader blows up the fxc optimizer
		// (10s+ compiles). Keep the whole thing nested to one exit.
		transmittance = float3(1.0, 1.0, 1.0);
		float  shadow = 1.0;                               // default: lit (empty slot / outside the light's shell)
		float3 W = P + FrameBuffer::CameraPosAdjust.xyz;   // absolute world space (atlas is rasterized here)
		LightRecord L = ShadowLights[vsmIndex];
		if (dot(L.positionWS.xyz, L.positionWS.xyz) >= kEmptySlotMin) {  // sentinel/OOB -> zeroed record -> lit
			// A3 normal-offset bias
			float3 nrmA3  = normalize(worldNormal);
			float3 toLA3  = L.positionWS.xyz - W;
			float  dLA3   = length(toLA3);
			float  ndlA3  = saturate(dot(nrmA3, toLA3 / max(dLA3, kDivEps)));
			float  footA3 = dLA3 * (2.0 / L.positionWS.w);
			float3 Woff   = W + nrmA3 * (footA3 * saturate(1.0 - ndlA3) * kNormalOffsetK);
			// cube-direct (no cubeVP matrix): the major axis of the light->receiver direction IS the cube
			// face; the other two components (sign-corrected per face) are the in-face coordinates. One
			// abs()+compare cascade yields the face AND the coords together — this is the per-light hot loop.
			float3 dir    = Woff - L.positionWS.xyz;
			float3 dirAbs = abs(dir);
			uint   face;
			float  majorAxisMag, faceX, faceY;
			if (dirAbs.x >= dirAbs.y && dirAbs.x >= dirAbs.z) { face = dir.x >= 0.0 ? 0u : 1u; majorAxisMag = dirAbs.x; faceX = (dir.x >= 0.0 ? dir.y : -dir.y); faceY = dir.z; }
			else if (dirAbs.y >= dirAbs.z)                    { face = dir.y >= 0.0 ? 2u : 3u; majorAxisMag = dirAbs.y; faceX = (dir.y >= 0.0 ? -dir.x : dir.x); faceY = dir.z; }
			else                                              { face = dir.z >= 0.0 ? 4u : 5u; majorAxisMag = dirAbs.z; faceX = (dir.z >= 0.0 ? dir.x : -dir.x); faceY = dir.y; }
			if (majorAxisMag <= L.farPlane && majorAxisMag >= L.nearPlane) {  // inside the light's [near, far] shell
				float  faceRes       = L.positionWS.w;
				float  faceEdgeInset = (faceRes - kCubeFaceNdcExtent) / faceRes;  // 1-texel border per side: keeps the UV off the cube-face seam
				float2 faceUV  = float2((faceX / majorAxisMag) * faceEdgeInset * 0.5 + 0.5, -(faceY / majorAxisMag) * faceEdgeInset * 0.5 + 0.5);
				faceUV         = clamp(faceUV, 0.5 / faceRes, 1.0 - 0.5 / faceRes);
				float2 tile    = float2(face % kAtlasBlockCols, face / kAtlasBlockCols);
				float2 pxc     = float2(L.atlasX, L.atlasY) + (tile + faceUV) * faceRes;
				float2 atlasUV = pxc / float2(gAtlasW, gAtlasH);
				if (gTranslucentOn > 0.5)
					transmittance = ShadowTransAtlas.SampleLevel(ShadowSampler, atlasUV, 0).rgb;
				float3 Ldir      = normalize(toLA3);   // light->receiver dir, reused from the A3 bias calc above
				float  ndl       = max(dot(worldNormal, Ldir), kNdotLFloor);
				float  slope     = sqrt(saturate(1.0 - ndl * ndl)) / ndl;
				float  footprint = majorAxisMag * (kCubeFaceNdcExtent / faceRes);
				float  bias      = kBiasTexelScale * footprint * slope * gBiasScale;
				int    pr  = clamp((int)gPCFRadius, 0, kMaxPCFRadius);
				float  sum = 0.0, cnt = 0.0;
				[loop] for (int oy = -pr; oy <= pr; ++oy)
					[loop] for (int ox = -pr; ox <= pr; ++ox) {
						float2 fuv  = clamp(faceUV + float2(ox, oy) / faceRes, 0.5 / faceRes, 1.0 - 0.5 / faceRes);
						float2 tpx  = float2(L.atlasX, L.atlasY) + (tile + fuv) * faceRes;
						float  occT = ShadowAtlas.SampleLevel(ShadowSampler, tpx / float2(gAtlasW, gAtlasH), 0);
						float  loT  = LinearizeCubeDepth(occT, L.nearPlane, L.farPlane);
						sum += (majorAxisMag - loT > bias) ? 0.0 : 1.0;
						cnt += 1.0;
					}
				shadow = sum / cnt;   // 0 = shadowed .. 1 = lit
			}
		}
		return shadow;
	}

}
#endif
