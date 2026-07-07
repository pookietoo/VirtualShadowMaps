#ifndef VSM_HLSLI
#define VSM_HLSLI

// Virtual Shadow Maps — samples our owned cube-shadow atlas for a local light.
// Bound by LightLimitFix::Prepass (t110/t111/s7 via VirtualShadowMaps.dll). The layout and
// cube-face order below MUST match VirtualShadowMaps.h (LightRecord / atlas) and
// VirtualShadowMaps.cpp::BuildCubeMatrices. This file ships with CS but is our logic.
namespace VSM
{
	static const int kFaceRes      = 256;
	static const int kLightsPerRow = 4;
	static const int kMaxLights    = 32;
	static const int kBlockW       = kFaceRes * 3;  // 768
	static const int kBlockH       = kFaceRes * 2;  // 512
	static const int kAtlasW       = kLightsPerRow * kBlockW;                                       // 3072
	static const int kAtlasH       = ((kMaxLights + kLightsPerRow - 1) / kLightsPerRow) * kBlockH;  // 4096

	struct LightRecord
	{
		float4             positionWS;  // xyz = light pos (float4 -> cubeVP lands at byte 32, matches C++)
		float              farPlane;
		float              nearPlane;
		uint               atlasCol;
		uint               atlasRow;
		row_major float4x4 cubeVP[6];
	};

	Texture2D<float>              ShadowAtlas   : register(t110);
	StructuredBuffer<LightRecord> ShadowLights  : register(t111);
	SamplerState                  ShadowSampler : register(s7);

	// Live tuning from the plugin menu (VirtualShadowMaps.cpp::UpdateDebugCB), bound at b13 by
	// LightLimitFix::Prepass. 48-byte layout = three float4 rows; MUST match the C++ struct.
	cbuffer VSMDebug : register(b13)
	{
		float gBiasWorld;    // 0  shadow bias in WORLD units (linear-distance compare)
		int   gMode;         // 4  0 = shadow  1 = off(lit)  >=2 = RGB diagnostic overlay
		float gVizScale;     // 8  grayscale scale for the atlas-depth views (debug)
		int   gSampleSpace;  // 12 0 = absP(CameraPosAdjust)  1 = P(cam-rel)  2 = P + altEye
		float gMatchThresh;  // 16 light-match distance (world units): buffer light vs shader light
		int   gCompareMode;  // 20 0 = linearized-distance compare  1 = raw ndc.z compare
		int   gMatchEye;     // 24 0 = CameraPosAdjust  1 = altEye(posAdjust.getEye) — eye for lightPosWS match
		float gDbgB;         // 28 spare
		float gAltEyeX;      // 32 render eye the atlas was rasterized around (posAdjust)
		float gAltEyeY;      // 36
		float gAltEyeZ;      // 40
		int   gProbeArmed;   // 44 1 = write the real-shader pixel probe (centre pixel) into gVSMPixelProbe
	};

	// Real-shader pixel probe: when armed, GetLocalShadow writes the FULL per-pixel state for the
	// centre screen pixel into this UAV (bound at u7 by VirtualShadowMaps.dll's OMSetRenderTargets
	// hook). This is the ground truth the C++/compute probe cannot see — the REAL input.WorldPosition,
	// the REAL FrameBuffer::CameraPosAdjust, and the resources CS actually bound. Layout MUST match
	// VirtualShadowMaps.cpp::PixelProbe.
	struct VSMPixelProbe
	{
		float4 pixel;       // xy = SV_Position.xy, z = written(1), w = mode
		float4 P;           // input.WorldPosition (camera-relative), the REAL value the shader got
		float4 camAdjust;   // FrameBuffer::CameraPosAdjust.xyz — the REAL shader-side eye (the unknown)
		float4 W;           // SampleP(P).xyz, w = sampleSpace
		float4 lightPosWS;  // the light as LLF passed it
		float4 absLight;    // match key (lightPosWS + matchEye)
		float4 matched;     // x = matched index, y = face, z = inBounds, w = inFront
		float4 ndc;         // xyz = ndc, w = clip.w
		float4 uv_occ;      // xy = atlasUV, z = occluder, w = reason (0 eval,1 mode!=0,2 no-match,3 behind,4 oob)
		float4 result;      // x = linPix, y = linOcc, z = diff, w = shadow (0 = shadowed)
	};
	// UAVs are only legal in pixel/compute shaders. VSM.hlsli is included by Lighting.hlsl ABOVE its
	// #ifdef PSHADER, so it also compiles as a vertex shader — guard the UAV (and its write below) to
	// PSHADER only, or the VS permutation fails to compile.
	// Slot: PS UAV registers share the OM namespace with render-target outputs. Lighting.hlsl outputs
	// up to 8 targets (u0..u7), so the UAV MUST be u8+ (compiler X4509). u8 requires FL11.1 at bind.
#ifdef PSHADER
	RWStructuredBuffer<VSMPixelProbe> gVSMPixelProbe : register(u8);
#endif

	// Convert the pixel's camera-relative position P into the space we PROJECT/MATCH in.
	// The atlas is rasterized in ABSOLUTE world space (geom->world + absolute-light cubeVP), so
	// space 0 (absP) is the physically-correct one; 1/2 are controls that expose the mismatch.
	float3 SampleP(float3 P)
	{
		if (gSampleSpace == 1)
			return P;                                                     // camera-relative (old, broken control)
		if (gSampleSpace == 2)
			return P + float3(gAltEyeX, gAltEyeY, gAltEyeZ);              // C++ render eye (posAdjust)
		return P + FrameBuffer::CameraPosAdjust.xyz;                       // absP — the fix (default)
	}

	// Standard LH perspective depth -> world-space distance from the light.
	float LinearizeCubeDepth(float d, float lnear, float lfar)
	{
		return lnear * lfar / max(lfar - d * (lfar - lnear), 1e-4);
	}

	// Which cube face a world-space direction falls on (matches BuildCubeMatrices order).
	int FaceFromDir(float3 v)
	{
		float3 a = abs(v);
		if (a.x >= a.y && a.x >= a.z)
			return v.x >= 0 ? 0 : 1;  // +X -X
		if (a.y >= a.z)
			return v.y >= 0 ? 2 : 3;  // +Y -Y
		return v.z >= 0 ? 4 : 5;      // +Z -Z
	}

	// Real per-light shadow — ONLY in mode 0. For any debug mode it returns 1 so lighting is
	// untouched and the RGB DebugColor() override (below) paints the diagnostic instead.
	// Match is ALWAYS in absolute space (buffer stores absLight = lightPosWS + CameraPosAdjust);
	// the PROJECTION space is chosen by SampleP so we can A/B the fix live (default = absP).
	// pixelPos = SV_Position.xy (screen pixels), used only to select the probe target pixel.
	// Restructured to a single exit so the armed probe can capture the FULL state (incl. the reason
	// a pixel came back lit) — the returned shadow value is identical to the original branch logic.
	float GetLocalShadow(float3 lightPosWS, float3 P, float2 pixelPos)
	{
		// Probe targets the centre pixel; DynamicResolutionParams2.xy == 1/renderSize (CS convention),
		// so 0.5/that is the render-space centre regardless of resolution / dynamic-res scaling.
		bool isTarget = false;
		if (gProbeArmed != 0) {
			float2 centre = 0.5 / max(FrameBuffer::DynamicResolutionParams2.xy, 1e-6);
			isTarget = all(abs(pixelPos - centre) <= 1.0);
		}

		float3 W        = SampleP(P);                                   // pixel in projection space (glued to atlas)
		float3 matchEye = (gMatchEye == 1) ? float3(gAltEyeX, gAltEyeY, gAltEyeZ) : FrameBuffer::CameraPosAdjust.xyz;
		float3 absLight = lightPosWS + matchEye;                        // match key (absolute)

		int    matched = -1, face = -1, reason = 0, inb = 0, front = 0;
		float  clipW = 0.0, occluder = 1.0, linPix = 0.0, linOcc = 0.0;
		float3 ndc = float3(9, 9, 9);
		float2 atlasUV = float2(0, 0);
		float  shadow = 1.0;

		if (gMode != 0) {
			reason = 1;                                                // debug overlay mode -> lit
		} else {
			[loop] for (uint i = 0; i < (uint)kMaxLights; ++i) {
				float3 rp = ShadowLights[i].positionWS.xyz;
				if (dot(rp, rp) < 1.0)
					continue;
				if (distance(rp, absLight) >= gMatchThresh)
					continue;
				matched         = (int)i;
				LightRecord L   = ShadowLights[i];
				face            = FaceFromDir(W - L.positionWS.xyz);
				float4 clip     = mul(float4(W, 1.0), L.cubeVP[face]);
				clipW           = clip.w;
				if (clip.w <= 1e-4) { reason = 3; break; }              // behind the light -> lit
				front           = 1;
				ndc             = clip.xyz / clip.w;
				if (ndc.x < -1 || ndc.x > 1 || ndc.y < -1 || ndc.y > 1 || ndc.z < 0 || ndc.z > 1) { reason = 4; break; }  // out of bounds -> lit
				inb             = 1;
				float2 faceUV   = ndc.xy * float2(0.5, -0.5) + 0.5;
				float2 tile     = float2(face % 3, face / 3);
				float2 pxc      = float2(L.atlasCol * kBlockW, L.atlasRow * kBlockH) + (tile + faceUV) * kFaceRes;
				atlasUV         = pxc / float2(kAtlasW, kAtlasH);
				occluder        = ShadowAtlas.SampleLevel(ShadowSampler, atlasUV, 0);
				if (gCompareMode == 1) {
					shadow = (ndc.z - occluder > 0.0) ? 0.0 : 1.0;     // raw depth compare (isolates linearize)
				} else {
					linPix = LinearizeCubeDepth(ndc.z,    L.nearPlane, L.farPlane);
					linOcc = LinearizeCubeDepth(occluder, L.nearPlane, L.farPlane);
					shadow = (linPix - linOcc > gBiasWorld) ? 0.0 : 1.0;
				}
				reason = 0;
				break;
			}
			if (matched < 0)
				reason = 2;                                            // no light matched -> lit
		}

		// Guarded to PSHADER: gVSMPixelProbe (the UAV) only exists in the pixel-shader compile.
#ifdef PSHADER
		if (isTarget) {
			VSMPixelProbe pr;
			pr.pixel      = float4(pixelPos, 1.0, (float)gMode);
			pr.P          = float4(P, 0.0);
			pr.camAdjust  = float4(FrameBuffer::CameraPosAdjust.xyz, 0.0);
			pr.W          = float4(W, (float)gSampleSpace);
			pr.lightPosWS = float4(lightPosWS, 0.0);
			pr.absLight   = float4(absLight, 0.0);
			pr.matched    = float4((float)matched, (float)face, (float)inb, (float)front);
			pr.ndc        = float4(ndc, clipW);
			pr.uv_occ     = float4(atlasUV, occluder, (float)reason);
			pr.result     = float4(linPix, linOcc, linPix - linOcc, shadow);
			gVSMPixelProbe[0] = pr;
		}
#endif
		return shadow;
	}

	// When true, Lighting.hlsl replaces the pixel color with DebugColor() so we can visualize any
	// pipeline stage as RGB (positions / directions / face / UV / depth) — many hypotheses, one build.
	bool WantsDebugOverride() { return gMode >= 2; }

	float3 DebugColor(float3 P)
	{
		float3 absP = P + FrameBuffer::CameraPosAdjust.xyz;  // true world space (control anchor)
		float3 W    = SampleP(P);                            // projection/match space (honors dropdown)

		// Nearest valid buffer light to this pixel, measured in the SAME space we project in.
		int   best  = -1;
		float bestD = 1e30;
		[loop] for (uint i = 0; i < (uint)kMaxLights; ++i) {
			float3 rp = ShadowLights[i].positionWS.xyz;
			if (dot(rp, rp) < 1.0)
				continue;
			float d = distance(rp, W);
			if (d < bestD) { bestD = d; best = (int)i; }
		}

		// Space-independent controls (always valid, even with no lights matched):
		if (gMode == 2)  return frac(absP / 256.0);                  // absP as RGB — GLUED to world if correct
		if (gMode == 3)  return frac(P    / 256.0);                  // P as RGB — camera-relative, EXPECTED to move
		if (gMode == 18) return frac((absP - W) / 256.0);           // space delta — BLACK if sample space == absP
		if (gMode == 21) return frac(FrameBuffer::CameraPosAdjust.xyz / 4096.0);  // CameraPosAdjust sign/magnitude
		if (gMode == 22)                                            // eye delta — BLACK if CameraPosAdjust == altEye
			return saturate(abs(FrameBuffer::CameraPosAdjust.xyz - float3(gAltEyeX, gAltEyeY, gAltEyeZ)) / 50.0);
		if (best < 0)
			return float3(0.0, 0.0, 0.0);                            // no lights in the buffer at all

		LightRecord L   = ShadowLights[best];
		float3 rp       = L.positionWS.xyz;
		float3 dir      = W - rp;
		int    face     = FaceFromDir(dir);
		float4 clip     = mul(float4(W, 1.0), L.cubeVP[face]);
		bool   front    = clip.w > 1e-4;
		float3 ndc      = front ? clip.xyz / clip.w : float3(9, 9, 9);
		bool   inBounds = abs(ndc.x) <= 1 && abs(ndc.y) <= 1 && ndc.z >= 0 && ndc.z <= 1;
		float2 faceUV   = ndc.xy * float2(0.5, -0.5) + 0.5;
		float2 tile     = float2(face % 3, face / 3);
		float2 px       = float2(L.atlasCol * kBlockW, L.atlasRow * kBlockH) + (tile + faceUV) * kFaceRes;
		float2 atlasUV  = px / float2(kAtlasW, kAtlasH);
		float  occluder = ShadowAtlas.SampleLevel(ShadowSampler, atlasUV, 0);
		float  linPix   = LinearizeCubeDepth(saturate(ndc.z), L.nearPlane, L.farPlane);
		float  linOcc   = LinearizeCubeDepth(occluder,        L.nearPlane, L.farPlane);
		float3 faceCol[6] = { float3(1,0,0), float3(0,1,0), float3(0,0,1), float3(1,1,0), float3(1,0,1), float3(0,1,1) };

		if (gMode == 4)  return frac(rp / 256.0);                                    // nearest light pos — CONSTANT per region
		if (gMode == 5)  return normalize(dir) * 0.5 + 0.5;                          // light->pixel direction (smooth)
		if (gMode == 6)  return faceCol[face];                                       // cube face (6 flat colors)
		if (gMode == 7)  return front ? float3(0,1,0) : float3(1,0,0);               // in FRONT of the light? green=yes
		if (gMode == 8)  return float3(saturate(ndc.xy * 0.5 + 0.5), 0.0);           // ndc.xy (position on the face)
		if (gMode == 9)  return saturate(ndc.z).xxx;                                 // pixel depth ndc.z (should be radial)
		if (gMode == 10) return inBounds ? float3(0,1,0) : float3(1,0,0);            // projected INSIDE the face? green=yes
		if (gMode == 11) return float3(atlasUV, 0.0);                                // where in the atlas we sample
		if (gMode == 12) return saturate(occluder).xxx;                             // RAW atlas depth we read (occluder)
		if (gMode == 13) return saturate(linOcc / gVizScale).xxx;                   // occluder distance
		if (gMode == 14) return saturate(linPix / gVizScale).xxx;                   // pixel distance from the light
		if (gMode == 15) return (linPix - linOcc > gBiasWorld) ? float3(1,0,0) : float3(0,1,0);  // shadow decision: red=shadow
		if (gMode == 16) return bestD < gMatchThresh ? float3(0,1,0) : float3(1, saturate(bestD / 500.0), 0);  // light matched? green=yes
		if (gMode == 17) return frac(float3(best * 0.1237, best * 0.3391, best * 0.5541) + 0.11);      // matched light index color
		if (gMode == 19) return (occluder < 0.9999) ? float3(0,1,0) : float3(0.2,0,0);  // atlas populated where we sample? green=depth<1
		if (gMode == 20) return (ndc.z - occluder > 0.0) ? float3(1,0,0) : float3(0,1,0);  // raw-depth shadow decision (no linearize)
		return float3(0,0,0);
	}
}
#endif
