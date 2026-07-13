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

	// Named tunable/sentinel constants (renamed and documented; the values are unchanged).
	// Namespace scope so every function below shares one definition.
	static const float kNdotLFloor  = 0.05;  // min N·L in the bias calc — clamps grazing slope so tan(incidence) stays finite
	static const float kProbeAccept = 1.5;   // probe target half-window (pixels) around the render centre
	static const float kEmptySlotMin = 1.0;  // dot(rp,rp) below this = empty ShadowLights slot -> skip
	static const float kOutOfCubeNdc = 9.0;  // ndc sentinel value meaning "no valid projection" (out of cube / behind)
	static const float kDivEps      = 1e-4;  // near-zero divide / behind-light guard
	static const float kArgMinInit  = 1e30;  // initial best-distance for the nearest-light argmin
		static const float kNormalOffsetK = 2.0; // A3: normal-offset bias strength (texel footprints of receiver shift at grazing)
		static const float kDebugMatchThresh = 5.0; // debug mode 16 ONLY: nearest-light match distance threshold (was the b13
		                                            // gMatchThresh field, now repurposed as gTranslucentOn).
	static const int   kMaxPCFRadius = 4;    // clamp cap for the runtime soft-shadow PCF half-width (gPCFRadius, from
	                                         // iBlurDeferredShadowMask). Fixed texels x per-light faceRes = constant screen penumbra.

	struct LightRecord
	{
		float4             positionWS;  // xyz = light pos; w = per-light cube-face resolution (faceRes, texels)
		float              farPlane;
		float              nearPlane;
		float              atlasX;      // pixel origin X of this light's 3x2 cube-face block in the atlas
		float              atlasY;      // pixel origin Y (cubeVP still lands at byte 32 — layout unchanged)
		row_major float4x4 cubeVP[6];
	};

	Texture2D<float>              ShadowAtlas   : register(t110);
	StructuredBuffer<LightRecord> ShadowLights  : register(t111);
	SamplerState                  ShadowSampler : register(s7);
	// A5 colored translucent shadows: per-texel transmittance of glass / alpha-blended casters in front of the
	// nearest opaque occluder. 1 = clear (no effect). Bound at t112 by LightLimitFix::Prepass (VirtualShadowMaps.dll).
	// ALWAYS bound and cleared white by the plugin, so when the A5 module is OFF this sample is a no-op multiply by 1.
	Texture2D<float4>             ShadowTransAtlas : register(t112);

	// Live tuning from the plugin menu (VirtualShadowMaps.cpp::UpdateDebugCB), bound at b13 by
	// LightLimitFix::Prepass. 64-byte layout = four float4 rows; MUST match the C++ struct (static_assert).
	cbuffer VSMDebug : register(b13)
	{
		float gBiasScale;    // 0  fShadowBiasScale: multiplier on the CALCULATED per-pixel bias (default 1; <0 = everything shadowed). Was gBiasWorld
		                     //    uses a CALCULATED receiver-plane bias (see GetLocalShadow), not this slider
		int   gMode;         // 4  debug view selector (see mode legend below the cbuffer): 0 = real shadow,
		                     //    1 = off(lit), 2..22 = RGB diagnostic overlay, 23/24/25 = shadow-tint (not overlays)
		float gVizScale;     // 8  grayscale scale for the atlas-depth views (debug)
		int   gSampleSpace;  // 12 0 = absP(CameraPosAdjust)  1 = P(cam-rel)  2 = P + altEye
		float gTranslucentOn;// 16 A5: 1 = the transmittance atlas (t112) is live -> sample + tint the light; 0 (default) =
		                     //    skip it entirely, so local lights NEVER depend on t112 being bound. (Was the dead gMatchThresh.)
		int   gCompareMode;  // 20 0 = linearized-distance compare  1 = raw ndc.z compare
		int   gMatchEye;     // 24 0 = CameraPosAdjust  1 = altEye(posAdjust.getEye) — eye for lightPosWS match
		float gPCFRadius;    // 28 soft-shadow PCF kernel half-width in texels (from iBlurDeferredShadowMask; 0 = hard). Was gFaceRes (now per-light in positionWS.w)
		float gAltEyeX;      // 32 render eye the atlas was rasterized around (posAdjust)
		float gAltEyeY;      // 36
		float gAltEyeZ;      // 40
		int   gProbeArmed;   // 44 1 = write the real-shader pixel probe (target pixel) into gVSMPixelProbe
		float gProbePixelX;  // 48 target pixel for the probe (SV_Position), fed from C++ = render centre
		float gProbePixelY;  // 52
		float gAtlasH;       // 56 RUNTIME atlas height  (rows x 2 x faceRes)
		float gAtlasW;       // 60 RUNTIME atlas width   (cols x 3 x faceRes)
	};

	// gMode legend — the debug ladder used by GetLocalShadow (shadow/tint paths) and DebugColor (RGB
	// overlays). Bare integer codes are the b13 cbuffer contract (kept as ints, NOT an enum, so the
	// C++ side in VirtualShadowMaps.cpp stays byte-compatible). What each code shows:
	//    0  real per-light shadow (the shipping path; darkens)
	//    1  off — scene left lit, no shadow
	//    2  absP as RGB (world-glued if the fix is correct)
	//    3  P as RGB (camera-relative; EXPECTED to move with the camera)
	//    4  nearest light position (constant per region)
	//    5  light->pixel direction (smooth)
	//    6  cube face (6 flat colours)
	//    7  in FRONT of the light? green=yes
	//    8  ndc.xy (position on the face)
	//    9  ndc.z (pixel depth; should be radial)
	//   10  projected INSIDE the face? green=yes
	//   11  atlas UV we sample
	//   12  raw atlas depth read (occluder)
	//   13  occluder linear distance
	//   14  pixel linear distance from the light
	//   15  LEGACY gBiasWorld shadow decision (NOT the calculated-bias path); red=shadow
	//   16  LEGACY light-matched? (gMatchThresh; unused by the real match); green=matched
	//   17  matched light index colour
	//   18  sample-space delta (BLACK if sample space == absP)
	//   19  atlas populated where we sample? green=depth>0 (reverse-Z empty=0)
	//   20  raw-depth shadow decision (no linearize)
	//   21  CameraPosAdjust sign/magnitude
	//   22  eye delta (BLACK if CameraPosAdjust == altEye)
	//   23  shadow-TINT: matched light index (repaints only our shadowed pixels)
	//   24  shadow-TINT: cube face
	//   25  shadow-TINT: per-occluder unique colour + stripe

	// Real-shader pixel probe: when armed, GetLocalShadow writes the FULL per-pixel state for the
	// centre screen pixel into this UAV (bound at u8 by VirtualShadowMaps.dll's OMSetRenderTargets
	// hook; see register(u8) below). This is the ground truth the offline analyzer cannot see — the REAL input.WorldPosition,
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

	// Perspective depth -> world-space distance from the light. The atlas is REVERSE-Z (near->1, far->0),
	// matching the plugin's kReverseZ=true (VSMInternal.h) — projection z-flip + clear 0.0 + DepthFunc GREATER.
	// Reverse-Z inverse: d=1 -> lnear, d=0 -> lfar  (standard-Z would be lfar - d*(lfar-lnear) in the denom).
	// MUST stay in sync with VSMInternal.h::kReverseZ and tools/shadow_truth.py::_lin.
	float LinearizeCubeDepth(float d, float lnear, float lfar)
	{
		return lnear * lfar / max(lnear + d * (lfar - lnear), kDivEps);
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

	// Shadow-source tint (menu modes 23/24). GetLocalShadow runs the REAL per-light shadow test but,
	// instead of darkening, records which light + cube face darkened this pixel; Lighting.hlsl then
	// paints those pixels a solid identifying colour so the shadow can be read off-screen and matched
	// to a light/face in the dump. HLSL globals are per-invocation (per pixel), and GetLocalShadow is
	// called once per light, so these accumulate across the per-light calls (last shadower wins).
	static bool   gVSMHit      = false;
	static int    gVSMHitLight = -1;
	static int    gVSMHitFace  = -1;
	static float3 gVSMHitOcc   = float3(0, 0, 0);  // reconstructed world pos of the OCCLUDER (mode 25 unique tint)

	// Real per-light shadow — mode 0 (and the tint modes 23/24, which compute it but don't darken).
	// For any OTHER debug mode it returns 1 so lighting is untouched and the RGB DebugColor() override
	// (below) paints the diagnostic instead.
	// Match is ALWAYS in absolute space (buffer stores absLight = lightPosWS + CameraPosAdjust);
	// the PROJECTION space is chosen by SampleP so we can A/B the fix live (default = absP).
	// pixelPos = SV_Position.xy (screen pixels), used only to select the probe target pixel.
	// Restructured to a single exit so the armed probe can capture the FULL state (incl. the reason
	// a pixel came back lit) — the returned shadow value is identical to the original branch logic.
	// LLF path: vsmIndex = this light's slot in ShadowLights, stamped by LLF into Light.vsmShadowIndex (a shared
	// BSLight* key; see docs/ARCHITECTURE §6.1 + notes/CS_INTEGRATION_MANIFEST.md). O(1) — no per-pixel search. A
	// position-argmin overload (non-LLF fallback) is defined below and delegates here.
	float GetLocalShadow(uint vsmIndex, float3 P, float2 pixelPos, float3 worldNormal, out float3 transmittance)
	{
		transmittance = float3(1.0, 1.0, 1.0);  // A5: default clear (white) — set from the transmittance atlas once a light is matched
		// Probe targets the centre pixel; the probe centre is supplied by C++ as gProbePixelX/Y (already in
		// SV_Position space), so no in-shader resolution math is needed.
		bool isTarget = false;
		if (gProbeArmed != 0) {
			// Target pixel comes straight from C++ (render-target centre) — DynamicResolutionParams2 is a
			// scale here, not 1/size, so deriving the centre in-shader put the probe at (0,0).
			float2 centre = float2(gProbePixelX, gProbePixelY);
			isTarget = all(abs(pixelPos - centre) <= kProbeAccept);
		}

		float3 W        = SampleP(P);                                   // pixel in projection space (glued to atlas)
		float3 matchEye = (gMatchEye == 1) ? float3(gAltEyeX, gAltEyeY, gAltEyeZ) : FrameBuffer::CameraPosAdjust.xyz;  // probe only now
		float3 matchedLightPos = float3(0, 0, 0);                       // matched light's world pos (for the diagnostic probe)

		int    matched = -1, face = -1, reason = 0, inb = 0, front = 0;
		float  clipW = 0.0, occluder = 0.0, linPix = 0.0, linOcc = 0.0;  // reverse-Z: 0.0 = far = empty/no-occluder
		float3 ndc = float3(kOutOfCubeNdc, kOutOfCubeNdc, kOutOfCubeNdc);
		float2 atlasUV = float2(0, 0);
		float  shadow = 1.0;

		if (gMode != 0 && gMode != 23 && gMode != 24 && gMode != 25) {
			reason = 1;                                                // pure-overlay debug mode -> lit
		} else {
			// >>> VSM PATCH (light-index alignment) — re-merge: notes/CS_INTEGRATION_MANIFEST.md P6
			// O(1) DIRECT INDEX. LLF stamped this light's ShadowLights slot into Light.vsmShadowIndex and passes it
			// in (Lighting.hlsl). Replaces the old O(lights²)/pixel nearest-position argmin over ALL ShadowLights —
			// the dominant per-pixel sampling cost (proven by testing/gpu/sample_main.cpp). FAIL-SAFE: a sentinel
			// (0xFFFFFFFF) or any out-of-range index reads a ZEROED record (D3D robust structured-buffer access), so
			// it collapses into the SAME empty-slot test the argmin used -> 'lit', never a wrong cube. The non-LLF
			// fallback overload below resolves the index by argmin (compiled only when LIGHT_LIMIT_FIX is off).
			LightRecord L = ShadowLights[vsmIndex];                    // sentinel / OOB -> zeroed record -> caught as empty
			if (dot(L.positionWS.xyz, L.positionWS.xyz) < kEmptySlotMin) {
				reason = 2;                                            // no VSM shadow for this light -> lit
			} else {
				matched        = (int)vsmIndex;
				matchedLightPos = L.positionWS.xyz;                    // for the probe
			// <<< VSM PATCH
				// A3 NORMAL-OFFSET BIAS: shift the receiver along its surface normal toward the light by a fraction
				// of the shadow texel's world footprint, scaled up at grazing angles (where acne + peter-panning
				// live). A shifted receiver samples the atlas depth in FRONT of its own surface, escaping texel
				// quantization at the SOURCE — kills acne AND peter-panning far better than a pure depth bias (which
				// only trades one for the other). The calculated slope bias below stays as a small floor. Reverse-Z
				// safe: it reduces linPix (receiver appears nearer the light) in the convention-independent
				// linear-distance compare, so it can only REDUCE self-shadowing. Foundation for the soft-shadow spine.
				float3 nrmA3  = normalize(worldNormal);
				float3 toLA3  = L.positionWS.xyz - W;
				float  dLA3   = length(toLA3);
				float  ndlA3  = saturate(dot(nrmA3, toLA3 / max(dLA3, kDivEps)));
				float  footA3 = dLA3 * (2.0 / L.positionWS.w);                 // shadow texel world size at this receiver distance
				float3 Woff   = W + nrmA3 * (footA3 * saturate(1.0 - ndlA3) * kNormalOffsetK);
				face          = FaceFromDir(Woff - L.positionWS.xyz);
				// >>> VSM PATCH (cube-direct) — face UV + compare depth DIRECTLY from the direction, NO cubeVP matrix
				// (no 4x4 multiply, no 64-byte matrix read, no ndc divide) -> ~-52% per-pixel sampling. Per-face LH
				// look-to axes matching BuildCubeMatrices (world-Z up for X/Y faces, world-Y up for +/-Z) + guard-band
				// scale (faceRes-2)/faceRes; linPix = axial distance. VALIDATED bit-equivalent to the old matrix path
				// (tools/cube_direct_check.py: 100% face, 99.89% UV, 99.81% verdict over 43,697 receivers). The two
				// range checks below (far, then near) replace the old behind/out-of-cube checks; FaceFromDir already
				// guarantees the point is within its face, so no ndc.xy bounds test is needed.
				float3 dcd  = Woff - L.positionWS.xyz;
				float3 acd  = abs(dcd);
				float  macd, xvcd, yvcd;
				if (acd.x >= acd.y && acd.x >= acd.z) { macd = acd.x; xvcd = (dcd.x >= 0.0 ? dcd.y : -dcd.y); yvcd = dcd.z; }
				else if (acd.y >= acd.z)              { macd = acd.y; xvcd = (dcd.y >= 0.0 ? -dcd.x : dcd.x); yvcd = dcd.z; }
				else                                  { macd = acd.z; xvcd = (dcd.z >= 0.0 ? dcd.x : -dcd.x); yvcd = dcd.y; }
				clipW = macd;                                          // (probe) = axial distance to the light
				if (macd > L.farPlane) {
					reason = 3;                                        // beyond the light's far -> lit
				} else {
					front = 1;
					if (macd < L.nearPlane) {
						reason = 4;                                    // nearer than the light's near plane -> lit
					} else {
						inb = 1;
						// RUNTIME atlas dims (from the b13 cbuffer) so sampling matches the texture the plugin
						// actually allocated this frame (sized for the active lights, at iShadowMapResolution).
						float faceRes = L.positionWS.w;  // per-light cube-face resolution (was global gFaceRes)
						float  scd    = (faceRes - 2.0) / faceRes;             // guard-band FOV scale (matches BuildCubeMatrices)
						float2 faceUV = float2((xvcd / macd) * scd * 0.5 + 0.5, -(yvcd / macd) * scd * 0.5 + 0.5);  // direction -> face UV, NO matrix
						faceUV        = clamp(faceUV, 0.5 / faceRes, 1.0 - 0.5 / faceRes);  // never floor into a neighbour tile
						ndc           = float3(faceUV * 2.0 - 1.0, saturate(macd / L.farPlane));  // (probe only; the compare uses linPix)
						float2 tile   = float2(face % 3, face / 3);
						float2 pxc    = float2(L.atlasX, L.atlasY) + (tile + faceUV) * faceRes;  // atlasX/Y = block pixel origin
						atlasUV       = pxc / float2(gAtlasW, gAtlasH);
						occluder      = ShadowAtlas.SampleLevel(ShadowSampler, atlasUV, 0);
						// A5: colored transmittance of any glass/alpha caster in front of the opaque occluder along this
						// texel's ray. Sampled ONLY when the module is live (gTranslucentOn) — otherwise transmittance stays
						// white, so a local light NEVER depends on t112 being bound (an unbound t112 samples black and would
						// zero the light). Multiplies the local light in Lighting.hlsl.
						if (gTranslucentOn > 0.5)
							transmittance = ShadowTransAtlas.SampleLevel(ShadowSampler, atlasUV, 0).rgb;
						if (gCompareMode == 1) {
							shadow = (occluder - ndc.z > 0.0) ? 0.0 : 1.0;   // raw compare (isolates linearize); reverse-Z: occluder nearer (larger z) than receiver = shadowed
						} else {
							linPix = macd;   // cube-direct: axial distance IS the compare depth (no linearize-of-ndc.z)
							linOcc = LinearizeCubeDepth(occluder, L.nearPlane, L.farPlane);
							// CALCULATED receiver-plane bias — replaces the gBiasWorld slider AND the dead
							// rasterizer DepthBias (=~0.19u on a D32_FLOAT atlas, useless). A receiver's distance
							// to the light can differ from the nearest surface stored in its atlas texel by up to
							// the texel's lateral footprint x the surface slope vs the light: footprint =
							// linPix*(2/gFaceRes) (a cube face spans tan[-1,1] over gFaceRes texels); slope =
							// tan(incidence) from the G-buffer normal; x2 = texel diagonal + point-sample jitter.
							// Face-on (normal toward light) -> slope~0 -> ~no bias; grazing -> large, exactly
							// where the self-shadow acne lives. No tuned constant, no slider.
							float3 Ldir      = normalize(L.positionWS.xyz - W);
							float  ndl       = max(dot(worldNormal, Ldir), kNdotLFloor);
							float  slope     = sqrt(saturate(1.0 - ndl * ndl)) / ndl;  // tan(incidence) from the G-buffer normal
							float  footprint = linPix * (2.0 / faceRes);               // texel lateral size at this distance
							float  bias      = 2.0 * footprint * slope * gBiasScale;   // x2 = texel diagonal + jitter; gBiasScale = fShadowBiasScale
							// SOFT SHADOWS (PCF): average the depth compare over a RUNTIME texel kernel (half-width
							// gPCFRadius from iBlurDeferredShadowMask; 0 = hard). A fixed texel radius x this light's
							// variable faceRes rung => ~constant SCREEN-space penumbra across lights (the coupling in
							// notes/VSM_VARIABLE_RESOLUTION.md). Each tap is clamped inside this face (no neighbour bleed).
							int    pr  = clamp((int)gPCFRadius, 0, kMaxPCFRadius);
							float  sum = 0.0, cnt = 0.0;
							[loop] for (int oy = -pr; oy <= pr; ++oy)
								[loop] for (int ox = -pr; ox <= pr; ++ox) {
									float2 fuv  = clamp(faceUV + float2(ox, oy) / faceRes, 0.5 / faceRes, 1.0 - 0.5 / faceRes);
									float2 tpx  = float2(L.atlasX, L.atlasY) + (tile + fuv) * faceRes;
									float  occT = ShadowAtlas.SampleLevel(ShadowSampler, tpx / float2(gAtlasW, gAtlasH), 0);
									float  loT  = LinearizeCubeDepth(occT, L.nearPlane, L.farPlane);
									sum += (linPix - loT > bias) ? 0.0 : 1.0;
									cnt += 1.0;
								}
							shadow = sum / cnt;  // 0=shadowed .. 1=lit (fractional=penumbra); pr=0 -> single tap (hard)
							// NO far-distance fade here — CS's light falloff is matched instead of imposing another. CS already
							// multiplies the shadow by the light's own attenuation (Lighting.hlsl: lightColor *= intensityMultiplier
							// = InverseSquareLighting::GetAttenuation, which smoothsteps to 0 at the radius), so the shadow's
							// visible effect vanishes exactly as the light does. An added linear fade would double-fade on a MISMATCHED
							// curve (linear vs inverse-square) and lift a visible border while the light is
							// still bright.
						}
						// Tint modes: record what shadowed this pixel so Lighting.hlsl can paint it.
						if ((gMode == 23 || gMode == 24 || gMode == 25) && shadow < 0.5) {
							gVSMHit = true; gVSMHitLight = matched; gVSMHitFace = face;
							float linOccTint = LinearizeCubeDepth(occluder, L.nearPlane, L.farPlane);
							gVSMHitOcc = L.positionWS.xyz + normalize(W - L.positionWS.xyz) * linOccTint;
						}
						reason = 0;
					}
				}
			}
		}

		// Guarded to PSHADER: gVSMPixelProbe (the UAV) only exists in the pixel-shader compile.
#ifdef PSHADER
		if (isTarget) {
			VSMPixelProbe pr;
			pr.pixel      = float4(pixelPos, 1.0, (float)gMode);
			pr.P          = float4(P, 0.0);
			pr.camAdjust  = float4(FrameBuffer::CameraPosAdjust.xyz, 0.0);
			pr.W          = float4(W, (float)gSampleSpace);
			pr.lightPosWS = float4(matchedLightPos, 0.0);              // VSM PATCH: was the lightPosWS param; now the matched light's pos
			pr.absLight   = float4(matchedLightPos + matchEye, 0.0);   // VSM PATCH: match key reconstructed from the matched light
			pr.matched    = float4((float)matched, (float)face, (float)inb, (float)front);
			pr.ndc        = float4(ndc, clipW);
			pr.uv_occ     = float4(atlasUV, occluder, (float)reason);
			pr.result     = float4(linPix, linOcc, linPix - linOcc, shadow);
			gVSMPixelProbe[0] = pr;
		}
#endif
		// Tint modes 23/24/25 don't darken — they leave the scene lit and Lighting.hlsl paints the hit
		// pixels (see ShadowTint). Every other non-zero mode returns 1 (DebugColor paints the overlay).
		return (gMode == 23 || gMode == 24 || gMode == 25) ? 1.0 : shadow;
	}

	// NON-LLF FALLBACK overload (position-argmin -> index, then delegate). Community Shaders with Light Limit Fix
	// (the shipping config) always calls the uint-index overload above; this float3 overload is compiled ONLY for
	// the !LIGHT_LIMIT_FIX permutation of Lighting.hlsl (the PointLightPosition path), where no LLF Light struct —
	// and thus no stamped vsmShadowIndex — exists. It reconstructs the index by the same nearest-position argmin the
	// index path replaced (O(lights) per call, only on that degenerate path), so behavior there is unchanged.
	float GetLocalShadow(float3 lightPosWS, float3 P, float2 pixelPos, float3 worldNormal, out float3 transmittance)
	{
		float3 matchEye = (gMatchEye == 1) ? float3(gAltEyeX, gAltEyeY, gAltEyeZ) : FrameBuffer::CameraPosAdjust.xyz;
		float3 absLight = lightPosWS + matchEye;                       // match key (absolute)
		int  best = -1; float bestD = kArgMinInit;
		uint lightCount, lightStride; ShadowLights.GetDimensions(lightCount, lightStride);
		[loop] for (uint i = 0; i < lightCount; ++i) {
			float3 rp = ShadowLights[i].positionWS.xyz;
			if (dot(rp, rp) < kEmptySlotMin) continue;
			float dl = distance(rp, absLight);
			if (dl < bestD) { bestD = dl; best = (int)i; }
		}
		return GetLocalShadow(best < 0 ? 0xFFFFFFFFu : (uint)best, P, pixelPos, worldNormal, transmittance);
	}

	// When true, Lighting.hlsl replaces the pixel color with DebugColor() so we can visualize any
	// pipeline stage as RGB (positions / directions / face / UV / depth) — many hypotheses, one build.
	// Excludes the shadow-tint modes (23/24/25), which use ShadowTint() instead.
	bool WantsDebugOverride() { return gMode >= 2 && gMode != 23 && gMode != 24 && gMode != 25; }

	// Shadow-source tint (modes 23/24/25): keep the real lit scene, but repaint pixels our shadow
	// darkened with a colour identifying the SOURCE.
	//   23 = matched LIGHT index (few colours; same palette as mode 17)
	//   24 = cube FACE (+X red, -X green, +Y blue, -Y yellow, +Z magenta, -Z cyan)
	//   25 = per-OCCLUDER unique colour + stripe pattern (thousands of distinct ids; SHIMMERS if the
	//        occluder is camera-relative). Read the problem shadow's colour/pattern off-screen and
	//        match it in the dump.
	bool WantsShadowTint() { return gMode == 23 || gMode == 24 || gMode == 25; }

	// Chaotic hash of a 3D cell -> [0,1)^3, so neighbouring occluder cells get very different colours.
	float3 Hash3(float3 p)
	{
		// standard hash constants (irrational-ish dot weights + 43758.5453 sin scale) — not tuned, don't rename
		p = float3(dot(p, float3(127.1, 311.7, 74.7)),
		           dot(p, float3(269.5, 183.3, 246.1)),
		           dot(p, float3(113.5, 271.9, 124.6)));
		return frac(sin(p) * 43758.5453);
	}

	float3 ShadowTint(float3 sceneColor)
	{
		if (!gVSMHit)
			return sceneColor;  // not shadowed by us -> untouched lit scene
		if (gMode == 23)
			return frac(float3(gVSMHitLight * 0.1237, gVSMHitLight * 0.3391, gVSMHitLight * 0.5541) + 0.11);
		if (gMode == 24) {
			float3 faceCol[6] = { float3(1,0,0), float3(0,1,0), float3(0,0,1), float3(1,1,0), float3(1,0,1), float3(0,1,1) };
			return (gVSMHitFace >= 0 && gVSMHitFace < 6) ? faceCol[gVSMHitFace] : float3(1,1,1);
		}
		// Mode 25: unique colour per ~8-unit occluder cell + a stripe pattern (orientation/phase from a
		// second hash) so even near-identical hues are distinguishable across a large number of shadows.
		float3 cell = floor(gVSMHitOcc / 8.0);
		float3 col  = Hash3(cell);
		float3 h2   = Hash3(cell + 17.0);
		float  freq = 0.15 + 0.5 * h2.x;                         // stripe frequency per id
		float  axis = dot(gVSMHitOcc, normalize(h2 - 0.5));      // stripe orientation per id
		float  stripe = 0.65 + 0.35 * step(0.5, frac(axis * freq));
		return saturate(col * stripe);
	}

	float3 DebugColor(float3 P)
	{
		float3 absP = P + FrameBuffer::CameraPosAdjust.xyz;  // true world space (control anchor)
		float3 W    = SampleP(P);                            // projection/match space (honors dropdown)

		// Nearest valid buffer light to this pixel, measured in the SAME space we project in.
		int   best  = -1;
		float bestD = kArgMinInit;
		uint lightCount, lightStride; ShadowLights.GetDimensions(lightCount, lightStride);  // runtime buffer size (NO cap)
		[loop] for (uint i = 0; i < lightCount; ++i) {
			float3 rp = ShadowLights[i].positionWS.xyz;
			if (dot(rp, rp) < kEmptySlotMin)
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
		bool   front    = clip.w > kDivEps;
		float3 ndc      = front ? clip.xyz / clip.w : float3(kOutOfCubeNdc, kOutOfCubeNdc, kOutOfCubeNdc);
		bool   inBounds = abs(ndc.x) <= 1 && abs(ndc.y) <= 1 && ndc.z >= 0 && ndc.z <= 1;
		float2 faceUV   = ndc.xy * float2(0.5, -0.5) + 0.5;
		float2 tile     = float2(face % 3, face / 3);
		// RUNTIME atlas dims (b13 cbuffer), matching the real GetLocalShadow path — NOT the compile-time
		// kBlockW/kFaceRes/kAtlasW/kAtlasH, which are wrong once iShadowMapResolution != 256.
		float  faceRes  = L.positionWS.w;  // per-light cube-face resolution (was global gFaceRes)
		float2 px       = float2(L.atlasX, L.atlasY) + (tile + faceUV) * faceRes;  // atlasX/Y = block pixel origin
		float2 atlasUV  = px / float2(gAtlasW, gAtlasH);
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
		if (gMode == 15) return (linPix - linOcc > gBiasScale) ? float3(1,0,0) : float3(0,1,0);  // legacy rough decision vs gBiasScale (NOT the calculated-bias path); red=shadow
		if (gMode == 16) return bestD < kDebugMatchThresh ? float3(0,1,0) : float3(1, saturate(bestD / 500.0), 0);  // LEGACY match viz (green=within threshold of nearest light)
		if (gMode == 17) return frac(float3(best * 0.1237, best * 0.3391, best * 0.5541) + 0.11);      // matched light index color
		if (gMode == 19) return (occluder > 0.0001) ? float3(0,1,0) : float3(0.2,0,0);  // atlas populated where we sample? reverse-Z: green=depth>0 (empty=0)
		if (gMode == 20) return (occluder - ndc.z > 0.0) ? float3(1,0,0) : float3(0,1,0);  // raw-depth shadow decision (no linearize); reverse-Z
		return float3(0,0,0);
	}
}
#endif
