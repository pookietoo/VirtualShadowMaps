# M0 Integration & Test Guide

`OwnedShadowTest` is a **standalone Community Shaders Feature**. It touches **one line**
of pre-existing CS code (feature registration); everything else is new files that CMake
auto-globs. No edits to `LightLimitFix`, `Lighting.hlsl`, or `CMakeLists`.

## 1. Drop in the files (copy to the same paths in your `owned-vsm` fork)

| This repo | → Fork path |
|---|---|
| `src/Features/OwnedShadowTest.h`   | `src/Features/OwnedShadowTest.h` |
| `src/Features/OwnedShadowTest.cpp` | `src/Features/OwnedShadowTest.cpp` |
| `package/Shaders/OwnedShadowTest/OwnedShadowDepthVS.hlsl` | same |

CMake picks these up automatically (`cmake/AddCXXFiles.cmake` globs `src/*.cpp`;
`package/Shaders/**` is deployed to `Data\Shaders\`). Re-run `cmake --preset ALL-VS2022`
after adding files so the globs refresh.

## 2. The ONE edit — register the feature

In `src/Feature.cpp`, `Feature::GetFeatureList()` returns a hardcoded list. Add our
singleton (and the include at the top):

```cpp
#include "Features/OwnedShadowTest.h"
// ...
const std::vector<Feature*>& Feature::GetFeatureList()
{
    static std::vector<Feature*> features = {
        // ... existing features ...
        OwnedShadowTest::GetSingleton(),   // <-- add
    };
```

That's the only change to existing code — an addition to a list, the standard way every
CS feature registers.

## 3. Build & deploy

```pwsh
cmake --preset ALL-VS2022
cmake --build --preset ALL-VS2022   # 'Dev' config for fast iteration
cmake --install --preset ALL-VS2022 -- --prefix $MOD_FOLDER
```

## 4. How to test (this is the M0 validation)

1. Launch Skyrim (SKSE). Open the **Community Shaders menu** (default **`END`** key).
2. Find **Owned Shadow Test** under the **Lighting** category. Tick **Enabled**.
3. Load a scene with at least one shadow-casting *local* light (a shadow-casting torch /
   spell / interior lamp — anything that isn't just the sun). `coc` to a controlled
   interior if you built one.
4. **Primary check — the depth-map preview.** The settings panel shows a live image of the
   depth map we rasterized ourselves. **You should see the scene silhouetted from the
   light's point of view** (rendered as R32 depth in the red channel — nearer = one end of
   the range). The "casters: N" counter should be non-zero.
   - This proves *registry → own render* end-to-end (geometry captured, light matrix
     borrowed, our own depth pass rasterized correctly).
5. **Secondary check — RenderDoc.** Capture a frame; find the `OwnedShadowTest::DepthMap`
   resource. Confirm it's a `D32`/`R32` texture, filled during the frame, with recognizable
   geometry from the light's viewpoint. Confirm the engine's own render targets look intact
   after our pass (no corruption elsewhere — validates the state guard).

M0 is **done** when the preview shows correct silhouettes from the light and toggling
Enabled turns it on/off with no frame corruption.

> Not in M0 (kept zero-touch): sampling the map back to darken the actual lit scene. That's
> the integration milestone; it will either feed the existing `t102/t103` sample contract
> (no shader edit) or add a minimal `Lighting.hlsl` include — decided at M3/M4.

## 5. Debug runbook (symptom → likely cause)

| Symptom | Look at |
|---|---|
| **Won't compile** | The VERIFY identifiers in §6 — fix against the real headers. |
| **"casters: 0"** | `PickTestLight` found no light (no local shadow caster in scene, or `shadowCasterLights`/`shadowDirLight` path wrong), or `CollectCasters` traversal (`AsTriShape`/`AsNode`/`GetChildren`). |
| **Preview is blank/all one color** | Depth cleared but nothing drawn: light matrix zero/garbage (`shadowmapDescriptors[0].lightTransform`), or all geometry culled by winding — try `CullMode = D3D11_CULL_NONE`. |
| **Preview geometry looks exploded/scrambled** | Position format: packed vs full mismatch (`VF_FULLPREC`), or a mesh with non-zero POSITION offset. |
| **Preview inverted (near/far swapped)** | Flip `kReverseZ` in `OwnedShadowTest.cpp` (one line). |
| **Other rendering breaks when enabled** | State guard: something our draw pass changed isn't restored — check `GraphicsStateGuard`. |
| **Crash on enable** | `globals::game::smState->shadowSceneNode[0]` path, or a null `rendererData`. |

## 6. VERIFY checklist (resolve on first compile / RenderDoc)

Resolved from CS/CommonLibSSE-NG source (real code written): `NiTransform`/`NiMatrix3` math,
scene traversal (`AsTriShape`/`AsNode`/`GetChildren`), light pick
(`shadowCasterLights` + `shadowmapDescriptors[0].lightTransform`), input-layout blob via
`D3DCompileFromFile`, packed positions via `VF_FULLPREC`, matrix convention (row-vector),
reverse-Z single toggle, settings via `NLOHMANN_*`, ImGui preview.

Additionally confirmed against the real CS tree:
- ✅ `RE::BSGraphics::Vertex::VF_FULLPREC` spelling (enumerators sit directly under `Vertex`,
  per `ShaderCache.cpp:1172`'s `VA_POSITION` usage).
- ✅ `light->GetRuntimeData().shadowmapDescriptors[i].lightTransform` — the exact idiom in
  `Deferred.cpp:566` (not a direct `sl->shadowmapDescriptors` member).
- ✅ `globals::game::smState->shadowSceneNode[0]` — identical to `LightLimitFix.cpp:394`.
- ✅ `FeatureCategories::kLighting`.

- ✅ `ShadowSceneNode::GetRuntimeData().shadowCasterLights` (`BSTArray<BSShadowLight*>`) and
  `.shadowDirLight` (`BSShadowDirectionalLight*`) — confirmed verbatim from the CommonLibSSE-NG
  `ShadowSceneNode.h` header.

**All compile-time identifiers are now resolved — the code should build first try.**
Remaining items are runtime-only, diagnosed from the preview (see the runbook):
1. `kReverseZ` — flip if the preview is inverted.
2. Skyrim triangle winding vs `FrontCounterClockwise = FALSE` — if the map is empty, try
   `CULL_NONE` first, then pick the correct winding.
