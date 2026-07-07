#pragma once

// ============================================================================
// Virtual Shadow Maps — Community Shaders feature (M0).
//
// Standalone-DLL testing showed CS has no add-on API, so for menu integration we
// build as a CS feature: appears under Lighting in the CS menu (END) with a live
// depth-map preview. Only edit to existing CS code is one registration line in
// Feature.cpp; everything else is these new files.
//
// M0: render our OWN depth map for one local shadow light from a minimal
// scene-graph registry, and validate it via the in-menu preview. No VSM machinery
// yet (no atlas/paging/caching/meshlets) — those are later milestones.
// ============================================================================

#include "Buffer.h"   // CS: Texture2D, ConstantBuffer, ConstantBufferDesc<T>
#include "Feature.h"  // CS: Feature base, FeatureCategories

#include <DirectXMath.h>

struct VirtualShadowMaps : Feature
{
	static VirtualShadowMaps* GetSingleton()
	{
		static VirtualShadowMaps singleton;
		return &singleton;
	}

	std::string GetName() override { return "Virtual Shadow Maps"; }
	std::string GetShortName() override { return "VirtualShadowMaps"; }
	std::string_view GetCategory() const override { return FeatureCategories::kLighting; }
	bool SupportsVR() override { return false; }

	void SetupResources() override;
	void Prepass() override;
	void DrawSettings() override;
	void LoadSettings(json& o_json) override;
	void SaveSettings(json& o_json) override;
	void Reset() override;

	struct Settings
	{
		bool  Enabled      = false;
		int   Resolution   = 2048;
		float CasterRadius = 0.0f;   // 0 = gather all (M0)
		float PreviewScale = 0.25f;  // depth-map preview scale in the menu
	};
	Settings settings;

private:
	struct CasterEntry
	{
		ID3D11Buffer*       vertexBuffer  = nullptr;
		ID3D11Buffer*       indexBuffer   = nullptr;
		uint32_t            vertexStride  = 0;
		bool                fullPrecision = false;
		uint32_t            indexCount    = 0;
		DirectX::XMFLOAT4X4 world         = {};
	};

	bool PickTestLight(RE::ShadowSceneNode* a_ssn);
	void CollectCasters(RE::NiAVObject* a_obj, float a_radius);
	void RenderDepth();

	std::unique_ptr<Texture2D>       depthMap;
	std::unique_ptr<ConstantBuffer>  perDrawCB;
	winrt::com_ptr<ID3D11VertexShader>      depthVS;
	winrt::com_ptr<ID3D11InputLayout>       ilFull;    // POSITION R32G32B32_FLOAT
	winrt::com_ptr<ID3D11InputLayout>       ilPacked;  // POSITION R16G16B16A16_FLOAT (xyz)
	winrt::com_ptr<ID3D11DepthStencilState> depthState;
	winrt::com_ptr<ID3D11RasterizerState>   rasterState;

	std::vector<CasterEntry> registry;
	DirectX::XMFLOAT4X4      lightViewProj = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
	bool haveTestLight  = false;
	bool resourcesReady = false;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    VirtualShadowMaps::Settings, Enabled, Resolution, CasterRadius, PreviewScale)
