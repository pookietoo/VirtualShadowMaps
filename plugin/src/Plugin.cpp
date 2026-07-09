#include "VirtualShadowMaps.h"
#include "Plugin.h"

#include <imgui.h>
#include <spdlog/sinks/basic_file_sink.h>

#define DLLEXPORT __declspec(dllexport)

// Exposes our shadow atlas + per-light buffer + sampler to Community Shaders'
// LightLimitFix::Prepass, which binds them (t110/t111/s7) so the lighting shader can sample
// our shadows.
extern "C" DLLEXPORT void VSM_GetShadowResources(
    ID3D11ShaderResourceView** a_atlas,
    ID3D11ShaderResourceView** a_lightBuffer,
    ID3D11SamplerState**       a_sampler,
    ID3D11Buffer**             a_debugCB,
    int*                       a_lightCount,
    bool*                      a_enabled)
{
	auto* vsm = VirtualShadowMaps::GetSingleton();
	vsm->NoteResourceFetch();  // record the CS<->plugin handshake for the diagnostic dump
	if (a_atlas)
		*a_atlas = vsm->GetAtlasSRV();
	if (a_lightBuffer)
		*a_lightBuffer = vsm->GetLightBufferSRV();
	if (a_sampler)
		*a_sampler = vsm->GetPointSampler();
	if (a_debugCB)
		*a_debugCB = vsm->GetDebugCB();
	if (a_lightCount)
		*a_lightCount = vsm->GetShadowLightCount();
	if (a_enabled)
		*a_enabled = vsm->IsEnabled();
}

namespace
{
	// ---- IDXGISwapChain::Present hook (vtable index 8) — our per-frame tick ----
	using PresentFn = HRESULT(WINAPI*)(IDXGISwapChain*, UINT, UINT);
	PresentFn g_origPresent = nullptr;

#if VSM_DIAGNOSTICS
	// ---- ID3D11DeviceContext::OMSetRenderTargets hook (vtable index 33) — binds our pixel-probe UAV
	// at u8 alongside whatever render targets a pass sets, ONLY while the probe is armed. This lets
	// the real Lighting.hlsl write its per-pixel probe. When disarmed it's a pure passthrough. ----
	using OMSetRTsFn     = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*);
	using OMSetRTsUAVsFn = void(STDMETHODCALLTYPE*)(ID3D11DeviceContext*, UINT, ID3D11RenderTargetView* const*, ID3D11DepthStencilView*, UINT, UINT, ID3D11UnorderedAccessView* const*, const UINT*);
	OMSetRTsFn     g_origOMSetRTs     = nullptr;
	OMSetRTsUAVsFn g_origOMSetRTsUAVs = nullptr;

	void STDMETHODCALLTYPE HookedOMSetRenderTargets(ID3D11DeviceContext* a_ctx, UINT a_num,
	    ID3D11RenderTargetView* const* a_rtvs, ID3D11DepthStencilView* a_dsv)
	{
		auto* vsm = VirtualShadowMaps::GetSingleton();
		// Only alter state while armed. The probe UAV lives at u8 (render-target outputs occupy u0..u7
		// in the shared PS OM namespace). Bind it just above the pass's render targets; passes whose PS
		// doesn't declare u8 simply ignore it, so this is inert for them. Needs FL11.1 at runtime.
		if (vsm->IsProbeArmed() && g_origOMSetRTsUAVs && a_num >= 1 && a_num <= 8) {
			if (auto* uav = vsm->GetPixelProbeUAV()) {
				ID3D11UnorderedAccessView* u = uav;
				g_origOMSetRTsUAVs(a_ctx, a_num, a_rtvs, a_dsv, 8, 1, &u, nullptr);
				return;
			}
		}
		g_origOMSetRTs(a_ctx, a_num, a_rtvs, a_dsv);
	}

	void InstallContextHook(ID3D11DeviceContext* a_ctx)
	{
		if (!a_ctx)
			return;
		void** vtbl = *reinterpret_cast<void***>(a_ctx);
		g_origOMSetRTsUAVs = reinterpret_cast<OMSetRTsUAVsFn>(vtbl[34]);  // OMSetRenderTargetsAndUnorderedAccessViews
		DWORD oldProtect = 0;
		VirtualProtect(&vtbl[33], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect);
		g_origOMSetRTs = reinterpret_cast<OMSetRTsFn>(vtbl[33]);
		vtbl[33] = reinterpret_cast<void*>(&HookedOMSetRenderTargets);
		VirtualProtect(&vtbl[33], sizeof(void*), oldProtect, &oldProtect);
		VSM_LOG("VSM: OMSetRenderTargets hook installed (real-shader pixel probe, u8)");
	}
#endif  // VSM_DIAGNOSTICS

	// Draw callback CS invokes inside its menu (registered via CS_RegisterExternalMenu).
	void ExternalDrawMenu()
	{
		VirtualShadowMaps::GetSingleton()->DrawMenu();
	}

	// Register our menu with Community Shaders once its ImGui context exists. CS
	// exposes a tiny add-on hook: CS_GetImGui (share context + allocators) and
	// CS_RegisterExternalMenu (register our draw callback). Retried each frame until
	// CS is loaded and its context is created. Our logic never lives in CS's DLL.
	void TryRegisterCSMenu()
	{
		static bool done = false;
		if (done)
			return;
		auto cs = GetModuleHandleA("CommunityShaders.dll");
		if (!cs)
			return;
		auto getImGui = reinterpret_cast<void (*)(ImGuiContext**, ImGuiMemAllocFunc*, ImGuiMemFreeFunc*, void**)>(
		    GetProcAddress(cs, "CS_GetImGui"));
		auto reg = reinterpret_cast<void (*)(void (*)())>(GetProcAddress(cs, "CS_RegisterExternalMenu"));
		if (!getImGui || !reg)
			return;  // CS build lacks the add-on hook
		ImGuiContext* ctx = nullptr;
		ImGuiMemAllocFunc alloc = nullptr;
		ImGuiMemFreeFunc free = nullptr;
		void* ud = nullptr;
		getImGui(&ctx, &alloc, &free, &ud);
		if (!ctx)
			return;  // CS menu not initialized yet; retry next frame
		ImGui::SetCurrentContext(ctx);
		ImGui::SetAllocatorFunctions(alloc, free, ud);
		reg(&ExternalDrawMenu);
		done = true;
		VSM_LOG("VSM: registered menu with Community Shaders");
	}

	HRESULT WINAPI HookedPresent(IDXGISwapChain* a_this, UINT a_sync, UINT a_flags)
	{
		TryRegisterCSMenu();
		VirtualShadowMaps::GetSingleton()->RenderFrame();
		return g_origPresent(a_this, a_sync, a_flags);
	}

	void InstallPresentHook(IDXGISwapChain* a_swapChain)
	{
		if (!a_swapChain)
			return;
		void** vtbl = *reinterpret_cast<void***>(a_swapChain);
		DWORD oldProtect = 0;
		VirtualProtect(&vtbl[8], sizeof(void*), PAGE_EXECUTE_READWRITE, &oldProtect);
		g_origPresent = reinterpret_cast<PresentFn>(vtbl[8]);
		vtbl[8] = reinterpret_cast<void*>(&HookedPresent);
		VirtualProtect(&vtbl[8], sizeof(void*), oldProtect, &oldProtect);
		VSM_LOG("VSM: Present hook installed");
	}

	void InitializeLog()
	{
		auto path = logger::log_directory();
		if (!path)
			return;
		*path /= std::format("{}.log", Plugin::NAME);
		auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path->string(), true);
		auto log = std::make_shared<spdlog::logger>("global log", std::move(sink));
		log->set_level(spdlog::level::info);
		log->flush_on(spdlog::level::info);
		spdlog::set_default_logger(std::move(log));
		spdlog::set_pattern("[%H:%M:%S.%e] [%l] [%s:%#] %v");
	}

	void MessageHandler(SKSE::MessagingInterface::Message* a_msg)
	{
		if (a_msg->type != SKSE::MessagingInterface::kDataLoaded)
			return;

		auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
		if (!renderer) {
			logger::error("VSM: no BSGraphics::Renderer at kDataLoaded");
			return;
		}
		auto& rd = renderer->GetRuntimeData();
		auto* device = reinterpret_cast<ID3D11Device*>(rd.forwarder);
		auto* context = reinterpret_cast<ID3D11DeviceContext*>(rd.context);
		auto* swapChain = reinterpret_cast<IDXGISwapChain*>(rd.renderWindows->swapChain);
		VSM_LOG("VSM: D3D device={} context={} swapChain={}",
		    (void*)device, (void*)context, (void*)swapChain);

		if (swapChain) {
			DXGI_SWAP_CHAIN_DESC scd{};
			if (SUCCEEDED(swapChain->GetDesc(&scd))) {
				VirtualShadowMaps::GetSingleton()->SetRenderSize(scd.BufferDesc.Width, scd.BufferDesc.Height);
				VSM_LOG("VSM: render size {}x{} (pixel-probe target = centre)", scd.BufferDesc.Width, scd.BufferDesc.Height);
			}
		}

		VirtualShadowMaps::GetSingleton()->OnD3DReady(device, context);
		InstallPresentHook(swapChain);
#if VSM_DIAGNOSTICS
		InstallContextHook(context);  // for the real-shader pixel probe (only acts while armed)
#endif

		VSM_LOG("VSM: ready. Settings appear in the Community Shaders menu under 'Virtual Shadow Maps'.");
	}
}

extern "C" DLLEXPORT bool SKSEAPI SKSEPlugin_Load(const SKSE::LoadInterface* a_skse)
{
	InitializeLog();
	VSM_LOG("Loaded {} {}", Plugin::NAME, Plugin::VERSION);
	SKSE::Init(a_skse);
	SKSE::AllocTrampoline(1 << 10);
	SKSE::GetMessagingInterface()->RegisterListener("SKSE", MessageHandler);
	return true;
}

extern "C" DLLEXPORT constinit auto SKSEPlugin_Version = []() noexcept {
	SKSE::PluginVersionData v;
	v.PluginName(Plugin::NAME.data());
	v.PluginVersion(REL::Version(Plugin::VERSION_MAJOR, Plugin::VERSION_MINOR, Plugin::VERSION_PATCH));
	v.UsesAddressLibrary();
	v.UsesNoStructs();
	return v;
}();

extern "C" DLLEXPORT bool SKSEAPI SKSEPlugin_Query(const SKSE::QueryInterface*, SKSE::PluginInfo* a_info)
{
	a_info->name = SKSEPlugin_Version.pluginName;
	a_info->infoVersion = SKSE::PluginInfo::kVersion;
	a_info->version = SKSEPlugin_Version.pluginVersion;
	return true;
}
