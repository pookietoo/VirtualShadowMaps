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
    ID3D11Buffer**             a_paramsCB,
    int*                       a_lightCount,
    bool*                      a_enabled)
{
	auto* vsm = VirtualShadowMaps::GetSingleton();
	vsm->NotifyResourcesFetched();  // N3: LLF is sampling our atlas this frame -> keep rendering (else RenderFrame idles)
	if (a_atlas)
		*a_atlas = vsm->GetAtlasSRV();
	if (a_lightBuffer)
		*a_lightBuffer = vsm->GetLightBufferSRV();
	if (a_sampler)
		*a_sampler = vsm->GetPointSampler();
	if (a_paramsCB)
		*a_paramsCB = vsm->GetParamsCB();
	if (a_lightCount)
		*a_lightCount = vsm->GetShadowLightCount();
	if (a_enabled)
		*a_enabled = vsm->IsEnabled();
}

// A5 (colored translucent shadows): the transmittance atlas SRV, bound by LightLimitFix::Prepass at t112. Kept as a
// SEPARATE export (not appended to VSM_GetShadowResources) so the original 6-arg contract is unchanged — an older CS
// that never calls this simply doesn't bind t112, and there is no calling-convention mismatch. Always non-null once
// resources are ready (the atlas is cleared white when the A5 module is off, so binding it is a no-op then).
extern "C" DLLEXPORT void VSM_GetTransmittanceResource(ID3D11ShaderResourceView** a_transAtlas)
{
	if (a_transAtlas)
		*a_transAtlas = VirtualShadowMaps::GetSingleton()->GetTransmittanceSRV();
}

// LLF light-index alignment (Option c): map a BSLight* to its slot in our ShadowLights (t111) buffer, so LLF can
// stamp that index into each Light struct (Lighting.hlsl then indexes our shadow record directly — O(1) — instead
// of the shader searching all lights per pixel). Returns 0xFFFFFFFF (VirtualShadowMaps::kNoShadowIndex) when the
// light has no shadow record this frame; LLF writes that sentinel and the shader renders the light unshadowed
// (fail-safe). SEPARATE additive export: an older CS that never calls it is unaffected. See docs/ARCHITECTURE §6.1.
extern "C" DLLEXPORT std::uint32_t VSM_GetLightShadowIndex(const void* a_bsLight)
{
	return VirtualShadowMaps::GetSingleton()->GetLightShadowIndex(a_bsLight);
}

// ABI/version handshake for the forked Community Shaders build (see vsm::kABIVersion). Lets a VSM-aware CS read
// our shared-layout id and refuse to bind our resources unless it matches the value CS was built against — so a
// mismatched plugin/CS pair falls back to "no VSM shadows" (safe) instead of silently rendering WRONG shadows.
// Additive, optional export: an older CS that never calls it is unaffected. Enforcement lives in CS (LightLimitFix
// Prepass); this half only publishes the id.
extern "C" DLLEXPORT std::uint32_t VSM_GetABIVersion()
{
	return vsm::kABIVersion;
}

namespace
{
	// ---- IDXGISwapChain::Present hook (vtable index 8) — our per-frame tick ----
	using PresentFn = HRESULT(WINAPI*)(IDXGISwapChain*, UINT, UINT);
	PresentFn g_origPresent = nullptr;


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
		auto reg = reinterpret_cast<void (*)(const char*, const char*, void (*)())>(GetProcAddress(cs, "CS_RegisterExternalMenu"));
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
		// Deployment build: list under the CS "Lighting" category beside Light Limit Fix, our host feature.
		reg("Virtual Shadow Maps", "Lighting", &ExternalDrawMenu);
		done = true;
	}

	// Kept out of SafeRenderFrame so that function has NO C++ objects needing unwind (a hard requirement for __try).
	void LogRenderFrameFault()
	{
		static bool logged = false;
		if (!logged) {
			logged = true;
			logger::error("VSM: exception in RenderFrame — shadow rendering skipped this frame (further occurrences suppressed)");
		}
	}

	// SEH net around the per-frame render. RenderFrame walks live game pointers (lights, geometry renderer buffers)
	// that can dangle on cell-transition / streaming frames; RebuildRegistry and skinning are already __try-guarded,
	// this catches the rest so an access violation can't unwind out of Present into the game's render thread (a CTD).
	// On a caught fault this frame's shadows are simply skipped (~GraphicsStateGuard may not run, so device state is
	// corrected on the next good frame). NOTE: __try requires this function to hold no unwinding C++ objects.
	void SafeRenderFrame()
	{
		__try {
			VirtualShadowMaps::GetSingleton()->RenderFrame();
		} __except (EXCEPTION_EXECUTE_HANDLER) {
			LogRenderFrameFault();
		}
	}

	HRESULT WINAPI HookedPresent(IDXGISwapChain* a_this, UINT a_sync, UINT a_flags)
	{
		TryRegisterCSMenu();
		SafeRenderFrame();
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

		// Runtime gate: VSM is built and tested only for Skyrim SE/AE 1.6.x and depends on a matching (AE-only)
		// Community Shaders + Light Limit Fix build. Skyrim VR and pre-AE (1.5.x) runtimes have different
		// renderer/light struct layouts and are unsupported — refuse cleanly here instead of dereferencing those
		// layouts and crashing. (SKSE would otherwise load us on any runtime; UsesNoStructs advertises no gate.)
		if (REL::Module::IsVR() || !REL::Module::IsAE()) {
			logger::warn("VSM: unsupported Skyrim runtime (VR or pre-AE) — Virtual Shadow Maps is disabled.");
			return;
		}

		auto* renderer = RE::BSGraphics::Renderer::GetSingleton();
		if (!renderer) {
			logger::error("VSM: no BSGraphics::Renderer at kDataLoaded");
			return;
		}
		auto& rd = renderer->GetRuntimeData();
		auto* device = reinterpret_cast<ID3D11Device*>(rd.forwarder);
		auto* context = reinterpret_cast<ID3D11DeviceContext*>(rd.context);
		auto* swapChain = rd.renderWindows ? reinterpret_cast<IDXGISwapChain*>(rd.renderWindows->swapChain) : nullptr;
		if (!device || !context || !swapChain) {
			logger::error("VSM: renderer not ready (device/context/swapchain null) — Virtual Shadow Maps is disabled.");
			return;
		}

		DXGI_SWAP_CHAIN_DESC scd{};
		if (SUCCEEDED(swapChain->GetDesc(&scd)))
			VirtualShadowMaps::GetSingleton()->SetRenderSize(scd.BufferDesc.Width, scd.BufferDesc.Height);

		VirtualShadowMaps::GetSingleton()->SetSwapChain(swapChain);  // m3: lets RenderFrame refresh screenH on a resolution change
		VirtualShadowMaps::GetSingleton()->OnD3DReady(device, context);
		InstallPresentHook(swapChain);
	}
}

extern "C" DLLEXPORT bool SKSEAPI SKSEPlugin_Load(const SKSE::LoadInterface* a_skse)
{
	InitializeLog();
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
