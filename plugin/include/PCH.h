#pragma once

// CommonLibSSE-NG + SKSE
#include <RE/Skyrim.h>
#include <REL/Relocation.h>
#include <SKSE/SKSE.h>

// std
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// D3D11
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

// math + logging
#include <DirectXMath.h>
#include <spdlog/spdlog.h>

using namespace std::literals;
namespace logger = SKSE::log;

template <class T>
using ComPtr = Microsoft::WRL::ComPtr<T>;
