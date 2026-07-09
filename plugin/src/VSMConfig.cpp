#include "VSMConfig.h"
#include "VSMConstants.h"
#include "VSMBuildConfig.h"  // VSM_LOG (dev-only info logging)

#include <toml++/toml.hpp>

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <system_error>

namespace
{
	// Relative to the game root; under MO2 this resolves through the virtual file system
	// (writes land in the active profile's overwrite, which is what we want).
	constexpr const char* kConfigPath    = "Data/SKSE/Plugins/VirtualShadowMaps.toml";
	constexpr const char* kConfigDir     = "Data/SKSE/Plugins";

	// Serialize a Config to documented TOML text (comments + current values). Written by Save()
	// and used for the auto-generated default so the options are discoverable in the file itself.
	std::string ToToml(const vsm::Config& c)
	{
		return std::format(
		    "# Virtual Shadow Maps configuration.\n"
		    "# Runtime tunables — also editable live from the in-game menu, which saves back here.\n"
		    "\n"
		    "[general]\n"
		    "enabled = {}\n"
		    "\n"
		    "# [atlas] is COMPILE-TIME (baked into the shipped shader + GPU buffer sizing). Shown for\n"
		    "# reference only; changing it here has NO effect until the plugin is rebuilt with new\n"
		    "# values in VSMConstants.h.\n"
		    "[atlas]\n"
		    "faceResolution = {}\n"
		    "maxLights      = {}\n",
		    c.enabled,
		    vsm::kFaceRes, vsm::kMaxLights);
	}
}

namespace vsm
{
	Config& GetConfig()
	{
		static Config instance;
		return instance;
	}

	void Config::Load()
	{
		std::error_code ec;
		if (!std::filesystem::exists(kConfigPath, ec)) {
			Save();  // write a documented default so the options are discoverable
			VSM_LOG("VSM config: no file found; wrote default {}", kConfigPath);
			return;
		}

		toml::table tbl;
		try {
			tbl = toml::parse_file(kConfigPath);
		} catch (const toml::parse_error& e) {
			logger::warn("VSM config: parse failed ({}); keeping defaults", e.description());
			return;
		}

		enabled = tbl["general"]["enabled"].value_or(enabled);

		// Atlas geometry is compile-time; warn (don't apply) if the file disagrees.
		const int faceRes   = tbl["atlas"]["faceResolution"].value_or(kFaceRes);
		const int maxLights = tbl["atlas"]["maxLights"].value_or(kMaxLights);
		if (faceRes != kFaceRes || maxLights != kMaxLights)
			logger::warn("VSM config: [atlas] faceResolution/maxLights ({}/{}) differ from the compiled "
			             "shader ({}/{}); atlas geometry is compile-time, so the file values are ignored "
			             "until the plugin is rebuilt.",
			    faceRes, maxLights, kFaceRes, kMaxLights);

		VSM_LOG("VSM config: loaded {} (enabled={})", kConfigPath, enabled);
	}

	void Config::Save() const
	{
		std::error_code ec;
		std::filesystem::create_directories(kConfigDir, ec);
		std::ofstream file(kConfigPath, std::ios::trunc);
		if (!file) {
			logger::warn("VSM config: cannot open {} for writing", kConfigPath);
			return;
		}
		file << ToToml(*this);
	}
}
