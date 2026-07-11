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
	// Serialize a pattern list as a TOML array of LITERAL (single-quoted) strings, so backslashes in
	// model paths (e.g. '\effects\') are taken verbatim — a TOML basic ("...") string would treat '\' as
	// an escape. Patterns must not contain a single quote (none of ours do).
	std::string ToTomlArray(const std::vector<std::string>& v)
	{
		std::string o = "[";
		for (size_t i = 0; i < v.size(); ++i) {
			if (i)
				o += ", ";
			o += '\'';
			o += v[i];
			o += '\'';
		}
		o += "]";
		return o;
	}

	std::string ToToml(const vsm::Config& c)
	{
		return std::format(
		    "# Virtual Shadow Maps configuration.\n"
		    "# Runtime tunables — also editable live from the in-game menu, which saves back here.\n"
		    "\n"
		    "[general]\n"
		    "enabled = {}\n"
		    "# Shadow-resolution quality: target shadow texels per screen pixel for the variable-resolution\n"
		    "# level picker. 1 = critically sampled; >1 sharper (more VRAM); <1 softer/cheaper. Default 1.\n"
		    "qualityFactor = {}\n"
		    "# Performance: cache static-geometry shadow depth and re-render only dynamic casters (actors, doors)\n"
		    "# each frame — a large win in static scenes. Default false (module under validation).\n"
		    "cacheStaticShadows = {}\n"
		    "# Performance: cull shadow lights that are behind the camera or entirely beyond fShadowDistance\n"
		    "# (they cast no visible shadow). Conservative. Default false (module under validation).\n"
		    "cullCasters = {}\n"
		    "# Performance (P4, requires cacheStaticShadows): per-light cache invalidation + budgeted re-bakes —\n"
		    "# only the light whose static occluders changed re-renders its block, amortized across frames.\n"
		    "# Default false (module under validation).\n"
		    "incrementalCache = {}\n"
		    "# Colored translucent shadows: glass / alpha-blended casters dim and tint the light passing through\n"
		    "# them (instead of being excluded). Adds a transmittance atlas. Default false (module under validation).\n"
		    "translucentShadows = {}\n"
		    "# Performance (O4, default false): skip a light's entire 6-face caster pass when NO caster lies within its\n"
		    "# radius (CPU sphere-vs-radius pre-check). Off-by-default optimization, UNTESTED in-game.\n"
		    "cullEmptyLightPasses = {}\n"
		    "# Alpha-tested cutout shadows (A6, default false): foliage / grates / chain / hair sample their diffuse alpha\n"
		    "# and clip() in the depth pass, casting punched-through silhouettes instead of solid quads. Fail-safe.\n"
		    "alphaTestedShadows = {}\n"
		    "\n"
		    "# Per-shape shadow-caster overrides. Match the caster's model NIF path (as a substring — use path\n"
		    "# fragments like '\\effects\\', '\\magic\\', '\\lod\\') and/or its shape name (WHOLE-TOKEN: a name is\n"
		    "# split on non-letters and camelCase, so 'marker' matches 'EditorMarker' but NOT 'Market'/'Moonlight').\n"
		    "# Case-insensitive; use single-quoted strings so backslashes are literal. forceCast BEATS every\n"
		    "# built-in exclusion (billboards, alpha/effect glass, decals, the kCastShadows-off flag); forceNoCast\n"
		    "# beats casting; forceCast wins a conflict. Empty (default) = today's behavior exactly.\n"
		    "# Examples (commented candidates — verify against your load order before enabling):\n"
		    "#   forceNoCast = ['\\effects\\', '\\magic\\', '\\lod\\', 'xmarker', 'editormarker', '1stperson']\n"
		    "#   forceCast   = ['glowingmushroom']\n"
		    "[classification]\n"
		    "forceCast   = {}\n"
		    "forceNoCast = {}\n"
		    "\n"
		    "# [atlas] is COMPILE-TIME (baked into the shipped shader + GPU buffer sizing). Shown for\n"
		    "# reference only; changing it here has NO effect until the plugin is rebuilt with new\n"
		    "# values in VSMConstants.h.\n"
		    "[atlas]\n"
		    "faceResolution = {}\n"
		    "maxLights      = {}\n",
		    c.enabled,
		    c.qualityFactor,
		    c.cacheStaticShadows,
		    c.cullCasters,
		    c.incrementalCache,
		    c.translucentShadows,
		    c.cullEmptyLightPasses,
		    c.alphaTestedShadows,
		    ToTomlArray(c.forceCast),
		    ToTomlArray(c.forceNoCast),
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

		enabled       = tbl["general"]["enabled"].value_or(enabled);
		qualityFactor      = tbl["general"]["qualityFactor"].value_or(qualityFactor);
		cacheStaticShadows = tbl["general"]["cacheStaticShadows"].value_or(cacheStaticShadows);
		cullCasters        = tbl["general"]["cullCasters"].value_or(cullCasters);
		incrementalCache   = tbl["general"]["incrementalCache"].value_or(incrementalCache);
		translucentShadows = tbl["general"]["translucentShadows"].value_or(translucentShadows);
		cullEmptyLightPasses = tbl["general"]["cullEmptyLightPasses"].value_or(cullEmptyLightPasses);
		alphaTestedShadows   = tbl["general"]["alphaTestedShadows"].value_or(alphaTestedShadows);

		// [classification] pattern lists (arrays of strings). Absent section => leave empty (no override).
		forceCast.clear();
		forceNoCast.clear();
		if (auto* arr = tbl["classification"]["forceCast"].as_array())
			for (auto& el : *arr)
				if (auto s = el.value<std::string>())
					forceCast.push_back(*s);
		if (auto* arr = tbl["classification"]["forceNoCast"].as_array())
			for (auto& el : *arr)
				if (auto s = el.value<std::string>())
					forceNoCast.push_back(*s);

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
