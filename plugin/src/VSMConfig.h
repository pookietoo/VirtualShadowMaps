#pragma once

// ============================================================================
// Runtime configuration for Virtual Shadow Maps, persisted to
//   Data/SKSE/Plugins/VirtualShadowMaps.toml
//
// Only values that are SAFE to change at runtime live here — they feed the b13 tuning
// cbuffer or the CPU-side caster/light gathering. The atlas GEOMETRY (face resolution,
// light cap) stays compile-time in VSMConstants.h because it's baked into the shipped
// shader + GPU buffer sizing; Load() validates the file's [atlas] section against those
// compiled values and warns on mismatch rather than silently (and wrongly) applying it.
// ============================================================================
namespace vsm
{
	struct Config
	{
		// [general]
		bool  enabled     = false;  // start with the feature OFF

		// Read the .toml (writing a documented default file if none exists) and validate the
		// [atlas] section against the compiled constants. Safe to call once at startup.
		void Load();
		// Write the current values back to the .toml (called when the menu changes them).
		void Save() const;
	};

	Config& GetConfig();  // process-wide singleton
}
