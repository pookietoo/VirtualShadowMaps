#include "VSMConfig.h"

namespace vsm
{
	Config& GetConfig()
	{
		static Config instance;
		return instance;
	}

	// Deployment build: there is NO user-facing configuration file. Every setting is hardwired to its
	// VSMConfig.h default — the feature is always on (uninstall the mod to disable it), the strictly-better
	// modules are compiled in, and the experimental ones stay off. The runtime TOML (module toggles,
	// per-shape overrides, live-editable from the menu) exists only in the dev build, which has its own
	// copy of this file. Load()/Save() are no-ops here so nothing is written to the user's Data folder.
	void Config::Load() {}
	void Config::Save() const {}
}
