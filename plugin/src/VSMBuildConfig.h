#pragma once

// ============================================================================
// Development vs deployment build toggle.
//
//   VSM_DIAGNOSTICS = 0  (deployment, DEFAULT): compiles the diagnostics / logging / debug apparatus OUT
//                        — VsmDiagnostics (dumps, probes, preview), the debug menu controls, the real-shader
//                        pixel probe, and verbose (info-level) logging are all gone. Core rendering + a
//                        minimal menu + ONLY error-level logging remain. This is what ships / is checked in.
//   VSM_DIAGNOSTICS = 1  (development): compiles ALL of that back in. Opt in locally with -DVSM_DEV_BUILD=ON,
//                        which also re-adds VirtualShadowMaps_Diagnostics.cpp to the build.
//
// CMake always defines this explicitly (both branches). The fallback below only applies to a standalone /
// tooling compile that bypasses CMake — and it too defaults to deployment, per "shipping is the default".
// ============================================================================
#ifndef VSM_DIAGNOSTICS
    #define VSM_DIAGNOSTICS 0
#endif

// Informational logging is DEVELOPMENT-ONLY. Errors are ALWAYS retained (use logger::error directly) so
// a deployment user can still diagnose a real failure. Use VSM_LOG(...) for anything info/debug-level.
#if VSM_DIAGNOSTICS
    #define VSM_LOG(...) logger::info(__VA_ARGS__)
#else
    #define VSM_LOG(...) ((void)0)
#endif
