cmake_minimum_required(VERSION 3.21)
project(VirtualShadowMaps VERSION 0.1.0 LANGUAGES CXX)

# CommonLibSSE-NG is vendored inside the project (self-contained; relative path).
set(CommonLibPath "${CMAKE_CURRENT_SOURCE_DIR}/../extern/CommonLibSSE-NG")

add_compile_definitions(SKYRIM)
add_library(${PROJECT_NAME} SHARED)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp" "src/*.h")
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})

configure_file(cmake/Plugin.h.in ${CMAKE_CURRENT_BINARY_DIR}/cmake/Plugin.h @ONLY)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/cmake)

target_precompile_headers(${PROJECT_NAME} PRIVATE include/PCH.h)

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /MP /W4 /permissive- /EHsc /bigobj
        /Zc:preprocessor /Zc:__cplusplus)
endif()

# CommonLibSSE-NG (built as a subproject; no tests)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(${CommonLibPath} CommonLibSSE EXCLUDE_FROM_ALL)

find_package(spdlog CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)   # must match CS's imgui (1.92.6 + docking) for shared context

target_link_libraries(${PROJECT_NAME} PRIVATE
    CommonLibSSE::CommonLibSSE
    spdlog::spdlog
    imgui::imgui
    d3d11 dxgi d3dcompiler)

# ---- Deploy into the LOCAL mod folder (dist/) -------------------------------------------------
# dist/ is a local, git-ignored MO2-ready tree — it is NOT published to GitHub. Layout:
#   SKSE/Plugins/VirtualShadowMaps.dll   <- this target (auto-copied below)
#   SKSE/Plugins/CommunityShaders.dll    <- the minimally-hooked CS build (maintained separately)
#   Shaders/...                          <- copied from the tracked /Shaders source tree
# Every build refreshes the DLL and the shader overrides so dist/ mirrors the current source.
set(VSM_MOD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dist/Virtual Shadow Maps")
set(VSM_MOD_PLUGINS "${VSM_MOD_DIR}/SKSE/Plugins")
set(VSM_SHADER_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../Shaders")
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${VSM_MOD_PLUGINS}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${PROJECT_NAME}>" "${VSM_MOD_PLUGINS}/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${VSM_SHADER_SRC}" "${VSM_MOD_DIR}/Shaders"
    VERBATIM
    COMMENT "Deploying VirtualShadowMaps.dll + shaders -> dist/Virtual Shadow Maps/")
