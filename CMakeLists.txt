# Copyright (c) 2026 Advanced Micro Devices, Inc.
# Author: Jeff Daily <jeff.daily@amd.com>
#
# Cross-platform build for YarnBall. Upstream ships only a Visual Studio /
# CUDA solution; this adds a CMake build that targets either NVIDIA CUDA or
# AMD ROCm/HIP from the same .cu sources.
#
#   ROCm/HIP:  -DUSE_HIP=ON  -DCMAKE_HIP_ARCHITECTURES=gfx90a
#   NVIDIA:    (default)      -DCMAKE_CUDA_ARCHITECTURES=86
#
# On HIP the cuda_to_hip.h compat header is force-included on the GPU sources
# (so it precedes GLM) and the .cu files compile as HIP.

cmake_minimum_required(VERSION 3.21)

option(USE_HIP "Build the GPU path with ROCm/HIP instead of NVIDIA CUDA" OFF)

if(USE_HIP)
  project(YarnBall LANGUAGES C CXX HIP)
else()
  project(YarnBall LANGUAGES C CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(KE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/KittenEngine)
set(KE      ${KE_ROOT}/KittenEngine)

# --- third-party -----------------------------------------------------------
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Freetype REQUIRED)
find_package(CLI11 REQUIRED)
find_package(Eigen3 REQUIRED)

if(WIN32)
  # Windows: jsoncpp and assimp are sourced via FetchContent because the vcpkg
  # assimp port transitively depends on polyclipping (sourceforge.net) which
  # is network-blocked on this host, and pkg-config is not the canonical
  # Windows find mechanism for jsoncpp.
  set(CMAKE_TLS_VERIFY OFF)
  include(FetchContent)
  FetchContent_Declare(jsoncpp_fc
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp.git
    GIT_TAG 1.9.6
    GIT_SHALLOW TRUE)
  set(JSONCPP_WITH_TESTS OFF CACHE BOOL "" FORCE)
  set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "" FORCE)
  set(JSONCPP_WITH_CMAKE_PACKAGE ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(jsoncpp_fc)
  set(JSONCPP_INCLUDE_DIRS "")
  set(JSONCPP_LIBRARIES jsoncpp_lib)

  FetchContent_Declare(assimp_fc
    GIT_REPOSITORY https://github.com/assimp/assimp.git
    GIT_TAG v5.3.1
    GIT_SHALLOW TRUE)
  set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
  set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
  set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
  # assimp enables -Werror for itself; clang rejects its memcpy-of-non-trivial
  # type without this override.
  set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(assimp_fc)

  # stb is header-only; locate it via find_path (headers are in vcpkg include/)
  find_path(STB_INCLUDE_DIR stb_image.h)
  if(NOT STB_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find stb headers; install vcpkg stb")
  endif()
else()
  find_package(assimp REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(JSONCPP REQUIRED jsoncpp)
  # stb headers ship as system headers under /usr/include/stb
  find_path(STB_INCLUDE_DIR stb_image.h PATH_SUFFIXES stb)
endif()

# GLM: the ROCm build pins GLM 1.0.x, which detects hipcc via __HIP__ and
# decorates its math __host__ __device__ verbatim. The system GLM 0.9.9.8 has
# no __HIP__ path, so it is used only by the CUDA build (which GLM keys off
# __CUDACC__ itself).
if(USE_HIP)
  # hip::host gives the host .cpp files the HIP runtime headers + the
  # __HIP_PLATFORM_AMD__ define so cuda_to_hip.h can include <hip/hip_runtime.h>
  # and resolve the cuda* aliases there too.
  find_package(hip REQUIRED)
  include(FetchContent)
  FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(glm)
  set(GLM_TARGET glm::glm)
else()
  find_package(glm REQUIRED)
  set(GLM_TARGET glm::glm)
endif()

# Dear ImGui: fetch the upstream sources (core + GLFW/OpenGL3 backends) and
# compile them into the build. The distro libimgui-dev does not ship the
# platform/render backend sources, so a self-contained fetch is the portable
# choice on both CUDA and ROCm hosts.
include(FetchContent)
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.90.1
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)
set(IMGUI_DIR ${imgui_SOURCE_DIR})
set(IMGUI_SRC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp)

# glad (OpenGL loader) generated into third_party/glad1.
set(GLAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/glad1)

# --- engine + simulation library ------------------------------------------
set(ENGINE_CPP
  ${KE}/src/Algo.cpp
  ${KE}/src/ComputeBuffer.cpp
  ${KE}/src/Font.cpp
  ${KE}/src/FrameBuffer.cpp
  ${KE}/src/Gizmos.cpp
  ${KE}/src/KittenAssets.cpp
  ${KE}/src/KittenInit.cpp
  ${KE}/src/KittenPreprocessor.cpp
  ${KE}/src/KittenRendering.cpp
  ${KE}/src/Mesh.cpp
  ${KE}/src/MeshMoments.cpp
  ${KE}/src/Shader.cpp
  ${KE}/src/StopWatch.cpp
  ${KE}/src/Texture.cpp
  ${KE}/src/Timer.cpp
  ${KE}/opt/asa047.cpp
  ${KE}/opt/compass_search.cpp
  ${KE}/opt/math.cpp
  ${KE}/opt/praxis.cpp
  ${KE}/opt/svd/svd.cpp
  ${KE}/opt/toms178.cpp)

set(YARN_CPP
  ${KE_ROOT}/YarnBall/io/fiberMesher.cpp
  ${KE_ROOT}/YarnBall/io/fileExport.cpp
  ${KE_ROOT}/YarnBall/io/jsonBuilder.cpp
  ${KE_ROOT}/YarnBall/io/reader.cpp
  ${KE_ROOT}/YarnBall/io/render.cpp
  ${KE_ROOT}/YarnBall/sim/statistics.cpp
  ${KE_ROOT}/YarnBall/sim/step.cpp)

set(GPU_SRC
  ${KE_ROOT}/YarnBall/YarnBall.cu
  ${KE_ROOT}/YarnBall/sim/collision.cu
  ${KE_ROOT}/YarnBall/sim/cosserat.cu
  ${KE_ROOT}/YarnBall/sim/iteration.cu
  ${KE}/KittenGpuLBVH/lbvh.cu)

add_executable(Gui
  ${KE_ROOT}/main.cpp
  ${ENGINE_CPP}
  ${YARN_CPP}
  ${GPU_SRC}
  ${GLAD_DIR}/src/glad.c
  ${IMGUI_SRC})

if(NOT EIGEN3_INCLUDE_DIR)
  # Modern Eigen3 cmake exports only Eigen3::Eigen target; set the include path
  # from the target for use by sources that still need EIGEN3_INCLUDE_DIR.
  get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
endif()

target_include_directories(Gui PRIVATE
  ${KE_ROOT}
  ${KE}
  ${KE}/includes
  ${KE_ROOT}/YarnBall
  ${GLAD_DIR}/include
  ${STB_INCLUDE_DIR}
  ${IMGUI_DIR}
  ${IMGUI_DIR}/backends
  ${EIGEN3_INCLUDE_DIR}
  ${FREETYPE_INCLUDE_DIRS}
  ${JSONCPP_INCLUDE_DIRS})

if(WIN32)
  set(ASSIMP_TARGET assimp)
else()
  set(ASSIMP_TARGET assimp::assimp)
endif()

target_link_libraries(Gui PRIVATE
  ${GLM_TARGET}
  glfw
  OpenGL::GL
  ${CMAKE_DL_LIBS}
  Freetype::Freetype
  ${ASSIMP_TARGET}
  CLI11::CLI11
  Eigen3::Eigen
  ${JSONCPP_LIBRARIES})

if(WIN32)
  target_compile_definitions(Gui PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

target_compile_definitions(Gui PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLAD)

# --- GPU path configuration ------------------------------------------------
if(USE_HIP)
  if(NOT CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES gfx90a)
  endif()
  set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE HIP)
  set_target_properties(Gui PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  target_compile_definitions(Gui PRIVATE USE_HIP)
  target_link_libraries(Gui PRIVATE hip::host)
  # Force-include the compat header on the project's own sources (both the .cu
  # kernels and the host .cpp that pull Common.h). It brings in the HIP runtime
  # -- which makes __device__/__host__ host no-ops in the .cpp and supplies the
  # device math overloads ahead of GLM in the .cu -- and aliases the cuda*
  # surface to hip*. It is NOT applied to glad.c or the ImGui sources.
  set_source_files_properties(${KE_ROOT}/main.cpp ${ENGINE_CPP} ${YARN_CPP} ${GPU_SRC} PROPERTIES
    COMPILE_OPTIONS "-include;${KE_ROOT}/cuda_to_hip.h")
else()
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  set_source_files_properties(${GPU_SRC} PROPERTIES LANGUAGE CUDA)
  set_target_properties(Gui PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON)
endif()
