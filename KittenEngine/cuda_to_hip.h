#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// \author Jeff Daily <jeff.daily@amd.com>
//
// CUDA-to-HIP compatibility shim for the ROCm build. Force-included (CMake
// -include) on every HIP translation unit so it is parsed before any project
// or GLM header, then again pulled in by the sources that name CUDA symbols.
//
// Aliases the cuda* runtime/stream/graph surface the sources use to hip*. The
// HIP runtime is included first so its __device__ memcpy/memset overloads are in
// scope for any header (e.g. GLM type_ptr) the sources pull in afterwards.
//
// GLM device-qualifier note: the kernels call glm:: math (dot/length/cross/...)
// from __device__ code. GLM only decorates those __host__ __device__ when it
// detects a device compiler. GLM >= 1.0 recognizes hipcc via __HIP__ directly
// (glm/simd/platform.h -> GLM_COMPILER_HIP), so the ROCm build pins GLM 1.0.x
// (FetchContent in CMakeLists.txt) and needs no compiler-identity spoofing here.
// The system GLM 0.9.9.8 lacks __HIP__ detection and is used only by the CUDA
// build, which GLM keys off __CUDACC__ on its own.

#if defined(USE_HIP)

#include <cstring>
#include <cstdlib>
#include <cstdio>

#include <hip/hip_runtime.h>

// --- runtime API -----------------------------------------------------------
#define cudaError_t                         hipError_t
#define cudaSuccess                         hipSuccess
#define cudaGetErrorString                  hipGetErrorString
#define cudaGetLastError                    hipGetLastError
#define cudaPeekAtLastError                 hipPeekAtLastError
#define cudaDeviceSynchronize               hipDeviceSynchronize

#define cudaMalloc                          hipMalloc
#define cudaFree                            hipFree
#define cudaMemcpy                          hipMemcpy
#define cudaMemcpyAsync                     hipMemcpyAsync
#define cudaMemset                          hipMemset
#define cudaMemsetAsync                     hipMemsetAsync
#define cudaMemcpyHostToDevice              hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost              hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice            hipMemcpyDeviceToDevice
#define cudaMemcpyKind                      hipMemcpyKind

// --- streams ---------------------------------------------------------------
#define cudaStream_t                        hipStream_t
#define cudaStreamCreate                    hipStreamCreate
#define cudaStreamDestroy                   hipStreamDestroy
#define cudaStreamSynchronize               hipStreamSynchronize
#define cudaStreamBeginCapture              hipStreamBeginCapture
#define cudaStreamEndCapture                hipStreamEndCapture
#define cudaStreamCaptureModeGlobal         hipStreamCaptureModeGlobal
#define cudaStreamCaptureModeThreadLocal    hipStreamCaptureModeThreadLocal

// --- graphs ----------------------------------------------------------------
#define cudaGraph_t                         hipGraph_t
#define cudaGraphExec_t                     hipGraphExec_t
#define cudaGraphInstantiate                hipGraphInstantiate
#define cudaGraphLaunch                     hipGraphLaunch
#define cudaGraphDestroy                    hipGraphDestroy
#define cudaGraphExecDestroy                hipGraphExecDestroy

// --- OpenGL interop --------------------------------------------------------
// Name mappings only; <hip/hip_gl_interop.h> itself needs GL types in scope, so
// it is included by the consumer (ComputeBuffer.h) after glad, not here (this
// shim is force-included before any GL header).
#define cudaGraphicsResource                hipGraphicsResource
#define cudaGraphicsResource_t              hipGraphicsResource_t
#define cudaGraphicsRegisterFlagsNone       hipGraphicsRegisterFlagsNone
#define cudaGraphicsGLRegisterBuffer        hipGraphicsGLRegisterBuffer
#define cudaGraphicsMapResources            hipGraphicsMapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsUnmapResources          hipGraphicsUnmapResources
#define cudaGraphicsUnregisterResource      hipGraphicsUnregisterResource

#endif // USE_HIP
