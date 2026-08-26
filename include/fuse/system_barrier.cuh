// SPDX-License-Identifier: Apache-2.0
// System-scope barrier primitives adapted from ByteDance Flux.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifndef FUSE_ENABLE_PROFILING
#define FUSE_ENABLE_PROFILING 0
#endif

namespace fuse::detail {

__device__ __forceinline__ uint32_t load_acquire_gpu(const uint32_t* ptr) {
  uint32_t value;
  asm volatile(
      "ld.global.acquire.gpu.b32 %0, [%1];\n"
      : "=r"(value)
      : "l"(ptr)
      : "memory");
  return value;
}

__device__ __forceinline__ void store_release_gpu(uint32_t* ptr, uint32_t value) {
  asm volatile("st.global.release.gpu.b32 [%0], %1;\n" :: "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t* ptr) {
  uint32_t value;
  asm volatile(
      "ld.global.acquire.sys.b32 %0, [%1];\n"
      : "=r"(value)
      : "l"(ptr)
      : "memory");
  return value;
}

__device__ __forceinline__ void store_release_system(uint32_t* ptr, uint32_t value) {
  asm volatile("st.global.release.sys.b32 [%0], %1;\n" :: "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ void add_release_system(uint32_t* ptr, uint32_t value = 1) {
  asm volatile("fence.acq_rel.sys;\n" ::: "memory");
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;\n" :: "l"(ptr), "r"(value) : "memory");
}

__device__ __forceinline__ void add_release_gpu(uint32_t* ptr, uint32_t value = 1) {
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;\n" :: "l"(ptr), "r"(value) : "memory");
}

#if FUSE_ENABLE_PROFILING
// Telemetry-only returning form. The production path uses add_release_gpu's
// non-returning reduction instruction.
__device__ __forceinline__ uint32_t add_release_gpu_fetch_old(
    uint32_t* ptr,
    uint32_t value = 1) {
  uint32_t old;
  asm volatile("fence.acq_rel.gpu;\n" ::: "memory");
  asm volatile(
      "atom.relaxed.gpu.global.add.u32 %0, [%1], %2;\n"
      : "=r"(old)
      : "l"(ptr), "r"(value)
      : "memory");
  return old;
}
#endif

__device__ __forceinline__ void wait_acquire_system(
    const uint32_t* ptr,
    uint32_t target,
    int lane) {
  if (lane == 0) {
#pragma unroll 1
    while (load_acquire_system(ptr) < target) {
      __nanosleep(64);
    }
  }
  __syncwarp();
}

__device__ __forceinline__ void wait_acquire_gpu(
    const uint32_t* ptr,
    uint32_t target,
    int lane) {
  if (lane == 0) {
#pragma unroll 1
    while (load_acquire_gpu(ptr) < target) {
      __nanosleep(64);
    }
  }
  __syncwarp();
}

}  // namespace fuse::detail
