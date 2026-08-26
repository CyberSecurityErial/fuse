// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#ifndef FUSE_ENABLE_PROFILING
#define FUSE_ENABLE_PROFILING 0
#endif

namespace fuse {

constexpr int kMaxWorldSize = 8;
// Keep independently produced and consumed epochs on separate cache lines.
constexpr int kReadyFlagStride = 32;

using Bf16 = cutlass::bfloat16_t;
using Fp8E4m3 = cutlass::float_e4m3_t;

struct KernelTraits {
  int32_t block_m;
  int32_t block_n;
  int32_t block_k;
  int32_t threads;
  int32_t dynamic_smem_bytes;
};

}  // namespace fuse
