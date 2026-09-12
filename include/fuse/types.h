// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
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

// Packed row-major E4M3 [rows,K], with CUTLASS native SFA scale storage.
// Every consecutive K32 group shares one UE8M0 scale. Allocation sizes include
// native scale padding; each operator's size query defines its logical shape.
struct Mxfp8Activation {
  const Fp8E4m3* data = nullptr;
  const uint8_t* scales = nullptr;
  size_t data_bytes = 0, scale_bytes = 0;
};

struct KernelTraits {
  int32_t block_m;
  int32_t block_n;
  int32_t block_k;
  int32_t threads;
  int32_t dynamic_smem_bytes;
};

}  // namespace fuse
