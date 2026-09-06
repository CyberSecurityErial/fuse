// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace fuse {

// Immediate mode runs data-gradient and weight-gradient work in the same
// backward call. Deferred mode (the ZeroBubble B/W split) runs only the
// data-gradient phase; the caller must keep the documented operands alive and
// call the operator's weight-gradient entry during the later W phase.
enum class WeightGradientMode : int32_t {
  kImmediate = 0,
  kDeferred = 1,
};

// Explicit data-gradient tile choices are benchmark knobs. kAuto is the
// production entry and is resolved from the GEMM shape, route work and the
// available compute CTAs; it never depends on a model name.
enum class BackwardGemmPolicy : int32_t {
  kAuto = 0,
  kM128N64 = 1,
  kM128N128 = 2,
  kM128N160 = 3,
  kM128N192 = 4,
  kM128N256 = 5,
  kM128N64ClusterM2 = 6,
};

// Explicit FP32 main_grad candidates for the MXFP8-weight operators. kAuto
// intentionally preserves the measured baseline until tuning is complete.
enum class Mxfp8WgradPolicy : int32_t {
  kAuto = 0,
  kM128N256K64ClusterM2 = 1,
  kM128N128K64ClusterM2 = 2,
  kM128N128K128ClusterM2 = 3,
  kM128N256K64ClusterM1 = 4,
  kM128N192K64ClusterM2 = 5,
  kM128N256K32ClusterM2 = 6,
};

struct Mxfp8WgradKernelTraits {
  Mxfp8WgradPolicy policy = Mxfp8WgradPolicy::kAuto;
  int32_t block_m = 0;
  int32_t block_n = 0;
  int32_t block_k = 0;
  int32_t cluster_m = 0;
  int32_t stages = 0;
  int32_t dynamic_smem_bytes = 0;
  int32_t registers_per_thread = 0;
};

// Metadata query on the current CUDA device, outside the timed launch path.
// Registers come from cudaFuncGetAttributes, not an accumulator-size estimate.
cudaError_t mxfp8_wgrad_kernel_traits(
    Mxfp8WgradPolicy policy,
    Mxfp8WgradKernelTraits* traits);

}  // namespace fuse
