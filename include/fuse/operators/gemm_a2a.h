// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include "fuse/layout/gemm.h"
#include "fuse/layout/ulysses.h"
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"
#endif

#include <cuda_runtime_api.h>

namespace fuse {

// Dense projection followed by forward Ulysses QKV packing. peer_output[r]
// stores contiguous post-A2A Q, K and V tensors; defer_v_a2a leaves V local.
struct GemmA2AParams {
  const Bf16* lhs;
  const Bf16* rhs_nt;
  Bf16* local_output;
  Bf16* peer_output[kMaxWorldSize];
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready;
  uint32_t* completion_epoch = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas;
  uint32_t epoch;
  float alpha = 1.0f;
};

// E4M3 projection with FP32 accumulation and BF16 routed output.
struct Fp8GemmA2AParams {
  const Fp8E4m3* lhs;
  const Fp8E4m3* rhs_nt;
  Bf16* local_output;
  Bf16* peer_output[kMaxWorldSize];
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready;
  uint32_t* completion_epoch = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas;
  uint32_t epoch;
  float alpha = 1.0f;
};

KernelTraits projection_cutlass_kernel_traits();
KernelTraits qkv_cutlass_kernel_traits(const GemmProblem& problem);
KernelTraits fp8_cutlass_kernel_traits();

int32_t recommended_gemm_a2a_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route);

cudaError_t launch_gemm_a2a_cutlass(
    const GemmA2AParams& params,
    cudaStream_t stream);

#if FUSE_ENABLE_PROFILING
cudaError_t launch_gemm_a2a_role_telemetry(
    const GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream);
#endif

cudaError_t launch_gemm_a2a_fp8_cutlass(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream);

cudaError_t launch_batched_cutlass_reference(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas = 0);

cudaError_t launch_dense_fp8_cutlass_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas = 0);

cudaError_t launch_gemm_a2a_copy_reference(
    const GemmA2AParams& params,
    cudaStream_t stream);

cudaError_t launch_gemm_a2a_fp8_copy_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream);

}  // namespace fuse
