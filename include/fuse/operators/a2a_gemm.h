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

// Finite Hopper policies for inverse A2A followed by dense GEMM.
enum class A2ALhsGemmPolicy : int32_t {
  kAuto = 0,
  kM64N128 = 1,
  kM128N128 = 2,
  kM128N160 = 3,
  kM128N256ClusterM2 = 4,
  kM128N320ClusterM2 = 5,
};

struct A2ALhsPolicyInfo {
  A2ALhsGemmPolicy policy = A2ALhsGemmPolicy::kAuto;
  int32_t tile_m = 0;
  int32_t tile_n = 0;
  int32_t tile_k = 0;
  int32_t cluster_m = 0;
  int32_t compute_ctas = 0;
  int32_t compute_clusters = 0;
  int64_t tile_count = 0;
  int64_t cluster_tile_count = 0;
  int32_t n_tiles = 0;
  int32_t waves = 0;
  int32_t last_wave_clusters = 0;
  int32_t last_wave_ctas = 0;
  int32_t frontier_aligned = 0;
  int32_t full_last_wave = 0;
  double estimated_cycles = 0.0;
};

// Inverse head-to-sequence A2A produces the row-major lhs consumed by GEMM.
// ready has a2a_lhs_gemm_ready_elements(gemm, route) entries; epoch zero is
// reserved and a reused ready buffer must keep the same shape.
struct A2AGemmParams {
  const Bf16* peer_input[kMaxWorldSize]{};
  Bf16* input_staging = nullptr;
  const uint32_t* peer_input_ready[kMaxWorldSize]{};
  Bf16* rhs_nt = nullptr;
  Bf16* output = nullptr;
  uint32_t* ready = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas = 0;
  A2ALhsGemmPolicy lhs_policy = A2ALhsGemmPolicy::kAuto;
  uint32_t epoch = 0;
  uint32_t input_epoch = 0;
  float alpha = 1.0f;
};

KernelTraits cutlass_kernel_traits();

int64_t a2a_lhs_gemm_ready_elements(
    const GemmProblem& problem,
    const UlyssesRoute& route);

int32_t recommended_a2a_lhs_gemm_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route);

A2ALhsPolicyInfo select_a2a_lhs_gemm_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    A2ALhsGemmPolicy requested = A2ALhsGemmPolicy::kAuto);

cudaError_t launch_a2a_gemm_cutlass(
    const A2AGemmParams& params,
    cudaStream_t stream);

#if FUSE_ENABLE_PROFILING
cudaError_t launch_a2a_gemm_cutlass_role_telemetry(
    const A2AGemmParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    A2AGemmPeerTimeline* peer_timeline,
    int32_t peer_timeline_capacity,
    cudaStream_t stream);

cudaError_t query_a2a_gemm_role_resources(A2AGemmRoleResources* resources);
#endif

cudaError_t launch_a2a_gemm_cutlass_reference(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas = 0);

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream);

}  // namespace fuse
