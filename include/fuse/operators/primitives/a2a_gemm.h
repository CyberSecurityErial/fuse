// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include "fuse/layout/mxfp8.h"
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

// Explicit precision name for new integrations.  A2AGemmParams remains the
// source-compatible BF16 spelling used by existing callers.
using Bf16A2AGemmParams = A2AGemmParams;

// FP8 inverse A2A followed by FP8 GEMM with FP32 accumulation and E4M3 output.
// The caller owns quantization scales/amax; alpha is the already-combined
// scale applied before the result is rounded to E4M3.
struct Fp8A2AGemmParams {
  const Fp8E4m3* peer_input[kMaxWorldSize]{};
  Fp8E4m3* input_staging = nullptr;
  const uint32_t* peer_input_ready[kMaxWorldSize]{};
  Fp8E4m3* rhs_nt = nullptr;
  Fp8E4m3* output = nullptr;
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

// Optional, explicitly calibrated host-side model for MXFP8-weight OProj.
// These are service-model coefficients, not universal hardware constants.
// A zero-initialized model leaves the existing production policy unchanged.
struct Mxfp8OprojCommModel {
  int32_t world_size = 0;
  int32_t sm_count = 0;
  double compute_flop_us = 0.0;       // per tile TFLOP, per persistent wave
  double compute_tile_us = 0.0;       // per 128x256 output tile, per wave
  double copy_mib_us = 0.0;           // per MiB of remote payload
  double copy_task_wave_us = 0.0;     // per four-issuer communication wave
  double launch_prior_us = 5.0;      // modeling prior, not measured launch time
  double minimum_gain = 0.10;        // predicted time reduction, not speedup
};

// Only the measured bulk/tile family is eligible. Unsupported calibration or
// less than minimum_gain predicted improvement retains the existing policy.
int32_t select_mxfp8_oproj_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const Mxfp8OprojCommModel& model);

A2ALhsPolicyInfo select_a2a_lhs_gemm_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    A2ALhsGemmPolicy requested = A2ALhsGemmPolicy::kAuto);

cudaError_t launch_a2a_gemm_cutlass(
    const A2AGemmParams& params,
    cudaStream_t stream);

cudaError_t launch_a2a_gemm_fp8_cutlass(
    const Fp8A2AGemmParams& params,
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

cudaError_t launch_a2a_gemm_fp8_cutlass_reference(
    const Fp8A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas = 0);

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream);

// Opt-in copy service with the fused path's selected ready-M window. The
// original two-argument reference retains its fixed-window scheduling.
cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream,
    bool match_fused_schedule);

// Query the same resolved fused window without launching a kernel.
cudaError_t a2a_gemm_comm_window(
    const A2AGemmParams& params,
    int32_t* m_window);

cudaError_t launch_a2a_gemm_fp8_copy_reference(
    const Fp8A2AGemmParams& params,
    cudaStream_t stream);

// MXFP8-weight baseline: BF16 activations/communication/dX; FP32 dW.
// Offline original-axis weights are dequantized on every forward/B launch.
// Workspace and B-to-W input leases remain caller-owned. W does not read Wq.
struct Mxfp8A2AGemmParams {
  const Bf16* peer_input[kMaxWorldSize]{};
  Bf16* input_staging = nullptr;
  const uint32_t* peer_input_ready[kMaxWorldSize]{};
  Mxfp8Weight weight{};
  Mxfp8WeightWorkspace weight_workspace{};
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

cudaError_t launch_a2a_gemm_mxfp8_cutlass(
    const Mxfp8A2AGemmParams& params,
    cudaStream_t stream);

}  // namespace fuse
