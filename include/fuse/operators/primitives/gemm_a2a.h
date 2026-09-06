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

// Explicit precision name for new integrations.  GemmA2AParams remains the
// source-compatible BF16 spelling used by existing callers.
using Bf16GemmA2AParams = GemmA2AParams;

// E4M3 projection with FP32 accumulation and E4M3 routed output. The caller
// owns quantization scales/amax; alpha is applied before E4M3 rounding.
struct Fp8GemmA2AParams {
  const Fp8E4m3* lhs;
  const Fp8E4m3* rhs_nt;
  Fp8E4m3* local_output;
  Fp8E4m3* peer_output[kMaxWorldSize];
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
// Without route/communication information, return the finest supported QKV
// geometry so callers can size ready/workspace buffers safely. This does not
// report the kernel selected by auto. To query the actual selected geometry,
// use the overload below and pass its resolved, positive communication-CTA
// count rather than the auto sentinel 0.
KernelTraits qkv_cutlass_kernel_traits(const GemmProblem& problem);
KernelTraits qkv_cutlass_kernel_traits(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count);
// Conservative FP8 capacity/resource query. The N64 geometry has the finest
// ready-flag granularity and the largest production shared-storage footprint;
// this overload does not claim that auto will launch N64.
KernelTraits fp8_cutlass_kernel_traits();
// Actual FP8 auto geometry for a resolved positive communication-CTA count.
KernelTraits fp8_cutlass_kernel_traits(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count);

int32_t recommended_gemm_a2a_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route);

// Optional, explicitly calibrated host-side model for MXFP8-weight QKV.
// Coefficients describe a steady-state role/service-envelope model, not
// bare communication bandwidth or universal hardware constants. The route
// calibration includes producer readiness. Zero initialization retains auto.
struct Mxfp8QkvForwardCommModel {
  int32_t world_size = 0;
  int32_t sm_count = 0;
  double compute_gflop_sm_us = 0.0;  // us * compute SMs / GEMM GFLOP
  double route_slot_task_us = 0.0;  // us * active route slots / copy tasks
  double minimum_gain = 0.0;
};

int32_t select_mxfp8_qkv_forward_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const Mxfp8QkvForwardCommModel& model);

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

#if FUSE_ENABLE_PROFILING
cudaError_t launch_gemm_a2a_fp8_role_telemetry(
    const Fp8GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream);
#endif

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

// Independent BF16 QKV-forward services with the production-selected tile.
// DQ and input initialization are caller-owned. Producer pairs use the same
// kernel entry, monolithic scheduler and fused shared-memory reservation;
// NoSignal disables the epilogue's tile drain/release via a null ready pointer.
// Copy omits producer waits and cross-rank finalization. None of these APIs
// measures a complete forward operator or concurrent compute/route contention.
enum class QkvForwardReference : int32_t {
  kProducerSignalingSubgrid = 0,
  kProducerNoSignalSubgrid = 1,
  kProducerSignalingFullgrid = 2,
  kProducerNoSignalFullgrid = 3,
  kCopyFusedReservation = 4,
};

struct QkvForwardReferenceResources {
  int32_t tile_m = 0;
  int32_t tile_n = 0;
  int32_t tile_k = 0;
  int32_t selected_cluster_m = 0;
  int32_t selected_gemm_stages = 0;
  int32_t primitive_dynamic_smem_bytes = 0;
  int32_t primitive_registers_per_thread = 0;
  int32_t primitive_grid_x = 0;
  int32_t primitive_launch_cluster_m = 0;
  int32_t threads_per_cta = 0;
  int32_t ready_flag_stride = 0;
  int32_t ready_m_tiles = 0;
  int32_t ready_n_tiles = 0;
  int32_t copy_use_tma = 0;
  int32_t copy_use_tma_store = 0;
  int32_t copy_slots = 0;
};

cudaError_t launch_qkv_forward_reference(
    const GemmA2AParams& params,
    QkvForwardReference primitive,
    cudaStream_t stream);
cudaError_t query_qkv_forward_reference(
    const GemmA2AParams& params,
    QkvForwardReference primitive,
    QkvForwardReferenceResources* resources);

// MXFP8-weight baseline: BF16 activations/communication/dX; FP32 dW.
// Offline original-axis weights are dequantized on every forward/B launch.
// Workspace and B-to-W input leases remain caller-owned. W does not read Wq.
struct Mxfp8GemmA2AParams {
  const Bf16* lhs;
  Mxfp8Weight weight{};
  Mxfp8WeightWorkspace weight_workspace{};
  Bf16* local_output;
  Bf16* peer_output[kMaxWorldSize];
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready;
  uint32_t* completion_epoch = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;  // Rank-major QKV forward; causal rows are unsupported.
  int32_t num_comm_ctas;
  uint32_t epoch;
  float alpha = 1.0f;
};

cudaError_t launch_gemm_a2a_mxfp8_cutlass(
    const Mxfp8GemmA2AParams& params,
    cudaStream_t stream);

}  // namespace fuse
