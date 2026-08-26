#pragma once

#include "fuse/problem.h"
#include "fuse/route.h"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

#ifndef FUSE_ENABLE_PROFILING
#define FUSE_ENABLE_PROFILING 0
#endif

namespace fuse {

constexpr int kMaxWorldSize = 8;
// Keep independently produced/consumed epochs on separate 128-byte lines.
constexpr int kReadyFlagStride = 32;

using Bf16 = cutlass::bfloat16_t;
using Fp8E4m3 = cutlass::float_e4m3_t;

// Finite, precompiled Hopper policies for inverse A2A -> dense GEMM.  Auto
// ranks the same candidates from the actual problem and resident compute-CTA
// budget; explicit values are useful for reproducible policy sweeps.
enum class A2ALhsGemmPolicy : int32_t {
  kAuto = 0,
  kM64N128 = 1,
  kM128N128 = 2,
  kM128N160 = 3,
  kM128N256ClusterM2 = 4,
  // Wide-N cluster policy used when its cluster wave exactly covers complete
  // M frontiers.
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

// Inverse head-to-sequence A2A followed by dense O-projection. Communication
// prepares the row-major activation matrix consumed by GEMM; the compute role
// waits only at contiguous peer boundaries along K.
//   peer_input[r]: [batch, global_seq, local_heads, head_dim]
//   input_staging: [gemm.m, gemm.k]
//   rhs_nt:        [gemm.k, gemm.n]
//   output:      [gemm.l, gemm.m, gemm.n]
// ready requires a2a_lhs_gemm_ready_elements(gemm, route) entries. Epoch zero
// is reserved and the same shape must be used while a ready buffer is reused
// monotonically.
struct A2AGemmParams {
  const Bf16* peer_input[kMaxWorldSize]{};
  Bf16* input_staging = nullptr;
  // Optional producer epochs. When input_epoch is nonzero, communication waits
  // for peer_input_ready[peer] before reading that peer.
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

// Dense QKV projection followed by the forward Ulysses A2A pack.
//   lhs:          [m, k]
//   rhs_nt:       [n, k]
//   local_output: [m, n]
//   peer_output[r] is one allocation containing three contiguous tensors:
//     Q [batch, global_seq, q_heads/world, head_dim]
//     K [batch, global_seq, kv_heads/world, head_dim]
//     V [batch, global_seq, kv_heads/world, head_dim]
//   in that order. Each tensor exactly matches TE FlashAttention's post-A2A
//   bshd layout and can be exposed as a zero-copy view.
//   When route.defer_v_a2a is true, the allocation contains only Q followed
//   by K; V remains in local_output for an explicitly delayed consumer.
//   Communication is source-owned: rank s consumes ready tiles from its own
//   local_output and writes each destination's slice to peer_output[r].
//   peer_route_done_epoch[r] points to a destination-owned array of
//   world_size * kReadyFlagStride uint32_t entries. Source s publishes epoch
//   in slot s after all of its remote stores finish; destination r does not
//   return until every source has completed its routed output.
// ready requires
//   gemm.l * ceil(gemm.m / 128) * ceil(gemm.n / 128) * kReadyFlagStride
// uint32_t entries.
struct GemmA2AParams {
  const Bf16* lhs;
  const Bf16* rhs_nt;
  Bf16* local_output;
  Bf16* peer_output[kMaxWorldSize];
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready;
  // Optional completion epoch for a downstream consumer. QKV_GQA_PACK
  // publishes it after every V output tile is globally visible.
  uint32_t* completion_epoch = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas;
  uint32_t epoch;
  float alpha = 1.0f;
};

// E4M3 x E4M3 dense projection with FP32 accumulation and BF16 output,
// followed by QKV_GQA_PACK. alpha applies the product of input/weight
// dequantization scales in the stock CUTLASS epilogue. ready follows the
// GemmA2AParams capacity contract.
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

struct KernelTraits {
  int32_t block_m;
  int32_t block_n;
  int32_t block_k;
  int32_t threads;
  int32_t dynamic_smem_bytes;
};

// One diagnostic sample per physical CTA. SM90 %globaltimer is shared across
// SMs on a device, so entries from communication and compute CTAs can be put
// on one timeline. This type is unused by the production entry point.
struct A2AGemmCtaTimeline {
  uint64_t start = 0;
  uint64_t end = 0;
  // First point at which this CTA's GEMM producer passed its initial ready
  // dependency and entered the CUTLASS mainloop.
  uint64_t active_start = 0;
};

// Diagnostic-only peer publication/observation timestamps. Communication
// indexes release records by [ready_m, peer_slot]. Compute indexes acquire
// records by logical GEMM tile and stores one timestamp per peer slot.
struct A2AGemmPeerTimeline {
  uint64_t release = 0;
  uint64_t acquire[kMaxWorldSize]{};
  int32_t m_tile = 0;
  int32_t n_tile = 0;
  int32_t batch = 0;
  uint32_t valid = 0;
};

struct A2AGemmRoleResources {
  int32_t threads_per_cta = 0;
  int32_t registers_per_thread = 0;
  int32_t telemetry_registers_per_thread = 0;
  int32_t static_smem_bytes = 0;
  int32_t dynamic_smem_bytes = 0;
  int32_t cluster_ctas = 0;
  int32_t comm_active_warps = 0;
  int32_t compute_active_warps = 0;
  int32_t comm_working_smem_bytes = 0;
};

KernelTraits cutlass_kernel_traits();
KernelTraits projection_cutlass_kernel_traits();
KernelTraits qkv_cutlass_kernel_traits(const GemmProblem& problem);
KernelTraits fp8_cutlass_kernel_traits();

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

int32_t recommended_gemm_a2a_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route);

cudaError_t launch_a2a_gemm_cutlass(
    const A2AGemmParams& params,
    cudaStream_t stream);

// Diagnostic-only launch of the same monolithic policy. Production calls do
// not carry timeline state or execute instrumentation branches.
cudaError_t launch_a2a_gemm_cutlass_role_telemetry(
    const A2AGemmParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    A2AGemmPeerTimeline* peer_timeline,
    int32_t peer_timeline_capacity,
    cudaStream_t stream);

cudaError_t query_a2a_gemm_role_resources(A2AGemmRoleResources* resources);

cudaError_t launch_a2a_gemm_cutlass_reference(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas = 0);

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream);

cudaError_t launch_gemm_a2a_cutlass(const GemmA2AParams& params, cudaStream_t stream);
cudaError_t launch_gemm_a2a_deepgemm(
    const GemmA2AParams& params,
    cudaStream_t stream);
cudaError_t launch_gemm_a2a_deepgemm_compute_only(
    const GemmA2AParams& params,
    cudaStream_t stream);

cudaError_t launch_gemm_a2a_fp8_cutlass(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream);

cudaError_t launch_batched_cutlass_reference(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas = 0);
cudaError_t launch_deepgemm_bf16_reference(
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
cudaError_t launch_gemm_a2a_deepgemm_copy_reference(
    const GemmA2AParams& params,
    cudaStream_t stream);

cudaError_t launch_gemm_a2a_fp8_copy_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream);

}  // namespace fuse
