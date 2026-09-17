// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include <cuda_runtime_api.h>
#include "fuse/profiling/grouped.cuh"

namespace fuse {

struct GroupedTokenSource {
  int32_t rank = 0, token = 0, slot = 0;
};

struct GroupedGemmPolicy {
  int32_t num_comm_ctas = 20;
  int32_t num_compute_ctas = 128;
  int32_t swizzle = 1;
  bool along_n = false;
  // Explicit CUTLASS configuration; no shape-name heuristic or online search.
  // M is 128; N=128/256, K=64/128. Stage count follows the SMEM carveout.
  int32_t tile_n = 128, tile_k = 64;
  // Share only the last incomplete communication batch across CTAs. Delivery
  // stays full-K/128 rows. Opt-in: small workloads benefit; large ones can be
  // flat. Detailed per-panel profiling currently requires this to be false.
  bool balance_dispatch_tail = false;
};

// Device arrays contain one matrix pointer per LOCAL expert. Matrices are
// contiguous row-major A[M_e,K], physical weight W[N,K], and D[M_e,N].
// BF16 operands/output, FP32 accumulation, alpha=1 and beta=0.
//
// Dispatch: peer_input -> lhs staging -> GEMM -> output.
// Combine:  lhs -> GEMM -> output staging -> peer_output[token,slot,N].
// No router, activation, weighting or top-k reduction is included.
struct Bf16GroupedGemmParams {
  Bf16* const* lhs = nullptr;
  const Bf16* const* weight_nt = nullptr;
  Bf16* const* output = nullptr;
  const int64_t* row_offsets = nullptr;      // device [experts+1], actual row prefix
  const GroupedTokenSource* source = nullptr; // device [row_offsets[experts]]
  const Bf16* peer_input[kMaxWorldSize]{};
  Bf16* peer_output[kMaxWorldSize]{};
  // Caller-owned, peer-accessible device arrays [world_size*kReadyFlagStride].
  // Zero once before plan creation, use only for this collective plan, and keep
  // alive until all ranks' final stream work has completed.
  uint32_t* peer_started[kMaxWorldSize]{};
  uint32_t* peer_done[kMaxWorldSize]{};
  int32_t experts = 0, n = 0, k = 0;
  int32_t expert_row_capacity = 0;
  int32_t rank = 0, world_size = 0, topk = 0;
  GroupedGemmPolicy policy{};
  // Dispatch only, caller-allocated rows per lhs[e]. Default 0 uses the full
  // expert_row_capacity. A smaller positive multiple of 128 reuses input slots.
  // WARNING: bounded reuse can substantially slow execution (measured 16–79%
  // throughput loss). Enable ONLY under memory pressure, not for speed.
  // Allocate this capacity before plan creation; changing it requires a new
  // plan. Output/routes remain full-sized. Launch never silently retries OOM.
  int32_t dispatch_buffer_rows = 0;
#if FUSE_ENABLE_PROFILING
  GroupedProfile profile{};
#endif
};

// Opaque host plan owns private device metadata and CUTLASS workspace. Creation
// is outside capture; launch is capture-safe, allocates nothing and performs no
// host count readback. Pointers/configuration are fixed, device counts/routes
// may change between calls. Every rank launches once per collective invocation
// on one serial stream. Recreate all plans/state before 2^32 epoch wraparound.
// Caller enables peer access and supplies valid in-bounds, nonduplicated branch
// routes. Destroy only after all captured/replayed uses have finished.
struct Bf16GroupedGemmPlan;
cudaError_t create_bf16_a2a_grouped_gemm(
    const Bf16GroupedGemmParams&, Bf16GroupedGemmPlan**);
cudaError_t create_bf16_grouped_gemm_a2a(
    const Bf16GroupedGemmParams&, Bf16GroupedGemmPlan**);
cudaError_t launch_bf16_grouped_gemm(Bf16GroupedGemmPlan*, cudaStream_t);
cudaError_t destroy_bf16_grouped_gemm(Bf16GroupedGemmPlan*);

}  // namespace fuse
