// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/backward_common.h"
#include "fuse/types.h"
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"
#endif

#include <cuda_runtime_api.h>

namespace fuse {

// QKV projection backward, B phase:
//
//   [head-sharded dQ,dK,dV] --HeadToSequence A2A--> dQKV[M,QKV]
//   dX[M,H] = dQKV[M,QKV] * Wqkv[QKV,H]
//
// dQ/dK/dV are the ordinary planar FlashAttention gradients for this rank's
// local heads and all global sequence rows. The route writes directly into
// each destination-owned dQKV staging matrix in packed [all Q, all K, all V]
// order; no framework-side cat, index-select, permute, or contiguous copy is
// part of this interface.
struct QkvBackwardDataParams {
  const Bf16* grad_q = nullptr;
  const Bf16* grad_k = nullptr;
  const Bf16* grad_v = nullptr;
  Bf16* peer_dqkv_staging[kMaxWorldSize]{};
  uint32_t* peer_ready[kMaxWorldSize]{};
  uint32_t* peer_done_epoch[kMaxWorldSize]{};
  const Bf16* weight = nullptr;  // Stored forward weight [QKV, H].
  Bf16* grad_input = nullptr;    // [M, H].
  int32_t local_tokens = 0;      // M on this CP rank.
  int32_t hidden = 0;            // H.
  int32_t batch = 1;
  int32_t q_heads = 0;
  int32_t kv_heads = 0;
  int32_t head_dim = 0;
  int32_t world_size = 1;
  int32_t rank = 0;
  int32_t num_comm_ctas = 0;
  BackwardGemmPolicy gemm_policy = BackwardGemmPolicy::kAuto;
  uint32_t epoch = 0;
  bool causal_load_balanced = false;
  float alpha = 1.0f;
};

// QKV projection backward, W phase:
//
//   dWqkv[QKV,H] = dQKV^T[QKV,M] * X[M,H]
//
// In ZeroBubble mode dqkv_staging is the destination-owned buffer produced by
// the earlier B phase. It and saved_input form a lease: neither allocation may
// be overwritten or freed until this call finishes. One lease/slot is needed
// per outstanding B-to-W interval. No extra D2D stash is required when the
// framework can preserve that slot; otherwise the framework must explicitly
// copy these operands to its own delayed-wgrad storage.
struct QkvBackwardWeightParams {
  const Bf16* dqkv_staging = nullptr;  // [M, QKV], packed Q then K then V.
  const Bf16* saved_input = nullptr;   // Forward input X [M, H].
  Bf16* grad_weight = nullptr;         // [QKV, H].
  int32_t local_tokens = 0;
  int32_t hidden = 0;
  int32_t q_heads = 0;
  int32_t kv_heads = 0;
  int32_t head_dim = 0;
  float alpha = 1.0f;
  float beta = 0.0f;  // Set to one to accumulate into an existing main_grad.
};

struct QkvBackwardParams {
  QkvBackwardDataParams data;
  QkvBackwardWeightParams weight;
  WeightGradientMode weight_mode = WeightGradientMode::kImmediate;
};

int32_t recommended_qkv_backward_comm_ctas(
    const QkvBackwardDataParams& params);
BackwardGemmPolicy recommended_qkv_backward_gemm_policy(
    const QkvBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count);
KernelTraits qkv_backward_kernel_traits(
    const QkvBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count);
int64_t qkv_backward_ready_elements(const QkvBackwardDataParams& params);

cudaError_t launch_qkv_backward_data(
    const QkvBackwardDataParams& params,
    cudaStream_t stream);
#if FUSE_ENABLE_PROFILING
// Diagnostic-only launch. It preserves the production dataflow and records
// one timestamp record per physical CTA; profiling builds must never be used
// for formal performance numbers.
cudaError_t launch_qkv_backward_data_role_telemetry(
    const QkvBackwardDataParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream);
#endif
cudaError_t launch_qkv_backward_weight(
    const QkvBackwardWeightParams& params,
    cudaStream_t stream);

// Convenience entry. Immediate mode launches B then W on the same stream.
// Deferred mode launches only B; the caller later invokes
// launch_qkv_backward_weight() from its ZeroBubble W phase.
cudaError_t launch_qkv_backward(
    const QkvBackwardParams& params,
    cudaStream_t stream);

}  // namespace fuse
