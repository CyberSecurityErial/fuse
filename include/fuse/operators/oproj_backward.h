// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/backward_common.h"
#include "fuse/types.h"
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"
#endif

#include <cuda_runtime_api.h>

namespace fuse {

// Output projection backward, B phase:
//
//   dA[M,A] = dY[M,H] * Wo[H,A]
//   dA --SequenceToHead A2A--> head-sharded FlashAttention gradient
//
// A = q_heads * head_dim. peer_grad_attention[destination] stores that
// destination's local heads for all global sequence rows.
struct OprojBackwardDataParams {
  const Bf16* grad_output = nullptr;  // dY [M, H].
  const Bf16* weight = nullptr;       // Stored forward weight Wo [H, A].
  Bf16* local_grad_attention = nullptr;  // GEMM staging dA [M, A].
  Bf16* peer_grad_attention[kMaxWorldSize]{};
  uint32_t* peer_done_epoch[kMaxWorldSize]{};
  uint32_t* ready = nullptr;
  int32_t local_tokens = 0;  // M on this CP rank.
  int32_t hidden = 0;        // H.
  int32_t batch = 1;
  int32_t q_heads = 0;
  int32_t head_dim = 0;
  int32_t world_size = 1;
  int32_t rank = 0;
  int32_t num_comm_ctas = 0;
  BackwardGemmPolicy gemm_policy = BackwardGemmPolicy::kAuto;
  uint32_t epoch = 0;
  bool causal_load_balanced = false;
  float alpha = 1.0f;
};

// Output projection backward, W phase:
//
//   dWo[H,A] = dY^T[H,M] * A[M,A]
//
// ZeroBubble deferral only extends the lifetime of grad_output and the saved
// forward attention input. No layout transform is required. If the framework
// cannot retain either tensor, its explicit stash is the unavoidable extra
// ZeroBubble storage operation.
struct OprojBackwardWeightParams {
  const Bf16* grad_output = nullptr;       // dY [M, H].
  const Bf16* saved_attention = nullptr;   // Forward A [M, A].
  Bf16* grad_weight = nullptr;             // [H, A].
  int32_t local_tokens = 0;
  int32_t hidden = 0;
  int32_t q_heads = 0;
  int32_t head_dim = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
};

struct OprojBackwardParams {
  OprojBackwardDataParams data;
  OprojBackwardWeightParams weight;
  WeightGradientMode weight_mode = WeightGradientMode::kImmediate;
};

int32_t recommended_oproj_backward_comm_ctas(
    const OprojBackwardDataParams& params);
BackwardGemmPolicy recommended_oproj_backward_gemm_policy(
    const OprojBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count);
KernelTraits oproj_backward_kernel_traits(
    const OprojBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count);
int64_t oproj_backward_ready_elements(const OprojBackwardDataParams& params);

cudaError_t launch_oproj_backward_data(
    const OprojBackwardDataParams& params,
    cudaStream_t stream);
#if FUSE_ENABLE_PROFILING
// Diagnostic-only launch. It preserves the production dataflow and records
// one timestamp record per physical CTA; profiling builds must never be used
// for formal performance numbers.
cudaError_t launch_oproj_backward_data_role_telemetry(
    const OprojBackwardDataParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream);
#endif
cudaError_t launch_oproj_backward_weight(
    const OprojBackwardWeightParams& params,
    cudaStream_t stream);
cudaError_t launch_oproj_backward(
    const OprojBackwardParams& params,
    cudaStream_t stream);

}  // namespace fuse
