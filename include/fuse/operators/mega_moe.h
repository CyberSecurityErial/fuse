// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace fuse {

// All resident CTAs cooperatively execute dispatch and combine. During the
// compute phase CHAIN executes both GEMMs on each CTA; SPLIT reserves disjoint
// persistent worker pools for L1 and L2 and connects them through global-memory
// tiles plus release/acquire epochs.
enum class MegaMoeSchedule : int32_t {
  kChain = 0,
  kSplit = 1,
};

struct MegaMoeProblem {
  int32_t tokens_per_rank = 0;
  int32_t hidden = 0;
  int32_t intermediate = 0;
  int32_t num_experts = 0;
  int32_t topk = 0;
  int32_t max_tokens_per_local_expert = 0;
};

// Storage is deliberately explicit: the benchmark can audit every byte and a
// framework adapter can place the symmetric fields in its own registered
// allocation. All capacities are in elements, not bytes.
struct MegaMoeWorkspace {
  Fp8E4m3* expert_input = nullptr;
  Bf16* l1_output = nullptr;
  Fp8E4m3* l2_input = nullptr;
  Bf16* l2_output = nullptr;
  float* routed_weight = nullptr;
  int32_t* source_rank = nullptr;
  int32_t* source_token = nullptr;
  int32_t* source_topk = nullptr;

  // Double-buffered counters are indexed by epoch parity. Ready arrays carry
  // monotonically increasing epochs and are never cleared in steady state.
  uint32_t* control = nullptr;
  uint32_t* expert_counts = nullptr;
  uint32_t* scatter_n_done = nullptr;
  uint32_t* l1_tile_ready = nullptr;
  uint32_t* l2_k_ready = nullptr;
  uint32_t* l2_tile_ready = nullptr;

  // Source-owned final contribution slots and output.
  Bf16* combine = nullptr;
  Bf16* output = nullptr;
};

struct MegaMoeParams {
  const Fp8E4m3* peer_input[kMaxWorldSize]{};
  const int32_t* peer_topk_idx[kMaxWorldSize]{};
  const float* peer_topk_weight[kMaxWorldSize]{};

  // Each rank owns the weights for its contiguous local-expert range.
  // W1: [E_local, 2I, H], W2: [E_local, H, I], both stored NT.
  const Fp8E4m3* w1_nt = nullptr;
  const Fp8E4m3* w2_nt = nullptr;

  MegaMoeWorkspace local{};
  uint32_t* peer_control[kMaxWorldSize]{};
  Bf16* peer_combine[kMaxWorldSize]{};

  MegaMoeProblem problem{};
  int32_t world_size = 0;
  int32_t rank = 0;
  // Zero selects the largest grid proven simultaneously resident at launch.
  int32_t num_ctas = 0;
  // CTASP only. A small tail pool may begin scatter before FC1 finishes; the
  // middle pool runs L2. Must leave both GEMM pools non-empty.
  int32_t num_l1_ctas = 0;
  int32_t num_scatter_ctas = 0;
  MegaMoeSchedule schedule = MegaMoeSchedule::kChain;
  uint32_t epoch = 0;
  float l1_scale = 1.0f;
  float requant_scale = 1.0f;
  float l2_scale = 1.0f;
  float activation_clamp = 10.0f;
};

struct MegaMoeWorkspaceSizes {
  size_t expert_input = 0;
  size_t l1_output = 0;
  size_t l2_input = 0;
  size_t l2_output = 0;
  size_t routed_weight = 0;
  size_t metadata = 0;
  size_t control = 0;
  size_t expert_counts = 0;
  size_t scatter_n_done = 0;
  size_t l1_tile_ready = 0;
  size_t l2_k_ready = 0;
  size_t l2_tile_ready = 0;
  size_t combine = 0;
  size_t output = 0;
};

MegaMoeWorkspaceSizes mega_moe_workspace_sizes(
    const MegaMoeProblem& problem,
    int32_t world_size);

KernelTraits mega_moe_cutlass_kernel_traits();

cudaError_t launch_mega_moe_cta_specialized(
    const MegaMoeParams& params,
    cudaStream_t stream);

}  // namespace fuse
