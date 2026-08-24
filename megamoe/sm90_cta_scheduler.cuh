// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

#include <cute/tensor.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/sm90_mega_moe.cuh>

namespace fuse::megamoe {

// A deliberately small deterministic scheduler for the compute-only phase.
// CTAs [0, kNumL1CTAs) produce post-SwiGLU activation tiles, the next CTAs
// consume them for Linear2, and the final kNumScatterCTAs are communication
// only.  There is no global task atomic in either GEMM hot path.  Both compute
// pools walk the same expert/padded-M order, which is required for bounded-ring
// forward progress.
template <
    uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
    uint32_t L1_SHAPE_N, uint32_t L1_SHAPE_K,
    uint32_t L2_SHAPE_N, uint32_t L2_SHAPE_K,
    uint32_t kNumExpertsPerRank,
    uint32_t kNumL1CTAs, uint32_t kNumScatterCTAs,
    uint32_t kNumSMs, uint32_t kNumRanks,
    uint32_t kNumExpertsPerLane =
        deep_gemm::math::constexpr_ceil_div(kNumExpertsPerRank, 32u),
    uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N,
    uint32_t kNumL2BlockNs = L2_SHAPE_N / BLOCK_N>
struct CTASpecializedScheduler {
  using BlockPhase = deep_gemm::sched::BlockPhase;

  static constexpr uint32_t kNumL2CTAs =
      kNumSMs - kNumL1CTAs - kNumScatterCTAs;

  static_assert(kNumL1CTAs > 0);
  static_assert(kNumL1CTAs + kNumScatterCTAs < kNumSMs);
  static_assert(L1_SHAPE_N % BLOCK_N == 0);
  static_assert(L2_SHAPE_N % BLOCK_N == 0);
  static_assert(L1_SHAPE_K % BLOCK_K == 0);
  static_assert(L2_SHAPE_K % BLOCK_K == 0);

  const deep_gemm::layout::Workspace& workspace;
  const bool is_l1;
  const bool is_l2;
  const bool is_scatter;
  const uint32_t worker_count;
  uint32_t block_idx;
  uint32_t current_local_expert_idx = 0;
  uint32_t current_num_tokens = 0;
  uint32_t current_pool_block_offset = 0;
  uint32_t m_block_idx = 0;
  uint32_t n_block_idx = 0;
  uint32_t stored_num_tokens_per_expert[kNumExpertsPerLane] = {};

  CUTLASS_DEVICE explicit CTASpecializedScheduler(
      const deep_gemm::layout::Workspace& workspace_)
      : workspace(workspace_),
        is_l1(blockIdx.x < kNumL1CTAs),
        is_l2(blockIdx.x >= kNumL1CTAs and
              blockIdx.x < kNumL1CTAs + kNumL2CTAs),
        is_scatter(blockIdx.x >= kNumL1CTAs + kNumL2CTAs),
        worker_count(is_l1 ? kNumL1CTAs : (is_l2 ? kNumL2CTAs : 1u)),
        block_idx(is_l1 ? blockIdx.x
                        : (is_l2 ? blockIdx.x - kNumL1CTAs : 0u)) {}

  CUTLASS_DEVICE uint32_t get_num_tokens(uint32_t expert_idx) const {
    uint32_t value = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (expert_idx == i * 32 + deep_gemm::ptx::get_lane_idx()) {
        value = stored_num_tokens_per_expert[i];
      }
    }
    return deep_gemm::ptx::exchange(value, expert_idx % 32);
  }

  CUTLASS_DEVICE uint32_t get_pool_block_offset(uint32_t expert_idx) const {
    uint32_t blocks = 0;
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      if (i * 32 + deep_gemm::ptx::get_lane_idx() < expert_idx) {
        blocks += deep_gemm::math::ceil_div(
            stored_num_tokens_per_expert[i], BLOCK_M);
      }
    }
    return __reduce_add_sync(0xffffffff, blocks);
  }

  CUTLASS_DEVICE void fetch_expert_recv_count() {
#pragma unroll
    for (uint32_t i = 0; i < kNumExpertsPerLane; ++i) {
      const uint32_t expert_idx = i * 32 + deep_gemm::ptx::get_lane_idx();
      uint64_t value = 0;
      if (expert_idx < kNumExpertsPerRank) {
        do {
          value = deep_gemm::ptx::ld_volatile(
              workspace.get_expert_recv_count_sum_ptr(expert_idx));
        } while (static_cast<uint32_t>(value >> 32) != kNumSMs * kNumRanks);
      }
      stored_num_tokens_per_expert[i] = static_cast<uint32_t>(value);
    }
    __syncwarp();
  }

  CUTLASS_DEVICE void set_expert_idx(uint32_t expert_idx) {
    current_local_expert_idx = expert_idx;
    current_num_tokens = get_num_tokens(expert_idx);
    current_pool_block_offset = get_pool_block_offset(expert_idx);
  }

  CUTLASS_DEVICE uint32_t get_current_pool_block_offset() const {
    return current_pool_block_offset;
  }

  CUTLASS_DEVICE uint32_t get_current_num_m_blocks() const {
    return deep_gemm::math::ceil_div(current_num_tokens, BLOCK_M);
  }

  template <bool kAligned = false>
  CUTLASS_DEVICE uint32_t get_valid_m() const {
    const uint32_t start = m_block_idx * BLOCK_M;
    if (start >= current_num_tokens) {
      return 0;
    }
    const uint32_t valid = cute::min(current_num_tokens - start, BLOCK_M);
    return kAligned ? deep_gemm::math::align(valid, 16u) : valid;
  }

  CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t>
  get_next_block() {
    if (is_scatter)
      return {BlockPhase::None, 0, 0, 0};
    const uint32_t n_blocks = is_l1 ? kNumL1BlockNs : kNumL2BlockNs;
    while (current_local_expert_idx < kNumExpertsPerRank) {
      const uint32_t m_blocks = get_current_num_m_blocks();
      const uint32_t expert_tasks = m_blocks * n_blocks;
      if (block_idx < expert_tasks) {
        // M-major on both sides is a liveness property, not merely a raster
        // choice: FC2 drains the oldest physical ring slots first.
        m_block_idx = block_idx / n_blocks;
        n_block_idx = block_idx - m_block_idx * n_blocks;
        block_idx += worker_count;
        return {
            is_l1 ? BlockPhase::Linear1 : BlockPhase::Linear2,
            current_local_expert_idx,
            m_block_idx,
            n_block_idx};
      }
      block_idx -= expert_tasks;
      ++current_local_expert_idx;
      if (current_local_expert_idx < kNumExpertsPerRank) {
        current_pool_block_offset += m_blocks;
        current_num_tokens = get_num_tokens(current_local_expert_idx);
      }
    }
    return {BlockPhase::None, 0, 0, 0};
  }
};

}  // namespace fuse::megamoe
