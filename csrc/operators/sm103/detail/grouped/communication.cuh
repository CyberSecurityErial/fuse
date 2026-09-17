// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "producer_consumer.cuh"
#include "fuse/arch/common.cuh"
#include "fuse/operators/primitives/grouped_gemm.h"
#include <cute/tensor.hpp>
#include <cassert>

namespace fuse::detail {

// Packed expert-row -> original token branch. Routing is external metadata,
// not a model-name/top-k heuristic. Distinct slots may name the same token;
// combine writes each slot separately (no weighted reduction in this boundary).
using GroupedTokenSource = ::fuse::GroupedTokenSource;

struct GroupedCommParams {
#if FUSE_ENABLE_PROFILING
  GroupedProfile profile{};
#endif
  GroupedTileOrder order{};
  const int64_t* row_offsets = nullptr;       // actual rows, not padded tiles
  const GroupedTokenSource* source = nullptr;
  const Bf16* const* input = nullptr;         // dispatch: peers; combine: experts
  Bf16* const* output = nullptr;             // dispatch: experts; combine: peers
  uint32_t* ready = nullptr;
  uint32_t* arrivals = nullptr;              // dispatch only, one counter/panel
  uint32_t* consumed = nullptr;              // bounded Dispatch: N readers finished
  int32_t buffer_m = 0;                      // physical 128-row slots per expert
  bool balance_tail = false;
  uint32_t* peer_done[kMaxWorldSize]{};       // per receiver [source rank * stride]
  int32_t columns = 0, topk = 0, rank = 0, world_size = 0, num_comm_ctas = 0;
  uint32_t epoch = 0;
  const uint32_t* epoch_ptr = nullptr;
};

struct GroupedPrepareParams {
  const int64_t* row_offsets = nullptr;
  int64_t* row_tile_offsets = nullptr;
  cute::Shape<int32_t,int32_t,int32_t>* shapes = nullptr;
  uint32_t* epoch = nullptr;
  uint32_t* arrivals = nullptr;
  uint32_t* consumed = nullptr;
  bool balance_tail = false;
  int32_t num_comm_ctas = 0;
  uint32_t* peer_started[kMaxWorldSize]{};
  int32_t experts = 0, n = 0, k = 0, rank = 0, world_size = 0;
  int64_t row_capacity = 0;
  int32_t expert_row_capacity = INT32_MAX;
};

// Fixed-size Graph node, GPU-variable expert counts. No padded expert work or
// host readback/re-capture: regenerate only shapes and the compact tile prefix.
// Every rank invokes this collective the same number of times on one stream.
// State is zero-initialized ONCE, and must be collectively reinitialized before
// UINT32_MAX invocations. Ready flags need not be cleared on each replay.
// Entry publication follows prior input writes on the calling stream; exit
// publication follows all output writes. This also prevents a fast rank from
// reusing input storage while another rank is still reading the prior call.
template <int TileM = 128>
__global__ void prepare_grouped_invocation(GroupedPrepareParams p) {
  for (int e=threadIdx.x; e<p.experts; e+=blockDim.x) {
    const int64_t rows = p.row_offsets[e+1]-p.row_offsets[e];
    assert(rows >= 0 && rows <= p.expert_row_capacity);
    p.shapes[e] = cute::make_shape(int32_t(rows),p.n,p.k);
  }
  if (threadIdx.x == 0) {
    assert(p.row_offsets[0] == 0 && p.row_offsets[p.experts] <= p.row_capacity);
    int64_t tiles = 0;
    p.row_tile_offsets[0] = 0;
    for (int e=0; e<p.experts; ++e) {
      tiles += (p.row_offsets[e+1]-p.row_offsets[e]+TileM-1)/TileM;
      p.row_tile_offsets[e+1] = tiles;
    }
    assert(*p.epoch != UINT32_MAX);
    ++*p.epoch;
  }
  __syncthreads();
  if (p.consumed)
    for (int64_t panel=threadIdx.x; panel<p.row_tile_offsets[p.experts]; panel+=blockDim.x)
      p.consumed[panel]=0;
  if (p.arrivals && grouped_dispatch_splits(p.row_tile_offsets[p.experts],p.num_comm_ctas)>1)
    for (int64_t panel=threadIdx.x; panel<p.row_tile_offsets[p.experts]; panel+=blockDim.x)
      p.arrivals[panel]=0;
  else if (p.arrivals && p.balance_tail) {
    const int64_t panels=p.row_tile_offsets[p.experts];
    const int64_t begin=panels-panels%p.num_comm_ctas;
    if (begin) for(int64_t panel=begin+threadIdx.x;panel<panels;panel+=blockDim.x)
      p.arrivals[panel]=0;
  }
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const uint32_t epoch = *p.epoch;
    if (lane < p.world_size)
      store_release_system(p.peer_started[lane] + p.rank*kReadyFlagStride, epoch);
    __syncwarp();
    if (lane < p.world_size) {
      CUTLASS_PRAGMA_NO_UNROLL
      while (load_acquire_system(p.peer_started[p.rank] + lane*kReadyFlagStride) < epoch)
        __nanosleep(64);
    }
    __syncwarp();
  }
}

// Invocation state is private to Grouped; neither transport depends on Projection.
struct GroupedCommArguments { GroupedCommParams params; };

// Each producer first joins ALL its writers, then arrives with release/acquire
// semantics. The atomic modification chain carries preceding producers' stores
// to the last arrival. That leader publishes the SAME full-panel ready flag;
// consumers neither see partial K nor perform additional ready checks.
CUTLASS_DEVICE uint32_t arrive_grouped_panel(uint32_t* counter) {
  uint32_t previous;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
               : "=r"(previous) : "l"(counter) : "memory");
  return previous;
}

inline bool valid_grouped_transport(const GroupedCommParams& p) {
  if (!p.order.row_tile_offsets || !p.row_offsets || !p.source || !p.input || !p.output ||
      !p.ready || p.order.experts <= 0 || p.order.n_tiles <= 0 ||
      p.columns <= 0 || p.columns % 8 ||
      p.topk <= 0 || p.world_size < 1 || p.world_size > kMaxWorldSize ||
      p.rank < 0 || p.rank >= p.world_size || p.num_comm_ctas <= 0 ||
      (!p.epoch && !p.epoch_ptr))
    return false;
  for (int r = 0; r < p.world_size; ++r)
    if (!p.peer_done[r]) return false;
  return true;
}

CUTLASS_DEVICE void finalize_grouped_transport(const GroupedCommParams& p) {
  if (blockIdx.x != 0 || threadIdx.x >= 32) return;
  const int lane = threadIdx.x;
  const uint32_t epoch = p.epoch_ptr ? *p.epoch_ptr : p.epoch;
  if (lane < p.world_size)
    store_release_system(p.peer_done[lane] + p.rank * kReadyFlagStride, epoch);
  __syncwarp();
  if (lane < p.world_size) {
    CUTLASS_PRAGMA_NO_UNROLL
    while (load_acquire_system(p.peer_done[p.rank] + lane * kReadyFlagStride) < epoch)
      __nanosleep(64);
  }
  __syncwarp();
}

}  // namespace fuse::detail
