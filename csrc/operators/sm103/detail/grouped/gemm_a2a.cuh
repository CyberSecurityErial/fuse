// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "gemm.cuh"
#include "communication.cuh"

namespace fuse::detail {

// GEMM -> Combine: one CTA consumes one complete producer (expert,M,N) tile.
// Enumerate the SAME Along/swizzle order as GEMM; destination rank/token does
// not reorder this queue. Each branch writes its own (token,top-k slot) output.
//   GEMM tile -> ready[expert,m,n] -> scatter rows -> source(rank,token,slot)
// There is no top-k reduction or weighting inside this transport boundary.
template <int TileN = 128, int TileM = 128>
struct GroupedCombineComm {
  static constexpr int kMinThreads = 256;
  static constexpr size_t SharedStorageBytes = 0;
  static constexpr bool kNeedsGridFinalize = true;
  static constexpr bool kReuseIdleComputeCtas = false;
  using Arguments = GroupedCommArguments;
  using Params = Arguments;

  static bool can_implement(const Arguments& a) {
    return valid_grouped_transport(a.params) &&
        a.params.order.n_tiles == (a.params.columns + TileN - 1) / TileN;
  }
  static Params to_underlying_arguments(const Arguments& a) { return a; }

  CUTLASS_DEVICE void operator()(const Params& args, char*, int comm, int comm_ctas) {
    const auto& p = args.params;
    const uint32_t epoch = p.epoch_ptr ? *p.epoch_ptr : p.epoch;
    // One TileN=128 row contains 16 vectors, so a full warp per row wastes half
    // its lanes. Let each 16-lane group own one row (two rows per warp). Wider
    // future tiles retain this mapping and iterate columns; tails stay masked.
    const int lane = threadIdx.x % 16, row_group = threadIdx.x / 16;
    const int row_groups = blockDim.x / 16;
    for (int64_t task = comm; task < p.order.tiles(); task += comm_ctas) {
      const auto tile = p.order.decode(task);
      if (threadIdx.x == 0) {
        const auto index = p.order.output_ready_index(tile.expert, tile.m, tile.n);
        CUTLASS_PRAGMA_NO_UNROLL
        while (load_acquire_gpu(p.ready + index * kReadyFlagStride) < epoch)
          __nanosleep(64);
      }
      __syncthreads();
      const int64_t start = p.row_offsets[tile.expert];
      const int64_t rows = p.row_offsets[tile.expert + 1] - start;
      const int begin = tile.n * TileN;
      const int end = begin + TileN < p.columns ? begin + TileN : p.columns;
      for (int row = tile.m * TileM + row_group;
           row < (tile.m + 1) * TileM && row < rows; row += 4 * row_groups) {
        // Stage four independent row vectors in registers before remote stores.
        // This overlaps route/payload loads without changing the producer tile
        // or publishing partial output. No additional shared-memory stage.
        for (int col = begin + lane * 8; col < end; col += 16 * 8) {
          uint4 values[4];
          Bf16* destinations[4];
          CUTLASS_PRAGMA_UNROLL
          for(int i=0;i<4;++i) {
            const int r=row+i*row_groups;
            if(r<(tile.m+1)*TileM && r<rows) {
              const auto route=p.source[start+r];
              values[i]=*reinterpret_cast<const uint4*>(
                  p.input[tile.expert]+int64_t(r)*p.columns+col);
              destinations[i]=p.output[route.rank]+
                  (int64_t(route.token)*p.topk+route.slot)*p.columns+col;
            }
          }
          CUTLASS_PRAGMA_UNROLL
          for(int i=0;i<4;++i) {
            const int r=row+i*row_groups;
            if(r<(tile.m+1)*TileM && r<rows)
              *reinterpret_cast<uint4*>(destinations[i])=values[i];
          }
        }
      }
      __syncthreads();
    }
    // Every remote writer orders its stores before the outer cooperative grid
    // join; a fence only on the final publishing lane does not cover them.
    fence_system();
  }

  CUTLASS_DEVICE void finalize(const Params& args) {
    finalize_grouped_transport(args.params);
  }
};

template <int TileN = 128, int TileK = 64>
using GroupedGemmA2A = GroupedMonolithicGemm<
    typename Bf16GroupedGemmTypes<TileN, TileK>::CombineGemm, GroupedCombineComm<TileN>>;

}  // namespace fuse::detail
