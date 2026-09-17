// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace fuse::detail {

// If there are fewer independent panels than communication CTAs, share their
// row work. Never create more stripes than the panel's warp-sized row groups.
// Large workloads retain one producer per panel and the original queue.
CUTLASS_HOST_DEVICE int grouped_dispatch_splits(
    int64_t panels, int comm_ctas, int rows_per_panel = 128, int warps = 8) {
  if (panels <= 0 || panels >= comm_ctas) return 1;
  const int available = int(comm_ctas / panels);
  const int row_groups = (rows_per_panel + warps - 1) / warps;
  return available < row_groups ? available : row_groups;
}

// One shared ordering for grouped compute, dispatch first-use, and combine.
// The GPU builds row_tile_offsets[e+1] = sum_{j<=e} ceil(M_j / TileM).
// Empty experts have equal adjacent offsets; they produce no work or flags.
// N and K are common to this grouped invocation, M_e is genuinely variable.
//
// logical tile -> (expert, M tile, N tile) -> physical ready address
//                  |             |
// dispatch:        +-- M panel --+   copy the entire K once, then release
// combine:         +-- (M,N) tile    acquire, then route its output rows
//
// Along and swizzle change traversal, NOT ready granularity or row ownership.
// A partial swizzle band is compact, with no extra GEMM work for empty tiles.
// Dispatch enumerates M panels in increasing expert/M order: this is their
// first-use order for BOTH rasters below. All N consumers reuse that copy.
// Combine enumerates the same logical tiles as GEMM. Concurrent completion
// may differ from this priority order: always acquire the actual tile flag;
// a later tile's completion must never stand in for an earlier tile's release.
struct GroupedTileOrder {
  struct Tile {
    int32_t expert = -1, m = -1, n = -1;
    bool valid = false;
  };

  const int64_t* row_tile_offsets = nullptr;  // GPU, experts+1 entries
  int32_t experts = 0, n_tiles = 0, swizzle = 1;
  bool along_n = false;
  int32_t window_m = 0; // Dispatch ring: finish every N tile in this M window.

  CUTLASS_HOST_DEVICE int64_t row_tiles() const {
    return row_tile_offsets[experts];
  }
  CUTLASS_HOST_DEVICE int64_t tiles() const {
    return row_tiles() * n_tiles;
  }

  CUTLASS_HOST_DEVICE int expert_for_row_tile(int64_t row_tile) const {
    // upper_bound, including any run of empty experts at either boundary.
    int lo = 0, hi = experts;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      if (row_tile_offsets[mid + 1] <= row_tile) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  CUTLASS_HOST_DEVICE Tile decode(int64_t logical) const {
    if (logical < 0 || logical >= tiles()) return {};
    const int expert = expert_for_row_tile(logical / n_tiles);
    int64_t local = logical - row_tile_offsets[expert] * n_tiles;
    int64_t mt = row_tile_offsets[expert + 1] - row_tile_offsets[expert];
    const int64_t begin_m = window_m ? (local / (int64_t(window_m)*n_tiles))*window_m : 0;
    if (window_m) {
      local -= begin_m*n_tiles;
      mt = mt-begin_m < window_m ? mt-begin_m : window_m;
    }
    const int64_t major_count = along_n ? n_tiles : mt;
    const int64_t minor_count = along_n ? mt : n_tiles;
    const int64_t band = local / (major_count * swizzle);
    const int64_t begin = band * swizzle;
    const int64_t remaining = minor_count - begin;
    const int64_t width = remaining < swizzle ? remaining : swizzle;
    const int64_t in_band = local - begin * major_count;
    const int major = static_cast<int>(in_band / width);
    const int minor = static_cast<int>(begin + in_band % width);
    return {expert, int(begin_m)+(along_n ? minor : major), along_n ? major : minor, true};
  }

  CUTLASS_HOST_DEVICE int64_t linear(int expert, int m, int n) const {
    int64_t mt = row_tile_offsets[expert + 1] - row_tile_offsets[expert];
    const int64_t begin_m = window_m ? (m/window_m)*window_m : 0;
    if (window_m) {
      m -= int(begin_m);
      mt = mt-begin_m < window_m ? mt-begin_m : window_m;
    }
    const int64_t major_count = along_n ? n_tiles : mt;
    const int64_t minor_count = along_n ? mt : n_tiles;
    const int64_t minor = along_n ? m : n;
    const int64_t major = along_n ? n : m;
    const int64_t begin = minor / swizzle * swizzle;
    const int64_t remaining = minor_count - begin;
    const int64_t width = remaining < swizzle ? remaining : swizzle;
    return (row_tile_offsets[expert]+begin_m) * n_tiles + begin * major_count +
        major * width + minor - begin;
  }

  // Physical ready storage is independent of the tuning configuration.
  CUTLASS_HOST_DEVICE int64_t input_ready_index(int expert, int m) const {
    return row_tile_offsets[expert] + m;
  }
  CUTLASS_HOST_DEVICE int64_t output_ready_index(int expert, int m, int n) const {
    return input_ready_index(expert, m) * n_tiles + n;
  }
};

}  // namespace fuse::detail
