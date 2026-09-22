// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <cutlass/cutlass.h>

namespace fuse::detail {

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
    return decode_for_expert(logical, expert);
  }

  // The grouped GEMM scheduler already finds the expert cooperatively across
  // its scheduler warp. Keep the scalar lookup above for communication and
  // host reasoning, but do not repeat the same dependent binary search in all
  // 32 scheduler lanes.
  CUTLASS_HOST_DEVICE Tile decode_for_expert(int64_t logical, int expert) const {
    if (logical < 0 || logical >= tiles() || expert < 0 || expert >= experts) return {};
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

// Normalize the GEMM backend's traversal into logical (expert,M,N) order
// BEFORE deriving communication. Public raster names are not a schedule:
// the pinned device-only CUTLASS grouped scheduler has effective swizzle=1,
// and its AlongN advances N first, as does our native decoder at sw=1.
// A 2-SM pair consumes ONE 256-row tile; P physical CTAs expose P/2 workers.
// No model identity or token bucket participates in this translation.
CUTLASS_HOST_DEVICE GroupedTileOrder grouped_consumer_order(
    GroupedTileOrder requested, bool stock_scheduler = false) {
  if (stock_scheduler) {
    requested.swizzle = 1;
  }
  return requested;
}

// Number of distinct input panels touched by the first persistent-GEMM wave.
// Logical tasks are expert-contiguous and each panel owns exactly n_tiles,
// but a raster may interleave those panels. Count the prefix analytically;
// this runs once in invocation preparation, never in a communication CTA.
CUTLASS_HOST_DEVICE int grouped_first_wave_panels(
    const GroupedTileOrder& order, int compute_ctas) {
  int64_t remaining = compute_ctas < order.tiles() ? compute_ctas : order.tiles();
  int panels = 0;
  for (int expert=0; expert<order.experts && remaining>0; ++expert) {
    const int64_t expert_mt = order.row_tile_offsets[expert+1]-order.row_tile_offsets[expert];
    const int64_t window = order.window_m ? order.window_m : expert_mt;
    for (int64_t window_begin=0; window_begin<expert_mt && remaining>0;
         window_begin+=window) {
      const int64_t mt = expert_mt-window_begin < window ? expert_mt-window_begin : window;
      const int64_t tasks = mt*order.n_tiles;
      const int64_t take = remaining < tasks ? remaining : tasks;
      if (take == tasks) panels += int(mt);
      else if (!order.along_n) {
        const int width = order.n_tiles < order.swizzle ? order.n_tiles : order.swizzle;
        const int64_t touched = (take+width-1)/width;
        panels += int(touched < mt ? touched : mt);
      } else {
        int64_t left = take;
        for (int64_t begin=0; begin<mt && left>0; begin+=order.swizzle) {
          const int width = int(mt-begin < order.swizzle ? mt-begin : order.swizzle);
          const int64_t band = int64_t(width)*order.n_tiles;
          if (left >= band) { panels += width; left -= band; }
          else { panels += int(left < width ? left : width); left = 0; }
        }
      }
      remaining -= take;
    }
  }
  return panels;
}

// Workload branch, not a claim about the measured bottleneck:
//   all M_e < 192: emphasize first delivery and integer GEMM waves;
//   any M_e >=192: protect GEMM service and check sustained input delivery.
// M_e comes from this replay's router counts, not capacity or mean M. Thus
// [191,191] and [0,382] take different paths despite having the same mean.
// 192 is a provisional policy boundary, not a hardware crossover. In either
// branch, use the complete row/tile distribution to score exposed waiting;
// one large expert must not erase the small experts from the cost model.
CUTLASS_HOST_DEVICE bool grouped_use_latency_cohort(
    const int64_t* row_offsets, int experts, int threshold = 192) {
  for (int expert=0; expert<experts; ++expert)
    if (row_offsets[expert+1]-row_offsets[expert] >= threshold) return false;
  return true;
}

// Maximum number of independent 8-row producer stripes exposed by the actual
// routed rows.  Full panels contribute 16 stripes; a short tail contributes
// only ceil(tail/8).  This is the useful communication parallelism, unlike
// resident CTA count: lending more blocks than this only creates empty roles.
CUTLASS_HOST_DEVICE int64_t grouped_dispatch_producer_groups(
    const int64_t* row_offsets, int experts, int rows_per_panel = 128,
    int warps = 8) {
  int64_t groups = 0;
  for (int expert=0; expert<experts; ++expert) {
    int64_t rows = row_offsets[expert+1]-row_offsets[expert];
    groups += (rows/rows_per_panel)*(rows_per_panel/warps);
    rows %= rows_per_panel;
    groups += (rows+warps-1)/warps;
  }
  return groups;
}

// Reuse only compute blocks that have no initial tile in this invocation.
// The configured compute ceiling remains intact; large workloads return the
// original communication count. Preparation and the persistent kernel must
// agree on this pool so arrival counts include exactly the useful producers.
// Explicit-budget diagnostics do not lend; an Auto diagnostic must reproduce
// the selected base budget AND lending before its R can be called matched.
CUTLASS_HOST_DEVICE int grouped_effective_dispatch_ctas(
    int64_t panels, int n_tiles, int comm_ctas, int compute_ctas,
    int64_t producer_groups) {
  if (panels < 0 || n_tiles <= 0 || comm_ctas <= 0 || compute_ctas <= 0) return comm_ctas;
  const int64_t tiles=panels > INT64_MAX/n_tiles ? INT64_MAX : panels*n_tiles;
  const int active=tiles < compute_ctas ? int(tiles) : compute_ctas;
  const int candidate=comm_ctas+compute_ctas-active;
  if (producer_groups <= comm_ctas) return comm_ctas;
  return producer_groups < candidate ? int(producer_groups) : candidate;
}

// Match producer breadth to the panels that the first GEMM wave can consume.
// C=20, first-wave panels=6 => three producers/panel and seven early panels:
//   comm CTA:  0 1 2 | 3 4 5 | ... | 18 19
//   panel:     0 0 0 | 1 1 1 | ... |  6  6
// Later cohorts keep the same width, so faster first delivery does not starve
// the second wave. Ready remains one full-K panel; no consumer checks are added.
CUTLASS_HOST_DEVICE int grouped_dispatch_splits(
    int64_t panels, int comm_ctas, int first_wave_panels = 0,
    int64_t panel_bytes = 0, int rows_per_panel = 128, int warps = 8) {
  if (panels <= 0 || comm_ctas <= 0) return 1;
  if (first_wave_panels < 0) {
    // Large-token baseline: keep complete producer cohorts. Partial extra
    // cohorts change panel completion order; the small-token first-wave gain
    // alone is not evidence to enable them for sustained delivery.
    const int available = panels < comm_ctas ? comm_ctas/int(panels) : 1;
    const int row_groups = (rows_per_panel + warps - 1) / warps;
    return available < row_groups ? available : row_groups;
  }
  // If every panel already fits in the communication grid, distribute spare
  // CTAs over all panels. Narrowing that finite set to only wave zero delays
  // later panels without creating additional steady-state overlap.
  const bool all_panels_fit = panels < comm_ctas;
  const int target = !all_panels_fit && first_wave_panels > 0 && first_wave_panels < panels
      ? first_wave_panels : int(panels);
  if (target >= comm_ctas) return 1;
  // When all panels fit, ceil intentionally spends the otherwise-idle tail.
  // Otherwise floor keeps every first-wave panel represented in cohort zero.
  int available;
  if (target == panels) {
    available=comm_ctas/target;
    // With only one producer/panel, spend the otherwise-idle tail on the
    // earliest panels. If every panel already has >=2 producers, retain equal
    // complete cohorts instead of creating an imbalanced partial cohort.
    if (available==1 && comm_ctas%target) available=2;
  } else available=comm_ctas/target;
  // The same first-wave match can over-fragment a narrow-K panel. Measurements
  // on B300 put the useful boundary between 171 KiB/producer (regressed) and
  // 256 KiB/producer (improved). Keep at least the measured safe work quantum;
  // panel bytes and CTA budgets, not a model/shape identity, drive the cap.
  constexpr int64_t kMinProducerBytes = 256*1024;
  if (panel_bytes > 0) {
    const int byte_limited = int(panel_bytes/kMinProducerBytes);
    if (available > byte_limited) available = byte_limited > 0 ? byte_limited : 1;
  }
  const int row_groups = (rows_per_panel + warps - 1) / warps;
  return available < row_groups ? available : row_groups;
}

}  // namespace fuse::detail
