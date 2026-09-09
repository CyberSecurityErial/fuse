// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <cutlass/cutlass.h>

namespace fuse::detail {

// QKV producer / ready / consumer ordering contract (SM103 BF16 baseline):
// The old peer-major copy queue could wait for a late GEMM tile while earlier
// tiles were already ready behind it: logical-order head-of-line blocking.
// Build the copy queue from the GEMM scheduler's
// resolved raster/swizzle order. A copy rectangle belongs to the LAST logical
// producer among ALL ready tiles it intersects. Example: if a copy needs
// producer tiles 3 and 7, enumerate it under 7, not 3; acquiring tile 3 alone
// would neither make that rectangle readable nor preserve the intended order.
//
// These three granularities are deliberately separate but coordinated:
//   producer = one complete GEMM output tile;
//   release  = that output tile after all its GMEM stores complete;
//   consumer = one TMA copy rectangle, possibly spanning multiple releases.
// Inverse producer indexing assigns each copy exactly one owner. Communication
// warps stride this common ordered candidate space; padding/non-owner slots are
// skipped, then the source column determines Q/K/V, head and destination peer.
// Peer ownership therefore cannot force a warp to start at a distant producer.
//
// This removes the deterministic queue-order mismatch, NOT every possible
// ready wait. Persistent CTAs execute concurrently: logical producer 7 may
// finish before 3. Always acquire EVERY intersecting release, even after the
// owner is ready; never infer memory visibility from a logical index. Keep the
// existing release/acquire fences and final remote-store drain. No global
// in-order completion barrier, work queue atomics or extra grid synchronization
// is introduced. The scalar/vector fallback keeps its existing ordering.
// TODO: only if profiling demonstrates material residual completion-order HOL,
// evaluate bounded ready-aware lookahead; preserve exactly-once copy ownership,
// all dependency acquires and eventual progress, and measure its probing cost.

// Publication is currently one complete producer output tile, after its GMEM
// stores finish. This is NOT the MMA atom or the epilogue's SMEM subtile.
// Smaller/larger publication tiles would require a different release protocol,
// not merely changing these constants. Copy tiles may span several releases.
template <int M, int N>
struct PublishedTile {
  static constexpr int kM = M, kN = N;
  CUTLASS_HOST_DEVICE static int64_t index(int m, int n, int n_tiles) {
    return static_cast<int64_t>(m) * n_tiles + n;
  }
};

// The one-CTA-cluster CUTLASS static mapping, shared by producer and consumer.
// Read the RESOLVED scheduler parameters: padding, raster and swizzle must not
// be independently guessed by the communication path.
// Optional rank-dependent rotation of whole N bands. Rotate padded coordinates
// (not head ownership), preserving a bijection even for partial final bands.
// Both producer decode and consumer dependency inverse MUST use the same
// rotation, otherwise a consumer can wait for an unproduced band again.
// This dephases destination concentration across ranks; it does not impose
// completion order or guarantee less NVLink congestion. Group-local swizzle,
// physical output/ready addresses and all release/acquire fences stay intact.
struct NBandSwizzle {
  int offset = 0, extent = 0;

  template <class Params>
  CUTLASS_HOST_DEVICE static NBandSwizzle make(const Params& p, int rank) {
    const bool along_n = p.raster_order_ == Params::RasterOrder::AlongN;
    const int extent = static_cast<int>(along_n ? p.divmod_cluster_blk_major_.divisor
        : p.divmod_batch_.divisor / p.divmod_cluster_blk_major_.divisor);
    const int width = 1 << p.log_swizzle_size_;
    const int bands = extent / width;
    return {bands > 0 ? (rank % bands) * width : 0, extent};
  }

  CUTLASS_HOST_DEVICE int forward(int n) const {
    n += offset;
    return offset && n >= extent ? n - extent : n;
  }
  CUTLASS_HOST_DEVICE int inverse(int n) const {
    n -= offset;
    return n < 0 ? n + extent : n;
  }
};

struct ProducerTileOrder {
  struct Tile { int m = 0, n = 0, batch = 0; bool valid = false; };

  template <class Params>
  CUTLASS_HOST_DEVICE static Tile decode(
      const Params& p, uint64_t linear, NBandSwizzle swizzle = {}) {
    if (linear >= p.blocks_per_problem_) return {};
    uint64_t batch, rest, group, major;
    p.divmod_batch_(batch, rest, linear);
    const int log = p.log_swizzle_size_;
    p.divmod_cluster_blk_major_(group, major, rest >> log);
    const int minor = static_cast<int>((group << log) + (rest & ((1u << log) - 1)));
    return p.raster_order_ == Params::RasterOrder::AlongN
        ? Tile{minor, swizzle.forward(static_cast<int>(major)), static_cast<int>(batch), true}
        : Tile{static_cast<int>(major), swizzle.forward(minor), static_cast<int>(batch), true};
  }

  template <class Params>
  CUTLASS_HOST_DEVICE static uint64_t linear(
      const Params& p, int m, int n, int batch = 0, NBandSwizzle swizzle = {}) {
    n = swizzle.inverse(n);
    const bool along_n = p.raster_order_ == Params::RasterOrder::AlongN;
    const uint64_t minor = along_n ? m : n, major = along_n ? n : m;
    const int log = p.log_swizzle_size_;
    return batch * p.divmod_batch_.divisor +
        ((((minor >> log) * p.divmod_cluster_blk_major_.divisor + major) << log) |
         (minor & ((1u << log) - 1)));
  }
};

// GEMM-consumption-driven communication scheduling (OProj):
// OProj communication produces A[M, peer]; all N tiles at that M reuse it.
// GEMM's resolved tile/raster/swizzle/compute-CTA budget is the input to this
// mapping, not a quantity changed by communication. Worker c consumes logical
// tiles c, c+C, c+2C, ... . M blocks whose FIRST consumer falls in [wave*C,
// (wave+1)*C) share a window; each M occurs in exactly one window.
//
// The RESOLVED raster is also an input to delivery policy, not just indexing:
//
// AlongM, multiple N bands:
//   C[M, N=0..swizzle-1] visits every M before the remaining N bands.
//   That first band needs all A while doing only part of the GEMM work. Deliver
//   a bounded M cohort through ALL peers before opening the next cohort:
//
//       cohort A: P0 -> P1 -> P2 -> P3
//       cohort B:                         P0 -> P1 -> P2 -> P3
//
//   Inside each peer: M -> original copy chunks. Cohort width is bounded by
//   active copy slots / chunks per complete (M,peer), at least one M. A peer
//   stage therefore fits one set of copy slots when an individual ready unit
//   fits; a wider shard keeps its original chunks even when that is impossible.
//   This prioritizes K completion for fewer M blocks over broad first-peer
//   coverage. It is an experimental tradeoff, not a guaranteed deadline policy.
//
// AlongN, or only one N band:
//   A small M group serves all N without the same first-band demand burst.
//   Keep the existing group/peer diagonal order:
//
//                  peer0   peer1   peer2   peer3
//   M group A        0       1       2       3       numbers = diagonals
//   M group B        1       2       3       4
//   M group C        2       3       4       5
//
//   queue: A/P0 -> B/P0 -> A/P1 -> C/P0 -> B/P1 -> A/P2 -> ...
//
// Every cell contains all (M, chunk) tasks for that group and peer. This mixes
// starting new M groups with supplying later K stages to already-started groups.
// Within one diagonal, lower peer stages come first. Each group sees peers in
// GEMM K order; chunks still publish the SAME complete 128-row/peer ready unit.
// No extra ready checks, K subdivisions, fences or acknowledgements are added.
//
// Diagonal group size = max(resolved swizzle width, ceil(window M tiles / peer count)).
// This bounds groups by peer count, retains at least one swizzle-width group
// when it fits, and adapts to the actual first-use window rather than model names.
// A one-group window naturally becomes peer-major. This is a starting heuristic,
// not a fitted performance model. M/N, GEMM tile/raster/swizzle and compute CTA
// budget determine reuse/windows; K/CP, row/column layout and staging capacity
// determine copy chunks; communication CTA/slot count bounds cohort width.
// TODO: use measured GEMM tile/K-stage and communication service times (including
// pipeline lookahead) to model delivery deadlines and choose cohort width/lead.
// Do not substitute a model-name lookup or pretend geometry proves bandwidth.
//
// The queue defines priority, NOT execution/completion order: communication CTAs
// independently stride it and may finish out of order. Every (M, peer, chunk)
// appears exactly once; all existing arrivals and release/acquire fences remain
// mandatory. Padding and tail groups never introduce fictitious ready units.
struct A2AInputTileOrder {
  int32_t m_tiles = 0;
  int32_t n_tiles = 0;  // Resolved padded extent; padding cannot add an N band.
  int32_t ready_group_m_tiles = 1;
  int32_t log_swizzle = 0;
  uint64_t compute_ctas = 0;
  uint64_t group_stride = 1;
  bool along_n = true;

  template <class Params>
  CUTLASS_HOST_DEVICE static A2AInputTileOrder make(const Params& p, int32_t m) {
    A2AInputTileOrder order{};
    order.m_tiles = m;
    order.log_swizzle = p.log_swizzle_size_;
    order.compute_ctas = p.compute_grid_size;
    order.along_n = p.raster_order_ == Params::RasterOrder::AlongN;
    order.n_tiles = static_cast<int32_t>(order.along_n
        ? p.divmod_cluster_blk_major_.divisor
        : p.divmod_batch_.divisor / p.divmod_cluster_blk_major_.divisor);
    order.group_stride = p.divmod_cluster_blk_major_.divisor << order.log_swizzle;
    return order;
  }

  // Factored ProducerTileOrder::linear(m, 0). Its monotonicity lets us find
  // window endpoints in O(1), without scanning tiles or storing a lookup table.
  CUTLASS_HOST_DEVICE uint64_t first_use(uint64_t m) const {
    const uint64_t width = uint64_t{1} << log_swizzle;
    return along_n ? (m >> log_swizzle) * group_stride + (m & (width - 1))
                   : m << log_swizzle;
  }

  // First valid M whose first_use >= logical; m_tiles is the end sentinel.
  CUTLASS_HOST_DEVICE int32_t lower_bound(uint64_t logical) const {
    const uint64_t width = uint64_t{1} << log_swizzle;
    uint64_t m;
    if (along_n) {
      const uint64_t group = logical / group_stride;
      const uint64_t offset = logical - group * group_stride;
      m = group * width + (offset < width ? offset : width);
    } else {
      m = (logical + width - 1) >> log_swizzle;
    }
    return static_cast<int32_t>(m < static_cast<uint64_t>(m_tiles) ? m : m_tiles);
  }

  struct Task { int32_t m = 0, peer = 0, chunk = 0; };

  // Sum max(depth-g, 0) for g in [0, groups). Used to count the rectangular
  // group/peer grid before a diagonal without scanning M blocks or copy tasks.
  CUTLASS_HOST_DEVICE static uint64_t ramp_cells(int32_t depth, int32_t groups) {
    if (depth <= 0) return 0;
    const uint64_t count = depth < groups ? depth : groups;
    return count * depth - count * (count - 1) / 2;
  }

  CUTLASS_HOST_DEVICE static uint64_t m_tasks_before_diagonal(
      int32_t depth, int32_t groups, int32_t group_m_tiles,
      int32_t tail_m_tiles, int32_t world) {
    const uint64_t cells = ramp_cells(depth, groups) - ramp_cells(depth - world, groups);
    const int32_t tail_depth = depth - (groups - 1);
    const int32_t tail_peers = tail_depth <= 0 ? 0 : (tail_depth < world ? tail_depth : world);
    return cells * group_m_tiles - static_cast<uint64_t>(group_m_tiles - tail_m_tiles) * tail_peers;
  }

  CUTLASS_HOST_DEVICE Task decode(uint64_t task, int32_t world, int32_t chunks) const {
    const uint64_t tasks_per_m = static_cast<uint64_t>(world) * chunks;
    // Interleaving changes only a window's interior: its queue extent remains
    // [begin,end)*tasks_per_m. The probe M locates the window, not the copy's M.
    const uint64_t probe_m = task / tasks_per_m;
    const uint64_t wave = first_use(probe_m) / compute_ctas;
    const int32_t begin = lower_bound(wave * compute_ctas);
    const int32_t end = lower_bound((wave + 1) * compute_ctas);
    const uint64_t offset = task - static_cast<uint64_t>(begin) * tasks_per_m;
    const int32_t window_m_tiles = end - begin;
    const int32_t width = int32_t{1} << log_swizzle;
    if (!along_n && n_tiles > width) {
      // Cohorts are queue priorities only: independent copy slots may overlap
      // groups/windows. No completion barrier, new ready flag or acknowledgement.
      const int32_t cohort = ready_group_m_tiles > 0 ? ready_group_m_tiles : 1;
      const uint64_t group_tasks = static_cast<uint64_t>(cohort) * tasks_per_m;
      const int32_t group_begin = static_cast<int32_t>(offset / group_tasks) * cohort;
      const int32_t remaining = window_m_tiles - group_begin;
      const int32_t group_m = remaining < cohort ? remaining : cohort;
      const uint64_t in_group = offset % group_tasks;
      const uint64_t peer_tasks = static_cast<uint64_t>(group_m) * chunks;
      const uint64_t in_peer = in_group % peer_tasks;
      return {begin + group_begin + static_cast<int32_t>(in_peer / chunks),
              static_cast<int32_t>(in_group / peer_tasks),
              static_cast<int32_t>(in_peer % chunks)};
    }
    const int32_t share = (window_m_tiles + world - 1) / world;
    const int32_t group_m_tiles = share > width ? share : width;
    const int32_t groups = (window_m_tiles + group_m_tiles - 1) / group_m_tiles;
    if (groups == 1) {
      const uint64_t tasks_per_peer = static_cast<uint64_t>(window_m_tiles) * chunks;
      const int32_t peer = static_cast<int32_t>(offset / tasks_per_peer);
      const uint64_t in_peer = offset % tasks_per_peer;
      return {begin + static_cast<int32_t>(in_peer / chunks), peer,
              static_cast<int32_t>(in_peer % chunks)};
    }

    const int32_t tail_m_tiles = window_m_tiles - (groups - 1) * group_m_tiles;
    const uint64_t m_task = offset / chunks;
    // At most 2*world-1 diagonals; CP4/8 needs at most four prefix probes.
    // Search M/peer task counts (not padded cells), so a partial final group has no holes.
    int32_t lo = 0, hi = groups + world - 1;
    while (hi - lo > 1) {
      const int32_t mid = (lo + hi) / 2;
      if (m_tasks_before_diagonal(mid, groups, group_m_tiles, tail_m_tiles, world) <= m_task) lo = mid;
      else hi = mid;
    }
    uint64_t in_diagonal = m_task - m_tasks_before_diagonal(lo, groups, group_m_tiles, tail_m_tiles, world);
    int32_t group = lo < groups ? lo : groups - 1;
    // Descending group order is ascending peer order. A partial tail group, if
    // present on this diagonal, is first; all remaining groups have equal width.
    if (group == groups - 1) {
      if (in_diagonal < static_cast<uint64_t>(tail_m_tiles)) {
        return {begin + group * group_m_tiles + static_cast<int32_t>(in_diagonal),
                lo - group, static_cast<int32_t>(offset % chunks)};
      }
      in_diagonal -= tail_m_tiles;
      --group;
    }
    group -= static_cast<int32_t>(in_diagonal / group_m_tiles);
    return {begin + group * group_m_tiles + static_cast<int32_t>(in_diagonal % group_m_tiles),
            lo - group, static_cast<int32_t>(offset % chunks)};
  }
};

// Enumerate copy rectangles near their latest *logical* publication dependency.
// Each copy appears exactly once, even when N64/160/192 producer tiles intersect
// the fixed 128-column TMA rectangle. Invalid candidate slots are skipped.
// No global completion ordering is implied: execution still acquires EVERY
// intersecting ready tile. No queue atomics, additional fence or grid barrier.
template <class ReadyTile, int CopyM, int CopyN>
struct ConsumerTileOrder {
  CUTLASS_HOST_DEVICE static constexpr int gcd(int a, int b) {
    return b ? gcd(b, a % b) : a;
  }
  static constexpr int kRows = (ReadyTile::kM + CopyM - gcd(ReadyTile::kM, CopyM) + CopyM - 1) / CopyM;
  static constexpr int kColumns = (ReadyTile::kN + CopyN - gcd(ReadyTile::kN, CopyN) + CopyN - 1) / CopyN;
  static constexpr int kSlots = kRows * kColumns;
  struct Task {
    int row = 0, column = 0, rows = 0, columns = 0;
    int first_m = 0, last_m = 0, first_n = 0, last_n = 0;
    bool valid = false;
  };

  template <class Params>
  CUTLASS_HOST_DEVICE static Task decode(
      const Params& p, uint64_t work, int rows, int columns, NBandSwizzle swizzle = {}) {
    const uint64_t owner = work / kSlots;
    const auto producer = ProducerTileOrder::decode(p, owner, swizzle);
    if (!producer.valid || producer.batch != 0) return {}; // Batch is flattened into M.
    const int pm = producer.m * ReadyTile::kM, pn = producer.n * ReadyTile::kN;
    if (pm >= rows || pn >= columns) return {};
    const int slot = static_cast<int>(work % kSlots);
    Task t{};
    t.row = (pm / CopyM + slot / kColumns) * CopyM;
    t.column = (pn / CopyN + slot % kColumns) * CopyN;
    if (t.row >= rows || t.column >= columns ||
        t.row >= pm + ReadyTile::kM || t.column >= pn + ReadyTile::kN) return {};
    t.rows = rows - t.row < CopyM ? rows - t.row : CopyM;
    t.columns = columns - t.column < CopyN ? columns - t.column : CopyN;
    t.first_m = t.row / ReadyTile::kM;
    t.last_m = (t.row + t.rows - 1) / ReadyTile::kM;
    t.first_n = t.column / ReadyTile::kN;
    t.last_n = (t.column + t.columns - 1) / ReadyTile::kN;
    uint64_t last = 0;
    for (int m = t.first_m; m <= t.last_m; ++m) {
      for (int n = t.first_n; n <= t.last_n; ++n) {
        const uint64_t dependency = ProducerTileOrder::linear(p, m, n, 0, swizzle);
        if (dependency > last) last = dependency;
      }
    }
    t.valid = owner == last;
    return t;
  }
};

}  // namespace fuse::detail
