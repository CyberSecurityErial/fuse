// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <cutlass/cutlass.h>

namespace fuse::detail {

// QKV producer / ready / consumer ordering contract (SM103 BF16 baseline):
// The old peer-major copy queue could wait for a late GEMM tile while earlier
// tiles were already ready behind it: logical-order head-of-line blocking.
// Keep the GEMM scheduler unchanged and instead build the copy queue from its
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
struct ProducerTileOrder {
  struct Tile { int m = 0, n = 0, batch = 0; bool valid = false; };

  template <class Params>
  CUTLASS_HOST_DEVICE static Tile decode(const Params& p, uint64_t linear) {
    if (linear >= p.blocks_per_problem_) return {};
    uint64_t batch, rest, group, major;
    p.divmod_batch_(batch, rest, linear);
    const int log = p.log_swizzle_size_;
    p.divmod_cluster_blk_major_(group, major, rest >> log);
    const int minor = static_cast<int>((group << log) + (rest & ((1u << log) - 1)));
    return p.raster_order_ == Params::RasterOrder::AlongN
        ? Tile{minor, static_cast<int>(major), static_cast<int>(batch), true}
        : Tile{static_cast<int>(major), minor, static_cast<int>(batch), true};
  }

  template <class Params>
  CUTLASS_HOST_DEVICE static uint64_t linear(const Params& p, int m, int n, int batch = 0) {
    const bool along_n = p.raster_order_ == Params::RasterOrder::AlongN;
    const uint64_t minor = along_n ? m : n, major = along_n ? n : m;
    const int log = p.log_swizzle_size_;
    return batch * p.divmod_batch_.divisor +
        ((((minor >> log) * p.divmod_cluster_blk_major_.divisor + major) << log) |
         (minor & ((1u << log) - 1)));
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
  CUTLASS_HOST_DEVICE static Task decode(const Params& p, uint64_t work, int rows, int columns) {
    const uint64_t owner = work / kSlots;
    const auto producer = ProducerTileOrder::decode(p, owner);
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
        const uint64_t dependency = ProducerTileOrder::linear(p, m, n);
        if (dependency > last) last = dependency;
      }
    }
    t.valid = owner == last;
    return t;
  }
};

}  // namespace fuse::detail
