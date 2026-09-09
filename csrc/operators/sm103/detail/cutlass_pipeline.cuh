// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "producer_consumer.cuh"

// Blackwell CUTLASS producer/consumer adapters; geometry comes from the binding.
//
// Necessary differences from sm90/detail/cutlass_pipeline.cuh:
// - The SM100 mainloop retains Params through its constructor and accepts
//   hardware info during argument conversion. Its load() returns the advanced
//   pipeline state and K iterator; the kernel calls it for both the prologue
//   and the remaining K tiles. Keep that state continuous across both calls.
// - In the selected SM90 collective, the K loop (including iterator advance)
//   runs inside an elected-lane branch. In SM100, only the TMA copy is elected;
//   iterator advance is executed by the whole producer warp. Here lane zero
//   acquires readiness, then the warp rejoins and all producer lanes execute
//   the async-proxy fence before entering the stock TMA load. A single-lane
//   wait helper must not contain the warp barrier itself.
// - The SM100 epilogue consumes TMEM accumulators and its store() requires the
//   accumulator pipeline/state, MMA tile and ReuseTmem arguments. Those are
//   CUTLASS interface differences, not a change in the ready-epoch contract.
//
// Current implementation choices, not Blackwell requirements:
// - Keep peer-bounded Base::load() calls and the per-CTA peer-ready cache.
// - Publish output readiness at GPU scope because its consumer is a local
//   communication CTA. Wait on every lane of the issuing epilogue warp to
//   cover TMA stores issued by that warp before lane zero publishes. Scope
//   selection and this issuer-coverage strategy are not inherently SM100-only rules.
//
// TODO: Evaluate a ReadyKIterator-style adapter or another boundary-injection
// strategy to align more closely with SM90 and potentially reduce repeated
// Base::load() setup. It must preserve whole-warp participation, async-proxy
// ordering, returned state across prologue/remainder, and first-acquire
// telemetry. The current strategy may be improvable; compare correctness and
// measured performance before choosing a replacement, not just code size.

#include "fuse/arch/common.cuh"
#include "fuse/types.h"
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"
#include "epilogue_profiling.cuh"
#include "oproj_profiling.cuh"
#endif

#include <cute/tensor.hpp>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/kernel_hardware_info.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace fuse::detail {

// Same single-lane contract as SM90. The SM100 caller performs the producer
// warp rejoin and async-proxy fence after lane zero returns from this helper.
CUTLASS_DEVICE void wait_acquire_gpu_single_lane(
    const uint32_t* flag,
    uint32_t target) {
#pragma unroll 1
  while (load_acquire_gpu(flag) < target) {
    __nanosleep(64);
  }
}

#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsTimelineArguments {};

template <>
struct A2ALhsTimelineArguments<true> {
  OprojPipelineView probe{};
  A2AGemmCtaTimeline* timeline = nullptr;
  int32_t timeline_capacity = 0;
  A2AGemmPeerTimeline* peer_timeline = nullptr;
  int32_t peer_timeline_capacity = 0;
  int32_t n_tiles = 0;
};
#endif

// The receive slab is populated by this GPU's communication CTAs. Each
// peer owns one contiguous, tile-aligned K shard of A for every M tile.
template <
    class Base,
    class TileShape
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
struct A2ALhsReadyMainloop : Base {
  static constexpr int kTileM = cute::size<0>(TileShape{});
  static constexpr int kTileN = cute::size<1>(TileShape{});
  static constexpr int kTileK = cute::size<2>(TileShape{});
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;
  using MainloopClusterShape = typename Base::DispatchPolicy::ClusterShape;

  struct Arguments : BaseArguments
#if FUSE_ENABLE_PROFILING
      , A2ALhsTimelineArguments<Instrumented>
#endif
  {
    const uint32_t* ready = nullptr;
    int32_t world_size = 0;
    int32_t m_tiles = 0;
    int32_t arrivals_per_peer = 0;
    int32_t k_tiles_per_peer = 0;
    uint32_t epoch = 0;
  };

  struct Params : BaseParams
#if FUSE_ENABLE_PROFILING
      , A2ALhsTimelineArguments<Instrumented>
#endif
  {
    const uint32_t* ready = nullptr;
    int32_t world_size = 0;
    int32_t m_tiles = 0;
    int32_t arrivals_per_peer = 0;
    int32_t k_tiles_per_peer = 0;
    uint32_t epoch = 0;
  };

  template <class Problem>
  static Params to_underlying_arguments(
      const Problem& problem,
      const Arguments& args,
      void* workspace,
      const cutlass::KernelHardwareInfo& hardware = {}) {
    Params params{};
    static_cast<BaseParams&>(params) = Base::to_underlying_arguments(
        problem, static_cast<const BaseArguments&>(args), workspace, hardware);
    params.ready = args.ready;
    params.world_size = args.world_size;
    params.m_tiles = args.m_tiles;
    params.arrivals_per_peer = args.arrivals_per_peer;
    params.k_tiles_per_peer = args.k_tiles_per_peer;
    params.epoch = args.epoch;
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) {
      params.probe = args.probe;
      params.timeline = args.timeline;
      params.timeline_capacity = args.timeline_capacity;
      params.peer_timeline = args.peer_timeline;
      params.peer_timeline_capacity = args.peer_timeline_capacity;
      params.n_tiles = args.n_tiles;
    }
#endif
    return params;
  }

  template <class Problem>
  static bool can_implement(const Problem& problem, const Arguments& args) {
    const auto shape = cute::append<4>(problem, 1);
    const int64_t m = cute::get<0>(shape);
    const int64_t k = cute::get<2>(shape);
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) {
      const int64_t n = cute::get<1>(shape);
      if (!args.timeline || args.timeline_capacity <= 0 || n <= 0 ||
          cute::get<3>(shape) != 1 ||
          args.n_tiles != (n + kTileN - 1) / kTileN ||
          args.peer_timeline_capacity < 0) {
        return false;
      }
      // Release/comm fields use [M, peer], while acquire/tile metadata use
      // [L, M, N] in the same allocation. A null peer buffer requests only
      // CTA timing; a supplied buffer must hold both complete index spaces.
      const int64_t acquire_records = static_cast<int64_t>(args.m_tiles) *
          args.n_tiles * cute::get<3>(shape);
      const int64_t release_records =
          static_cast<int64_t>(args.m_tiles) * args.world_size;
      if (args.peer_timeline &&
          (args.peer_timeline_capacity < acquire_records ||
           args.peer_timeline_capacity < release_records)) {
        return false;
      }
    }
#endif
    // The ready layout has no batch dimension. The host flattens batch into
    // M; multiple independent L batches would alias the same arrivals.
    return args.ready && args.world_size > 0 &&
        args.world_size <= kMaxWorldSize && args.arrivals_per_peer > 0 &&
        args.k_tiles_per_peer > 0 && args.epoch > 0 &&
        static_cast<uint64_t>(args.epoch) * args.arrivals_per_peer <=
            std::numeric_limits<uint32_t>::max() &&
        m > 0 && args.m_tiles == (m + kTileM - 1) / kTileM &&
        cute::get<3>(shape) == 1 &&
        k == static_cast<int64_t>(args.world_size) *
                 args.k_tiles_per_peer * kTileK &&
        Base::can_implement(problem, static_cast<const BaseArguments&>(args));
  }

  CUTLASS_DEVICE A2ALhsReadyMainloop(
      const Params& params,
      MainloopClusterShape cluster_shape,
      uint32_t block_rank_in_cluster)
      : Base(static_cast<const BaseParams&>(params), cluster_shape,
             block_rank_in_cluster),
        params_(&params) {}

  template <class LoadParams, class TileCoord, class KTileIterator>
  CUTLASS_DEVICE auto load(
      typename Base::MainloopPipeline pipeline,
      typename Base::MainloopPipelineState state,
      const LoadParams& inputs,
      const TileCoord& tile_coord,
      KTileIterator k_iter,
      int k_tiles) {
    const int32_t m = static_cast<int32_t>(cute::get<0>(tile_coord));
    // Static scheduling can pad M for swizzling. TMA handles those OOB
    // loads; there is no communication flag to wait for outside the slab.
    if (m < 0 || m >= params_->m_tiles) {
      return Base::load(pipeline, state, inputs, tile_coord, k_iter, k_tiles);
    }

    const uint32_t target = params_->epoch * params_->arrivals_per_peer;
    // A kernel calls load twice per output tile (prologue and remainder).
    // Split both calls at peer boundaries, preserving CUTLASS's pipeline
    // state and never reinitializing or splitting the MMA accumulator.
    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tiles > 0) {
      const int32_t first_k = static_cast<int32_t>(*k_iter);
      const int32_t peer = first_k / params_->k_tiles_per_peer;
      CUTLASS_ASSERT(peer >= 0 && peer < params_->world_size);
      const int32_t to_peer_end =
          params_->k_tiles_per_peer - first_k % params_->k_tiles_per_peer;
      const int count = k_tiles < to_peer_end ? k_tiles : to_peer_end;

      const bool cache_hit = acquired_m_ == m && acquired_peer_ == peer;
#if FUSE_ENABLE_PROFILING
      OprojReadyRecord* probe = nullptr;
      uint64_t check_begin = 0, check_end = 0;
      if constexpr (Instrumented) {
        const int64_t index = params_->probe.tile_index(tile_coord);
        if (index >= 0 && threadIdx.x % 32 == 0) {
          auto* candidate = &params_->probe.tiles[index].ready[peer];
          if (!candidate->begin) probe = candidate; // First prologue/remainder observation only.
        }
        if (probe) check_begin = oproj_timestamp();
      }
#endif
      if (!cache_hit) {
        const uint32_t* flag = params_->ready +
            (static_cast<int64_t>(m) * params_->world_size + peer) *
                kReadyFlagStride;
        if (threadIdx.x % 32 == 0) {
          wait_acquire_gpu_single_lane(flag, target);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            if (probe) check_end = oproj_timestamp();
          }
#endif
        }
        __syncwarp();
        // The later elected TMA issuer need not be the lane which polled.
        // All producer lanes bridge the acquired generic-global writes to
        // the async proxy before entering CUTLASS's TMA load function.
        fence_proxy_async_global();
        acquired_m_ = m;
        acquired_peer_ = peer;
      }

#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (probe) {
          const uint64_t joined = oproj_timestamp();
          *probe = {check_begin, cache_hit ? check_begin : check_end,
                    joined, static_cast<uint32_t>(cache_hit)};
        }
        // This is the mainloop-load warp (warp 2), not CTA thread zero.
        // Record cache hits too: another N tile can reuse this peer's ready
        // observation without repeating the semaphore acquire.
        if (threadIdx.x % 32 == 0) {
          record_peer_acquire(tile_coord, peer);
        }
      }
#endif

      auto next = Base::load(
          pipeline, state, inputs, tile_coord, k_iter, count);
      state = cute::get<0>(next);
      // CUTLASS ForwardCoordIterator binds a const Shape&, so assignment of
      // the iterator is deleted. Keep that shape and copy its advanced coord.
      k_iter.coord = *cute::get<1>(next);
      k_tiles -= count;
    }
    return cute::make_tuple(state, k_iter);
  }

#if FUSE_ENABLE_PROFILING
  template <class Pipelines, class States, class Accumulators, class Inputs, class Coord>
  CUTLASS_DEVICE auto mma(Pipelines pipelines, States states, Accumulators accumulators,
                         Inputs inputs, Coord coord, int k_tiles) {
    if constexpr (Instrumented) {
      const int64_t index = params_->probe.tile_index(coord);
      if (index >= 0) {
        CUTLASS_ASSERT(k_tiles == params_->probe.k_tiles);
        return profile_oproj_mma(pipelines, states, accumulators, inputs, k_tiles,
            params_->probe.tiles + index, params_->probe.stages + index * k_tiles);
      }
    }
    return Base::mma(pipelines, states, accumulators, inputs, coord, k_tiles);
  }
#endif

 private:
#if FUSE_ENABLE_PROFILING
  template <class TileCoord>
  CUTLASS_DEVICE void record_peer_acquire(
      const TileCoord& tile_coord, int32_t peer) {
    if constexpr (Instrumented) {
      const int32_t m = static_cast<int32_t>(cute::get<0>(tile_coord));
      const int32_t n = static_cast<int32_t>(cute::get<1>(tile_coord));
      const int32_t l = static_cast<int32_t>(cute::get<3>(tile_coord));
      // Swizzle padding in N must not alias the next M tile's event. This
      // collective's ready protocol accepts only L=1, with batch in M.
      if (m < 0 || m >= params_->m_tiles ||
          n < 0 || n >= params_->n_tiles || l != 0) {
        return;
      }
      if (active_start_recorded_ && !params_->peer_timeline) {
        return;
      }
      const uint64_t now = read_global_timer();
      const int32_t cta = static_cast<int32_t>(blockIdx.x);
      if (!active_start_recorded_) {
        if (params_->timeline && cta < params_->timeline_capacity) {
          atomicCAS(
              reinterpret_cast<unsigned long long*>(
                  &params_->timeline[cta].active_start),
              0ull, static_cast<unsigned long long>(now));
        }
        active_start_recorded_ = true;
      }
      const int64_t tile_id =
          (static_cast<int64_t>(l) * params_->m_tiles + m) *
              params_->n_tiles + n;
      if (params_->peer_timeline &&
          tile_id < params_->peer_timeline_capacity) {
        auto& event = params_->peer_timeline[tile_id];
        // The caller clears both timeline buffers before each diagnostic
        // launch, independently of the cumulative ready/epoch state. Keep
        // the first observation when prologue and remainder share a peer.
        if (atomicCAS(
                reinterpret_cast<unsigned long long*>(&event.acquire[peer]),
                0ull, static_cast<unsigned long long>(now)) == 0ull) {
          event.m_tile = m;
          event.n_tile = n;
          event.batch = l;
          event.valid = 1;
        }
      }
    }
  }
  // Only lane zero of the single mainloop-load warp accesses this instance
  // state; it survives prologue/remainder and persistent tiles, not launches.
  bool active_start_recorded_ = false;
#endif

  const Params* params_;
  int32_t acquired_m_ = -1;
  int32_t acquired_peer_ = -1;
};

// The first epilogue warp issues CUTLASS's TMA stores. Publish its output
// tile only after destination-global completion, not merely SMEM reuse.
template <class Base, class TileShape>
struct SignalingEpilogue : Base {
  static constexpr int kTileM = cute::size<0>(TileShape{});
  static constexpr int kTileN = cute::size<1>(TileShape{});
  using ReadyTile = PublishedTile<kTileM, kTileN>;
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;

  struct Arguments : BaseArguments {
    uint32_t* ready = nullptr;
    int32_t m_tiles = 0;
    int32_t n_tiles = 0;
    uint32_t epoch = 0;
  };

  struct Params : BaseParams {
    uint32_t* ready = nullptr;
    int32_t m_tiles = 0;
    int32_t n_tiles = 0;
    uint32_t epoch = 0;
  };

  template <class Problem>
  static Params to_underlying_arguments(
      const Problem& problem, const Arguments& args, void* workspace) {
    Params params{};
    static_cast<BaseParams&>(params) = Base::to_underlying_arguments(
        problem, static_cast<const BaseArguments&>(args), workspace);
    params.ready = args.ready;
    params.m_tiles = args.m_tiles;
    params.n_tiles = args.n_tiles;
    params.epoch = args.epoch;
    return params;
  }

  template <class Problem>
  static bool can_implement(const Problem& problem, const Arguments& args) {
    const auto shape = cute::append<4>(problem, 1);
    const int64_t m = cute::get<0>(shape);
    const int64_t n = cute::get<1>(shape);
    return args.ready && args.epoch > 0 && m > 0 && n > 0 &&
        cute::get<3>(shape) > 0 &&
        args.m_tiles == (m + kTileM - 1) / kTileM &&
        args.n_tiles == (n + kTileN - 1) / kTileN &&
        Base::can_implement(problem, static_cast<const BaseArguments&>(args));
  }

  template <class Problem>
  static size_t get_workspace_size(const Problem& problem, const Arguments& args) {
    return Base::get_workspace_size(
        problem, static_cast<const BaseArguments&>(args));
  }

  template <class Problem>
  static cutlass::Status initialize_workspace(
      const Problem& problem,
      const Arguments& args,
      void* workspace,
      cudaStream_t stream,
      cutlass::CudaHostAdapter* adapter = nullptr) {
    return Base::initialize_workspace(
        problem, static_cast<const BaseArguments&>(args), workspace,
        stream, adapter);
  }

  CUTLASS_DEVICE SignalingEpilogue(
      const Params& params, typename Base::TensorStorage& storage)
      : Base(static_cast<const BaseParams&>(params), storage), params_(&params) {}

  template <
      bool ReuseTmem = false,
      class AccumulatorPipeline,
      class AccumulatorPipelineState,
      class Problem,
      class CtaTile,
      class TileCoord,
      class MmaTile,
      class TiledMma,
      class AccEngine,
      class AccLayout>
  CUTLASS_DEVICE auto store(
      typename Base::LoadPipeline load_pipeline,
      typename Base::LoadPipelineState load_state,
      typename Base::StorePipeline store_pipeline,
      typename Base::StorePipelineState store_state,
      AccumulatorPipeline acc_pipeline,
      AccumulatorPipelineState acc_state,
      Problem problem,
      CtaTile cta_tile,
      TileCoord tile_coord,
      MmaTile mma_tile,
      TiledMma tiled_mma,
      cute::Tensor<AccEngine, AccLayout> accumulators,
      typename Base::TensorStorage& storage) {
    auto states = Base::template store<ReuseTmem>(
        load_pipeline, load_state, store_pipeline, store_state,
        acc_pipeline, acc_state, problem, cta_tile, tile_coord,
        mma_tile, tiled_mma, accumulators, storage);

    const int epilogue_thread = threadIdx.x % Base::ThreadCount;
    if (epilogue_thread < 32) {
      // Drain on every lane in the issuing warp: TMA bulk groups are
      // thread-local. A wait in an unrelated warp cannot complete them.
      // The common primitive must be cp.async.bulk.wait_group 0, without
      // .read. Neither Base::store() nor Base::store_tail() guarantees this.
      fuse::detail::tma_store_wait_all();
      __syncwarp();
      if (epilogue_thread == 0) {
        const int32_t m = static_cast<int32_t>(cute::get<0>(tile_coord));
        const int32_t n = static_cast<int32_t>(cute::get<1>(tile_coord));
        const int32_t l = static_cast<int32_t>(cute::get<3>(tile_coord));
        if (m >= 0 && m < params_->m_tiles &&
            n >= 0 && n < params_->n_tiles && l >= 0) {
          const int64_t tile = ReadyTile::index(l * params_->m_tiles + m, n, params_->n_tiles);
          fuse::detail::store_release_gpu(
              params_->ready + tile * kReadyFlagStride, params_->epoch);
        }
      }
    }
    return states;
  }

 private:
  const Params* params_;
};

#if FUSE_ENABLE_PROFILING
// The original SignalingEpilogue above is deliberately unchanged. This
// diagnostic adapter duplicates only its narrow store/drain/publish bridge,
// calling the same CUTLASS Base::store and preserving the 32-lane drain.
// It is instantiated only by the private N256/K64/e32 probe, not CTA-only
// telemetry or production. No shared-memory storage is added.
template <class Base, class TileShape>
struct QkvEpilogueProbe : SignalingEpilogue<Base, TileShape> {
  using Parent = SignalingEpilogue<Base, TileShape>;
  struct Arguments : Parent::Arguments {
    QkvEpilogueRecord* records = nullptr;
    int32_t record_capacity = 0;
  };
  struct Params : Parent::Params {
    QkvEpilogueRecord* records = nullptr;
    int32_t record_capacity = 0;
  };

  template <class Problem>
  static Params to_underlying_arguments(
      const Problem& problem, const Arguments& args, void* workspace) {
    Params params{};
    static_cast<typename Parent::Params&>(params) = Parent::to_underlying_arguments(
        problem, static_cast<const typename Parent::Arguments&>(args), workspace);
    params.records = args.records;
    params.record_capacity = args.record_capacity;
    return params;
  }

  template <class Problem>
  static bool can_implement(const Problem& problem, const Arguments& args) {
    return args.records && args.record_capacity > 0 &&
        cute::get<3>(cute::append<4>(problem, 1)) == 1 &&
        Parent::can_implement(problem, static_cast<const typename Parent::Arguments&>(args));
  }

  CUTLASS_DEVICE QkvEpilogueProbe(
      const Params& params, typename Base::TensorStorage& storage)
      : Parent(static_cast<const typename Parent::Params&>(params), storage), params_(&params) {}

  template <
      bool ReuseTmem = false,
      class AccumulatorPipeline, class AccumulatorPipelineState,
      class Problem, class CtaTile, class TileCoord, class MmaTile,
      class TiledMma, class AccEngine, class AccLayout>
  CUTLASS_DEVICE auto store(
      typename Base::LoadPipeline load_pipeline,
      typename Base::LoadPipelineState load_state,
      typename Base::StorePipeline store_pipeline,
      typename Base::StorePipelineState store_state,
      AccumulatorPipeline acc_pipeline, AccumulatorPipelineState acc_state,
      Problem problem, CtaTile cta_tile, TileCoord tile_coord,
      MmaTile mma_tile, TiledMma tiled_mma,
      cute::Tensor<AccEngine, AccLayout> accumulators,
      typename Base::TensorStorage& storage) {
    const int epilogue_thread = threadIdx.x % Base::ThreadCount;
    uint64_t store_begin = 0;
    if (epilogue_thread == 0) store_begin = epilogue_timestamp();
    auto states = Base::template store<ReuseTmem>(
        load_pipeline, load_state, store_pipeline, store_state,
        acc_pipeline, acc_state, problem, cta_tile, tile_coord,
        mma_tile, tiled_mma, accumulators, storage);

    if (epilogue_thread < 32) {
      uint64_t store_end = 0;
      if (epilogue_thread == 0) store_end = epilogue_timestamp();
      fuse::detail::tma_store_wait_all();
      __syncwarp();
      if (epilogue_thread == 0) {
        const uint64_t drain_end = epilogue_timestamp();
        const int32_t m = static_cast<int32_t>(cute::get<0>(tile_coord));
        const int32_t n = static_cast<int32_t>(cute::get<1>(tile_coord));
        const int32_t l = static_cast<int32_t>(cute::get<3>(tile_coord));
        if (m >= 0 && m < params_->m_tiles &&
            n >= 0 && n < params_->n_tiles && l >= 0) {
          const int64_t tile =
              (static_cast<int64_t>(l) * params_->m_tiles + m) *
                  params_->n_tiles + n;
          fuse::detail::store_release_gpu(
              params_->ready + tile * kReadyFlagStride, params_->epoch);
          const uint64_t ready_after = epilogue_timestamp();
          if (record_.tile_count == 0) {
            record_.first_store_begin = store_begin;
            record_.first_store_end = store_end;
            record_.first_drain_end = drain_end;
            record_.first_ready_after = ready_after;
            record_.first_m_tile = m;
            record_.first_n_tile = n;
            record_.first_batch = l;
            record_.epoch = params_->epoch;
          }
          const uint64_t store_ns = store_end - store_begin;
          const uint64_t drain_ns = drain_end - store_end;
          record_.store_ns_sum += store_ns;
          record_.drain_ns_sum += drain_ns;
          record_.store_ns_max = store_ns > record_.store_ns_max ? store_ns : record_.store_ns_max;
          record_.drain_ns_max = drain_ns > record_.drain_ns_max ? drain_ns : record_.drain_ns_max;
          record_.last_ready_after = ready_after;
          ++record_.tile_count;
        }
      }
    }
    return states;
  }

  template <class CtaTile>
  CUTLASS_DEVICE void store_tail(
      typename Base::LoadPipeline load_pipeline,
      typename Base::LoadPipelineState load_state,
      typename Base::StorePipeline store_pipeline,
      typename Base::StorePipelineState store_state,
      CtaTile cta_tile) {
    Base::store_tail(load_pipeline, load_state, store_pipeline, store_state, cta_tile);
    // CUTLASS constructs this collective once per CTA and calls store_tail
    // after its persistent tile loop. Other lanes own separate empty state.
    const int32_t cta = static_cast<int32_t>(blockIdx.x);
    if (threadIdx.x % Base::ThreadCount == 0 && record_.tile_count > 0 &&
        params_->records && cta >= 0 && cta < params_->record_capacity) {
      params_->records[cta] = record_;
    }
  }

 private:
  const Params* params_;
  QkvEpilogueRecord record_{};
};
#endif

}  // namespace fuse::detail
