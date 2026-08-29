// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/arch/sm90.cuh"
#include "fuse/types.h"
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"
#endif

#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/cuda_host_adapter.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace fuse::detail {

// Single-lane wait for use inside CUTLASS's elect_one_sync() producer branch.
// Do not add a warp barrier here: the other lanes do not enter that branch.
CUTLASS_DEVICE void wait_acquire_gpu_single_lane(
    const uint32_t* flag,
    uint32_t target) {
#pragma unroll 1
  while (load_acquire_gpu(flag) < target) {
    __nanosleep(64);
  }
}

// Source-push backward routes publish readiness from a peer GPU. The elected
// TMA producer therefore needs a system-scope acquire; forward pull routes
// keep the cheaper GPU-scope path.
CUTLASS_DEVICE void wait_acquire_system_single_lane(
    const uint32_t* flag,
    uint32_t target) {
#pragma unroll 1
  while (load_acquire_system(flag) < target) {
    __nanosleep(64);
  }
}

#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct PeerAcquireRecorder {
  CUTLASS_DEVICE explicit PeerAcquireRecorder(uint64_t*) {}
  CUTLASS_DEVICE void record(int32_t) {}
};

template <>
struct PeerAcquireRecorder<true> {
  uint64_t* timestamps = nullptr;

  CUTLASS_DEVICE explicit PeerAcquireRecorder(uint64_t* values)
      : timestamps(values) {}

  CUTLASS_DEVICE void record(int32_t group) {
    if (timestamps) {
      timestamps[group] = read_global_timer();
    }
  }
};
#endif

// Sequential K-group adapter used by the generic A2A -> GEMM path.
template <
    class Iterator,
    bool SystemScope = false
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
class ReadyKIterator
#if FUSE_ENABLE_PROFILING
    : private PeerAcquireRecorder<Instrumented>
#endif
{
 public:
  CUTLASS_DEVICE ReadyKIterator(
      Iterator iterator,
      const uint32_t* ready,
      int32_t ready_group_stride,
      uint32_t target,
      int32_t tiles_per_group,
      int32_t first_k_tile,
      int32_t work_tile_count
#if FUSE_ENABLE_PROFILING
      , uint64_t* acquire_timestamps = nullptr
#endif
      )
#if FUSE_ENABLE_PROFILING
      : PeerAcquireRecorder<Instrumented>(acquire_timestamps),
#else
      :
#endif
        iterator_(iterator),
        ready_(ready),
        target_(target),
        ready_group_stride_(ready_group_stride),
        group_(first_k_tile / tiles_per_group),
        tiles_left_in_group_(
            tiles_per_group - first_k_tile % tiles_per_group),
        tiles_per_group_(tiles_per_group),
        work_tiles_left_(work_tile_count) {}

  CUTLASS_DEVICE decltype(auto) operator*() const { return *iterator_; }

  CUTLASS_DEVICE ReadyKIterator& operator++() {
    ++iterator_;
    --work_tiles_left_;
    --tiles_left_in_group_;
    if (tiles_left_in_group_ == 0 && work_tiles_left_ > 0) {
      ++group_;
      tiles_left_in_group_ = tiles_per_group_;
      if constexpr (SystemScope) {
        wait_acquire_system_single_lane(
            ready_ + static_cast<int64_t>(group_) * ready_group_stride_,
            target_);
      } else {
        wait_acquire_gpu_single_lane(
            ready_ + static_cast<int64_t>(group_) * ready_group_stride_,
            target_);
      }
#if FUSE_ENABLE_PROFILING
      this->record(group_);
#endif
    }
    return *this;
  }

 private:
  Iterator iterator_;
  const uint32_t* ready_;
  uint32_t target_;
  int32_t ready_group_stride_;
  int32_t group_;
  int32_t tiles_left_in_group_;
  int32_t tiles_per_group_;
  int32_t work_tiles_left_;
};

template <
#if FUSE_ENABLE_PROFILING
    bool Instrumented = false,
#endif
    bool SystemScope = false,
    class Iterator>
CUTLASS_DEVICE auto make_ready_k_iterator(
    Iterator iterator,
    const uint32_t* ready,
    int32_t ready_group_stride,
    uint32_t target,
    int32_t tiles_per_group,
    int32_t first_k_tile,
    int32_t work_tile_count
#if FUSE_ENABLE_PROFILING
    , uint64_t* acquire_timestamps = nullptr
#endif
    ) {
  return ReadyKIterator<
      Iterator,
      SystemScope
#if FUSE_ENABLE_PROFILING
      , Instrumented
#endif
      >(
      iterator, ready, ready_group_stride, target, tiles_per_group,
      first_k_tile, work_tile_count
#if FUSE_ENABLE_PROFILING
      , acquire_timestamps
#endif
      );
}

}  // namespace fuse::detail


// CUTLASS adapters for A2A readiness and GEMM tile publication.
namespace fuse::detail {

// Readiness adapter for inverse head-to-sequence A2A -> dense GEMM. Every
// peer contributes one contiguous K shard of operand A for a given M tile.
// The elected TMA producer waits at peer/K boundaries while the stock CUTLASS
// load pipeline, WGMMA, and epilogue remain unchanged.
#if FUSE_ENABLE_PROFILING
template <bool Enabled>
struct A2ALhsTimelineArguments {};

template <>
struct A2ALhsTimelineArguments<true> {
  A2AGemmCtaTimeline* timeline = nullptr;
  int32_t timeline_capacity = 0;
  A2AGemmPeerTimeline* peer_timeline = nullptr;
  int32_t peer_timeline_capacity = 0;
  int32_t n_tiles = 0;
};
#endif

template <
    class Base,
    int32_t MTilesPerReady = 1,
    bool SystemScope = false
#if FUSE_ENABLE_PROFILING
    ,
    bool Instrumented = false>
#else
    >
#endif
struct A2ALhsReadyMainloop : Base {
  static_assert(MTilesPerReady > 0);
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;

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

  template <class ProblemShape>
  static Params to_underlying_arguments(
      const ProblemShape& problem,
      const Arguments& args,
      void* workspace) {
    Params params{};
    static_cast<BaseParams&>(params) = Base::to_underlying_arguments(
        problem, static_cast<const BaseArguments&>(args), workspace);
    params.ready = args.ready;
    params.world_size = args.world_size;
    params.m_tiles = args.m_tiles;
    params.arrivals_per_peer = args.arrivals_per_peer;
    params.k_tiles_per_peer = args.k_tiles_per_peer;
    params.epoch = args.epoch;
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) {
      params.timeline = args.timeline;
      params.timeline_capacity = args.timeline_capacity;
      params.peer_timeline = args.peer_timeline;
      params.peer_timeline_capacity = args.peer_timeline_capacity;
      params.n_tiles = args.n_tiles;
    }
#endif
    return params;
  }

  template <class ProblemShape>
  static bool can_implement(const ProblemShape& problem, const Arguments& args) {
    return Base::can_implement(problem, static_cast<const BaseArguments&>(args));
  }

  template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      const Params& params,
      typename Base::MainloopPipeline pipeline,
      typename Base::PipelineState write_state,
      const cute::tuple<TensorA, TensorB>& inputs,
      const BlockCoord& block_coord,
      KTileIterator k_iter,
      int k_tiles,
      int lane,
      uint32_t block_rank_in_cluster,
      typename Base::TensorStorage& storage) {
    const int32_t m = static_cast<int32_t>(cute::get<0>(block_coord));
    if (m < 0 || m >= params.m_tiles) {
      Base::load(
          static_cast<const BaseParams&>(params), pipeline, write_state,
          inputs, block_coord, k_iter, k_tiles, lane,
          block_rank_in_cluster, storage);
      return;
    }
    // Communication publishes one flag group per physical M tile.  A smaller
    // GEMM tile may split that producer tile across multiple compute CTAs.
    const int32_t ready_m = m / MTilesPerReady;
    const uint32_t* tile_ready = params.ready +
        static_cast<int64_t>(ready_m) * params.world_size * kReadyFlagStride;
    const uint32_t target = params.epoch * params.arrivals_per_peer;
    const int32_t first_k_tile = static_cast<int32_t>(*k_iter);
    const int32_t first_peer = first_k_tile / params.k_tiles_per_peer;
#if FUSE_ENABLE_PROFILING
    uint64_t* acquire_timestamps = nullptr;
    A2AGemmPeerTimeline* peer_event = nullptr;
    if constexpr (Instrumented) {
      const int32_t n = static_cast<int32_t>(cute::get<1>(block_coord));
      const int32_t l = static_cast<int32_t>(cute::get<3>(block_coord));
      const int64_t tile_id =
          (static_cast<int64_t>(l) * params.m_tiles + m) * params.n_tiles + n;
      if (params.peer_timeline && tile_id >= 0 &&
          tile_id < params.peer_timeline_capacity) {
        peer_event = params.peer_timeline + tile_id;
        acquire_timestamps = peer_event->acquire;
      }
    }
#endif
    if (lane == 0) {
      if constexpr (SystemScope) {
        wait_acquire_system_single_lane(
            tile_ready + first_peer * kReadyFlagStride, target);
      } else {
        wait_acquire_gpu_single_lane(
            tile_ready + first_peer * kReadyFlagStride, target);
      }
#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        const uint64_t now = read_global_timer();
        if (peer_event) {
          peer_event->m_tile = m;
          peer_event->n_tile = static_cast<int32_t>(cute::get<1>(block_coord));
          peer_event->batch = static_cast<int32_t>(cute::get<3>(block_coord));
          peer_event->acquire[first_peer] = now;
          peer_event->valid = 1;
        }
        const int32_t cta = static_cast<int32_t>(blockIdx.x);
        if (params.timeline && cta < params.timeline_capacity) {
          atomicCAS(
              reinterpret_cast<unsigned long long*>(
                  &params.timeline[cta].active_start),
              0ull,
              static_cast<unsigned long long>(now));
          }
      }
#endif
    }
#if FUSE_ENABLE_PROFILING
    auto ready_k_iter = make_ready_k_iterator<Instrumented, SystemScope>(
#else
    auto ready_k_iter = make_ready_k_iterator<SystemScope>(
#endif
        k_iter, tile_ready, kReadyFlagStride, target,
        params.k_tiles_per_peer, first_k_tile, k_tiles
#if FUSE_ENABLE_PROFILING
        , acquire_timestamps
#endif
        );
    Base::load(
        static_cast<const BaseParams&>(params), pipeline, write_state,
        inputs, block_coord, ready_k_iter, k_tiles, lane,
        block_rank_in_cluster, storage);
  }
};

// Publishes one epoch after the stock CUTLASS TMA epilogue has made the D tile
// globally visible. It deliberately does not perform communication itself.
template <class Base>
struct SignalingEpilogue : Base {
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;
  using SharedStorage = typename Base::SharedStorage;

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

  template <class ProblemShape>
  static Params to_underlying_arguments(
      const ProblemShape& problem,
      const Arguments& args,
      void* workspace) {
    Params params{};
    static_cast<BaseParams&>(params) = Base::to_underlying_arguments(
        problem, static_cast<const BaseArguments&>(args), workspace);
    params.ready = args.ready;
    params.m_tiles = args.m_tiles;
    params.n_tiles = args.n_tiles;
    params.epoch = args.epoch;
    return params;
  }

  template <class ProblemShape>
  static bool can_implement(const ProblemShape& problem, const Arguments& args) {
    return Base::can_implement(problem, static_cast<const BaseArguments&>(args));
  }

  template <class ProblemShape>
  static size_t get_workspace_size(const ProblemShape& problem, const Arguments& args) {
    return Base::get_workspace_size(problem, static_cast<const BaseArguments&>(args));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      const ProblemShape& problem,
      const Arguments& args,
      void* workspace,
      cudaStream_t stream,
      cutlass::CudaHostAdapter* adapter = nullptr) {
    return Base::initialize_workspace(
        problem, static_cast<const BaseArguments&>(args), workspace, stream, adapter);
  }

  SignalingEpilogue() = default;

  CUTLASS_HOST_DEVICE SignalingEpilogue(
      const Params& params,
      typename Base::TensorStorage& storage)
      : Base(static_cast<const BaseParams&>(params), storage), params_(&params) {}

  template <
      class ProblemShapeMNKL,
      class TileShapeMNK,
      class TileCoordMNKL,
      class AccEngine,
      class AccLayout,
      class TiledMma>
  CUTLASS_DEVICE auto store(
      typename Base::LoadPipeline load_pipeline,
      typename Base::LoadPipelineState load_state,
      typename Base::StorePipeline store_pipeline,
      typename Base::StorePipelineState store_state,
      ProblemShapeMNKL problem,
      TileShapeMNK tile_shape,
      TileCoordMNKL tile_coord,
      cute::Tensor<AccEngine, AccLayout> accumulators,
      TiledMma tiled_mma,
      int thread_idx,
      typename Base::TensorStorage& storage,
      int subtile_idx = -1) {
    auto states = Base::store(
        load_pipeline,
        load_state,
        store_pipeline,
        store_state,
        problem,
        tile_shape,
        tile_coord,
        accumulators,
        tiled_mma,
        thread_idx,
        storage,
        subtile_idx);

    if (thread_idx == 0 && params_->ready) {
      // ready is consumed by a different CTA.  The `.read` TMA wait used by
      // CUTE is sufficient for SMEM reuse, but it does not guarantee that D
      // is visible in global memory.  Fully drain this tile before release.
      tma_store_wait_all();
      const int32_t m = static_cast<int32_t>(cute::get<0>(tile_coord));
      const int32_t n = static_cast<int32_t>(cute::get<1>(tile_coord));
      const int32_t l = static_cast<int32_t>(cute::get<3>(tile_coord));
      const bool valid_tile =
          m >= 0 && m < params_->m_tiles && n >= 0 && n < params_->n_tiles;
      const int32_t tile =
          (l * params_->m_tiles + m) * params_->n_tiles + n;
      if (valid_tile) {
        store_release_system(
            params_->ready + tile * kReadyFlagStride, params_->epoch);
      }
    }
    return states;
  }

 private:
  const Params* params_ = nullptr;
};

}  // namespace fuse::detail
