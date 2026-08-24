// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/ready_k_iterator.cuh"
#include "fuse/system_barrier.cuh"
#include "fuse/kernels.h"

#include <cute/arch/copy_sm90.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>

namespace fuse::detail {

// Readiness adapter for A2A -> strided-batched GEMM. RHS readiness is
// independent of the output M tile: every local GEMM group maps to one local
// input group, and every peer contributes one contiguous K interval. Keeping
// one flag per (outer batch, local input group, peer, K group) avoids duplicate
// publication for aliases and output tiles while allowing the TMA producer to
// start as soon as a contiguous subset of a peer shard is visible.
template <class Base>
struct A2AReadyMainloop : Base {
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;

  struct Arguments : BaseArguments {
    const uint32_t* ready = nullptr;
    int32_t world_size = 0;
    int32_t local_gemm_groups = 0;
    int32_t local_input_groups = 0;
    int32_t ready_groups_per_peer = 1;
    int32_t arrivals_per_group = 0;
    int32_t k_tiles_per_group = 0;
    uint32_t epoch = 0;
  };

  struct Params : BaseParams {
    const uint32_t* ready = nullptr;
    int32_t world_size = 0;
    int32_t local_gemm_groups = 0;
    int32_t local_input_groups = 0;
    int32_t ready_groups_per_peer = 1;
    int32_t arrivals_per_group = 0;
    int32_t k_tiles_per_group = 0;
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
    params.local_gemm_groups = args.local_gemm_groups;
    params.local_input_groups = args.local_input_groups;
    params.ready_groups_per_peer = args.ready_groups_per_peer;
    params.arrivals_per_group = args.arrivals_per_group;
    params.k_tiles_per_group = args.k_tiles_per_group;
    params.epoch = args.epoch;
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
    const int32_t l = static_cast<int32_t>(cute::get<3>(block_coord));
    const int32_t local_group = l % params.local_gemm_groups;
    const int32_t batch = l / params.local_gemm_groups;
    const int32_t aliases =
        params.local_gemm_groups / params.local_input_groups;
    const int32_t local_input_group = local_group / aliases;
    const int32_t ready_group =
        batch * params.local_input_groups + local_input_group;
    const int32_t total_ready_groups =
        params.world_size * params.ready_groups_per_peer;
    const uint32_t* tile_ready = params.ready +
        static_cast<int64_t>(ready_group) * total_ready_groups *
            kReadyFlagStride;
    const uint32_t target = params.epoch * params.arrivals_per_group;

    if (params.k_tiles_per_group > 0) {
      const int32_t first_k_tile = static_cast<int32_t>(*k_iter);
      const int32_t first_ready_group =
          first_k_tile / params.k_tiles_per_group;
      if (lane == 0) {
        wait_acquire_gpu_single_lane(
            tile_ready + first_ready_group * kReadyFlagStride, target);
      }
      auto ready_k_iter = make_ready_k_iterator(
          k_iter,
          tile_ready,
          kReadyFlagStride,
          target,
          params.k_tiles_per_group,
          first_k_tile,
          k_tiles);
      Base::load(
          static_cast<const BaseParams&>(params),
          pipeline,
          write_state,
          inputs,
          block_coord,
          ready_k_iter,
          k_tiles,
          lane,
          block_rank_in_cluster,
          storage);
      return;
    }

    for (int32_t group = 0; group < total_ready_groups; ++group) {
      wait_acquire_gpu(
          tile_ready + group * kReadyFlagStride, target, lane);
    }
    Base::load(
        static_cast<const BaseParams&>(params),
        pipeline,
        write_state,
        inputs,
        block_coord,
        k_iter,
        k_tiles,
        lane,
        block_rank_in_cluster,
        storage);
  }
};

// Readiness adapter for inverse head-to-sequence A2A -> dense GEMM. Every
// peer contributes one contiguous K shard of operand A for a given M tile.
// The elected TMA producer waits at peer/K boundaries while the stock CUTLASS
// load pipeline, WGMMA, and epilogue remain unchanged.
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

template <
    class Base,
    int32_t MTilesPerReady = 1,
    bool Instrumented = false>
struct A2ALhsReadyMainloop : Base {
  static_assert(MTilesPerReady > 0);
  using BaseArguments = typename Base::Arguments;
  using BaseParams = typename Base::Params;

  struct Arguments : BaseArguments, A2ALhsTimelineArguments<Instrumented> {
    const uint32_t* ready = nullptr;
    int32_t world_size = 0;
    int32_t m_tiles = 0;
    int32_t arrivals_per_peer = 0;
    int32_t k_tiles_per_peer = 0;
    uint32_t epoch = 0;
  };

  struct Params : BaseParams, A2ALhsTimelineArguments<Instrumented> {
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
    if constexpr (Instrumented) {
      params.timeline = args.timeline;
      params.timeline_capacity = args.timeline_capacity;
      params.peer_timeline = args.peer_timeline;
      params.peer_timeline_capacity = args.peer_timeline_capacity;
      params.n_tiles = args.n_tiles;
    }
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
    if (lane == 0) {
      wait_acquire_gpu_single_lane(
          tile_ready + first_peer * kReadyFlagStride, target);
      if constexpr (Instrumented) {
        const uint64_t now = read_ready_timer();
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
    }
    auto ready_k_iter = make_ready_k_iterator<Instrumented>(
        k_iter, tile_ready, kReadyFlagStride, target,
        params.k_tiles_per_peer, first_k_tile, k_tiles,
        acquire_timestamps);
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
      cute::tma_store_wait<0>();
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
