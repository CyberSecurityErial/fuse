// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cutlass/kernel_hardware_info.hpp>

// CUTLASS boundary adapters: acquire a delivered input panel, or publish a
// completed output tile. Traversal and peer transport live in other modules.
#include "producer_consumer.cuh"
#include "fuse/profiling/grouped.cuh"
#include "fuse/arch/common.cuh"
#include <cute/tensor.hpp>

namespace fuse::detail {

// Y[M,N] = X[M,K] W[N,K]^T becomes Y^T[N,M] = W[N,K] X[M,K]^T.
// As in MegaMoE, the varying token extent now occupies UMMA N. Round the
// current task's valid tokens to 16, not the fixed 128-row delivery capacity.
// Only the MMA descriptor is narrowed; TMA, TMEM allocation, epilogue masks and
// full-K ready publication retain their original capacities/lifetimes. There
// is no extra transpose kernel or per-K polling. Output is a column-major
// view of the SAME row-major Y allocation. One-CTA CUTLASS is deliberately
// retained to isolate this change from MegaMoE's two-CTA multicast design.
template <class Base, bool TrimTokens = true>
struct GroupedSwapABMainloop : Base {
  using Shape = cute::Shape<int32_t,int32_t,int32_t>;
  struct Params : Base::Params { const Shape* shapes = nullptr; };
  template <class Problem>
  static Params to_underlying_arguments(const Problem& shape, const typename Base::Arguments& a,
      void* workspace, const cutlass::KernelHardwareInfo& hw = {}) {
    Params p{};
    static_cast<typename Base::Params&>(p) = Base::to_underlying_arguments(shape,a,workspace,hw);
    p.shapes = shape.problem_shapes;
    return p;
  }
  template <class Cluster>
  CUTLASS_DEVICE GroupedSwapABMainloop(const Params& p, Cluster cluster, uint32_t rank)
      : Base(p,cluster,rank), shapes_(p.shapes) {}
  template <class Pipelines, class States, class Accumulator, class Inputs, class Coord>
  CUTLASS_DEVICE auto mma(Pipelines pipelines, States states, Accumulator& accum,
      const Inputs& inputs, Coord coord, int k_count) {
    auto narrowed = inputs;
    if constexpr (TrimTokens) {
      const int tokens = cute::get<1>(shapes_[int(cute::get<3>(coord))]);
      const int remaining = tokens - int(cute::get<1>(coord)) * 128;
      const int valid = remaining < 128 ? remaining : 128;
      // InstrDescriptor stores N/8; BF16 M128 requires N in multiples of16.
      cute::get<0>(narrowed).idesc_.n_dim_ = ((valid + 15) / 16 * 16) >> 3;
    }
    return Base::mma(pipelines,states,accum,narrowed,coord,k_count);
  }
 private:
  const Shape* shapes_;
};

// Ptr-array CUTLASS intentionally passes L=0 to load(): the expert has already
// been installed in the TMA descriptor. Retain that expert at descriptor update
// instead of treating load's L=0 as expert zero. A full M panel is acquired once
// across both load calls (prologue/remainder), not at every K tile.
template <class Base, bool SwapAB = false>
struct GroupedInputReadyMainloop : Base {
  using Base::Base;
  struct Arguments : Base::Arguments {
#if FUSE_ENABLE_PROFILING
    GroupedProfile profile{};
#endif
    const int64_t* row_tile_offsets = nullptr;
    const uint32_t* ready = nullptr;
    const uint32_t* epoch_ptr = nullptr;
    uint32_t epoch = 0;
    int32_t buffer_m = 0;
  };
  struct Params : Base::Params {
#if FUSE_ENABLE_PROFILING
    GroupedProfile profile{};
#endif
    const int64_t* row_tile_offsets = nullptr;
    const uint32_t* ready = nullptr;
    const uint32_t* epoch_ptr = nullptr;
    uint32_t epoch = 0;
    int32_t buffer_m = 0;
  };
#if FUSE_ENABLE_PROFILING
  // Each warp has its own collective object. Only the elected load-warp lane
  // accumulates observations; all other objects retain checks==0. No atomics,
  // extra barrier, per-tile arrays, or global writes in the polling loop.
  CUTLASS_DEVICE ~GroupedInputReadyMainloop() {
    if (summary_output_ && summary_.checks)
      summary_output_[blockIdx.x] = summary_;
  }
#endif
  template <class Problem>
  static Params to_underlying_arguments(const Problem& shape, const Arguments& a,
      void* workspace, const cutlass::KernelHardwareInfo& hw = {}) {
    Params p{};
    static_cast<typename Base::Params&>(p) = Base::to_underlying_arguments(shape, a, workspace, hw);
    p.row_tile_offsets = a.row_tile_offsets; p.ready = a.ready; p.epoch = a.epoch;
    p.epoch_ptr = a.epoch_ptr;
    p.buffer_m = a.buffer_m;
#if FUSE_ENABLE_PROFILING
    p.profile = a.profile;
#endif
    return p;
  }
  template <class Problem>
  static bool can_implement(const Problem& shape, const Arguments& a) {
    return (!a.ready || (a.row_tile_offsets && (a.epoch || a.epoch_ptr))) && Base::can_implement(shape, a);
  }
  template <class Maps, class Problem>
  CUTLASS_DEVICE void tensormaps_perform_update(typename Base::TensorMapStorage& storage,
      const Params& p, const Maps& maps, Problem shape, int expert) {
    expert_ = expert;
    Base::tensormaps_perform_update(storage, p, maps, shape, expert);
  }
  template <class Inputs, class Coord, class Iterator>
  CUTLASS_DEVICE auto load(const Params& p, typename Base::MainloopPipeline pipeline,
      typename Base::MainloopPipelineState state, const Inputs& inputs,
      const Coord& coord, Iterator k, int count, bool changed) {
#if FUSE_ENABLE_PROFILING
    // One record per output tile's FIRST load() call. Prologue/remainder
    // reuse must not overwrite it. No new ready checks or synchronization.
    const int64_t profile_row = p.profile.tiles ? p.row_tile_offsets[expert_] + int(cute::get<SwapAB ? 1 : 0>(coord)) : -1;
    const int64_t profile_tile = profile_row * p.profile.n_tiles + int(cute::get<SwapAB ? 0 : 1>(coord));
    const bool record = p.profile.tiles && count > 0 && profile_tile != profiled_tile_;
    const bool leader = threadIdx.x % 32 == 0;
    const uint64_t wait_begin = record && leader ? read_global_timer() : 0;
    uint64_t observed = wait_begin;
    bool polled = false;
#endif
    if (p.ready && count > 0) {
      const int64_t row = p.row_tile_offsets[expert_] + int(cute::get<SwapAB ? 1 : 0>(coord));
      if (row != acquired_) {
        const uint32_t epoch = p.epoch_ptr ? *p.epoch_ptr : p.epoch;
        if (threadIdx.x % 32 == 0) {
#if FUSE_ENABLE_PROFILING
          const uint64_t summary_begin = p.profile.ready_summary ? read_global_timer() : 0;
#endif
          CUTLASS_PRAGMA_NO_UNROLL
          while (load_acquire_gpu(p.ready + row * kReadyFlagStride) < epoch)
            __nanosleep(64);
#if FUSE_ENABLE_PROFILING
          if (p.profile.ready_summary) {
            const uint64_t elapsed = read_global_timer() - summary_begin;
            summary_output_ = p.profile.ready_summary;
            if (!summary_.checks) summary_.first_wait_ns = elapsed;
            ++summary_.checks;
            summary_.wait_ns += elapsed;
            summary_.max_wait_ns = elapsed > summary_.max_wait_ns ? elapsed : summary_.max_wait_ns;
            summary_.waits_ge_1us += elapsed >= 1000;
          }
          if (record) { observed = read_global_timer(); polled = true; }
#endif
        }
        __syncwarp();
        fence_proxy_async_global();
        acquired_ = row;
      }
    }
    // Only the activation coordinate wraps (A/M normally, B/N after swap).
    // Readiness and output coordinates remain logical. Physical buffers have
    // complete 128-row slots, including storage for the last partial tile.
    auto load_coord = coord;
    if (p.buffer_m) cute::get<SwapAB ? 1 : 0>(load_coord) = int(cute::get<SwapAB ? 1 : 0>(coord)) % p.buffer_m;
#if FUSE_ENABLE_PROFILING
    const uint64_t load_begin = record && leader ? read_global_timer() : 0;
    auto result = Base::load(p, pipeline, state, inputs, load_coord, k, count, changed);
    if (record) {
      if (leader && profile_row < p.profile.panel_capacity)
        p.profile.tiles[profile_tile] = {wait_begin, observed, load_begin, read_global_timer(),
            int(blockIdx.x), expert_, int(cute::get<SwapAB ? 1 : 0>(coord)), int(cute::get<SwapAB ? 0 : 1>(coord)), int(polled)};
      profiled_tile_ = profile_tile;
    }
    return result;
#else
    return Base::load(p, pipeline, state, inputs, load_coord, k, count, changed);
#endif
  }
 private:
  int expert_ = -1;
  int64_t acquired_ = -1;
#if FUSE_ENABLE_PROFILING
  int64_t profiled_tile_ = -1;
  GroupedReadySummary summary_{};
  GroupedReadySummary* summary_output_ = nullptr;
#endif
};

// Dispatch input lifetime, independent of Combine's output-ready protocol.
// Base::store has waited for this tile's final MMA before returning. Thus all
// its A reads have completed; one epilogue leader contributes one N consumer.
// Release/acquire RMW chains all readers before the writer reuses the slot.
// We deliberately do not release at load() return: TMA is asynchronous there.
template <class Base, bool SwapAB = false>
struct GroupedInputReleaseEpilogue : Base {
  struct Arguments : Base::Arguments {
    const int64_t* row_tile_offsets = nullptr;
    uint32_t* consumed = nullptr;
  };
  struct Params : Base::Params {
    const int64_t* row_tile_offsets = nullptr;
    uint32_t* consumed = nullptr;
  };
  template <class Problem>
  static Params to_underlying_arguments(const Problem& shape, const Arguments& a, void* workspace) {
    Params p{};
    static_cast<typename Base::Params&>(p) = Base::to_underlying_arguments(shape,a,workspace);
    p.row_tile_offsets=a.row_tile_offsets; p.consumed=a.consumed;
    return p;
  }
  CUTLASS_DEVICE GroupedInputReleaseEpilogue(const Params& p, typename Base::TensorStorage& storage)
      : Base(p,storage), p_(p) {}
  template <bool ReuseTmem = false, class AccPipeline, class AccState,
      class Problem, class CtaTile, class Coord, class MmaTile, class Mma,
      class AccEngine, class AccLayout, class Maps>
  CUTLASS_DEVICE auto store(typename Base::LoadPipeline load_pipe,
      typename Base::LoadPipelineState load_state, typename Base::StorePipeline store_pipe,
      typename Base::StorePipelineState store_state, AccPipeline acc_pipe, AccState acc_state,
      Problem shape, CtaTile cta_tile, Coord coord, MmaTile mma_tile, Mma mma,
      cute::Tensor<AccEngine,AccLayout> accum, typename Base::TensorStorage& storage, Maps maps) {
    auto states=Base::template store<ReuseTmem>(load_pipe,load_state,store_pipe,store_state,
        acc_pipe,acc_state,shape,cta_tile,coord,mma_tile,mma,accum,storage,maps);
    if (p_.consumed && threadIdx.x % Base::ThreadCount == 0) {
      auto* counter=p_.consumed+p_.row_tile_offsets[int(cute::get<3>(coord))]+int(cute::get<SwapAB ? 1 : 0>(coord));
      uint32_t previous;
      asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                   : "=r"(previous) : "l"(counter) : "memory");
    }
    return states;
  }
 private:
  const Params& p_;
};

// Publish one complete (expert,M,N) output tile, not one epilogue subtile.
// The remote routing CTA is local to this GPU, so GPU-scope release suffices.
// Its later remote-store completion requires a separate system-scope handoff.
template <class Base, bool SwapAB = false>
struct GroupedSignalingEpilogue : Base {
  struct Arguments : Base::Arguments {
    GroupedTileOrder order{};
    uint32_t* ready = nullptr;
    const uint32_t* epoch_ptr = nullptr;
    uint32_t epoch = 0;
  };
  struct Params : Base::Params {
    GroupedTileOrder order{};
    uint32_t* ready = nullptr;
    const uint32_t* epoch_ptr = nullptr;
    uint32_t epoch = 0;
  };
  template <class Problem>
  static Params to_underlying_arguments(const Problem& shape, const Arguments& a, void* workspace) {
    Params p{};
    static_cast<typename Base::Params&>(p) = Base::to_underlying_arguments(shape, a, workspace);
    p.order = a.order; p.ready = a.ready; p.epoch = a.epoch;
    p.epoch_ptr = a.epoch_ptr;
    return p;
  }
  template <class Problem>
  static bool can_implement(const Problem& shape, const Arguments& a) {
    return (!a.ready || (a.order.row_tile_offsets && a.order.n_tiles > 0 && (a.epoch || a.epoch_ptr))) &&
        Base::can_implement(shape, a);
  }
  CUTLASS_DEVICE GroupedSignalingEpilogue(const Params& p, typename Base::TensorStorage& storage)
      : Base(p, storage), p_(p) {}
  template <bool ReuseTmem = false, class AccPipeline, class AccState,
      class Problem, class CtaTile, class Coord, class MmaTile, class Mma,
      class AccEngine, class AccLayout, class Maps>
  CUTLASS_DEVICE auto store(typename Base::LoadPipeline load_pipe,
      typename Base::LoadPipelineState load_state, typename Base::StorePipeline store_pipe,
      typename Base::StorePipelineState store_state, AccPipeline acc_pipe, AccState acc_state,
      Problem shape, CtaTile cta_tile, Coord coord, MmaTile mma_tile, Mma mma,
      cute::Tensor<AccEngine, AccLayout> accum, typename Base::TensorStorage& storage, Maps maps) {
    auto states = Base::template store<ReuseTmem>(load_pipe, load_state, store_pipe, store_state,
        acc_pipe, acc_state, shape, cta_tile, coord, mma_tile, mma, accum, storage, maps);
    if (p_.ready && threadIdx.x % Base::ThreadCount < 32) {
      // TMA groups are per issuing thread; cover all lanes of the store warp.
      // .read would only make SMEM reusable, not make GMEM data consumable.
      tma_store_wait_all();
      __syncwarp();
      if (threadIdx.x % Base::ThreadCount == 0) {
        const auto i = p_.order.output_ready_index(int(cute::get<3>(coord)),
            int(cute::get<SwapAB ? 1 : 0>(coord)), int(cute::get<SwapAB ? 0 : 1>(coord)));
        store_release_gpu(p_.ready + i * kReadyFlagStride, p_.epoch_ptr ? *p_.epoch_ptr : p_.epoch);
      }
    }
    return states;
  }
 private:
  const Params& p_;
};

}  // namespace fuse::detail
