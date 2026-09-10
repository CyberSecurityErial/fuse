// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// SM103-specific diagnostics; shared profiling records remain one directory up.

// Opt-in OProj pipeline diagnostics. No production ready granularity, MMA
// schedule, or synchronization changes. The harness samples GPU0 and three
// persistent workers only; records belong to logical (M,N,K), not clock bins.
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cstdint>

namespace fuse::detail {

struct OprojReadyRecord {
  uint64_t begin = 0, end = 0, joined = 0;
  uint32_t cache_hit = 0;
};

struct OprojMmaStageRecord {
  uint64_t wait_begin = 0, wait_end = 0, issue_end = 0;
};

struct OprojPipelineRecord {
  uint64_t first_input = 0;
  OprojReadyRecord ready[kMaxWorldSize]{};
  uint64_t mma_begin = 0, tmem_acquired = 0, mma_return = 0;
  uint64_t epi_begin = 0, acc_wait_begin = 0, acc_wait_end = 0;
  uint64_t tmem_release_begin = 0, tmem_release_end = 0, epi_return = 0;
  int32_t cta = -1;
};

struct OprojPipelineView {
  OprojPipelineRecord* tiles = nullptr;
  OprojMmaStageRecord* stages = nullptr;
  int32_t m_tiles = 0, n_tiles = 0, k_tiles = 0;
  int32_t comm_ctas = 0, compute_ctas = 0, swizzle = 1;
  bool all_workers = false;

#if defined(__CUDACC__)
  template <class Coord>
  CUTLASS_DEVICE int64_t tile_index(const Coord& coord) const {
    if (!tiles) return -1;
    const int worker = static_cast<int>(blockIdx.x) - comm_ctas;
    const int tail = compute_ctas - (swizzle < compute_ctas ? swizzle : compute_ctas);
    if (!all_workers && worker != 0 && worker != 1 && worker != tail) return -1;
    const int m = static_cast<int>(cute::get<0>(coord));
    const int n = static_cast<int>(cute::get<1>(coord));
    if (m < 0 || m >= m_tiles || n < 0 || n >= n_tiles || cute::get<3>(coord) != 0) return -1;
    return static_cast<int64_t>(m) * n_tiles + n;
  }
#endif
};

// Bind only on the host thread that enqueues this rank. The copied device
// view outlives the binding; its buffers live until stream completion.
extern thread_local const OprojPipelineView* oproj_pipeline_sink;
class OprojPipelineBinding {
 public:
  explicit OprojPipelineBinding(const OprojPipelineView* view)
      : previous_(oproj_pipeline_sink) { oproj_pipeline_sink = view; }
  ~OprojPipelineBinding() { oproj_pipeline_sink = previous_; }
  OprojPipelineBinding(const OprojPipelineBinding&) = delete;
  OprojPipelineBinding& operator=(const OprojPipelineBinding&) = delete;
 private:
  const OprojPipelineView* previous_;
};

#if defined(__CUDACC__)
CUTLASS_DEVICE uint64_t oproj_timestamp() {
  asm volatile("" ::: "memory");
  const uint64_t now = read_global_timer();
  asm volatile("" ::: "memory");
  return now;
}

// The stock BF16 SM100 mma() takes a concrete MainloopPipeline, so a derived
// pipeline would be sliced and cannot observe consumer_wait(). This small
// diagnostic mirror preserves its instruction/barrier order (pinned CUTLASS
// 57e3cfb, sm100_mma_warpspecialized.hpp). No extra GPU wait/fence is inserted.
// TODO: Replace this mirror with upstream observer hooks if CUTLASS adds them.
// issue_end/return mean instruction SUBMISSION, never tensor execution end.
template <class Pipelines, class States, class Accumulators, class Inputs>
CUTLASS_DEVICE auto profile_oproj_mma(
    Pipelines pipelines, States states, Accumulators accumulators_pair,
    Inputs inputs, int k_tiles, OprojPipelineRecord* record,
    OprojMmaStageRecord* stages) {
  auto accumulators = cute::get<0>(accumulators_pair);
  auto [tiled_mma, tCrA, tCrB] = inputs;
  auto [mainloop_pipeline, accumulator_pipeline] = pipelines;
  auto [mainloop_state, accumulator_state] = states;
  const bool writer = threadIdx.x % 32 == 0;
  uint32_t skip_wait = k_tiles <= 0;
  auto token = mainloop_pipeline.consumer_try_wait(mainloop_state, skip_wait);
  tiled_mma.accumulate_ = cute::UMMA::ScaleOut::Zero;
  const uint64_t begin = writer ? oproj_timestamp() : 0;
  accumulator_pipeline.producer_acquire(accumulator_state);
  if (writer) {
    const uint64_t acquired = oproj_timestamp();
    record->mma_begin = begin;
    record->tmem_acquired = acquired;
    record->cta = static_cast<int>(blockIdx.x);
  }
  int k = 0;
  CUTLASS_PRAGMA_NO_UNROLL
  while (k_tiles > 0) {
    const uint64_t wait_begin = writer && stages ? oproj_timestamp() : 0;
    mainloop_pipeline.consumer_wait(mainloop_state, token);
    const uint64_t wait_end = writer && (stages || k == 0) ? oproj_timestamp() : 0;
    if (writer && k == 0) record->first_input = wait_end;
    const int read_stage = mainloop_state.index();
    auto current = mainloop_state;
    ++mainloop_state;
    --k_tiles;
    skip_wait = k_tiles <= 0;
    token = mainloop_pipeline.consumer_try_wait(mainloop_state, skip_wait);
    CUTLASS_PRAGMA_UNROLL
    for (int block = 0; block < cute::size<2>(tCrA); ++block) {
      cute::gemm(tiled_mma, tCrA(cute::_,cute::_,block,read_stage),
                 tCrB(cute::_,cute::_,block,read_stage), accumulators);
      tiled_mma.accumulate_ = cute::UMMA::ScaleOut::One;
    }
    mainloop_pipeline.consumer_release(current);
    if (writer && stages) {
      const uint64_t issued = oproj_timestamp();
      stages[k] = {wait_begin, wait_end, issued};
    }
    ++k;
  }
  if (writer) record->mma_return = oproj_timestamp();
  return mainloop_state;
}

// Independent GEMM observer: inherit the stock load() unchanged. Unlike the
// fused ready mainloop this introduces no peer partitioning or ready checks.
// A null stages buffer records tile boundaries only, not every K-stage clock.
template <class Base>
struct OprojPureMainloopObserver : Base {
  struct Arguments : Base::Arguments { OprojPipelineView probe{}; };
  struct Params : Base::Params { OprojPipelineView probe{}; };
  template <class Problem>
  static Params to_underlying_arguments(const Problem& problem, const Arguments& args,
      void* workspace, const cutlass::KernelHardwareInfo& hardware = {}) {
    Params params{};
    static_cast<typename Base::Params&>(params) = Base::to_underlying_arguments(
        problem, static_cast<const typename Base::Arguments&>(args), workspace, hardware);
    params.probe = args.probe;
    return params;
  }
  template <class Cluster>
  CUTLASS_DEVICE OprojPureMainloopObserver(const Params& params, Cluster cluster, uint32_t rank)
      : Base(static_cast<const typename Base::Params&>(params), cluster, rank), probe_(params.probe) {}
  template <class Pipelines, class States, class Accumulators, class Inputs, class Coord>
  CUTLASS_DEVICE auto mma(Pipelines pipelines, States states, Accumulators accumulators,
                         Inputs inputs, Coord coord, int k_tiles) {
    const int64_t index = probe_.tile_index(coord);
    if (index < 0) return Base::mma(pipelines, states, accumulators, inputs, coord, k_tiles);
    return profile_oproj_mma(pipelines, states, accumulators, inputs, k_tiles,
        probe_.tiles + index, probe_.stages ? probe_.stages + index * k_tiles : nullptr);
  }
 private:
  OprojPipelineView probe_;
};

// Observe the existing epilogue accumulator barrier, not a new completion
// wait. acc_wait_end is when this epilogue lane OBSERVED MMA completion: an
// upper bound on completion, potentially late if epilogue was busy earlier.
// Likewise TMEM release is this lane's arrival, not all consumers' completion.
template <class Base>
struct OprojAccumulatorObserver : Base {
  OprojPipelineRecord* record;
  bool writer;
  CUTLASS_DEVICE OprojAccumulatorObserver(Base base, OprojPipelineRecord* out, bool lane)
      : Base(base), record(out), writer(lane) {}
  template <class State, class Token>
  CUTLASS_DEVICE void consumer_wait(State state, Token token) {
    const uint64_t begin = writer ? oproj_timestamp() : 0;
    Base::consumer_wait(state, token);
    if (writer) {
      const uint64_t end = oproj_timestamp();
      record->acc_wait_begin = begin;
      record->acc_wait_end = end;
    }
  }
  template <class State>
  CUTLASS_DEVICE void consumer_release(State state) {
    const uint64_t begin = writer ? oproj_timestamp() : 0;
    Base::consumer_release(state);
    if (writer) {
      const uint64_t end = oproj_timestamp();
      record->tmem_release_begin = begin;
      record->tmem_release_end = end;
    }
  }
};

template <class Base>
struct OprojEpilogueObserver : Base {
  struct Arguments : Base::Arguments { OprojPipelineView probe{}; };
  struct Params : Base::Params { OprojPipelineView probe{}; };
  template <class Problem>
  static Params to_underlying_arguments(const Problem& problem, const Arguments& args, void* workspace) {
    Params params{};
    static_cast<typename Base::Params&>(params) = Base::to_underlying_arguments(
        problem, static_cast<const typename Base::Arguments&>(args), workspace);
    params.probe = args.probe;
    return params;
  }
  CUTLASS_DEVICE OprojEpilogueObserver(const Params& params, typename Base::TensorStorage& storage)
      : Base(static_cast<const typename Base::Params&>(params), storage), probe_(params.probe) {}

  template <bool ReuseTmem = false, class AccPipeline, class AccState,
            class Problem, class Tile, class Coord, class MmaTile, class Mma,
            class Engine, class Layout>
  CUTLASS_DEVICE auto store(
      typename Base::LoadPipeline load, typename Base::LoadPipelineState load_state,
      typename Base::StorePipeline store, typename Base::StorePipelineState store_state,
      AccPipeline acc, AccState acc_state, Problem problem, Tile tile, Coord coord,
      MmaTile mma_tile, Mma mma, cute::Tensor<Engine, Layout> accumulators,
      typename Base::TensorStorage& storage) {
    const int64_t index = probe_.tile_index(coord);
    if (index < 0) return Base::template store<ReuseTmem>(load, load_state, store, store_state,
        acc, acc_state, problem, tile, coord, mma_tile, mma, accumulators, storage);
    auto* record = probe_.tiles + index;
    const bool writer = threadIdx.x % Base::ThreadCount == 0;
    if (writer) record->epi_begin = oproj_timestamp();
    auto states = Base::template store<ReuseTmem>(load, load_state, store, store_state,
        OprojAccumulatorObserver<AccPipeline>(acc, record, writer), acc_state,
        problem, tile, coord, mma_tile, mma, accumulators, storage);
    if (writer) record->epi_return = oproj_timestamp();
    return states;
  }
 private:
  OprojPipelineView probe_;
};
#endif
}  // namespace fuse::detail
#endif
