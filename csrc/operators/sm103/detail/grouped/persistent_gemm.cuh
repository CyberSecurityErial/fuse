// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Grouped scheduling and private CTASP kernel; independent of Projection.
#include <cooperative_groups.h>
#include "producer_consumer.cuh"
#include "fuse/profiling/grouped.cuh"
#include "fuse/arch/common.cuh"
#include <cute/tensor.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>

namespace fuse::detail {

// A distinct type scopes the CUTLASS scheduler specialization to our grouped
// kernels. Ordinary SM103 dense kernels and stock grouped kernels are intact.
struct GroupedProblemShape
    : cutlass::gemm::GroupProblemShape<cute::Shape<int32_t,int32_t,int32_t>> {};
struct SwappedGroupedProblemShape : GroupedProblemShape {};

template <int Stages, bool SwapAB = false>
class PersistentTileSchedulerSm100Grouped
    : public cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100Group<
          GroupedProblemShape, Stages> {
  using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100Group<
      GroupedProblemShape, Stages>;
 public:
  using typename Base::WorkTileInfo;
  using typename Base::CLCResponse;
  using typename Base::RasterOrderOptions;
  static constexpr bool IsDynamicPersistent = false;
  struct Arguments : Base::Arguments {
    const int64_t* row_tile_offsets = nullptr;
    int32_t n = 0;
    int32_t compute_ctas = 0, block_offset = 0;
    int32_t window_m = 0;
  };
  struct Params : Base::Params {
    GroupedTileOrder order{};
    int32_t compute_ctas = 0, block_offset = 0;
  };
  static bool can_implement(const Arguments& a) {
    return a.row_tile_offsets && a.n > 0 && a.compute_ctas > 0 && a.block_offset >= 0 &&
        (a.max_swizzle_size == 1 || a.max_swizzle_size == 2 ||
         a.max_swizzle_size == 4 || a.max_swizzle_size == 8) &&
        (a.raster_order == RasterOrderOptions::AlongM ||
         a.raster_order == RasterOrderOptions::AlongN);
  }
  template <class Tile, class Atom, class Cluster>
  static Params to_underlying_arguments(GroupedProblemShape shapes, Tile, Atom, Cluster,
      const cutlass::KernelHardwareInfo&, const Arguments& a, void* = nullptr) {
    static_assert(cute::size(Cluster{}) == 1 && cute::size(Atom{}) == 1);
    Params p{};
    p.order = {a.row_tile_offsets, shapes.groups(),
        (a.n + cute::size<SwapAB ? 0 : 1>(Tile{}) - 1) / cute::size<SwapAB ? 0 : 1>(Tile{}),
        a.max_swizzle_size, a.raster_order == RasterOrderOptions::AlongN};
    p.compute_ctas = a.compute_ctas;
    p.order.window_m = a.window_m;
    p.block_offset = a.block_offset;
    // Do NOT shrink hw_info.sm_count to this budget. Ptr-array CUTLASS uses
    // physical blockIdx (including the comm prefix) to index TMA workspace.
    // The outer launch validates total grid <= physical SM count/occupancy.
    return p;
  }
  template <class Tile, class Atom, class Cluster>
  static dim3 get_grid_shape(const Params& p, GroupedProblemShape, Tile, Atom, Cluster,
      const cutlass::KernelHardwareInfo&) {
    return dim3(p.compute_ctas, 1, 1);
  }
  CUTLASS_DEVICE static bool valid_initial_worker(const Params& p, uint64_t block) {
    return block >= uint64_t(p.block_offset) &&
        block - p.block_offset < uint64_t(p.compute_ctas) &&
        block - p.block_offset < uint64_t(p.order.tiles());
  }
  CUTLASS_DEVICE PersistentTileSchedulerSm100Grouped(CLCResponse* response, const Params& p, dim3)
      : Base(), p_(p), response_(response), current_(int64_t(blockIdx.x) - p.block_offset) {}
  template <class Cluster>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(Cluster) const { return current_work(); }
  template <class Pipeline, class State>
  CUTLASS_DEVICE auto advance_to_next_work(Pipeline& pipeline, State state, uint32_t count = 1) {
    current_ += int64_t(p_.compute_ctas) * count;
    const auto work = current_work();
    // Unlike the ordinary dense static kernel, the grouped kernel has an
    // active scheduler warp and a shared response pipeline. Preserve BOTH
    // producer and consumer halves: only this warp decodes subsequent work,
    // then mainload/MMA/epilogue consume the same response and release it.
    pipeline.producer_acquire(state);
    if (cute::elect_one_sync()) {
      response_[state.index()] = work;
      cutlass::arch::fence_view_async_shared();
      pipeline.producer_commit(state);
    }
    return cute::make_tuple(work, true);
  }
  template <class Pipeline, class State>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo, Pipeline& pipeline, State state) {
    pipeline.consumer_wait(state);
    const auto work = response_[state.index()];
    cutlass::arch::fence_view_async_shared();
    pipeline.consumer_release(state);
    return cute::make_tuple(work, true);
  }
 private:
  CUTLASS_DEVICE WorkTileInfo current_work() const {
    const auto t = p_.order.decode(current_);
    // Decode in logical (token, feature) coordinates for BOTH layouts. Swap
    // only at the CUTLASS boundary: communication order and ready IDs stay put.
    return {SwapAB ? t.n : t.m, SwapAB ? t.m : t.n, t.expert, int(t.valid)};
  }
  const Params& p_;
  CLCResponse* response_;
  int64_t current_;
};

// Private copy of the Projection CTASP wrapper (v24.1, 7671339). Keep its
// role dispatch, shared-storage maximum, and collective grid join here so
// grouped scheduling/resource tuning never changes Projection hot paths.
template <class GemmKernel, class CommOp>
struct GroupedMonolithicGemm {
  using ArchTag = typename GemmKernel::ArchTag;
  using TileShape = typename GemmKernel::TileShape;
  using ClusterShape = typename GemmKernel::ClusterShape;
  static constexpr int MaxThreadsPerBlock = GemmKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static_assert(MaxThreadsPerBlock == 256);
  static_assert(MaxThreadsPerBlock >= CommOp::kMinThreads);
  static_assert(cute::size(typename GemmKernel::ClusterShape{}) == 1);
  static constexpr size_t SharedStorageSize =
      sizeof(typename GemmKernel::SharedStorage) > CommOp::SharedStorageBytes
          ? sizeof(typename GemmKernel::SharedStorage) : CommOp::SharedStorageBytes;

  struct alignas(128) SharedStorage {
    char bytes[SharedStorageSize];
  };

  struct Arguments {
#if FUSE_ENABLE_PROFILING
    GroupedProfile profile{};
#endif
    typename GemmKernel::Arguments gemm;
    typename CommOp::Arguments comm;
    int32_t num_comm_ctas = 0;
  };

  struct Params {
#if FUSE_ENABLE_PROFILING
    GroupedProfile profile{};
#endif
    typename GemmKernel::Params gemm;
    typename CommOp::Params comm;
    int32_t compute_ctas = 0;
    int32_t num_comm_ctas = 0;
  };

  static bool can_implement(const Arguments& args) {
    return args.num_comm_ctas > 0 && args.gemm.hw_info.sm_count > 0 &&
        args.gemm.scheduler.block_offset == args.num_comm_ctas &&
        args.comm.params.num_comm_ctas == args.num_comm_ctas &&
        GemmKernel::can_implement(args.gemm) && CommOp::can_implement(args.comm);
  }

  static size_t get_workspace_size(const Arguments& args) {
    return GemmKernel::get_workspace_size(args.gemm);
  }

  static cutlass::Status initialize_workspace(
      const Arguments& args, void* workspace, cudaStream_t stream) {
    return GemmKernel::initialize_workspace(args.gemm, workspace, stream);
  }

  static Params to_underlying_arguments(const Arguments& args, void* workspace) {
    Params params{};
#if FUSE_ENABLE_PROFILING
    params.profile = args.profile;
#endif
    params.gemm = GemmKernel::to_underlying_arguments(args.gemm, workspace);
    params.comm = CommOp::to_underlying_arguments(args.comm);
    params.num_comm_ctas = args.num_comm_ctas;
    const dim3 compute_grid = GemmKernel::get_grid_shape(params.gemm);
    params.compute_ctas = static_cast<int32_t>(
        static_cast<uint64_t>(compute_grid.x) * compute_grid.y * compute_grid.z);
    return params;
  }

  static dim3 get_grid_shape(const Params& params) {
    return dim3(params.compute_ctas + params.num_comm_ctas, 1, 1);
  }

  static dim3 get_block_shape() { return GemmKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
#if FUSE_ENABLE_PROFILING
    if (params.profile.roles && threadIdx.x == 0)
      params.profile.roles[blockIdx.x].begin = read_global_timer();
#endif
    if (static_cast<int32_t>(blockIdx.x) < params.num_comm_ctas) {
      CommOp{}(params.comm, smem, static_cast<int32_t>(blockIdx.x),
               params.num_comm_ctas);
    } else if (GemmKernel::TileScheduler::valid_initial_worker(
                   params.gemm.scheduler, blockIdx.x)) {
      GemmKernel{}(params.gemm, smem);
    }
#if FUSE_ENABLE_PROFILING
    if (params.profile.roles) {
      __syncthreads();  // Diagnostic role endpoint means ALL CTA threads returned.
      if (threadIdx.x == 0) params.profile.roles[blockIdx.x].role_end = read_global_timer();
    }
#endif
    if constexpr (CommOp::kNeedsGridFinalize) {
      cooperative_groups::this_grid().sync();
      CommOp{}.finalize(params.comm);
    }
#if FUSE_ENABLE_PROFILING
    if (params.profile.roles) {
      __syncthreads();
      if (threadIdx.x == 0) params.profile.roles[blockIdx.x].end = read_global_timer();
    }
#endif
  }
};

}  // namespace fuse::detail

namespace cutlass::gemm::kernel::detail {
template <class Tile, class Cluster, uint32_t Stages>
struct TileSchedulerSelector<GroupScheduler, arch::Sm100, Tile, Cluster, Stages,
    fuse::detail::GroupedProblemShape> {
  using Scheduler = fuse::detail::PersistentTileSchedulerSm100Grouped<Stages>;
};
template <class Tile, class Cluster, uint32_t Stages>
struct TileSchedulerSelector<GroupScheduler, arch::Sm100, Tile, Cluster, Stages,
    fuse::detail::SwappedGroupedProblemShape> {
  using Scheduler = fuse::detail::PersistentTileSchedulerSm100Grouped<Stages, true>;
};
}  // namespace cutlass::gemm::kernel::detail
