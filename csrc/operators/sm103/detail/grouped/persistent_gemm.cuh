// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Grouped scheduling and private CTASP kernel; independent of Projection.
#include <cooperative_groups.h>
#include "producer_consumer.cuh"
#include "communication.cuh"
#include "fuse/profiling/grouped.cuh"
#include "fuse/arch/common.cuh"
#include <cute/tensor.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <type_traits>

namespace fuse::detail {

// A distinct type scopes the CUTLASS scheduler specialization to our grouped
// kernels. Ordinary SM103 dense kernels and stock grouped kernels are intact.
struct GroupedProblemShape
    : cutlass::gemm::GroupProblemShape<cute::Shape<int32_t,int32_t,int32_t>> {};
struct SwappedGroupedProblemShape : GroupedProblemShape {};

template <int Stages, bool SwapAB = false, int SmMode = 1>
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
        a.compute_ctas % SmMode == 0 && a.block_offset % SmMode == 0 &&
        (a.max_swizzle_size == 1 || a.max_swizzle_size == 2 ||
         a.max_swizzle_size == 4 || a.max_swizzle_size == 8) &&
        (a.raster_order == RasterOrderOptions::AlongM ||
         a.raster_order == RasterOrderOptions::AlongN);
  }
  template <class Tile, class Atom, class Cluster>
  static Params to_underlying_arguments(GroupedProblemShape shapes, Tile, Atom, Cluster,
      const cutlass::KernelHardwareInfo&, const Arguments& a, void* = nullptr) {
    static_assert(SmMode == 1 || SmMode == 2);
    static_assert(cute::size(Cluster{}) == SmMode && cute::size(Atom{}) == SmMode);
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
        (block - p.block_offset) / SmMode < uint64_t(p.order.tiles());
  }
  CUTLASS_DEVICE PersistentTileSchedulerSm100Grouped(CLCResponse* response, const Params& p, dim3)
      : Base(), p_(p), response_(response),
        current_((int64_t(blockIdx.x) - p.block_offset) / SmMode),
        cluster_rank_((int32_t(blockIdx.x) - p.block_offset) % SmMode) {}
  template <class Cluster>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(Cluster) const { return current_work(); }
  template <class Pipeline, class State>
  CUTLASS_DEVICE auto advance_to_next_work(Pipeline& pipeline, State state, uint32_t count = 1) {
    current_ += int64_t(p_.compute_ctas / SmMode) * count;
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
    if (current_ < 0 || current_ >= p_.order.tiles()) return {-1,-1,-1,0};
    const int64_t row_tile = current_ / p_.order.n_tiles;
    const int lane = int(threadIdx.x) & 31;
    int expert = p_.order.experts;
    // Expert counts are dynamic, but all scheduler lanes seek the same panel.
    // Test 32 prefix boundaries in parallel instead of issuing log2(E)
    // dependent global loads independently from every lane. Empty experts are
    // handled by selecting the first boundary strictly greater than row_tile.
    for (int base=0; base<p_.order.experts; base+=32) {
      const int candidate=base+lane;
      const bool after=candidate<p_.order.experts &&
          p_.order.row_tile_offsets[candidate+1]>row_tile;
      const unsigned hits=__ballot_sync(0xffffffffu,after);
      if (hits) {
        expert=base+__ffs(hits)-1;
        break;
      }
    }
    const auto t = p_.order.decode_for_expert(current_,expert);
    // Decode in logical (token, feature) coordinates for BOTH layouts. Swap
    // only at the CUTLASS boundary: communication order and ready IDs stay put.
    const int physical_m=t.m*SmMode+cluster_rank_;
    return {SwapAB ? t.n : physical_m, SwapAB ? physical_m : t.n,
        t.expert, int(t.valid)};
  }
  const Params& p_;
  CLCResponse* response_;
  int64_t current_;
  int32_t cluster_rank_;
};

// Private copy of the Projection CTASP wrapper (v24.1, 7671339). Keep its
// role dispatch, shared-storage maximum, and collective grid join here so
// grouped scheduling/resource tuning never changes Projection hot paths.
template <class GemmKernel, class CommOp, class Selector = GroupedExplicitPreparePolicy,
    bool SeparateRoles = false>
struct GroupedMonolithicGemm {
  static constexpr bool DynamicPolicy =
      !std::is_same_v<Selector,GroupedExplicitPreparePolicy>;
  using ArchTag = typename GemmKernel::ArchTag;
  using TileShape = typename GemmKernel::TileShape;
  using ClusterShape = typename GemmKernel::ClusterShape;
  static constexpr int MaxThreadsPerBlock = GemmKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static_assert(MaxThreadsPerBlock == 256);
  static_assert(MaxThreadsPerBlock >= CommOp::kMinThreads);
  static constexpr int ClusterSize=cute::size(typename GemmKernel::ClusterShape{});
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  static_assert(SeparateRoles || ClusterSize == 1);
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
    bool borrow_idle_compute_ctas = false;
  };

  struct Params;
  struct Params {
#if FUSE_ENABLE_PROFILING
    GroupedProfile profile{};
#endif
    typename GemmKernel::Params gemm;
    typename CommOp::Params comm;
    int32_t compute_ctas = 0;
    int32_t num_comm_ctas = 0;
    // Stock or paired-MMA plans use two Graph branches with disjoint budgets.
    // A compute cluster always contains only GEMM roles. 0 is the original
    // native-1SM launch; 1 is GEMM only; 2 is communication only.
    int32_t split_role = 0;
    bool borrow_idle_compute_ctas = false;
    // Optional GPU-prepared invocation state. The host copy still fixes the
    // cooperative grid ceiling; explicit plans leave this pointer null.
    const Params* invocation_params = nullptr;
  };

  static bool can_implement(const Arguments& args) {
    if (args.num_comm_ctas <= 0 || args.gemm.hw_info.sm_count <= 0 ||
        args.comm.params.num_comm_ctas != args.num_comm_ctas ||
        !GemmKernel::can_implement(args.gemm) || !CommOp::can_implement(args.comm)) return false;
    if constexpr (SeparateRoles)
      return args.num_comm_ctas % ClusterSize == 0 &&
          args.gemm.hw_info.sm_count % ClusterSize == 0;
    else return args.gemm.scheduler.block_offset == args.num_comm_ctas;
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
    params.borrow_idle_compute_ctas = args.borrow_idle_compute_ctas;
    const dim3 compute_grid = GemmKernel::get_grid_shape(params.gemm);
    params.compute_ctas = static_cast<int32_t>(
        static_cast<uint64_t>(compute_grid.x) * compute_grid.y * compute_grid.z);
    return params;
  }

  static dim3 get_grid_shape(const Params& params) {
    if constexpr (SeparateRoles) return dim3(params.compute_ctas,1,1);
    else return dim3(params.compute_ctas + params.num_comm_ctas, 1, 1);
  }

  static dim3 get_block_shape() { return GemmKernel::get_block_shape(); }

  CUTLASS_DEVICE static const Params& resolve_invocation(const Params& launch) {
    // Preserve direct kernel-argument access for explicit plans. Only the
    // separately instantiated dynamic kernel loads GPU-prepared parameters.
    if constexpr (DynamicPolicy) return *launch.invocation_params;
    else return launch;
  }

  CUTLASS_DEVICE void operator()(const Params& launch, char* smem) {
    // The preceding prepare node has published shapes, counters and the
    // optional dynamic policy on this stream. Do not copy CUTLASS's
    // descriptor-rich Params into thread-local storage.
    const Params& params = resolve_invocation(launch);
#if FUSE_ENABLE_PROFILING
    const int profile_cta = params.profile.cta_offset + int(blockIdx.x +
        gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z));
    if (params.profile.roles && threadIdx.x == 0)
      params.profile.roles[profile_cta].begin = read_global_timer();
#endif
    const int32_t block = static_cast<int32_t>(blockIdx.x);
    // A short grouped workload may expose fewer initial GEMM tiles than its
    // fixed compute ceiling. Those blocks cannot execute GEMM in this launch.
    // Dispatch temporarily reuses them as extra panel producers:
    //
    //   [ configured comm ][ GEMM tiles ][ idle compute blocks ]
    //             |                              |
    //             +--------- producer pool ------+
    //
    // The configured GEMM ceiling and tile order do not change. Once the
    // workload has at least compute_ctas tiles, idle_compute is zero and the
    // ordinary large-token role partition is unchanged. The decision uses
    // device row counts, never a model or benchmark identity.
    if constexpr (SeparateRoles) {
      if (params.split_role == 1) {
        GemmKernel{}(params.gemm,smem);
      } else if (params.split_role == 2) {
        CommOp{}(params.comm,smem,block,params.num_comm_ctas);
      }
    } else {
      const int64_t tiles = params.gemm.scheduler.order.tiles();
      const int32_t active_compute = tiles < params.compute_ctas
          ? int32_t(tiles) : params.compute_ctas;
      const int32_t effective_comm = CommOp::kReuseIdleComputeCtas &&
          params.borrow_idle_compute_ctas
          ? *params.comm.params.effective_comm_ctas : params.num_comm_ctas;
      if (block < params.num_comm_ctas) {
        CommOp{}(params.comm, smem, block, effective_comm);
      } else if (GemmKernel::TileScheduler::valid_initial_worker(
                     params.gemm.scheduler, blockIdx.x)) {
        GemmKernel{}(params.gemm, smem);
      } else if constexpr (CommOp::kReuseIdleComputeCtas) {
        const int32_t compute_index = block - params.num_comm_ctas;
        if (params.borrow_idle_compute_ctas && compute_index >= active_compute)
          CommOp{}(params.comm, smem,
              params.num_comm_ctas + compute_index - active_compute,effective_comm);
      }
    }
#if FUSE_ENABLE_PROFILING
    if (params.profile.roles) {
      __syncthreads();  // Diagnostic role endpoint means ALL CTA threads returned.
      if (threadIdx.x == 0) params.profile.roles[profile_cta].role_end = read_global_timer();
    }
#endif
    if constexpr (CommOp::kNeedsGridFinalize) {
      if constexpr (SeparateRoles) {
        if (params.split_role != 1) {
          cooperative_groups::this_grid().sync();
          CommOp{}.finalize(params.comm);
        }
      } else {
        cooperative_groups::this_grid().sync();
        CommOp{}.finalize(params.comm);
      }
    }
#if FUSE_ENABLE_PROFILING
    if (params.profile.roles) {
      __syncthreads();
      if (threadIdx.x == 0) params.profile.roles[profile_cta].end = read_global_timer();
    }
#endif
  }
};

}  // namespace fuse::detail

namespace cutlass::gemm::kernel::detail {
template <class Tile, class Cluster, uint32_t Stages>
struct TileSchedulerSelector<GroupScheduler, arch::Sm100, Tile, Cluster, Stages,
    fuse::detail::GroupedProblemShape> {
  using Scheduler = fuse::detail::PersistentTileSchedulerSm100Grouped<
      Stages,false,cute::size(Cluster{})>;
};
template <class Tile, class Cluster, uint32_t Stages>
struct TileSchedulerSelector<GroupScheduler, arch::Sm100, Tile, Cluster, Stages,
    fuse::detail::SwappedGroupedProblemShape> {
  using Scheduler = fuse::detail::PersistentTileSchedulerSm100Grouped<
      Stages,true,cute::size(Cluster{})>;
};
}  // namespace cutlass::gemm::kernel::detail
