// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/device_kernel.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>

namespace fuse::detail {

struct MonolithicPersistentScheduler {};

// CUTLASS's stock persistent scheduler advances by gridDim. A monolithic grid
// also contains communication CTAs, so GEMM workers must advance by the compute
// sub-grid size instead. All tile mapping logic stays in the stock scheduler.
class PersistentTileSchedulerSm90Monolithic
    : public cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90 {
  using Base = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;
  using BaseParams = typename Base::Params;

 public:
  using BaseArguments = typename Base::Arguments;
  struct Arguments : BaseArguments {
    int32_t block_offset = 0;
  };
  using WorkTileInfo = typename Base::WorkTileInfo;
  using RasterOrder = typename Base::RasterOrder;
  using RasterOrderOptions = typename Base::RasterOrderOptions;
  using Pipeline = typename Base::Pipeline;
  using SharedStorage = typename Base::SharedStorage;
  using ThrottlePipeline = typename Base::ThrottlePipeline;
  using CLCResponse = typename Base::CLCResponse;
  static constexpr bool IsDynamicPersistent = false;

  struct Params : BaseParams {
    uint64_t compute_grid_size = 0;
    int32_t block_offset = 0;
  };

  template <class ProblemShape, class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      ProblemShape problem,
      TileShape tile,
      ClusterShape cluster,
      const cutlass::KernelHardwareInfo& hardware,
      const Arguments& arguments,
      void* workspace = nullptr,
      uint32_t epilogue_subtiles = 1,
      uint32_t k_tile_alignment = 1) {
    Params params{};
    const BaseArguments& base_arguments = arguments;
    BaseParams base_params = Base::to_underlying_arguments(
        problem,
        tile,
        cluster,
        hardware,
        base_arguments,
        workspace,
        epilogue_subtiles,
        k_tile_alignment);
    static_cast<BaseParams&>(params) = base_params;
    const dim3 grid = Base::get_grid_shape(
        base_params,
        problem,
        tile,
        cluster,
        hardware,
        base_arguments);
    params.compute_grid_size =
        static_cast<uint64_t>(grid.x) * grid.y * grid.z;
    params.block_offset = arguments.block_offset;
    return params;
  }

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90Monolithic(const Params& params)
      : Base(static_cast<const BaseParams&>(params)),
        current_(static_cast<uint64_t>(blockIdx.x - params.block_offset) +
                 static_cast<uint64_t>(blockIdx.y) * params.compute_grid_size),
        stride_(params.compute_grid_size) {}

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    return Base::get_current_work_for_linear_idx(current_);
  }

  CUTLASS_DEVICE void advance_to_next_work(uint32_t count = 1) {
    current_ += stride_ * count;
  }

  CUTLASS_DEVICE bool is_last_tile(WorkTileInfo& work, uint32_t count = 1) const {
    if (Base::continue_current_work(work)) {
      return false;
    }
    return !Base::get_current_work_for_linear_idx(current_ + stride_ * count).is_valid();
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo work) {
    if (Base::continue_current_work(work)) {
      return cute::make_tuple(work, true);
    }
    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  template <class SchedulerPipeline, class PipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo work,
      SchedulerPipeline&,
      PipelineState) {
    return fetch_next_work(work);
  }

 private:
  uint64_t current_;
  uint64_t stride_;
};

}  // namespace fuse::detail

namespace cutlass::gemm::kernel::detail {

template <
    class ArchTag,
    class TileShape,
    class ClusterShape,
    uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    fuse::detail::MonolithicPersistentScheduler,
    ArchTag,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount> {
  using Scheduler = fuse::detail::PersistentTileSchedulerSm90Monolithic;
};

}  // namespace cutlass::gemm::kernel::detail


// Cooperative role dispatcher and launch helper.
namespace fuse::detail {

template <class CommOp, class = void>
struct NeedsGridFinalize : std::false_type {};

template <class CommOp>
struct NeedsGridFinalize<
    CommOp,
    std::void_t<decltype(CommOp::kNeedsGridFinalize)>>
    : std::bool_constant<CommOp::kNeedsGridFinalize> {};

// Thin dispatcher around an unmodified CUTLASS GemmUniversal kernel. Compute
// CTAs retain CUTLASS block ids [0, compute_ctas); communication CTAs occupy the
// prefix of the same cooperative grid.
template <class GemmKernel, class CommOp>
struct MonolithicGemm {
  using ArchTag = typename GemmKernel::ArchTag;
  using ClusterShape = typename GemmKernel::ClusterShape;
  static constexpr int SharedStorageSize =
      GemmKernel::SharedStorageSize > static_cast<int>(CommOp::SharedStorageBytes)
      ? GemmKernel::SharedStorageSize
      : static_cast<int>(CommOp::SharedStorageBytes);
  struct alignas(128) SharedStorage {
    char bytes[SharedStorageSize];
  };
  static constexpr int MaxThreadsPerBlock = GemmKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = GemmKernel::MinBlocksPerMultiprocessor;

  struct Arguments {
    typename GemmKernel::Arguments gemm;
    typename CommOp::Arguments comm;
    int32_t num_comm_ctas = 0;
  };

  struct Params {
    typename GemmKernel::Params gemm;
    typename CommOp::Params comm;
    int32_t compute_ctas = 0;
    int32_t num_comm_ctas = 0;
  };

  static bool can_implement(const Arguments& args) {
    constexpr int cluster_x = cute::size<0>(ClusterShape{});
    constexpr int cluster_y = cute::size<1>(ClusterShape{});
    constexpr int cluster_z = cute::size<2>(ClusterShape{});
    return cluster_x > 0 && cluster_y == 1 && cluster_z == 1 &&
        args.num_comm_ctas > 0 && args.num_comm_ctas % cluster_x == 0 &&
        GemmKernel::can_implement(args.gemm) &&
        CommOp::can_implement(args.comm);
  }

  static size_t get_workspace_size(const Arguments& args) {
    return GemmKernel::get_workspace_size(args.gemm);
  }

  static cutlass::Status initialize_workspace(
      const Arguments& args,
      void* workspace,
      cudaStream_t stream) {
    return GemmKernel::initialize_workspace(args.gemm, workspace, stream);
  }

  static Params to_underlying_arguments(const Arguments& args, void* workspace) {
    Params params{};
    params.gemm = GemmKernel::to_underlying_arguments(args.gemm, workspace);
    params.comm = CommOp::to_underlying_arguments(args.comm);
    const dim3 compute_grid = GemmKernel::get_grid_shape(params.gemm);
    params.compute_ctas = static_cast<int32_t>(
        static_cast<uint64_t>(compute_grid.x) * compute_grid.y * compute_grid.z);
    params.num_comm_ctas = args.num_comm_ctas;
    return params;
  }

  static dim3 get_grid_shape(const Params& params) {
    return dim3(params.compute_ctas + params.num_comm_ctas, 1, 1);
  }

  static dim3 get_block_shape() { return GemmKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    const bool is_comm =
        static_cast<int32_t>(blockIdx.x) < params.num_comm_ctas;
    if (is_comm) {
      CommOp{}(
          params.comm,
          static_cast<int32_t>(blockIdx.x),
          params.num_comm_ctas);
    } else {
      GemmKernel{}(params.gemm, smem);
    }
    if constexpr (NeedsGridFinalize<CommOp>::value) {
      cooperative_groups::this_grid().sync();
      CommOp{}.finalize(params.comm);
    }
  }
};

template <class Kernel>
cudaError_t launch_cooperative(
    const typename Kernel::Params& params,
    cudaStream_t stream,
    int32_t sm_count) {
  constexpr size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
  constexpr int cluster_x = cute::size<0>(typename Kernel::ClusterShape{});
  constexpr int cluster_y = cute::size<1>(typename Kernel::ClusterShape{});
  constexpr int cluster_z = cute::size<2>(typename Kernel::ClusterShape{});
  constexpr int cluster_size = cluster_x * cluster_y * cluster_z;
  auto entry = cutlass::device_kernel<Kernel>;

  cudaError_t status = cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  if (status != cudaSuccess) {
    return status;
  }

  const dim3 grid = Kernel::get_grid_shape(params);
  if constexpr (cluster_y != 1 || cluster_z != 1) {
    return cudaErrorNotSupported;
  }
  if (params.num_comm_ctas % cluster_x != 0 ||
      params.compute_ctas % cluster_x != 0 || grid.x % cluster_x != 0 ||
      grid.y != 1 || grid.z != 1) {
    return cudaErrorInvalidConfiguration;
  }

  cudaLaunchAttribute attributes[2]{};
  int attribute_count = 0;
  if constexpr (cluster_size > 1) {
    cudaLaunchAttribute cluster_attribute{};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim = {cluster_x, cluster_y, cluster_z};

    cudaLaunchConfig_t occupancy_config{};
    occupancy_config.gridDim = dim3(cluster_x, cluster_y, cluster_z);
    occupancy_config.blockDim = Kernel::get_block_shape();
    occupancy_config.dynamicSmemBytes = smem_bytes;
    occupancy_config.stream = stream;
    occupancy_config.attrs = &cluster_attribute;
    occupancy_config.numAttrs = 1;

    int active_clusters = 0;
    status = cudaOccupancyMaxActiveClusters(
        &active_clusters, entry, &occupancy_config);
    if (status != cudaSuccess) {
      return status;
    }
    if (static_cast<int32_t>(grid.x) / cluster_x > active_clusters) {
      return cudaErrorCooperativeLaunchTooLarge;
    }

    attributes[attribute_count++] = cluster_attribute;
  } else {
    int active_per_sm = 0;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_per_sm,
        entry,
        static_cast<int>(Kernel::get_block_shape().x),
        smem_bytes);
    if (status != cudaSuccess) {
      return status;
    }
    if (static_cast<int32_t>(grid.x) > active_per_sm * sm_count) {
      return cudaErrorCooperativeLaunchTooLarge;
    }
  }

  attributes[attribute_count].id = cudaLaunchAttributeCooperative;
  attributes[attribute_count].val.cooperative = 1;
  ++attribute_count;

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = Kernel::get_block_shape();
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = attribute_count;
  return cudaLaunchKernelEx(&config, entry, params);
}

}  // namespace fuse::detail
