// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <type_traits>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

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
