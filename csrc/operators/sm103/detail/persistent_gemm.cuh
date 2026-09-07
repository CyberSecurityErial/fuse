// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "host_profiling.cuh"
#include "producer_consumer.cuh"

// Blackwell scheduling, CTA roles, and cooperative launch resources.
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fuse::detail {

struct MonolithicPersistentScheduler {};

// The physical one-dimensional grid begins with communication CTAs. CLC
// cannot reserve that prefix; the compute workers use a static scheduler
// whose stride is the compute sub-grid, not the full physical gridDim.
class PersistentTileSchedulerSm100Monolithic
    : public cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100 {
  using Base =
      cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100;
  using BaseParams = Base::Params;

 public:
  using BaseArguments = Base::Arguments;
  using WorkTileInfo = Base::WorkTileInfo;
  static constexpr bool IsDynamicPersistent = false;

  struct Arguments : BaseArguments {
    int32_t block_offset = 0;
  };

  struct Params : BaseParams {
    uint64_t compute_grid_size = 0;
    int32_t block_offset = 0;
  };

  static bool can_implement(const Arguments& args) {
    const int swizzle = args.max_swizzle_size;
    return args.block_offset >= 0 &&
        (swizzle == 1 || swizzle == 2 || swizzle == 4 || swizzle == 8) &&
        Base::can_implement(static_cast<const BaseArguments&>(args));
  }

  template <class Problem, class Tile, class AtomThreadShape, class Cluster>
  static Params to_underlying_arguments(
      Problem problem,
      Tile tile,
      AtomThreadShape atom_thread_shape,
      Cluster cluster,
      const cutlass::KernelHardwareInfo& hardware,
      const Arguments& args,
      void* workspace = nullptr,
      uint32_t epilogue_subtiles = 1) {
    static_assert(cute::size(Cluster{}) == 1,
                  "The mixed SM103 grid requires one-CTA clusters.");
    static_assert(cute::size(AtomThreadShape{}) == 1,
                  "The mixed SM103 grid requires a one-SM MMA.");
    Params params{};
    BaseParams& base_params = params;
    base_params = Base::to_underlying_arguments(
        problem, tile, atom_thread_shape, cluster, hardware,
        args, workspace, epilogue_subtiles);
    params.block_offset = args.block_offset;
    if (hardware.sm_count > 0) {
      // hardware.sm_count is the caller's COMPUTE budget, excluding comm.
      // CUTLASS truncates this grid to the number of initial work tiles.
      const dim3 grid = Base::get_grid_shape(
          base_params, problem, tile,
          atom_thread_shape, cluster, hardware);
      params.compute_grid_size =
          static_cast<uint64_t>(grid.x) * grid.y * grid.z;
    }
    return params;
  }

  template <class Problem, class Tile, class AtomThreadShape, class Cluster>
  static dim3 get_grid_shape(
      const Params& params,
      Problem,
      Tile,
      AtomThreadShape,
      Cluster,
      const cutlass::KernelHardwareInfo&) {
    // The outer role-dispatch kernel adds block_offset to this grid.
    return dim3(static_cast<uint32_t>(params.compute_grid_size), 1, 1);
  }

  // Call this before entering CUTLASS's unconditional initial do/while. A
  // worker with no initial tile must never initialize its MMA/load pipelines.
  CUTLASS_HOST_DEVICE static bool valid_initial_worker(
      const Params& params, uint64_t physical_cta) {
    if (params.block_offset < 0 || params.compute_grid_size == 0 ||
        physical_cta < static_cast<uint64_t>(params.block_offset)) {
      return false;
    }
    const uint64_t worker = physical_cta - params.block_offset;
    return worker < params.compute_grid_size &&
        worker < params.blocks_per_problem_;
  }

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm100Monolithic(
      CLCResponse* clc_response,
      const Params& params,
      dim3 block_id_in_cluster)
      : Base(clc_response, static_cast<const BaseParams&>(params),
             block_id_in_cluster),
        current_(static_cast<uint64_t>(blockIdx.x) - params.block_offset),
        stride_(params.compute_grid_size) {
    CUTLASS_ASSERT(blockIdx.y == 0 && blockIdx.z == 0);
    CUTLASS_ASSERT(valid_initial_worker(params, blockIdx.x));
  }

  template <class Cluster>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(Cluster) const {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    const auto tile = ProducerTileOrder::decode(this->scheduler_params, current_);
    return {tile.m, tile.n, tile.batch, tile.valid};
  }

  CUTLASS_DEVICE void advance_to_next_work(uint32_t count = 1) {
    current_ += stride_ * count;
  }

  CUTLASS_DEVICE bool is_last_tile(
      WorkTileInfo&, uint32_t count = 1) const {
    return !Base::get_current_work_for_linear_idx(
                current_ + stride_ * count).is_valid();
  }

  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo) {
    advance_to_next_work();
    return cute::make_tuple(get_current_work(), true);
  }

  template <class SchedulerPipeline, class PipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo work, SchedulerPipeline&, PipelineState) {
    return fetch_next_work(work);
  }

 private:
  uint64_t current_;
  uint64_t stride_;
};

// One physical cooperative grid, two persistent CTA roles. Keep exactly
// CUTLASS's block size: adding idle warps would break its CTA barriers.
template <class GemmKernel, class CommOp>
struct MonolithicGemm {
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
    if (static_cast<int32_t>(blockIdx.x) < params.num_comm_ctas) {
      CommOp{}(params.comm, smem, static_cast<int32_t>(blockIdx.x),
               params.num_comm_ctas);
    } else if (PersistentTileSchedulerSm100Monolithic::valid_initial_worker(
                   params.gemm.scheduler, blockIdx.x)) {
      GemmKernel{}(params.gemm, smem);
    }
    if constexpr (CommOp::kNeedsGridFinalize) {
      cooperative_groups::this_grid().sync();
      CommOp{}.finalize(params.comm);
    }
  }
};

// Standalone calibration kernels keep production's physical CTA geometry
// and shared-memory floor, but do not launch dummy communication/compute CTAs.
// They deliberately do not change MonolithicGemm's mixed-role contract.
template <class GemmKernel, class FusedKernel>
struct GemmReferenceKernel {
  using ProductionKernel = FusedKernel;
  using Params = typename GemmKernel::Params;
  using ArchTag = typename GemmKernel::ArchTag;
  using ClusterShape = typename GemmKernel::ClusterShape;
  static constexpr int MaxThreadsPerBlock = GemmKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr size_t SharedStorageSize =
      sizeof(typename GemmKernel::SharedStorage) > FusedKernel::SharedStorageSize
          ? sizeof(typename GemmKernel::SharedStorage) : FusedKernel::SharedStorageSize;
  static_assert(MaxThreadsPerBlock == FusedKernel::MaxThreadsPerBlock);
  static_assert(MaxThreadsPerBlock == 256 && cute::size(ClusterShape{}) == 1);

  static dim3 get_grid_shape(const Params& params) {
    return GemmKernel::get_grid_shape(params);
  }

  static dim3 get_block_shape() { return GemmKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    if (PersistentTileSchedulerSm100Monolithic::valid_initial_worker(
            params.scheduler, blockIdx.x)) {
      GemmKernel{}(params, smem);
    }
  }
};

template <class CommOp, class FusedKernel>
struct CopyReferenceKernel {
  using ProductionKernel = FusedKernel;
  using Params = typename CommOp::Params;
  using ArchTag = typename FusedKernel::ArchTag;
  using ClusterShape = typename FusedKernel::ClusterShape;
  static constexpr int MaxThreadsPerBlock = FusedKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor = 1;
  static constexpr size_t SharedStorageSize = FusedKernel::SharedStorageSize;
  static_assert(MaxThreadsPerBlock == 256 && cute::size(ClusterShape{}) == 1);
  static_assert(MaxThreadsPerBlock >= CommOp::kMinThreads);
  static_assert(SharedStorageSize >= CommOp::SharedStorageBytes);

  static dim3 get_grid_shape(const Params& params) {
    return dim3(params.params.num_comm_ctas, 1, 1);
  }

  static dim3 get_block_shape() { return FusedKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    const int32_t comm_id = static_cast<int32_t>(blockIdx.x);
    const int32_t comm_ctas = params.params.num_comm_ctas;
    if constexpr (CommOp::kNeedsGridFinalize) {
      // QKV input is already materialized. Do not wait for GEMM ready flags,
      // but retain the full production cross-rank routing completion tail.
      CommOp{}.run(params, smem, comm_id, comm_ctas, false);
      cooperative_groups::this_grid().sync();
      CommOp{}.finalize(params);
    } else {
      // A2A reads actual peer input and publishes actual arrival counters.
      CommOp{}(params, smem, comm_id, comm_ctas);
    }
  }
};

struct DeviceInfo {
  int device = -1;
  int sm_count = 0;
  int max_block_smem = 0;
  int max_sm_smem = 0;
};

template <class Kernel>
cudaError_t launch_shared_memory(const DeviceInfo& info, size_t* bytes) {
  struct Resources { int device; size_t bytes; };
  static thread_local std::vector<Resources> resources;
  for (const auto& item : resources) {
    if (item.device == info.device) {
      *bytes = item.bytes;
      return cudaSuccess;
    }
  }
  auto entry = cutlass::device_kernel<Kernel>;
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, entry);
  if (status != cudaSuccess) {
    return status;
  }
  // __launch_bounds__(...,1) is NOT a one-CTA-per-SM limit. Reserve more
  // than half of SM shared memory (including static SMEM), then verify
  // actual occupancy. Thus each explicit comm CTA occupies a distinct SM.
  const size_t half_sm = static_cast<size_t>(info.max_sm_smem) / 2;
  const size_t static_smem = attributes.sharedSizeBytes;
  const size_t padding = half_sm >= static_smem ? half_sm - static_smem + 1 : 0;
  const size_t dynamic_smem =
      (std::max<size_t>(Kernel::SharedStorageSize, padding) + 127) / 128 * 128;
  if (dynamic_smem + static_smem > static_cast<size_t>(info.max_block_smem)) {
    return cudaErrorInvalidConfiguration;
  }
  status = cudaFuncSetAttribute(entry, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                static_cast<int>(dynamic_smem));
  if (status != cudaSuccess) {
    return status;
  }
  int active_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_per_sm, entry, Kernel::MaxThreadsPerBlock, dynamic_smem);
  if (status != cudaSuccess) {
    return status;
  }
  if (active_per_sm != 1) {
    return cudaErrorInvalidConfiguration;
  }
  *bytes = dynamic_smem;
  resources.push_back({info.device, dynamic_smem});
  return cudaSuccess;
}

template <class Kernel>
cudaError_t launch_cooperative(
    const typename Kernel::Params& params,
    const DeviceInfo& info,
    cudaStream_t stream) {
  size_t smem_bytes = 0;
  cudaError_t status = launch_shared_memory<Kernel>(info, &smem_bytes);
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 grid = Kernel::get_grid_shape(params);
  if (params.num_comm_ctas <= 0 || params.num_comm_ctas >= info.sm_count ||
      params.compute_ctas <= 0 || grid.y != 1 || grid.z != 1 ||
      params.compute_ctas > info.sm_count - params.num_comm_ctas ||
      params.gemm.scheduler.block_offset != params.num_comm_ctas) {
    return cudaErrorInvalidConfiguration;
  }
  if (grid.x > static_cast<uint32_t>(info.sm_count)) {
    return cudaErrorCooperativeLaunchTooLarge;
  }
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = Kernel::get_block_shape();
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  FUSE_SM103_HOST_MARK(kCudaEnqueue);
#if FUSE_ENABLE_PROFILING
  status = cudaLaunchKernelEx(&config, cutlass::device_kernel<Kernel>, params);
  FUSE_SM103_HOST_MARK(kEnd);
  return status;
#else
  return cudaLaunchKernelEx(&config, cutlass::device_kernel<Kernel>, params);
#endif
}

template <class Kernel>
cudaError_t launch_reference_cooperative(
    const typename Kernel::Params& params,
    const DeviceInfo& info,
    int32_t cta_budget,
    cudaStream_t stream) {
  const dim3 grid = Kernel::get_grid_shape(params);
  const dim3 block = Kernel::get_block_shape();
  if (cta_budget <= 0 || cta_budget > info.sm_count ||
      grid.x == 0 || grid.y != 1 || grid.z != 1 ||
      block.x != 256 || block.y != 1 || block.z != 1) {
    return cudaErrorInvalidConfiguration;
  }
  if (grid.x > static_cast<uint32_t>(cta_budget)) {
    return cudaErrorCooperativeLaunchTooLarge;
  }
  size_t production_smem = 0;
  cudaError_t status = launch_shared_memory<typename Kernel::ProductionKernel>(
      info, &production_smem);
  if (status != cudaSuccess) {
    return status;
  }
  size_t smem_bytes = 0;
  // Each real reference entry is checked independently for register/shared
  // memory limits and exactly one resident CTA/SM; launch bounds alone do
  // not establish that. Resource caches contain actual per-device results.
  status = launch_shared_memory<Kernel>(info, &smem_bytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (smem_bytes < production_smem) {
    return cudaErrorInvalidConfiguration;
  }
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, cutlass::device_kernel<Kernel>, params);
}

}  // namespace fuse::detail

namespace cutlass::gemm::kernel::detail {

// Deliberately specialize only Sm100. Do not route another architecture's
// scheduler through this Blackwell implementation via a generic ArchTag.
template <class Tile, class Cluster, uint32_t SchedulerPipelineStageCount>
struct TileSchedulerSelector<
    fuse::detail::MonolithicPersistentScheduler,
    cutlass::arch::Sm100,
    Tile,
    Cluster,
    SchedulerPipelineStageCount> {
  using Scheduler = fuse::detail::PersistentTileSchedulerSm100Monolithic;
};

}  // namespace cutlass::gemm::kernel::detail
