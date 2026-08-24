// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cute/layout.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cstdint>

namespace fuse::detail {

// A host-built work list keeps variable-length grouped GEMM scheduling cheap
// on device and, unlike CUTLASS's stock grouped scheduler, allows a prefix of
// the physical grid to be reserved for communication CTAs.
struct PackedGroupWorkTile {
  int32_t M_idx = 0;
  int32_t N_idx = 0;
  int32_t L_idx = 0;
  int32_t is_valid_tile = 0;

  CUTLASS_HOST_DEVICE bool is_valid() const { return is_valid_tile != 0; }
  CUTLASS_HOST_DEVICE static PackedGroupWorkTile invalid_work_tile() {
    return {-1, -1, -1, 0};
  }
  CUTLASS_HOST_DEVICE bool is_final_split(uint32_t) const { return true; }
  CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const { return -1; }
};

template <class UnderlyingProblemShape_>
struct PackedGroupProblemShape {
  using UnderlyingProblemShape = UnderlyingProblemShape_;
  int32_t num_groups = 0;
  UnderlyingProblemShape* problem_shapes = nullptr;
  const UnderlyingProblemShape* host_problem_shapes = nullptr;

  CUTLASS_HOST_DEVICE int32_t groups() const { return num_groups; }
  CUTLASS_HOST_DEVICE UnderlyingProblemShape get_problem_shape(int32_t group) const {
    return problem_shapes[group];
  }
  CUTLASS_HOST_DEVICE UnderlyingProblemShape get_host_problem_shape(int32_t group) const {
    return host_problem_shapes ? host_problem_shapes[group] : UnderlyingProblemShape{};
  }
  CUTLASS_HOST_DEVICE bool is_host_problem_shape_available() const {
    return host_problem_shapes != nullptr;
  }
};

template <class GroupProblemShape, int SchedulerPipelineStageCount>
class PackedGroupSchedulerSm90 {
 public:
  using WorkTileInfo = PackedGroupWorkTile;
  using ProblemShape = typename GroupProblemShape::UnderlyingProblemShape;
  using RasterParams =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params;
  using RasterOrder = typename RasterParams::RasterOrder;
  using RasterOrderOptions = typename RasterParams::RasterOrderOptions;
  static constexpr bool IsDynamicPersistent = false;

  using Pipeline = cutlass::PipelineAsync<SchedulerPipelineStageCount>;
  using PipelineStorage =
      cutlass::PipelineDetail::PipelineAsyncSharedStorage<SchedulerPipelineStageCount>;
  using PipelineState =
      cutlass::PipelineDetail::PipelineAsyncPipelineState<SchedulerPipelineStageCount>;
  using ThrottlePipeline = cutlass::PipelineEmpty;
  using ThrottlePipelineStorage = typename ThrottlePipeline::SharedStorage;
  using SchedulerResponse = WorkTileInfo;

  struct SharedStorage {
    CUTLASS_DEVICE PipelineStorage pipeline() { return pipeline_; }
    CUTLASS_DEVICE ThrottlePipelineStorage throttle_pipeline() { return {}; }
    CUTLASS_DEVICE SchedulerResponse* data() { return data_; }

   private:
    alignas(16) PipelineStorage pipeline_;
    alignas(16) SchedulerResponse data_[SchedulerPipelineStageCount];
  };

  struct Arguments {
    const WorkTileInfo* work_list = nullptr;
    int64_t work_count = 0;
    int32_t block_offset = 0;
    int32_t worker_count = 0;
    int max_swizzle_size = 1;
    RasterOrderOptions raster_order = RasterOrderOptions::AlongM;
  };

  struct Params {
    const WorkTileInfo* work_list = nullptr;
    int64_t work_count = 0;
    int32_t block_offset = 0;
    int32_t worker_count = 0;
    int32_t log_swizzle_size_ = 0;
    RasterOrder raster_order_ = RasterOrder::AlongM;
  };

  template <class TileShape, class ClusterShape>
  static Params to_underlying_arguments(
      GroupProblemShape,
      TileShape,
      ClusterShape,
      const cutlass::KernelHardwareInfo& hw_info,
      const Arguments& args,
      void* = nullptr,
      uint32_t = 1,
      uint32_t = 1) {
    Params result{};
    result.work_list = args.work_list;
    result.work_count = args.work_count;
    result.block_offset = args.block_offset;
    result.worker_count = args.worker_count > 0 ? args.worker_count : hw_info.sm_count;
    result.raster_order_ = args.raster_order == RasterOrderOptions::AlongN
        ? RasterOrder::AlongN
        : RasterOrder::AlongM;
    return result;
  }

  template <class TileShape, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3 get_grid_shape(
      const Params& params,
      const GroupProblemShape&,
      TileShape,
      ClusterShape,
      cutlass::KernelHardwareInfo,
      Arguments,
      bool = true) {
    return dim3(static_cast<uint32_t>(params.worker_count), 1, 1);
  }

  static bool can_implement(const Arguments& args) {
    return args.work_list && args.work_count > 0 && args.block_offset >= 0 &&
        args.worker_count > 0;
  }

  template <class Shape, class ElementAccumulator>
  static size_t get_workspace_size(
      const Arguments&,
      Shape,
      const cutlass::KernelHardwareInfo&,
      uint32_t,
      uint32_t = 1,
      uint32_t = 1) {
    return 0;
  }

  template <class Shape, class ElementAccumulator>
  static cutlass::Status initialize_workspace(
      const Arguments&,
      void*,
      cudaStream_t,
      Shape,
      const cutlass::KernelHardwareInfo&,
      uint32_t,
      uint32_t = 1,
      uint32_t = 1,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  PackedGroupSchedulerSm90() = default;
  CUTLASS_DEVICE PackedGroupSchedulerSm90(
      const Params& params,
      SchedulerResponse* response)
      : params_(params),
        response_(response),
        current_(static_cast<int64_t>(blockIdx.x) - params.block_offset) {}

  template <class ClusterShape>
  CUTLASS_DEVICE WorkTileInfo initial_work_tile_info(ClusterShape) const {
    return get_current_work();
  }

  CUTLASS_DEVICE WorkTileInfo get_current_work() const {
    if (current_ < 0 || current_ >= params_.work_count) {
      return WorkTileInfo::invalid_work_tile();
    }
    return params_.work_list[current_];
  }

  template <class SchedulerPipeline, class SchedulerPipelineState>
  CUTLASS_DEVICE auto advance_to_next_work(
      SchedulerPipeline& pipeline,
      SchedulerPipelineState state,
      uint32_t count = 1) {
    current_ += static_cast<int64_t>(params_.worker_count) * count;
    const WorkTileInfo work = get_current_work();
    pipeline.producer_acquire(state);
    if (cute::elect_one_sync()) {
      response_[state.index()] = work;
      cutlass::arch::fence_view_async_shared();
      pipeline.producer_commit(state);
    }
    return cute::make_tuple(work, true);
  }

  template <class SchedulerPipeline, class SchedulerPipelineState>
  CUTLASS_DEVICE auto fetch_next_work(
      WorkTileInfo,
      SchedulerPipeline& pipeline,
      SchedulerPipelineState state) {
    pipeline.consumer_wait(state);
    const WorkTileInfo work = response_[state.index()];
    cutlass::arch::fence_view_async_shared();
    pipeline.consumer_release(state);
    return cute::make_tuple(work, true);
  }

  CUTLASS_HOST_DEVICE static bool compute_epilogue(
      const WorkTileInfo&, const Params&) {
    return true;
  }
  template <class Fragment>
  CUTLASS_DEVICE static void fixup(
      const Params&, const WorkTileInfo&, Fragment&, uint32_t, uint32_t) {}
  CUTLASS_DEVICE static bool continue_current_work(WorkTileInfo&) { return false; }
  template <class ProblemShapeMNKL, class TileShape>
  CUTLASS_HOST_DEVICE static int get_work_k_tile_count(
      const WorkTileInfo&, ProblemShapeMNKL problem, TileShape tile) {
    return cute::size(cute::ceil_div(cute::get<2>(problem), cute::get<2>(tile)));
  }
  CUTLASS_HOST_DEVICE static uint32_t get_work_k_tile_start(const WorkTileInfo&) {
    return 0;
  }
  CUTLASS_DEVICE static bool valid_warpgroup_in_work_tile(const WorkTileInfo&) {
    return true;
  }
  CUTLASS_DEVICE static bool need_separate_reduction(const Params&) { return false; }
  CUTLASS_DEVICE static bool requires_separate_reduction(const Params&) { return false; }
  CUTLASS_DEVICE bool is_work_tile_for_reduction(const WorkTileInfo&, const Params&) {
    return false;
  }
  CUTLASS_DEVICE uint32_t epilgoue_subtile_idx(const WorkTileInfo&, const Params&) const {
    return 0;
  }
  template <class Fragment>
  CUTLASS_DEVICE void separate_reduction(
      const Params&, const WorkTileInfo&, Fragment&, uint32_t, uint32_t) {}
  template <class Fragment>
  CUTLASS_DEVICE static void share(
      const Params&, const WorkTileInfo&, Fragment&, uint32_t, uint32_t) {}

 private:
  Params params_{};
  SchedulerResponse* response_ = nullptr;
  int64_t current_ = 0;
};

}  // namespace fuse::detail

namespace cutlass::gemm::kernel::detail {

template <
    class TileShape,
    class ClusterShape,
    uint32_t SchedulerPipelineStageCount,
    class UnderlyingProblemShape>
struct TileSchedulerSelector<
    GroupScheduler,
    cutlass::arch::Sm90,
    TileShape,
    ClusterShape,
    SchedulerPipelineStageCount,
    fuse::detail::PackedGroupProblemShape<UnderlyingProblemShape>> {
  using Scheduler = fuse::detail::PackedGroupSchedulerSm90<
      fuse::detail::PackedGroupProblemShape<UnderlyingProblemShape>,
      SchedulerPipelineStageCount>;
};

}  // namespace cutlass::gemm::kernel::detail
