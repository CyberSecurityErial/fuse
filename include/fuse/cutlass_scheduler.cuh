// SPDX-License-Identifier: BSD-3-Clause
#pragma once

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
