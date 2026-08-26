// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/kernels.h"

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>

#if FUSE_ENABLE_PROFILING
namespace fuse::detail {

CUTLASS_DEVICE uint64_t read_global_timer() {
  uint64_t value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

// Diagnostic-only wrapper. The production monolithic kernel remains
// unchanged; this type is instantiated only by the telemetry entry point.
template <class BaseKernel>
struct RoleTelemetryKernel {
  using ArchTag = typename BaseKernel::ArchTag;
  using ClusterShape = typename BaseKernel::ClusterShape;
  using SharedStorage = typename BaseKernel::SharedStorage;
  static constexpr int SharedStorageSize = BaseKernel::SharedStorageSize;
  static constexpr int MaxThreadsPerBlock = BaseKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor =
      BaseKernel::MinBlocksPerMultiprocessor;

  struct Arguments : BaseKernel::Arguments {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
  };

  struct Params : BaseKernel::Params {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
  };

  static bool can_implement(const Arguments& args) {
    return args.timeline != nullptr && args.timeline_capacity > 0 &&
        BaseKernel::can_implement(
            static_cast<const typename BaseKernel::Arguments&>(args));
  }

  static size_t get_workspace_size(const Arguments& args) {
    return BaseKernel::get_workspace_size(
        static_cast<const typename BaseKernel::Arguments&>(args));
  }

  static cutlass::Status initialize_workspace(
      const Arguments& args,
      void* workspace,
      cudaStream_t stream) {
    return BaseKernel::initialize_workspace(
        static_cast<const typename BaseKernel::Arguments&>(args),
        workspace,
        stream);
  }

  static Params to_underlying_arguments(const Arguments& args, void* workspace) {
    Params params{};
    static_cast<typename BaseKernel::Params&>(params) =
        BaseKernel::to_underlying_arguments(
            static_cast<const typename BaseKernel::Arguments&>(args),
            workspace);
    params.timeline = args.timeline;
    params.timeline_capacity = args.timeline_capacity;
    return params;
  }

  static dim3 get_grid_shape(const Params& params) {
    return BaseKernel::get_grid_shape(
        static_cast<const typename BaseKernel::Params&>(params));
  }

  static dim3 get_block_shape() { return BaseKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    const int32_t cta = static_cast<int32_t>(blockIdx.x);
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].start = read_global_timer();
    }

    BaseKernel{}(
        static_cast<const typename BaseKernel::Params&>(params), smem);

    // Comm CTAs use only a subset of the physical warps. Rejoin every warp so
    // the end stamp means the whole CTA, rather than just slot zero, is done.
    __syncthreads();
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].end = read_global_timer();
    }
  }
};

}  // namespace fuse::detail
#endif
