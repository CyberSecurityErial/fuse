// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#if defined(__CUDACC__)
#include "fuse/arch/sm90.cuh"
#endif

#include <cuda_runtime.h>

#include <cutlass/cutlass.h>

#if FUSE_ENABLE_PROFILING
namespace fuse {

// One diagnostic sample per physical CTA. SM90 %globaltimer is shared across
// SMs on a device, so communication and compute entries share one timeline.
struct A2AGemmCtaTimeline {
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t active_start = 0;
  // GEMM->A2A diagnostic kernels use this for the point immediately before
  // the cooperative grid barrier: GEMM has finished on compute CTAs, while
  // all locally assigned output-route tasks have finished on comm CTAs.
  uint64_t role_done = 0;
  // GEMM->A2A finalize breakdown. grid_sync_done is recorded by every CTA;
  // the remaining fields are populated only by CTA0/thread0.
  uint64_t grid_sync_done = 0;
  uint64_t fence_done = 0;
  uint64_t publish_done = 0;
  uint64_t source_ready[kMaxWorldSize]{};
};

// Per-peer publication and observation timestamps for one logical GEMM tile.
struct A2AGemmPeerTimeline {
  uint64_t release = 0;
  uint64_t acquire[kMaxWorldSize]{};
  uint64_t task_begin = 0;
  uint64_t input_ready = 0;
  uint64_t g2s_issue = 0;
  uint64_t g2s_done = 0;
  uint64_t s2g_issue = 0;
  uint64_t s2g_done = 0;
  uint64_t publish_issue = 0;
  int32_t m_tile = 0;
  int32_t n_tile = 0;
  int32_t batch = 0;
  uint32_t valid = 0;
  int32_t comm_cta = 0;
  int32_t comm_slot = 0;
  int32_t task_id = 0;
  int32_t row_chunk = 0;
  int32_t copy_rows = 0;
  int32_t source_rank = 0;
  // 0: vector GMEM copy, 1: bulk G2S + row S2G,
  // 2: bulk G2S + tensor-store S2G.
  int32_t copy_path = 0;
  uint32_t comm_valid = 0;
};

struct A2AGemmRoleResources {
  int32_t threads_per_cta = 0;
  int32_t registers_per_thread = 0;
  int32_t telemetry_registers_per_thread = 0;
  int32_t static_smem_bytes = 0;
  int32_t dynamic_smem_bytes = 0;
  int32_t cluster_ctas = 0;
  int32_t comm_active_warps = 0;
  int32_t compute_active_warps = 0;
  int32_t comm_working_smem_bytes = 0;
};

}  // namespace fuse

#if defined(__CUDACC__)
namespace fuse::detail {

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
#endif  // defined(__CUDACC__)
#endif
