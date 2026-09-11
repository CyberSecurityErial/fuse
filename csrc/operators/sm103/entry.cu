// SPDX-License-Identifier: BSD-3-Clause
#include "fuse/operators/primitives/a2a_gemm.h"
#include "fuse/operators/primitives/gemm_a2a.h"
#include "fuse/operators/primitives/gemm_a2a_mxfp8.h"
#include "fuse/profiling/sm103/host.cuh"
#include "fuse/profiling/sm103/oproj.cuh"

#if FUSE_ENABLE_PROFILING
namespace fuse::detail {
thread_local HostLaunchRecord* host_launch_sink = nullptr;
thread_local const OprojPipelineView* oproj_pipeline_sink = nullptr;
}  // namespace fuse::detail
#endif

// Like SM90, each architecture is assembled in one CUDA translation unit.
// Common public BF16 entry points resolve to the CMake-selected backend.
#include "detail/cutlass_pipeline.cuh"
#include "detail/persistent_gemm.cuh"
#include "detail/gemm.cuh"
#include "detail/quantization.cuh"
#include "detail/a2a_gemm.cuh"
#include "detail/gemm_a2a.cuh"
#include "detail/backward.cuh"
#include "detail/launch.cuh"
#include "api/policy.cuh"
#include "api/forward.cuh"
#include "api/forward_mxfp8.cuh"
#include "api/reference.cuh"
#include "api/backward.cuh"
