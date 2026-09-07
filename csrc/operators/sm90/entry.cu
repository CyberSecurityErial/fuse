// SPDX-License-Identifier: BSD-3-Clause
#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED 1
#include "fuse/operators/semantics/ulysses/projection.h"
#include "fuse/operators/ulysses/heterogeneous_cp.h"

#include "fuse/arch/sm90.cuh"
#include "fuse/profiling/timeline.cuh"
#include "detail/cutlass_pipeline.cuh"
#include "detail/persistent_gemm.cuh"

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

// Private device types, communication operators and launch machinery. These
// remain one CUDA translation unit so CUTLASS instantiation and SASS stay
// identical while each dataflow can be maintained independently.
#include "detail/gemm.cuh"
#include "detail/a2a_gemm.cuh"
#include "detail/backward.cuh"
#include "detail/gemm_a2a.cuh"
#include "detail/launch.cuh"

// Public API implementations, grouped by responsibility rather than appended
// behind the device code that they launch.
#include "api/backward.cuh"
#include "api/policy.cuh"
#include "api/heterogeneous.cuh"
#include "api/forward.cuh"
#include "api/reference.cuh"
