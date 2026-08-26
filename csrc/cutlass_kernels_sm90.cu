// SPDX-License-Identifier: BSD-3-Clause
#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED 1
#include "fuse/kernels.h"

#include "fuse/cutlass_collectives.cuh"
#include "fuse/cutlass_scheduler.cuh"
#include "fuse/deepgemm_qkv_sm90.cuh"
#include "fuse/monolithic_gemm.cuh"
#if FUSE_ENABLE_PROFILING
#include "fuse/role_telemetry.cuh"
#endif
#include "fuse/system_barrier.cuh"

#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

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
#include <type_traits>

namespace fuse {
namespace {

using namespace cute;

using Element = Bf16;
using Fp8Element = Fp8E4m3;
using Accumulator = float;
using TileShape = Shape<_128, _128, _64>;
using M64TileShape = Shape<_64, _128, _64>;
using N160TileShape = Shape<_128, _160, _64>;
using ProjectionTileShape = Shape<_128, _256, _64>;
using WideN320TileShape = Shape<_128, Int<320>, _64>;
using Fp8TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using M64ClusterShape = Shape<_1, _1, _1>;
using ProjectionClusterShape = Shape<_2, _1, _1>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kA2ALhsCommRows = 32;
constexpr int kA2ALhsBulkSlots = 4;
constexpr int kA2ALhsBulkStageBytes = 48 * 1024;
constexpr int kQkvBulkRows = 64;
constexpr int kQkvBulkSlots = 12;
constexpr int kQkvBulkColumns = 128;
constexpr int kQkvBulkStageBytes =
    kQkvBulkRows * kQkvBulkColumns * sizeof(Element);
constexpr int kAlignment = 16 / sizeof(Element);
constexpr int kFp8Alignment = 16 / sizeof(Fp8Element);

using RasterOptions =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::
        RasterOrderOptions;

RasterOptions raster_option(GemmRaster requested, RasterOptions fallback) {
  switch (requested) {
    case GemmRaster::kAlongM:
      return RasterOptions::AlongM;
    case GemmRaster::kAlongN:
      return RasterOptions::AlongN;
    default:
      return fallback;
  }
}

__host__ __device__ constexpr int64_t stride_or(int64_t stride, int64_t packed) {
  return stride < 0 ? packed : stride;
}

__host__ __device__ constexpr int64_t a_row_stride(const GemmProblem& p) {
  return stride_or(p.stride_a.row, p.k);
}

__host__ __device__ constexpr int64_t a_batch_stride(const GemmProblem& p) {
  return stride_or(p.stride_a.batch, a_row_stride(p) * p.m);
}

__host__ __device__ constexpr int64_t b_row_stride(const GemmProblem& p) {
  return stride_or(p.stride_b.row, p.k);
}

__host__ __device__ constexpr int64_t b_batch_stride(const GemmProblem& p) {
  return stride_or(p.stride_b.batch, b_row_stride(p) * p.n);
}

__host__ __device__ constexpr int64_t d_row_stride(const GemmProblem& p) {
  return stride_or(p.stride_d.row, p.n);
}

__host__ __device__ constexpr int64_t d_batch_stride(const GemmProblem& p) {
  return stride_or(p.stride_d.batch, d_row_stride(p) * p.m);
}

bool supported_problem(const GemmProblem& p) {
  const bool unit_strides =
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1);
  const bool leading_dimensions =
      a_row_stride(p) >= p.k && b_row_stride(p) >= p.k &&
      d_row_stride(p) >= p.n && a_row_stride(p) % kAlignment == 0 &&
      a_batch_stride(p) % kAlignment == 0 &&
      b_row_stride(p) % kAlignment == 0 &&
      b_batch_stride(p) % kAlignment == 0 &&
      d_row_stride(p) % kAlignment == 0 && d_batch_stride(p) % kAlignment == 0;
  const bool batches = p.l == 1 ||
      (a_batch_stride(p) >= a_row_stride(p) * p.m &&
       (b_batch_stride(p) == 0 || b_batch_stride(p) >= b_row_stride(p) * p.n) &&
       d_batch_stride(p) >= d_row_stride(p) * p.m);
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l > 0 && unit_strides &&
      leading_dimensions && batches && !p.transpose_a && !p.transpose_b &&
      p.input_dtype == DType::kBfloat16 && p.weight_dtype == DType::kBfloat16 &&
      p.output_dtype == DType::kBfloat16 &&
      (p.max_swizzle_size == 1 || p.max_swizzle_size == 2 ||
       p.max_swizzle_size == 4 || p.max_swizzle_size == 8);
}

bool supported_fp8_problem(const GemmProblem& p) {
  const bool unit_strides =
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1);
  const bool leading_dimensions =
      a_row_stride(p) >= p.k && b_row_stride(p) >= p.k &&
      d_row_stride(p) >= p.n && a_row_stride(p) % kFp8Alignment == 0 &&
      a_batch_stride(p) % kFp8Alignment == 0 &&
      b_row_stride(p) % kFp8Alignment == 0 &&
      b_batch_stride(p) % kFp8Alignment == 0 &&
      d_row_stride(p) % kAlignment == 0 && d_batch_stride(p) % kAlignment == 0;
  const bool batches = p.l == 1 ||
      (a_batch_stride(p) >= a_row_stride(p) * p.m &&
       (b_batch_stride(p) == 0 || b_batch_stride(p) >= b_row_stride(p) * p.n) &&
       d_batch_stride(p) >= d_row_stride(p) * p.m);
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l > 0 && unit_strides &&
      leading_dimensions && batches && !p.transpose_a && !p.transpose_b &&
      p.input_dtype == DType::kFloat8E4M3 &&
      p.weight_dtype == DType::kFloat8E4M3 &&
      p.output_dtype == DType::kBfloat16 &&
      (p.max_swizzle_size == 1 || p.max_swizzle_size == 2 ||
       p.max_swizzle_size == 4 || p.max_swizzle_size == 8);
}

bool supported_route_base(const UlyssesRoute& route) {
  return route.world_size > 0 && route.world_size <= kMaxWorldSize &&
      route.rank >= 0 && route.rank < route.world_size && route.batch > 0 &&
      route.seq_local > 0 && route.global_seq > 0;
}

bool use_wide_qkv_policy(const GemmProblem& problem) {
  return problem.m >= 2048;
}

using BaseEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator,
    Accumulator,
    void,
    LayoutD,
    kAlignment,
    Element,
    LayoutD,
    kAlignment,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Element,
    LayoutA,
    kAlignment,
    Element,
    LayoutB,
    kAlignment,
    Accumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename BaseEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using ObservedEpilogue = detail::SignalingEpilogue<BaseEpilogue>;

using PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BaseMainloop,
    BaseEpilogue,
    cutlass::gemm::PersistentScheduler>;

using A2ALhsInputMainloop = detail::A2ALhsReadyMainloop<BaseMainloop>;
using A2ALhsInputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsInputMainloop,
    BaseEpilogue,
    detail::MonolithicPersistentScheduler>;

using N160Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        N160TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using N160Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        N160TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename N160Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using A2ALhsN160Mainloop = detail::A2ALhsReadyMainloop<N160Mainloop>;
using A2ALhsN160Gemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsN160Mainloop,
    N160Epilogue,
    detail::MonolithicPersistentScheduler>;
using N160PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N160Mainloop,
    N160Epilogue,
    cutlass::gemm::PersistentScheduler>;

#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryMainloop =
    detail::A2ALhsReadyMainloop<BaseMainloop, 1, true>;
using A2ALhsTelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsTelemetryMainloop,
    BaseEpilogue,
    detail::MonolithicPersistentScheduler>;
#endif
// Small-M dense GEMMs need twice as many M tiles to fill H200.  Keep these
// CTAs independent so an early-ready M64 tile is never held behind its peer.
using M64Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        M64TileShape,
        M64ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using M64Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        M64TileShape,
        M64ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename M64Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using M64PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    M64Mainloop,
    M64Epilogue,
    cutlass::gemm::PersistentScheduler>;

using A2ALhsM64Mainloop = detail::A2ALhsReadyMainloop<M64Mainloop>;
using A2ALhsM64Gemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsM64Mainloop,
    M64Epilogue,
    detail::MonolithicPersistentScheduler>;
#if FUSE_ENABLE_PROFILING
using A2ALhsM64TelemetryMainloop =
    detail::A2ALhsReadyMainloop<M64Mainloop, 1, true>;
using A2ALhsM64TelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsM64TelemetryMainloop,
    M64Epilogue,
    detail::MonolithicPersistentScheduler>;
#endif
using OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BaseMainloop,
    ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;

// Wide-N dense output routing follows the Flux-style cooperative policy. The
// generic A2A -> GEMM path keeps its independent 128x128 ping-pong policy.
using ProjectionEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using ProjectionMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCount<4>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using ProjectionObservedEpilogue = detail::SignalingEpilogue<ProjectionEpilogue>;
using ProjectionOutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    ProjectionMainloop,
    ProjectionObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using ProjectionPureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    ProjectionMainloop,
    ProjectionEpilogue,
    cutlass::gemm::PersistentScheduler>;

using A2ALhsProjectionMainloop =
    detail::A2ALhsReadyMainloop<ProjectionMainloop>;
using A2ALhsProjectionGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsProjectionMainloop,
    ProjectionEpilogue,
    detail::MonolithicPersistentScheduler>;
#if FUSE_ENABLE_PROFILING
using A2ALhsProjectionTelemetryMainloop =
    detail::A2ALhsReadyMainloop<ProjectionMainloop, 1, true>;
using A2ALhsProjectionTelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsProjectionTelemetryMainloop,
    ProjectionEpilogue,
    detail::MonolithicPersistentScheduler>;
#endif

// Wide-N cluster policy for frontier-aligned persistent waves.
using WideN320Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        WideN320TileShape,
        ProjectionClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using WideN320Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        WideN320TileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename WideN320Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using A2ALhsWideN320Mainloop =
    detail::A2ALhsReadyMainloop<WideN320Mainloop>;
using A2ALhsWideN320Gemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsWideN320Mainloop,
    WideN320Epilogue,
    detail::MonolithicPersistentScheduler>;
using WideN320PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    WideN320Mainloop,
    WideN320Epilogue,
    cutlass::gemm::PersistentScheduler>;
#if FUSE_ENABLE_PROFILING
using A2ALhsWideN320TelemetryMainloop =
    detail::A2ALhsReadyMainloop<WideN320Mainloop, 1, true>;
using A2ALhsWideN320TelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsWideN320TelemetryMainloop,
    WideN320Epilogue,
    detail::MonolithicPersistentScheduler>;
#endif

using Fp8BaseEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Fp8TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator,
    Accumulator,
    void,
    LayoutD,
    kAlignment,
    Element,
    LayoutD,
    kAlignment,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using Fp8BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Fp8Element,
    LayoutA,
    kFp8Alignment,
    Fp8Element,
    LayoutB,
    kFp8Alignment,
    Accumulator,
    Fp8TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Fp8BaseEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::CollectiveOp;

using Fp8ObservedEpilogue = detail::SignalingEpilogue<Fp8BaseEpilogue>;
using Fp8OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BaseMainloop,
    Fp8ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BaseMainloop,
    Fp8BaseEpilogue,
    cutlass::gemm::PersistentScheduler>;

__host__ __device__ constexpr int32_t ceil_div(int32_t x, int32_t y) {
  return (x + y - 1) / y;
}

// Pull the sequence rows owned by this destination rank from every
// head-sharded peer output and place them directly in the row-major GEMM A
// staging matrix. A peer completes one contiguous K interval for each M tile;
// the CUTLASS producer can therefore advance peer by peer without waiting for
// the full inverse A2A.
#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsCommTimelineArguments {};

template <>
struct A2ALhsCommTimelineArguments<true> {
  A2AGemmPeerTimeline* peer_timeline = nullptr;
  int32_t peer_timeline_capacity = 0;
};
#endif

#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsCommStageSample {};

template <>
struct A2ALhsCommStageSample<true> {
  uint64_t task_begin = 0;
  uint64_t input_ready = 0;
  uint64_t g2s_issue = 0;
  uint64_t g2s_done = 0;
  uint64_t s2g_issue = 0;
  uint64_t s2g_done = 0;
  int32_t comm_cta = 0;
  int32_t comm_slot = 0;
  int32_t task_id = 0;
  int32_t row_chunk = 0;
  int32_t copy_rows = 0;
  int32_t source_rank = 0;
  int32_t copy_path = 0;
};
#endif

cudaError_t make_a2a_lhs_store_tma_3d(
    CUtensorMap* tensor_map,
    void* pointer,
    int32_t inner_u64,
    int32_t total_groups,
    int32_t rows,
    int32_t peer_groups,
    int32_t box_rows) {
  const uint64_t global_dims[3] = {
      static_cast<uint64_t>(inner_u64),
      static_cast<uint64_t>(total_groups),
      static_cast<uint64_t>(rows)};
  const uint64_t global_strides[2] = {
      static_cast<uint64_t>(inner_u64) * sizeof(uint64_t),
      static_cast<uint64_t>(inner_u64) * total_groups * sizeof(uint64_t)};
  const uint32_t box_dims[3] = {
      static_cast<uint32_t>(inner_u64),
      static_cast<uint32_t>(peer_groups),
      static_cast<uint32_t>(box_rows)};
  constexpr uint32_t element_strides[3] = {1, 1, 1};
  const CUresult result =
      CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          tensor_map,
          CU_TENSOR_MAP_DATA_TYPE_UINT64,
          3,
          pointer,
          global_dims,
          global_strides,
          box_dims,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <
    int32_t ReadyBlockM
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
struct A2ALhsInputCommT {
  static constexpr int32_t kReadyBlockM = ReadyBlockM;
  static constexpr size_t SharedStorageBytes =
      kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
      kA2ALhsBulkSlots * sizeof(uint64_t);
  struct Arguments
#if FUSE_ENABLE_PROFILING
      : A2ALhsCommTimelineArguments<Instrumented>
#endif
  {
    A2AGemmParams params{};
    CUtensorMap store_tma_full{};
    int32_t comm_rows = kA2ALhsCommRows;
    int32_t m_window = 2;
    int32_t store_peer_groups = 0;
    int32_t store_rows = 0;
    bool use_bulk = false;
    bool use_tensor_store = false;
  };
  using Params = Arguments;

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    if (route.world_size <= 0) {
      return cudaErrorInvalidValue;
    }
    const int32_t shard_width = route.local_heads * route.head_dim;
    const int32_t row_bytes = shard_width * sizeof(Element);
    const int32_t max_rows = row_bytes > 0
        ? kA2ALhsBulkStageBytes / row_bytes
        : 0;
    args.comm_rows = min(ReadyBlockM, max_rows);
    args.use_bulk = args.comm_rows > 0 && row_bytes % 16 == 0 &&
        route.seq_local % ReadyBlockM == 0 &&
        (!route.causal_load_balanced ||
         (route.seq_local / 2) % ReadyBlockM == 0);
    if (!args.use_bulk) {
      args.comm_rows = kA2ALhsCommRows;
      return cudaSuccess;
    }

    constexpr int32_t kTensorStoreBytes = 8 * 1024;
    if (row_bytes > 1024) {
      return cudaSuccess;
    }
    args.store_rows = min(args.comm_rows, kTensorStoreBytes / row_bytes);
    if (args.store_rows < 2) {
      return cudaSuccess;
    }

    // Store a narrow peer shard as a small 3D tensor instead of one TMA per
    // row. Four BF16 values form one raw 64-bit tensor element; the first two
    // dimensions factor the contiguous K shard and the third is M.
    const int32_t shard_u64 = row_bytes / sizeof(uint64_t);
    int32_t inner_u64 = 0;
    for (int32_t candidate = 256; candidate >= 2; candidate >>= 1) {
      if (shard_u64 % candidate == 0) {
        inner_u64 = candidate;
        break;
      }
    }
    const int32_t peer_groups = inner_u64 > 0
        ? shard_u64 / inner_u64
        : 0;
    if (inner_u64 == 0 || peer_groups <= 0 || peer_groups > 256) {
      return cudaSuccess;
    }

    const int32_t total_groups = peer_groups * route.world_size;
    const cudaError_t status = make_a2a_lhs_store_tma_3d(
        &args.store_tma_full,
        p.input_staging,
        inner_u64,
        total_groups,
        p.gemm.m,
        peer_groups,
        args.store_rows);
    if (status != cudaSuccess) {
      return cudaSuccess;
    }
    args.store_peer_groups = peer_groups;
    args.use_tensor_store = true;
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    bool pointers = p.input_staging && p.rhs_nt && p.output && p.ready;
    for (int32_t peer = 0; peer < route.world_size && pointers; ++peer) {
      pointers = p.peer_input[peer] != nullptr &&
          (p.input_epoch == 0 || p.peer_input_ready[peer] != nullptr);
    }
    return pointers && p.epoch > 0 && supported_problem(p.gemm) &&
        supported_route_base(route) &&
        route.kind == RouteKind::kHeadToSequence &&
        route.direction == RouteDirection::kInverse &&
        route.q_heads > 0 && route.local_heads > 0 &&
        route.q_heads == route.local_heads * route.world_size &&
        route.head_dim > 0 && route.head_dim % kAlignment == 0 &&
        route.global_seq == route.seq_local * route.world_size &&
        (!route.causal_load_balanced || route.seq_local % 2 == 0) &&
        p.gemm.l == 1 && p.gemm.m == route.batch * route.seq_local &&
        p.gemm.k == route.q_heads * route.head_dim &&
        a_row_stride(p.gemm) == p.gemm.k;
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  __host__ __device__ static int32_t arrivals_per_peer(
      const Params& args) {
    return ceil_div(ReadyBlockM, args.comm_rows);
  }

  CUTLASS_DEVICE static void publish_ready(
      const Params& args,
      int32_t tile_m,
      int32_t peer_slot) {
    const auto& p = args.params;
    const auto& route = p.route;
    auto* ready = p.ready +
        (static_cast<int64_t>(tile_m) * route.world_size + peer_slot) *
            kReadyFlagStride;
    detail::add_release_gpu(ready);
  }

#if FUSE_ENABLE_PROFILING
  CUTLASS_DEVICE static void publish_ready_instrumented(
      const Params& args,
      int32_t tile_m,
      int32_t peer_slot,
      const A2ALhsCommStageSample<true>& sample) {
    const auto& p = args.params;
    const auto& route = p.route;
    auto* ready = p.ready +
        (static_cast<int64_t>(tile_m) * route.world_size + peer_slot) *
            kReadyFlagStride;
    const uint64_t release_issue = static_cast<unsigned long long>(
        detail::read_global_timer());
    const uint32_t old = detail::add_release_gpu_fetch_old(ready);
    const uint64_t release_done = static_cast<unsigned long long>(
        detail::read_global_timer());
    const uint32_t target = p.epoch * arrivals_per_peer(args);
    const int32_t index = tile_m * route.world_size + peer_slot;
    if (old + 1 == target && args.peer_timeline && index >= 0 &&
        index < args.peer_timeline_capacity) {
      auto& event = args.peer_timeline[index];
      event.release = release_done;
      event.task_begin = sample.task_begin;
      event.input_ready = sample.input_ready;
      event.g2s_issue = sample.g2s_issue;
      event.g2s_done = sample.g2s_done;
      event.s2g_issue = sample.s2g_issue;
      event.s2g_done = sample.s2g_done;
      event.publish_issue = release_issue;
      event.comm_cta = sample.comm_cta;
      event.comm_slot = sample.comm_slot;
      event.task_id = sample.task_id;
      event.row_chunk = sample.row_chunk;
      event.copy_rows = sample.copy_rows;
      event.source_rank = sample.source_rank;
      event.copy_path = sample.copy_path;
      event.comm_valid = 1;
    }
  }
#endif

  CUTLASS_DEVICE static int32_t global_sequence_row(
      const UlyssesRoute& route,
      int32_t local_sequence) {
    if (!route.causal_load_balanced) {
      return route.rank * route.seq_local + local_sequence;
    }
    const int32_t chunk_rows = route.seq_local / 2;
    const int32_t chunk = local_sequence < chunk_rows
        ? route.rank
        : 2 * route.world_size - route.rank - 1;
    const int32_t row_in_chunk = local_sequence < chunk_rows
        ? local_sequence
        : local_sequence - chunk_rows;
    return chunk * chunk_rows + row_in_chunk;
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas) {
    const auto& p = args.params;
    const auto& route = p.route;
    const int32_t m_tiles = ceil_div(p.gemm.m, ReadyBlockM);
    const int32_t chunks_per_tile = arrivals_per_peer(args);
    const int32_t tasks_per_m = route.world_size * chunks_per_tile;
    const int32_t tasks = m_tiles * tasks_per_m;
    const int32_t shard_width = route.local_heads * route.head_dim;
    constexpr int32_t elements_per_vector = kAlignment;
    const int32_t vectors_per_row = shard_width / elements_per_vector;
    int32_t waited_peer = -1;

    if (args.use_bulk) {
      extern __shared__ char dynamic_smem[];
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kA2ALhsBulkSlots * kA2ALhsBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot >= kA2ALhsBulkSlots) {
        return;
      }
      auto* stage = reinterpret_cast<Element*>(
          dynamic_smem + slot * kA2ALhsBulkStageBytes);
      uint64_t* barrier = barriers + slot;
      int32_t phase = 0;
      if (lane == 0) {
        cute::initialize_barrier(*barrier, 1);
        if (args.use_tensor_store) {
          cute::prefetch_tma_descriptor(&args.store_tma_full);
        }
      }
      __syncwarp();

      const int32_t task_peer_count = route.world_size;
      const int32_t bulk_tasks =
          m_tiles * task_peer_count * chunks_per_tile;
      const int32_t task_stride = comm_ctas * kA2ALhsBulkSlots;
      for (int32_t task = slot * comm_ctas + comm_id;
           task < bulk_tasks;
           task += task_stride) {
#if FUSE_ENABLE_PROFILING
        A2ALhsCommStageSample<Instrumented> sample{};
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.task_begin = detail::read_global_timer();
            sample.comm_cta = comm_id;
            sample.comm_slot = slot;
            sample.task_id = task;
            sample.copy_path = args.use_tensor_store ? 2 : 1;
          }
        }
#endif
        const int32_t m_window = args.m_window;
        const int32_t full_windows = m_tiles / m_window;
        const int32_t tasks_per_window =
            m_window * task_peer_count * chunks_per_tile;
        const int32_t full_window_tasks = full_windows * tasks_per_window;
        int32_t tile_m = 0;
        int32_t peer_slot = 0;
        int32_t row_chunk = 0;
        if (task < full_window_tasks) {
          const int32_t window = task / tasks_per_window;
          const int32_t task_in_window = task - window * tasks_per_window;
          peer_slot = task_in_window / (m_window * chunks_per_tile);
          const int32_t task_in_peer =
              task_in_window - peer_slot * m_window * chunks_per_tile;
          const int32_t m_in_window = task_in_peer / chunks_per_tile;
          row_chunk = task_in_peer - m_in_window * chunks_per_tile;
          tile_m = window * m_window + m_in_window;
        } else {
          const int32_t tail = task - full_window_tasks;
          const int32_t tail_m_tiles = m_tiles - full_windows * m_window;
          peer_slot = tail / (tail_m_tiles * chunks_per_tile);
          const int32_t task_in_peer =
              tail - peer_slot * tail_m_tiles * chunks_per_tile;
          const int32_t m_in_window = task_in_peer / chunks_per_tile;
          row_chunk = task_in_peer - m_in_window * chunks_per_tile;
          tile_m = full_windows * m_window + m_in_window;
        }
        const int32_t row_in_tile = row_chunk * args.comm_rows;
        const int32_t source_peer = route.cyclic_peer_order
            ? (route.rank + peer_slot) % route.world_size
            : peer_slot;
        const int32_t m_begin = tile_m * ReadyBlockM + row_in_tile;
        const int32_t copy_rows = max(
            0,
            min(min(args.comm_rows, ReadyBlockM - row_in_tile),
                p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.row_chunk = row_chunk;
            sample.copy_rows = copy_rows;
            sample.source_rank = source_peer;
          }
        }
#endif

        if (p.input_epoch != 0 && source_peer != waited_peer) {
          detail::wait_acquire_system(
              p.peer_input_ready[source_peer], p.input_epoch, lane);
          waited_peer = source_peer;
        }

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.input_ready = detail::read_global_timer();
          }
        }
#endif

        if (lane == 0 && copy_rows > 0) {
          const int32_t batch = m_begin / route.seq_local;
          const int32_t local_sequence =
              m_begin - batch * route.seq_local;
          const int32_t source_sequence =
              global_sequence_row(route, local_sequence);
          const auto* source = p.peer_input[source_peer] +
              (static_cast<int64_t>(batch) * route.global_seq +
               source_sequence) * shard_width;
          const int32_t copy_bytes =
              copy_rows * shard_width * sizeof(Element);
          cute::set_barrier_transaction_bytes(*barrier, copy_bytes);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_issue = detail::read_global_timer();
          }
#endif
          cute::SM90_BULK_COPY_G2S::copy(
              source, barrier, stage, copy_bytes);
          cute::wait_barrier(*barrier, phase);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_done = detail::read_global_timer();
          }
#endif
          phase ^= 1;
          cute::tma_store_fence();
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.s2g_issue = detail::read_global_timer();
          }
#endif
          if (args.use_tensor_store) {
            int32_t row = 0;
            for (; row + args.store_rows <= copy_rows;
                 row += args.store_rows) {
              cute::SM90_TMA_STORE_3D::copy(
                  &args.store_tma_full,
                  stage + static_cast<int64_t>(row) * shard_width,
                  0,
                  peer_slot * args.store_peer_groups,
                  m_begin + row);
            }
            if (row > 0) {
              cute::tma_store_arrive();
            }
            int32_t residual_ops = 0;
            for (; row < copy_rows; ++row) {
              cute::SM90_BULK_COPY_S2G::copy(
                  stage + static_cast<int64_t>(row) * shard_width,
                  p.input_staging +
                      static_cast<int64_t>(m_begin + row) * p.gemm.k +
                      peer_slot * shard_width,
                  shard_width * sizeof(Element));
              if (++residual_ops == 8) {
                cute::tma_store_arrive();
                residual_ops = 0;
              }
            }
            if (residual_ops != 0) {
              cute::tma_store_arrive();
            }
          } else {
            for (int32_t row = 0; row < copy_rows; ++row) {
              cute::SM90_BULK_COPY_S2G::copy(
                  stage + static_cast<int64_t>(row) * shard_width,
                  p.input_staging +
                      static_cast<int64_t>(m_begin + row) * p.gemm.k +
                      peer_slot * shard_width,
                  shard_width * sizeof(Element));
              if ((row & 7) == 7) {
                cute::tma_store_arrive();
              }
            }
            if ((copy_rows & 7) != 0) {
              cute::tma_store_arrive();
            }
          }
          cute::tma_store_wait<0>();
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.s2g_done = detail::read_global_timer();
            publish_ready_instrumented(args, tile_m, peer_slot, sample);
          } else
#endif
          {
            publish_ready(args, tile_m, peer_slot);
          }
        } else if (lane == 0) {
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            publish_ready_instrumented(args, tile_m, peer_slot, sample);
          } else
#endif
          {
            publish_ready(args, tile_m, peer_slot);
          }
        }
        __syncwarp();
      }
      if (lane == 0) {
        cutlass::arch::ClusterBarrier::invalidate(barrier);
      }
      return;
    }

    for (int32_t task = comm_id; task < tasks; task += comm_ctas) {
#if FUSE_ENABLE_PROFILING
      A2ALhsCommStageSample<Instrumented> sample{};
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.task_begin = detail::read_global_timer();
          sample.comm_cta = comm_id;
          sample.comm_slot = -1;
          sample.task_id = task;
          sample.copy_path = 0;
        }
      }
#endif
      const int32_t tile_m = task / tasks_per_m;
      const int32_t task_in_m = task - tile_m * tasks_per_m;
      const int32_t peer_slot = task_in_m / chunks_per_tile;
      const int32_t row_chunk = task_in_m - peer_slot * chunks_per_tile;
      const int32_t source_peer = route.cyclic_peer_order
          ? (route.rank + peer_slot) % route.world_size
          : peer_slot;
      const int32_t m_begin = tile_m * ReadyBlockM +
          row_chunk * args.comm_rows;
      const int32_t copy_rows = max(
          0, min(args.comm_rows, p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.row_chunk = row_chunk;
          sample.copy_rows = copy_rows;
          sample.source_rank = source_peer;
        }
      }
#endif

      if (p.input_epoch != 0 && source_peer != waited_peer) {
        if (threadIdx.x < 32) {
          detail::wait_acquire_system(
              p.peer_input_ready[source_peer], p.input_epoch,
              static_cast<int32_t>(threadIdx.x));
        }
        __syncthreads();
        waited_peer = source_peer;
      }
#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.input_ready = detail::read_global_timer();
          sample.g2s_issue = sample.input_ready;
        }
      }
#endif

      const auto* source =
          reinterpret_cast<const uint4*>(p.peer_input[source_peer]);
      auto* destination = reinterpret_cast<uint4*>(p.input_staging);
      const int32_t vector_count = copy_rows * vectors_per_row;
      for (int32_t index = static_cast<int32_t>(threadIdx.x);
           index < vector_count;
           index += static_cast<int32_t>(blockDim.x)) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_k = index - row * vectors_per_row;
        const int32_t destination_row = m_begin + row;
        const int32_t batch = destination_row / route.seq_local;
        const int32_t local_sequence =
            destination_row - batch * route.seq_local;
        const int32_t source_sequence =
            global_sequence_row(route, local_sequence);
        const int64_t src =
            (static_cast<int64_t>(batch) * route.global_seq +
             source_sequence) *
                vectors_per_row +
            vector_k;
        const int64_t dst =
            static_cast<int64_t>(destination_row) *
                (p.gemm.k / kAlignment) +
            peer_slot * vectors_per_row + vector_k;
        destination[dst] = source[src];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          sample.g2s_done = detail::read_global_timer();
          publish_ready_instrumented(args, tile_m, peer_slot, sample);
        } else
#endif
        {
          publish_ready(args, tile_m, peer_slot);
        }
      }
      __syncthreads();
    }
  }
};

using A2ALhsInputComm = A2ALhsInputCommT<kBlockM>;
using A2ALhsM64InputComm = A2ALhsInputCommT<64>;
#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryInputComm = A2ALhsInputCommT<kBlockM, true>;
using A2ALhsM64TelemetryInputComm = A2ALhsInputCommT<64, true>;
#endif

using A2ALhsGemmKernel =
    detail::MonolithicGemm<A2ALhsInputGemm, A2ALhsInputComm>;
using A2ALhsN160GemmKernel =
    detail::MonolithicGemm<A2ALhsN160Gemm, A2ALhsInputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryBase =
    detail::MonolithicGemm<A2ALhsTelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsTelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsTelemetryBase>;
#endif
using A2ALhsProjectionGemmKernel =
    detail::MonolithicGemm<A2ALhsProjectionGemm, A2ALhsInputComm>;
using A2ALhsWideN320GemmKernel =
    detail::MonolithicGemm<A2ALhsWideN320Gemm, A2ALhsInputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsProjectionTelemetryBase = detail::MonolithicGemm<
    A2ALhsProjectionTelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsProjectionTelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsProjectionTelemetryBase>;
using A2ALhsWideN320TelemetryBase = detail::MonolithicGemm<
    A2ALhsWideN320TelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsWideN320TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsWideN320TelemetryBase>;
#endif
using A2ALhsM64GemmKernel =
    detail::MonolithicGemm<A2ALhsM64Gemm, A2ALhsM64InputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsM64TelemetryBase =
    detail::MonolithicGemm<
        A2ALhsM64TelemetryGemm, A2ALhsM64TelemetryInputComm>;
using A2ALhsM64TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsM64TelemetryBase>;
#endif
static_assert(
    sizeof(typename A2ALhsGemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsN160GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "N160 A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsProjectionGemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "wide A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsWideN320GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "N320 A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsM64GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "M64 A2A LHS monolithic shared storage must hold every bulk slot");

template <
    class ParamsType,
    bool IsFp8,
    int32_t BlockM,
    int32_t BlockN,
    int32_t MTilesPerTask = 1,
    bool PeerInterleaved = false,
    int32_t CopyBlockN = BlockN>
struct QkvGqaPackCommT {
  struct Arguments {
    ParamsType params{};
    cute::TmaDescriptor local_output_tma{};
    cute::TmaDescriptor peer_output_tma[kMaxWorldSize][3]{};
    bool use_tma = false;
    bool use_tma_store = false;
  };
  using Params = Arguments;
  static constexpr size_t SharedStorageBytes =
      kQkvBulkSlots * kQkvBulkStageBytes +
      kQkvBulkSlots * sizeof(uint64_t);
  static constexpr bool kNeedsGridFinalize = true;

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    args.use_tma = p.route.head_dim == kQkvBulkColumns &&
        d_row_stride(p.gemm) >= p.gemm.n &&
        d_row_stride(p.gemm) % kAlignment == 0;
    if (!args.use_tma) {
      return cudaSuccess;
    }
    const uint64_t global_dims[2] = {
        static_cast<uint64_t>(p.gemm.n),
        static_cast<uint64_t>(p.gemm.m)};
    const uint64_t global_strides[1] = {
        static_cast<uint64_t>(d_row_stride(p.gemm)) * sizeof(Element)};
    constexpr uint32_t box_dims[2] = {
        kQkvBulkColumns, kQkvBulkRows};
    constexpr uint32_t element_strides[2] = {1, 1};
    CUresult result =
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.local_output_tma,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<Element*>(p.local_output),
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
      return cudaErrorInvalidValue;
    }

    args.use_tma_store = p.gemm.m % kQkvBulkRows == 0 &&
        p.route.seq_local % kQkvBulkRows == 0;
    if (!args.use_tma_store) {
      return cudaSuccess;
    }
    const int32_t q_local_width =
        p.route.q_heads / p.route.world_size * p.route.head_dim;
    const int32_t kv_local_width =
        p.route.kv_heads / p.route.world_size * p.route.head_dim;
    const uint64_t output_rows =
        static_cast<uint64_t>(p.route.batch) * p.route.global_seq;
    for (int32_t peer = 0; peer < p.route.world_size; ++peer) {
      const int32_t descriptor_count = p.route.defer_v_a2a ? 1 : 3;
      for (int32_t segment = 0; segment < descriptor_count; ++segment) {
        const int32_t segment_width = p.route.defer_v_a2a
            ? q_local_width + kv_local_width
            : (segment == 0 ? q_local_width : kv_local_width);
        const int64_t segment_offset = p.route.defer_v_a2a || segment == 0
            ? 0
            : static_cast<int64_t>(output_rows) * q_local_width +
                (segment == 2
                     ? static_cast<int64_t>(output_rows) * kv_local_width
                     : 0);
        const uint64_t destination_dims[2] = {
            static_cast<uint64_t>(segment_width), output_rows};
        const uint64_t destination_strides[1] = {
            static_cast<uint64_t>(segment_width) * sizeof(Element)};
        result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.peer_output_tma[peer][segment],
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<Element*>(p.peer_output[peer] + segment_offset),
            destination_dims,
            destination_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          return cudaErrorInvalidValue;
        }
      }
    }
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    bool pointers = supported_route_base(p.route) && p.lhs && p.rhs_nt &&
        p.local_output && p.ready;
    for (int32_t peer = 0; peer < p.route.world_size && pointers; ++peer) {
      pointers = p.peer_output[peer] && p.peer_route_done_epoch[peer];
    }
    const int32_t packed_heads = p.route.q_heads + 2 * p.route.kv_heads;
    const bool problem_supported = IsFp8
        ? supported_fp8_problem(p.gemm)
        : supported_problem(p.gemm);
    return pointers && p.epoch > 0 && problem_supported &&
        p.route.kind == RouteKind::kQkvGqaPack &&
        p.route.direction == RouteDirection::kForward && p.gemm.l == 1 &&
        p.route.q_heads > 0 && p.route.kv_heads > 0 &&
        p.route.q_heads % p.route.kv_heads == 0 &&
        p.route.q_heads % p.route.world_size == 0 &&
        p.route.kv_heads % p.route.world_size == 0 &&
        p.route.head_dim > 0 && p.route.head_dim % kAlignment == 0 &&
        p.gemm.m == p.route.batch * p.route.seq_local &&
        p.route.global_seq == p.route.seq_local * p.route.world_size &&
        p.gemm.n == packed_heads * p.route.head_dim &&
        p.gemm.n % kAlignment == 0 &&
        p.route.qkv_peer_interleaved == PeerInterleaved &&
        (!PeerInterleaved || p.gemm.raster != GemmRaster::kAlongN);
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  CUTLASS_DEVICE static int32_t source_feature(
      const UlyssesRoute& route,
      int32_t destination_rank,
      int32_t segment,
      int32_t local_head,
      int32_t head_offset) {
    const int32_t q_width = route.q_heads * route.head_dim;
    const int32_t kv_width = route.kv_heads * route.head_dim;
    const int32_t segment_base =
        segment == 0 ? 0 : q_width + (segment == 2 ? kv_width : 0);
    const int32_t global_heads =
        segment == 0 ? route.q_heads : route.kv_heads;
    const int32_t local_heads = global_heads / route.world_size;
    const int32_t physical_head =
        PeerInterleaved && segment < 2
        ? local_head * route.world_size + destination_rank
        : destination_rank * local_heads + local_head;
    return segment_base + physical_head * route.head_dim + head_offset;
  }

  CUTLASS_DEVICE void run(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas,
      bool wait_for_gemm) {
    const auto& p = args.params;
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t m_groups = ceil_div(m_tiles, MTilesPerTask);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t q_local_heads = p.route.q_heads / p.route.world_size;
    const int32_t kv_local_heads = p.route.kv_heads / p.route.world_size;
    const int32_t route_heads =
        q_local_heads + (p.route.defer_v_a2a ? 1 : 2) * kv_local_heads;
    const int32_t chunks_per_head = ceil_div(p.route.head_dim, CopyBlockN);
    const int32_t head_chunks = route_heads * chunks_per_head;
    const int32_t tasks =
        p.route.world_size * m_groups * head_chunks;
    const int64_t source_row_vectors = d_row_stride(p.gemm) / kAlignment;
    const int32_t q_local_width = q_local_heads * p.route.head_dim;
    const int32_t kv_local_width = kv_local_heads * p.route.head_dim;
    const int64_t segment_rows =
        static_cast<int64_t>(p.route.batch) * p.route.global_seq;

    if (args.use_tma) {
      static_assert(CopyBlockN >= kQkvBulkColumns);
      extern __shared__ char dynamic_smem[];
      auto* stages = reinterpret_cast<Element*>(dynamic_smem);
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kQkvBulkSlots * kQkvBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot < kQkvBulkSlots) {
        Element* stage = stages +
            static_cast<int64_t>(slot) * kQkvBulkRows * kQkvBulkColumns;
        uint64_t* barrier = barriers + slot;
        int32_t phase = 0;
        if (lane == 0) {
          cute::initialize_barrier(*barrier, 1);
        }
        __syncwarp();

        const int32_t m_chunks = ceil_div(p.gemm.m, kQkvBulkRows);
        const int32_t bulk_tasks =
            p.route.world_size * m_chunks * head_chunks;
        const int32_t task_stride = comm_ctas * kQkvBulkSlots;
        for (int32_t work = slot * comm_ctas + comm_id;
             work < bulk_tasks;
             work += task_stride) {
          const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
          const int32_t destination_rank = work % p.route.world_size;
          const int32_t rank_work = work / p.route.world_size;
          const int32_t m_chunk = along_n
              ? rank_work / head_chunks
              : rank_work % m_chunks;
          const int32_t head_chunk = along_n
              ? rank_work % head_chunks
              : rank_work / m_chunks;
          const int32_t head_slot = head_chunk / chunks_per_head;
          const int32_t chunk = head_chunk - head_slot * chunks_per_head;
          const int32_t segment = head_slot < q_local_heads
              ? 0
              : (head_slot < q_local_heads + kv_local_heads ? 1 : 2);
          const int32_t local_head = segment == 0
              ? head_slot
              : head_slot - q_local_heads -
                  (segment == 2 ? kv_local_heads : 0);
          const int32_t head_offset = chunk * CopyBlockN;
          const int32_t copy_n =
              min(CopyBlockN, p.route.head_dim - head_offset);
          const int32_t physical_feature = source_feature(
              p.route,
              destination_rank,
              segment,
              local_head,
              head_offset);
          const int32_t m_begin = m_chunk * kQkvBulkRows;
          const int32_t copy_m =
              min(kQkvBulkRows, p.gemm.m - m_begin);

          if (wait_for_gemm) {
            const int32_t producer_m = m_begin / BlockM;
            const int32_t first_producer_n = physical_feature / BlockN;
            const int32_t last_producer_n =
                (physical_feature + copy_n - 1) / BlockN;
            for (int32_t producer_n = first_producer_n;
                 producer_n <= last_producer_n;
                 ++producer_n) {
              detail::wait_acquire_system(
                  p.ready +
                      (producer_m * output_n_tiles + producer_n) *
                          kReadyFlagStride,
                  p.epoch,
                  lane);
            }
          }
          __syncwarp();

          if (lane == 0) {
            cute::set_barrier_transaction_bytes(
                *barrier, kQkvBulkStageBytes);
            cute::SM90_TMA_LOAD_2D::copy(
                &args.local_output_tma,
                barrier,
                0x12f0000000000000ull,
                stage,
                physical_feature,
                m_begin);
            cute::wait_barrier(*barrier, phase);
            phase ^= 1;
            cute::tma_store_fence();

            auto* destination = p.peer_output[destination_rank];
            const int32_t segment_width =
                segment == 0 ? q_local_width : kv_local_width;
            const int32_t local_segment_base =
                segment == 0 ? 0 : q_local_width +
                    (segment == 2 ? kv_local_width : 0);
            const int64_t segment_offset = segment == 0
                ? 0
                : segment_rows * q_local_width +
                    (segment == 2 ? segment_rows * kv_local_width : 0);
            const int32_t local_feature =
                local_segment_base + local_head * p.route.head_dim +
                head_offset;
            if (args.use_tma_store) {
              const int32_t batch = m_begin / p.route.seq_local;
              const int32_t local_seq =
                  m_begin - batch * p.route.seq_local;
              const int32_t destination_row =
                  batch * p.route.global_seq +
                  p.route.rank * p.route.seq_local + local_seq;
              const int32_t destination_feature = p.route.defer_v_a2a
                  ? local_feature
                  : local_feature - local_segment_base;
              const int32_t descriptor =
                  p.route.defer_v_a2a ? 0 : segment;
              cute::SM90_TMA_STORE_2D::copy(
                  &args.peer_output_tma[destination_rank][descriptor],
                  stage,
                  destination_feature,
                  destination_row);
              cute::tma_store_arrive();
            } else {
              for (int32_t row = 0; row < copy_m; ++row) {
                const int32_t source_row = m_begin + row;
                const int32_t batch = source_row / p.route.seq_local;
                const int32_t local_seq =
                    source_row - batch * p.route.seq_local;
                const int64_t destination_row =
                    static_cast<int64_t>(batch) * p.route.global_seq +
                    p.route.rank * p.route.seq_local + local_seq;
                const int64_t dst = p.route.defer_v_a2a
                    ? destination_row * (q_local_width + kv_local_width) +
                        local_feature
                    : segment_offset + destination_row * segment_width +
                        local_feature - local_segment_base;
                cute::SM90_BULK_COPY_S2G::copy(
                    stage + static_cast<int64_t>(row) * kQkvBulkColumns,
                    destination + dst,
                    kQkvBulkColumns * sizeof(Element));
                if ((row & 7) == 7) {
                  cute::tma_store_arrive();
                }
              }
              if ((copy_m & 7) != 0) {
                cute::tma_store_arrive();
              }
            }
            cute::tma_store_wait<0>();
          }
          __syncwarp();
        }
        if (lane == 0) {
          cutlass::arch::ClusterBarrier::invalidate(barrier);
        }
      }
      __syncthreads();
      if (wait_for_gemm && p.route.defer_v_a2a && p.completion_epoch &&
          comm_id == 0) {
        wait_and_publish_completion(args);
      }
      return;
    }

    for (int32_t work = comm_id; work < tasks; work += comm_ctas) {
      const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
      const int32_t destination_rank = work % p.route.world_size;
      const int32_t rank_work = work / p.route.world_size;
      const int32_t m_group = along_n
          ? rank_work / head_chunks
          : rank_work % m_groups;
      const int32_t head_chunk = along_n
          ? rank_work % head_chunks
          : rank_work / m_groups;
      const int32_t head_slot = head_chunk / chunks_per_head;
      const int32_t chunk = head_chunk - head_slot * chunks_per_head;
      const int32_t segment = head_slot < q_local_heads
          ? 0
          : (head_slot < q_local_heads + kv_local_heads ? 1 : 2);
      const int32_t local_head = segment == 0
          ? head_slot
          : head_slot - q_local_heads -
              (segment == 2 ? kv_local_heads : 0);
      const int32_t head_offset = chunk * CopyBlockN;
      const int32_t copy_n =
          min(CopyBlockN, p.route.head_dim - head_offset);
      const int32_t physical_feature = source_feature(
          p.route, destination_rank, segment, local_head, head_offset);
      if (wait_for_gemm) {
        const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
        const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
        const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
        const int32_t first_producer_n = physical_feature / BlockN;
        const int32_t last_producer_n =
            (physical_feature + copy_n - 1) / BlockN;
        const int32_t producer_n_tiles =
            last_producer_n - first_producer_n + 1;
        const int32_t waits = MTilesPerTask * producer_n_tiles;
        for (int32_t item = warp; item < waits; item += warps) {
          const int32_t local_m_tile = item / producer_n_tiles;
          const int32_t tile_m =
              m_group * MTilesPerTask + local_m_tile;
          if (tile_m < m_tiles) {
            const int32_t producer_n =
                first_producer_n + item % producer_n_tiles;
            const int32_t signal =
                tile_m * output_n_tiles + producer_n;
            detail::wait_acquire_system(
                p.ready + signal * kReadyFlagStride,
                p.epoch,
                lane);
          }
        }
      }
      __syncthreads();

      const int32_t m_begin = m_group * MTilesPerTask * BlockM;
      const int32_t copy_m =
          min(BlockM * MTilesPerTask, p.gemm.m - m_begin);
      const int32_t vectors_per_row = copy_n / kAlignment;
      const int32_t vector_count = copy_m * vectors_per_row;
      const auto* source = reinterpret_cast<const uint4*>(
          p.local_output);
      auto* destination =
          reinterpret_cast<uint4*>(p.peer_output[destination_rank]);
      const int32_t segment_width =
          segment == 0 ? q_local_width : kv_local_width;
      const int32_t local_segment_base =
          segment == 0 ? 0 : q_local_width +
              (segment == 2 ? kv_local_width : 0);
      const int64_t segment_offset = segment == 0
          ? 0
          : segment_rows * q_local_width +
              (segment == 2 ? segment_rows * kv_local_width : 0);
      const int32_t local_feature =
          local_segment_base + local_head * p.route.head_dim + head_offset;
      for (int32_t index = threadIdx.x; index < vector_count; index += blockDim.x) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_n = index - row * vectors_per_row;
        const int32_t source_row = m_begin + row;
        const int64_t src =
            static_cast<int64_t>(source_row) * source_row_vectors +
            physical_feature / kAlignment + vector_n;
        const int32_t batch = source_row / p.route.seq_local;
        const int32_t local_seq = source_row - batch * p.route.seq_local;
        const int64_t destination_row =
            static_cast<int64_t>(batch) * p.route.global_seq +
            p.route.rank * p.route.seq_local + local_seq;
        const int64_t dst = p.route.defer_v_a2a
            ? (destination_row *
                   (q_local_width + kv_local_width) +
               local_feature) /
                    kAlignment +
                vector_n
            : (segment_offset +
               destination_row * segment_width +
               local_feature - local_segment_base) /
                    kAlignment +
                vector_n;
        destination[dst] = source[src];
      }
    }

    // The downstream A2A->GEMM consumes V but not the routed Q/K buffer. A
    // single communication CTA waits only for V's epilogue tiles and publishes
    // their transitive completion after the Q/K route has drained.
    if (wait_for_gemm && p.route.defer_v_a2a && p.completion_epoch &&
        comm_id == 0) {
      wait_and_publish_completion(args);
    }

  }

  CUTLASS_DEVICE static void wait_and_publish_completion(const Params& args) {
    const auto& p = args.params;
    if (!p.completion_epoch) {
      return;
    }
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t v_begin =
        (p.route.q_heads + p.route.kv_heads) * p.route.head_dim;
    const int32_t v_first_tile = v_begin / BlockN;
    const int32_t v_n_tiles = output_n_tiles - v_first_tile;
    const int32_t v_signals = m_tiles * v_n_tiles;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
    for (int32_t item = warp; item < v_signals; item += warps) {
      const int32_t tile_m = item / v_n_tiles;
      const int32_t tile_n = v_first_tile + item - tile_m * v_n_tiles;
      const int32_t signal = tile_m * output_n_tiles + tile_n;
      detail::wait_acquire_system(
          p.ready + signal * kReadyFlagStride, p.epoch, lane);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      detail::store_release_system(p.completion_epoch, p.epoch);
    }
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas) {
    run(args, comm_id, comm_ctas, true);
  }

  // All local compute and communication CTAs have crossed the cooperative
  // grid barrier before this hook runs, so every remote store issued by this
  // source rank is complete. Publish one source-complete epoch into every
  // destination-owned array, then hold this rank's kernel open until all
  // sources have completed its routed output.
  CUTLASS_DEVICE void finalize(const Params& args) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
      return;
    }
    const auto& p = args.params;
    asm volatile("fence.sc.sys;\n" ::: "memory");
    for (int32_t destination_rank = 0;
         destination_rank < p.route.world_size;
         ++destination_rank) {
      detail::store_release_system(
          p.peer_route_done_epoch[destination_rank] +
              p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    const uint32_t* local_sources_done =
        p.peer_route_done_epoch[p.route.rank];
    for (int32_t source_rank = 0;
         source_rank < p.route.world_size;
         ++source_rank) {
#pragma unroll 1
      while (detail::load_acquire_system(
                 local_sources_done + source_rank * kReadyFlagStride) <
             p.epoch) {
        __nanosleep(64);
      }
    }
  }
};

using QkvGqaPackCommWide = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(ProjectionTileShape{})),
    static_cast<int32_t>(cute::size<1>(ProjectionTileShape{}))>;
using QkvGqaPackCommSmall = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    4>;
using QkvGqaPackCommSmallInterleaved = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    4,
    true>;
// Keep the GEMM producer on DeepGEMM's fastest wide-N policies while routing
// Q/K in head-aligned 128-column chunks.  Producer and copy tile boundaries
// are intentionally independent; a copy waits for every producer tile it
// overlaps before reading the output.
using DeepGemmQkvComm176 =
    QkvGqaPackCommT<GemmA2AParams, false, 128, 176, 4, false, 128>;
using DeepGemmQkvComm176Interleaved =
    QkvGqaPackCommT<GemmA2AParams, false, 128, 176, 4, true, 128>;
using DeepGemmQkvComm192 =
    QkvGqaPackCommT<GemmA2AParams, false, 128, 192, 4, false, 128>;
using DeepGemmQkvComm192Interleaved =
    QkvGqaPackCommT<GemmA2AParams, false, 128, 192, 4, true, 128>;
using Fp8QkvGqaPackComm = QkvGqaPackCommT<
    Fp8GemmA2AParams,
    true,
    static_cast<int32_t>(cute::size<0>(Fp8TileShape{})),
    static_cast<int32_t>(cute::size<1>(Fp8TileShape{}))>;

using QkvGemmA2AKernelWide =
    detail::MonolithicGemm<ProjectionOutputGemm, QkvGqaPackCommWide>;
using QkvGemmA2AKernelSmall =
    detail::MonolithicGemm<OutputGemm, QkvGqaPackCommSmall>;
using Fp8GemmA2AKernel =
    detail::MonolithicGemm<Fp8OutputGemm, Fp8QkvGqaPackComm>;

static_assert(
    sizeof(typename QkvGemmA2AKernelWide::SharedStorage) >=
        QkvGqaPackCommWide::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelSmall::SharedStorage) >=
        QkvGqaPackCommSmall::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8GemmA2AKernel::SharedStorage) >=
        Fp8QkvGqaPackComm::SharedStorageBytes);

template <class Comm>
__global__ __launch_bounds__(384)
void a2a_lhs_input_copy_reference_kernel(
    CUTLASS_GRID_CONSTANT const typename Comm::Params params) {
  Comm{}(
      params, static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(gridDim.x));
}

template <class Comm>
__global__ __launch_bounds__(384) void qkv_gqa_copy_reference_kernel(
    CUTLASS_GRID_CONSTANT const typename Comm::Params params) {
  Comm{}.run(
      params,
      static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(gridDim.x),
      false);
}

template <class Comm>
cudaError_t launch_qkv_gqa_copy_reference(
    typename Comm::Arguments& args,
    int32_t comm_ctas,
    cudaStream_t stream) {
  cudaError_t status = Comm::initialize(args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!Comm::can_implement(args)) {
    return cudaErrorNotSupported;
  }
  auto entry = qkv_gqa_copy_reference_kernel<Comm>;
  status = cudaFuncSetAttribute(
      entry,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(Comm::SharedStorageBytes));
  if (status != cudaSuccess) {
    return status;
  }
  entry<<<comm_ctas, 384, Comm::SharedStorageBytes, stream>>>(
      Comm::to_underlying_arguments(args));
  return cudaGetLastError();
}

cudaError_t device_sm_count(int32_t* count, int32_t* device) {
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceGetAttribute(count, cudaDevAttrMultiProcessorCount, *device);
}

template <class Kernel>
cudaError_t launch_regular(const typename Kernel::Params& params, cudaStream_t stream) {
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
  if constexpr (cluster_size == 1) {
    entry<<<grid, Kernel::get_block_shape(), smem_bytes, stream>>>(params);
    return cudaGetLastError();
  } else {
    if (grid.x % cluster_x != 0 || grid.y % cluster_y != 0 ||
        grid.z % cluster_z != 0) {
      return cudaErrorInvalidConfiguration;
    }
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {cluster_x, cluster_y, cluster_z};
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = Kernel::get_block_shape();
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, entry, params);
  }
}

template <class Kernel>
cudaError_t launch_gemm_reference_impl(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas,
    RasterOptions fallback) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.lhs;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.local_output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order = raster_option(params.gemm.raster, fallback);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream);
}

template <class Kernel>
cudaError_t launch_a2a_lhs_reference_impl(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.input_staging;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongN);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream);
}

template <
    class InputGemm,
    class Kernel,
    class Comm = A2ALhsInputComm
#if FUSE_ENABLE_PROFILING
    ,
    bool Instrumented = false>
#else
    >
#endif
cudaError_t launch_a2a_lhs_gemm_policy(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0,
    A2AGemmPeerTimeline* peer_timeline = nullptr,
    int32_t peer_timeline_capacity = 0
#endif
    ) {
  typename Comm::Arguments comm_args{};
  comm_args.params = params;
  cudaError_t status = Comm::initialize(comm_args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!Comm::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }

  constexpr int32_t tile_m = static_cast<int32_t>(
      cute::size<0>(typename InputGemm::TileShape{}));
  constexpr int32_t tile_n = static_cast<int32_t>(
      cute::size<1>(typename InputGemm::TileShape{}));
  constexpr int32_t tile_k = static_cast<int32_t>(
      cute::size<2>(typename InputGemm::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  static_assert(Comm::kReadyBlockM % tile_m == 0);
  constexpr int32_t m_tiles_per_ready = Comm::kReadyBlockM / tile_m;
  const int32_t ready_m_tiles = ceil_div(params.gemm.m, Comm::kReadyBlockM);
  const int32_t compute_m_frontier =
      ceil_div(sm_count - params.num_comm_ctas, n_tiles);
  comm_args.m_window = min(
      ready_m_tiles,
      max(1, ceil_div(compute_m_frontier, m_tiles_per_ready)));
  const int32_t k_per_peer = params.gemm.k / params.route.world_size;
  if (params.gemm.k % params.route.world_size != 0 ||
      k_per_peer % tile_k != 0) {
    return cudaErrorNotSupported;
  }

  typename Kernel::Arguments args{};
  args.num_comm_ctas = params.num_comm_ctas;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.comm = comm_args;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.comm.peer_timeline = peer_timeline;
    args.comm.peer_timeline_capacity = peer_timeline_capacity;
  }
#endif
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.gemm.mainloop.ptr_A = params.input_staging;
  args.gemm.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  args.gemm.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.gemm.mainloop.ready = params.ready;
  args.gemm.mainloop.world_size = params.route.world_size;
  args.gemm.mainloop.m_tiles = m_tiles;
  args.gemm.mainloop.arrivals_per_peer =
      Comm::arrivals_per_peer(comm_args);
  args.gemm.mainloop.k_tiles_per_peer = k_per_peer / tile_k;
  args.gemm.mainloop.epoch = params.epoch;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.gemm.mainloop.timeline = timeline;
    args.gemm.mainloop.timeline_capacity = timeline_capacity;
    args.gemm.mainloop.peer_timeline = peer_timeline;
    args.gemm.mainloop.peer_timeline_capacity = peer_timeline_capacity;
    args.gemm.mainloop.n_tiles = n_tiles;
  }
#endif
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ptr_D = params.output;
  args.gemm.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongN);
  if (!Kernel::can_implement(args)) {
    return cudaErrorNotSupported;
  }
  if (Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  return detail::launch_cooperative<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr),
      stream,
      sm_count);
}

struct LhsPolicyCandidate {
  A2ALhsGemmPolicy policy;
  int32_t tile_m;
  int32_t tile_n;
  int32_t cluster_m;
};

constexpr std::array<LhsPolicyCandidate, 5> kLhsPolicyCandidates{{
    {A2ALhsGemmPolicy::kM64N128, 64, 128, 1},
    {A2ALhsGemmPolicy::kM128N128, 128, 128, 1},
    {A2ALhsGemmPolicy::kM128N160, 128, 160, 1},
    {A2ALhsGemmPolicy::kM128N256ClusterM2, 128, 256, 2},
    {A2ALhsGemmPolicy::kM128N320ClusterM2, 128, 320, 2},
}};

constexpr LhsPolicyCandidate kWideN320ManualCandidate{
    A2ALhsGemmPolicy::kM128N320ClusterM2, 128, 320, 2};

A2ALhsPolicyInfo score_a2a_lhs_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    const LhsPolicyCandidate& candidate) {
  A2ALhsPolicyInfo info{};
  info.policy = candidate.policy;
  info.tile_m = candidate.tile_m;
  info.tile_n = candidate.tile_n;
  info.tile_k = 64;
  info.cluster_m = candidate.cluster_m;
  info.compute_ctas = sm_count - num_comm_ctas;
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 || problem.l <= 0 ||
      info.compute_ctas <= 0 ||
      (candidate.cluster_m > 1 &&
       (num_comm_ctas % candidate.cluster_m != 0 ||
        info.compute_ctas % candidate.cluster_m != 0))) {
    info.estimated_cycles = std::numeric_limits<double>::infinity();
    return info;
  }
  info.compute_clusters = info.compute_ctas / candidate.cluster_m;

  const int64_t m_tiles = ceil_div(problem.m, candidate.tile_m);
  const int64_t n_tiles = ceil_div(problem.n, candidate.tile_n);
  const int64_t m_cluster_tiles = ceil_div(m_tiles, candidate.cluster_m);
  info.n_tiles = static_cast<int32_t>(n_tiles);
  info.tile_count = m_tiles * n_tiles * problem.l;
  info.cluster_tile_count = m_cluster_tiles * n_tiles * problem.l;
  info.waves = static_cast<int32_t>(
      ceil_div(
          info.cluster_tile_count,
          static_cast<int64_t>(info.compute_clusters)));
  info.last_wave_clusters = static_cast<int32_t>(
      info.cluster_tile_count - static_cast<int64_t>(info.waves - 1) *
          info.compute_clusters);
  info.last_wave_ctas = info.last_wave_clusters * candidate.cluster_m;
  // A cluster is the indivisible scheduler work unit.  A wave boundary is
  // frontier-aligned only when it contains an integer number of complete
  // N-frontiers, or one frontier occupies an integer number of whole waves.
  // This prevents an already-published M frontier from being split across
  // two persistent-CTA waves merely because its N fan-out does not divide the
  // resident cluster budget.
  info.frontier_aligned =
      info.waves == 1 || info.compute_clusters % n_tiles == 0 ||
      n_tiles % info.compute_clusters == 0;
  info.full_last_wave =
      info.cluster_tile_count % info.compute_clusters == 0;

  // DeepGEMM-style SM90 wave model.  Tensor-core work and HBM bytes are
  // invariant across candidates; this compares the variable L1/L2 traffic
  // after accounting for cluster-B multicast and partial-wave occupancy.
  constexpr double kL2BytesPerCycle = 8.0e6 / 1.3e3;
  const double l2_bandwidth = std::min(
      64.0 * info.compute_ctas, kL2BytesPerCycle);
  const double l1_bandwidth = 128.0 * info.compute_ctas;
  constexpr double element_bytes = sizeof(Element);
  const double l2_bytes_per_tile =
      static_cast<double>(problem.k) *
          (candidate.tile_m +
           static_cast<double>(candidate.tile_n) / candidate.cluster_m) *
          element_bytes +
      static_cast<double>(candidate.tile_m) * candidate.tile_n *
          element_bytes;
  const double l1_bytes_per_tile =
      static_cast<double>(problem.k) *
          (candidate.tile_m + candidate.tile_n) * element_bytes +
      static_cast<double>(problem.k) *
          (std::max(64, candidate.tile_m) + candidate.tile_n) *
          element_bytes +
      2.0 * candidate.tile_m * candidate.tile_n * element_bytes;
  const double wave_efficiency =
      static_cast<double>(info.cluster_tile_count) /
      (static_cast<double>(info.waves) * info.compute_clusters);
  const double l2_cycles =
      l2_bytes_per_tile * info.tile_count / l2_bandwidth;
  const double l1_cycles =
      l1_bytes_per_tile * info.tile_count / l1_bandwidth;
  info.estimated_cycles =
      std::max(l1_cycles, l2_cycles) / wave_efficiency;
  return info;
}

A2ALhsPolicyInfo select_a2a_lhs_policy_impl(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    A2ALhsGemmPolicy requested) {
  A2ALhsPolicyInfo best{};
  best.estimated_cycles = std::numeric_limits<double>::infinity();
  if (requested == A2ALhsGemmPolicy::kM128N320ClusterM2) {
    return score_a2a_lhs_policy(
        problem, num_comm_ctas, sm_count, kWideN320ManualCandidate);
  }
  for (const auto& candidate : kLhsPolicyCandidates) {
    if (requested != A2ALhsGemmPolicy::kAuto &&
        requested != candidate.policy) {
      continue;
    }
    const auto current = score_a2a_lhs_policy(
        problem, num_comm_ctas, sm_count, candidate);
    if (requested != A2ALhsGemmPolicy::kAuto) {
      return current;
    }
    // A two-CTA multicast cluster makes a one-wave input consumer advance at
    // the slower ready tile in each pair. Independent CTAs are stronger when
    // the whole GEMM already fits in one wave.
    if (candidate.cluster_m > 1 && current.waves == 1) {
      continue;
    }
    // N320 is admitted automatically only when it fixes a scheduling
    // geometry problem: aligning an M frontier or improving a partial wave.
    // If an unaligned candidate already ends on a full wave, keep the mature
    // N256 family instead of treating larger N as a generic GEMM tuning knob.
    if (candidate.policy == A2ALhsGemmPolicy::kM128N320ClusterM2 &&
        !current.frontier_aligned && current.full_last_wave) {
      continue;
    }
    const bool better_frontier =
        current.frontier_aligned > best.frontier_aligned;
    const bool equal_frontier =
        current.frontier_aligned == best.frontier_aligned;
    // Frontier splitting directly delays ready-data consumption and is a
    // structural scheduling hazard.  A partial final wave is softer: its cost
    // is already represented by wave_efficiency, so never choose an otherwise
    // weak GEMM tile solely to make the last wave full.
    if (better_frontier ||
        (equal_frontier &&
         current.estimated_cycles < best.estimated_cycles)) {
      best = current;
    }
  }
  return best;
}

cudaError_t launch_a2a_lhs_gemm_impl(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  switch (selected.policy) {
    case A2ALhsGemmPolicy::kM64N128:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsM64Gemm, A2ALhsM64GemmKernel, A2ALhsM64InputComm>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N128:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsInputGemm, A2ALhsGemmKernel, A2ALhsInputComm>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N160:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsN160Gemm, A2ALhsN160GemmKernel>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsProjectionGemm, A2ALhsProjectionGemmKernel>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsWideN320Gemm, A2ALhsWideN320GemmKernel>(
              params, stream, sm_count, device);
    default:
      return cudaErrorNotSupported;
  }
}

template <class GemmKernel, class Comm, class Params>
cudaError_t launch_gemm_a2a_impl(
    const Params& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  using Kernel = detail::MonolithicGemm<GemmKernel, Comm>;
  constexpr int32_t tile_m =
      static_cast<int32_t>(cute::size<0>(typename GemmKernel::TileShape{}));
  constexpr int32_t tile_n =
      static_cast<int32_t>(cute::size<1>(typename GemmKernel::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  typename Kernel::Arguments args{};
  args.num_comm_ctas = params.num_comm_ctas;
  args.comm.params = params;
  cudaError_t status = Comm::initialize(args.comm);
  if (status != cudaSuccess) {
    return status;
  }
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.gemm.mainloop.ptr_A = params.lhs;
  args.gemm.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  args.gemm.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ptr_D = params.local_output;
  args.gemm.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ready = params.ready;
  args.gemm.epilogue.m_tiles = m_tiles;
  args.gemm.epilogue.n_tiles = n_tiles;
  args.gemm.epilogue.epoch = params.epoch;
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  const auto kernel_params = Kernel::to_underlying_arguments(args, nullptr);
  return detail::launch_cooperative<Kernel>(kernel_params, stream, sm_count);
}

cudaError_t make_deepgemm_tma_2d(
    CUtensorMap* tensor_map,
    const void* pointer,
    uint64_t global_inner,
    uint64_t global_outer,
    uint32_t box_inner,
    uint32_t box_outer,
    uint64_t outer_stride_elements,
    CUtensorMapSwizzle swizzle) {
  const uint64_t global_dims[2] = {global_inner, global_outer};
  const uint64_t global_strides[1] = {
      outer_stride_elements * sizeof(Element)};
  const uint32_t box_dims[2] = {box_inner, box_outer};
  constexpr uint32_t element_strides[2] = {1, 1};
  const CUresult result =
      CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          tensor_map,
          CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          2,
          const_cast<void*>(pointer),
          global_dims,
          global_strides,
          box_dims,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

template <
    int32_t BlockN,
    int32_t NumStages,
    int32_t SwizzleD,
    bool MulticastA,
    int32_t NumComputeCtas>
cudaError_t launch_deepgemm_reference_config(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  constexpr int32_t block_m = 128;
  constexpr int32_t block_k = 64;
  constexpr int32_t threads = 384;
  constexpr size_t smem_d =
      (block_m * BlockN * sizeof(Element) + 1023) / 1024 * 1024;
  constexpr size_t smem_per_stage =
      (block_m + BlockN) * block_k * sizeof(Element);
  // DeepGEMM reserves all sixteen full/empty barriers in its host policy.
  constexpr size_t smem_bytes = smem_d + 16 * 8 * 2 +
      NumStages * smem_per_stage;
  constexpr int32_t d_box_inner = SwizzleD / sizeof(Element);

  CUtensorMap tensor_map_a{};
  CUtensorMap tensor_map_b{};
  CUtensorMap tensor_map_d{};
  cudaError_t status = make_deepgemm_tma_2d(
      &tensor_map_a,
      params.lhs,
      params.gemm.k,
      params.gemm.m,
      128 / sizeof(Element),
      block_m,
      a_row_stride(params.gemm),
      CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }
  status = make_deepgemm_tma_2d(
      &tensor_map_b,
      params.rhs_nt,
      params.gemm.k,
      params.gemm.n,
      128 / sizeof(Element),
      BlockN,
      b_row_stride(params.gemm),
      CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }
  status = make_deepgemm_tma_2d(
      &tensor_map_d,
      params.local_output,
      params.gemm.n,
      params.gemm.m,
      d_box_inner,
      block_m,
      d_row_stride(params.gemm),
      SwizzleD == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                      : (SwizzleD == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                        : CU_TENSOR_MAP_SWIZZLE_32B));
  if (status != cudaSuccess) {
    return status;
  }

  auto entry = deep_gemm::sm90_bf16_gemm_impl<
      cute::UMMA::Major::K,
      cute::UMMA::Major::K,
      0,
      10240,
      8192,
      1,
      block_m,
      BlockN,
      block_k,
      128,
      128,
      SwizzleD,
      NumStages,
      128,
      256,
      2,
      MulticastA,
      NumComputeCtas,
      deep_gemm::GemmType::Normal,
      false,
      cutlass::bfloat16_t>;
  status = cudaFuncSetAttribute(
      entry,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (status != cudaSuccess) {
    return status;
  }

  cudaLaunchAttribute cluster_attribute{};
  cluster_attribute.id = cudaLaunchAttributeClusterDimension;
  cluster_attribute.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(NumComputeCtas, 1, 1);
  config.blockDim = dim3(threads, 1, 1);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = &cluster_attribute;
  config.numAttrs = 1;
  const uint32_t m = params.gemm.m;
  const uint32_t n = params.gemm.n;
  const uint32_t k = params.gemm.k;
  return cudaLaunchKernelEx(
      &config,
      entry,
      static_cast<int*>(nullptr),
      m,
      n,
      k,
      tensor_map_a,
      tensor_map_b,
      tensor_map_d);
}

template <
    class CommOp,
    int32_t BlockN,
    int32_t NumStages,
    int32_t SwizzleD,
    bool MulticastA,
    int32_t NumComputeCtas,
    int32_t NumCommCtas>
cudaError_t launch_deepgemm_fused_config(
    const GemmA2AParams& params,
    cudaStream_t stream,
    bool enable_comm) {
  static_assert(NumComputeCtas + NumCommCtas == 132);
  constexpr int32_t block_m = 128;
  constexpr int32_t block_k = 64;
  constexpr int32_t threads = 384;
  constexpr size_t smem_d =
      (block_m * BlockN * sizeof(Element) + 1023) / 1024 * 1024;
  constexpr size_t smem_per_stage =
      (block_m + BlockN) * block_k * sizeof(Element);
  constexpr size_t smem_bytes = smem_d + 16 * 8 * 2 +
      NumStages * smem_per_stage;
  constexpr int32_t d_box_inner = SwizzleD / sizeof(Element);

  typename CommOp::Arguments comm_args{};
  comm_args.params = params;
  cudaError_t status = CommOp::initialize(comm_args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!CommOp::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }

  CUtensorMap tensor_map_a{};
  CUtensorMap tensor_map_b{};
  CUtensorMap tensor_map_d{};
  status = make_deepgemm_tma_2d(
      &tensor_map_a,
      params.lhs,
      params.gemm.k,
      params.gemm.m,
      128 / sizeof(Element),
      block_m,
      a_row_stride(params.gemm),
      CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }
  status = make_deepgemm_tma_2d(
      &tensor_map_b,
      params.rhs_nt,
      params.gemm.k,
      params.gemm.n,
      128 / sizeof(Element),
      BlockN,
      b_row_stride(params.gemm),
      CU_TENSOR_MAP_SWIZZLE_128B);
  if (status != cudaSuccess) {
    return status;
  }
  status = make_deepgemm_tma_2d(
      &tensor_map_d,
      params.local_output,
      params.gemm.n,
      params.gemm.m,
      d_box_inner,
      block_m,
      d_row_stride(params.gemm),
      SwizzleD == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                      : (SwizzleD == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                                        : CU_TENSOR_MAP_SWIZZLE_32B));
  if (status != cudaSuccess) {
    return status;
  }

  auto entry = deep_gemm::sm90_bf16_gemm_fused_impl<
      cute::UMMA::Major::K,
      cute::UMMA::Major::K,
      0,
      10240,
      8192,
      1,
      block_m,
      BlockN,
      block_k,
      128,
      128,
      SwizzleD,
      NumStages,
      128,
      256,
      2,
      MulticastA,
      NumComputeCtas,
      deep_gemm::GemmType::Normal,
      false,
      cutlass::bfloat16_t,
      CommOp>;
  status = cudaFuncSetAttribute(
      entry,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_bytes));
  if (status != cudaSuccess) {
    return status;
  }

  cudaLaunchAttribute cluster_attribute{};
  cluster_attribute.id = cudaLaunchAttributeClusterDimension;
  cluster_attribute.val.clusterDim = {2, 1, 1};
  cudaLaunchConfig_t occupancy_config{};
  occupancy_config.gridDim = dim3(2, 1, 1);
  occupancy_config.blockDim = dim3(threads, 1, 1);
  occupancy_config.dynamicSmemBytes = smem_bytes;
  occupancy_config.stream = stream;
  occupancy_config.attrs = &cluster_attribute;
  occupancy_config.numAttrs = 1;
  int32_t active_clusters = 0;
  status = cudaOccupancyMaxActiveClusters(
      &active_clusters, entry, &occupancy_config);
  if (status != cudaSuccess) {
    return status;
  }
  if (active_clusters < 66) {
    return cudaErrorCooperativeLaunchTooLarge;
  }

  cudaLaunchAttribute attributes[2]{};
  attributes[0] = cluster_attribute;
  attributes[1].id = cudaLaunchAttributeCooperative;
  attributes[1].val.cooperative = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(132, 1, 1);
  config.blockDim = dim3(threads, 1, 1);
  config.dynamicSmemBytes = smem_bytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 2;
  const uint32_t m = params.gemm.m;
  const uint32_t n = params.gemm.n;
  const uint32_t k = params.gemm.k;
  return cudaLaunchKernelEx(
      &config,
      entry,
      comm_args,
      static_cast<uint32_t>(NumCommCtas),
      enable_comm,
      m,
      n,
      k,
      tensor_map_a,
      tensor_map_b,
      tensor_map_d);
}

template <bool PeerInterleaved>
cudaError_t launch_deepgemm_decoupled(
    const GemmA2AParams& params,
    cudaStream_t stream,
    bool enable_comm) {
  switch (params.num_comm_ctas) {
    case 8:
      return launch_deepgemm_fused_config<
          std::conditional_t<
              PeerInterleaved,
              DeepGemmQkvComm176Interleaved,
              DeepGemmQkvComm176>,
          176, 4, 32, false, 124, 8>(
              params, stream, enable_comm);
    case 12:
      return launch_deepgemm_fused_config<
          std::conditional_t<
              PeerInterleaved,
              DeepGemmQkvComm176Interleaved,
              DeepGemmQkvComm176>,
          176, 4, 32, true, 120, 12>(
              params, stream, enable_comm);
    case 16:
      return launch_deepgemm_fused_config<
          std::conditional_t<
              PeerInterleaved,
              DeepGemmQkvComm192Interleaved,
              DeepGemmQkvComm192>,
          192, 4, 128, true, 116, 16>(
              params, stream, enable_comm);
    case 20:
      return launch_deepgemm_fused_config<
          std::conditional_t<
              PeerInterleaved,
              DeepGemmQkvComm192Interleaved,
              DeepGemmQkvComm192>,
          192, 4, 128, true, 112, 20>(
              params, stream, enable_comm);
    case 24:
      return launch_deepgemm_fused_config<
          std::conditional_t<
              PeerInterleaved,
              DeepGemmQkvComm192Interleaved,
              DeepGemmQkvComm192>,
          192, 4, 128, true, 108, 24>(
              params, stream, enable_comm);
    default:
      return cudaErrorNotSupported;
  }
}

}  // namespace

KernelTraits cutlass_kernel_traits() {
  return {
      kBlockM,
      kBlockN,
      64,
      static_cast<int32_t>(OutputGemm::get_block_shape().x),
      static_cast<int32_t>(sizeof(typename A2ALhsGemmKernel::SharedStorage))};
}

KernelTraits projection_cutlass_kernel_traits() {
  return {
      static_cast<int32_t>(cute::size<0>(ProjectionTileShape{})),
      static_cast<int32_t>(cute::size<1>(ProjectionTileShape{})),
      static_cast<int32_t>(cute::size<2>(ProjectionTileShape{})),
      static_cast<int32_t>(ProjectionOutputGemm::get_block_shape().x),
      static_cast<int32_t>(sizeof(typename QkvGemmA2AKernelWide::SharedStorage))};
}

KernelTraits qkv_cutlass_kernel_traits(const GemmProblem& problem) {
  if (use_wide_qkv_policy(problem)) {
    return projection_cutlass_kernel_traits();
  }
  return cutlass_kernel_traits();
}

KernelTraits fp8_cutlass_kernel_traits() {
  return {
      kBlockM,
      kBlockN,
      128,
      static_cast<int32_t>(Fp8OutputGemm::get_block_shape().x),
      static_cast<int32_t>(sizeof(typename Fp8GemmA2AKernel::SharedStorage))};
}

int64_t a2a_lhs_gemm_ready_elements(
    const GemmProblem& problem,
    const UlyssesRoute& route) {
  if (problem.m <= 0 || problem.l != 1 || route.world_size <= 0 ||
      route.world_size > kMaxWorldSize || route.batch <= 0 ||
      route.seq_local <= 0 || problem.m != route.batch * route.seq_local) {
    return 0;
  }
  // Reserve for the finest production policy.  Coarser M128 policies use a
  // prefix of the same allocation.
  return static_cast<int64_t>(ceil_div(problem.m, 64)) *
      route.world_size * kReadyFlagStride;
}

double a2a_lhs_nvlink_bidirectional_gbps(int32_t device) {
  if (const char* value = std::getenv("FUSE_NVLINK_BIDIR_GBPS")) {
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end != value && parsed > 0.0) {
      return parsed;
    }
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
      std::strstr(properties.name, "H800") != nullptr) {
    return 400.0;
  }
  return 900.0;
}

double a2a_lhs_compute_time_us(
    const A2ALhsPolicyInfo& policy,
    int32_t world_size) {
  // H200 calibration of
  //   Tcompute = launch + policy_scale * estimated_cycles / rate(waves).
  // The four constants are a nonlinear least-squares fit to the standalone
  // compute-subgrid measurements from the CP4 3-width x 6-sequence sweep.
  // rate is in model-cycles/us; it decays from the short one-wave rate to the
  // steady persistent-grid rate as the number of waves grows.
  constexpr double kRateSteady = 1542.6;
  constexpr double kRateFirstWave = 1739.1;
  constexpr double kWaveDecay = 10.56;
  constexpr double kLaunchUs = 5.80;
  double policy_scale = 1.0;
  switch (policy.policy) {
    case A2ALhsGemmPolicy::kM64N128:
      policy_scale = 1.205;
      break;
    case A2ALhsGemmPolicy::kM128N160:
      policy_scale = 1.019;
      break;
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      policy_scale = 1.410;
      break;
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      policy_scale = 1.478;
      break;
    default:
      break;
  }
  const double wave_rate = kRateSteady +
      (kRateFirstWave - kRateSteady) *
          std::exp(-static_cast<double>(policy.waves - 1) / kWaveDecay);
  const double cp_scale = std::pow(
      std::max(1.0, static_cast<double>(world_size) / 4.0), 0.4883);
  return (kLaunchUs + policy.estimated_cycles * policy_scale / wave_rate) *
      cp_scale;
}

double a2a_lhs_route_time_us(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const A2ALhsPolicyInfo& policy,
    int32_t comm_ctas,
    double nvlink_bidirectional_gbps) {
  // H200 calibration of
  //   Troute = launch + payload/B(comm) + rows/(comm*row_rate)
  //            + task_waves/task_rate.
  // These constants are a log-error least-squares fit to the matching CP4
  // standalone route sweep.  B(comm)=Bmax*(1-exp(-comm/cta_scale)); the
  // explicit row and task terms retain the effects of shard width, ReadyBlockM
  // and the four bulk-TMA issuers in each communication CTA.
  constexpr double kLaunchUs = 9.20;
  constexpr double kCopyBmaxGbs = 528.8;
  constexpr double kCopyCtaScale = 4.961;
  constexpr double kStoreRowsPerUs = 41912.0;
  constexpr double kTasksPerUs = 0.3874;
  const int32_t world_size = route.world_size;
  const int32_t row_bytes = problem.k / world_size * sizeof(Element);
  const int32_t max_rows = row_bytes > 0
      ? kA2ALhsBulkStageBytes / row_bytes
      : 0;
  const int32_t comm_rows = max_rows > 0
      ? std::min(policy.tile_m, max_rows)
      : kA2ALhsCommRows;
  const int64_t chunks_per_tile = ceil_div(policy.tile_m, comm_rows);
  const int64_t tasks =
      ceil_div(problem.m, policy.tile_m) * world_size * chunks_per_tile;
  const int64_t task_waves =
      ceil_div(tasks, static_cast<int64_t>(comm_ctas * kA2ALhsBulkSlots));

  const double issuer_bandwidth = kCopyBmaxGbs *
      (1.0 - std::exp(-static_cast<double>(comm_ctas) / kCopyCtaScale));
  const double remote_fraction =
      static_cast<double>(world_size - 1) / world_size;
  const double fabric_bandwidth = remote_fraction > 0.0
      ? 0.5 * nvlink_bidirectional_gbps / remote_fraction
      : kCopyBmaxGbs;
  const double bandwidth = std::min(issuer_bandwidth, fabric_bandwidth);
  const double payload_bytes =
      static_cast<double>(problem.m) * problem.k * sizeof(Element);
  const double copy_us = payload_bytes / bandwidth / 1000.0;
  const double store_us =
      static_cast<double>(problem.m) * world_size /
      (comm_ctas * kStoreRowsPerUs);
  const double task_us = task_waves / kTasksPerUs;
  const double cp_scale = std::pow(
      std::max(1.0, static_cast<double>(world_size) / 4.0), 0.2022);
  return (kLaunchUs + copy_us + store_us + task_us) * cp_scale;
}

int32_t experimental_a2a_lhs_gemm_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t sm_count,
    int32_t device) {
  constexpr std::array<int32_t, 10> kCommCandidates{
      2, 4, 6, 8, 10, 12, 14, 16, 20, 24};
  // Candidate score:
  //   compute + route_weight*max(0, route-hidden_fraction*compute)
  //           + contention_weight*compute*(comm_ctas/sm_count)^power.
  // The four dimensionless coefficients minimize selection regret over the
  // complete CP4/CP8 3-width x 6-sequence sweeps; CP4 10+50 finalists carry
  // twice the quick-sweep weight.  Shape enters only through the compute and
  // route models above: there is no fitted sequence-length threshold.
  constexpr double kExposedRouteWeight = 0.584;
  constexpr double kHiddenRouteFraction = 0.365;
  constexpr double kCommContentionWeight = 7.84;
  constexpr double kCommContentionPower = 1.737;
  const double nvlink_gbps = a2a_lhs_nvlink_bidirectional_gbps(device);
  int32_t best_comm_ctas = 4;
  double best_score = std::numeric_limits<double>::infinity();
  for (const int32_t comm_ctas : kCommCandidates) {
    if (comm_ctas >= sm_count) {
      continue;
    }
    const auto policy = select_a2a_lhs_policy_impl(
        problem, comm_ctas, sm_count, A2ALhsGemmPolicy::kAuto);
    if (!std::isfinite(policy.estimated_cycles)) {
      continue;
    }
    const double compute_us =
        a2a_lhs_compute_time_us(policy, route.world_size);
    const double route_us = a2a_lhs_route_time_us(
        problem, route, policy, comm_ctas, nvlink_gbps);
    const double exposed_route =
        std::max(0.0, route_us - kHiddenRouteFraction * compute_us);
    const double comm_fraction =
        static_cast<double>(comm_ctas) / sm_count;
    const double score = compute_us +
        kExposedRouteWeight * exposed_route +
        kCommContentionWeight * compute_us *
            std::pow(comm_fraction, kCommContentionPower);
    if (score < best_score) {
      best_score = score;
      best_comm_ctas = comm_ctas;
    }
  }
  return best_comm_ctas;
}

int32_t default_a2a_lhs_gemm_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t device) {
  // The default deliberately uses the small, auditable rule established by
  // the long-sequence sweep.  Short/medium M keeps the mature comm4 policy.
  // For M>=32768, BF16 remote bytes / GEMM FLOPs simplifies to
  //   (world-1) / (world*N).
  // On the CP4 H200 long-sequence matrix, threshold 1.65e-4 selects comm8 for
  // N=4096 and comm6 for N=5120/7168 (8/9 oracle matches; <=0.022% worst loss).
  // Scaling by 900/device_bandwidth preserves the same balance on H800.
  if (problem.m < 32768 || route.world_size <= 1 || problem.n <= 0) {
    return 4;
  }
  constexpr double kH200NvlinkGbs = 900.0;
  constexpr double kHighPressure = 1.65e-4;
  const double remote_bytes_per_flop =
      static_cast<double>(route.world_size - 1) /
      (static_cast<double>(route.world_size) * problem.n);
  const double normalized_pressure = remote_bytes_per_flop *
      kH200NvlinkGbs / a2a_lhs_nvlink_bidirectional_gbps(device);
  return normalized_pressure >= kHighPressure ? 8 : 6;
}

int32_t recommended_a2a_lhs_gemm_comm_ctas_impl(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t sm_count,
    int32_t device) {
  // The fitted whole-range model is retained for experiments only.  Its
  // cross-shape generality has not been established well enough for the
  // default ABI; explicit comm_ctas always overrides both policies.
  const char* policy = std::getenv("FUSE_A2A_LHS_COMM_POLICY");
  if (policy != nullptr && std::strcmp(policy, "experimental_model") == 0) {
    return experimental_a2a_lhs_gemm_comm_ctas(
        problem, route, sm_count, device);
  }
  return default_a2a_lhs_gemm_comm_ctas(problem, route, device);
}

int32_t recommended_a2a_lhs_gemm_comm_ctas(
    const GemmProblem& problem,
    const UlyssesRoute& route) {
  int32_t sm_count = 0;
  int32_t device = 0;
  if (device_sm_count(&sm_count, &device) != cudaSuccess) {
    return 4;
  }
  return recommended_a2a_lhs_gemm_comm_ctas_impl(
      problem, route, sm_count, device);
}

A2ALhsPolicyInfo select_a2a_lhs_gemm_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    A2ALhsGemmPolicy requested) {
  return select_a2a_lhs_policy_impl(
      problem, num_comm_ctas, sm_count, requested);
}

int32_t recommended_gemm_a2a_comm_ctas(
  const GemmProblem& problem,
  const UlyssesRoute& route) {
  if (route.kind != RouteKind::kQkvGqaPack) {
    return 0;
  }
  const int64_t output_bytes =
      static_cast<int64_t>(problem.m) * problem.n * sizeof(Bf16);
  if (problem.input_dtype == DType::kFloat8E4M3 &&
      output_bytes >= 64ll * 1024 * 1024) {
    return 40;
  }
  if (output_bytes >= 32ll * 1024 * 1024) {
    return 32;
  }
  return problem.n >= 4096 ? 24 : 16;
}

cudaError_t launch_a2a_gemm_cutlass(const A2AGemmParams& params, cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  A2AGemmParams launch_params = params;
  if (launch_params.num_comm_ctas == 0) {
    launch_params.num_comm_ctas = recommended_a2a_lhs_gemm_comm_ctas_impl(
        params.gemm, params.route, sm_count, device);
  }
  if (launch_params.num_comm_ctas <= 0 || launch_params.num_comm_ctas >= sm_count) {
    return cudaErrorInvalidValue;
  }

  return launch_a2a_lhs_gemm_impl(
      launch_params, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_a2a_gemm_cutlass_role_telemetry(
    const A2AGemmParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    A2AGemmPeerTimeline* peer_timeline,
    int32_t peer_timeline_capacity,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  A2AGemmParams launch_params = params;
  if (launch_params.num_comm_ctas == 0) {
    launch_params.num_comm_ctas = recommended_a2a_lhs_gemm_comm_ctas_impl(
        params.gemm, params.route, sm_count, device);
  }
  if (launch_params.num_comm_ctas <= 0 ||
      launch_params.num_comm_ctas >= sm_count || timeline == nullptr ||
      timeline_capacity < sm_count) {
    return cudaErrorInvalidValue;
  }
  const auto selected = select_a2a_lhs_policy_impl(
      launch_params.gemm,
      launch_params.num_comm_ctas,
      sm_count,
      launch_params.lhs_policy);
  if (selected.policy == A2ALhsGemmPolicy::kM64N128) {
    return launch_a2a_lhs_gemm_policy<
        A2ALhsM64TelemetryGemm,
        A2ALhsM64TelemetryKernel,
        A2ALhsM64TelemetryInputComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  }
  if (selected.policy == A2ALhsGemmPolicy::kM128N256ClusterM2) {
    return launch_a2a_lhs_gemm_policy<
        A2ALhsProjectionTelemetryGemm,
        A2ALhsProjectionTelemetryKernel,
        A2ALhsTelemetryInputComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  }
  if (selected.policy == A2ALhsGemmPolicy::kM128N320ClusterM2) {
    return launch_a2a_lhs_gemm_policy<
        A2ALhsWideN320TelemetryGemm,
        A2ALhsWideN320TelemetryKernel,
        A2ALhsTelemetryInputComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  }
  if (selected.policy != A2ALhsGemmPolicy::kM128N128) {
    return cudaErrorNotSupported;
  }
  return launch_a2a_lhs_gemm_policy<
      A2ALhsTelemetryGemm,
      A2ALhsTelemetryKernel,
      A2ALhsTelemetryInputComm,
      true>(
          launch_params,
          stream,
          sm_count,
          device,
          timeline,
          timeline_capacity,
          peer_timeline,
          peer_timeline_capacity);
}

cudaError_t query_a2a_gemm_role_resources(A2AGemmRoleResources* resources) {
  if (resources == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaFuncAttributes production{};
  cudaFuncAttributes instrumented{};
  cudaError_t status = cudaFuncGetAttributes(
      &production, cutlass::device_kernel<A2ALhsGemmKernel>);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaFuncGetAttributes(
      &instrumented, cutlass::device_kernel<A2ALhsTelemetryKernel>);
  if (status != cudaSuccess) {
    return status;
  }
  constexpr int32_t cluster_ctas =
      cute::size<0>(typename A2ALhsGemmKernel::ClusterShape{}) *
      cute::size<1>(typename A2ALhsGemmKernel::ClusterShape{}) *
      cute::size<2>(typename A2ALhsGemmKernel::ClusterShape{});
  *resources = {
      static_cast<int32_t>(A2ALhsGemmKernel::get_block_shape().x),
      production.numRegs,
      instrumented.numRegs,
      static_cast<int32_t>(production.sharedSizeBytes),
      static_cast<int32_t>(sizeof(typename A2ALhsGemmKernel::SharedStorage)),
      cluster_ctas,
      kA2ALhsBulkSlots,
      static_cast<int32_t>(A2ALhsGemmKernel::get_block_shape().x / 32),
      static_cast<int32_t>(
          kA2ALhsBulkSlots *
          (kA2ALhsBulkStageBytes + sizeof(uint64_t)))};
  return cudaSuccess;
}
#endif

cudaError_t launch_gemm_a2a_cutlass(const GemmA2AParams& params, cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  GemmA2AParams launch_params = params;
  if (launch_params.num_comm_ctas == 0) {
    launch_params.num_comm_ctas =
        recommended_gemm_a2a_comm_ctas(params.gemm, params.route);
  }
  if (launch_params.num_comm_ctas <= 0 || launch_params.num_comm_ctas >= sm_count) {
    return cudaErrorInvalidValue;
  }

  if (launch_params.route.kind == RouteKind::kQkvGqaPack &&
      launch_params.route.direction == RouteDirection::kForward) {
    if (use_wide_qkv_policy(launch_params.gemm)) {
      return launch_gemm_a2a_impl<ProjectionOutputGemm, QkvGqaPackCommWide>(
          launch_params, stream, sm_count, device);
    }
    if (launch_params.route.qkv_peer_interleaved) {
      return launch_gemm_a2a_impl<
          OutputGemm, QkvGqaPackCommSmallInterleaved>(
              launch_params, stream, sm_count, device);
    }
    return launch_gemm_a2a_impl<OutputGemm, QkvGqaPackCommSmall>(
        launch_params, stream, sm_count, device);
  }
  return cudaErrorNotSupported;
}

cudaError_t launch_gemm_a2a_deepgemm(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const bool packed =
      a_row_stride(params.gemm) == params.gemm.k &&
      b_row_stride(params.gemm) == params.gemm.k &&
      d_row_stride(params.gemm) == params.gemm.n;
  if (sm_count != 132 || !packed || params.gemm.m != 512 ||
      params.gemm.n != 10240 || params.gemm.k != 8192 ||
      params.gemm.l != 1 || params.alpha != 1.0f ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorNotSupported;
  }
  return params.route.qkv_peer_interleaved
      ? launch_deepgemm_decoupled<true>(params, stream, true)
      : launch_deepgemm_decoupled<false>(params, stream, true);
}

cudaError_t launch_gemm_a2a_deepgemm_compute_only(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  return params.route.qkv_peer_interleaved
      ? launch_deepgemm_decoupled<true>(params, stream, false)
      : launch_deepgemm_decoupled<false>(params, stream, false);
}

cudaError_t launch_gemm_a2a_fp8_cutlass(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  Fp8GemmA2AParams launch_params = params;
  if (launch_params.num_comm_ctas == 0) {
    launch_params.num_comm_ctas =
        recommended_gemm_a2a_comm_ctas(params.gemm, params.route);
  }
  if (launch_params.num_comm_ctas <= 0 || launch_params.num_comm_ctas >= sm_count) {
    return cudaErrorInvalidValue;
  }
  if (launch_params.route.kind != RouteKind::kQkvGqaPack ||
      launch_params.route.direction != RouteDirection::kForward) {
    return cudaErrorNotSupported;
  }
  Fp8QkvGqaPackComm::Arguments comm_args{};
  comm_args.params = launch_params;
  if (!Fp8QkvGqaPackComm::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }
  return launch_gemm_a2a_impl<Fp8OutputGemm, Fp8QkvGqaPackComm>(
      launch_params, stream, sm_count, device);
}

cudaError_t launch_batched_cutlass_reference(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }

  if (use_wide_qkv_policy(params.gemm)) {
    return launch_gemm_reference_impl<ProjectionPureGemm>(
        params, stream, sm_count, device, reserved_comm_ctas,
        RasterOptions::AlongN);
  }
  return launch_gemm_reference_impl<PureGemm>(
      params, stream, sm_count, device, reserved_comm_ctas,
      RasterOptions::AlongN);
}

cudaError_t launch_deepgemm_bf16_reference(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const bool packed =
      a_row_stride(params.gemm) == params.gemm.k &&
      b_row_stride(params.gemm) == params.gemm.k &&
      d_row_stride(params.gemm) == params.gemm.n;
  if (!params.lhs || !params.rhs_nt || !params.local_output ||
      params.gemm.m != 512 || params.gemm.n != 10240 ||
      params.gemm.k != 8192 || params.gemm.l != 1 || !packed ||
      params.alpha != 1.0f || sm_count != 132) {
    return cudaErrorNotSupported;
  }
  if (reserved_comm_ctas == 0) {
    return launch_deepgemm_reference_config<160, 5, 64, false, 132>(
        params, stream);
  }
  if (reserved_comm_ctas == 4) {
    return launch_deepgemm_reference_config<160, 5, 64, false, 128>(
        params, stream);
  }
  if (reserved_comm_ctas == 8) {
    return launch_deepgemm_reference_config<176, 4, 32, false, 124>(
        params, stream);
  }
  if (reserved_comm_ctas == 12) {
    return launch_deepgemm_reference_config<176, 4, 32, true, 120>(
        params, stream);
  }
  if (reserved_comm_ctas == 16) {
    return launch_deepgemm_reference_config<192, 4, 128, true, 116>(
        params, stream);
  }
  if (reserved_comm_ctas == 20) {
    return launch_deepgemm_reference_config<192, 4, 128, true, 112>(
        params, stream);
  }
  if (reserved_comm_ctas == 24) {
    return launch_deepgemm_reference_config<192, 4, 128, true, 108>(
        params, stream);
  }
  return cudaErrorNotSupported;
}

cudaError_t launch_a2a_gemm_cutlass_reference(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.input_staging || !params.rhs_nt || !params.output ||
      !supported_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }

  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  switch (selected.policy) {
    case A2ALhsGemmPolicy::kM64N128:
      return launch_a2a_lhs_reference_impl<M64PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N128:
      return launch_a2a_lhs_reference_impl<PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N160:
      return launch_a2a_lhs_reference_impl<N160PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      if (reserved_comm_ctas % 2 != 0 ||
          (sm_count - reserved_comm_ctas) % 2 != 0) {
        return cudaErrorInvalidValue;
      }
      return launch_a2a_lhs_reference_impl<ProjectionPureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      if (reserved_comm_ctas % 2 != 0 ||
          (sm_count - reserved_comm_ctas) % 2 != 0) {
        return cudaErrorInvalidValue;
      }
      return launch_a2a_lhs_reference_impl<WideN320PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    default:
      return cudaErrorNotSupported;
  }
}

cudaError_t launch_dense_fp8_cutlass_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.lhs || !params.rhs_nt || !params.local_output ||
      !supported_fp8_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }

  typename Fp8PureGemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.lhs;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.local_output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  if (!Fp8PureGemm::can_implement(args) || Fp8PureGemm::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Fp8PureGemm::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Fp8PureGemm>(
      Fp8PureGemm::to_underlying_arguments(args, nullptr), stream);
}

template <class Comm>
cudaError_t launch_a2a_lhs_copy_reference_impl(
    const A2AGemmParams& params, cudaStream_t stream) {
    typename Comm::Arguments args{};
    args.params = params;
    cudaError_t status = Comm::initialize(args);
    if (status != cudaSuccess) {
      return status;
    }
    if (params.num_comm_ctas <= 0 ||
        !Comm::can_implement(args)) {
      return cudaErrorNotSupported;
    }
    constexpr size_t smem_bytes =
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
        kA2ALhsBulkSlots * sizeof(uint64_t);
    auto entry = a2a_lhs_input_copy_reference_kernel<Comm>;
    status = cudaFuncSetAttribute(
        entry,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (status != cudaSuccess) {
      return status;
    }
    entry<<<
        params.num_comm_ctas, 384, smem_bytes, stream>>>(
            Comm::to_underlying_arguments(args));
    return cudaGetLastError();
}

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  if (selected.policy == A2ALhsGemmPolicy::kM64N128) {
    return launch_a2a_lhs_copy_reference_impl<A2ALhsM64InputComm>(
        params, stream);
  }
  return launch_a2a_lhs_copy_reference_impl<A2ALhsInputComm>(
      params, stream);
}

cudaError_t launch_gemm_a2a_copy_reference(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  if (use_wide_qkv_policy(params.gemm)) {
    QkvGqaPackCommWide::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<QkvGqaPackCommWide>(
        args, params.num_comm_ctas, stream);
  }
  if (params.route.qkv_peer_interleaved) {
    QkvGqaPackCommSmallInterleaved::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<QkvGqaPackCommSmallInterleaved>(
        args, params.num_comm_ctas, stream);
  }
  QkvGqaPackCommSmall::Arguments args{};
  args.params = params;
  return launch_qkv_gqa_copy_reference<QkvGqaPackCommSmall>(
      args, params.num_comm_ctas, stream);
}

cudaError_t launch_gemm_a2a_deepgemm_copy_reference(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  if (params.route.qkv_peer_interleaved) {
    DeepGemmQkvComm192Interleaved::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<DeepGemmQkvComm192Interleaved>(
        args, params.num_comm_ctas, stream);
  } else {
    DeepGemmQkvComm192::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<DeepGemmQkvComm192>(
        args, params.num_comm_ctas, stream);
  }
}

cudaError_t launch_gemm_a2a_fp8_copy_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  Fp8QkvGqaPackComm::Arguments args{};
  args.params = params;
  return launch_qkv_gqa_copy_reference<Fp8QkvGqaPackComm>(
      args, params.num_comm_ctas, stream);
}

}  // namespace fuse
