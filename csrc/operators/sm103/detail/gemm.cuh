// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include "fuse/layout/gemm.h"

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>

namespace fuse {
namespace {

constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 64;
constexpr int kAlignment = 16 / sizeof(Bf16);

using Element = Bf16;
using Accumulator = float;
// Sm100 is the minimum architecture of this BF16 tcgen05/TMEM collective.
// The translation unit must be compiled for sm_103a on the target device.
using ArchTag = cutlass::arch::Sm100;
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ProblemShape = cute::Shape<int, int, int, int>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

CUTLASS_HOST_DEVICE constexpr int32_t ceil_div(int32_t value, int32_t divisor) {
  return value / divisor + (value % divisor != 0);
}

__host__ __device__ constexpr int64_t a_row_stride(const GemmProblem& p) {
  return p.stride_a.row < 0 ? p.k : p.stride_a.row;
}

__host__ __device__ constexpr int64_t b_row_stride(const GemmProblem& p) {
  return p.stride_b.row < 0 ? p.k : p.stride_b.row;
}

__host__ __device__ constexpr int64_t d_row_stride(const GemmProblem& p) {
  return p.stride_d.row < 0 ? p.n : p.stride_d.row;
}

inline bool supported_problem(const GemmProblem& p) {
  // Batched Ulysses rows are flattened into M. There is one dense weight
  // matrix, not an independently-strided L dimension in this BF16 version.
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l == 1 &&
      !p.transpose_a && !p.transpose_b &&
      p.input_dtype == DType::kBfloat16 && p.weight_dtype == DType::kBfloat16 &&
      p.output_dtype == DType::kBfloat16 &&
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1) &&
      a_row_stride(p) >= p.k && a_row_stride(p) % kAlignment == 0 &&
      b_row_stride(p) >= p.k && b_row_stride(p) % kAlignment == 0 &&
      d_row_stride(p) >= p.n && d_row_stride(p) % kAlignment == 0 &&
      (p.max_swizzle_size == 1 || p.max_swizzle_size == 2 ||
       p.max_swizzle_size == 4 || p.max_swizzle_size == 8) &&
      (p.raster == GemmRaster::kHeuristic || p.raster == GemmRaster::kAlongM ||
       p.raster == GemmRaster::kAlongN);
}

inline auto raster_option(GemmRaster value, GemmRaster fallback) {
  using Raster = detail::PersistentTileSchedulerSm100Monolithic::RasterOrderOptions;
  return (value == GemmRaster::kHeuristic ? fallback : value) == GemmRaster::kAlongM
      ? Raster::AlongM : Raster::AlongN;
}

// TODO: These Hopper tile widths are starting points; tune them on B300 after
// correctness and resource validation, keeping communication SMs explicit.
// TODO: Calibrate B300 compute costs for the required performance-model tuner;
// do not reuse Hopper's measured wave tables or treat these defaults as optimal.
template <int BlockN, int BlockK = 64, int EpilogueN = 0, bool SwapAB = false>
struct Bf16GemmTypes {
  // Precision-specific family, not a generic BF16-versus-FP8 branch. A future
  // block-scaled family needs its actual operand/scale contract and CUTLASS
  // collective; changing Element or alpha alone would not implement it.
  using Element = Bf16;
  using Accumulator = float;
  // Both forward APIs compute D = alpha * A * B; neither accepts a C source.
  using ElementC = void;
  static constexpr int kAlignment = 16 / sizeof(Element);
  static_assert((SwapAB && BlockN == 32) || BlockN == 64 || BlockN == 128 || BlockN == 160 ||
                BlockN == 192 || BlockN == 256, "Unsupported BF16 tile width");
  static_assert(!SwapAB || (BlockN == 32 && BlockK == 64 && EpilogueN == 0),
                "The row-owned input plan uses physical M128/N32/K64/Auto");
  static_assert(BlockK == 64 || BlockK == 128, "Unsupported BF16 K tile");
  static_assert(EpilogueN == 0 || EpilogueN == 32 || EpilogueN == 64,
                "Epilogue N is Auto (0), 32 or 64");
  static_assert(EpilogueN != 64 || (BlockN == 256 && BlockK == 64),
                "E64 is the explicit N256/K64 no-residual candidate");
  // Retain the registered K128 pairings; no new policy is implied by the
  // no-residual epilogue's lower shared-memory requirement.
  static_assert(BlockK == 64 || BlockN == 128 ||
                (BlockN == 256 && EpilogueN == 32),
                "K128 supports N128/Auto or N256/epilogue-N32");
  static constexpr int kTileM = 128;
  static constexpr int kTileN = BlockN;
  static constexpr int kTileK = BlockK;
  // One-SM MMA makes this the physical producer/ready tile, not a cluster
  // aggregate. Launch bindings must derive communication and flags from it.
  using TileShape = cute::Shape<cute::Int<kTileM>, cute::Int<kTileN>, cute::Int<kTileK>>;

  // Without C, BF16 Auto targets a 128x32 epilogue subtile for this family.
  // E64 changes only the N256/K64 epilogue subtile, not its producer/ready
  // tile. Auto and the explicit N32 bindings retain their previous choices.
  using EpilogueTile = cute::conditional_t<
      EpilogueN == 64, cute::Shape<cute::_128, cute::_64>,
      cute::conditional_t<EpilogueN == 32 || BlockN == 160,
          cute::Shape<cute::_128, cute::_32>,
          cutlass::epilogue::collective::EpilogueTileAuto>>;
  // SwapAB computes D^T = W^T A^T from the existing [N,K]/[M,K] buffers.
  // A column-major D^T view writes the caller's row-major D directly: no
  // staging transpose or post-GEMM transpose kernel is part of this plan.
  using OutputLayout = cute::conditional_t<SwapAB, cutlass::layout::ColumnMajor, LayoutD>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      TileShape, ClusterShape, EpilogueTile,
      Accumulator, Accumulator,
      ElementC, OutputLayout, kAlignment,
      Element, OutputLayout, kAlignment,
      cutlass::epilogue::TmaWarpSpecialized1Sm>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassTensorOp,
      Element, LayoutA, kAlignment,
      Element, LayoutB, kAlignment,
      Accumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecialized1SmSm100>::CollectiveOp;
  static_assert(BlockN != 256 || BlockK != 64 ||
                Mainloop::DispatchPolicy::Stages == 4,
                "N256/K64 no-residual BF16 must retain four A/B stages");

  // Independent compute calibration uses the same BF16 collective and
  // scheduler, without a ready-wait mainloop or ready-publishing epilogue.
  using PureGemm = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, Mainloop, Epilogue, detail::MonolithicPersistentScheduler>;
  // Only QKV publishes output tiles. The swapped input plan must not bind
  // that publisher to its physical N32 tile or expose it as QKV geometry.
  using OutputGemm = cute::conditional_t<SwapAB, PureGemm,
      cutlass::gemm::kernel::GemmUniversal<
          ProblemShape, Mainloop, detail::SignalingEpilogue<Epilogue, TileShape>,
          detail::MonolithicPersistentScheduler>>;
  static_assert(cute::size(typename Mainloop::AtomThrShapeMNK{}) == 1,
                "Communication and compute require independent one-SM CTAs.");
  static_assert(OutputGemm::MaxThreadsPerBlock == 256,
                "Communication and compute must share a 256-thread CTA.");
  static_assert(PureGemm::MaxThreadsPerBlock == OutputGemm::MaxThreadsPerBlock);
  static_assert(EpilogueN != 64 ||
                (Epilogue::DispatchPolicy::StagesD == 2 &&
                 OutputGemm::AccumulatorPipelineStageCount == 2 &&
                 PureGemm::AccumulatorPipelineStageCount == 2),
                "E64 must retain two D buffers and two TMEM accumulator stages");
};

#if FUSE_ENABLE_PROFILING
// One explicitly bounded probe, not a new production policy or a second
// all-policy diagnostic grid. Its collective geometry matches N256/K64/e32.
using QkvEpilogueProbeTypes = Bf16GemmTypes<256, 64, 32>;
using QkvEpilogueProbeGemm = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, typename QkvEpilogueProbeTypes::Mainloop,
    detail::QkvEpilogueProbe<typename QkvEpilogueProbeTypes::Epilogue,
                             typename QkvEpilogueProbeTypes::TileShape>,
    detail::MonolithicPersistentScheduler>;
#endif

using N64TileShape = typename Bf16GemmTypes<64>::TileShape;
using N160TileShape = typename Bf16GemmTypes<160>::TileShape;
using N192TileShape = typename Bf16GemmTypes<192>::TileShape;
using ProjectionTileShape = typename Bf16GemmTypes<256>::TileShape;
using N64OutputGemm = typename Bf16GemmTypes<64>::OutputGemm;
using OutputGemm = typename Bf16GemmTypes<128>::OutputGemm;
using N160OutputGemm = typename Bf16GemmTypes<160>::OutputGemm;
using N192OutputGemm = typename Bf16GemmTypes<192>::OutputGemm;
using ProjectionOutputGemm = typename Bf16GemmTypes<256>::OutputGemm;

// Inverse A2A uses the same BF16 collective family, but consumes ready lhs
// shards and stores ordinary GEMM output: no QKV signaling epilogue here.
template <int BlockN, int BlockK = 64, int EpilogueN = 0, bool SwapAB = false>
struct A2ALhsGemmTypes {
  static_assert((SwapAB && BlockN == 32) || (!SwapAB && (BlockN == 128 || BlockN == 256)));
  using Dense = Bf16GemmTypes<BlockN, BlockK, EpilogueN, SwapAB>;
  using TileShape = typename Dense::TileShape;
  using PureGemm = typename Dense::PureGemm;
  using Mainloop = detail::A2ALhsReadyMainloop<typename Dense::Mainloop, TileShape, false, SwapAB>;
  using Gemm = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, Mainloop, typename Dense::Epilogue,
      detail::MonolithicPersistentScheduler>;

#if FUSE_ENABLE_PROFILING
  using TelemetryMainloop =
      detail::A2ALhsReadyMainloop<typename Dense::Mainloop, TileShape, true, SwapAB>;
  using TelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, TelemetryMainloop, typename Dense::Epilogue,
      detail::MonolithicPersistentScheduler>;
#endif
  static_assert(Gemm::MaxThreadsPerBlock == 256);
};

}  // namespace
}  // namespace fuse
