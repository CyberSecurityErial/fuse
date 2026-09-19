// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// BF16 grouped collective configurations, not scheduling or transport policy.
#include "cutlass_pipeline.cuh"
#include "persistent_gemm.cuh"
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>

namespace fuse::detail {

// Keep the stock Blackwell UMMA/TMEM mainloop and TMA epilogue. Only grouped
// traversal and the two ready boundaries are adapted; no copied MMA loop.
template <int TileN = 128, int TileK = 64, bool SwapAB = false,
    bool TrimTokens = true, int SmMode = 1>
struct Bf16GroupedGemmTypes {
  static_assert(SmMode == 1 || SmMode == 2);
  static_assert(SmMode == 1 || !SwapAB,
      "the swapAB pilot has not been adapted to two-SM MMA");
  static_assert(!SwapAB || TileN == 128, "swapAB pilot uses one M128 UMMA atom");
  using Tile = cute::Shape<cute::Int<128 * SmMode>, cute::Int<TileN>, cute::Int<TileK>>;
  using Cluster = cute::Shape<cute::Int<SmMode>,cute::_1,cute::_1>;
  using MainloopSchedule = std::conditional_t<SmMode == 1,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>;
  using EpilogueSchedule = std::conditional_t<SmMode == 1,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>;
  using OutputLayout = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor*, cutlass::layout::RowMajor*>;
  using Problem = std::conditional_t<SwapAB, SwappedGroupedProblemShape, GroupedProblemShape>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, OutputLayout, 8, Bf16, OutputLayout, 8, EpilogueSchedule>::CollectiveOp;
  using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Bf16, cutlass::layout::RowMajor*, 8, Bf16, cutlass::layout::ColumnMajor*, 8,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
      MainloopSchedule>::CollectiveOp;
  using Mainloop = std::conditional_t<SwapAB, GroupedSwapABMainloop<BaseMainloop,TrimTokens>, BaseMainloop>;
  using PureGemm = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epi>;
  using DispatchGemm = cutlass::gemm::kernel::GemmUniversal<Problem,
      GroupedInputReadyMainloop<Mainloop,SwapAB>, GroupedInputReleaseEpilogue<Epi,SwapAB>>;
  using CombineGemm = cutlass::gemm::kernel::GemmUniversal<Problem,
      Mainloop, GroupedSignalingEpilogue<Epi,SwapAB>>;
};

// Long-token Dispatch candidate: stock CUTLASS grouped scheduling and
// Blackwell two-CTA UMMA. Two physical 128-row CTAs share one logical 256-row
// input-ready panel. Communication CTAs later retire into this full-SM GEMM.
template <int TileN = 128, int TileK = 64>
struct Bf16GroupedRetiringGemmTypes {
  using Tile = cute::Shape<cute::_256, cute::Int<TileN>, cute::Int<TileK>>;
  using Cluster = cute::Shape<cute::_2,cute::_1,cute::_1>;
  using Problem = cutlass::gemm::GroupProblemShape<
      cute::Shape<int32_t,int32_t,int32_t>>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, cutlass::layout::RowMajor*, 8, Bf16, cutlass::layout::RowMajor*, 8,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;
  using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Bf16, cutlass::layout::RowMajor*, 8, Bf16, cutlass::layout::ColumnMajor*, 8,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmSm100>::CollectiveOp;
  using DispatchGemm = cutlass::gemm::kernel::GemmUniversal<Problem,
      GroupedInputReadyMainloop<BaseMainloop,false,2>,
      GroupedInputReleaseEpilogue<Epi>>;
  // The plan's common type plumbing names both directions before the
  // Dispatch-only static assertion is applied. This alias is never launched.
  using CombineGemm = DispatchGemm;
};

}  // namespace fuse::detail
