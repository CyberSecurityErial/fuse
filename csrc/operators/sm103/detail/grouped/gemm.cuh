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
template <int TileN = 128, int TileK = 64, bool SwapAB = false, bool TrimTokens = true>
struct Bf16GroupedGemmTypes {
  static_assert(!SwapAB || TileN == 128, "swapAB pilot uses one M128 UMMA atom");
  using Tile = cute::Shape<cute::_128, cute::Int<TileN>, cute::Int<TileK>>;
  using Cluster = cute::Shape<cute::_1,cute::_1,cute::_1>;
  using OutputLayout = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor*, cutlass::layout::RowMajor*>;
  using Problem = std::conditional_t<SwapAB, SwappedGroupedProblemShape, GroupedProblemShape>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      void, OutputLayout, 8, Bf16, OutputLayout, 8,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
  using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp,
      Bf16, cutlass::layout::RowMajor*, 8, Bf16, cutlass::layout::ColumnMajor*, 8,
      float, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epi::SharedStorage)>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>::CollectiveOp;
  using Mainloop = std::conditional_t<SwapAB, GroupedSwapABMainloop<BaseMainloop,TrimTokens>, BaseMainloop>;
  using PureGemm = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epi>;
  using DispatchGemm = cutlass::gemm::kernel::GemmUniversal<Problem,
      GroupedInputReadyMainloop<Mainloop,SwapAB>, GroupedInputReleaseEpilogue<Epi,SwapAB>>;
  using CombineGemm = cutlass::gemm::kernel::GemmUniversal<Problem,
      Mainloop, GroupedSignalingEpilogue<Epi,SwapAB>>;
};

}  // namespace fuse::detail
