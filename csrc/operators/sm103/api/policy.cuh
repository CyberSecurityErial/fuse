// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "../detail/autotune.cuh"
#include <type_traits>

namespace fuse {
namespace {

// Only the communication budget is automatic. The caller's GEMM collective,
// raster and effective swizzle remain fixed: every candidate's SM split then
// drives the same first-use windows/cohorts used by the production copy queue.
// Shape-only queries deliberately omit pointer/epoch validation; the existing
// launchers still validate those resources after this common resolution step.
inline cudaError_t resolve_oproj_communication(
    const A2AGemmParams& params, A2AGemmParams* resolved) {
  if (!resolved || params.num_comm_ctas < 0) return cudaErrorInvalidValue;
  *resolved = params;
  if (params.num_comm_ctas > 0) return cudaSuccess;

  const auto& p = params.gemm;
  const auto& r = params.route;
  // Both calibrated collectives use 128-row ready units and 64-wide K tiles.
  // Each causal half must contain whole ready units to select the bulk queue.
  if (!supported_problem(p) || params.lhs_policy != A2ALhsGemmPolicy::kAuto ||
      a_row_stride(p) != p.k || b_row_stride(p) != p.k || d_row_stride(p) != p.n ||
      (r.world_size != 4 && r.world_size != 8) ||
      r.rank < 0 || r.rank >= r.world_size || r.batch <= 0 ||
      r.seq_local <= 0 || r.global_seq <= 0 || r.q_heads <= 0 ||
      r.local_heads <= 0 || r.head_dim <= 0 || r.head_dim % kAlignment != 0 ||
      r.kind != RouteKind::kHeadToSequence || r.direction != RouteDirection::kInverse ||
      r.channel_count != 1 || !r.causal_load_balanced || r.cyclic_peer_order ||
      r.qkv_peer_interleaved || r.defer_v_a2a || r.packed_source_row ||
      r.packed_row_granularity != 0 || r.seq_local % 256 != 0 ||
      int64_t{r.seq_local} * r.world_size != r.global_seq ||
      int64_t{r.batch} * r.seq_local != p.m ||
      int64_t{r.local_heads} * r.world_size != r.q_heads ||
      int64_t{r.q_heads} * r.head_dim != p.k ||
      (int64_t{r.local_heads} * r.head_dim) % 64 != 0) {
    return cudaErrorNotSupported;
  }
  // Match A2ALhsInputCommT::initialize: unset/empty/rows means row pull.
  // Column delivery has no measured curves. Preserve input_epoch and its
  // existing acquire: calibration starts with all upstream inputs published,
  // so C/R curves do not estimate an unfinished upstream producer's delay.
  const char* layout = std::getenv("FUSE_SM103_OPROJ_COMM_LAYOUT");
  if (layout && *layout && std::strcmp(layout, "rows") != 0) {
    return std::strcmp(layout, "columns") == 0
        ? cudaErrorNotSupported : cudaErrorInvalidValue;
  }
  OprojGemmPolicy policy{};
  cudaError_t status = select_oproj_gemm_policy(&policy);
  if (status != cudaSuccess) return status;
  const int policy_index = policy == OprojGemmPolicy::kM128N256 ? 0 :
      policy == OprojGemmPolicy::kM128N256K64E32 ? 1 : -1;
  if (policy_index < 0) return cudaErrorNotSupported;

  // Reject out-of-model work before CUTLASS's scheduler lowering, whose tile
  // products use int32 arithmetic. Bounding unpadded work to the core's 1M
  // limit also bounds swizzle padding (width <= 8) far below INT32_MAX; the
  // core applies its exact padded-work limit after the real scheduler resolves
  // the width. No descriptors or potentially overflowing tile products yet.
  const int64_t m_tiles = (int64_t{p.m} + 127) / 128;
  const int64_t n_tiles = (int64_t{p.n} + 255) / 256;
  if (p.k < 8192 || p.k > 16384 || m_tiles * n_tiles > 1000000) {
    return cudaErrorNotSupported;
  }

  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  if (info.sm_count != 148) return cudaErrorNotSupported;
  auto select = [&](auto binding_tag) {
    using Gemm = typename decltype(binding_tag)::type::Gemm;
    // This lowers scheduler geometry only, without encoding TMA descriptors.
    // Zero here obtains the unchanged raster/swizzle, not a measured 148-SM
    // cost: the selector uses each candidate's own budget-matched C/R curves.
    const auto args = gemm_arguments<Gemm>(
        p, params.input_staging, params.rhs_nt, params.output,
        params.alpha, 0, info, GemmRaster::kAlongN);
    const auto order = a2a_input_order<Gemm>(args);
    detail::OProjTuningRequest request;
    request.m = p.m; request.n = p.n; request.k = p.k;
    request.world = r.world_size; request.sm_count = info.sm_count;
    request.device = info.device; request.policy_index = policy_index;
    request.raster = order.along_n ? 1 : 0;
    request.max_swizzle_size = p.max_swizzle_size;
    request.swizzle = 1 << order.log_swizzle;
    const auto result = detail::select_oproj_plan_cached(request);
    if (result.status != detail::OProjTuningStatus::Success) {
      return result.status == detail::OProjTuningStatus::InvalidInput
          ? cudaErrorInvalidValue : cudaErrorNotSupported;
    }
    resolved->num_comm_ctas = result.comm_ctas;
    return cudaSuccess;
  };
  return visit_oproj_forward_policy(policy, select);
}

// Match the BF16 resolver contract: a positive budget is a strict override;
// zero selects a HOST plan before any descriptor construction or GPU launch.
// GEMM owns the layout. The selected budget changes both j + wave*C compute
// ownership and w + step*(8*c) quantization/copy ownership, so the scorer must
// rebuild those chains together. No tile policy, ready granularity or device
// work queue is changed here.
inline cudaError_t resolve_mxfp8_qkv_communication(
    const Mxfp8GemmA2AParams& params, Mxfp8GemmA2AParams* resolved) {
  if (!resolved || params.projection.num_comm_ctas < 0) return cudaErrorInvalidValue;
  *resolved = params;
  if (params.projection.num_comm_ctas > 0) return cudaSuccess;

  const auto& p = params.projection.gemm;
  const auto& r = params.projection.route;
  // First calibration domain: packed activation/native K32 scales, contiguous
  // BF16 master W, full Q/K/V bulk routing, ordinary eight-warp side work.
  // Unsupported layouts remain usable with explicit budgets where the kernel
  // supports them; they must not silently inherit these service measurements.
  if (!supported_mxfp8_problem(p) || p.m % 128 || p.n % 256 ||
      (params.epilogue_n != 32 && params.epilogue_n != 64) ||
      params.weight_preparation != Mxfp8WeightPreparation::kCommunicationCtas ||
      a_row_stride(p) != p.k || b_row_stride(p) != p.k || d_row_stride(p) != p.n ||
      (r.world_size != 4 && r.world_size != 8) ||
      r.rank < 0 || r.rank >= r.world_size || r.batch <= 0 ||
      r.seq_local <= 0 || r.seq_local % 64 || r.global_seq <= 0 ||
      r.q_heads <= 0 || r.kv_heads <= 0 || r.head_dim != 128 ||
      r.q_heads % r.world_size || r.kv_heads % r.world_size ||
      r.q_heads % r.kv_heads || r.channel_count != 1 ||
      r.kind != RouteKind::kQkvGqaPack || r.direction != RouteDirection::kForward ||
      r.qkv_peer_interleaved || r.defer_v_a2a || r.causal_load_balanced ||
      r.cyclic_peer_order || r.packed_source_row || r.packed_row_granularity != 0 ||
      int64_t{r.seq_local} * r.world_size != r.global_seq ||
      int64_t{r.batch} * r.seq_local != p.m ||
      int64_t{r.batch} * r.global_seq > INT32_MAX ||
      (int64_t{r.q_heads} + 2 * int64_t{r.kv_heads}) * r.head_dim != p.n) {
    return cudaErrorNotSupported;
  }
#if FUSE_SM103_QKV_RANK_SWIZZLE
  // Rank rotation changes destination concentration. It requires its own
  // mixed-service calibration, not a per-rank reinterpretation of this plan.
  return cudaErrorNotSupported;
#endif
  // Bound scheduler int32 products before lowering; the scorer applies its
  // own tighter event budget including quantization and communication work.
  if ((int64_t{p.m} / 128) * (p.n / 256) > 1000000) return cudaErrorNotSupported;
  DeviceInfo info{};
  auto status = device_info(&info); // Also verifies CUDA-reported 10.3.
  if (status != cudaSuccess) return status;
  if (info.sm_count != 148) return cudaErrorNotSupported;
  auto select = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    using Gemm = typename Binding::Gemm;
    using Mainloop = typename Binding::Types::Mainloop;
    using Kernel = detail::InputProductionKernel<typename Binding::Kernel, Mxfp8WeightProducer>;
    size_t dynamic_smem = 0;
    auto result = detail::launch_shared_memory<Kernel>(info, &dynamic_smem);
    if (result != cudaSuccess) return result;
    // Lower only scheduler geometry, as the BF16 resolver does. Null operands
    // are never encoded or dereferenced. Budget zero here resolves the caller's
    // raster/swizzle; scoring uses each candidate's actual compute grid.
    const auto args = gemm_arguments<Gemm, Fp8E4m3>(
        p, nullptr, nullptr, nullptr, 1.0f, 0, info, GemmRaster::kAlongM);
    using Scheduler = typename Gemm::TileScheduler;
    const auto order = Scheduler::to_underlying_arguments(
        args.problem_shape, typename Gemm::TileShape{}, typename Gemm::AtomThrShapeMNK{},
        typename Gemm::ClusterShape{}, args.hw_info, args.scheduler);
    using Order = std::decay_t<decltype(order)>;
    detail::Mxfp8QkvTuningRequest request;
    request.m = p.m; request.n = p.n; request.k = p.k;
    request.world = r.world_size; request.sm_count = info.sm_count; request.device = info.device;
    request.capability = 103;
    request.tile_m = cute::size<0>(typename Gemm::TileShape{});
    request.tile_n = cute::size<1>(typename Gemm::TileShape{});
    request.tile_k = cute::size<2>(typename Gemm::TileShape{});
    request.epilogue_n = params.epilogue_n;
    request.stages = Mainloop::DispatchPolicy::Stages;
    request.cluster_ctas = cute::size(typename Gemm::ClusterShape{});
    request.dynamic_smem_bytes = static_cast<int32_t>(dynamic_smem);
    request.raster = order.raster_order_ == Order::RasterOrder::AlongN ? 1 : 0;
    request.max_swizzle_size = p.max_swizzle_size;
    request.swizzle = 1 << order.log_swizzle_size_;
    request.q_heads = r.q_heads; request.kv_heads = r.kv_heads; request.head_dim = r.head_dim;
    const auto plan = detail::select_mxfp8_qkv_plan_cached(request);
    if (plan.status != detail::Mxfp8QkvTuningStatus::Success) {
      return plan.status == detail::Mxfp8QkvTuningStatus::InvalidInput
          ? cudaErrorInvalidValue : cudaErrorNotSupported;
    }
    resolved->projection.num_comm_ctas = plan.comm_ctas;
    return cudaSuccess;
  };
  return params.epilogue_n == 32 ? select(TypeTag<Mxfp8QkvBinding<32>>{})
                               : select(TypeTag<Mxfp8QkvBinding<64>>{});
}

}  // namespace

int32_t recommended_gemm_a2a_mxfp8_comm_ctas(
    const GemmProblem& problem, const UlyssesRoute& route, int32_t epilogue_n) {
  Mxfp8GemmA2AParams params{}, resolved{};
  params.projection.gemm = problem;
  params.projection.route = route;
  params.epilogue_n = epilogue_n;
  return resolve_mxfp8_qkv_communication(params, &resolved) == cudaSuccess
      ? resolved.projection.num_comm_ctas : 0;
}

KernelTraits cutlass_kernel_traits() {
  OprojGemmPolicy policy{};
  if (select_oproj_gemm_policy(&policy) != cudaSuccess) {
    return {};
  }
  KernelTraits traits{};
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    traits = kernel_traits<typename Binding::Kernel>();
    return cudaSuccess;
  };
  return visit_oproj_forward_policy(policy, read_traits) == cudaSuccess
      ? traits : KernelTraits{};
}

KernelTraits projection_cutlass_kernel_traits() {
  return kernel_traits<GemmA2AKernel>();
}

KernelTraits qkv_cutlass_kernel_traits(const GemmProblem&) {
  // Conservative ready capacity for every registered tile, independent of
  // the override. This is not the geometry of the selected launch.
  return kernel_traits<typename QkvForwardN64Binding::Kernel>();
}

KernelTraits qkv_cutlass_kernel_traits(
    const GemmProblem& problem, const UlyssesRoute& route,
    int32_t num_comm_ctas, int32_t sm_count) {
  KernelTraits traits{};
  QkvGemmPolicy policy{};
  if (!supported_problem(problem) || num_comm_ctas <= 0 ||
      num_comm_ctas >= sm_count || select_qkv_gemm_policy(&policy) != cudaSuccess) {
    return traits;
  }
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    traits = kernel_traits<typename Binding::Kernel>();
    return cudaSuccess;
  };
  return visit_qkv_forward_policy(policy, route.qkv_peer_interleaved, read_traits) ==
          cudaSuccess ? traits : KernelTraits{};
}

int64_t a2a_lhs_gemm_ready_elements(const GemmProblem& p, const UlyssesRoute& route) {
  if (p.m <= 0 || p.l != 1 || route.world_size <= 0 || route.world_size > kMaxWorldSize) {
    return 0;
  }
  OprojGemmPolicy policy{};
  if (select_oproj_gemm_policy(&policy) != cudaSuccess) {
    return 0;
  }
  int64_t elements = 0;
  auto count_ready = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    using ConsumerTile = typename Binding::Gemm::TileShape;
    elements = static_cast<int64_t>(ceil_div(p.m, cute::size<0>(ConsumerTile{}))) *
        route.world_size * kReadyFlagStride;
    return cudaSuccess;
  };
  return visit_oproj_forward_policy(policy, count_ready) == cudaSuccess ? elements : 0;
}

// Same geometry resolver as production, telemetry and copy calibration. Zero
// means unavailable, not a usable budget. GEMM auto still selects N128; this
// limited calibration cannot silently replace it with a different collective.
int32_t recommended_a2a_lhs_gemm_comm_ctas(
    const GemmProblem& problem, const UlyssesRoute& route) {
  A2AGemmParams params{};
  params.gemm = problem;
  params.route = route;
  A2AGemmParams resolved{};
  return resolve_oproj_communication(params, &resolved) == cudaSuccess
      ? resolved.num_comm_ctas : 0;
}

int32_t recommended_gemm_a2a_comm_ctas(const GemmProblem&, const UlyssesRoute&) {
  return 0;
}

}  // namespace fuse
