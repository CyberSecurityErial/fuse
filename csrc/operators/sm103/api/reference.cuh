// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Independent BF16 compute/route calibration, not an alternative fused path.
// Inputs must be materialized before timing. The caller owns buffers and
// cross-rank launch coordination; these APIs never allocate or clear flags.
// Use separate calibration ready/done buffers and an epoch sequence counting
// actual communication launches only. A2A copy increments arrival counters;
// pure GEMM does not. Reusing production counters without accounting for that
// can release a later consumer early or leave it waiting for missing arrivals.
namespace fuse {
namespace {

inline cudaError_t reference_device_info(DeviceInfo* info, int32_t reserved_ctas) {
  const cudaError_t status = device_info(info);
  if (status != cudaSuccess) {
    return status;
  }
  return reserved_ctas >= 0 && reserved_ctas < info->sm_count
      ? cudaSuccess : cudaErrorInvalidValue;
}

template <class Binding, class Gemm = typename Binding::PureGemm, bool Instrumented = false>
cudaError_t launch_gemm_reference_impl(
    const GemmProblem& problem, const Bf16* lhs, const Bf16* rhs_nt,
    Bf16* output, float alpha, int32_t reserved_comm_ctas,
    const DeviceInfo& info, GemmRaster fallback, cudaStream_t stream) {
  auto aligned = [](const void* pointer) {
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
  };
  if (!supported_problem(problem) || !std::isfinite(alpha) ||
      !aligned(lhs) || !aligned(rhs_nt) || !aligned(output)) {
    return cudaErrorInvalidValue;
  }
  using Kernel = detail::GemmReferenceKernel<Gemm, typename Binding::Kernel>;
  auto args = gemm_arguments<Gemm>(
      problem, lhs, rhs_nt, output, alpha, reserved_comm_ctas, info, fallback);
  // Keep the reduced compute budget and scheduler stride. Only the physical
  // CTA prefix disappears: worker 0 now starts at blockIdx.x == 0.
  args.scheduler.block_offset = 0;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    if (!detail::oproj_pipeline_sink) return cudaErrorInvalidValue;
    args.mainloop.probe = *detail::oproj_pipeline_sink;
    args.epilogue.probe = *detail::oproj_pipeline_sink;
  }
#endif
  if (!Gemm::can_implement(args) || Gemm::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Gemm::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  const auto params = Gemm::to_underlying_arguments(args, nullptr);
  if (params.scheduler.block_offset != 0) {
    return cudaErrorInvalidConfiguration;
  }
  return detail::launch_reference_cooperative<Kernel>(
      params, info, info.sm_count - reserved_comm_ctas, stream);
}

}  // namespace

cudaError_t launch_batched_cutlass_reference(
    const GemmA2AParams& params, cudaStream_t stream, int32_t reserved_comm_ctas) {
  if (params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorNotSupported;
  }
  DeviceInfo info{};
  cudaError_t status = reference_device_info(&info, reserved_comm_ctas);
  if (status != cudaSuccess) {
    return status;
  }
  QkvGemmPolicy policy{};
  status = select_qkv_gemm_policy(&policy);
  if (status != cudaSuccess) {
    return status;
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_gemm_reference_impl<Binding>(
        params.gemm, params.lhs, params.rhs_nt, params.local_output,
        params.alpha, reserved_comm_ctas, info, GemmRaster::kAlongM, stream);
  };
  return visit_qkv_forward_policy(policy, params.route.qkv_peer_interleaved, launch);
}

cudaError_t launch_a2a_gemm_cutlass_reference(
    const A2AGemmParams& params, cudaStream_t stream, int32_t reserved_comm_ctas) {
  // Independent GEMM: reserved=0 means every SM, never automatic communication.
  // For a budget-matched reference, obtain the positive reservation with
  // recommended_a2a_lhs_gemm_comm_ctas and pass it explicitly after checking it.
  if (params.lhs_policy != A2ALhsGemmPolicy::kAuto) {
    return cudaErrorNotSupported;
  }
  DeviceInfo info{};
  cudaError_t status = reference_device_info(&info, reserved_comm_ctas);
  if (status != cudaSuccess) {
    return status;
  }
  OprojGemmPolicy policy{};
  status = select_oproj_gemm_policy(&policy);
  if (status != cudaSuccess) {
    return status;
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
#if FUSE_ENABLE_PROFILING
    if (detail::oproj_pipeline_sink) {
      return launch_gemm_reference_impl<Binding, typename Binding::TelemetryPureGemm, true>(
          params.gemm, params.input_staging, params.rhs_nt, params.output,
          params.alpha, reserved_comm_ctas, info, GemmRaster::kAlongN, stream);
    }
#endif
    return launch_gemm_reference_impl<Binding>(
        params.gemm, params.input_staging, params.rhs_nt, params.output,
        params.alpha, reserved_comm_ctas, info, GemmRaster::kAlongN, stream);
  };
  return visit_oproj_forward_policy(policy, launch);
}

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& input, cudaStream_t stream) {
  A2AGemmParams params{};
  cudaError_t status = resolve_oproj_communication(input, &params);
  if (status != cudaSuccess) return status;
  if (params.num_comm_ctas <= 0 || params.epoch == 0 ||
      !std::isfinite(params.alpha) || params.lhs_policy != A2ALhsGemmPolicy::kAuto) {
    return cudaErrorInvalidValue;
  }
  DeviceInfo info{};
  status = reference_device_info(&info, params.num_comm_ctas);
  if (status != cudaSuccess) {
    return status;
  }
  OprojGemmPolicy policy{};
  status = select_oproj_gemm_policy(&policy);
  if (status != cudaSuccess) {
    return status;
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    using Comm = typename Binding::Comm;
    using Kernel = detail::CopyReferenceKernel<Comm, typename Binding::Kernel>;
    static_assert(!Comm::kNeedsGridFinalize);
    typename Comm::Arguments comm{};
    comm.params = params;
    using Gemm = typename Binding::Gemm;
    const auto gemm = gemm_arguments<Gemm>(
        params.gemm, params.input_staging, params.rhs_nt, params.output,
        params.alpha, params.num_comm_ctas, info, GemmRaster::kAlongN);
    comm.input_order = a2a_input_order<Gemm>(gemm);
    cudaError_t result = Comm::initialize(comm);
    if (result != cudaSuccess) {
      return result;
    }
    if (!Comm::can_implement(comm)) {
      return cudaErrorNotSupported;
    }
    return detail::launch_reference_cooperative<Kernel>(
        Comm::to_underlying_arguments(comm), info, params.num_comm_ctas, stream);
  };
  return visit_oproj_forward_policy(policy, launch);
}

cudaError_t launch_gemm_a2a_copy_reference(
    const GemmA2AParams& params, cudaStream_t stream) {
  // This calibration measures the full Q/K/V route. A deferred-V boundary
  // would both omit V traffic and need a different producer-completion rule.
  if (params.route.defer_v_a2a) {
    return cudaErrorNotSupported;
  }
  if (params.num_comm_ctas <= 0 || params.epoch == 0 || !std::isfinite(params.alpha)) {
    return cudaErrorInvalidValue;
  }
  DeviceInfo info{};
  cudaError_t status = reference_device_info(&info, params.num_comm_ctas);
  if (status != cudaSuccess) {
    return status;
  }
  QkvGemmPolicy policy{};
  status = select_qkv_gemm_policy(&policy);
  if (status != cudaSuccess) {
    return status;
  }
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    using Comm = typename Binding::Comm;
    using Kernel = detail::CopyReferenceKernel<Comm, typename Binding::Kernel>;
    static_assert(Comm::kNeedsGridFinalize,
                  "QKV route calibration must include cross-rank completion.");
    typename Comm::Arguments comm{};
    comm.params = params;
    if (comm.params.gemm.raster == GemmRaster::kHeuristic) {
      comm.params.gemm.raster = GemmRaster::kAlongM;
    }
    const cudaError_t result = Comm::initialize(comm);
    if (result != cudaSuccess) {
      return result;
    }
    return detail::launch_reference_cooperative<Kernel>(
        Comm::to_underlying_arguments(comm), info, params.num_comm_ctas, stream);
  };
  return visit_qkv_forward_policy(policy, params.route.qkv_peer_interleaved, launch);
}

}  // namespace fuse
