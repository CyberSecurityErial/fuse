// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace fuse {

cudaError_t launch_a2a_gemm_cutlass(const A2AGemmParams& params, cudaStream_t stream) {
  return launch_oproj_forward_policy(params, stream);
}

cudaError_t launch_gemm_a2a_cutlass(const GemmA2AParams& params, cudaStream_t stream) {
  return launch_qkv_forward_policy(params, stream);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_a2a_gemm_cutlass_role_telemetry(
    const A2AGemmParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    A2AGemmPeerTimeline* peer_timeline,
    int32_t peer_timeline_capacity,
    cudaStream_t stream) {
  return launch_oproj_forward_policy<true>(
      params, stream, timeline, timeline_capacity,
      peer_timeline, peer_timeline_capacity);
}

cudaError_t launch_gemm_a2a_role_telemetry(
    const GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  return launch_qkv_forward_policy<true>(params, stream, timeline, timeline_capacity);
}

cudaError_t launch_gemm_a2a_route_telemetry(
    const GemmA2AParams& params, A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity, QkvRouteTimeline* route_timeline,
    int32_t route_capacity, cudaStream_t stream) {
  if (!route_timeline || route_capacity <= 0) return cudaErrorInvalidValue;
  return launch_qkv_forward_policy<true>(params, stream, timeline,
      timeline_capacity, route_timeline, route_capacity);
}

cudaError_t query_gemm_a2a_route_timeline_capacity(const GemmA2AParams& params, int32_t* capacity) {
  if (!capacity || params.gemm.m <= 0 || params.gemm.n <= 0 || params.num_comm_ctas <= 0) return cudaErrorInvalidValue;
  QkvGemmPolicy policy{};
  const auto status = select_qkv_gemm_policy(&policy);
  if (status != cudaSuccess) return status;
  auto query = [&](auto tag) {
    using Comm = typename decltype(tag)::type::Comm;
    const uint64_t count = Comm::route_slots(params) + params.num_comm_ctas * Comm::kQkvBulkSlots;
    if (count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) return cudaErrorInvalidValue;
    *capacity = static_cast<int32_t>(count);
    return cudaSuccess;
  };
  return visit_qkv_forward_policy(policy, params.route.qkv_peer_interleaved, query);
}

cudaError_t query_a2a_gemm_role_resources(A2AGemmRoleResources* resources) {
  if (resources == nullptr) {
    return cudaErrorInvalidValue;
  }
  OprojGemmPolicy policy{};
  cudaError_t status = select_oproj_gemm_policy(&policy);
  if (status != cudaSuccess) {
    return status;
  }
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) {
    return status;
  }
  auto read_resources = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    using Kernel = typename Binding::Kernel;
    using Comm = typename Binding::Comm;
    cudaFuncAttributes production{};
    cudaFuncAttributes instrumented{};
    cudaError_t result = cudaFuncGetAttributes(
        &production, cutlass::device_kernel<Kernel>);
    if (result != cudaSuccess) {
      return result;
    }
    result = cudaFuncGetAttributes(
        &instrumented, cutlass::device_kernel<typename Binding::TelemetryKernel>);
    if (result != cudaSuccess) {
      return result;
    }
    size_t dynamic_smem = 0;
    result = detail::launch_shared_memory<Kernel>(info, &dynamic_smem);
    if (result != cudaSuccess) {
      return result;
    }
    *resources = {
        Kernel::MaxThreadsPerBlock,
        production.numRegs,
        instrumented.numRegs,
        static_cast<int32_t>(production.sharedSizeBytes),
        static_cast<int32_t>(dynamic_smem),
        static_cast<int32_t>(cute::size(typename Kernel::ClusterShape{})),
        Comm::kMinThreads / 32,
        // Legacy field name: physical CTA warp budget, not measured activity.
        Kernel::MaxThreadsPerBlock / 32,
        static_cast<int32_t>(Comm::SharedStorageBytes)};
    return cudaSuccess;
  };
  return visit_oproj_forward_policy(policy, read_resources);
}

namespace detail {

cudaError_t launch_qkv_epilogue_telemetry(
    const GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    QkvEpilogueRecord* records,
    int32_t record_capacity,
    cudaStream_t stream) {
  FUSE_SM103_HOST_BEGIN();
  QkvGemmPolicy policy{};
  const cudaError_t status = select_qkv_gemm_policy(&policy);
  if (status != cudaSuccess) FUSE_SM103_HOST_RETURN(status);
  if (policy != QkvGemmPolicy::kM128N256K64E32 || params.route.qkv_peer_interleaved) {
    FUSE_SM103_HOST_RETURN(cudaErrorNotSupported);
  }
  FUSE_SM103_HOST_RETURN((launch_gemm_a2a_impl<
      QkvEpilogueProbeGemm, QkvEpilogueProbeKernel,
      typename QkvForwardN256K64E32Binding::Comm, true, true>(
          params, stream, timeline, timeline_capacity, records, record_capacity)));
}

cudaError_t query_qkv_epilogue_resources(
    const GemmA2AParams& params, QkvEpilogueResources* resources) {
  if (!resources || !supported_problem(params.gemm) || params.num_comm_ctas <= 0) {
    return cudaErrorInvalidValue;
  }
  QkvGemmPolicy policy{};
  cudaError_t status = select_qkv_gemm_policy(&policy);
  if (status != cudaSuccess) return status;
  if (policy != QkvGemmPolicy::kM128N256K64E32 || params.route.qkv_peer_interleaved) {
    return cudaErrorNotSupported;
  }
  DeviceInfo info{};
  status = device_info(&info);
  if (status != cudaSuccess) return status;
  if (params.num_comm_ctas >= info.sm_count) return cudaErrorInvalidValue;

  using Binding = QkvForwardN256K64E32Binding;
  QkvEpilogueResources result{};
  status = cudaFuncGetAttributes(&result.production, cutlass::device_kernel<typename Binding::Kernel>);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&result.role_telemetry, cutlass::device_kernel<typename Binding::TelemetryKernel>);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&result.epilogue_telemetry, cutlass::device_kernel<QkvEpilogueProbeKernel>);
  if (status != cudaSuccess) return status;
  size_t dynamic_smem = 0;
  status = launch_shared_memory<QkvEpilogueProbeKernel>(info, &dynamic_smem);
  if (status != cudaSuccess) return status;
  result.dynamic_smem_bytes = static_cast<int32_t>(dynamic_smem);
  result.tile_m = cute::size<0>(typename Binding::TileShape{});
  result.tile_n = cute::size<1>(typename Binding::TileShape{});
  result.tile_k = cute::size<2>(typename Binding::TileShape{});
  result.cluster_ctas = cute::size(typename QkvEpilogueProbeKernel::ClusterShape{});
  *resources = result;
  return cudaSuccess;
}

}  // namespace detail
#endif

}  // namespace fuse
