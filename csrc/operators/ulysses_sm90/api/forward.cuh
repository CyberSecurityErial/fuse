// SPDX-License-Identifier: BSD-3-Clause
// Public API implementation; assembled by csrc/operators/ulysses_sm90.cu.
// Declarations: fuse/operators/semantics/ulysses/projection.h.
//
// Module index:
//   - BF16/FP8 A2A-then-GEMM production and telemetry entry points
//   - cached QKV launch-plan construction
//   - BF16/FP8 GEMM-then-A2A production and telemetry entry points

namespace fuse {

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

cudaError_t launch_a2a_gemm_fp8_cutlass(
    const Fp8A2AGemmParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  Fp8A2AGemmParams launch_params = params;
  if (launch_params.num_comm_ctas == 0) {
    launch_params.num_comm_ctas = recommended_a2a_lhs_gemm_comm_ctas_impl(
        params.gemm, params.route, sm_count, device);
  }
  if (launch_params.num_comm_ctas <= 0 ||
      launch_params.num_comm_ctas >= sm_count ||
      (launch_params.lhs_policy != A2ALhsGemmPolicy::kAuto &&
       launch_params.lhs_policy != A2ALhsGemmPolicy::kM128N128)) {
    return cudaErrorInvalidValue;
  }
  if (selected_fp8_pipeline(launch_params.gemm) ==
      Fp8PipelinePolicy::kCooperative) {
    return launch_a2a_lhs_gemm_policy<
        Fp8CooperativeA2ALhsGemm,
        Fp8CooperativeA2ALhsGemmKernel,
        Fp8A2ALhsInputComm>(
            launch_params, stream, sm_count, device);
  }
  return launch_a2a_lhs_gemm_policy<
      Fp8A2ALhsGemm,
      Fp8A2ALhsGemmKernel,
      Fp8A2ALhsInputComm>(
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
  auto launch = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    return launch_a2a_lhs_gemm_policy<
        typename Binding::TelemetryGemm,
        typename Binding::TelemetryKernel,
        typename Binding::TelemetryComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  };
  return visit_oproj_forward_policy(selected.policy, launch);
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

namespace {

enum class QkvGemmPolicyRequest : uint64_t {
  kAuto,
  kLegacy,
  kM128N64,
  kM128N128,
  kM128N160,
  kM128N192,
  kM128N256,
  kM128N320,
  kWaveTimeModel,
};

QkvGemmPolicyRequest normalized_qkv_gemm_policy_request() {
  const char* value = std::getenv("FUSE_QKV_GEMM_POLICY");
  if (value == nullptr) {
    return QkvGemmPolicyRequest::kAuto;
  }
  if (std::strcmp(value, "legacy") == 0) {
    return QkvGemmPolicyRequest::kLegacy;
  }
  if (std::strcmp(value, "m128n64") == 0) {
    return QkvGemmPolicyRequest::kM128N64;
  }
  if (std::strcmp(value, "m128n128") == 0) {
    return QkvGemmPolicyRequest::kM128N128;
  }
  if (std::strcmp(value, "m128n160") == 0) {
    return QkvGemmPolicyRequest::kM128N160;
  }
  if (std::strcmp(value, "m128n192") == 0) {
    return QkvGemmPolicyRequest::kM128N192;
  }
  if (std::strcmp(value, "m128n256") == 0) {
    return QkvGemmPolicyRequest::kM128N256;
  }
  if (std::strcmp(value, "m128n320") == 0) {
    return QkvGemmPolicyRequest::kM128N320;
  }
  if (std::strcmp(value, "wave_time_model") == 0) {
    return QkvGemmPolicyRequest::kWaveTimeModel;
  }
  // Unknown values have always fallen through to the normal automatic path.
  return QkvGemmPolicyRequest::kAuto;
}

struct QkvLaunchPlanKey {
  // Only immutable policy inputs belong here.  Device pointers, epoch and
  // alpha remain per-launch values in GemmA2AParams.
  std::array<uint64_t, 34> words{};

  bool operator==(const QkvLaunchPlanKey& other) const {
    return words == other.words;
  }
};

using QkvLaunchPlanKeyHash = WordArrayHash<QkvLaunchPlanKey>;

QkvLaunchPlanKey make_qkv_launch_plan_key(
    const GemmA2AParams& params,
    int32_t device) {
  const GemmProblem& p = params.gemm;
  const UlyssesRoute& r = params.route;
  const auto comm_policy = normalized_qkv_comm_policy_request();
  const auto gemm_policy = normalized_qkv_gemm_policy_request();
  const auto nvlink_override = normalized_qkv_nvlink_override();
  return {{
      static_cast<uint64_t>(device),
      static_cast<uint64_t>(params.num_comm_ctas),
      static_cast<uint64_t>(p.m),
      static_cast<uint64_t>(p.n),
      static_cast<uint64_t>(p.k),
      static_cast<uint64_t>(p.l),
      static_cast<uint64_t>(p.input_dtype),
      static_cast<uint64_t>(p.weight_dtype),
      static_cast<uint64_t>(p.output_dtype),
      static_cast<uint64_t>(p.transpose_a),
      static_cast<uint64_t>(p.transpose_b),
      static_cast<uint64_t>(p.raster),
      static_cast<uint64_t>(p.max_swizzle_size),
      static_cast<uint64_t>(r.world_size),
      static_cast<uint64_t>(r.rank),
      static_cast<uint64_t>(r.batch),
      static_cast<uint64_t>(r.global_seq),
      static_cast<uint64_t>(r.seq_local),
      static_cast<uint64_t>(r.q_heads),
      static_cast<uint64_t>(r.kv_heads),
      static_cast<uint64_t>(r.local_heads),
      static_cast<uint64_t>(r.head_dim),
      static_cast<uint64_t>(r.channel_count),
      static_cast<uint64_t>(r.kind),
      static_cast<uint64_t>(r.direction),
      static_cast<uint64_t>(r.qkv_peer_interleaved),
      static_cast<uint64_t>(r.defer_v_a2a),
      static_cast<uint64_t>(r.causal_load_balanced),
      static_cast<uint64_t>(r.cyclic_peer_order),
      static_cast<uint64_t>(r.packed_row_granularity),
      static_cast<uint64_t>(comm_policy),
      static_cast<uint64_t>(gemm_policy),
      static_cast<uint64_t>(nvlink_override.present),
      nvlink_override.bits,
  }};
}

struct QkvLaunchPlan {
  int32_t device = 0;
  int32_t sm_count = 0;
  int32_t num_comm_ctas = 0;
  QkvGemmPolicy policy = QkvGemmPolicy::kM128N128;
};

using QkvLaunchPlanCache = BoundedLaunchCache<
    QkvLaunchPlanKey, QkvLaunchPlan, QkvLaunchPlanKeyHash>;

cudaError_t cached_qkv_launch_plan(
    const GemmA2AParams& params,
    QkvLaunchPlan* result) {
  if (result == nullptr) {
    return cudaErrorInvalidValue;
  }

  int32_t device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  const QkvLaunchPlanKey key = make_qkv_launch_plan_key(params, device);
  static QkvLaunchPlanCache cache;
  auto create = [&](QkvLaunchPlan* plan) {
    // Resolve the device, communication split and GEMM kernel as one immutable
    // plan. Hot eager launches then bypass the property and candidate queries.
    plan->device = device;
    cudaError_t create_status = cudaDeviceGetAttribute(
        &plan->sm_count, cudaDevAttrMultiProcessorCount, device);
    if (create_status != cudaSuccess) {
      return create_status;
    }
    plan->num_comm_ctas = params.num_comm_ctas == 0
        ? recommended_gemm_a2a_comm_ctas(params.gemm, params.route)
        : params.num_comm_ctas;
    if (plan->num_comm_ctas <= 0 ||
        plan->num_comm_ctas >= plan->sm_count) {
      return cudaErrorInvalidValue;
    }
    plan->policy = select_qkv_gemm_policy(
        params.gemm,
        plan->num_comm_ctas,
        plan->sm_count,
        params.route.qkv_peer_interleaved);
    return cudaSuccess;
  };
  return cache.get_or_create(key, result, create);
}

}  // namespace

cudaError_t launch_gemm_a2a_cutlass(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  QkvLaunchPlan plan{};
  cudaError_t status = cached_qkv_launch_plan(params, &plan);
  if (status != cudaSuccess) {
    return status;
  }
  GemmA2AParams launch_params = params;
  launch_params.num_comm_ctas = plan.num_comm_ctas;

  if (launch_params.route.kind == RouteKind::kQkvGqaPack &&
      launch_params.route.direction == RouteDirection::kForward) {
    return launch_qkv_forward_policy(
        launch_params,
        plan.policy,
        stream,
        plan.sm_count,
        plan.device);
  }
  return cudaErrorNotSupported;
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_gemm_a2a_role_telemetry(
    const GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  QkvLaunchPlan plan{};
  cudaError_t status = cached_qkv_launch_plan(params, &plan);
  if (status != cudaSuccess) {
    return status;
  }
  GemmA2AParams launch_params = params;
  launch_params.num_comm_ctas = plan.num_comm_ctas;
  if (timeline == nullptr || timeline_capacity < plan.sm_count ||
      launch_params.num_comm_ctas <= 0 ||
      launch_params.num_comm_ctas >= plan.sm_count) {
    return cudaErrorInvalidValue;
  }
  if (launch_params.route.kind != RouteKind::kQkvGqaPack ||
      launch_params.route.direction != RouteDirection::kForward) {
    return cudaErrorNotSupported;
  }
  return launch_qkv_forward_policy<true>(
      launch_params,
      plan.policy,
      stream,
      plan.sm_count,
      plan.device,
      timeline,
      timeline_capacity);
}
#endif


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
  const Fp8TilePolicy tile = select_fp8_qkv_tile(
      launch_params.gemm,
      launch_params.route,
      launch_params.num_comm_ctas,
      sm_count,
      device);
  if (tile == Fp8TilePolicy::kM128N64) {
    Fp8QkvGqaPackCommN64::Arguments comm_args{};
    comm_args.params = launch_params;
    if (!Fp8QkvGqaPackCommN64::can_implement(comm_args)) {
      return cudaErrorNotSupported;
    }
    return launch_gemm_a2a_impl<
        Fp8N64OutputGemm, Fp8QkvGqaPackCommN64>(
            launch_params, stream, sm_count, device);
  }
  if (tile == Fp8TilePolicy::kM128N256ClusterM2) {
    Fp8QkvGqaPackCommWide::Arguments comm_args{};
    comm_args.params = launch_params;
    if (!Fp8QkvGqaPackCommWide::can_implement(comm_args)) {
      return cudaErrorNotSupported;
    }
    return launch_gemm_a2a_impl<
        Fp8WideOutputGemm, Fp8QkvGqaPackCommWide>(
            launch_params, stream, sm_count, device);
  }
  Fp8QkvGqaPackComm::Arguments comm_args{};
  comm_args.params = launch_params;
  if (!Fp8QkvGqaPackComm::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }
  if (selected_fp8_pipeline(launch_params.gemm) ==
      Fp8PipelinePolicy::kCooperative) {
    return launch_gemm_a2a_impl<
        Fp8CooperativeOutputGemm, Fp8QkvGqaPackComm>(
            launch_params, stream, sm_count, device);
  }
  return launch_gemm_a2a_impl<Fp8OutputGemm, Fp8QkvGqaPackComm>(
      launch_params, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_gemm_a2a_fp8_role_telemetry(
    const Fp8GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
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
  if (timeline == nullptr || timeline_capacity < sm_count ||
      launch_params.num_comm_ctas <= 0 ||
      launch_params.num_comm_ctas >= sm_count ||
      launch_params.route.kind != RouteKind::kQkvGqaPack ||
      launch_params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  const Fp8TilePolicy tile = select_fp8_qkv_tile(
      launch_params.gemm,
      launch_params.route,
      launch_params.num_comm_ctas,
      sm_count,
      device);
  if (tile == Fp8TilePolicy::kM128N64) {
    return launch_gemm_a2a_impl<
        Fp8N64OutputGemm,
        Fp8QkvGqaPackCommN64,
        Fp8GemmA2AParams,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity);
  }
  if (tile == Fp8TilePolicy::kM128N256ClusterM2) {
    return launch_gemm_a2a_impl<
        Fp8WideOutputGemm,
        Fp8QkvGqaPackCommWide,
        Fp8GemmA2AParams,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity);
  }
  if (selected_fp8_pipeline(launch_params.gemm) ==
      Fp8PipelinePolicy::kCooperative) {
    return launch_gemm_a2a_impl<
        Fp8CooperativeOutputGemm,
        Fp8QkvGqaPackComm,
        Fp8GemmA2AParams,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity);
  }
  return launch_gemm_a2a_impl<
      Fp8OutputGemm,
      Fp8QkvGqaPackComm,
      Fp8GemmA2AParams,
      true>(
          launch_params,
          stream,
          sm_count,
          device,
          timeline,
          timeline_capacity);
}
#endif

}  // namespace fuse
