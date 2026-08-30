// SPDX-License-Identifier: BSD-3-Clause
// Public API implementation; assembled by csrc/operators/ulysses_sm90.cu.
// Declarations: fuse/operators/ulysses/qkv_backward.h and oproj_backward.h.
//
// Module index:
//   - QKV/OProj BF16 dgrad launch plans and launch entry points
//   - immediate/deferred wgrad entry points and combined backward wrappers
//   - matching pure-E4M3 backward entry points

namespace fuse {

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

int64_t qkv_backward_ready_elements(const QkvBackwardDataParams& params) {
  if (params.local_tokens <= 0 || params.q_heads <= 0 ||
      params.kv_heads <= 0) {
    return 0;
  }
  const int64_t m_tiles = ceil_div(params.local_tokens, kBlockM);
  return m_tiles * (params.q_heads + 2LL * params.kv_heads) *
      kReadyFlagStride;
}

BackwardGemmPolicy recommended_qkv_backward_gemm_policy(
    const QkvBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count) {
  const int32_t packed_width =
      (params.q_heads + 2 * params.kv_heads) * params.head_dim;
  const BackwardGemmPolicy selected = select_backward_gemm_policy(
      params.gemm_policy,
      params.local_tokens,
      params.hidden,
      packed_width,
      resolved_comm_ctas,
      sm_count);
  // For a long-K dgrad with an even number of M128 tiles, cluster-M2 lets
  // each CTA pair multicast the same weight tile.  The output tile count is
  // unchanged, so this avoids the extra wave that made M64 tiles unattractive.
  // Explicit requests remain exact benchmark knobs; only kAuto upgrades the
  // independently selected N64 policy.
  const int32_t m_tiles = ceil_div(params.local_tokens, kBlockM);
  if (params.gemm_policy == BackwardGemmPolicy::kAuto &&
      selected == BackwardGemmPolicy::kM128N64 &&
      packed_width >= 4096 && m_tiles >= 2 && m_tiles % 2 == 0) {
    return BackwardGemmPolicy::kM128N64ClusterM2;
  }
  return selected;
}

KernelTraits qkv_backward_kernel_traits(
    const QkvBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count) {
  return qkv_backward_traits_for_policy(
      recommended_qkv_backward_gemm_policy(
          params, resolved_comm_ctas, sm_count));
}

namespace {

struct BackwardDeviceInfo {
  int32_t device = 0;
  int32_t sm_count = 0;
};

cudaError_t cached_backward_device_info(BackwardDeviceInfo* result) {
  if (result == nullptr) {
    return cudaErrorInvalidValue;
  }
  int32_t device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  struct Cache {
    std::mutex mutex;
    std::unordered_map<int32_t, int32_t> sm_counts;
  };
  static Cache cache;
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.sm_counts.find(device);
    if (found != cache.sm_counts.end()) {
      *result = {device, found->second};
      return cudaSuccess;
    }
  }
  int32_t sm_count = 0;
  status = cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.sm_counts.emplace(device, sm_count);
  }
  *result = {device, sm_count};
  return cudaSuccess;
}

int32_t recommended_qkv_backward_comm_ctas_impl(
    const QkvBackwardDataParams& params,
    int32_t sm_count) {
  if (params.local_tokens <= 0 || params.hidden <= 0 || params.batch <= 0 ||
      params.q_heads <= 0 || params.kv_heads <= 0 ||
      params.head_dim <= 0 || params.world_size <= 1 ||
      params.world_size > kMaxWorldSize ||
      params.q_heads % params.world_size != 0 ||
      params.kv_heads % params.world_size != 0 || sm_count <= 2) {
    return 0;
  }

  // One QKV push task moves one M128 x head_dim tile.  A communication CTA
  // owns twelve independent warp/TMA issue slots.  H200 measurements show
  // that ten CTAs are enough to reach the useful issuer bandwidth; beyond
  // that point extra CTAs are useful only when they remove long task waves
  // without forcing the GEMM out of a single persistent wave.
  constexpr int32_t kRouteSlotsPerCta = 12;
  constexpr int32_t kTargetRouteWaves = 2;
  constexpr int32_t kRouteIssuerSaturation = 10;
  constexpr int32_t kMultiComputeWaveCap = 12;
  constexpr int32_t kMaxCommCtas = 32;
  const int64_t m_tiles = ceil_div(params.local_tokens, kBlockM);
  const int64_t packed_heads = params.q_heads + 2LL * params.kv_heads;
  const int64_t route_tasks = m_tiles * packed_heads;
  const int32_t issuer_floor = static_cast<int32_t>(
      min<int64_t>(kRouteIssuerSaturation, route_tasks));
  const int32_t two_wave_target = static_cast<int32_t>(ceil_div(
      route_tasks,
      static_cast<int64_t>(kRouteSlotsPerCta) * kTargetRouteWaves));
  int32_t desired = max(issuer_floor, two_wave_target);
  desired = (desired + 1) & ~1;
  const int32_t max_comm = min(kMaxCommCtas, (sm_count - 2) & ~1);
  desired = min(desired, max_comm);
  if (desired <= 0) {
    return 0;
  }

  // A larger route subgrid is effectively free only while every GEMM tile
  // can still reside in one physical compute wave.  Once compute needs more
  // than one wave, cap the route side near its measured saturation point so
  // it cannot steal additional GEMM workers.  This rule uses only physical
  // task counts and the selected tile geometry; it has no model/shape table.
  if (desired > kMultiComputeWaveCap) {
    const BackwardGemmPolicy policy =
        recommended_qkv_backward_gemm_policy(params, desired, sm_count);
    const KernelTraits traits = qkv_backward_traits_for_policy(policy);
    if (traits.block_m <= 0 || traits.block_n <= 0) {
      return 0;
    }
    int64_t physical_m_tiles = ceil_div(params.local_tokens, traits.block_m);
    if (policy == BackwardGemmPolicy::kM128N64ClusterM2) {
      physical_m_tiles = ceil_div(physical_m_tiles, int64_t{2}) * 2;
    }
    const int64_t compute_ctas =
        physical_m_tiles * ceil_div(params.hidden, traits.block_n);
    if (compute_ctas > sm_count - desired) {
      desired = min(max_comm, kMultiComputeWaveCap);
    }
  }
  return desired;
}

struct QkvBackwardLaunchPlanKey {
  std::array<uint64_t, 12> words{};

  bool operator==(const QkvBackwardLaunchPlanKey& other) const {
    return words == other.words;
  }
};

using QkvBackwardLaunchPlanKeyHash =
    WordArrayHash<QkvBackwardLaunchPlanKey>;

QkvBackwardLaunchPlanKey make_qkv_backward_launch_plan_key(
    const QkvBackwardDataParams& params,
    int32_t device) {
  return {{
      static_cast<uint64_t>(device),
      static_cast<uint64_t>(params.num_comm_ctas),
      static_cast<uint64_t>(params.gemm_policy),
      static_cast<uint64_t>(params.local_tokens),
      static_cast<uint64_t>(params.hidden),
      static_cast<uint64_t>(params.batch),
      static_cast<uint64_t>(params.q_heads),
      static_cast<uint64_t>(params.kv_heads),
      static_cast<uint64_t>(params.head_dim),
      static_cast<uint64_t>(params.world_size),
      static_cast<uint64_t>(params.rank),
      static_cast<uint64_t>(params.causal_load_balanced),
  }};
}

struct QkvBackwardLaunchPlan {
  int32_t device = 0;
  int32_t sm_count = 0;
  int32_t num_comm_ctas = 0;
  BackwardGemmPolicy policy = BackwardGemmPolicy::kM128N128;
};

using QkvBackwardLaunchPlanCache = BoundedLaunchCache<
    QkvBackwardLaunchPlanKey,
    QkvBackwardLaunchPlan,
    QkvBackwardLaunchPlanKeyHash>;

cudaError_t cached_qkv_backward_launch_plan(
    const QkvBackwardDataParams& params,
    QkvBackwardLaunchPlan* result) {
  if (result == nullptr) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  const QkvBackwardLaunchPlanKey key =
      make_qkv_backward_launch_plan_key(params, device_info.device);
  static QkvBackwardLaunchPlanCache cache;
  auto create = [&](QkvBackwardLaunchPlan* plan) {
    plan->device = device_info.device;
    plan->sm_count = device_info.sm_count;
    plan->num_comm_ctas = params.num_comm_ctas == 0
        ? recommended_qkv_backward_comm_ctas_impl(params, plan->sm_count)
        : params.num_comm_ctas;
    if (plan->num_comm_ctas <= 0 ||
        plan->num_comm_ctas >= plan->sm_count) {
      return cudaErrorInvalidConfiguration;
    }
    plan->policy = recommended_qkv_backward_gemm_policy(
        params, plan->num_comm_ctas, plan->sm_count);
    return cudaSuccess;
  };
  return cache.get_or_create(key, result, create);
}

}  // namespace

int32_t recommended_qkv_backward_comm_ctas(
    const QkvBackwardDataParams& params) {
  QkvBackwardDataParams automatic = params;
  automatic.num_comm_ctas = 0;
  QkvBackwardLaunchPlan plan{};
  if (cached_qkv_backward_launch_plan(automatic, &plan) != cudaSuccess) {
    return 0;
  }
  return plan.num_comm_ctas;
}

static cudaError_t prepare_qkv_backward_data_launch(
    const QkvBackwardDataParams& params,
    QkvBackwardKernelParams* launch,
    int32_t* sm_count,
    int32_t* device) {
  if (launch == nullptr || sm_count == nullptr || device == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (!params.grad_q || !params.grad_k || !params.grad_v || !params.weight ||
      !params.grad_input || params.local_tokens <= 0 || params.hidden <= 0 ||
      params.batch <= 0 || params.local_tokens % params.batch != 0 ||
      params.q_heads <= 0 || params.kv_heads <= 0 ||
      params.head_dim <= 0 || params.head_dim % kAlignment != 0 ||
      params.world_size <= 1 || params.world_size > kMaxWorldSize ||
      params.rank < 0 || params.rank >= params.world_size ||
      params.q_heads % params.kv_heads != 0 ||
      params.q_heads % params.world_size != 0 ||
      params.kv_heads % params.world_size != 0 || params.epoch == 0 ||
      (params.causal_load_balanced &&
       (params.local_tokens / params.batch) % 2 != 0)) {
    return cudaErrorInvalidValue;
  }
  *launch = {};
  QkvBackwardLaunchPlan plan{};
  cudaError_t status = cached_qkv_backward_launch_plan(params, &plan);
  if (status != cudaSuccess) {
    return status;
  }
  *sm_count = plan.sm_count;
  *device = plan.device;
  launch->local_q = params.grad_q;
  launch->local_k = params.grad_k;
  launch->local_v = params.grad_v;
  for (int32_t peer = 0; peer < params.world_size; ++peer) {
    launch->peer_staging[peer] = params.peer_dqkv_staging[peer];
    launch->peer_ready[peer] = params.peer_ready[peer];
    launch->peer_done_epoch[peer] = params.peer_done_epoch[peer];
  }
  launch->weight = params.weight;
  launch->grad_input = params.grad_input;
  const int32_t packed_width =
      (params.q_heads + 2 * params.kv_heads) * params.head_dim;
  launch->gemm = {
      params.local_tokens, params.hidden, packed_width, 1};
  launch->gemm.transpose_b = true;
  launch->gemm.stride_b.row = params.hidden;
  launch->gemm.raster = GemmRaster::kAlongN;
  launch->gemm.max_swizzle_size = 1;
  launch->route.world_size = params.world_size;
  launch->route.rank = params.rank;
  launch->route.batch = params.batch;
  launch->route.seq_local = params.local_tokens / params.batch;
  launch->route.global_seq =
      launch->route.seq_local * params.world_size;
  launch->route.q_heads = params.q_heads;
  launch->route.kv_heads = params.kv_heads;
  launch->route.local_heads = params.q_heads / params.world_size;
  launch->route.head_dim = params.head_dim;
  launch->route.causal_load_balanced = params.causal_load_balanced;
  launch->route.kind = RouteKind::kQkvGqaPack;
  launch->route.direction = RouteDirection::kInverse;
  launch->num_comm_ctas = plan.num_comm_ctas;
  launch->epoch = params.epoch;
  launch->alpha = params.alpha;

  if (launch->num_comm_ctas <= 0 || launch->num_comm_ctas >= *sm_count ||
      launch->num_comm_ctas % 2 != 0) {
    return cudaErrorInvalidConfiguration;
  }
  launch->gemm_policy = plan.policy;
  return cudaSuccess;
}

cudaError_t launch_qkv_backward_data(
    const QkvBackwardDataParams& params,
    cudaStream_t stream) {
  QkvBackwardKernelParams launch{};
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = prepare_qkv_backward_data_launch(
      params, &launch, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_qkv_backward_data_impl(
      launch, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_qkv_backward_data_role_telemetry(
    const QkvBackwardDataParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  QkvBackwardKernelParams launch{};
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = prepare_qkv_backward_data_launch(
      params, &launch, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_qkv_backward_data_telemetry_impl(
      launch, timeline, timeline_capacity, stream, sm_count, device);
}
#endif

cudaError_t launch_qkv_backward_weight(
    const QkvBackwardWeightParams& params,
    cudaStream_t stream) {
  if (params.q_heads <= 0 || params.kv_heads <= 0 ||
      params.head_dim <= 0 || params.q_heads % params.kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  const int32_t packed_width =
      (params.q_heads + 2 * params.kv_heads) * params.head_dim;
  return launch_backward_wgrad_impl(
      params.dqkv_staging,
      params.saved_input,
      params.grad_weight,
      packed_width,
      params.hidden,
      params.local_tokens,
      params.alpha,
      params.beta,
      stream,
      device_info.sm_count,
      device_info.device);
}

cudaError_t launch_qkv_backward(
    const QkvBackwardParams& params,
    cudaStream_t stream) {
  cudaError_t status = launch_qkv_backward_data(params.data, stream);
  if (status != cudaSuccess ||
      params.weight_mode == WeightGradientMode::kDeferred) {
    return status;
  }
  if (params.weight_mode != WeightGradientMode::kImmediate) {
    return cudaErrorInvalidValue;
  }
  QkvBackwardWeightParams weight = params.weight;
  if (!weight.dqkv_staging) {
    weight.dqkv_staging =
        params.data.peer_dqkv_staging[params.data.rank];
  }
  if (weight.local_tokens == 0) {
    weight.local_tokens = params.data.local_tokens;
  }
  if (weight.hidden == 0) {
    weight.hidden = params.data.hidden;
  }
  if (weight.q_heads == 0) {
    weight.q_heads = params.data.q_heads;
  }
  if (weight.kv_heads == 0) {
    weight.kv_heads = params.data.kv_heads;
  }
  if (weight.head_dim == 0) {
    weight.head_dim = params.data.head_dim;
  }
  if (weight.local_tokens != params.data.local_tokens ||
      weight.hidden != params.data.hidden ||
      weight.q_heads != params.data.q_heads ||
      weight.kv_heads != params.data.kv_heads ||
      weight.head_dim != params.data.head_dim) {
    return cudaErrorInvalidValue;
  }
  return launch_qkv_backward_weight(weight, stream);
}

namespace {

QkvBackwardDataParams fp8_qkv_backward_shape_view(
    const Fp8QkvBackwardDataParams& params) {
  QkvBackwardDataParams view{};
  view.local_tokens = params.local_tokens;
  view.hidden = params.hidden;
  view.batch = params.batch;
  view.q_heads = params.q_heads;
  view.kv_heads = params.kv_heads;
  view.head_dim = params.head_dim;
  view.world_size = params.world_size;
  view.rank = params.rank;
  view.num_comm_ctas = params.num_comm_ctas;
  view.gemm_policy = params.gemm_policy;
  view.epoch = params.epoch;
  view.causal_load_balanced = params.causal_load_balanced;
  view.alpha = params.alpha;
  return view;
}

cudaError_t prepare_qkv_backward_fp8_data_launch(
    const Fp8QkvBackwardDataParams& params,
    Fp8QkvBackwardKernelParams* launch,
    int32_t* sm_count,
    int32_t* device) {
  if (launch == nullptr || sm_count == nullptr || device == nullptr ||
      !params.grad_q || !params.grad_k || !params.grad_v ||
      !params.weight_nt ||
      !params.grad_input || params.local_tokens <= 0 || params.hidden <= 0 ||
      params.batch <= 0 || params.local_tokens % params.batch != 0 ||
      params.q_heads <= 0 || params.kv_heads <= 0 ||
      params.head_dim <= 0 || params.head_dim % kFp8Alignment != 0 ||
      params.world_size <= 1 || params.world_size > kMaxWorldSize ||
      params.rank < 0 || params.rank >= params.world_size ||
      params.q_heads % params.kv_heads != 0 ||
      params.q_heads % params.world_size != 0 ||
      params.kv_heads % params.world_size != 0 || params.epoch == 0 ||
      (params.gemm_policy != BackwardGemmPolicy::kAuto &&
       params.gemm_policy != BackwardGemmPolicy::kM128N128) ||
      (params.causal_load_balanced &&
       (params.local_tokens / params.batch) % 2 != 0)) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  const int32_t comm_ctas = params.num_comm_ctas == 0
      ? recommended_qkv_backward_comm_ctas(
            fp8_qkv_backward_shape_view(params))
      : params.num_comm_ctas;
  if (comm_ctas <= 0 || comm_ctas >= device_info.sm_count ||
      comm_ctas % 2 != 0) {
    return cudaErrorInvalidConfiguration;
  }

  *launch = {};
  launch->local_q = params.grad_q;
  launch->local_k = params.grad_k;
  launch->local_v = params.grad_v;
  for (int32_t peer = 0; peer < params.world_size; ++peer) {
    launch->peer_staging[peer] = params.peer_dqkv_staging[peer];
    launch->peer_ready[peer] = params.peer_ready[peer];
    launch->peer_done_epoch[peer] = params.peer_done_epoch[peer];
  }
  launch->weight = params.weight_nt;
  launch->grad_input = params.grad_input;
  const int32_t packed_width =
      (params.q_heads + 2 * params.kv_heads) * params.head_dim;
  launch->gemm = {
      params.local_tokens, params.hidden, packed_width, 1};
  launch->gemm.input_dtype = DType::kFloat8E4M3;
  launch->gemm.weight_dtype = DType::kFloat8E4M3;
  launch->gemm.output_dtype = DType::kFloat8E4M3;
  launch->gemm.transpose_b = false;
  launch->gemm.stride_b.row = packed_width;
  launch->gemm.raster = GemmRaster::kAlongN;
  launch->gemm.max_swizzle_size = 1;
  launch->route.world_size = params.world_size;
  launch->route.rank = params.rank;
  launch->route.batch = params.batch;
  launch->route.seq_local = params.local_tokens / params.batch;
  launch->route.global_seq = launch->route.seq_local * params.world_size;
  launch->route.q_heads = params.q_heads;
  launch->route.kv_heads = params.kv_heads;
  launch->route.local_heads = params.q_heads / params.world_size;
  launch->route.head_dim = params.head_dim;
  launch->route.causal_load_balanced = params.causal_load_balanced;
  launch->route.kind = RouteKind::kQkvGqaPack;
  launch->route.direction = RouteDirection::kInverse;
  launch->num_comm_ctas = comm_ctas;
  launch->gemm_policy = BackwardGemmPolicy::kM128N128;
  launch->epoch = params.epoch;
  launch->alpha = params.alpha;
  *sm_count = device_info.sm_count;
  *device = device_info.device;
  return cudaSuccess;
}

}  // namespace

int64_t qkv_backward_fp8_ready_elements(
    const Fp8QkvBackwardDataParams& params) {
  if (params.local_tokens <= 0 || params.q_heads <= 0 ||
      params.kv_heads <= 0) {
    return 0;
  }
  return static_cast<int64_t>(ceil_div(params.local_tokens, kBlockM)) *
      (params.q_heads + 2LL * params.kv_heads) * kReadyFlagStride;
}

int32_t recommended_qkv_backward_fp8_comm_ctas(
    const Fp8QkvBackwardDataParams& params) {
  return recommended_qkv_backward_comm_ctas(
      fp8_qkv_backward_shape_view(params));
}

KernelTraits qkv_backward_fp8_kernel_traits(
    const Fp8QkvBackwardDataParams&,
    int32_t,
    int32_t) {
  return {
      128,
      128,
      128,
      static_cast<int32_t>(Fp8BackwardReadyGemm::get_block_shape().x),
      static_cast<int32_t>(
          sizeof(typename Fp8QkvBackwardDataKernel::SharedStorage))};
}

cudaError_t launch_qkv_backward_fp8_data(
    const Fp8QkvBackwardDataParams& params,
    cudaStream_t stream) {
  Fp8QkvBackwardKernelParams launch{};
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = prepare_qkv_backward_fp8_data_launch(
      params, &launch, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_qkv_backward_fp8_data_impl(
      launch, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_qkv_backward_fp8_data_role_telemetry(
    const Fp8QkvBackwardDataParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  Fp8QkvBackwardKernelParams launch{};
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = prepare_qkv_backward_fp8_data_launch(
      params, &launch, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_qkv_backward_fp8_data_telemetry_impl(
      launch,
      timeline,
      timeline_capacity,
      stream,
      sm_count,
      device);
}
#endif

cudaError_t launch_qkv_backward_fp8_weight(
    const Fp8QkvBackwardWeightParams& params,
    cudaStream_t stream) {
  if (params.q_heads <= 0 || params.kv_heads <= 0 ||
      params.head_dim <= 0 || params.q_heads % params.kv_heads != 0) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  const int32_t packed_width =
      (params.q_heads + 2 * params.kv_heads) * params.head_dim;
  return launch_backward_fp8_wgrad_impl(
      params.dqkv_t,
      params.saved_input_t,
      params.grad_weight,
      packed_width,
      params.hidden,
      params.local_tokens,
      params.alpha,
      params.beta,
      stream,
      device_info.sm_count,
      device_info.device);
}

cudaError_t launch_qkv_backward_fp8(
    const Fp8QkvBackwardParams& params,
    cudaStream_t stream) {
  cudaError_t status = launch_qkv_backward_fp8_data(params.data, stream);
  if (status != cudaSuccess ||
      params.weight_mode == WeightGradientMode::kDeferred) {
    return status;
  }
  if (params.weight_mode != WeightGradientMode::kImmediate) {
    return cudaErrorInvalidValue;
  }
  Fp8QkvBackwardWeightParams weight = params.weight;
  if (!weight.dqkv_t || !weight.saved_input_t) {
    return cudaErrorInvalidValue;
  }
  if (weight.local_tokens == 0) weight.local_tokens = params.data.local_tokens;
  if (weight.hidden == 0) weight.hidden = params.data.hidden;
  if (weight.q_heads == 0) weight.q_heads = params.data.q_heads;
  if (weight.kv_heads == 0) weight.kv_heads = params.data.kv_heads;
  if (weight.head_dim == 0) weight.head_dim = params.data.head_dim;
  if (weight.local_tokens != params.data.local_tokens ||
      weight.hidden != params.data.hidden ||
      weight.q_heads != params.data.q_heads ||
      weight.kv_heads != params.data.kv_heads ||
      weight.head_dim != params.data.head_dim) {
    return cudaErrorInvalidValue;
  }
  return launch_qkv_backward_fp8_weight(weight, stream);
}

int64_t oproj_backward_ready_elements(
    const OprojBackwardDataParams& params) {
  if (params.local_tokens <= 0 || params.q_heads <= 0 ||
      params.head_dim <= 0) {
    return 0;
  }
  const int32_t attention_width = params.q_heads * params.head_dim;
  BackwardDeviceInfo device_info{};
  if (cached_backward_device_info(&device_info) != cudaSuccess) {
    return 0;
  }
  const int32_t comm_ctas = params.num_comm_ctas == 0
      ? recommended_oproj_backward_comm_ctas(params)
      : params.num_comm_ctas;
  const KernelTraits traits = oproj_backward_kernel_traits(
      params, comm_ctas, device_info.sm_count);
  if (traits.block_m <= 0 || traits.block_n <= 0) {
    return 0;
  }
  return static_cast<int64_t>(
      ceil_div(params.local_tokens, traits.block_m)) *
      ceil_div(attention_width, traits.block_n) * kReadyFlagStride;
}

namespace {

int32_t recommended_oproj_backward_comm_ctas_impl(
    const OprojBackwardDataParams& params,
    int32_t sm_count) {
  if (params.local_tokens <= 0 || params.hidden <= 0 || params.batch <= 0 ||
      params.q_heads <= 0 || params.head_dim <= 0 ||
      params.world_size <= 1 || params.world_size > kMaxWorldSize ||
      params.q_heads % params.world_size != 0 || sm_count <= 2) {
    return 0;
  }

  // OProj backward publishes one 64-row x head_dim route task per logical
  // head.  Ten route CTAs reach the useful H200 issuer bandwidth.  A short
  // GEMM may spend otherwise-idle SMs to make the whole route a single wave;
  // a multi-wave GEMM keeps only the saturation floor so communication does
  // not take workers away from sustained matrix math.
  constexpr int32_t kRouteRows = 64;
  constexpr int32_t kRouteSlotsPerCta = 12;
  constexpr int32_t kRouteIssuerSaturation = 10;
  constexpr int32_t kMaxCommCtas = 32;
  const int64_t route_tasks =
      ceil_div(params.local_tokens, kRouteRows) *
      static_cast<int64_t>(params.q_heads) *
      ceil_div(params.head_dim, kQkvBulkColumns);
  const int32_t issuer_floor = static_cast<int32_t>(
      min<int64_t>(kRouteIssuerSaturation, route_tasks));
  int32_t desired = max(
      issuer_floor,
      static_cast<int32_t>(ceil_div(route_tasks, kRouteSlotsPerCta)));
  desired = (desired + 1) & ~1;
  const int32_t max_comm = min(kMaxCommCtas, (sm_count - 2) & ~1);
  desired = min(desired, max_comm);
  if (desired <= 0) {
    return 0;
  }

  if (desired > kRouteIssuerSaturation) {
    const BackwardGemmPolicy policy =
        recommended_oproj_backward_gemm_policy(params, desired, sm_count);
    const KernelTraits traits = oproj_backward_traits_for_policy(policy);
    if (traits.block_m <= 0 || traits.block_n <= 0) {
      return 0;
    }
    int64_t physical_m_tiles = ceil_div(params.local_tokens, traits.block_m);
    if (policy == BackwardGemmPolicy::kM128N256) {
      physical_m_tiles = ceil_div(physical_m_tiles, int64_t{2}) * 2;
    }
    const int64_t attention_width =
        static_cast<int64_t>(params.q_heads) * params.head_dim;
    const int64_t compute_ctas =
        physical_m_tiles * ceil_div(attention_width, traits.block_n);
    if (compute_ctas > sm_count - desired) {
      desired = min(max_comm, kRouteIssuerSaturation);
    }
  }
  return desired;
}

struct OprojBackwardLaunchPlanKey {
  std::array<uint64_t, 11> words{};

  bool operator==(const OprojBackwardLaunchPlanKey& other) const {
    return words == other.words;
  }
};

using OprojBackwardLaunchPlanKeyHash =
    WordArrayHash<OprojBackwardLaunchPlanKey>;

OprojBackwardLaunchPlanKey make_oproj_backward_launch_plan_key(
    const OprojBackwardDataParams& params,
    int32_t device) {
  return {{
      static_cast<uint64_t>(device),
      static_cast<uint64_t>(params.num_comm_ctas),
      static_cast<uint64_t>(params.gemm_policy),
      static_cast<uint64_t>(params.local_tokens),
      static_cast<uint64_t>(params.hidden),
      static_cast<uint64_t>(params.batch),
      static_cast<uint64_t>(params.q_heads),
      static_cast<uint64_t>(params.head_dim),
      static_cast<uint64_t>(params.world_size),
      static_cast<uint64_t>(params.rank),
      static_cast<uint64_t>(params.causal_load_balanced),
  }};
}

struct OprojBackwardLaunchPlan {
  int32_t device = 0;
  int32_t sm_count = 0;
  int32_t num_comm_ctas = 0;
  BackwardGemmPolicy policy = BackwardGemmPolicy::kM128N128;
};

using OprojBackwardLaunchPlanCache = BoundedLaunchCache<
    OprojBackwardLaunchPlanKey,
    OprojBackwardLaunchPlan,
    OprojBackwardLaunchPlanKeyHash>;

cudaError_t cached_oproj_backward_launch_plan(
    const OprojBackwardDataParams& params,
    OprojBackwardLaunchPlan* result) {
  if (result == nullptr) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  const OprojBackwardLaunchPlanKey key =
      make_oproj_backward_launch_plan_key(params, device_info.device);
  static OprojBackwardLaunchPlanCache cache;
  auto create = [&](OprojBackwardLaunchPlan* plan) {
    plan->device = device_info.device;
    plan->sm_count = device_info.sm_count;
    plan->num_comm_ctas = params.num_comm_ctas == 0
        ? recommended_oproj_backward_comm_ctas_impl(params, plan->sm_count)
        : params.num_comm_ctas;
    if (plan->num_comm_ctas <= 0 ||
        plan->num_comm_ctas >= plan->sm_count) {
      return cudaErrorInvalidConfiguration;
    }
    const int32_t attention_width = params.q_heads * params.head_dim;
    plan->policy = select_backward_gemm_policy(
        params.gemm_policy,
        params.local_tokens,
        attention_width,
        params.hidden,
        plan->num_comm_ctas,
        plan->sm_count);
    return cudaSuccess;
  };
  return cache.get_or_create(key, result, create);
}

}  // namespace

int32_t recommended_oproj_backward_comm_ctas(
    const OprojBackwardDataParams& params) {
  OprojBackwardDataParams automatic = params;
  automatic.num_comm_ctas = 0;
  OprojBackwardLaunchPlan plan{};
  if (cached_oproj_backward_launch_plan(automatic, &plan) != cudaSuccess) {
    return 0;
  }
  return plan.num_comm_ctas;
}

BackwardGemmPolicy recommended_oproj_backward_gemm_policy(
    const OprojBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count) {
  const int32_t attention_width = params.q_heads * params.head_dim;
  return select_backward_gemm_policy(
      params.gemm_policy,
      params.local_tokens,
      attention_width,
      params.hidden,
      resolved_comm_ctas,
      sm_count);
}

KernelTraits oproj_backward_kernel_traits(
    const OprojBackwardDataParams& params,
    int32_t resolved_comm_ctas,
    int32_t sm_count) {
  return oproj_backward_traits_for_policy(
      recommended_oproj_backward_gemm_policy(
          params, resolved_comm_ctas, sm_count));
}

cudaError_t launch_oproj_backward_data(
    const OprojBackwardDataParams& params,
    cudaStream_t stream) {
  if (!params.grad_output || !params.weight ||
      !params.local_grad_attention || !params.ready ||
      params.local_tokens <= 0 || params.hidden <= 0 || params.batch <= 0 ||
      params.local_tokens % params.batch != 0 || params.q_heads <= 0 ||
      params.head_dim <= 0 || params.head_dim % kAlignment != 0 ||
      params.world_size <= 1 || params.world_size > kMaxWorldSize ||
      params.rank < 0 || params.rank >= params.world_size ||
      params.q_heads % params.world_size != 0 || params.epoch == 0 ||
      (params.causal_load_balanced &&
       (params.local_tokens / params.batch) % 2 != 0)) {
    return cudaErrorInvalidValue;
  }
  OprojBackwardLaunchPlan plan{};
  cudaError_t status = cached_oproj_backward_launch_plan(params, &plan);
  if (status != cudaSuccess) {
    return status;
  }
  OprojBackwardKernelParams launch{};
  launch.lhs = params.grad_output;
  launch.rhs_nt = params.weight;
  launch.local_output = params.local_grad_attention;
  for (int32_t peer = 0; peer < params.world_size; ++peer) {
    launch.peer_output[peer] = params.peer_grad_attention[peer];
    launch.peer_route_done_epoch[peer] = params.peer_done_epoch[peer];
  }
  launch.ready = params.ready;
  const int32_t attention_width = params.q_heads * params.head_dim;
  launch.gemm = {
      params.local_tokens, attention_width, params.hidden, 1};
  launch.gemm.transpose_b = true;
  launch.gemm.stride_b.row = attention_width;
  launch.gemm.raster = GemmRaster::kAlongN;
  launch.gemm.max_swizzle_size = 1;
  launch.route.world_size = params.world_size;
  launch.route.rank = params.rank;
  launch.route.batch = params.batch;
  launch.route.seq_local = params.local_tokens / params.batch;
  launch.route.global_seq =
      launch.route.seq_local * params.world_size;
  launch.route.q_heads = params.q_heads;
  launch.route.kv_heads = 0;
  launch.route.local_heads = params.q_heads / params.world_size;
  launch.route.head_dim = params.head_dim;
  launch.route.causal_load_balanced = params.causal_load_balanced;
  launch.route.kind = RouteKind::kHeadToSequence;
  launch.route.direction = RouteDirection::kForward;
  launch.num_comm_ctas = plan.num_comm_ctas;
  launch.epoch = params.epoch;
  launch.alpha = params.alpha;

  if (launch.num_comm_ctas <= 0 || launch.num_comm_ctas >= plan.sm_count ||
      launch.num_comm_ctas % 2 != 0) {
    return cudaErrorInvalidConfiguration;
  }
  return launch_oproj_backward_data_impl(
      launch, plan.policy, stream, plan.sm_count, plan.device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_oproj_backward_data_role_telemetry(
    const OprojBackwardDataParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  if (!params.grad_output || !params.weight ||
      !params.local_grad_attention || !params.ready ||
      params.local_tokens <= 0 || params.hidden <= 0 || params.batch <= 0 ||
      params.local_tokens % params.batch != 0 || params.q_heads <= 0 ||
      params.head_dim <= 0 || params.head_dim % kAlignment != 0 ||
      params.world_size <= 1 || params.world_size > kMaxWorldSize ||
      params.rank < 0 || params.rank >= params.world_size ||
      params.q_heads % params.world_size != 0 || params.epoch == 0 ||
      (params.causal_load_balanced &&
       (params.local_tokens / params.batch) % 2 != 0)) {
    return cudaErrorInvalidValue;
  }
  OprojBackwardLaunchPlan plan{};
  cudaError_t status = cached_oproj_backward_launch_plan(params, &plan);
  if (status != cudaSuccess) {
    return status;
  }
  OprojBackwardKernelParams launch{};
  launch.lhs = params.grad_output;
  launch.rhs_nt = params.weight;
  launch.local_output = params.local_grad_attention;
  for (int32_t peer = 0; peer < params.world_size; ++peer) {
    launch.peer_output[peer] = params.peer_grad_attention[peer];
    launch.peer_route_done_epoch[peer] = params.peer_done_epoch[peer];
  }
  launch.ready = params.ready;
  const int32_t attention_width = params.q_heads * params.head_dim;
  launch.gemm = {
      params.local_tokens, attention_width, params.hidden, 1};
  launch.gemm.transpose_b = true;
  launch.gemm.stride_b.row = attention_width;
  launch.gemm.raster = GemmRaster::kAlongN;
  launch.gemm.max_swizzle_size = 1;
  launch.route.world_size = params.world_size;
  launch.route.rank = params.rank;
  launch.route.batch = params.batch;
  launch.route.seq_local = params.local_tokens / params.batch;
  launch.route.global_seq = launch.route.seq_local * params.world_size;
  launch.route.q_heads = params.q_heads;
  launch.route.local_heads = params.q_heads / params.world_size;
  launch.route.head_dim = params.head_dim;
  launch.route.causal_load_balanced = params.causal_load_balanced;
  launch.route.kind = RouteKind::kHeadToSequence;
  launch.route.direction = RouteDirection::kForward;
  launch.num_comm_ctas = plan.num_comm_ctas;
  launch.epoch = params.epoch;
  launch.alpha = params.alpha;
  if (launch.num_comm_ctas <= 0 || launch.num_comm_ctas >= plan.sm_count ||
      launch.num_comm_ctas % 2 != 0) {
    return cudaErrorInvalidConfiguration;
  }
  return launch_oproj_backward_data_telemetry_impl(
      launch,
      plan.policy,
      timeline,
      timeline_capacity,
      stream,
      plan.sm_count,
      plan.device);
}
#endif

cudaError_t launch_oproj_backward_weight(
    const OprojBackwardWeightParams& params,
    cudaStream_t stream) {
  if (params.q_heads <= 0 || params.head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_backward_wgrad_impl(
      params.grad_output,
      params.saved_attention,
      params.grad_weight,
      params.hidden,
      params.q_heads * params.head_dim,
      params.local_tokens,
      params.alpha,
      params.beta,
      stream,
      device_info.sm_count,
      device_info.device);
}

cudaError_t launch_oproj_backward(
    const OprojBackwardParams& params,
    cudaStream_t stream) {
  cudaError_t status = launch_oproj_backward_data(params.data, stream);
  if (status != cudaSuccess ||
      params.weight_mode == WeightGradientMode::kDeferred) {
    return status;
  }
  if (params.weight_mode != WeightGradientMode::kImmediate) {
    return cudaErrorInvalidValue;
  }
  OprojBackwardWeightParams weight = params.weight;
  if (!weight.grad_output) {
    weight.grad_output = params.data.grad_output;
  }
  if (weight.local_tokens == 0) {
    weight.local_tokens = params.data.local_tokens;
  }
  if (weight.hidden == 0) {
    weight.hidden = params.data.hidden;
  }
  if (weight.q_heads == 0) {
    weight.q_heads = params.data.q_heads;
  }
  if (weight.head_dim == 0) {
    weight.head_dim = params.data.head_dim;
  }
  if (weight.local_tokens != params.data.local_tokens ||
      weight.hidden != params.data.hidden ||
      weight.q_heads != params.data.q_heads ||
      weight.head_dim != params.data.head_dim) {
    return cudaErrorInvalidValue;
  }
  return launch_oproj_backward_weight(weight, stream);
}

namespace {

OprojBackwardDataParams fp8_oproj_backward_shape_view(
    const Fp8OprojBackwardDataParams& params) {
  OprojBackwardDataParams view{};
  view.local_tokens = params.local_tokens;
  view.hidden = params.hidden;
  view.batch = params.batch;
  view.q_heads = params.q_heads;
  view.head_dim = params.head_dim;
  view.world_size = params.world_size;
  view.rank = params.rank;
  view.num_comm_ctas = params.num_comm_ctas;
  view.gemm_policy = params.gemm_policy;
  view.epoch = params.epoch;
  view.causal_load_balanced = params.causal_load_balanced;
  view.alpha = params.alpha;
  return view;
}

cudaError_t prepare_oproj_backward_fp8_data_launch(
    const Fp8OprojBackwardDataParams& params,
    Fp8OprojBackwardKernelParams* launch,
    int32_t* sm_count,
    int32_t* device) {
  if (launch == nullptr || sm_count == nullptr || device == nullptr ||
      !params.grad_output || !params.weight_nt ||
      !params.local_grad_attention || !params.ready ||
      params.local_tokens <= 0 || params.hidden <= 0 || params.batch <= 0 ||
      params.local_tokens % params.batch != 0 || params.q_heads <= 0 ||
      params.head_dim <= 0 || params.head_dim % kFp8Alignment != 0 ||
      params.world_size <= 1 || params.world_size > kMaxWorldSize ||
      params.rank < 0 || params.rank >= params.world_size ||
      params.q_heads % params.world_size != 0 || params.epoch == 0 ||
      (params.gemm_policy != BackwardGemmPolicy::kAuto &&
       params.gemm_policy != BackwardGemmPolicy::kM128N128) ||
      (params.causal_load_balanced &&
       (params.local_tokens / params.batch) % 2 != 0)) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  const int32_t comm_ctas = params.num_comm_ctas == 0
      ? recommended_oproj_backward_comm_ctas(
            fp8_oproj_backward_shape_view(params))
      : params.num_comm_ctas;
  if (comm_ctas <= 0 || comm_ctas >= device_info.sm_count ||
      comm_ctas % 2 != 0) {
    return cudaErrorInvalidConfiguration;
  }

  *launch = {};
  launch->lhs = params.grad_output;
  launch->rhs_nt = params.weight_nt;
  launch->local_output = params.local_grad_attention;
  for (int32_t peer = 0; peer < params.world_size; ++peer) {
    launch->peer_output[peer] = params.peer_grad_attention[peer];
    launch->peer_route_done_epoch[peer] = params.peer_done_epoch[peer];
  }
  launch->ready = params.ready;
  const int32_t attention_width = params.q_heads * params.head_dim;
  launch->gemm = {
      params.local_tokens, attention_width, params.hidden, 1};
  launch->gemm.input_dtype = DType::kFloat8E4M3;
  launch->gemm.weight_dtype = DType::kFloat8E4M3;
  launch->gemm.output_dtype = DType::kFloat8E4M3;
  launch->gemm.transpose_b = false;
  launch->gemm.stride_b.row = params.hidden;
  launch->gemm.raster = GemmRaster::kAlongN;
  launch->gemm.max_swizzle_size = 1;
  launch->route.world_size = params.world_size;
  launch->route.rank = params.rank;
  launch->route.batch = params.batch;
  launch->route.seq_local = params.local_tokens / params.batch;
  launch->route.global_seq = launch->route.seq_local * params.world_size;
  launch->route.q_heads = params.q_heads;
  launch->route.kv_heads = 0;
  launch->route.local_heads = params.q_heads / params.world_size;
  launch->route.head_dim = params.head_dim;
  launch->route.causal_load_balanced = params.causal_load_balanced;
  launch->route.kind = RouteKind::kHeadToSequence;
  launch->route.direction = RouteDirection::kForward;
  launch->num_comm_ctas = comm_ctas;
  launch->epoch = params.epoch;
  launch->alpha = params.alpha;
  *sm_count = device_info.sm_count;
  *device = device_info.device;
  return cudaSuccess;
}

}  // namespace

int32_t recommended_oproj_backward_fp8_comm_ctas(
    const Fp8OprojBackwardDataParams& params) {
  return recommended_oproj_backward_comm_ctas(
      fp8_oproj_backward_shape_view(params));
}

KernelTraits oproj_backward_fp8_kernel_traits(
    const Fp8OprojBackwardDataParams&,
    int32_t,
    int32_t) {
  return {
      128,
      128,
      128,
      static_cast<int32_t>(Fp8BackwardSignalingGemm::get_block_shape().x),
      static_cast<int32_t>(
          sizeof(typename Fp8OprojBackwardDataKernel::SharedStorage))};
}

int64_t oproj_backward_fp8_ready_elements(
    const Fp8OprojBackwardDataParams& params) {
  if (params.local_tokens <= 0 || params.q_heads <= 0 ||
      params.head_dim <= 0) {
    return 0;
  }
  const int32_t attention_width = params.q_heads * params.head_dim;
  return static_cast<int64_t>(ceil_div(params.local_tokens, kBlockM)) *
      ceil_div(attention_width, kBlockN) * kReadyFlagStride;
}

cudaError_t launch_oproj_backward_fp8_data(
    const Fp8OprojBackwardDataParams& params,
    cudaStream_t stream) {
  Fp8OprojBackwardKernelParams launch{};
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = prepare_oproj_backward_fp8_data_launch(
      params, &launch, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_oproj_backward_fp8_data_impl(
      launch, stream, sm_count, device);
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_oproj_backward_fp8_data_role_telemetry(
    const Fp8OprojBackwardDataParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  Fp8OprojBackwardKernelParams launch{};
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = prepare_oproj_backward_fp8_data_launch(
      params, &launch, &sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_oproj_backward_fp8_data_telemetry_impl(
      launch,
      timeline,
      timeline_capacity,
      stream,
      sm_count,
      device);
}
#endif

cudaError_t launch_oproj_backward_fp8_weight(
    const Fp8OprojBackwardWeightParams& params,
    cudaStream_t stream) {
  if (params.q_heads <= 0 || params.head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  BackwardDeviceInfo device_info{};
  cudaError_t status = cached_backward_device_info(&device_info);
  if (status != cudaSuccess) {
    return status;
  }
  return launch_backward_fp8_wgrad_impl(
      params.grad_output_t,
      params.saved_attention_t,
      params.grad_weight,
      params.hidden,
      params.q_heads * params.head_dim,
      params.local_tokens,
      params.alpha,
      params.beta,
      stream,
      device_info.sm_count,
      device_info.device);
}

cudaError_t launch_oproj_backward_fp8(
    const Fp8OprojBackwardParams& params,
    cudaStream_t stream) {
  cudaError_t status = launch_oproj_backward_fp8_data(params.data, stream);
  if (status != cudaSuccess ||
      params.weight_mode == WeightGradientMode::kDeferred) {
    return status;
  }
  if (params.weight_mode != WeightGradientMode::kImmediate) {
    return cudaErrorInvalidValue;
  }
  Fp8OprojBackwardWeightParams weight = params.weight;
  if (!weight.grad_output_t || !weight.saved_attention_t) {
    return cudaErrorInvalidValue;
  }
  if (weight.local_tokens == 0) weight.local_tokens = params.data.local_tokens;
  if (weight.hidden == 0) weight.hidden = params.data.hidden;
  if (weight.q_heads == 0) weight.q_heads = params.data.q_heads;
  if (weight.head_dim == 0) weight.head_dim = params.data.head_dim;
  if (weight.local_tokens != params.data.local_tokens ||
      weight.hidden != params.data.hidden ||
      weight.q_heads != params.data.q_heads ||
      weight.head_dim != params.data.head_dim) {
    return cudaErrorInvalidValue;
  }
  return launch_oproj_backward_fp8_weight(weight, stream);
}

}  // namespace fuse
