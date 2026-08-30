// SPDX-License-Identifier: BSD-3-Clause
// Public API implementation; assembled by csrc/operators/ulysses_sm90.cu.
// Declarations: fuse/operators/primitives/a2a_gemm.h and gemm_a2a.h.
//
// Module index:
//   - resolved BF16/FP8 kernel traits and ready-storage sizing
//   - H200/H800 compute, route, and NVLink cost functions
//   - A2A-then-GEMM and GEMM-then-A2A communication-CTA selection

namespace fuse {

KernelTraits qkv_cutlass_kernel_traits(const GemmProblem& problem) {
  (void)problem;
  // This overload has no route, communication split or SM count, so it
  // cannot know the auto-selected kernel. Return the finest real geometry as
  // a conservative ready/workspace sizing contract. The route-aware overload
  // below reports the actual selected kernel.
  return {
      128,
      64,
      64,
      static_cast<int32_t>(N64OutputGemm::get_block_shape().x),
      static_cast<int32_t>(
          sizeof(typename QkvGemmA2AKernelN64::SharedStorage))};
}

KernelTraits qkv_cutlass_kernel_traits(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count) {
  const QkvGemmPolicy policy = select_qkv_gemm_policy(
      problem,
      num_comm_ctas,
      sm_count,
      route.qkv_peer_interleaved);
  KernelTraits traits{};
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    if constexpr (
        std::is_same_v<Binding, QkvForwardN128Binding> ||
        std::is_same_v<Binding, QkvForwardN128InterleavedBinding>) {
      traits = cutlass_kernel_traits();
    } else if constexpr (std::is_same_v<Binding, QkvForwardN256Binding>) {
      traits = projection_cutlass_kernel_traits();
    } else {
      traits = {
          static_cast<int32_t>(
              cute::size<0>(typename Binding::Gemm::TileShape{})),
          static_cast<int32_t>(
              cute::size<1>(typename Binding::Gemm::TileShape{})),
          static_cast<int32_t>(
              cute::size<2>(typename Binding::Gemm::TileShape{})),
          static_cast<int32_t>(Binding::Gemm::get_block_shape().x),
          static_cast<int32_t>(
              sizeof(typename Binding::Kernel::SharedStorage))};
    }
    return cudaSuccess;
  };
  return visit_qkv_forward_policy(
             policy, route.qkv_peer_interleaved, read_traits) == cudaSuccess
      ? traits
      : cutlass_kernel_traits();
}

KernelTraits fp8_cutlass_kernel_traits() {
  // N64 is both the finest ready-flag geometry and the largest production
  // shared-storage footprint, so it safely sizes callers that do not yet know
  // the resolved route/communication plan.
  return {
      128,
      64,
      128,
      static_cast<int32_t>(Fp8N64OutputGemm::get_block_shape().x),
      static_cast<int32_t>(
          sizeof(typename Fp8N64GemmA2AKernel::SharedStorage))};
}

KernelTraits fp8_cutlass_kernel_traits(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count) {
  int32_t device = 0;
  if (num_comm_ctas <= 0 || cudaGetDevice(&device) != cudaSuccess) {
    return fp8_cutlass_kernel_traits();
  }
  const Fp8TilePolicy tile = select_fp8_qkv_tile(
      problem, route, num_comm_ctas, sm_count, device);
  if (tile == Fp8TilePolicy::kM128N64) {
    return fp8_cutlass_kernel_traits();
  }
  if (tile == Fp8TilePolicy::kM128N256ClusterM2) {
    return {
        128,
        256,
        128,
        static_cast<int32_t>(Fp8WideOutputGemm::get_block_shape().x),
        static_cast<int32_t>(
            sizeof(typename Fp8WideGemmA2AKernel::SharedStorage))};
  }
  return {
      kBlockM,
      kBlockN,
      128,
      static_cast<int32_t>(Fp8OutputGemm::get_block_shape().x),
      static_cast<int32_t>(std::max(
          sizeof(typename Fp8GemmA2AKernel::SharedStorage),
          sizeof(typename Fp8CooperativeGemmA2AKernel::SharedStorage)))};
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

namespace {

enum class QkvCommPolicyRequest : uint64_t {
  kLegacy,
  kPipeline,
};

QkvCommPolicyRequest normalized_qkv_comm_policy_request() {
  const char* value = std::getenv("FUSE_QKV_COMM_POLICY");
  return value == nullptr || std::strcmp(value, "pipeline") == 0 ||
          std::strcmp(value, "roofline") == 0
      ? QkvCommPolicyRequest::kPipeline
      : QkvCommPolicyRequest::kLegacy;
}

struct QkvNvlinkOverride {
  bool present = false;
  uint64_t bits = 0;
};

QkvNvlinkOverride normalized_qkv_nvlink_override() {
  QkvNvlinkOverride result{};
  const char* value = std::getenv("FUSE_NVLINK_BIDIR_GBPS");
  if (value == nullptr) {
    return result;
  }
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end != value && parsed > 0.0) {
    result.present = true;
    static_assert(sizeof(result.bits) == sizeof(parsed));
    std::memcpy(&result.bits, &parsed, sizeof(parsed));
  }
  return result;
}

}  // namespace

double a2a_lhs_nvlink_bidirectional_gbps(int32_t device) {
  const auto override = normalized_qkv_nvlink_override();
  if (override.present) {
    double parsed = 0.0;
    std::memcpy(&parsed, &override.bits, sizeof(parsed));
    return parsed;
  }

  // Device identity is immutable for a process.  In particular, do not put
  // cudaGetDeviceProperties on every eager-launch policy path: on H800 it can
  // synchronize with outstanding CUDA work.  The environment override above
  // intentionally remains dynamic and therefore bypasses this cache.
  struct DeviceBandwidthCache {
    std::mutex mutex;
    std::unordered_map<int32_t, double> values;
  };
  static DeviceBandwidthCache cache;
  thread_local bool has_last_device = false;
  thread_local int32_t last_device = -1;
  thread_local double last_value = 0.0;
  if (has_last_device && last_device == device) {
    return last_value;
  }

  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto cached = cache.values.find(device);
  if (cached != cache.values.end()) {
    has_last_device = true;
    last_device = device;
    last_value = cached->second;
    return last_value;
  }

  cudaDeviceProp properties{};
  const cudaError_t status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    // Preserve the previous conservative fallback, but do not cache a
    // transient runtime failure.
    return 900.0;
  }
  const double bandwidth = std::strstr(properties.name, "H800") != nullptr
      ? 400.0
      : 900.0;
  cache.values.emplace(device, bandwidth);
  has_last_device = true;
  last_device = device;
  last_value = bandwidth;
  return bandwidth;
}

namespace {

constexpr std::array<Fp8TilePolicy, 3> kFp8QkvTileCandidates{
    Fp8TilePolicy::kM128N64,
    Fp8TilePolicy::kM128N128,
    Fp8TilePolicy::kM128N256ClusterM2};

double fp8_qkv_pipeline_score_ns(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count,
    int32_t device,
    Fp8TilePolicy policy) {
  const auto compute = estimate_fp8_qkv_compute(
      problem, num_comm_ctas, sm_count, policy);
  if (!compute.valid || route.world_size <= 1 || route.head_dim <= 0 ||
      route.q_heads <= 0 || route.kv_heads <= 0 ||
      route.q_heads % route.world_size != 0 ||
      route.kv_heads % route.world_size != 0) {
    return std::numeric_limits<double>::infinity();
  }

  const int32_t q_local_heads = route.q_heads / route.world_size;
  const int32_t kv_local_heads = route.kv_heads / route.world_size;
  const int32_t routed_local_heads = q_local_heads +
      (route.defer_v_a2a ? 1 : 2) * kv_local_heads;
  const int64_t m_chunks = ceil_div(problem.m, kQkvBulkRows);
  const int64_t route_tasks =
      static_cast<int64_t>(route.world_size) * m_chunks *
      routed_local_heads;
  const int64_t route_slots =
      static_cast<int64_t>(num_comm_ctas) * kQkvBulkSlots;
  const int64_t route_waves = ceil_div(route_tasks, route_slots);
  const double route_task_ns =
      route_waves * static_cast<double>(kFp8QkvRouteTaskWaveNs);

  const int64_t routed_global_width =
      static_cast<int64_t>(route.q_heads +
          (route.defer_v_a2a ? 1 : 2) * route.kv_heads) *
      route.head_dim;
  const double remote_bytes =
      static_cast<double>(problem.m) * routed_global_width *
      sizeof(Fp8Element) * (route.world_size - 1) / route.world_size;
  const double one_way_nvlink_gbps =
      0.5 * a2a_lhs_nvlink_bidirectional_gbps(device);
  const double comm_fraction = std::min(
      1.0,
      static_cast<double>(num_comm_ctas) /
          kFp8QkvFabricSaturationCommCtas);
  if (!(one_way_nvlink_gbps > 0.0) || !(comm_fraction > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  const double route_link_ns =
      remote_bytes / (one_way_nvlink_gbps * comm_fraction);
  const double route_ns = std::max(route_task_ns, route_link_ns);
  // Producer and route run as one dependency pipeline. Route can consume an
  // early ready tile while later GEMM waves continue, leaving one measured
  // 16-KiB task wave as the irreducible drain after compute.
  const double exposed_route_ns = std::max(
      static_cast<double>(kFp8QkvRouteTaskWaveNs),
      route_ns - compute.total_ns);
  return compute.total_ns + exposed_route_ns;
}

int32_t fp8_tile_tie_rank(Fp8TilePolicy policy) {
  // When the calibrated model cannot distinguish two choices, retain N128:
  // it uses fewer ready flags than N64 and does not couple two CTAs like N256.
  if (policy == Fp8TilePolicy::kM128N128) {
    return 0;
  }
  return policy == Fp8TilePolicy::kM128N64 ? 1 : 2;
}

Fp8TilePolicy select_fp8_qkv_tile(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count,
    int32_t device) {
  Fp8TilePolicy requested{};
  if (requested_fp8_tile_override(&requested)) {
    return requested;
  }
  Fp8TilePolicy best = Fp8TilePolicy::kM128N128;
  double best_score_ns = std::numeric_limits<double>::infinity();
  int32_t best_tie_rank = fp8_tile_tie_rank(best);
  for (Fp8TilePolicy candidate : kFp8QkvTileCandidates) {
    const double score_ns = fp8_qkv_pipeline_score_ns(
        problem, route, num_comm_ctas, sm_count, device, candidate);
    const int32_t tie_rank = fp8_tile_tie_rank(candidate);
    constexpr double kTieToleranceNs = 1.0e-6;
    if (score_ns + kTieToleranceNs < best_score_ns ||
        (std::abs(score_ns - best_score_ns) <= kTieToleranceNs &&
         tie_rank < best_tie_rank)) {
      best = candidate;
      best_score_ns = score_ns;
      best_tie_rank = tie_rank;
    }
  }
  return best;
}

}  // namespace

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

namespace {

struct QkvCommRecommendationKey {
  // Keep every scalar problem/route input so future policy changes cannot
  // accidentally reuse an older result. Device pointers are deliberately
  // excluded: they do not affect the communication split.
  std::array<uint64_t, 42> words{};

  bool operator==(const QkvCommRecommendationKey& other) const {
    return words == other.words;
  }
};

using QkvCommRecommendationKeyHash =
    WordArrayHash<QkvCommRecommendationKey>;

QkvCommRecommendationKey make_qkv_comm_recommendation_key(
    const GemmProblem& p,
    const UlyssesRoute& r,
    int32_t device) {
  const auto comm_policy = normalized_qkv_comm_policy_request();
  const auto nvlink_override = normalized_qkv_nvlink_override();
  Fp8TilePolicy fp8_tile{};
  const bool has_fp8_tile_override = requested_fp8_tile_override(&fp8_tile);
  return {{
      static_cast<uint64_t>(device),
      static_cast<uint64_t>(p.m),
      static_cast<uint64_t>(p.n),
      static_cast<uint64_t>(p.k),
      static_cast<uint64_t>(p.l),
      static_cast<uint64_t>(p.stride_a.row),
      static_cast<uint64_t>(p.stride_a.column),
      static_cast<uint64_t>(p.stride_a.batch),
      static_cast<uint64_t>(p.stride_b.row),
      static_cast<uint64_t>(p.stride_b.column),
      static_cast<uint64_t>(p.stride_b.batch),
      static_cast<uint64_t>(p.stride_d.row),
      static_cast<uint64_t>(p.stride_d.column),
      static_cast<uint64_t>(p.stride_d.batch),
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
      static_cast<uint64_t>(nvlink_override.present),
      nvlink_override.bits,
      has_fp8_tile_override
          ? static_cast<uint64_t>(fp8_tile) + 1
          : 0,
  }};
}

using QkvCommRecommendationCache = BoundedLaunchCache<
    QkvCommRecommendationKey, int32_t, QkvCommRecommendationKeyHash>;

QkvCommRecommendationCache& qkv_comm_recommendation_cache() {
  static QkvCommRecommendationCache cache;
  return cache;
}

bool find_qkv_comm_recommendation(
    const QkvCommRecommendationKey& key,
    int32_t* comm_ctas) {
  return qkv_comm_recommendation_cache().find(key, comm_ctas);
}

void store_qkv_comm_recommendation(
    const QkvCommRecommendationKey& key,
    int32_t comm_ctas) {
  if (comm_ctas <= 0) {
    return;
  }
  qkv_comm_recommendation_cache().store(key, comm_ctas);
}

}  // namespace

int32_t recommended_gemm_a2a_comm_ctas(
  const GemmProblem& problem,
  const UlyssesRoute& route) {
  if (route.kind != RouteKind::kQkvGqaPack) {
    return 0;
  }
  const bool fp8_problem =
      problem.input_dtype == DType::kFloat8E4M3 &&
      problem.weight_dtype == DType::kFloat8E4M3 &&
      problem.output_dtype == DType::kFloat8E4M3;
  const bool bf16_problem =
      problem.input_dtype == DType::kBfloat16 &&
      problem.weight_dtype == DType::kBfloat16 &&
      problem.output_dtype == DType::kBfloat16;
  const int32_t output_element_bytes = fp8_problem
      ? static_cast<int32_t>(sizeof(Fp8Element))
      : static_cast<int32_t>(sizeof(Bf16));
  const int64_t output_bytes =
      static_cast<int64_t>(problem.m) * problem.n * output_element_bytes;
  const int32_t legacy_comm_ctas =
      output_bytes >= 32ll * 1024 * 1024
      ? 32
      : (problem.n >= 4096 ? 24 : 16);

  const bool pipeline_policy = qkv_pipeline_policy_enabled();
  if (!pipeline_policy ||
      (!bf16_problem && !fp8_problem) || route.world_size <= 1) {
    return legacy_comm_ctas;
  }
  if (fp8_problem &&
      selected_fp8_pipeline(problem) == Fp8PipelinePolicy::kCooperative) {
    // The independent comm model below is calibrated for ping-pong. Manual
    // cooperative experiments must request an explicit split until they have
    // their own wave table.
    return legacy_comm_ctas;
  }

  int32_t device = 0;
  // The current CUDA device is thread-local and may change between calls, so
  // keep this cheap lookup outside the cache.  The cache removes the device
  // property/SM queries and the comm-by-tile search that caused the eager
  // host stall.
  if (cudaGetDevice(&device) != cudaSuccess) {
    return legacy_comm_ctas;
  }
  const auto cache_key = make_qkv_comm_recommendation_key(
      problem, route, device);
  int32_t cached_comm_ctas = 0;
  if (find_qkv_comm_recommendation(cache_key, &cached_comm_ctas)) {
    return cached_comm_ctas;
  }

  int32_t sm_count = 0;
  if (cudaDeviceGetAttribute(
          &sm_count, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
    return legacy_comm_ctas;
  }
  if (sm_count != 132) {
    store_qkv_comm_recommendation(cache_key, legacy_comm_ctas);
    return legacy_comm_ctas;
  }

  // Jointly choose the communication split and GEMM tile from a finite
  // two-stage pipeline model.  Every input is known before launch:
  //
  //   C = calibrated full-wave tile time * persistent GEMM waves
  //   R = max(16-KiB route-task waves, remote bytes / NVLink rate)
  //   T = C + max(one route-task wave, R - later GEMM waves)
  //
  // GEMM is the producer.  Route may start after the first GEMM wave, so all
  // later GEMM waves are available to hide its total work; at least one fine
  // 16-KiB route-task wave remains as pipeline drain.  This is the actual
  // dependency graph, not an overlap coefficient fitted from model winners.
  // The GEMM wave table above and the 16-KiB task cost below are reusable H200
  // kernel primitive calibrations.  TE-UB results, model names and per-shape
  // winner tables are deliberately absent from this policy.
  constexpr int32_t kMinCommCtas = 4;
  constexpr int32_t kMaxCommCtas = 32;
  // Standalone H200 route sweeps stop gaining useful one-way fabric bandwidth
  // beyond 16 CTAs.  This is a topology/primitive saturation point; it only
  // scales the NVLink lower bound and does not encode a model or shape winner.
  constexpr int32_t kRouteSlotsPerCta = kQkvBulkSlots;
  if (route.head_dim != kQkvBulkColumns ||
      route.q_heads <= 0 || route.kv_heads <= 0 ||
      route.q_heads % route.world_size != 0 ||
      route.kv_heads % route.world_size != 0) {
    store_qkv_comm_recommendation(cache_key, legacy_comm_ctas);
    return legacy_comm_ctas;
  }

  const double one_way_nvlink_gbps =
      0.5 * a2a_lhs_nvlink_bidirectional_gbps(device);
  if (!(one_way_nvlink_gbps > 0.0)) {
    store_qkv_comm_recommendation(cache_key, legacy_comm_ctas);
    return legacy_comm_ctas;
  }

  const int32_t q_local_heads = route.q_heads / route.world_size;
  const int32_t kv_local_heads = route.kv_heads / route.world_size;
  const int32_t routed_local_heads = q_local_heads +
      (route.defer_v_a2a ? 1 : 2) * kv_local_heads;
  const int64_t m_chunks = ceil_div(problem.m, kQkvBulkRows);
  const int64_t route_tasks =
      static_cast<int64_t>(route.world_size) * m_chunks *
      routed_local_heads;
  const int64_t routed_global_width =
      static_cast<int64_t>(route.q_heads +
          (route.defer_v_a2a ? 1 : 2) * route.kv_heads) *
      route.head_dim;
  if (fp8_problem) {
    int32_t best_comm_ctas = legacy_comm_ctas;
    int32_t best_tie_rank = std::numeric_limits<int32_t>::max();
    double best_score_ns = std::numeric_limits<double>::infinity();
    for (int32_t comm_ctas = kMinCommCtas;
         comm_ctas <= kMaxCommCtas && comm_ctas < sm_count;
         comm_ctas += 2) {
      const Fp8TilePolicy tile = select_fp8_qkv_tile(
          problem, route, comm_ctas, sm_count, device);
      const double score_ns = fp8_qkv_pipeline_score_ns(
          problem, route, comm_ctas, sm_count, device, tile);
      if (!std::isfinite(score_ns)) {
        continue;
      }
      // An exact model tie means extra route CTAs do not change the predicted
      // compute waves or exposed communication. Prefer the measured FP8 route
      // saturation point, rather than starving route issue or consuming extra
      // compute headroom for no modeled benefit.
      constexpr double kTieToleranceNs = 1.0e-6;
      const int32_t tie_rank =
          std::abs(comm_ctas - kFp8QkvLatencySaturationCommCtas);
      if (score_ns + kTieToleranceNs < best_score_ns ||
          (std::abs(score_ns - best_score_ns) <= kTieToleranceNs &&
           (tie_rank < best_tie_rank ||
            (tie_rank == best_tie_rank && comm_ctas < best_comm_ctas)))) {
        best_score_ns = score_ns;
        best_tie_rank = tie_rank;
        best_comm_ctas = comm_ctas;
      }
    }
    store_qkv_comm_recommendation(cache_key, best_comm_ctas);
    return best_comm_ctas;
  }
  const double remote_bytes =
      static_cast<double>(problem.m) * routed_global_width * sizeof(Bf16) *
      (route.world_size - 1) / route.world_size;

  int32_t best_comm_ctas = legacy_comm_ctas;
  double best_score_ns = std::numeric_limits<double>::infinity();
  int32_t best_tie_rank = std::numeric_limits<int32_t>::max();
  for (int32_t comm_ctas = kMinCommCtas;
       comm_ctas <= kMaxCommCtas && comm_ctas < sm_count;
       comm_ctas += 2) {
    const auto compute = estimate_qkv_compute(
        problem,
        comm_ctas,
        sm_count,
        route.qkv_peer_interleaved,
        true);
    if (!compute.valid || compute.waves <= 0) {
      continue;
    }

    const int64_t route_slots =
        static_cast<int64_t>(comm_ctas) * kRouteSlotsPerCta;
    const int64_t route_waves = ceil_div(route_tasks, route_slots);
    const double route_task_ns =
        route_waves * static_cast<double>(kQkvRouteTaskWaveNs);
    const double comm_fraction = std::min(
        1.0,
        static_cast<double>(comm_ctas) /
            kQkvFabricSaturationCommCtas);
    // bytes / (GB/s) is numerically nanoseconds.
    const double route_link_ns =
        remote_bytes / (one_way_nvlink_gbps * comm_fraction);
    const double route_ns = std::max(route_task_ns, route_link_ns);
    const double compute_ns = static_cast<double>(compute.total_ns);
    const double later_compute_ns =
        compute_ns - static_cast<double>(compute.wave_ns);
    const double exposed_route_ns = std::max(
        static_cast<double>(kQkvRouteTaskWaveNs),
        route_ns - later_compute_ns);
    // A cluster-M2 worker exposes two CTAs to the same multicast/barrier
    // progress dependency.  Charge one route service quantum per bound CTA;
    // this is the same independent-progress risk used by the tile selector
    // above, now applied across different communication splits as well.
    const double cluster_progress_risk_ns = compute.cluster_m == 1
        ? 0.0
        : compute.cluster_m * static_cast<double>(kQkvRouteTaskWaveNs);
    const double score_ns =
        compute_ns + exposed_route_ns + cluster_progress_risk_ns;

    // Exact model ties retain the mature v6 split.  This prevents a policy
    // change when the physical model predicts no end-to-end benefit.
    const int32_t tie_rank = comm_ctas == legacy_comm_ctas
        ? 0
        : std::abs(comm_ctas - legacy_comm_ctas) + 1;
    constexpr double kTieToleranceNs = 1.0e-6;
    if (score_ns + kTieToleranceNs < best_score_ns ||
        (std::abs(score_ns - best_score_ns) <= kTieToleranceNs &&
         tie_rank < best_tie_rank)) {
      best_score_ns = score_ns;
      best_comm_ctas = comm_ctas;
      best_tie_rank = tie_rank;
    }
  }
  store_qkv_comm_recommendation(cache_key, best_comm_ctas);
  return best_comm_ctas;
}

}  // namespace fuse
