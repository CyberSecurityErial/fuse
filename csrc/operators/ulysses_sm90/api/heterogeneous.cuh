// SPDX-License-Identifier: BSD-3-Clause
// Public API implementation; assembled by csrc/operators/ulysses_sm90.cu.
// Declarations: fuse/operators/ulysses/heterogeneous_cp.h.
//
// Module index:
//   - rank capacity and communication-cost model
//   - generic weighted-row plan solver
//   - QKV/OProj weighted planning and launch entry points

namespace fuse {

namespace {

// The heterogeneous planner is deliberately a cold-path, finite physical
// model.  It never observes clocks.  The caller supplies three independent
// capacity ratios, and every candidate is an integer row ownership plus a
// precompiled communication/tile policy.
struct WeightedRankModel {
  WeightedCpRankDecision decision{};
  // The minimum of several discrete kernel policies can have tiny model
  // inversions at a policy boundary.  Allocation uses the monotone envelope;
  // the report retains the actual selected policy time.
  double allocation_critical_us = 0.0;
  bool valid = false;
};

constexpr double kH200DenseBf16FlopsPerUs = 989.0e6;
constexpr double kH200HbmBytesPerUs = 4.8e6;
// Sustained locked-frequency validation on the 700-W H200 system brackets
// the reference-rank power transition between about 0.49 ms (1980 MHz held)
// and 0.97 ms (reference ranks collapsed to roughly 1500--1650 MHz).  Stay on
// the conservative side when the caller supplies nominal clock ratios.  This
// is an architecture/workload envelope, not a model-name or shape winner.
constexpr double kH200WeightedPowerSafeDenseComputeUs = 750.0;

double dense_bf16_compute_floor_us(const GemmProblem& problem) {
  return 2.0 * problem.m * problem.n * problem.k * problem.l /
      kH200DenseBf16FlopsPerUs;
}

bool valid_rank_resources(const HeterogeneousCpRankResources& resources) {
  return std::isfinite(resources.sm) && resources.sm > 0.0 &&
      std::isfinite(resources.hbm) && resources.hbm > 0.0 &&
      std::isfinite(resources.nvlink) && resources.nvlink > 0.0;
}

double gemm_resource_time_scale(
    const GemmProblem& problem,
    const HeterogeneousCpRankResources& resources) {
  const double flops = 2.0 * problem.m * problem.n * problem.k * problem.l;
  // This is compulsory tensor traffic, not a cache-hit fit.  It is used only
  // to decide whether SM or HBM capacity is the active scaling resource; the
  // absolute time remains the measured CUTLASS primitive time.
  const double bytes = sizeof(Element) * static_cast<double>(problem.l) *
      (static_cast<double>(problem.m) * problem.k +
       static_cast<double>(problem.n) * problem.k +
       static_cast<double>(problem.m) * problem.n);
  const double compute_floor_us = flops / kH200DenseBf16FlopsPerUs;
  const double hbm_floor_us = bytes / kH200HbmBytesPerUs;
  const double base_floor_us = std::max(compute_floor_us, hbm_floor_us);
  if (!(base_floor_us > 0.0)) {
    return 1.0;
  }
  return std::max(
      compute_floor_us / resources.sm,
      hbm_floor_us / resources.hbm) / base_floor_us;
}

WeightedRankModel score_weighted_qkv_rank(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const WeightedCpPlannerOptions& options,
    const HeterogeneousCpRankResources& resources,
    int32_t fixed_comm_ctas = 0) {
  WeightedRankModel best{};
  best.decision.critical_us = std::numeric_limits<double>::infinity();
  if (problem.m <= 0 || options.sm_count != 132 ||
      route.head_dim != kQkvBulkColumns || route.world_size <= 1 ||
      route.q_heads <= 0 || route.kv_heads <= 0 ||
      route.q_heads % route.world_size != 0 ||
      route.kv_heads % route.world_size != 0 ||
      route.qkv_peer_interleaved || !valid_rank_resources(resources)) {
    return best;
  }

  const int32_t q_local_heads = route.q_heads / route.world_size;
  const int32_t kv_local_heads = route.kv_heads / route.world_size;
  const int32_t routed_local_heads = q_local_heads +
      (route.defer_v_a2a ? 1 : 2) * kv_local_heads;
  const int64_t route_tasks = static_cast<int64_t>(route.world_size) *
      ceil_div(problem.m, kQkvBulkRows) * routed_local_heads;
  const int64_t routed_global_width =
      static_cast<int64_t>(route.q_heads +
          (route.defer_v_a2a ? 1 : 2) * route.kv_heads) *
      route.head_dim;
  const double remote_bytes =
      static_cast<double>(problem.m) * routed_global_width * sizeof(Element) *
      (route.world_size - 1) / route.world_size;
  const double gemm_scale = gemm_resource_time_scale(problem, resources);
  const double link_scale = std::min(resources.hbm, resources.nvlink);
  const double one_way_nvlink_gbps =
      0.5 * options.baseline_nvlink_bidirectional_gbps * link_scale;
  if (!(one_way_nvlink_gbps > 0.0)) {
    return best;
  }

  // One sequence-chunk frontier contains one task per routed destination
  // head.  Keep it within one 12-slot communication wave; otherwise the next
  // GEMM frontier can be ready while the previous frontier still occupies the
  // route service loop, a dependency bubble that the aggregate-byte bound
  // cannot hide.  This derives the lower bound from layout, not from P/x/y.
  const int32_t frontier_tasks = route.world_size * routed_local_heads;
  const int32_t frontier_comm_ctas = static_cast<int32_t>(
      ceil_div(frontier_tasks, kQkvBulkSlots));
  const int32_t kMinCommCtas = std::max(
      4, frontier_comm_ctas + (frontier_comm_ctas & 1));
  constexpr int32_t kMaxCommCtas = 32;
  const int64_t output_bytes =
      static_cast<int64_t>(problem.m) * problem.n * sizeof(Element);
  const int32_t legacy_comm_ctas = output_bytes >= 32ll * 1024 * 1024
      ? 32
      : (problem.n >= 4096 ? 24 : 16);
  int32_t best_tie_rank = std::numeric_limits<int32_t>::max();
  for (int32_t comm_ctas = kMinCommCtas;
       comm_ctas <= kMaxCommCtas && comm_ctas < options.sm_count;
       comm_ctas += 2) {
    if (fixed_comm_ctas > 0 && comm_ctas != fixed_comm_ctas) {
      continue;
    }
    const auto compute = estimate_qkv_compute(
        problem,
        comm_ctas,
        options.sm_count,
        route.qkv_peer_interleaved,
        true);
    if (!compute.valid || compute.waves <= 0) {
      continue;
    }
    const int64_t route_slots =
        static_cast<int64_t>(comm_ctas) * kQkvBulkSlots;
    const int64_t route_waves = ceil_div(route_tasks, route_slots);
    // TMA/NVLink payload bandwidth is governed by HBM/link clocks.  Only the
    // issue/service quantum follows SM clock.  This distinction is why a
    // 24.2% SM downclock measured only about a 10% route slowdown.
    const double route_quantum_us =
        kQkvRouteTaskWaveNs / 1000.0 / resources.sm;
    const double route_task_us = route_waves * route_quantum_us;
    const double comm_fraction = std::min(
        1.0,
        static_cast<double>(comm_ctas) /
            kQkvFabricSaturationCommCtas);
    const double route_link_us =
        remote_bytes / (one_way_nvlink_gbps * comm_fraction) / 1000.0;
    const double route_us = std::max(route_task_us, route_link_us);
    const double compute_us = compute.total_ns / 1000.0 * gemm_scale;
    const double first_wave_us = compute.wave_ns / 1000.0 * gemm_scale;
    const double later_compute_us = std::max(0.0, compute_us - first_wave_us);
    const double exposed_route_us = std::max(
        route_quantum_us, route_us - later_compute_us);
    const double cluster_progress_us = compute.cluster_m == 1
        ? 0.0
        : compute.cluster_m * route_quantum_us;
    const double critical_us =
        compute_us + exposed_route_us + cluster_progress_us;

    const bool better = critical_us < best.decision.critical_us - 1.0e-9;
    const bool tie =
        std::abs(critical_us - best.decision.critical_us) <= 1.0e-9;
    const int32_t tie_rank = comm_ctas == legacy_comm_ctas
        ? 0
        : std::abs(comm_ctas - legacy_comm_ctas) + 1;
    if (better || (tie && tie_rank < best_tie_rank)) {
      best.valid = true;
      best_tie_rank = tie_rank;
      best.decision.rows = problem.m;
      best.decision.comm_ctas = comm_ctas;
      best.decision.tile_m = kBlockM;
      best.decision.tile_n = compute.tile_n;
      best.decision.cluster_m = compute.cluster_m;
      best.decision.waves = static_cast<int32_t>(compute.waves);
      best.decision.compute_us = compute_us;
      best.decision.route_us = route_us;
      best.decision.critical_us = critical_us;
    }
  }
  best.allocation_critical_us = best.decision.critical_us;
  return best;
}

double scaled_a2a_lhs_compute_time_us(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const A2ALhsPolicyInfo& policy,
    const HeterogeneousCpRankResources& resources) {
  const double base_us = a2a_lhs_compute_time_us(policy, route.world_size);
  const double cp_scale = std::pow(
      std::max(1.0, static_cast<double>(route.world_size) / 4.0), 0.4883);
  const double launch_us = 5.80 * cp_scale;
  return launch_us + std::max(0.0, base_us - launch_us) *
      gemm_resource_time_scale(problem, resources);
}

double scaled_a2a_lhs_route_time_us(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const A2ALhsPolicyInfo& policy,
    int32_t comm_ctas,
    const WeightedCpPlannerOptions& options,
    const HeterogeneousCpRankResources& resources,
    double* first_ready_us) {
  constexpr double kLaunchUs = 9.20;
  constexpr double kCopyBmaxGbs = 528.8;
  constexpr double kCopyCtaScale = 4.961;
  constexpr double kStoreRowsPerUs = 41912.0;
  constexpr double kTasksPerUs = 0.3874;
  const int32_t row_bytes =
      problem.k / route.world_size * sizeof(Element);
  const int32_t max_rows = row_bytes > 0
      ? kA2ALhsBulkStageBytes / row_bytes
      : 0;
  const int32_t comm_rows = max_rows > 0
      ? std::min(policy.tile_m, max_rows)
      : kA2ALhsCommRows;
  const int64_t chunks_per_tile = ceil_div(policy.tile_m, comm_rows);
  const int64_t m_frontiers = ceil_div(problem.m, policy.tile_m);
  const int64_t tasks = m_frontiers * route.world_size * chunks_per_tile;
  const int64_t task_waves = ceil_div(
      tasks, static_cast<int64_t>(comm_ctas * kA2ALhsBulkSlots));

  // Bulk payload throughput follows HBM and NVLink capacities, not SM clock.
  // The scalar store/task service around it follows SM clock.
  const double issuer_bandwidth = kCopyBmaxGbs * resources.hbm *
      (1.0 - std::exp(-static_cast<double>(comm_ctas) / kCopyCtaScale));
  const double remote_fraction =
      static_cast<double>(route.world_size - 1) / route.world_size;
  const double fabric_bandwidth = remote_fraction > 0.0
      ? 0.5 * options.baseline_nvlink_bidirectional_gbps * resources.nvlink /
          remote_fraction
      : issuer_bandwidth;
  const double bandwidth = std::min(issuer_bandwidth, fabric_bandwidth);
  const double payload_bytes =
      static_cast<double>(problem.m) * problem.k * sizeof(Element);
  const double copy_us = payload_bytes / bandwidth / 1000.0;
  const double store_us =
      static_cast<double>(problem.m) * route.world_size /
      (comm_ctas * kStoreRowsPerUs * resources.sm);
  const double task_us =
      task_waves / (kTasksPerUs * resources.sm);
  const double cp_scale = std::pow(
      std::max(1.0, static_cast<double>(route.world_size) / 4.0), 0.2022);
  const double work_us = (copy_us + store_us + task_us) * cp_scale;
  const double launch_us = kLaunchUs * cp_scale;
  if (first_ready_us != nullptr) {
    *first_ready_us = launch_us + work_us / std::max<int64_t>(1, m_frontiers);
  }
  return launch_us + work_us;
}

WeightedRankModel score_weighted_oproj_rank(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const WeightedCpPlannerOptions& options,
    const HeterogeneousCpRankResources& resources,
    int32_t fixed_comm_ctas = 0) {
  WeightedRankModel best{};
  best.decision.critical_us = std::numeric_limits<double>::infinity();
  if (problem.m <= 0 || options.sm_count != 132 ||
      route.world_size <= 1 || problem.k % route.world_size != 0 ||
      !valid_rank_resources(resources)) {
    return best;
  }
  constexpr std::array<int32_t, 10> kCommCandidates{
      2, 4, 6, 8, 10, 12, 14, 16, 20, 24};
  // Eight CTAs are the measured H200 bulk-route issuer saturation point for
  // the weighted contiguous path.  An SM-only downclock lowers instruction
  // issue service but does not lower HBM/NVLink payload bandwidth, so preserve
  // issuer capacity with ceil_even(8 / sm_ratio), capped at the 16-CTA fabric
  // saturation point.  HBM- or NVLink-only degradation deliberately does not
  // add CTAs: more issuers cannot restore those bandwidth resources.
  const int32_t issuer_target = std::min(
      16,
      2 * static_cast<int32_t>(std::ceil(4.0 / resources.sm)));
  for (const int32_t comm_ctas : kCommCandidates) {
    if (comm_ctas >= options.sm_count ||
        (fixed_comm_ctas > 0 && comm_ctas != fixed_comm_ctas) ||
        (fixed_comm_ctas == 0 && comm_ctas != issuer_target)) {
      continue;
    }
    const auto policy = select_a2a_lhs_policy_impl(
        problem, comm_ctas, options.sm_count, A2ALhsGemmPolicy::kAuto);
    if (!std::isfinite(policy.estimated_cycles) || policy.waves <= 0) {
      continue;
    }
    const double compute_us = scaled_a2a_lhs_compute_time_us(
        problem, route, policy, resources);
    double first_ready_us = 0.0;
    const double route_us = scaled_a2a_lhs_route_time_us(
        problem,
        route,
        policy,
        comm_ctas,
        options,
        resources,
        &first_ready_us);
    const double route_after_first = std::max(0.0, route_us - first_ready_us);
    // Route produces the first M frontier; the remaining route work and GEMM
    // then run concurrently.  This is a dependency bound, not an empirical
    // overlap percentage.
    const double critical_us =
        first_ready_us + std::max(compute_us, route_after_first);
    const bool better = critical_us < best.decision.critical_us - 1.0e-9;
    const bool tie =
        std::abs(critical_us - best.decision.critical_us) <= 1.0e-9;
    if (better || (tie && comm_ctas < best.decision.comm_ctas)) {
      best.valid = true;
      best.decision.rows = problem.m;
      best.decision.comm_ctas = comm_ctas;
      best.decision.tile_m = policy.tile_m;
      best.decision.tile_n = policy.tile_n;
      best.decision.cluster_m = policy.cluster_m;
      best.decision.waves = policy.waves;
      best.decision.compute_us = compute_us;
      best.decision.route_us = route_us;
      best.decision.critical_us = critical_us;
    }
  }
  best.allocation_critical_us = best.decision.critical_us;
  return best;
}

bool valid_weighted_planner_inputs(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    const WeightedCpPlannerOptions& options,
    WeightedCpPlan* plan) {
  if (plan == nullptr || options.world_size <= 1 ||
      options.world_size > kMaxWorldSize ||
      route.world_size != options.world_size ||
      options.uniform_local_rows <= 0 ||
      problem.m != options.uniform_local_rows ||
      options.row_quantum <= 0 ||
      options.uniform_local_rows % options.row_quantum != 0 ||
      options.sm_count != 132 ||
      !std::isfinite(options.baseline_nvlink_bidirectional_gbps) ||
      options.baseline_nvlink_bidirectional_gbps <= 0.0 ||
      problem.input_dtype != DType::kBfloat16 ||
      problem.weight_dtype != DType::kBfloat16 ||
      problem.output_dtype != DType::kBfloat16 ||
      problem.l != 1 || route.batch != 1 ||
      route.global_seq != options.world_size * options.uniform_local_rows ||
      route.causal_load_balanced || route.packed_source_row != nullptr) {
    return false;
  }
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    if (!valid_rank_resources(options.rank[rank])) {
      return false;
    }
  }
  return true;
}

std::vector<int32_t> proportional_units(
    const std::vector<double>& capacity,
    int32_t total_units) {
  const int32_t world_size = static_cast<int32_t>(capacity.size());
  std::vector<int32_t> units(world_size, 0);
  std::vector<double> remainder(world_size, 0.0);
  double capacity_sum = 0.0;
  for (double value : capacity) {
    capacity_sum += value;
  }
  int32_t assigned = 0;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const double ideal =
        total_units * capacity[rank] / capacity_sum;
    units[rank] = static_cast<int32_t>(std::floor(ideal));
    remainder[rank] = ideal - units[rank];
    assigned += units[rank];
  }
  while (assigned < total_units) {
    int32_t best = 0;
    for (int32_t rank = 1; rank < world_size; ++rank) {
      if (remainder[rank] > remainder[best] ||
          (remainder[rank] == remainder[best] && rank < best)) {
        best = rank;
      }
    }
    ++units[best];
    remainder[best] = -1.0;
    ++assigned;
  }
  return units;
}

std::vector<int32_t> proportional_units_by_sm(
    const WeightedCpPlannerOptions& options,
    int32_t total_units) {
  std::vector<double> capacity(options.world_size, 0.0);
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    capacity[rank] = options.rank[rank].sm;
  }
  return proportional_units(capacity, total_units);
}

template <class ScoreRank, class ScoreUniform>
cudaError_t solve_weighted_cp_plan(
    const WeightedCpPlannerOptions& options,
    ScoreRank&& score_rank,
    ScoreUniform&& score_uniform,
    bool permit_redistribution,
    WeightedCpPlan* plan) {
  const int32_t equal_units =
      options.uniform_local_rows / options.row_quantum;
  const int32_t total_units = equal_units * options.world_size;
  if (equal_units <= 0 || total_units < options.world_size) {
    return cudaErrorNotSupported;
  }
  const int32_t max_units = total_units - (options.world_size - 1);
  std::vector<std::vector<WeightedRankModel>> table(options.world_size);
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    table[rank].resize(max_units + 1);
    double monotone = 0.0;
    for (int32_t units = 1; units <= max_units; ++units) {
      auto candidate = score_rank(
          rank, units * options.row_quantum);
      if (!candidate.valid ||
          !std::isfinite(candidate.decision.critical_us)) {
        return cudaErrorNotSupported;
      }
      monotone = std::max(monotone, candidate.decision.critical_us);
      candidate.allocation_critical_us = monotone;
      table[rank][units] = candidate;
    }
  }

  // The physical endpoint is inferred from total resource service demand,
  // compute + route, at equal work.  These roles overlap in elapsed time but
  // still consume finite SM/HBM/NVLink service; using their sum only for the
  // capacity endpoint prevents an ideal-overlap assumption from assigning a
  // compute-proportional overload to the fast rank.  The actual winner below
  // is still scored with the dependency-aware critical path.  Search only
  // between this endpoint and equal ownership.
  std::vector<double> effective_capacity(options.world_size, 0.0);
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    const auto& equal = table[rank][equal_units].decision;
    effective_capacity[rank] = equal_units /
        std::max(1.0e-12, equal.compute_us + equal.route_us);
  }
  const auto effective_endpoint =
      proportional_units(effective_capacity, total_units);
  std::vector<int32_t> units(options.world_size, 0);
  std::vector<int32_t> upper_units(options.world_size, 0);
  int32_t initially_assigned = 0;
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    units[rank] = std::min(equal_units, effective_endpoint[rank]);
    upper_units[rank] = std::max(equal_units, effective_endpoint[rank]);
    initially_assigned += units[rank];
  }
  for (int32_t assigned = initially_assigned;
       assigned < total_units;
       ++assigned) {
    int32_t best_rank = -1;
    double best_next_us = std::numeric_limits<double>::infinity();
    for (int32_t rank = 0; rank < options.world_size; ++rank) {
      if (units[rank] >= upper_units[rank]) {
        continue;
      }
      const double next_us =
          table[rank][units[rank] + 1].allocation_critical_us;
      if (next_us < best_next_us - 1.0e-12 ||
          (std::abs(next_us - best_next_us) <= 1.0e-12 &&
           (best_rank < 0 || units[rank] < units[best_rank] ||
            (units[rank] == units[best_rank] && rank < best_rank)))) {
        best_rank = rank;
        best_next_us = next_us;
      }
    }
    if (best_rank < 0) {
      return cudaErrorNotSupported;
    }
    ++units[best_rank];
  }

  WeightedCpPlan result{};
  result.world_size = options.world_size;
  result.uniform_local_rows = options.uniform_local_rows;
  result.row_quantum = options.row_quantum;
  result.uniform_critical_us = 0.0;
  result.weighted_critical_us = 0.0;
  int32_t cursor = 0;
  int64_t l1_redistribution = 0;
  bool resources_equal = true;
  for (int32_t rank = 1; rank < options.world_size; ++rank) {
    resources_equal = resources_equal &&
        std::abs(options.rank[rank].sm - options.rank[0].sm) <= 1.0e-12 &&
        std::abs(options.rank[rank].hbm - options.rank[0].hbm) <= 1.0e-12 &&
        std::abs(options.rank[rank].nvlink - options.rank[0].nvlink) <= 1.0e-12;
  }
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    const auto uniform = score_uniform(rank);
    if (!uniform.valid) {
      return cudaErrorNotSupported;
    }
    if (uniform.decision.critical_us > result.uniform_critical_us) {
      result.uniform_critical_us = uniform.decision.critical_us;
      result.uniform_bottleneck_rank = rank;
    }
    auto decision = table[rank][units[rank]].decision;
    decision.global_sequence_begin = cursor;
    cursor += decision.rows;
    result.rank[rank] = decision;
    l1_redistribution += std::abs(decision.rows - options.uniform_local_rows);
    if (decision.critical_us > result.weighted_critical_us) {
      result.weighted_critical_us = decision.critical_us;
      result.weighted_bottleneck_rank = rank;
    }
  }
  if (cursor != total_units * options.row_quantum) {
    return cudaErrorUnknown;
  }
  result.redistributed_rows = l1_redistribution / 2;
  result.predicted_speedup = result.weighted_critical_us > 0.0
      ? result.uniform_critical_us / result.weighted_critical_us
      : 1.0;

  const auto proportional = proportional_units_by_sm(options, total_units);
  double projection = 0.0;
  double endpoint_norm = 0.0;
  for (int32_t rank = 0; rank < options.world_size; ++rank) {
    const double endpoint = proportional[rank] - equal_units;
    const double selected = units[rank] - equal_units;
    projection += selected * endpoint;
    endpoint_norm += endpoint * endpoint;
  }
  result.equivalent_alpha = endpoint_norm > 0.0
      ? projection / endpoint_norm
      : 0.0;
  result.weighted = permit_redistribution && !resources_equal &&
      result.predicted_speedup > 1.0 + 1.0e-12;

  // Exact ties and model losses return the ordinary equal operator.  This is
  // not a fitted safety margin: the physical model must predict an actual
  // critical-path reduction before ownership changes.
  if (!result.weighted) {
    result.weighted_critical_us = result.uniform_critical_us;
    result.weighted_bottleneck_rank = result.uniform_bottleneck_rank;
    result.redistributed_rows = 0;
    result.predicted_speedup = 1.0;
    result.equivalent_alpha = 0.0;
    cursor = 0;
    for (int32_t rank = 0; rank < options.world_size; ++rank) {
      auto uniform = score_uniform(rank).decision;
      uniform.global_sequence_begin = cursor;
      cursor += options.uniform_local_rows;
      result.rank[rank] = uniform;
    }
  }
  *plan = result;
  return cudaSuccess;
}

}  // namespace

cudaError_t plan_weighted_gemm_a2a(
    const GemmProblem& uniform_problem,
    const UlyssesRoute& uniform_route,
    const WeightedCpPlannerOptions& options,
    WeightedCpPlan* plan) {
  if (!valid_weighted_planner_inputs(
          uniform_problem, uniform_route, options, plan) ||
      uniform_route.kind != RouteKind::kQkvGqaPack ||
      uniform_route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  const HeterogeneousCpRankResources reference{};
  const auto baseline = score_weighted_qkv_rank(
      uniform_problem, uniform_route, options, reference);
  if (!baseline.valid) {
    return cudaErrorNotSupported;
  }
  const int32_t uniform_comm_ctas = baseline.decision.comm_ctas;
  auto score_rank = [&](int32_t rank, int32_t rows) {
    GemmProblem problem = uniform_problem;
    problem.m = rows;
    UlyssesRoute route = uniform_route;
    route.seq_local = rows;
    return score_weighted_qkv_rank(
        problem, route, options, options.rank[rank]);
  };
  auto score_uniform = [&](int32_t rank) {
    return score_weighted_qkv_rank(
        uniform_problem,
        uniform_route,
        options,
        options.rank[rank],
        uniform_comm_ctas);
  };
  const bool permit_redistribution =
      (options.allow_long_qkv_redistribution ||
       uniform_route.global_seq <= kDefaultWeightedQkvMaxGlobalSequence) &&
      (options.allow_power_limited_redistribution ||
       dense_bf16_compute_floor_us(uniform_problem) <=
           kH200WeightedPowerSafeDenseComputeUs);
  return solve_weighted_cp_plan(
      options, score_rank, score_uniform, permit_redistribution, plan);
}

cudaError_t plan_weighted_a2a_gemm(
    const GemmProblem& uniform_problem,
    const UlyssesRoute& uniform_route,
    const WeightedCpPlannerOptions& options,
    WeightedCpPlan* plan) {
  if (!valid_weighted_planner_inputs(
          uniform_problem, uniform_route, options, plan) ||
      uniform_route.kind != RouteKind::kHeadToSequence ||
      uniform_route.direction != RouteDirection::kInverse) {
    return cudaErrorInvalidValue;
  }
  int32_t uniform_comm_ctas = 4;
  if (uniform_problem.m >= 32768) {
    constexpr double kHighPressure = 1.65e-4;
    const double remote_bytes_per_flop =
        static_cast<double>(uniform_route.world_size - 1) /
        (static_cast<double>(uniform_route.world_size) * uniform_problem.n);
    const double normalized_pressure = remote_bytes_per_flop *
        900.0 / options.baseline_nvlink_bidirectional_gbps;
    uniform_comm_ctas = normalized_pressure >= kHighPressure ? 8 : 6;
  }
  auto score_rank = [&](int32_t rank, int32_t rows) {
    GemmProblem problem = uniform_problem;
    problem.m = rows;
    UlyssesRoute route = uniform_route;
    route.seq_local = rows;
    return score_weighted_oproj_rank(
        problem, route, options, options.rank[rank]);
  };
  auto score_uniform = [&](int32_t rank) {
    return score_weighted_oproj_rank(
        uniform_problem,
        uniform_route,
        options,
        options.rank[rank],
        uniform_comm_ctas);
  };
  const bool permit_redistribution =
      options.allow_power_limited_redistribution ||
      dense_bf16_compute_floor_us(uniform_problem) <=
          kH200WeightedPowerSafeDenseComputeUs;
  return solve_weighted_cp_plan(
      options,
      score_rank,
      score_uniform,
      permit_redistribution,
      plan);
}

cudaError_t launch_weighted_gemm_a2a_cutlass(
    const WeightedGemmA2AParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const int32_t resolved = params.num_comm_ctas == 0
      ? recommended_gemm_a2a_comm_ctas(params.gemm, params.route)
      : params.num_comm_ctas;
  if (resolved <= 0 || resolved >= sm_count || params.epoch == 0 ||
      params.route.rank < 0 || params.route.rank >= params.route.world_size ||
      params.global_sequence_begin < 0 ||
      params.global_sequence_begin + params.route.seq_local >
          params.route.global_seq ||
      params.gemm.m != params.route.batch * params.route.seq_local ||
      params.route.defer_v_a2a || !params.lhs || !params.rhs_nt ||
      !params.local_output || !params.ready) {
    return cudaErrorInvalidValue;
  }

  WeightedGemmA2AKernelParams segment{};
  segment.lhs = params.lhs;
  segment.rhs_nt = params.rhs_nt;
  segment.local_output = params.local_output;
  for (int32_t peer = 0; peer < params.route.world_size; ++peer) {
    if (!params.peer_output[peer] || !params.peer_route_done_epoch[peer]) {
      return cudaErrorInvalidValue;
    }
    segment.peer_output[peer] = params.peer_output[peer];
    segment.peer_route_done_epoch[peer] =
        params.peer_route_done_epoch[peer];
  }
  segment.ready = params.ready;
  segment.gemm = params.gemm;
  segment.route = params.route;
  segment.num_comm_ctas = resolved;
  segment.epoch = params.epoch;
  segment.alpha = params.alpha;
  segment.executor_rank = params.route.rank;
  segment.logical_source_rank = params.route.rank;
  segment.source_row_begin = 0;
  segment.global_sequence_begin = params.global_sequence_begin;
  segment.weighted_partition = true;

  const QkvGemmPolicy policy = select_qkv_gemm_policy(
      params.gemm,
      resolved,
      sm_count,
      params.route.qkv_peer_interleaved);
  switch (policy) {
    case QkvGemmPolicy::kM128N64:
      return launch_gemm_a2a_impl<
          N64OutputGemm, WeightedQkvGqaPackCommN64<true>>(
              segment, stream, sm_count, device);
    case QkvGemmPolicy::kM128N128:
      if (params.route.qkv_peer_interleaved) {
        return launch_gemm_a2a_impl<
            OutputGemm,
            WeightedQkvGqaPackCommN128Interleaved<true>>(
                segment, stream, sm_count, device);
      }
      return launch_gemm_a2a_impl<
          OutputGemm, WeightedQkvGqaPackCommN128<true>>(
              segment, stream, sm_count, device);
    case QkvGemmPolicy::kM128N160:
      return launch_gemm_a2a_impl<
          N160OutputGemm, WeightedQkvGqaPackCommN160<true>>(
              segment, stream, sm_count, device);
    case QkvGemmPolicy::kM128N192:
      return launch_gemm_a2a_impl<
          N192OutputGemm, WeightedQkvGqaPackCommN192<true>>(
              segment, stream, sm_count, device);
    case QkvGemmPolicy::kM128N256ClusterM2:
      return launch_gemm_a2a_impl<
          ProjectionOutputGemm, WeightedQkvGqaPackCommN256<true>>(
              segment, stream, sm_count, device);
    case QkvGemmPolicy::kM128N320ClusterM2:
      return launch_gemm_a2a_impl<
          WideN320OutputGemm, WeightedQkvGqaPackCommN320<true>>(
              segment, stream, sm_count, device);
  }
  return cudaErrorNotSupported;
}

cudaError_t launch_weighted_a2a_gemm_cutlass(
    const WeightedA2AGemmParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const int32_t resolved = params.num_comm_ctas == 0
      ? recommended_a2a_lhs_gemm_comm_ctas_impl(
            params.gemm, params.route, sm_count, device)
      : params.num_comm_ctas;
  if (resolved <= 0 || resolved >= sm_count || params.epoch == 0 ||
      params.route.rank < 0 || params.route.rank >= params.route.world_size ||
      params.global_sequence_begin < 0 ||
      params.global_sequence_begin + params.route.seq_local >
          params.route.global_seq ||
      params.gemm.m != params.route.batch * params.route.seq_local ||
      params.route.causal_load_balanced || !params.input_staging ||
      !params.rhs_nt || !params.output || !params.ready) {
    return cudaErrorInvalidValue;
  }

  WeightedA2AGemmKernelParams segment{};
  for (int32_t peer = 0; peer < params.route.world_size; ++peer) {
    if (!params.peer_input[peer] || !params.peer_done_epoch[peer] ||
        (params.input_epoch != 0 && !params.peer_input_ready[peer])) {
      return cudaErrorInvalidValue;
    }
    segment.peer_input[peer] = params.peer_input[peer];
    segment.peer_input_ready[peer] = params.peer_input_ready[peer];
    segment.peer_done_epoch[peer] = params.peer_done_epoch[peer];
  }
  segment.input_staging = params.input_staging;
  segment.rhs_nt = params.rhs_nt;
  segment.output = params.output;
  segment.ready = params.ready;
  segment.gemm = params.gemm;
  segment.route = params.route;
  segment.num_comm_ctas = resolved;
  segment.lhs_policy = params.lhs_policy;
  segment.epoch = params.epoch;
  segment.input_epoch = params.input_epoch;
  segment.alpha = params.alpha;
  segment.executor_rank = params.route.rank;
  segment.logical_source_rank = params.route.rank;
  segment.source_row_begin = 0;
  segment.global_sequence_begin = params.global_sequence_begin;
  segment.weighted_partition = true;

  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm, resolved, sm_count, params.lhs_policy);
  switch (selected.policy) {
    case A2ALhsGemmPolicy::kM64N128:
      return launch_weighted_a2a_policy<A2ALhsM64Gemm, 64>(
          segment, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N128:
      return launch_weighted_a2a_policy<A2ALhsInputGemm, 128>(
          segment, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N160:
      return launch_weighted_a2a_policy<A2ALhsN160Gemm, 128>(
          segment, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      return launch_weighted_a2a_policy<A2ALhsProjectionGemm, 128>(
          segment, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      return launch_weighted_a2a_policy<A2ALhsWideN320Gemm, 128>(
          segment, stream, sm_count, device);
    default:
      return cudaErrorNotSupported;
  }
}

}  // namespace fuse
