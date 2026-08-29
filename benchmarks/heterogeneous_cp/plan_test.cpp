// SPDX-License-Identifier: BSD-3-Clause

#include "fuse/operators/heterogeneous_cp.h"

#include <cstdio>
#include <cstdlib>

namespace {

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "weighted CP plan test failed: %s\n", message);
    std::exit(1);
  }
}

fuse::WeightedCpPlannerOptions planner_options(
    int32_t world_size,
    int32_t local_rows,
    int32_t slow_count,
    double slow_sm_ratio = 1500.0 / 1980.0) {
  fuse::WeightedCpPlannerOptions options{};
  options.world_size = world_size;
  options.uniform_local_rows = local_rows;
  options.row_quantum = 256;
  options.sm_count = 132;
  for (int32_t rank = world_size - slow_count; rank < world_size; ++rank) {
    options.rank[rank].sm = slow_sm_ratio;
  }
  return options;
}

fuse::GemmProblem qkv_problem(int32_t local_rows) {
  fuse::GemmProblem problem{};
  problem.m = local_rows;
  problem.n = (24 + 2 * 24) * 128;
  problem.k = 4096;
  problem.raster = fuse::GemmRaster::kAlongN;
  return problem;
}

fuse::UlyssesRoute qkv_route(int32_t world_size, int32_t local_rows) {
  fuse::UlyssesRoute route{};
  route.world_size = world_size;
  route.batch = 1;
  route.global_seq = world_size * local_rows;
  route.seq_local = local_rows;
  route.q_heads = 24;
  route.kv_heads = 24;
  route.head_dim = 128;
  route.kind = fuse::RouteKind::kQkvGqaPack;
  route.direction = fuse::RouteDirection::kForward;
  return route;
}

fuse::GemmProblem oproj_problem(int32_t local_rows) {
  fuse::GemmProblem problem{};
  problem.m = local_rows;
  problem.n = 4096;
  problem.k = 24 * 128;
  problem.raster = fuse::GemmRaster::kAlongN;
  return problem;
}

fuse::UlyssesRoute oproj_route(int32_t world_size, int32_t local_rows) {
  fuse::UlyssesRoute route{};
  route.world_size = world_size;
  route.batch = 1;
  route.global_seq = world_size * local_rows;
  route.seq_local = local_rows;
  route.q_heads = 24;
  route.local_heads = 24 / world_size;
  route.head_dim = 128;
  route.kind = fuse::RouteKind::kHeadToSequence;
  route.direction = fuse::RouteDirection::kInverse;
  return route;
}

void check_plan_shape(
    const fuse::WeightedCpPlan& plan,
    int32_t world_size,
    int32_t local_rows) {
  check(plan.world_size == world_size, "world size");
  check(plan.row_quantum == 256, "row quantum");
  int32_t cursor = 0;
  for (int32_t rank = 0; rank < world_size; ++rank) {
    check(plan.rank[rank].rows > 0, "rank must own rows");
    check(plan.rank[rank].rows % plan.row_quantum == 0,
          "row ownership must preserve the quantum");
    check(plan.rank[rank].global_sequence_begin == cursor,
          "rank ranges must be contiguous");
    cursor += plan.rank[rank].rows;
  }
  check(cursor == world_size * local_rows,
        "rank ranges must cover the global sequence");
}

void check_useful_or_uniform(
    const fuse::WeightedCpPlan& plan,
    int32_t slow_count) {
  if (plan.weighted) {
    check(slow_count > 0 && slow_count < plan.world_size,
          "only mixed clocks may redistribute");
    check(plan.predicted_speedup > 1.0 && plan.redistributed_rows > 0,
          "redistribution must reduce the modeled critical path");
  } else {
    check(plan.predicted_speedup == 1.0 && plan.redistributed_rows == 0,
          "fallback must be the exact equal plan");
  }
}

}  // namespace

int main() {
  constexpr int32_t kWorlds[] = {2, 4, 6, 8};
  for (const int32_t world_size : kWorlds) {
    bool saw_short_qkv_redistribution = false;
    bool saw_oproj_redistribution = false;
    for (int32_t slow_count = 0; slow_count <= world_size; ++slow_count) {
      const auto short_options = planner_options(world_size, 2048, slow_count);
      fuse::WeightedCpPlan short_qkv{};
      check(
          fuse::plan_weighted_gemm_a2a(
              qkv_problem(2048),
              qkv_route(world_size, 2048),
              short_options,
              &short_qkv) == cudaSuccess,
          "short QKV planning");
      check_plan_shape(short_qkv, world_size, 2048);
      check_useful_or_uniform(short_qkv, slow_count);
      saw_short_qkv_redistribution |= short_qkv.weighted;

      const auto long_options = planner_options(world_size, 16384, slow_count);
      fuse::WeightedCpPlan long_qkv{};
      check(
          fuse::plan_weighted_gemm_a2a(
              qkv_problem(16384),
              qkv_route(world_size, 16384),
              long_options,
              &long_qkv) == cudaSuccess,
          "long QKV planning");
      check_plan_shape(long_qkv, world_size, 16384);
      check(!long_qkv.weighted && long_qkv.redistributed_rows == 0,
            "long QKV must default to equal ownership");

      fuse::WeightedCpPlan oproj{};
      check(
          fuse::plan_weighted_a2a_gemm(
              oproj_problem(16384),
              oproj_route(world_size, 16384),
              long_options,
              &oproj) == cudaSuccess,
          "OProj planning");
      check_plan_shape(oproj, world_size, 16384);
      check_useful_or_uniform(oproj, slow_count);
      saw_oproj_redistribution |= oproj.weighted;
    }
    check(saw_short_qkv_redistribution,
          "each CP degree needs a useful short-QKV plan");
    check(saw_oproj_redistribution,
          "each CP degree needs a useful OProj plan");
  }

  // The override is intentionally explicit: it is only valid when the
  // caller has measured a stable long-QKV capacity ratio under the workload.
  auto override_options = planner_options(8, 16384, 3);
  override_options.allow_long_qkv_redistribution = true;
  fuse::WeightedCpPlan override_plan{};
  check(
      fuse::plan_weighted_gemm_a2a(
          qkv_problem(16384),
          qkv_route(8, 16384),
          override_options,
          &override_plan) == cudaSuccess,
      "explicit long-QKV planning override");
  check_plan_shape(override_plan, 8, 16384);

  std::printf("weighted CP planner PASS: CP2/4/6/8 fixed-clock matrix\n");
  return 0;
}
