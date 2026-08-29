// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include "fuse/layout/gemm.h"
#include "fuse/layout/ulysses.h"
#include "fuse/operators/a2a_gemm.h"

#include <cstdint>

#include <cuda_runtime_api.h>

namespace fuse {

// Weighted-sequence variants keep one contiguous persistent kernel per rank.
// Every rank may own a different number of sequence rows, but the row ranges
// must form one gap-free global sequence in rank order. This avoids the
// multi-segment scheduling tax of remote work donation. The framework is
// responsible for carrying the same partition through residual/MLP layers.
struct WeightedGemmA2AParams {
  const Bf16* lhs = nullptr;
  const Bf16* rhs_nt = nullptr;
  Bf16* local_output = nullptr;
  Bf16* peer_output[kMaxWorldSize]{};
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  // Row offset within one global sequence, not including batch.
  int32_t global_sequence_begin = 0;
  int32_t num_comm_ctas = 0;
  uint32_t epoch = 0;
  float alpha = 1.0f;
};

struct WeightedA2AGemmParams {
  const Bf16* peer_input[kMaxWorldSize]{};
  const uint32_t* peer_input_ready[kMaxWorldSize]{};
  Bf16* input_staging = nullptr;
  Bf16* rhs_nt = nullptr;
  Bf16* output = nullptr;
  uint32_t* peer_done_epoch[kMaxWorldSize]{};
  uint32_t* ready = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t global_sequence_begin = 0;
  int32_t num_comm_ctas = 0;
  A2ALhsGemmPolicy lhs_policy = A2ALhsGemmPolicy::kAuto;
  uint32_t epoch = 0;
  uint32_t input_epoch = 0;
  float alpha = 1.0f;
};

// Caller-supplied, dimensionless effective resource capacity relative to the
// reference rank under the same workload. The planner never reads clocks or
// device names. A pure SM downclock from 1980 to 1500 MHz is
// {1500/1980, 1, 1} only when the reference rank actually sustains 1980 MHz;
// if it also hits a power limit, pass the measured/known Tensor-service ratio
// instead. A pure memory downclock changes hbm only.
struct HeterogeneousCpRankResources {
  double sm = 1.0;
  double hbm = 1.0;
  double nvlink = 1.0;
};

// Inputs shared by the two weighted-sequence planners.  uniform_local_rows is
// the ordinary equal CP ownership before planning.  row_quantum is a
// structural constraint, not a tuning parameter: it must preserve the row
// alignment required by the surrounding framework.  The benchmark uses 256
// rows so every existing Hopper M-tile/cluster geometry remains aligned.
struct WeightedCpPlannerOptions {
  int32_t world_size = 0;
  int32_t uniform_local_rows = 0;
  int32_t row_quantum = 256;
  int32_t sm_count = 132;
  // H200 SXM is 900 GB/s bidirectional for the topology used by the primitive
  // model.  H800 callers pass 400; per-rank nvlink scales are applied on top.
  double baseline_nvlink_bidirectional_gbps = 900.0;
  // Long, sustained QKV projections can drive nominally faster ranks into
  // their power limit, so a fixed clock ratio no longer describes their
  // effective throughput.  The default locked-frequency policy therefore
  // redistributes QKV rows only through S=16K.  Callers may opt in for longer
  // sequences only when they have a stable workload-level capacity ratio.
  bool allow_long_qkv_redistribution = false;
  // A nominal SM-clock ratio stops describing effective capacity once the
  // reference ranks hit their sustained power limit.  By default the H200
  // planner redistributes only inside its measured power-safe work envelope.
  // Set this only when rank[].sm already contains a workload-level effective
  // throughput ratio rather than a nominal locked-clock ratio.
  bool allow_power_limited_redistribution = false;
  HeterogeneousCpRankResources rank[kMaxWorldSize]{};
};

constexpr int32_t kDefaultWeightedQkvMaxGlobalSequence = 16 * 1024;

struct WeightedCpRankDecision {
  int32_t rows = 0;
  int32_t global_sequence_begin = 0;
  int32_t comm_ctas = 0;
  int32_t tile_m = 0;
  int32_t tile_n = 0;
  int32_t cluster_m = 0;
  int32_t waves = 0;
  double compute_us = 0.0;
  double route_us = 0.0;
  double critical_us = 0.0;
};

// The planner minimizes the predicted slowest-rank time over integer row
// allocations and communication-CTA choices.  equivalent_alpha is output-only
// diagnostics: 0 is equal ownership and 1 is the raw-SM-proportional endpoint;
// values outside [0,1] honestly report a plan beyond that line. It never
// participates in selection and callers never need to provide it.
struct WeightedCpPlan {
  int32_t world_size = 0;
  int32_t uniform_local_rows = 0;
  int32_t row_quantum = 0;
  int32_t uniform_bottleneck_rank = -1;
  int32_t weighted_bottleneck_rank = -1;
  int64_t redistributed_rows = 0;
  bool weighted = false;
  double uniform_critical_us = 0.0;
  double weighted_critical_us = 0.0;
  double predicted_speedup = 1.0;
  double equivalent_alpha = 0.0;
  WeightedCpRankDecision rank[kMaxWorldSize]{};
};

// These are cold-path planning calls.  They are deterministic functions of
// the shape, route, and supplied resource ratios; no runtime profiling or
// model-name/case table is consulted.  Shapes outside the calibrated Hopper
// BF16 domain return cudaErrorNotSupported instead of extrapolating silently.
cudaError_t plan_weighted_gemm_a2a(
    const GemmProblem& uniform_problem,
    const UlyssesRoute& uniform_route,
    const WeightedCpPlannerOptions& options,
    WeightedCpPlan* plan);

cudaError_t plan_weighted_a2a_gemm(
    const GemmProblem& uniform_problem,
    const UlyssesRoute& uniform_route,
    const WeightedCpPlannerOptions& options,
    WeightedCpPlan* plan);

cudaError_t launch_weighted_gemm_a2a_cutlass(
    const WeightedGemmA2AParams& params,
    cudaStream_t stream);

cudaError_t launch_weighted_a2a_gemm_cutlass(
    const WeightedA2AGemmParams& params,
    cudaStream_t stream);

}  // namespace fuse
