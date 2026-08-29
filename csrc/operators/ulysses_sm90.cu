// SPDX-License-Identifier: BSD-3-Clause
#define CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED 1
#include "fuse/operators/a2a_gemm.h"
#include "fuse/operators/gemm_a2a.h"
#include "fuse/operators/heterogeneous_cp.h"

#include "fuse/arch/sm90.cuh"
#include "fuse/profiling/timeline.cuh"
#include "fuse/schedule/cutlass_pipeline.cuh"
#include "fuse/schedule/persistent_gemm.cuh"


#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cuda_host_adapter.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace fuse {
namespace {

using namespace cute;

using Element = Bf16;
using Fp8Element = Fp8E4m3;
using Accumulator = float;
using N64TileShape = Shape<_128, _64, _64>;
using TileShape = Shape<_128, _128, _64>;
using M64TileShape = Shape<_64, _128, _64>;
using N160TileShape = Shape<_128, _160, _64>;
using N192TileShape = Shape<_128, _192, _64>;
using ProjectionTileShape = Shape<_128, _256, _64>;
using WideN320TileShape = Shape<_128, Int<320>, _64>;
using Fp8TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using M64ClusterShape = Shape<_1, _1, _1>;
using ProjectionClusterShape = Shape<_2, _1, _1>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kBlockM = 128;
constexpr int kBlockN = 128;
constexpr int kA2ALhsCommRows = 32;
constexpr int kA2ALhsBulkSlots = 4;
constexpr int kA2ALhsBulkStageBytes = 48 * 1024;
constexpr int kQkvBulkRows = 64;
constexpr int kQkvBulkSlots = 12;
constexpr int kQkvBulkColumns = 128;
constexpr int kQkvBulkStageBytes =
    kQkvBulkRows * kQkvBulkColumns * sizeof(Element);
constexpr int kAlignment = 16 / sizeof(Element);
constexpr int kFp8Alignment = 16 / sizeof(Fp8Element);

using RasterOptions =
    cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::
        RasterOrderOptions;

RasterOptions raster_option(GemmRaster requested, RasterOptions fallback) {
  switch (requested) {
    case GemmRaster::kAlongM:
      return RasterOptions::AlongM;
    case GemmRaster::kAlongN:
      return RasterOptions::AlongN;
    default:
      return fallback;
  }
}

__host__ __device__ constexpr int64_t stride_or(int64_t stride, int64_t packed) {
  return stride < 0 ? packed : stride;
}

__host__ __device__ constexpr int64_t a_row_stride(const GemmProblem& p) {
  return stride_or(p.stride_a.row, p.k);
}

__host__ __device__ constexpr int64_t a_batch_stride(const GemmProblem& p) {
  return stride_or(p.stride_a.batch, a_row_stride(p) * p.m);
}

__host__ __device__ constexpr int64_t b_row_stride(const GemmProblem& p) {
  return stride_or(p.stride_b.row, p.k);
}

__host__ __device__ constexpr int64_t b_batch_stride(const GemmProblem& p) {
  return stride_or(p.stride_b.batch, b_row_stride(p) * p.n);
}

__host__ __device__ constexpr int64_t d_row_stride(const GemmProblem& p) {
  return stride_or(p.stride_d.row, p.n);
}

__host__ __device__ constexpr int64_t d_batch_stride(const GemmProblem& p) {
  return stride_or(p.stride_d.batch, d_row_stride(p) * p.m);
}

bool supported_problem(const GemmProblem& p) {
  const bool unit_strides =
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1);
  const bool leading_dimensions =
      a_row_stride(p) >= p.k && b_row_stride(p) >= p.k &&
      d_row_stride(p) >= p.n && a_row_stride(p) % kAlignment == 0 &&
      a_batch_stride(p) % kAlignment == 0 &&
      b_row_stride(p) % kAlignment == 0 &&
      b_batch_stride(p) % kAlignment == 0 &&
      d_row_stride(p) % kAlignment == 0 && d_batch_stride(p) % kAlignment == 0;
  const bool batches = p.l == 1 ||
      (a_batch_stride(p) >= a_row_stride(p) * p.m &&
       (b_batch_stride(p) == 0 || b_batch_stride(p) >= b_row_stride(p) * p.n) &&
       d_batch_stride(p) >= d_row_stride(p) * p.m);
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l > 0 && unit_strides &&
      leading_dimensions && batches && !p.transpose_a && !p.transpose_b &&
      p.input_dtype == DType::kBfloat16 && p.weight_dtype == DType::kBfloat16 &&
      p.output_dtype == DType::kBfloat16 &&
      (p.max_swizzle_size == 1 || p.max_swizzle_size == 2 ||
       p.max_swizzle_size == 4 || p.max_swizzle_size == 8);
}

bool supported_fp8_problem(const GemmProblem& p) {
  const bool unit_strides =
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1);
  const bool leading_dimensions =
      a_row_stride(p) >= p.k && b_row_stride(p) >= p.k &&
      d_row_stride(p) >= p.n && a_row_stride(p) % kFp8Alignment == 0 &&
      a_batch_stride(p) % kFp8Alignment == 0 &&
      b_row_stride(p) % kFp8Alignment == 0 &&
      b_batch_stride(p) % kFp8Alignment == 0 &&
      d_row_stride(p) % kAlignment == 0 && d_batch_stride(p) % kAlignment == 0;
  const bool batches = p.l == 1 ||
      (a_batch_stride(p) >= a_row_stride(p) * p.m &&
       (b_batch_stride(p) == 0 || b_batch_stride(p) >= b_row_stride(p) * p.n) &&
       d_batch_stride(p) >= d_row_stride(p) * p.m);
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l > 0 && unit_strides &&
      leading_dimensions && batches && !p.transpose_a && !p.transpose_b &&
      p.input_dtype == DType::kFloat8E4M3 &&
      p.weight_dtype == DType::kFloat8E4M3 &&
      p.output_dtype == DType::kBfloat16 &&
      (p.max_swizzle_size == 1 || p.max_swizzle_size == 2 ||
       p.max_swizzle_size == 4 || p.max_swizzle_size == 8);
}

bool supported_route_base(const UlyssesRoute& route) {
  return route.world_size > 0 && route.world_size <= kMaxWorldSize &&
      route.rank >= 0 && route.rank < route.world_size && route.batch > 0 &&
      route.seq_local > 0 && route.global_seq > 0;
}

// Kernel parameters are private to the new weighted-sequence operators. The
// existing public parameter structs and uniform kernels remain unchanged.
struct WeightedGemmA2AKernelParams {
  const Bf16* lhs = nullptr;
  const Bf16* rhs_nt = nullptr;
  Bf16* local_output = nullptr;
  Bf16* peer_output[kMaxWorldSize]{};
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready = nullptr;
  uint32_t* completion_epoch = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas = 0;
  uint32_t epoch = 0;
  float alpha = 1.0f;
  int32_t executor_rank = 0;
  int32_t logical_source_rank = 0;
  int32_t source_row_begin = 0;
  int32_t global_sequence_begin = 0;
  bool weighted_partition = false;
};

struct WeightedA2AGemmKernelParams {
  const Bf16* peer_input[kMaxWorldSize]{};
  Bf16* input_staging = nullptr;
  const uint32_t* peer_input_ready[kMaxWorldSize]{};
  Bf16* rhs_nt = nullptr;
  Bf16* output = nullptr;
  uint32_t* peer_done_epoch[kMaxWorldSize]{};
  uint32_t* ready = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas = 0;
  A2ALhsGemmPolicy lhs_policy = A2ALhsGemmPolicy::kAuto;
  uint32_t epoch = 0;
  uint32_t input_epoch = 0;
  float alpha = 1.0f;
  int32_t executor_rank = 0;
  int32_t logical_source_rank = 0;
  int32_t source_row_begin = 0;
  int32_t global_sequence_begin = 0;
  bool weighted_partition = false;
};

enum class QkvGemmPolicy {
  kM128N64,
  kM128N128,
  kM128N160,
  kM128N192,
  kM128N256ClusterM2,
  kM128N320ClusterM2,
};

struct QkvPolicyGeometry {
  QkvGemmPolicy policy;
  int32_t tile_n;
  int32_t cluster_m;
};

constexpr std::array<QkvPolicyGeometry, 6> kQkvPolicyGeometries{{
    {QkvGemmPolicy::kM128N64, 64, 1},
    {QkvGemmPolicy::kM128N128, 128, 1},
    {QkvGemmPolicy::kM128N160, 160, 1},
    {QkvGemmPolicy::kM128N192, 192, 1},
    {QkvGemmPolicy::kM128N256ClusterM2, 256, 2},
    {QkvGemmPolicy::kM128N320ClusterM2, 320, 2},
}};

struct QkvWaveCalibrationRow {
  int32_t k;
  // One full compute wave in nanoseconds, indexed like kQkvPolicyGeometries.
  std::array<int32_t, 6> wave_ns;
};

// H200 BF16 primitive calibration for the QKV tile selector.  These are
// pure CUTLASS compute-subgrid p50 times, not fitted coefficients:
//   * H200 SXM, 132 SMs; CUDA events; warmup=5, iterations=20.
//   * comm=24 leaves 108 compute CTAs; comm=32 leaves 100 compute CTAs.
//   * Each calibration shape supplies one full (or nearest legal) wave for
//     exactly one policy.  N64/N128/N160/N192 use cluster-M1;
//     N256/N320 use cluster-M2.
//     N64 uses (M,N)=(128,6912), exactly 108 N64 tiles at comm=24.
//     comm=24 uses (M,N)=(128,13824), (128,17152), (256,13824),
//     (256,17152); comm=32 uses N=12800/15872 with the same M mapping.
//     N64 and N192 were measured once at comm=24.  Their values are duplicated
//     in the comm=32 array only to keep a uniform policy index; the general
//     model treats tile service cost as independent of the worker split, while
//     the mature v6 path excludes both experimental geometries.
// The selector multiplies this measured one-wave latency by the number of
// persistent waves required by the real MxN problem.  This is an
// architecture-level calibration, not a per-shape winner table: runtime M/N
// only determine the number of waves.  Regenerate it with
// qkvproj_a2a_bench, forcing each m128n* policy and timing
// compute_subgrid_cutlass on the calibration shapes documented above,
// whenever the GPU, CUDA/CUTLASS version, or kernel policy changes.
constexpr std::array<QkvWaveCalibrationRow, 5> kQkvWaveCalibrationComm24{{
    {2048, {15312, 25248, 28768, 32736, 31792, 40400}},
    {3072, {18816, 32304, 37264, 43440, 41600, 53296}},
    {4096, {25520, 39136, 46496, 54592, 50656, 65728}},
    {5120, {28752, 45632, 54400, 64304, 60128, 78496}},
    {16384, {78352, 126704, 150656, 181360, 167328, 219744}},
}};

constexpr std::array<QkvWaveCalibrationRow, 5> kQkvWaveCalibrationComm32{{
    {2048, {15312, 24288, 27680, 32736, 31456, 39440}},
    {3072, {18816, 30880, 35840, 43440, 41056, 52304}},
    {4096, {25520, 36768, 44496, 54592, 50256, 64896}},
    {5120, {28752, 43920, 51504, 64304, 59392, 77152}},
    {16384, {78352, 117696, 141040, 181360, 165456, 213024}},
}};

// Incremental cost of one additional 16-KiB route-task wave.  The value is
// the difference between otherwise identical 64-task and 128-task QKV route
// calibrations (14.7 us and 19.0 us) on H200.  It is a communication-kernel
// primitive cost, not a model-shape winner.
constexpr int64_t kQkvRouteTaskWaveNs = 4300;

// Standalone H200 route sweeps reach their useful one-way fabric bandwidth at
// 16 communication CTAs.  The joint model uses this architecture-level
// saturation point to cap its NVLink bandwidth estimate; it is not a tile
// eligibility threshold.
constexpr int32_t kQkvFabricSaturationCommCtas = 16;

bool qkv_pipeline_policy_enabled() {
  const char* value = std::getenv("FUSE_QKV_COMM_POLICY");
  // The physical pipeline model is the production default.  Keep the mature
  // legacy selector available for reproducibility and rollback; unknown values also
  // fall back instead of silently enabling a misspelled experimental mode.
  return value == nullptr || std::strcmp(value, "pipeline") == 0 ||
      std::strcmp(value, "roofline") == 0;
}

QkvGemmPolicy legacy_qkv_gemm_policy(const GemmProblem& problem) {
  return problem.m >= 2048 ? QkvGemmPolicy::kM128N256ClusterM2
                           : QkvGemmPolicy::kM128N128;
}

template <size_t NumRows>
int64_t interpolate_qkv_wave_ns(
    const std::array<QkvWaveCalibrationRow, NumRows>& table,
    int32_t k,
    size_t policy_index) {
  if (k < table.front().k || k > table.back().k) {
    return -1;
  }
  for (size_t row = 0; row < NumRows; ++row) {
    if (k == table[row].k) {
      return table[row].wave_ns[policy_index];
    }
    if (k < table[row].k) {
      const auto& lo = table[row - 1];
      const auto& hi = table[row];
      const int64_t numerator =
          static_cast<int64_t>(hi.wave_ns[policy_index] -
                               lo.wave_ns[policy_index]) *
          (k - lo.k);
      const int64_t denominator = hi.k - lo.k;
      return lo.wave_ns[policy_index] +
          (numerator + denominator / 2) / denominator;
    }
  }
  return table.back().wave_ns[policy_index];
}

struct QkvComputeEstimate {
  QkvGemmPolicy policy = QkvGemmPolicy::kM128N128;
  int32_t tile_n = 128;
  int32_t cluster_m = 1;
  int64_t waves = 0;
  int64_t wave_ns = 0;
  int64_t total_ns = 0;
  bool valid = false;
};

QkvComputeEstimate estimate_qkv_compute(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    bool peer_interleaved,
    bool allow_general_comm = false) {
  const QkvGemmPolicy fallback = legacy_qkv_gemm_policy(problem);
  QkvComputeEstimate best{};
  best.policy = fallback;
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 || problem.l <= 0 ||
      peer_interleaved || sm_count != 132 ||
      ((!allow_general_comm) &&
       num_comm_ctas != 24 && num_comm_ctas != 32)) {
    return best;
  }

  const int32_t compute_ctas = sm_count - num_comm_ctas;
  if (compute_ctas <= 0) {
    return best;
  }
  const auto ceil_div_i64 = [](int64_t value, int64_t divisor) {
    return (value + divisor - 1) / divisor;
  };

  int64_t best_score_ns = std::numeric_limits<int64_t>::max();
  int32_t best_tie_rank = std::numeric_limits<int32_t>::max();
  QkvComputeEstimate best_independent{};
  int64_t best_independent_ns = std::numeric_limits<int64_t>::max();
  for (size_t index = 0; index < kQkvPolicyGeometries.size(); ++index) {
    const auto& candidate = kQkvPolicyGeometries[index];
    // N64 and N192 are general-model candidates.  The mature v6 selector
    // remains bit-for-bit on its original four-policy search space.
    if (!allow_general_comm &&
        (candidate.policy == QkvGemmPolicy::kM128N64 ||
         candidate.policy == QkvGemmPolicy::kM128N192)) {
      continue;
    }
    if (candidate.cluster_m == 2 &&
        (num_comm_ctas % 2 != 0 || compute_ctas % 2 != 0)) {
      continue;
    }
    // The general comm model uses one policy latency independent of the
    // resource split: changing comm CTAs changes the number of resident GEMM
    // workers and therefore the wave count, not the work performed by a tile.
    // The mature v6 path retains its exact comm24/comm32 calibration.
    const int64_t wave_ns = allow_general_comm || num_comm_ctas == 24
        ? interpolate_qkv_wave_ns(
              kQkvWaveCalibrationComm24, problem.k, index)
        : interpolate_qkv_wave_ns(
              kQkvWaveCalibrationComm32, problem.k, index);
    if (wave_ns <= 0) {
      return best;
    }

    const int64_t workers = compute_ctas / candidate.cluster_m;
    const int64_t m_tiles = ceil_div_i64(problem.m, kBlockM);
    const int64_t m_work_units = ceil_div_i64(
        m_tiles, candidate.cluster_m);
    const int64_t n_work_units = ceil_div_i64(
        problem.n, candidate.tile_n);
    const int64_t work_units =
        m_work_units * n_work_units * problem.l;
    const int64_t waves = ceil_div_i64(work_units, workers);
    // Treat every persistent wave as one calibrated full wave.  The score is
    // intentionally small and auditable: it does not fit per-shape winners or
    // use TE-UB results.  Shapes outside the calibrated H200/comm/K domain use
    // the v5 policy instead of extrapolating this model.
    const int64_t score_ns = waves * wave_ns;
    QkvComputeEstimate estimate{};
    estimate.policy = candidate.policy;
    estimate.tile_n = candidate.tile_n;
    estimate.cluster_m = candidate.cluster_m;
    estimate.waves = waves;
    estimate.wave_ns = wave_ns;
    estimate.total_ns = score_ns;
    estimate.valid = true;
    if (candidate.cluster_m == 1 && score_ns < best_independent_ns) {
      best_independent = estimate;
      best_independent_ns = score_ns;
    }
    const int32_t tie_rank = candidate.policy == fallback
        ? 0
        : static_cast<int32_t>(index) + 1;
    if (score_ns < best_score_ns ||
        (score_ns == best_score_ns && tie_rank < best_tie_rank)) {
      best = estimate;
      best_score_ns = score_ns;
      best_tie_rank = tie_rank;
    }
  }
  // Cluster-M2 saves B traffic, but its two CTAs share one cluster progress
  // unit.  Under concurrent route TMA, a delayed peer can hold both CTAs.  If
  // an independent cluster-M1 policy needs the same number of persistent
  // waves and gives up no more than two route service quanta, preserve the
  // independent progress.  The margin is tied to the cluster width and the
  // measured route primitive above; it is not selected from the 96 cases.
  if (allow_general_comm && best.valid && best.cluster_m == 2 &&
      best_independent.valid && best_independent.waves == best.waves &&
      best_independent.total_ns <=
          best.total_ns + best.cluster_m * kQkvRouteTaskWaveNs) {
    return best_independent;
  }
  return best;
}

QkvGemmPolicy select_qkv_wave_time_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    bool peer_interleaved,
    bool allow_general_comm = false) {
  const auto estimate = estimate_qkv_compute(
      problem,
      num_comm_ctas,
      sm_count,
      peer_interleaved,
      allow_general_comm);
  return estimate.valid ? estimate.policy : legacy_qkv_gemm_policy(problem);
}

QkvGemmPolicy select_qkv_gemm_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    bool peer_interleaved) {
  if (const char* value = std::getenv("FUSE_QKV_GEMM_POLICY")) {
    if (std::strcmp(value, "legacy") == 0) {
      return legacy_qkv_gemm_policy(problem);
    }
    if (std::strcmp(value, "m128n64") == 0) {
      return QkvGemmPolicy::kM128N64;
    }
    if (std::strcmp(value, "m128n128") == 0) {
      return QkvGemmPolicy::kM128N128;
    }
    if (std::strcmp(value, "m128n160") == 0) {
      return QkvGemmPolicy::kM128N160;
    }
    if (std::strcmp(value, "m128n192") == 0) {
      return QkvGemmPolicy::kM128N192;
    }
    if (std::strcmp(value, "m128n256") == 0) {
      return QkvGemmPolicy::kM128N256ClusterM2;
    }
    if (std::strcmp(value, "m128n320") == 0) {
      return QkvGemmPolicy::kM128N320ClusterM2;
    }
    if (std::strcmp(value, "wave_time_model") == 0) {
      return select_qkv_wave_time_policy(
          problem, num_comm_ctas, sm_count, peer_interleaved, true);
    }
  }
  const bool pipeline_comm = qkv_pipeline_policy_enabled();
  return select_qkv_wave_time_policy(
      problem,
      num_comm_ctas,
      sm_count,
      peer_interleaved,
      pipeline_comm);
}

// Narrow-N latency geometry for short, under-filled QKV projections.  The
// production selector evaluates it with the same calibrated compute/route
// cost model as every other geometry, without a shape- or comm-specific gate.
using N64Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        N64TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using N64Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        N64TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename N64Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using N64ObservedEpilogue = detail::SignalingEpilogue<N64Epilogue>;
using N64OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N64Mainloop,
    N64ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using N64PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N64Mainloop,
    N64Epilogue,
    cutlass::gemm::PersistentScheduler>;

using BaseEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator,
    Accumulator,
    void,
    LayoutD,
    kAlignment,
    Element,
    LayoutD,
    kAlignment,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Element,
    LayoutA,
    kAlignment,
    Element,
    LayoutB,
    kAlignment,
    Accumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename BaseEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using ObservedEpilogue = detail::SignalingEpilogue<BaseEpilogue>;

using PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BaseMainloop,
    BaseEpilogue,
    cutlass::gemm::PersistentScheduler>;

using A2ALhsInputMainloop = detail::A2ALhsReadyMainloop<BaseMainloop>;
using A2ALhsInputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsInputMainloop,
    BaseEpilogue,
    detail::MonolithicPersistentScheduler>;

using N160Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        N160TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using N160Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        N160TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename N160Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using N160ObservedEpilogue = detail::SignalingEpilogue<N160Epilogue>;
using N160OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N160Mainloop,
    N160ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;

using A2ALhsN160Mainloop = detail::A2ALhsReadyMainloop<N160Mainloop>;
using A2ALhsN160Gemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsN160Mainloop,
    N160Epilogue,
    detail::MonolithicPersistentScheduler>;
using N160PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N160Mainloop,
    N160Epilogue,
    cutlass::gemm::PersistentScheduler>;

// Cluster-M1 candidate between N160 and the cluster-M2 N256 policy.  It
// fills wave-count gaps without introducing a second-CTA progress dependency.
using N192Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        N192TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using N192Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        N192TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename N192Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using N192ObservedEpilogue = detail::SignalingEpilogue<N192Epilogue>;
using N192OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N192Mainloop,
    N192ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using N192PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    N192Mainloop,
    N192Epilogue,
    cutlass::gemm::PersistentScheduler>;

#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryMainloop =
    detail::A2ALhsReadyMainloop<BaseMainloop, 1, true>;
using A2ALhsTelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsTelemetryMainloop,
    BaseEpilogue,
    detail::MonolithicPersistentScheduler>;
#endif
// Small-M dense GEMMs need twice as many M tiles to fill H200.  Keep these
// CTAs independent so an early-ready M64 tile is never held behind its peer.
using M64Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        M64TileShape,
        M64ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

using M64Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        M64TileShape,
        M64ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename M64Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using M64PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    M64Mainloop,
    M64Epilogue,
    cutlass::gemm::PersistentScheduler>;

using A2ALhsM64Mainloop = detail::A2ALhsReadyMainloop<M64Mainloop>;
using A2ALhsM64Gemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsM64Mainloop,
    M64Epilogue,
    detail::MonolithicPersistentScheduler>;
#if FUSE_ENABLE_PROFILING
using A2ALhsM64TelemetryMainloop =
    detail::A2ALhsReadyMainloop<M64Mainloop, 1, true>;
using A2ALhsM64TelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsM64TelemetryMainloop,
    M64Epilogue,
    detail::MonolithicPersistentScheduler>;
#endif
using OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BaseMainloop,
    ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;

// Wide-N dense output routing follows the Flux-style cooperative policy. The
// generic A2A -> GEMM path keeps its independent 128x128 ping-pong policy.
using ProjectionEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using ProjectionMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCount<4>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using ProjectionObservedEpilogue = detail::SignalingEpilogue<ProjectionEpilogue>;
using ProjectionOutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    ProjectionMainloop,
    ProjectionObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using ProjectionPureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    ProjectionMainloop,
    ProjectionEpilogue,
    cutlass::gemm::PersistentScheduler>;

using A2ALhsProjectionMainloop =
    detail::A2ALhsReadyMainloop<ProjectionMainloop>;
using A2ALhsProjectionGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsProjectionMainloop,
    ProjectionEpilogue,
    detail::MonolithicPersistentScheduler>;
#if FUSE_ENABLE_PROFILING
using A2ALhsProjectionTelemetryMainloop =
    detail::A2ALhsReadyMainloop<ProjectionMainloop, 1, true>;
using A2ALhsProjectionTelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsProjectionTelemetryMainloop,
    ProjectionEpilogue,
    detail::MonolithicPersistentScheduler>;
#endif

// Wide-N cluster policy for frontier-aligned persistent waves.
using WideN320Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        WideN320TileShape,
        ProjectionClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using WideN320Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutB,
        kAlignment,
        Accumulator,
        WideN320TileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename WideN320Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using WideN320ObservedEpilogue =
    detail::SignalingEpilogue<WideN320Epilogue>;
using WideN320OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    WideN320Mainloop,
    WideN320ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;

using A2ALhsWideN320Mainloop =
    detail::A2ALhsReadyMainloop<WideN320Mainloop>;
using A2ALhsWideN320Gemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsWideN320Mainloop,
    WideN320Epilogue,
    detail::MonolithicPersistentScheduler>;
using WideN320PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    WideN320Mainloop,
    WideN320Epilogue,
    cutlass::gemm::PersistentScheduler>;
#if FUSE_ENABLE_PROFILING
using A2ALhsWideN320TelemetryMainloop =
    detail::A2ALhsReadyMainloop<WideN320Mainloop, 1, true>;
using A2ALhsWideN320TelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsWideN320TelemetryMainloop,
    WideN320Epilogue,
    detail::MonolithicPersistentScheduler>;
#endif

using Fp8BaseEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Fp8TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    Accumulator,
    Accumulator,
    void,
    LayoutD,
    kAlignment,
    Element,
    LayoutD,
    kAlignment,
    cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;

using Fp8BaseMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    Fp8Element,
    LayoutA,
    kFp8Alignment,
    Fp8Element,
    LayoutB,
    kFp8Alignment,
    Accumulator,
    Fp8TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Fp8BaseEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::CollectiveOp;

using Fp8ObservedEpilogue = detail::SignalingEpilogue<Fp8BaseEpilogue>;
using Fp8OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BaseMainloop,
    Fp8ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BaseMainloop,
    Fp8BaseEpilogue,
    cutlass::gemm::PersistentScheduler>;

__host__ __device__ constexpr int32_t ceil_div(int32_t x, int32_t y) {
  return (x + y - 1) / y;
}

CUTLASS_DEVICE void finalize_weighted_cp_epoch(
    uint32_t* const* peer_done_epoch,
    int32_t world_size,
    int32_t executor_rank,
    uint32_t epoch) {
  if (blockIdx.x != 0 || threadIdx.x >= 32) {
    return;
  }
  const int32_t lane = static_cast<int32_t>(threadIdx.x);
  if (lane == 0) {
    // The cooperative grid barrier has completed every routed store issued by
    // this executor. Publish those stores as one cross-rank completion point.
    detail::fence_system();
  }
  __syncwarp();
  if (lane < world_size) {
    detail::store_release_system(
        peer_done_epoch[lane] + executor_rank * kReadyFlagStride,
        epoch);
  }
  __syncwarp();
  const uint32_t* local_done = peer_done_epoch[executor_rank];
  if (lane < world_size) {
    while (detail::load_acquire_system(
               local_done + lane * kReadyFlagStride) < epoch) {
      __nanosleep(64);
    }
  }
  __syncwarp();
}

// Pull the sequence rows owned by this destination rank from every
// head-sharded peer output and place them directly in the row-major GEMM A
// staging matrix. A peer completes one contiguous K interval for each M tile;
// the CUTLASS producer can therefore advance peer by peer without waiting for
// the full inverse A2A.
#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsCommTimelineArguments {};

template <>
struct A2ALhsCommTimelineArguments<true> {
  A2AGemmPeerTimeline* peer_timeline = nullptr;
  int32_t peer_timeline_capacity = 0;
};
#endif

#if FUSE_ENABLE_PROFILING
template <bool Instrumented>
struct A2ALhsCommStageSample {};

template <>
struct A2ALhsCommStageSample<true> {
  uint64_t task_begin = 0;
  uint64_t input_ready = 0;
  uint64_t g2s_issue = 0;
  uint64_t g2s_done = 0;
  uint64_t s2g_issue = 0;
  uint64_t s2g_done = 0;
  int32_t comm_cta = 0;
  int32_t comm_slot = 0;
  int32_t task_id = 0;
  int32_t row_chunk = 0;
  int32_t copy_rows = 0;
  int32_t source_rank = 0;
  int32_t copy_path = 0;
};
#endif

cudaError_t make_a2a_lhs_store_tma_3d(
    CUtensorMap* tensor_map,
    void* pointer,
    int32_t inner_u64,
    int32_t total_groups,
    int32_t rows,
    int32_t peer_groups,
    int32_t box_rows) {
  const uint64_t global_dims[3] = {
      static_cast<uint64_t>(inner_u64),
      static_cast<uint64_t>(total_groups),
      static_cast<uint64_t>(rows)};
  const uint64_t global_strides[2] = {
      static_cast<uint64_t>(inner_u64) * sizeof(uint64_t),
      static_cast<uint64_t>(inner_u64) * total_groups * sizeof(uint64_t)};
  const uint32_t box_dims[3] = {
      static_cast<uint32_t>(inner_u64),
      static_cast<uint32_t>(peer_groups),
      static_cast<uint32_t>(box_rows)};
  constexpr uint32_t element_strides[3] = {1, 1, 1};
  const CUresult result =
      CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
          tensor_map,
          CU_TENSOR_MAP_DATA_TYPE_UINT64,
          3,
          pointer,
          global_dims,
          global_strides,
          box_dims,
          element_strides,
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return result == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue;
}

// A2A -> GEMM: remote input staging and per-peer K-shard publication.
template <
    int32_t ReadyBlockM,
    class ParamsType = A2AGemmParams,
    bool Heterogeneous = false,
    bool FinalizeAcrossRanks = false
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
struct A2ALhsInputCommT {
  static constexpr int32_t kReadyBlockM = ReadyBlockM;
  static constexpr size_t SharedStorageBytes =
      kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
      kA2ALhsBulkSlots * sizeof(uint64_t);
  struct Arguments
#if FUSE_ENABLE_PROFILING
      : A2ALhsCommTimelineArguments<Instrumented>
#endif
  {
    ParamsType params{};
    CUtensorMap store_tma_full{};
    int32_t comm_rows = kA2ALhsCommRows;
    int32_t m_window = 2;
    int32_t store_peer_groups = 0;
    int32_t store_rows = 0;
    bool use_bulk = false;
    bool use_tensor_store = false;
  };
  using Params = Arguments;
  static constexpr bool kNeedsGridFinalize = FinalizeAcrossRanks;

  CUTLASS_HOST_DEVICE static int32_t logical_source_rank(
      const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.logical_source_rank;
    }
    return p.route.rank;
  }

  CUTLASS_HOST_DEVICE static int32_t source_row_begin(
      const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.source_row_begin;
    }
    return 0;
  }

  CUTLASS_HOST_DEVICE static int32_t executor_rank(const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.executor_rank;
    }
    return p.route.rank;
  }

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    if (route.world_size <= 0) {
      return cudaErrorInvalidValue;
    }
    const int32_t shard_width = route.local_heads * route.head_dim;
    const int32_t row_bytes = shard_width * sizeof(Element);
    const int32_t max_rows = row_bytes > 0
        ? kA2ALhsBulkStageBytes / row_bytes
        : 0;
    args.comm_rows = min(ReadyBlockM, max_rows);
    args.use_bulk = args.comm_rows > 0 && row_bytes % 16 == 0 &&
        route.seq_local % ReadyBlockM == 0 &&
        source_row_begin(p) % ReadyBlockM == 0 &&
        (!route.causal_load_balanced ||
         (route.seq_local / 2) % ReadyBlockM == 0);
    if (!args.use_bulk) {
      args.comm_rows = kA2ALhsCommRows;
      return cudaSuccess;
    }

    constexpr int32_t kTensorStoreBytes = 8 * 1024;
    if (row_bytes > 1024) {
      return cudaSuccess;
    }
    args.store_rows = min(args.comm_rows, kTensorStoreBytes / row_bytes);
    if (args.store_rows < 2) {
      return cudaSuccess;
    }

    // Store a narrow peer shard as a small 3D tensor instead of one TMA per
    // row. Four BF16 values form one raw 64-bit tensor element; the first two
    // dimensions factor the contiguous K shard and the third is M.
    const int32_t shard_u64 = row_bytes / sizeof(uint64_t);
    int32_t inner_u64 = 0;
    for (int32_t candidate = 256; candidate >= 2; candidate >>= 1) {
      if (shard_u64 % candidate == 0) {
        inner_u64 = candidate;
        break;
      }
    }
    const int32_t peer_groups = inner_u64 > 0
        ? shard_u64 / inner_u64
        : 0;
    if (inner_u64 == 0 || peer_groups <= 0 || peer_groups > 256) {
      return cudaSuccess;
    }

    const int32_t total_groups = peer_groups * route.world_size;
    const cudaError_t status = make_a2a_lhs_store_tma_3d(
        &args.store_tma_full,
        p.input_staging,
        inner_u64,
        total_groups,
        p.gemm.m,
        peer_groups,
        args.store_rows);
    if (status != cudaSuccess) {
      return cudaSuccess;
    }
    args.store_peer_groups = peer_groups;
    args.use_tensor_store = true;
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    const auto& route = p.route;
    bool pointers = p.input_staging && p.rhs_nt && p.output && p.ready;
    for (int32_t peer = 0; peer < route.world_size && pointers; ++peer) {
      pointers = p.peer_input[peer] != nullptr &&
          (p.input_epoch == 0 || p.peer_input_ready[peer] != nullptr);
      if constexpr (Heterogeneous) {
        pointers = pointers && p.peer_done_epoch[peer] != nullptr;
      }
    }
    bool rows_supported = p.gemm.m == route.batch * route.seq_local;
    bool sequence_supported =
        route.global_seq == route.seq_local * route.world_size;
    if constexpr (Heterogeneous) {
      const bool rank_supported = p.executor_rank >= 0 &&
          p.executor_rank < route.world_size &&
          p.logical_source_rank >= 0 &&
          p.logical_source_rank < route.world_size;
      if (p.weighted_partition) {
        rows_supported = rank_supported &&
            p.gemm.m == route.batch * route.seq_local &&
            p.source_row_begin == 0 && p.global_sequence_begin >= 0;
        sequence_supported = !route.causal_load_balanced &&
            p.global_sequence_begin + route.seq_local <= route.global_seq;
      } else {
        rows_supported = rank_supported && p.source_row_begin >= 0 &&
            p.source_row_begin + p.gemm.m <=
                route.batch * route.seq_local;
      }
    }
    return pointers && p.epoch > 0 && supported_problem(p.gemm) &&
        supported_route_base(route) &&
        route.kind == RouteKind::kHeadToSequence &&
        route.direction == RouteDirection::kInverse &&
        route.q_heads > 0 && route.local_heads > 0 &&
        route.q_heads == route.local_heads * route.world_size &&
        route.head_dim > 0 && route.head_dim % kAlignment == 0 &&
        sequence_supported &&
        (!route.causal_load_balanced || route.seq_local % 2 == 0) &&
        p.gemm.l == 1 && rows_supported &&
        p.gemm.k == route.q_heads * route.head_dim &&
        a_row_stride(p.gemm) == p.gemm.k;
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  __host__ __device__ static int32_t arrivals_per_peer(
      const Params& args) {
    return ceil_div(ReadyBlockM, args.comm_rows);
  }

  CUTLASS_DEVICE static void publish_ready(
      const Params& args,
      int32_t tile_m,
      int32_t peer_slot) {
    const auto& p = args.params;
    const auto& route = p.route;
    auto* ready = p.ready +
        (static_cast<int64_t>(tile_m) * route.world_size + peer_slot) *
            kReadyFlagStride;
    detail::add_release_gpu(ready);
  }

#if FUSE_ENABLE_PROFILING
  CUTLASS_DEVICE static void publish_ready_instrumented(
      const Params& args,
      int32_t tile_m,
      int32_t peer_slot,
      const A2ALhsCommStageSample<true>& sample) {
    const auto& p = args.params;
    const auto& route = p.route;
    auto* ready = p.ready +
        (static_cast<int64_t>(tile_m) * route.world_size + peer_slot) *
            kReadyFlagStride;
    const uint64_t release_issue = static_cast<unsigned long long>(
        detail::read_global_timer());
    const uint32_t old = detail::add_release_gpu_fetch_old(ready);
    const uint64_t release_done = static_cast<unsigned long long>(
        detail::read_global_timer());
    const uint32_t target = p.epoch * arrivals_per_peer(args);
    const int32_t index = tile_m * route.world_size + peer_slot;
    if (old + 1 == target && args.peer_timeline && index >= 0 &&
        index < args.peer_timeline_capacity) {
      auto& event = args.peer_timeline[index];
      event.release = release_done;
      event.task_begin = sample.task_begin;
      event.input_ready = sample.input_ready;
      event.g2s_issue = sample.g2s_issue;
      event.g2s_done = sample.g2s_done;
      event.s2g_issue = sample.s2g_issue;
      event.s2g_done = sample.s2g_done;
      event.publish_issue = release_issue;
      event.comm_cta = sample.comm_cta;
      event.comm_slot = sample.comm_slot;
      event.task_id = sample.task_id;
      event.row_chunk = sample.row_chunk;
      event.copy_rows = sample.copy_rows;
      event.source_rank = sample.source_rank;
      event.copy_path = sample.copy_path;
      event.comm_valid = 1;
    }
  }
#endif

  CUTLASS_DEVICE static int32_t global_sequence_row(
      const ParamsType& p,
      int32_t local_sequence) {
    const auto& route = p.route;
    if constexpr (Heterogeneous) {
      if (p.weighted_partition) {
        return p.global_sequence_begin + local_sequence;
      }
    }
    if (!route.causal_load_balanced) {
      return logical_source_rank(p) * route.seq_local + local_sequence;
    }
    const int32_t chunk_rows = route.seq_local / 2;
    const int32_t chunk = local_sequence < chunk_rows
        ? logical_source_rank(p)
        : 2 * route.world_size - logical_source_rank(p) - 1;
    const int32_t row_in_chunk = local_sequence < chunk_rows
        ? local_sequence
        : local_sequence - chunk_rows;
    return chunk * chunk_rows + row_in_chunk;
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas) {
    const auto& p = args.params;
    const auto& route = p.route;
    const int32_t m_tiles = ceil_div(p.gemm.m, ReadyBlockM);
    const int32_t chunks_per_tile = arrivals_per_peer(args);
    const int32_t tasks_per_m = route.world_size * chunks_per_tile;
    const int32_t tasks = m_tiles * tasks_per_m;
    const int32_t shard_width = route.local_heads * route.head_dim;
    constexpr int32_t elements_per_vector = kAlignment;
    const int32_t vectors_per_row = shard_width / elements_per_vector;
    int32_t waited_peer = -1;

    if (args.use_bulk) {
      extern __shared__ char dynamic_smem[];
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kA2ALhsBulkSlots * kA2ALhsBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot >= kA2ALhsBulkSlots) {
        return;
      }
      auto* stage = reinterpret_cast<Element*>(
          dynamic_smem + slot * kA2ALhsBulkStageBytes);
      uint64_t* barrier = barriers + slot;
      int32_t phase = 0;
      if (lane == 0) {
        cute::initialize_barrier(*barrier, 1);
        if (args.use_tensor_store) {
          cute::prefetch_tma_descriptor(&args.store_tma_full);
        }
      }
      __syncwarp();

      const int32_t task_peer_count = route.world_size;
      const int32_t bulk_tasks =
          m_tiles * task_peer_count * chunks_per_tile;
      const int32_t task_stride = comm_ctas * kA2ALhsBulkSlots;
      for (int32_t task = slot * comm_ctas + comm_id;
           task < bulk_tasks;
           task += task_stride) {
#if FUSE_ENABLE_PROFILING
        A2ALhsCommStageSample<Instrumented> sample{};
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.task_begin = detail::read_global_timer();
            sample.comm_cta = comm_id;
            sample.comm_slot = slot;
            sample.task_id = task;
            sample.copy_path = args.use_tensor_store ? 2 : 1;
          }
        }
#endif
        const int32_t m_window = args.m_window;
        const int32_t full_windows = m_tiles / m_window;
        const int32_t tasks_per_window =
            m_window * task_peer_count * chunks_per_tile;
        const int32_t full_window_tasks = full_windows * tasks_per_window;
        int32_t tile_m = 0;
        int32_t peer_slot = 0;
        int32_t row_chunk = 0;
        if (task < full_window_tasks) {
          const int32_t window = task / tasks_per_window;
          const int32_t task_in_window = task - window * tasks_per_window;
          peer_slot = task_in_window / (m_window * chunks_per_tile);
          const int32_t task_in_peer =
              task_in_window - peer_slot * m_window * chunks_per_tile;
          const int32_t m_in_window = task_in_peer / chunks_per_tile;
          row_chunk = task_in_peer - m_in_window * chunks_per_tile;
          tile_m = window * m_window + m_in_window;
        } else {
          const int32_t tail = task - full_window_tasks;
          const int32_t tail_m_tiles = m_tiles - full_windows * m_window;
          peer_slot = tail / (tail_m_tiles * chunks_per_tile);
          const int32_t task_in_peer =
              tail - peer_slot * tail_m_tiles * chunks_per_tile;
          const int32_t m_in_window = task_in_peer / chunks_per_tile;
          row_chunk = task_in_peer - m_in_window * chunks_per_tile;
          tile_m = full_windows * m_window + m_in_window;
        }
        const int32_t row_in_tile = row_chunk * args.comm_rows;
        const int32_t source_peer = route.cyclic_peer_order
            ? (logical_source_rank(p) + peer_slot) % route.world_size
            : peer_slot;
        const int32_t m_begin = tile_m * ReadyBlockM + row_in_tile;
        const int32_t copy_rows = max(
            0,
            min(min(args.comm_rows, ReadyBlockM - row_in_tile),
                p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.row_chunk = row_chunk;
            sample.copy_rows = copy_rows;
            sample.source_rank = source_peer;
          }
        }
#endif

        if (p.input_epoch != 0 && source_peer != waited_peer) {
          detail::wait_acquire_system(
              p.peer_input_ready[source_peer], p.input_epoch, lane);
          waited_peer = source_peer;
        }

#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          if (lane == 0) {
            sample.input_ready = detail::read_global_timer();
          }
        }
#endif

        if (lane == 0 && copy_rows > 0) {
          const int32_t logical_row = source_row_begin(p) + m_begin;
          const int32_t batch = logical_row / route.seq_local;
          const int32_t local_sequence =
              logical_row - batch * route.seq_local;
          const int32_t source_sequence =
              global_sequence_row(p, local_sequence);
          const auto* source = p.peer_input[source_peer] +
              (static_cast<int64_t>(batch) * route.global_seq +
               source_sequence) * shard_width;
          const int32_t copy_bytes =
              copy_rows * shard_width * sizeof(Element);
          cute::set_barrier_transaction_bytes(*barrier, copy_bytes);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_issue = detail::read_global_timer();
          }
#endif
          cute::SM90_BULK_COPY_G2S::copy(
              source, barrier, stage, copy_bytes);
          cute::wait_barrier(*barrier, phase);
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.g2s_done = detail::read_global_timer();
          }
#endif
          phase ^= 1;
          cute::tma_store_fence();
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.s2g_issue = detail::read_global_timer();
          }
#endif
          if (args.use_tensor_store) {
            int32_t row = 0;
            for (; row + args.store_rows <= copy_rows;
                 row += args.store_rows) {
              cute::SM90_TMA_STORE_3D::copy(
                  &args.store_tma_full,
                  stage + static_cast<int64_t>(row) * shard_width,
                  0,
                  peer_slot * args.store_peer_groups,
                  m_begin + row);
            }
            if (row > 0) {
              cute::tma_store_arrive();
            }
            int32_t residual_ops = 0;
            for (; row < copy_rows; ++row) {
              cute::SM90_BULK_COPY_S2G::copy(
                  stage + static_cast<int64_t>(row) * shard_width,
                  p.input_staging +
                      static_cast<int64_t>(m_begin + row) * p.gemm.k +
                      peer_slot * shard_width,
                  shard_width * sizeof(Element));
              if (++residual_ops == 8) {
                cute::tma_store_arrive();
                residual_ops = 0;
              }
            }
            if (residual_ops != 0) {
              cute::tma_store_arrive();
            }
          } else {
            for (int32_t row = 0; row < copy_rows; ++row) {
              cute::SM90_BULK_COPY_S2G::copy(
                  stage + static_cast<int64_t>(row) * shard_width,
                  p.input_staging +
                      static_cast<int64_t>(m_begin + row) * p.gemm.k +
                      peer_slot * shard_width,
                  shard_width * sizeof(Element));
              if ((row & 7) == 7) {
                cute::tma_store_arrive();
              }
            }
            if ((copy_rows & 7) != 0) {
              cute::tma_store_arrive();
            }
          }
          // The ready flag is consumed by another CTA.  Waiting only for the
          // source-SMEM read would allow that consumer to observe ready before
          // the destination-global writes have completed.
          detail::tma_store_wait_all();
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            sample.s2g_done = detail::read_global_timer();
            publish_ready_instrumented(args, tile_m, peer_slot, sample);
          } else
#endif
          {
            publish_ready(args, tile_m, peer_slot);
          }
        } else if (lane == 0) {
#if FUSE_ENABLE_PROFILING
          if constexpr (Instrumented) {
            publish_ready_instrumented(args, tile_m, peer_slot, sample);
          } else
#endif
          {
            publish_ready(args, tile_m, peer_slot);
          }
        }
        __syncwarp();
      }
      if (lane == 0) {
        cutlass::arch::ClusterBarrier::invalidate(barrier);
      }
      return;
    }

    for (int32_t task = comm_id; task < tasks; task += comm_ctas) {
#if FUSE_ENABLE_PROFILING
      A2ALhsCommStageSample<Instrumented> sample{};
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.task_begin = detail::read_global_timer();
          sample.comm_cta = comm_id;
          sample.comm_slot = -1;
          sample.task_id = task;
          sample.copy_path = 0;
        }
      }
#endif
      const int32_t tile_m = task / tasks_per_m;
      const int32_t task_in_m = task - tile_m * tasks_per_m;
      const int32_t peer_slot = task_in_m / chunks_per_tile;
      const int32_t row_chunk = task_in_m - peer_slot * chunks_per_tile;
      const int32_t source_peer = route.cyclic_peer_order
          ? (logical_source_rank(p) + peer_slot) % route.world_size
          : peer_slot;
      const int32_t m_begin = tile_m * ReadyBlockM +
          row_chunk * args.comm_rows;
      const int32_t copy_rows = max(
          0, min(args.comm_rows, p.gemm.m - m_begin));

#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.row_chunk = row_chunk;
          sample.copy_rows = copy_rows;
          sample.source_rank = source_peer;
        }
      }
#endif

      if (p.input_epoch != 0 && source_peer != waited_peer) {
        if (threadIdx.x < 32) {
          detail::wait_acquire_system(
              p.peer_input_ready[source_peer], p.input_epoch,
              static_cast<int32_t>(threadIdx.x));
        }
        __syncthreads();
        waited_peer = source_peer;
      }
#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented) {
        if (threadIdx.x == 0) {
          sample.input_ready = detail::read_global_timer();
          sample.g2s_issue = sample.input_ready;
        }
      }
#endif

      const auto* source =
          reinterpret_cast<const uint4*>(p.peer_input[source_peer]);
      auto* destination = reinterpret_cast<uint4*>(p.input_staging);
      const int32_t vector_count = copy_rows * vectors_per_row;
      for (int32_t index = static_cast<int32_t>(threadIdx.x);
           index < vector_count;
           index += static_cast<int32_t>(blockDim.x)) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_k = index - row * vectors_per_row;
        const int32_t destination_row = m_begin + row;
        const int32_t logical_row =
            source_row_begin(p) + destination_row;
        const int32_t batch = logical_row / route.seq_local;
        const int32_t local_sequence =
            logical_row - batch * route.seq_local;
        const int32_t source_sequence =
            global_sequence_row(p, local_sequence);
        const int64_t src =
            (static_cast<int64_t>(batch) * route.global_seq +
             source_sequence) *
                vectors_per_row +
            vector_k;
        const int64_t dst =
            static_cast<int64_t>(destination_row) *
                (p.gemm.k / kAlignment) +
            peer_slot * vectors_per_row + vector_k;
        destination[dst] = source[src];
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented) {
          sample.g2s_done = detail::read_global_timer();
          publish_ready_instrumented(args, tile_m, peer_slot, sample);
        } else
#endif
        {
          publish_ready(args, tile_m, peer_slot);
        }
      }
      __syncthreads();
    }
  }

  CUTLASS_DEVICE void finalize(const Params& args) {
    if constexpr (Heterogeneous) {
      const auto& p = args.params;
      finalize_weighted_cp_epoch(
          p.peer_done_epoch,
          p.route.world_size,
          executor_rank(p),
          p.epoch);
    }
  }
};

using A2ALhsInputComm = A2ALhsInputCommT<kBlockM>;
using A2ALhsM64InputComm = A2ALhsInputCommT<64>;
template <int32_t ReadyBlockM, bool Finalize>
using WeightedA2ALhsInputComm = A2ALhsInputCommT<
    ReadyBlockM,
    WeightedA2AGemmKernelParams,
    true,
    Finalize>;
#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryInputComm =
    A2ALhsInputCommT<kBlockM, A2AGemmParams, false, false, true>;
using A2ALhsM64TelemetryInputComm =
    A2ALhsInputCommT<64, A2AGemmParams, false, false, true>;
#endif

using A2ALhsGemmKernel =
    detail::MonolithicGemm<A2ALhsInputGemm, A2ALhsInputComm>;
using A2ALhsN160GemmKernel =
    detail::MonolithicGemm<A2ALhsN160Gemm, A2ALhsInputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsTelemetryBase =
    detail::MonolithicGemm<A2ALhsTelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsTelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsTelemetryBase>;
#endif
using A2ALhsProjectionGemmKernel =
    detail::MonolithicGemm<A2ALhsProjectionGemm, A2ALhsInputComm>;
using A2ALhsWideN320GemmKernel =
    detail::MonolithicGemm<A2ALhsWideN320Gemm, A2ALhsInputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsProjectionTelemetryBase = detail::MonolithicGemm<
    A2ALhsProjectionTelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsProjectionTelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsProjectionTelemetryBase>;
using A2ALhsWideN320TelemetryBase = detail::MonolithicGemm<
    A2ALhsWideN320TelemetryGemm, A2ALhsTelemetryInputComm>;
using A2ALhsWideN320TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsWideN320TelemetryBase>;
#endif
using A2ALhsM64GemmKernel =
    detail::MonolithicGemm<A2ALhsM64Gemm, A2ALhsM64InputComm>;
#if FUSE_ENABLE_PROFILING
using A2ALhsM64TelemetryBase =
    detail::MonolithicGemm<
        A2ALhsM64TelemetryGemm, A2ALhsM64TelemetryInputComm>;
using A2ALhsM64TelemetryKernel =
    detail::RoleTelemetryKernel<A2ALhsM64TelemetryBase>;
#endif
static_assert(
    sizeof(typename A2ALhsGemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsN160GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "N160 A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsProjectionGemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "wide A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsWideN320GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "N320 A2A LHS monolithic shared storage must hold every bulk slot");
static_assert(
    sizeof(typename A2ALhsM64GemmKernel::SharedStorage) >=
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
            kA2ALhsBulkSlots * sizeof(uint64_t),
    "M64 A2A LHS monolithic shared storage must hold every bulk slot");

// GEMM -> A2A: consume published output tiles and route Q/K/V to peers.
template <
    class ParamsType,
    bool IsFp8,
    int32_t BlockM,
    int32_t BlockN,
    int32_t MTilesPerTask = 1,
    bool PeerInterleaved = false,
    int32_t CopyBlockN = BlockN,
    bool FinalizeAcrossRanks = true,
    bool Heterogeneous = false>
struct QkvGqaPackCommT {
  struct Arguments {
    ParamsType params{};
    cute::TmaDescriptor local_output_tma{};
    cute::TmaDescriptor peer_output_tma[kMaxWorldSize][3]{};
    bool use_tma = false;
    bool use_tma_store = false;
  };
  using Params = Arguments;
  static constexpr size_t SharedStorageBytes =
      kQkvBulkSlots * kQkvBulkStageBytes +
      kQkvBulkSlots * sizeof(uint64_t);
  static constexpr bool kNeedsGridFinalize = FinalizeAcrossRanks;

  CUTLASS_HOST_DEVICE static int32_t logical_source_rank(
      const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.logical_source_rank;
    }
    return p.route.rank;
  }

  CUTLASS_HOST_DEVICE static int32_t source_row_begin(
      const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.source_row_begin;
    }
    return 0;
  }

  CUTLASS_HOST_DEVICE static int32_t executor_rank(const ParamsType& p) {
    if constexpr (Heterogeneous) {
      return p.executor_rank;
    }
    return p.route.rank;
  }

  CUTLASS_DEVICE static int64_t destination_row(
      const ParamsType& p,
      int32_t source_row) {
    const int32_t logical_row = source_row_begin(p) + source_row;
    const int32_t batch = logical_row / p.route.seq_local;
    const int32_t local_sequence =
        logical_row - batch * p.route.seq_local;
    int32_t sequence_begin = logical_source_rank(p) * p.route.seq_local;
    if constexpr (Heterogeneous) {
      if (p.weighted_partition) {
        sequence_begin = p.global_sequence_begin;
      }
    }
    return static_cast<int64_t>(batch) * p.route.global_seq +
        sequence_begin + local_sequence;
  }

  static cudaError_t initialize(Arguments& args) {
    const auto& p = args.params;
    args.use_tma = p.route.head_dim == kQkvBulkColumns &&
        d_row_stride(p.gemm) >= p.gemm.n &&
        d_row_stride(p.gemm) % kAlignment == 0;
    if (!args.use_tma) {
      return cudaSuccess;
    }
    const uint64_t global_dims[2] = {
        static_cast<uint64_t>(p.gemm.n),
        static_cast<uint64_t>(p.gemm.m)};
    const uint64_t global_strides[1] = {
        static_cast<uint64_t>(d_row_stride(p.gemm)) * sizeof(Element)};
    constexpr uint32_t box_dims[2] = {
        kQkvBulkColumns, kQkvBulkRows};
    constexpr uint32_t element_strides[2] = {1, 1};
    CUresult result =
        CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.local_output_tma,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<Element*>(p.local_output),
            global_dims,
            global_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (result != CUDA_SUCCESS) {
      return cudaErrorInvalidValue;
    }

    args.use_tma_store = p.gemm.m % kQkvBulkRows == 0 &&
        p.route.seq_local % kQkvBulkRows == 0 &&
        source_row_begin(p) % kQkvBulkRows == 0;
    if (!args.use_tma_store) {
      return cudaSuccess;
    }
    const int32_t q_local_width =
        p.route.q_heads / p.route.world_size * p.route.head_dim;
    const int32_t kv_local_width =
        p.route.kv_heads / p.route.world_size * p.route.head_dim;
    const uint64_t output_rows =
        static_cast<uint64_t>(p.route.batch) * p.route.global_seq;
    for (int32_t peer = 0; peer < p.route.world_size; ++peer) {
      const int32_t descriptor_count = p.route.defer_v_a2a ? 1 : 3;
      for (int32_t segment = 0; segment < descriptor_count; ++segment) {
        const int32_t segment_width = p.route.defer_v_a2a
            ? q_local_width + kv_local_width
            : (segment == 0 ? q_local_width : kv_local_width);
        const int64_t segment_offset = p.route.defer_v_a2a || segment == 0
            ? 0
            : static_cast<int64_t>(output_rows) * q_local_width +
                (segment == 2
                     ? static_cast<int64_t>(output_rows) * kv_local_width
                     : 0);
        const uint64_t destination_dims[2] = {
            static_cast<uint64_t>(segment_width), output_rows};
        const uint64_t destination_strides[1] = {
            static_cast<uint64_t>(segment_width) * sizeof(Element)};
        result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &args.peer_output_tma[peer][segment],
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            const_cast<Element*>(p.peer_output[peer] + segment_offset),
            destination_dims,
            destination_strides,
            box_dims,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (result != CUDA_SUCCESS) {
          return cudaErrorInvalidValue;
        }
      }
    }
    return cudaSuccess;
  }

  static bool can_implement(const Arguments& args) {
    const auto& p = args.params;
    bool pointers = supported_route_base(p.route) && p.lhs && p.rhs_nt &&
        p.local_output && p.ready;
    for (int32_t peer = 0; peer < p.route.world_size && pointers; ++peer) {
      pointers = p.peer_output[peer] && p.peer_route_done_epoch[peer];
    }
    const int32_t packed_heads = p.route.q_heads + 2 * p.route.kv_heads;
    const bool problem_supported = IsFp8
        ? supported_fp8_problem(p.gemm)
        : supported_problem(p.gemm);
    bool rows_supported =
        p.gemm.m == p.route.batch * p.route.seq_local;
    bool sequence_supported =
        p.route.global_seq == p.route.seq_local * p.route.world_size;
    if constexpr (Heterogeneous) {
      const bool rank_supported = !IsFp8 && !p.route.defer_v_a2a &&
          p.executor_rank >= 0 &&
          p.executor_rank < p.route.world_size &&
          p.logical_source_rank >= 0 &&
          p.logical_source_rank < p.route.world_size;
      if (p.weighted_partition) {
        rows_supported = rank_supported &&
            p.gemm.m == p.route.batch * p.route.seq_local &&
            p.source_row_begin == 0 && p.global_sequence_begin >= 0;
        sequence_supported =
            p.global_sequence_begin + p.route.seq_local <=
                p.route.global_seq;
      } else {
        rows_supported = rank_supported && p.source_row_begin >= 0 &&
            p.source_row_begin + p.gemm.m <=
                p.route.batch * p.route.seq_local;
      }
    }
    return pointers && p.epoch > 0 && problem_supported &&
        p.route.kind == RouteKind::kQkvGqaPack &&
        p.route.direction == RouteDirection::kForward && p.gemm.l == 1 &&
        p.route.q_heads > 0 && p.route.kv_heads > 0 &&
        p.route.q_heads % p.route.kv_heads == 0 &&
        p.route.q_heads % p.route.world_size == 0 &&
        p.route.kv_heads % p.route.world_size == 0 &&
        p.route.head_dim > 0 && p.route.head_dim % kAlignment == 0 &&
        rows_supported &&
        sequence_supported &&
        p.gemm.n == packed_heads * p.route.head_dim &&
        p.gemm.n % kAlignment == 0 &&
        p.route.qkv_peer_interleaved == PeerInterleaved &&
        (!PeerInterleaved || p.gemm.raster != GemmRaster::kAlongN);
  }

  static Params to_underlying_arguments(const Arguments& args) { return args; }

  CUTLASS_DEVICE static int32_t source_feature(
      const UlyssesRoute& route,
      int32_t destination_rank,
      int32_t segment,
      int32_t local_head,
      int32_t head_offset) {
    const int32_t q_width = route.q_heads * route.head_dim;
    const int32_t kv_width = route.kv_heads * route.head_dim;
    const int32_t segment_base =
        segment == 0 ? 0 : q_width + (segment == 2 ? kv_width : 0);
    const int32_t global_heads =
        segment == 0 ? route.q_heads : route.kv_heads;
    const int32_t local_heads = global_heads / route.world_size;
    const int32_t physical_head =
        PeerInterleaved && segment < 2
        ? local_head * route.world_size + destination_rank
        : destination_rank * local_heads + local_head;
    return segment_base + physical_head * route.head_dim + head_offset;
  }

  CUTLASS_DEVICE void run(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas,
      bool wait_for_gemm) {
    const auto& p = args.params;
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t m_groups = ceil_div(m_tiles, MTilesPerTask);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t q_local_heads = p.route.q_heads / p.route.world_size;
    const int32_t kv_local_heads = p.route.kv_heads / p.route.world_size;
    const int32_t route_heads =
        q_local_heads + (p.route.defer_v_a2a ? 1 : 2) * kv_local_heads;
    const int32_t chunks_per_head = ceil_div(p.route.head_dim, CopyBlockN);
    const int32_t head_chunks = route_heads * chunks_per_head;
    const int32_t tasks =
        p.route.world_size * m_groups * head_chunks;
    const int64_t source_row_vectors = d_row_stride(p.gemm) / kAlignment;
    const int32_t q_local_width = q_local_heads * p.route.head_dim;
    const int32_t kv_local_width = kv_local_heads * p.route.head_dim;
    const int64_t segment_rows =
        static_cast<int64_t>(p.route.batch) * p.route.global_seq;

    if (args.use_tma) {
      static_assert(CopyBlockN >= kQkvBulkColumns);
      extern __shared__ char dynamic_smem[];
      auto* stages = reinterpret_cast<Element*>(dynamic_smem);
      auto* barriers = reinterpret_cast<uint64_t*>(
          dynamic_smem + kQkvBulkSlots * kQkvBulkStageBytes);
      const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
      const int32_t slot = static_cast<int32_t>(threadIdx.x) >> 5;
      if (slot < kQkvBulkSlots) {
        Element* stage = stages +
            static_cast<int64_t>(slot) * kQkvBulkRows * kQkvBulkColumns;
        uint64_t* barrier = barriers + slot;
        int32_t phase = 0;
        if (lane == 0) {
          cute::initialize_barrier(*barrier, 1);
        }
        __syncwarp();

        const int32_t m_chunks = ceil_div(p.gemm.m, kQkvBulkRows);
        const int32_t bulk_tasks =
            p.route.world_size * m_chunks * head_chunks;
        const int32_t task_stride = comm_ctas * kQkvBulkSlots;
        for (int32_t work = slot * comm_ctas + comm_id;
             work < bulk_tasks;
             work += task_stride) {
          const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
          const int32_t destination_rank = work % p.route.world_size;
          const int32_t rank_work = work / p.route.world_size;
          const int32_t m_chunk = along_n
              ? rank_work / head_chunks
              : rank_work % m_chunks;
          const int32_t head_chunk = along_n
              ? rank_work % head_chunks
              : rank_work / m_chunks;
          const int32_t head_slot = head_chunk / chunks_per_head;
          const int32_t chunk = head_chunk - head_slot * chunks_per_head;
          const int32_t segment = head_slot < q_local_heads
              ? 0
              : (head_slot < q_local_heads + kv_local_heads ? 1 : 2);
          const int32_t local_head = segment == 0
              ? head_slot
              : head_slot - q_local_heads -
                  (segment == 2 ? kv_local_heads : 0);
          const int32_t head_offset = chunk * CopyBlockN;
          const int32_t copy_n =
              min(CopyBlockN, p.route.head_dim - head_offset);
          const int32_t physical_feature = source_feature(
              p.route,
              destination_rank,
              segment,
              local_head,
              head_offset);
          const int32_t m_begin = m_chunk * kQkvBulkRows;
          const int32_t copy_m =
              min(kQkvBulkRows, p.gemm.m - m_begin);

          if (wait_for_gemm) {
            const int32_t producer_m = m_begin / BlockM;
            const int32_t first_producer_n = physical_feature / BlockN;
            const int32_t last_producer_n =
                (physical_feature + copy_n - 1) / BlockN;
            for (int32_t producer_n = first_producer_n;
                 producer_n <= last_producer_n;
                 ++producer_n) {
              detail::wait_acquire_system(
                  p.ready +
                      (producer_m * output_n_tiles + producer_n) *
                          kReadyFlagStride,
                  p.epoch,
                  lane);
            }
          }
          __syncwarp();

          if (lane == 0) {
            cute::set_barrier_transaction_bytes(
                *barrier, kQkvBulkStageBytes);
            cute::SM90_TMA_LOAD_2D::copy(
                &args.local_output_tma,
                barrier,
                0x12f0000000000000ull,
                stage,
                physical_feature,
                m_begin);
            cute::wait_barrier(*barrier, phase);
            phase ^= 1;
            cute::tma_store_fence();

            auto* destination = p.peer_output[destination_rank];
            const int32_t segment_width =
                segment == 0 ? q_local_width : kv_local_width;
            const int32_t local_segment_base =
                segment == 0 ? 0 : q_local_width +
                    (segment == 2 ? kv_local_width : 0);
            const int64_t segment_offset = segment == 0
                ? 0
                : segment_rows * q_local_width +
                    (segment == 2 ? segment_rows * kv_local_width : 0);
            const int32_t local_feature =
                local_segment_base + local_head * p.route.head_dim +
                head_offset;
            if (args.use_tma_store) {
              const int64_t destination_row_value =
                  destination_row(p, m_begin);
              const int32_t destination_feature = p.route.defer_v_a2a
                  ? local_feature
                  : local_feature - local_segment_base;
              const int32_t descriptor =
                  p.route.defer_v_a2a ? 0 : segment;
              cute::SM90_TMA_STORE_2D::copy(
                  &args.peer_output_tma[destination_rank][descriptor],
                  stage,
                  destination_feature,
                  destination_row_value);
              cute::tma_store_arrive();
            } else {
              for (int32_t row = 0; row < copy_m; ++row) {
                const int32_t source_row = m_begin + row;
                const int64_t destination_row_value =
                    destination_row(p, source_row);
                const int64_t dst = p.route.defer_v_a2a
                    ? destination_row_value *
                          (q_local_width + kv_local_width) +
                        local_feature
                    : segment_offset +
                        destination_row_value * segment_width +
                        local_feature - local_segment_base;
                cute::SM90_BULK_COPY_S2G::copy(
                    stage + static_cast<int64_t>(row) * kQkvBulkColumns,
                    destination + dst,
                    kQkvBulkColumns * sizeof(Element));
                if ((row & 7) == 7) {
                  cute::tma_store_arrive();
                }
              }
              if ((copy_m & 7) != 0) {
                cute::tma_store_arrive();
              }
            }
            cute::tma_store_wait<0>();
          }
          __syncwarp();
        }
        if (lane == 0) {
          // Per-task `.read` waits above make each stage reusable.  Before
          // this CTA publishes completion, also wait for every destination
          // global write issued by this lane to finish.
          detail::tma_store_wait_all();
          cutlass::arch::ClusterBarrier::invalidate(barrier);
        }
      }
      __syncthreads();
      if (wait_for_gemm && p.route.defer_v_a2a && p.completion_epoch &&
          comm_id == 0) {
        wait_and_publish_completion(args);
      }
      return;
    }

    for (int32_t work = comm_id; work < tasks; work += comm_ctas) {
      const bool along_n = p.gemm.raster == GemmRaster::kAlongN;
      const int32_t destination_rank = work % p.route.world_size;
      const int32_t rank_work = work / p.route.world_size;
      const int32_t m_group = along_n
          ? rank_work / head_chunks
          : rank_work % m_groups;
      const int32_t head_chunk = along_n
          ? rank_work % head_chunks
          : rank_work / m_groups;
      const int32_t head_slot = head_chunk / chunks_per_head;
      const int32_t chunk = head_chunk - head_slot * chunks_per_head;
      const int32_t segment = head_slot < q_local_heads
          ? 0
          : (head_slot < q_local_heads + kv_local_heads ? 1 : 2);
      const int32_t local_head = segment == 0
          ? head_slot
          : head_slot - q_local_heads -
              (segment == 2 ? kv_local_heads : 0);
      const int32_t head_offset = chunk * CopyBlockN;
      const int32_t copy_n =
          min(CopyBlockN, p.route.head_dim - head_offset);
      const int32_t physical_feature = source_feature(
          p.route, destination_rank, segment, local_head, head_offset);
      if (wait_for_gemm) {
        const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
        const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
        const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
        const int32_t first_producer_n = physical_feature / BlockN;
        const int32_t last_producer_n =
            (physical_feature + copy_n - 1) / BlockN;
        const int32_t producer_n_tiles =
            last_producer_n - first_producer_n + 1;
        const int32_t waits = MTilesPerTask * producer_n_tiles;
        for (int32_t item = warp; item < waits; item += warps) {
          const int32_t local_m_tile = item / producer_n_tiles;
          const int32_t tile_m =
              m_group * MTilesPerTask + local_m_tile;
          if (tile_m < m_tiles) {
            const int32_t producer_n =
                first_producer_n + item % producer_n_tiles;
            const int32_t signal =
                tile_m * output_n_tiles + producer_n;
            detail::wait_acquire_system(
                p.ready + signal * kReadyFlagStride,
                p.epoch,
                lane);
          }
        }
      }
      __syncthreads();

      const int32_t m_begin = m_group * MTilesPerTask * BlockM;
      const int32_t copy_m =
          min(BlockM * MTilesPerTask, p.gemm.m - m_begin);
      const int32_t vectors_per_row = copy_n / kAlignment;
      const int32_t vector_count = copy_m * vectors_per_row;
      const auto* source = reinterpret_cast<const uint4*>(
          p.local_output);
      auto* destination =
          reinterpret_cast<uint4*>(p.peer_output[destination_rank]);
      const int32_t segment_width =
          segment == 0 ? q_local_width : kv_local_width;
      const int32_t local_segment_base =
          segment == 0 ? 0 : q_local_width +
              (segment == 2 ? kv_local_width : 0);
      const int64_t segment_offset = segment == 0
          ? 0
          : segment_rows * q_local_width +
              (segment == 2 ? segment_rows * kv_local_width : 0);
      const int32_t local_feature =
          local_segment_base + local_head * p.route.head_dim + head_offset;
      for (int32_t index = threadIdx.x; index < vector_count; index += blockDim.x) {
        const int32_t row = index / vectors_per_row;
        const int32_t vector_n = index - row * vectors_per_row;
        const int32_t source_row = m_begin + row;
        const int64_t src =
            static_cast<int64_t>(source_row) * source_row_vectors +
            physical_feature / kAlignment + vector_n;
        const int64_t destination_row_value =
            destination_row(p, source_row);
        const int64_t dst = p.route.defer_v_a2a
            ? (destination_row_value *
                   (q_local_width + kv_local_width) +
               local_feature) /
                    kAlignment +
                vector_n
            : (segment_offset +
               destination_row_value * segment_width +
               local_feature - local_segment_base) /
                    kAlignment +
                vector_n;
        destination[dst] = source[src];
      }
    }

    // The downstream A2A->GEMM consumes V but not the routed Q/K buffer. A
    // single communication CTA waits only for V's epilogue tiles and publishes
    // their transitive completion after the Q/K route has drained.
    if (wait_for_gemm && p.route.defer_v_a2a && p.completion_epoch &&
        comm_id == 0) {
      wait_and_publish_completion(args);
    }

  }

  CUTLASS_DEVICE static void wait_and_publish_completion(const Params& args) {
    const auto& p = args.params;
    if (!p.completion_epoch) {
      return;
    }
    const int32_t m_tiles = ceil_div(p.gemm.m, BlockM);
    const int32_t output_n_tiles = ceil_div(p.gemm.n, BlockN);
    const int32_t v_begin =
        (p.route.q_heads + p.route.kv_heads) * p.route.head_dim;
    const int32_t v_first_tile = v_begin / BlockN;
    const int32_t v_n_tiles = output_n_tiles - v_first_tile;
    const int32_t v_signals = m_tiles * v_n_tiles;
    const int32_t lane = static_cast<int32_t>(threadIdx.x) & 31;
    const int32_t warp = static_cast<int32_t>(threadIdx.x) >> 5;
    const int32_t warps = static_cast<int32_t>(blockDim.x) >> 5;
    for (int32_t item = warp; item < v_signals; item += warps) {
      const int32_t tile_m = item / v_n_tiles;
      const int32_t tile_n = v_first_tile + item - tile_m * v_n_tiles;
      const int32_t signal = tile_m * output_n_tiles + tile_n;
      detail::wait_acquire_system(
          p.ready + signal * kReadyFlagStride, p.epoch, lane);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      detail::store_release_system(p.completion_epoch, p.epoch);
    }
  }

  CUTLASS_DEVICE void operator()(
      const Params& args,
      int32_t comm_id,
      int32_t comm_ctas) {
    run(args, comm_id, comm_ctas, true);
  }

  // All local compute and communication CTAs have crossed the cooperative
  // grid barrier before this hook runs, so every remote store issued by this
  // source rank is complete. Publish one source-complete epoch into every
  // destination-owned array, then hold this rank's kernel open until all
  // sources have completed its routed output.
  CUTLASS_DEVICE void finalize(const Params& args) {
    if constexpr (Heterogeneous) {
      const auto& p = args.params;
      finalize_weighted_cp_epoch(
          p.peer_route_done_epoch,
          p.route.world_size,
          executor_rank(p),
          p.epoch);
      return;
    }
    if (blockIdx.x != 0 || threadIdx.x >= 32) {
      return;
    }
    const auto& p = args.params;
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    if (lane == 0) {
      detail::fence_system();
    }
    __syncwarp();
    if (lane < p.route.world_size) {
      detail::store_release_system(
          p.peer_route_done_epoch[lane] +
              p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    __syncwarp();
    const uint32_t* local_sources_done =
        p.peer_route_done_epoch[p.route.rank];
    if (lane < p.route.world_size) {
      while (detail::load_acquire_system(
                 local_sources_done + lane * kReadyFlagStride) <
             p.epoch) {
        __nanosleep(64);
      }
    }
    __syncwarp();
  }

#if FUSE_ENABLE_PROFILING
  // Diagnostic copy of finalize() with phase boundaries.  Production Params,
  // control flow, and generated kernel remain untouched when profiling is off.
  CUTLASS_DEVICE void finalize_profile(
      const Params& args,
      A2AGemmCtaTimeline* event) {
    if (blockIdx.x != 0 || threadIdx.x >= 32 || event == nullptr) {
      return;
    }
    const auto& p = args.params;
    const int32_t lane = static_cast<int32_t>(threadIdx.x);
    if (lane == 0) {
      detail::fence_system();
      event->fence_done = detail::read_global_timer();
    }
    __syncwarp();
    if (lane < p.route.world_size) {
      detail::store_release_system(
          p.peer_route_done_epoch[lane] +
              p.route.rank * kReadyFlagStride,
          p.epoch);
    }
    __syncwarp();
    if (lane == 0) {
      event->publish_done = detail::read_global_timer();
    }
    const uint32_t* local_sources_done =
        p.peer_route_done_epoch[p.route.rank];
    if (lane < p.route.world_size) {
      while (detail::load_acquire_system(
                 local_sources_done + lane * kReadyFlagStride) <
             p.epoch) {
        __nanosleep(64);
      }
      event->source_ready[lane] = detail::read_global_timer();
    }
    __syncwarp();
  }
#endif
};

using QkvGqaPackCommWide = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(ProjectionTileShape{})),
    static_cast<int32_t>(cute::size<1>(ProjectionTileShape{}))>;
using QkvGqaPackCommN64 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(N64TileShape{})),
    static_cast<int32_t>(cute::size<1>(N64TileShape{})),
    4,
    false,
    kQkvBulkColumns>;
using QkvGqaPackCommN160 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(N160TileShape{})),
    static_cast<int32_t>(cute::size<1>(N160TileShape{})),
    4>;
using QkvGqaPackCommN192 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(N192TileShape{})),
    static_cast<int32_t>(cute::size<1>(N192TileShape{})),
    4>;
using QkvGqaPackCommN320 = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(WideN320TileShape{})),
    static_cast<int32_t>(cute::size<1>(WideN320TileShape{}))>;
using QkvGqaPackCommSmall = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    4>;
using QkvGqaPackCommSmallInterleaved = QkvGqaPackCommT<
    GemmA2AParams,
    false,
    static_cast<int32_t>(cute::size<0>(TileShape{})),
    static_cast<int32_t>(cute::size<1>(TileShape{})),
    4,
    true>;
using Fp8QkvGqaPackComm = QkvGqaPackCommT<
    Fp8GemmA2AParams,
    true,
    static_cast<int32_t>(cute::size<0>(Fp8TileShape{})),
    static_cast<int32_t>(cute::size<1>(Fp8TileShape{}))>;

template <bool Finalize>
using WeightedQkvGqaPackCommN64 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    64,
    4,
    false,
    kQkvBulkColumns,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN128 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    128,
    4,
    false,
    128,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN128Interleaved = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    128,
    4,
    true,
    128,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN160 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    160,
    4,
    false,
    160,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN192 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    192,
    4,
    false,
    192,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN256 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    256,
    1,
    false,
    256,
    Finalize,
    true>;
template <bool Finalize>
using WeightedQkvGqaPackCommN320 = QkvGqaPackCommT<
    WeightedGemmA2AKernelParams,
    false,
    128,
    320,
    1,
    false,
    320,
    Finalize,
    true>;

using QkvGemmA2AKernelWide =
    detail::MonolithicGemm<ProjectionOutputGemm, QkvGqaPackCommWide>;
using QkvGemmA2AKernelN64 =
    detail::MonolithicGemm<N64OutputGemm, QkvGqaPackCommN64>;
using QkvGemmA2AKernelN160 =
    detail::MonolithicGemm<N160OutputGemm, QkvGqaPackCommN160>;
using QkvGemmA2AKernelN192 =
    detail::MonolithicGemm<N192OutputGemm, QkvGqaPackCommN192>;
using QkvGemmA2AKernelN320 =
    detail::MonolithicGemm<WideN320OutputGemm, QkvGqaPackCommN320>;
using QkvGemmA2AKernelSmall =
    detail::MonolithicGemm<OutputGemm, QkvGqaPackCommSmall>;
using Fp8GemmA2AKernel =
    detail::MonolithicGemm<Fp8OutputGemm, Fp8QkvGqaPackComm>;

#if FUSE_ENABLE_PROFILING
// Diagnostic-only GEMM->A2A wrapper. Unlike the generic outer telemetry
// wrapper, this records the local role completion before the cooperative grid
// barrier, which separates GEMM/route work from the cross-rank finalize tail.
template <class GemmKernel, class CommOp>
struct GemmA2ARoleTelemetryKernel
    : detail::MonolithicGemm<GemmKernel, CommOp> {
  using BaseKernel = detail::MonolithicGemm<GemmKernel, CommOp>;
  using ArchTag = typename BaseKernel::ArchTag;
  using ClusterShape = typename BaseKernel::ClusterShape;
  using SharedStorage = typename BaseKernel::SharedStorage;
  static constexpr int MaxThreadsPerBlock = BaseKernel::MaxThreadsPerBlock;
  static constexpr int MinBlocksPerMultiprocessor =
      BaseKernel::MinBlocksPerMultiprocessor;

  struct Arguments : BaseKernel::Arguments {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
  };

  struct Params : BaseKernel::Params {
    A2AGemmCtaTimeline* timeline = nullptr;
    int32_t timeline_capacity = 0;
  };

  static bool can_implement(const Arguments& args) {
    return args.timeline != nullptr && args.timeline_capacity > 0 &&
        BaseKernel::can_implement(
            static_cast<const typename BaseKernel::Arguments&>(args));
  }

  static size_t get_workspace_size(const Arguments& args) {
    return BaseKernel::get_workspace_size(
        static_cast<const typename BaseKernel::Arguments&>(args));
  }

  static cutlass::Status initialize_workspace(
      const Arguments& args,
      void* workspace,
      cudaStream_t stream) {
    return BaseKernel::initialize_workspace(
        static_cast<const typename BaseKernel::Arguments&>(args),
        workspace,
        stream);
  }

  static Params to_underlying_arguments(const Arguments& args, void* workspace) {
    Params params{};
    static_cast<typename BaseKernel::Params&>(params) =
        BaseKernel::to_underlying_arguments(
            static_cast<const typename BaseKernel::Arguments&>(args),
            workspace);
    params.timeline = args.timeline;
    params.timeline_capacity = args.timeline_capacity;
    return params;
  }

  static dim3 get_grid_shape(const Params& params) {
    return BaseKernel::get_grid_shape(
        static_cast<const typename BaseKernel::Params&>(params));
  }

  static dim3 get_block_shape() { return BaseKernel::get_block_shape(); }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    const int32_t cta = static_cast<int32_t>(blockIdx.x);
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].start = detail::read_global_timer();
    }

    const bool is_comm = cta < params.num_comm_ctas;
    if (is_comm) {
      CommOp{}(params.comm, cta, params.num_comm_ctas);
    } else {
      GemmKernel{}(params.gemm, smem);
    }

    __syncthreads();
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].role_done = detail::read_global_timer();
    }
    __syncthreads();

    if constexpr (detail::NeedsGridFinalize<CommOp>::value) {
      cooperative_groups::this_grid().sync();
      A2AGemmCtaTimeline* event =
          cta < params.timeline_capacity ? params.timeline + cta : nullptr;
      if (threadIdx.x == 0 && event != nullptr) {
        event->grid_sync_done = detail::read_global_timer();
      }
      CommOp{}.finalize_profile(params.comm, event);
    }

    __syncthreads();
    if (threadIdx.x == 0 && cta < params.timeline_capacity) {
      params.timeline[cta].end = detail::read_global_timer();
    }
  }
};
#endif

static_assert(
    sizeof(typename QkvGemmA2AKernelWide::SharedStorage) >=
        QkvGqaPackCommWide::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelN64::SharedStorage) >=
        QkvGqaPackCommN64::SharedStorageBytes);
static_assert(N64OutputGemm::MaxThreadsPerBlock == 384);
static_assert(
    sizeof(typename QkvGemmA2AKernelN160::SharedStorage) >=
        QkvGqaPackCommN160::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelN320::SharedStorage) >=
        QkvGqaPackCommN320::SharedStorageBytes);
static_assert(
    sizeof(typename QkvGemmA2AKernelSmall::SharedStorage) >=
        QkvGqaPackCommSmall::SharedStorageBytes);
static_assert(
    sizeof(typename Fp8GemmA2AKernel::SharedStorage) >=
        Fp8QkvGqaPackComm::SharedStorageBytes);

template <class Comm>
__global__ __launch_bounds__(384)
void a2a_lhs_input_copy_reference_kernel(
    CUTLASS_GRID_CONSTANT const typename Comm::Params params) {
  Comm{}(
      params, static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(gridDim.x));
}

template <class Comm>
__global__ __launch_bounds__(384) void qkv_gqa_copy_reference_kernel(
    CUTLASS_GRID_CONSTANT const typename Comm::Params params) {
  Comm{}.run(
      params,
      static_cast<int32_t>(blockIdx.x),
      static_cast<int32_t>(gridDim.x),
      false);
}

template <class Comm>
cudaError_t launch_qkv_gqa_copy_reference(
    typename Comm::Arguments& args,
    int32_t comm_ctas,
    cudaStream_t stream) {
  cudaError_t status = Comm::initialize(args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!Comm::can_implement(args)) {
    return cudaErrorNotSupported;
  }
  auto entry = qkv_gqa_copy_reference_kernel<Comm>;
  status = cudaFuncSetAttribute(
      entry,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(Comm::SharedStorageBytes));
  if (status != cudaSuccess) {
    return status;
  }
  entry<<<comm_ctas, 384, Comm::SharedStorageBytes, stream>>>(
      Comm::to_underlying_arguments(args));
  return cudaGetLastError();
}

// Shared host launch helpers. Keeping both dataflow directions in one TU
// avoids duplicate registration of the CUTLASS reference kernels they share.
cudaError_t device_sm_count(int32_t* count, int32_t* device) {
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaDeviceGetAttribute(count, cudaDevAttrMultiProcessorCount, *device);
}

template <class Kernel>
cudaError_t launch_regular(const typename Kernel::Params& params, cudaStream_t stream) {
  constexpr size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
  constexpr int cluster_x = cute::size<0>(typename Kernel::ClusterShape{});
  constexpr int cluster_y = cute::size<1>(typename Kernel::ClusterShape{});
  constexpr int cluster_z = cute::size<2>(typename Kernel::ClusterShape{});
  constexpr int cluster_size = cluster_x * cluster_y * cluster_z;
  auto entry = cutlass::device_kernel<Kernel>;
  cudaError_t status = cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 grid = Kernel::get_grid_shape(params);
  if constexpr (cluster_size == 1) {
    entry<<<grid, Kernel::get_block_shape(), smem_bytes, stream>>>(params);
    return cudaGetLastError();
  } else {
    if (grid.x % cluster_x != 0 || grid.y % cluster_y != 0 ||
        grid.z % cluster_z != 0) {
      return cudaErrorInvalidConfiguration;
    }
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {cluster_x, cluster_y, cluster_z};
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = Kernel::get_block_shape();
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    config.attrs = &attribute;
    config.numAttrs = 1;
    return cudaLaunchKernelEx(&config, entry, params);
  }
}

template <class Kernel>
cudaError_t launch_gemm_reference_impl(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas,
    RasterOptions fallback) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.lhs;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.local_output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order = raster_option(params.gemm.raster, fallback);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream);
}

template <class Kernel>
cudaError_t launch_a2a_lhs_reference_impl(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device,
    int32_t reserved_comm_ctas) {
  typename Kernel::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.input_staging;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongN);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr), stream);
}

template <
    class InputGemm,
    class Kernel,
    class Comm = A2ALhsInputComm
#if FUSE_ENABLE_PROFILING
    ,
    bool Instrumented = false
#endif
    ,
    class ParamsType = A2AGemmParams>
cudaError_t launch_a2a_lhs_gemm_policy(
    const ParamsType& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0,
    A2AGemmPeerTimeline* peer_timeline = nullptr,
    int32_t peer_timeline_capacity = 0
#endif
    ) {
  typename Comm::Arguments comm_args{};
  comm_args.params = params;
  cudaError_t status = Comm::initialize(comm_args);
  if (status != cudaSuccess) {
    return status;
  }
  if (!Comm::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }

  constexpr int32_t tile_m = static_cast<int32_t>(
      cute::size<0>(typename InputGemm::TileShape{}));
  constexpr int32_t tile_n = static_cast<int32_t>(
      cute::size<1>(typename InputGemm::TileShape{}));
  constexpr int32_t tile_k = static_cast<int32_t>(
      cute::size<2>(typename InputGemm::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  static_assert(Comm::kReadyBlockM % tile_m == 0);
  constexpr int32_t m_tiles_per_ready = Comm::kReadyBlockM / tile_m;
  const int32_t ready_m_tiles = ceil_div(params.gemm.m, Comm::kReadyBlockM);
  const int32_t compute_m_frontier =
      ceil_div(sm_count - params.num_comm_ctas, n_tiles);
  comm_args.m_window = min(
      ready_m_tiles,
      max(1, ceil_div(compute_m_frontier, m_tiles_per_ready)));
  const int32_t k_per_peer = params.gemm.k / params.route.world_size;
  if (params.gemm.k % params.route.world_size != 0 ||
      k_per_peer % tile_k != 0) {
    return cudaErrorNotSupported;
  }

  typename Kernel::Arguments args{};
  args.num_comm_ctas = params.num_comm_ctas;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.comm = comm_args;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.comm.peer_timeline = peer_timeline;
    args.comm.peer_timeline_capacity = peer_timeline_capacity;
  }
#endif
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.gemm.mainloop.ptr_A = params.input_staging;
  args.gemm.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  args.gemm.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.gemm.mainloop.ready = params.ready;
  args.gemm.mainloop.world_size = params.route.world_size;
  args.gemm.mainloop.m_tiles = m_tiles;
  args.gemm.mainloop.arrivals_per_peer =
      Comm::arrivals_per_peer(comm_args);
  args.gemm.mainloop.k_tiles_per_peer = k_per_peer / tile_k;
  args.gemm.mainloop.epoch = params.epoch;
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.gemm.mainloop.timeline = timeline;
    args.gemm.mainloop.timeline_capacity = timeline_capacity;
    args.gemm.mainloop.peer_timeline = peer_timeline;
    args.gemm.mainloop.peer_timeline_capacity = peer_timeline_capacity;
    args.gemm.mainloop.n_tiles = n_tiles;
  }
#endif
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ptr_D = params.output;
  args.gemm.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongN);
  if (!Kernel::can_implement(args)) {
    return cudaErrorNotSupported;
  }
  if (Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  return detail::launch_cooperative<Kernel>(
      Kernel::to_underlying_arguments(args, nullptr),
      stream,
      sm_count);
}

// Runtime policy model for A2A -> GEMM; the candidate kernels stay finite and
// precompiled above.
struct LhsPolicyCandidate {
  A2ALhsGemmPolicy policy;
  int32_t tile_m;
  int32_t tile_n;
  int32_t cluster_m;
};

constexpr std::array<LhsPolicyCandidate, 5> kLhsPolicyCandidates{{
    {A2ALhsGemmPolicy::kM64N128, 64, 128, 1},
    {A2ALhsGemmPolicy::kM128N128, 128, 128, 1},
    {A2ALhsGemmPolicy::kM128N160, 128, 160, 1},
    {A2ALhsGemmPolicy::kM128N256ClusterM2, 128, 256, 2},
    {A2ALhsGemmPolicy::kM128N320ClusterM2, 128, 320, 2},
}};

constexpr LhsPolicyCandidate kWideN320ManualCandidate{
    A2ALhsGemmPolicy::kM128N320ClusterM2, 128, 320, 2};

A2ALhsPolicyInfo score_a2a_lhs_policy(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    const LhsPolicyCandidate& candidate) {
  A2ALhsPolicyInfo info{};
  info.policy = candidate.policy;
  info.tile_m = candidate.tile_m;
  info.tile_n = candidate.tile_n;
  info.tile_k = 64;
  info.cluster_m = candidate.cluster_m;
  info.compute_ctas = sm_count - num_comm_ctas;
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 || problem.l <= 0 ||
      info.compute_ctas <= 0 ||
      (candidate.cluster_m > 1 &&
       (num_comm_ctas % candidate.cluster_m != 0 ||
        info.compute_ctas % candidate.cluster_m != 0))) {
    info.estimated_cycles = std::numeric_limits<double>::infinity();
    return info;
  }
  info.compute_clusters = info.compute_ctas / candidate.cluster_m;

  const int64_t m_tiles = ceil_div(problem.m, candidate.tile_m);
  const int64_t n_tiles = ceil_div(problem.n, candidate.tile_n);
  const int64_t m_cluster_tiles = ceil_div(m_tiles, candidate.cluster_m);
  info.n_tiles = static_cast<int32_t>(n_tiles);
  info.tile_count = m_tiles * n_tiles * problem.l;
  info.cluster_tile_count = m_cluster_tiles * n_tiles * problem.l;
  info.waves = static_cast<int32_t>(
      ceil_div(
          info.cluster_tile_count,
          static_cast<int64_t>(info.compute_clusters)));
  info.last_wave_clusters = static_cast<int32_t>(
      info.cluster_tile_count - static_cast<int64_t>(info.waves - 1) *
          info.compute_clusters);
  info.last_wave_ctas = info.last_wave_clusters * candidate.cluster_m;
  // A cluster is the indivisible scheduler work unit.  A wave boundary is
  // frontier-aligned only when it contains an integer number of complete
  // N-frontiers, or one frontier occupies an integer number of whole waves.
  // This prevents an already-published M frontier from being split across
  // two persistent-CTA waves merely because its N fan-out does not divide the
  // resident cluster budget.
  info.frontier_aligned =
      info.waves == 1 || info.compute_clusters % n_tiles == 0 ||
      n_tiles % info.compute_clusters == 0;
  info.full_last_wave =
      info.cluster_tile_count % info.compute_clusters == 0;

  // SM90 wave model.  Tensor-core work and HBM bytes are
  // invariant across candidates; this compares the variable L1/L2 traffic
  // after accounting for cluster-B multicast and partial-wave occupancy.
  constexpr double kL2BytesPerCycle = 8.0e6 / 1.3e3;
  const double l2_bandwidth = std::min(
      64.0 * info.compute_ctas, kL2BytesPerCycle);
  const double l1_bandwidth = 128.0 * info.compute_ctas;
  constexpr double element_bytes = sizeof(Element);
  const double l2_bytes_per_tile =
      static_cast<double>(problem.k) *
          (candidate.tile_m +
           static_cast<double>(candidate.tile_n) / candidate.cluster_m) *
          element_bytes +
      static_cast<double>(candidate.tile_m) * candidate.tile_n *
          element_bytes;
  const double l1_bytes_per_tile =
      static_cast<double>(problem.k) *
          (candidate.tile_m + candidate.tile_n) * element_bytes +
      static_cast<double>(problem.k) *
          (std::max(64, candidate.tile_m) + candidate.tile_n) *
          element_bytes +
      2.0 * candidate.tile_m * candidate.tile_n * element_bytes;
  const double wave_efficiency =
      static_cast<double>(info.cluster_tile_count) /
      (static_cast<double>(info.waves) * info.compute_clusters);
  const double l2_cycles =
      l2_bytes_per_tile * info.tile_count / l2_bandwidth;
  const double l1_cycles =
      l1_bytes_per_tile * info.tile_count / l1_bandwidth;
  info.estimated_cycles =
      std::max(l1_cycles, l2_cycles) / wave_efficiency;
  return info;
}

A2ALhsPolicyInfo select_a2a_lhs_policy_impl(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    A2ALhsGemmPolicy requested) {
  A2ALhsPolicyInfo best{};
  best.estimated_cycles = std::numeric_limits<double>::infinity();
  if (requested == A2ALhsGemmPolicy::kM128N320ClusterM2) {
    return score_a2a_lhs_policy(
        problem, num_comm_ctas, sm_count, kWideN320ManualCandidate);
  }
  for (const auto& candidate : kLhsPolicyCandidates) {
    if (requested != A2ALhsGemmPolicy::kAuto &&
        requested != candidate.policy) {
      continue;
    }
    const auto current = score_a2a_lhs_policy(
        problem, num_comm_ctas, sm_count, candidate);
    if (requested != A2ALhsGemmPolicy::kAuto) {
      return current;
    }
    // A two-CTA multicast cluster makes a one-wave input consumer advance at
    // the slower ready tile in each pair. Independent CTAs are stronger when
    // the whole GEMM already fits in one wave.
    if (candidate.cluster_m > 1 && current.waves == 1) {
      continue;
    }
    // N320 is admitted automatically only when it fixes a scheduling
    // geometry problem: aligning an M frontier or improving a partial wave.
    // If an unaligned candidate already ends on a full wave, keep the mature
    // N256 family instead of treating larger N as a generic GEMM tuning knob.
    if (candidate.policy == A2ALhsGemmPolicy::kM128N320ClusterM2 &&
        !current.frontier_aligned && current.full_last_wave) {
      continue;
    }
    const bool better_frontier =
        current.frontier_aligned > best.frontier_aligned;
    const bool equal_frontier =
        current.frontier_aligned == best.frontier_aligned;
    // Frontier splitting directly delays ready-data consumption and is a
    // structural scheduling hazard.  A partial final wave is softer: its cost
    // is already represented by wave_efficiency, so never choose an otherwise
    // weak GEMM tile solely to make the last wave full.
    if (better_frontier ||
        (equal_frontier &&
         current.estimated_cycles < best.estimated_cycles)) {
      best = current;
    }
  }
  return best;
}

cudaError_t launch_a2a_lhs_gemm_impl(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  switch (selected.policy) {
    case A2ALhsGemmPolicy::kM64N128:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsM64Gemm, A2ALhsM64GemmKernel, A2ALhsM64InputComm>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N128:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsInputGemm, A2ALhsGemmKernel, A2ALhsInputComm>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N160:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsN160Gemm, A2ALhsN160GemmKernel>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsProjectionGemm, A2ALhsProjectionGemmKernel>(
              params, stream, sm_count, device);
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      return launch_a2a_lhs_gemm_policy<
          A2ALhsWideN320Gemm, A2ALhsWideN320GemmKernel>(
              params, stream, sm_count, device);
    default:
      return cudaErrorNotSupported;
  }
}

template <
    class GemmKernel,
    class Comm,
    class Params
#if FUSE_ENABLE_PROFILING
    , bool Instrumented = false
#endif
    >
cudaError_t launch_gemm_a2a_impl(
    const Params& params,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device
#if FUSE_ENABLE_PROFILING
    , A2AGemmCtaTimeline* timeline = nullptr,
    int32_t timeline_capacity = 0
#endif
    ) {
#if FUSE_ENABLE_PROFILING
  using Kernel = std::conditional_t<
      Instrumented,
      GemmA2ARoleTelemetryKernel<GemmKernel, Comm>,
      detail::MonolithicGemm<GemmKernel, Comm>>;
#else
  using Kernel = detail::MonolithicGemm<GemmKernel, Comm>;
#endif
  constexpr int32_t tile_m =
      static_cast<int32_t>(cute::size<0>(typename GemmKernel::TileShape{}));
  constexpr int32_t tile_n =
      static_cast<int32_t>(cute::size<1>(typename GemmKernel::TileShape{}));
  const int32_t m_tiles = ceil_div(params.gemm.m, tile_m);
  const int32_t n_tiles = ceil_div(params.gemm.n, tile_n);
  typename Kernel::Arguments args{};
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) {
    args.timeline = timeline;
    args.timeline_capacity = timeline_capacity;
  }
#endif
  args.num_comm_ctas = params.num_comm_ctas;
  args.comm.params = params;
  cudaError_t status = Comm::initialize(args.comm);
  if (status != cudaSuccess) {
    return status;
  }
  args.gemm.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.gemm.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.gemm.mainloop.ptr_A = params.lhs;
  args.gemm.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.gemm.mainloop.ptr_B = params.rhs_nt;
  args.gemm.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.gemm.epilogue.thread.alpha = params.alpha;
  args.gemm.epilogue.thread.beta = 0.0f;
  args.gemm.epilogue.ptr_C = nullptr;
  args.gemm.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ptr_D = params.local_output;
  args.gemm.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.gemm.epilogue.ready = params.ready;
  args.gemm.epilogue.m_tiles = m_tiles;
  args.gemm.epilogue.n_tiles = n_tiles;
  args.gemm.epilogue.epoch = params.epoch;
  args.gemm.hw_info.device_id = device;
  args.gemm.hw_info.sm_count = sm_count - params.num_comm_ctas;
  args.gemm.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.gemm.scheduler.block_offset = params.num_comm_ctas;
  args.gemm.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  if (!Kernel::can_implement(args) || Kernel::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Kernel::initialize_workspace(args, nullptr, stream) != cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  const auto kernel_params = Kernel::to_underlying_arguments(args, nullptr);
  return detail::launch_cooperative<Kernel>(kernel_params, stream, sm_count);
}

template <class InputGemm, int32_t ReadyBlockM>
cudaError_t launch_weighted_a2a_policy(
    const WeightedA2AGemmKernelParams& segment,
    cudaStream_t stream,
    int32_t sm_count,
    int32_t device) {
  using Comm = WeightedA2ALhsInputComm<ReadyBlockM, true>;
  using Kernel = detail::MonolithicGemm<InputGemm, Comm>;
#if FUSE_ENABLE_PROFILING
  return launch_a2a_lhs_gemm_policy<
      InputGemm,
      Kernel,
      Comm,
      false,
      WeightedA2AGemmKernelParams>(
          segment, stream, sm_count, device);
#else
  return launch_a2a_lhs_gemm_policy<
      InputGemm,
      Kernel,
      Comm,
      WeightedA2AGemmKernelParams>(
          segment, stream, sm_count, device);
#endif
}

}  // namespace

// Public operator API.

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
  switch (select_qkv_gemm_policy(
      problem,
      num_comm_ctas,
      sm_count,
      route.qkv_peer_interleaved)) {
    case QkvGemmPolicy::kM128N64:
      return {
          128,
          64,
          64,
          static_cast<int32_t>(N64OutputGemm::get_block_shape().x),
          static_cast<int32_t>(
              sizeof(typename QkvGemmA2AKernelN64::SharedStorage))};
    case QkvGemmPolicy::kM128N160:
      return {
          128,
          160,
          64,
          static_cast<int32_t>(N160OutputGemm::get_block_shape().x),
          static_cast<int32_t>(
              sizeof(typename QkvGemmA2AKernelN160::SharedStorage))};
    case QkvGemmPolicy::kM128N192:
      return {
          128,
          192,
          64,
          static_cast<int32_t>(N192OutputGemm::get_block_shape().x),
          static_cast<int32_t>(
              sizeof(typename QkvGemmA2AKernelN192::SharedStorage))};
    case QkvGemmPolicy::kM128N256ClusterM2:
      return projection_cutlass_kernel_traits();
    case QkvGemmPolicy::kM128N320ClusterM2:
      return {
          128,
          320,
          64,
          static_cast<int32_t>(WideN320OutputGemm::get_block_shape().x),
          static_cast<int32_t>(
              sizeof(typename QkvGemmA2AKernelN320::SharedStorage))};
    case QkvGemmPolicy::kM128N128:
      break;
  }
  return cutlass_kernel_traits();
}

KernelTraits fp8_cutlass_kernel_traits() {
  return {
      kBlockM,
      kBlockN,
      128,
      static_cast<int32_t>(Fp8OutputGemm::get_block_shape().x),
      static_cast<int32_t>(sizeof(typename Fp8GemmA2AKernel::SharedStorage))};
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
  std::array<uint64_t, 41> words{};

  bool operator==(const QkvCommRecommendationKey& other) const {
    return words == other.words;
  }
};

struct QkvCommRecommendationKeyHash {
  size_t operator()(const QkvCommRecommendationKey& key) const {
    size_t result = 0xcbf29ce484222325ull;
    for (uint64_t word : key.words) {
      result ^= std::hash<uint64_t>{}(word) + 0x9e3779b97f4a7c15ull +
          (result << 6) + (result >> 2);
    }
    return result;
  }
};

QkvCommRecommendationKey make_qkv_comm_recommendation_key(
    const GemmProblem& p,
    const UlyssesRoute& r,
    int32_t device) {
  const auto comm_policy = normalized_qkv_comm_policy_request();
  const auto nvlink_override = normalized_qkv_nvlink_override();
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
  }};
}

struct QkvCommRecommendationCache {
  std::mutex mutex;
  std::unordered_map<
      QkvCommRecommendationKey,
      int32_t,
      QkvCommRecommendationKeyHash> entries;
};

struct QkvLastCommRecommendation {
  bool valid = false;
  QkvCommRecommendationKey key{};
  int32_t comm_ctas = 0;
};

QkvCommRecommendationCache& qkv_comm_recommendation_cache() {
  static QkvCommRecommendationCache cache;
  return cache;
}

QkvLastCommRecommendation& qkv_last_comm_recommendation() {
  thread_local QkvLastCommRecommendation last;
  return last;
}

bool find_qkv_comm_recommendation(
    const QkvCommRecommendationKey& key,
    int32_t* comm_ctas) {
  auto& last = qkv_last_comm_recommendation();
  if (last.valid && last.key == key) {
    *comm_ctas = last.comm_ctas;
    return true;
  }
  auto& cache = qkv_comm_recommendation_cache();
  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto found = cache.entries.find(key);
  if (found == cache.entries.end()) {
    return false;
  }
  last = {true, key, found->second};
  *comm_ctas = found->second;
  return true;
}

void store_qkv_comm_recommendation(
    const QkvCommRecommendationKey& key,
    int32_t comm_ctas) {
  if (comm_ctas <= 0) {
    return;
  }
  auto& cache = qkv_comm_recommendation_cache();
  {
    std::lock_guard<std::mutex> lock(cache.mutex);
    const auto found = cache.entries.find(key);
    if (found == cache.entries.end()) {
      constexpr size_t kMaxEntries = 256;
      if (cache.entries.size() >= kMaxEntries) {
        cache.entries.erase(cache.entries.begin());
      }
      cache.entries.emplace(key, comm_ctas);
    } else {
      comm_ctas = found->second;
    }
  }
  qkv_last_comm_recommendation() = {true, key, comm_ctas};
}

}  // namespace

int32_t recommended_gemm_a2a_comm_ctas(
  const GemmProblem& problem,
  const UlyssesRoute& route) {
  if (route.kind != RouteKind::kQkvGqaPack) {
    return 0;
  }
  const int64_t output_bytes =
      static_cast<int64_t>(problem.m) * problem.n * sizeof(Bf16);
  if (problem.input_dtype == DType::kFloat8E4M3 &&
      output_bytes >= 64ll * 1024 * 1024) {
    return 40;
  }
  const int32_t legacy_comm_ctas =
      output_bytes >= 32ll * 1024 * 1024
      ? 32
      : (problem.n >= 4096 ? 24 : 16);

  const bool pipeline_policy = qkv_pipeline_policy_enabled();
  if (!pipeline_policy ||
      problem.input_dtype != DType::kBfloat16 || route.world_size <= 1) {
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
  if (selected.policy == A2ALhsGemmPolicy::kM64N128) {
    return launch_a2a_lhs_gemm_policy<
        A2ALhsM64TelemetryGemm,
        A2ALhsM64TelemetryKernel,
        A2ALhsM64TelemetryInputComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  }
  if (selected.policy == A2ALhsGemmPolicy::kM128N256ClusterM2) {
    return launch_a2a_lhs_gemm_policy<
        A2ALhsProjectionTelemetryGemm,
        A2ALhsProjectionTelemetryKernel,
        A2ALhsTelemetryInputComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  }
  if (selected.policy == A2ALhsGemmPolicy::kM128N320ClusterM2) {
    return launch_a2a_lhs_gemm_policy<
        A2ALhsWideN320TelemetryGemm,
        A2ALhsWideN320TelemetryKernel,
        A2ALhsTelemetryInputComm,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity,
            peer_timeline,
            peer_timeline_capacity);
  }
  if (selected.policy != A2ALhsGemmPolicy::kM128N128) {
    return cudaErrorNotSupported;
  }
  return launch_a2a_lhs_gemm_policy<
      A2ALhsTelemetryGemm,
      A2ALhsTelemetryKernel,
      A2ALhsTelemetryInputComm,
      true>(
          launch_params,
          stream,
          sm_count,
          device,
          timeline,
          timeline_capacity,
          peer_timeline,
          peer_timeline_capacity);
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

struct QkvLaunchPlanKeyHash {
  size_t operator()(const QkvLaunchPlanKey& key) const {
    size_t result = 0xcbf29ce484222325ull;
    for (uint64_t word : key.words) {
      result ^= std::hash<uint64_t>{}(word) + 0x9e3779b97f4a7c15ull +
          (result << 6) + (result >> 2);
    }
    return result;
  }
};

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

struct QkvLaunchPlanCache {
  std::mutex mutex;
  std::unordered_map<
      QkvLaunchPlanKey,
      QkvLaunchPlan,
      QkvLaunchPlanKeyHash> entries;
};

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

  struct LastPlan {
    bool valid = false;
    QkvLaunchPlanKey key{};
    QkvLaunchPlan plan{};
  };
  thread_local LastPlan last;
  if (last.valid && last.key == key) {
    *result = last.plan;
    return cudaSuccess;
  }

  static QkvLaunchPlanCache cache;
  std::lock_guard<std::mutex> lock(cache.mutex);
  const auto cached = cache.entries.find(key);
  if (cached != cache.entries.end()) {
    last = {true, key, cached->second};
    *result = cached->second;
    return cudaSuccess;
  }

  // Cold path: resolve the device, communication split and GEMM kernel as one
  // immutable plan.  All subsequent eager launches for this key bypass both
  // cudaGetDeviceProperties and the comm/tile candidate search.
  QkvLaunchPlan plan{};
  plan.device = device;
  status = cudaDeviceGetAttribute(
      &plan.sm_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }
  plan.num_comm_ctas = params.num_comm_ctas == 0
      ? recommended_gemm_a2a_comm_ctas(params.gemm, params.route)
      : params.num_comm_ctas;
  if (plan.num_comm_ctas <= 0 || plan.num_comm_ctas >= plan.sm_count) {
    return cudaErrorInvalidValue;
  }
  plan.policy = select_qkv_gemm_policy(
      params.gemm,
      plan.num_comm_ctas,
      plan.sm_count,
      params.route.qkv_peer_interleaved);

  constexpr size_t kMaxEntries = 256;
  if (cache.entries.size() >= kMaxEntries) {
    cache.entries.erase(cache.entries.begin());
  }
  cache.entries.emplace(key, plan);
  last = {true, key, plan};
  *result = plan;
  return cudaSuccess;
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
    switch (plan.policy) {
      case QkvGemmPolicy::kM128N64:
        return launch_gemm_a2a_impl<N64OutputGemm, QkvGqaPackCommN64>(
            launch_params, stream, plan.sm_count, plan.device);
      case QkvGemmPolicy::kM128N160:
        return launch_gemm_a2a_impl<N160OutputGemm, QkvGqaPackCommN160>(
            launch_params, stream, plan.sm_count, plan.device);
      case QkvGemmPolicy::kM128N192:
        return launch_gemm_a2a_impl<N192OutputGemm, QkvGqaPackCommN192>(
            launch_params, stream, plan.sm_count, plan.device);
      case QkvGemmPolicy::kM128N256ClusterM2:
        return launch_gemm_a2a_impl<ProjectionOutputGemm, QkvGqaPackCommWide>(
            launch_params, stream, plan.sm_count, plan.device);
      case QkvGemmPolicy::kM128N320ClusterM2:
        return launch_gemm_a2a_impl<WideN320OutputGemm, QkvGqaPackCommN320>(
            launch_params, stream, plan.sm_count, plan.device);
      case QkvGemmPolicy::kM128N128:
        break;
    }
    if (launch_params.route.qkv_peer_interleaved) {
      return launch_gemm_a2a_impl<
          OutputGemm, QkvGqaPackCommSmallInterleaved>(
              launch_params, stream, plan.sm_count, plan.device);
    }
    return launch_gemm_a2a_impl<OutputGemm, QkvGqaPackCommSmall>(
        launch_params, stream, plan.sm_count, plan.device);
  }
  return cudaErrorNotSupported;
}

#if FUSE_ENABLE_PROFILING
cudaError_t launch_gemm_a2a_role_telemetry(
    const GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  GemmA2AParams launch_params = params;
  if (launch_params.num_comm_ctas == 0) {
    launch_params.num_comm_ctas =
        recommended_gemm_a2a_comm_ctas(params.gemm, params.route);
  }
  if (timeline == nullptr || timeline_capacity < sm_count ||
      launch_params.num_comm_ctas <= 0 ||
      launch_params.num_comm_ctas >= sm_count) {
    return cudaErrorInvalidValue;
  }
  if (launch_params.route.kind != RouteKind::kQkvGqaPack ||
      launch_params.route.direction != RouteDirection::kForward) {
    return cudaErrorNotSupported;
  }
  switch (select_qkv_gemm_policy(
      launch_params.gemm,
      launch_params.num_comm_ctas,
      sm_count,
      launch_params.route.qkv_peer_interleaved)) {
    case QkvGemmPolicy::kM128N64:
      return launch_gemm_a2a_impl<
          N64OutputGemm, QkvGqaPackCommN64, GemmA2AParams, true>(
              launch_params,
              stream,
              sm_count,
              device,
              timeline,
              timeline_capacity);
    case QkvGemmPolicy::kM128N160:
      return launch_gemm_a2a_impl<
          N160OutputGemm, QkvGqaPackCommN160, GemmA2AParams, true>(
              launch_params,
              stream,
              sm_count,
              device,
              timeline,
              timeline_capacity);
    case QkvGemmPolicy::kM128N192:
      return launch_gemm_a2a_impl<
          N192OutputGemm, QkvGqaPackCommN192, GemmA2AParams, true>(
              launch_params,
              stream,
              sm_count,
              device,
              timeline,
              timeline_capacity);
    case QkvGemmPolicy::kM128N256ClusterM2:
      return launch_gemm_a2a_impl<
          ProjectionOutputGemm,
          QkvGqaPackCommWide,
          GemmA2AParams,
          true>(
              launch_params,
              stream,
              sm_count,
              device,
              timeline,
              timeline_capacity);
    case QkvGemmPolicy::kM128N320ClusterM2:
      return launch_gemm_a2a_impl<
          WideN320OutputGemm,
          QkvGqaPackCommN320,
          GemmA2AParams,
          true>(
              launch_params,
              stream,
              sm_count,
              device,
              timeline,
              timeline_capacity);
    case QkvGemmPolicy::kM128N128:
      break;
  }
  if (launch_params.route.qkv_peer_interleaved) {
    return launch_gemm_a2a_impl<
        OutputGemm,
        QkvGqaPackCommSmallInterleaved,
        GemmA2AParams,
        true>(
            launch_params,
            stream,
            sm_count,
            device,
            timeline,
            timeline_capacity);
  }
  return launch_gemm_a2a_impl<
      OutputGemm,
      QkvGqaPackCommSmall,
      GemmA2AParams,
      true>(
          launch_params,
          stream,
          sm_count,
          device,
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
  Fp8QkvGqaPackComm::Arguments comm_args{};
  comm_args.params = launch_params;
  if (!Fp8QkvGqaPackComm::can_implement(comm_args)) {
    return cudaErrorNotSupported;
  }
  return launch_gemm_a2a_impl<Fp8OutputGemm, Fp8QkvGqaPackComm>(
      launch_params, stream, sm_count, device);
}

cudaError_t launch_batched_cutlass_reference(
    const GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }

  const int32_t policy_comm_ctas = params.num_comm_ctas == 0
      ? recommended_gemm_a2a_comm_ctas(params.gemm, params.route)
      : params.num_comm_ctas;
  switch (select_qkv_gemm_policy(
      params.gemm,
      policy_comm_ctas,
      sm_count,
      params.route.qkv_peer_interleaved)) {
    case QkvGemmPolicy::kM128N64:
      return launch_gemm_reference_impl<N64PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas,
          RasterOptions::AlongN);
    case QkvGemmPolicy::kM128N160:
      return launch_gemm_reference_impl<N160PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas,
          RasterOptions::AlongN);
    case QkvGemmPolicy::kM128N192:
      return launch_gemm_reference_impl<N192PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas,
          RasterOptions::AlongN);
    case QkvGemmPolicy::kM128N256ClusterM2:
      return launch_gemm_reference_impl<ProjectionPureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas,
          RasterOptions::AlongN);
    case QkvGemmPolicy::kM128N320ClusterM2:
      return launch_gemm_reference_impl<WideN320PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas,
          RasterOptions::AlongN);
    case QkvGemmPolicy::kM128N128:
      break;
  }
  return launch_gemm_reference_impl<PureGemm>(
      params, stream, sm_count, device, reserved_comm_ctas,
      RasterOptions::AlongN);
}


cudaError_t launch_a2a_gemm_cutlass_reference(
    const A2AGemmParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.input_staging || !params.rhs_nt || !params.output ||
      !supported_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }

  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  switch (selected.policy) {
    case A2ALhsGemmPolicy::kM64N128:
      return launch_a2a_lhs_reference_impl<M64PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N128:
      return launch_a2a_lhs_reference_impl<PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N160:
      return launch_a2a_lhs_reference_impl<N160PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N256ClusterM2:
      if (reserved_comm_ctas % 2 != 0 ||
          (sm_count - reserved_comm_ctas) % 2 != 0) {
        return cudaErrorInvalidValue;
      }
      return launch_a2a_lhs_reference_impl<ProjectionPureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    case A2ALhsGemmPolicy::kM128N320ClusterM2:
      if (reserved_comm_ctas % 2 != 0 ||
          (sm_count - reserved_comm_ctas) % 2 != 0) {
        return cudaErrorInvalidValue;
      }
      return launch_a2a_lhs_reference_impl<WideN320PureGemm>(
          params, stream, sm_count, device, reserved_comm_ctas);
    default:
      return cudaErrorNotSupported;
  }
}

cudaError_t launch_dense_fp8_cutlass_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream,
    int32_t reserved_comm_ctas) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  if (reserved_comm_ctas < 0 || reserved_comm_ctas >= sm_count ||
      !params.lhs || !params.rhs_nt || !params.local_output ||
      !supported_fp8_problem(params.gemm)) {
    return cudaErrorInvalidValue;
  }

  typename Fp8PureGemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape =
      make_shape(params.gemm.m, params.gemm.n, params.gemm.k, params.gemm.l);
  args.mainloop.ptr_A = params.lhs;
  args.mainloop.dA = make_stride(
      a_row_stride(params.gemm), _1{}, a_batch_stride(params.gemm));
  args.mainloop.ptr_B = params.rhs_nt;
  args.mainloop.dB = make_stride(
      b_row_stride(params.gemm), _1{}, b_batch_stride(params.gemm));
  args.epilogue.thread.alpha = params.alpha;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.epilogue.ptr_D = params.local_output;
  args.epilogue.dD = make_stride(
      d_row_stride(params.gemm), _1{}, d_batch_stride(params.gemm));
  args.hw_info.device_id = device;
  args.hw_info.sm_count = sm_count - reserved_comm_ctas;
  args.scheduler.max_swizzle_size = params.gemm.max_swizzle_size;
  args.scheduler.raster_order =
      raster_option(params.gemm.raster, RasterOptions::AlongM);
  if (!Fp8PureGemm::can_implement(args) || Fp8PureGemm::get_workspace_size(args) != 0) {
    return cudaErrorNotSupported;
  }
  if (Fp8PureGemm::initialize_workspace(args, nullptr, stream) !=
      cutlass::Status::kSuccess) {
    return cudaErrorInitializationError;
  }
  return launch_regular<Fp8PureGemm>(
      Fp8PureGemm::to_underlying_arguments(args, nullptr), stream);
}

template <class Comm>
cudaError_t launch_a2a_lhs_copy_reference_impl(
    const A2AGemmParams& params, cudaStream_t stream) {
    typename Comm::Arguments args{};
    args.params = params;
    cudaError_t status = Comm::initialize(args);
    if (status != cudaSuccess) {
      return status;
    }
    if (params.num_comm_ctas <= 0 ||
        !Comm::can_implement(args)) {
      return cudaErrorNotSupported;
    }
    constexpr size_t smem_bytes =
        kA2ALhsBulkSlots * kA2ALhsBulkStageBytes +
        kA2ALhsBulkSlots * sizeof(uint64_t);
    auto entry = a2a_lhs_input_copy_reference_kernel<Comm>;
    status = cudaFuncSetAttribute(
        entry,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));
    if (status != cudaSuccess) {
      return status;
    }
    entry<<<
        params.num_comm_ctas, 384, smem_bytes, stream>>>(
            Comm::to_underlying_arguments(args));
    return cudaGetLastError();
}

cudaError_t launch_a2a_gemm_copy_reference(
    const A2AGemmParams& params,
    cudaStream_t stream) {
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess) {
    return status;
  }
  const auto selected = select_a2a_lhs_policy_impl(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.lhs_policy);
  if (selected.policy == A2ALhsGemmPolicy::kM64N128) {
    return launch_a2a_lhs_copy_reference_impl<A2ALhsM64InputComm>(
        params, stream);
  }
  return launch_a2a_lhs_copy_reference_impl<A2ALhsInputComm>(
      params, stream);
}

cudaError_t launch_gemm_a2a_copy_reference(
    const GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  int32_t sm_count = 0;
  int32_t device = 0;
  cudaError_t status = device_sm_count(&sm_count, &device);
  if (status != cudaSuccess || params.num_comm_ctas >= sm_count) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  switch (select_qkv_gemm_policy(
      params.gemm,
      params.num_comm_ctas,
      sm_count,
      params.route.qkv_peer_interleaved)) {
    case QkvGemmPolicy::kM128N64: {
      QkvGqaPackCommN64::Arguments args{};
      args.params = params;
      return launch_qkv_gqa_copy_reference<QkvGqaPackCommN64>(
          args, params.num_comm_ctas, stream);
    }
    case QkvGemmPolicy::kM128N160: {
      QkvGqaPackCommN160::Arguments args{};
      args.params = params;
      return launch_qkv_gqa_copy_reference<QkvGqaPackCommN160>(
          args, params.num_comm_ctas, stream);
    }
    case QkvGemmPolicy::kM128N192: {
      QkvGqaPackCommN192::Arguments args{};
      args.params = params;
      return launch_qkv_gqa_copy_reference<QkvGqaPackCommN192>(
          args, params.num_comm_ctas, stream);
    }
    case QkvGemmPolicy::kM128N256ClusterM2: {
      QkvGqaPackCommWide::Arguments args{};
      args.params = params;
      return launch_qkv_gqa_copy_reference<QkvGqaPackCommWide>(
          args, params.num_comm_ctas, stream);
    }
    case QkvGemmPolicy::kM128N320ClusterM2: {
      QkvGqaPackCommN320::Arguments args{};
      args.params = params;
      return launch_qkv_gqa_copy_reference<QkvGqaPackCommN320>(
          args, params.num_comm_ctas, stream);
    }
    case QkvGemmPolicy::kM128N128:
      break;
  }
  if (params.route.qkv_peer_interleaved) {
    QkvGqaPackCommSmallInterleaved::Arguments args{};
    args.params = params;
    return launch_qkv_gqa_copy_reference<QkvGqaPackCommSmallInterleaved>(
        args, params.num_comm_ctas, stream);
  }
  QkvGqaPackCommSmall::Arguments args{};
  args.params = params;
  return launch_qkv_gqa_copy_reference<QkvGqaPackCommSmall>(
      args, params.num_comm_ctas, stream);
}


cudaError_t launch_gemm_a2a_fp8_copy_reference(
    const Fp8GemmA2AParams& params,
    cudaStream_t stream) {
  if (params.num_comm_ctas <= 0 ||
      params.route.kind != RouteKind::kQkvGqaPack ||
      params.route.direction != RouteDirection::kForward) {
    return cudaErrorInvalidValue;
  }
  Fp8QkvGqaPackComm::Arguments args{};
  args.params = params;
  return launch_qkv_gqa_copy_reference<Fp8QkvGqaPackComm>(
      args, params.num_comm_ctas, stream);
}

}  // namespace fuse
