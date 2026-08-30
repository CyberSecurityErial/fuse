// SPDX-License-Identifier: BSD-3-Clause
// Private implementation; assembled by csrc/operators/ulysses_sm90.cu.
//
// Module index:
//   - scalar validation and normalized private parameter types
//   - BF16/FP8 calibration data and compute-cost primitives
//   - CUTLASS mainloop, epilogue, tile, and kernel type construction

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
using Fp8N64TileShape = Shape<_128, _64, _128>;
using Fp8WideTileShape = Shape<_128, _256, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using M64ClusterShape = Shape<_1, _1, _1>;
using ProjectionClusterShape = Shape<_2, _1, _1>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutAColumn = cutlass::layout::ColumnMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutBRow = cutlass::layout::RowMajor;
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

enum class Fp8PipelinePolicy {
  kPingpong,
  kCooperative,
};

enum class Fp8TilePolicy {
  kM128N64,
  kM128N128,
  kM128N256ClusterM2,
};

bool requested_fp8_tile_override(Fp8TilePolicy* policy) {
  const char* value = std::getenv("FUSE_FP8_GEMM_TILE");
  if (value != nullptr && std::strcmp(value, "m128n64") == 0) {
    *policy = Fp8TilePolicy::kM128N64;
    return true;
  }
  if (value != nullptr && std::strcmp(value, "m128n256") == 0) {
    *policy = Fp8TilePolicy::kM128N256ClusterM2;
    return true;
  }
  if (value != nullptr && std::strcmp(value, "m128n128") == 0) {
    *policy = Fp8TilePolicy::kM128N128;
    return true;
  }
  return false;
}

Fp8TilePolicy select_fp8_qkv_tile(
    const GemmProblem& problem,
    const UlyssesRoute& route,
    int32_t num_comm_ctas,
    int32_t sm_count,
    int32_t device);

Fp8PipelinePolicy selected_fp8_pipeline(const GemmProblem&) {
  const char* value = std::getenv("FUSE_FP8_GEMM_PIPELINE");
  if (value != nullptr && std::strcmp(value, "cooperative") == 0) {
    return Fp8PipelinePolicy::kCooperative;
  }
  // Until the independent FP8 wave table is complete, preserve the measured
  // V12 development baseline.  `pingpong` and `auto` therefore mean the same
  // thing here; the automatic branch is replaced by the calibrated model.
  return Fp8PipelinePolicy::kPingpong;
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
      d_row_stride(p) % kAlignment == 0 &&
      d_batch_stride(p) % kAlignment == 0;
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

bool supported_transpose_b_problem(const GemmProblem& p) {
  const bool unit_strides =
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1);
  const bool leading_dimensions =
      a_row_stride(p) >= p.k && b_row_stride(p) >= p.n &&
      d_row_stride(p) >= p.n && a_row_stride(p) % kAlignment == 0 &&
      b_row_stride(p) % kAlignment == 0 &&
      d_row_stride(p) % kAlignment == 0;
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l == 1 && unit_strides &&
      leading_dimensions && !p.transpose_a && p.transpose_b &&
      p.input_dtype == DType::kBfloat16 &&
      p.weight_dtype == DType::kBfloat16 &&
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
      d_row_stride(p) % kFp8Alignment == 0 &&
      d_batch_stride(p) % kFp8Alignment == 0;
  const bool batches = p.l == 1 ||
      (a_batch_stride(p) >= a_row_stride(p) * p.m &&
       (b_batch_stride(p) == 0 || b_batch_stride(p) >= b_row_stride(p) * p.n) &&
       d_batch_stride(p) >= d_row_stride(p) * p.m);
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l > 0 && unit_strides &&
      leading_dimensions && batches && !p.transpose_a && !p.transpose_b &&
      p.input_dtype == DType::kFloat8E4M3 &&
      p.weight_dtype == DType::kFloat8E4M3 &&
      p.output_dtype == DType::kFloat8E4M3 &&
      (p.max_swizzle_size == 1 || p.max_swizzle_size == 2 ||
       p.max_swizzle_size == 4 || p.max_swizzle_size == 8);
}

bool supported_fp8_transpose_b_problem(const GemmProblem& p) {
  const bool unit_strides =
      (p.stride_a.column < 0 || p.stride_a.column == 1) &&
      (p.stride_b.column < 0 || p.stride_b.column == 1) &&
      (p.stride_d.column < 0 || p.stride_d.column == 1);
  const bool leading_dimensions =
      a_row_stride(p) >= p.k && b_row_stride(p) >= p.n &&
      d_row_stride(p) >= p.n &&
      a_row_stride(p) % kFp8Alignment == 0 &&
      b_row_stride(p) % kFp8Alignment == 0 &&
      d_row_stride(p) % kFp8Alignment == 0;
  return p.m > 0 && p.n > 0 && p.k > 0 && p.l == 1 && unit_strides &&
      leading_dimensions && !p.transpose_a && p.transpose_b &&
      p.input_dtype == DType::kFloat8E4M3 &&
      p.weight_dtype == DType::kFloat8E4M3 &&
      p.output_dtype == DType::kFloat8E4M3 &&
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

// Private launch form for QKV backward. The public API is expressed in model
// semantics; this normalized form carries the exact dgrad GEMM and route used
// by the cooperative kernel.
struct QkvBackwardKernelParams {
  const Bf16* local_q = nullptr;
  const Bf16* local_k = nullptr;
  const Bf16* local_v = nullptr;
  Bf16* peer_staging[kMaxWorldSize]{};
  uint32_t* peer_ready[kMaxWorldSize]{};
  uint32_t* peer_done_epoch[kMaxWorldSize]{};
  const Bf16* weight = nullptr;
  Bf16* grad_input = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas = 0;
  BackwardGemmPolicy gemm_policy = BackwardGemmPolicy::kAuto;
  uint32_t epoch = 0;
  float alpha = 1.0f;
};

struct Fp8QkvBackwardKernelParams {
  const Fp8Element* local_q = nullptr;
  const Fp8Element* local_k = nullptr;
  const Fp8Element* local_v = nullptr;
  Fp8Element* peer_staging[kMaxWorldSize]{};
  uint32_t* peer_ready[kMaxWorldSize]{};
  uint32_t* peer_done_epoch[kMaxWorldSize]{};
  const Fp8Element* weight = nullptr;
  Fp8Element* grad_input = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas = 0;
  BackwardGemmPolicy gemm_policy = BackwardGemmPolicy::kAuto;
  uint32_t epoch = 0;
  float alpha = 1.0f;
};

// Private normalized form for OProj backward. Keeping it distinct from the
// public forward GemmA2AParams lets the shared route template select its
// q-head-only semantics without adding a new template argument to every
// existing forward kernel instantiation.
struct OprojBackwardKernelParams {
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
};

struct Fp8OprojBackwardKernelParams {
  const Fp8Element* lhs = nullptr;
  const Fp8Element* rhs_nt = nullptr;
  Fp8Element* local_output = nullptr;
  Fp8Element* peer_output[kMaxWorldSize]{};
  uint32_t* peer_route_done_epoch[kMaxWorldSize]{};
  uint32_t* ready = nullptr;
  uint32_t* completion_epoch = nullptr;
  GemmShape4D gemm;
  UlyssesRoute route;
  int32_t num_comm_ctas = 0;
  uint32_t epoch = 0;
  float alpha = 1.0f;
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

struct Fp8QkvWaveCalibrationRow {
  int32_t k;
  // Index order is N64/C1, N128/C1, N256/C2.
  std::array<int32_t, 3> one_wave_elapsed_ns;
  std::array<int32_t, 3> steady_wave_increment_ns;
};

// H200 E4M3 calibration for the three production FP8 candidates.  The role
// telemetry kernels run with 120 resident compute CTAs (comm=12), the route
// role enabled, raster-N, and report the maximum compute-role interval across
// the four ranks.  Each geometry is measured at exactly one and eight physical
// worker waves; the latter gives (T8 - T1) / 7.  This deliberately measures the
// signaling GEMM while route traffic is resident instead of borrowing a pure
// GEMM table that cannot see ready-publication interference.
//
// Geometry         one-wave shape        eight-wave shape
// N64/C1           M128 x N7680          M1024 x N7680
// N128/C1          M128 x N15360         M1024 x N15360
// N256/ClusterM2   M256 x N15360         M2048 x N15360
//
// The model is T(w) = T(one wave) + (w - 1) * steady increment.  The first
// term is not a reusable "wave cost": it contains the one-time prefetch and
// final drain around one tile per active worker. Regenerate whenever the FP8
// pipeline, signaling path, route kernel, or toolchain changes.
constexpr std::array<Fp8QkvWaveCalibrationRow, 5>
    kFp8QkvWaveCalibration{{
        {2048, {5790, 8030, 13220}, {3836, 5587, 13174}},
        {3072, {7420, 14460, 19870}, {5271, 11411, 20301}},
        {4096, {9180, 19680, 25340}, {6643, 16640, 25793}},
        {5120, {10720, 24030, 30750}, {8854, 20869, 30533}},
        {16384, {40860, 72260, 97250}, {43109, 68909, 94263}},
    }};

// FP8 routes half the bytes of BF16, but each route task retains most of the
// fixed TMA/ready protocol cost. CP8 role profiles at 43 task waves (comm=4)
// measured 178.1 us, or 4.14 us/wave; use the rounded primitive cost below.
constexpr int64_t kFp8QkvRouteTaskWaveNs = 4150;
constexpr int32_t kFp8QkvFabricSaturationCommCtas = 12;
// Short one-wave sweeps have a wider latency plateau: c16 is consistently
// within 1.5% of the winner across CP4/CP8 and K=2K..16K, while using c16
// costs no compute residency when the problem grid is already truncated.
// Keep this distinct from the c12 steady-bandwidth saturation point above.
constexpr int32_t kFp8QkvLatencySaturationCommCtas = 16;

struct Fp8QkvComputeEstimate {
  Fp8TilePolicy policy = Fp8TilePolicy::kM128N128;
  int32_t tile_n = 128;
  int32_t cluster_m = 1;
  int64_t waves = 0;
  double wave_equivalents = 0.0;
  int64_t one_wave_elapsed_ns = 0;
  int64_t steady_wave_increment_ns = 0;
  double total_ns = 0.0;
  bool valid = false;
};

int64_t interpolate_fp8_qkv_wave_field(
    int32_t k,
    size_t policy_index,
    bool steady) {
  const auto value = [steady, policy_index](
                         const Fp8QkvWaveCalibrationRow& row) {
    return steady ? row.steady_wave_increment_ns[policy_index] :
        row.one_wave_elapsed_ns[policy_index];
  };
  if (k < kFp8QkvWaveCalibration.front().k ||
      k > kFp8QkvWaveCalibration.back().k) {
    return -1;
  }
  for (size_t row = 0; row < kFp8QkvWaveCalibration.size(); ++row) {
    if (k == kFp8QkvWaveCalibration[row].k) {
      return value(kFp8QkvWaveCalibration[row]);
    }
    if (k < kFp8QkvWaveCalibration[row].k) {
      const auto& lo = kFp8QkvWaveCalibration[row - 1];
      const auto& hi = kFp8QkvWaveCalibration[row];
      const int64_t numerator =
          (value(hi) - value(lo)) * static_cast<int64_t>(k - lo.k);
      return value(lo) +
          (numerator + (hi.k - lo.k) / 2) / (hi.k - lo.k);
    }
  }
  return value(kFp8QkvWaveCalibration.back());
}

Fp8QkvComputeEstimate estimate_fp8_qkv_compute(
    const GemmProblem& problem,
    int32_t num_comm_ctas,
    int32_t sm_count,
    Fp8TilePolicy policy) {
  Fp8QkvComputeEstimate estimate{};
  estimate.policy = policy;
  size_t policy_index = 1;
  if (policy == Fp8TilePolicy::kM128N64) {
    estimate.tile_n = 64;
    policy_index = 0;
  } else if (policy == Fp8TilePolicy::kM128N256ClusterM2) {
    estimate.tile_n = 256;
    estimate.cluster_m = 2;
    policy_index = 2;
  }
  const int32_t compute_ctas = sm_count - num_comm_ctas;
  if (problem.m <= 0 || problem.n <= 0 || problem.k <= 0 ||
      problem.l <= 0 || compute_ctas <= 0 || sm_count != 132 ||
      (estimate.cluster_m == 2 &&
       (num_comm_ctas % 2 != 0 || compute_ctas % 2 != 0))) {
    return estimate;
  }
  estimate.one_wave_elapsed_ns =
      interpolate_fp8_qkv_wave_field(problem.k, policy_index, false);
  estimate.steady_wave_increment_ns =
      interpolate_fp8_qkv_wave_field(problem.k, policy_index, true);
  if (estimate.one_wave_elapsed_ns <= 0 ||
      estimate.steady_wave_increment_ns <= 0) {
    return estimate;
  }
  const int64_t m_tiles = (problem.m + kBlockM - 1) / kBlockM;
  const int64_t m_work_units =
      (m_tiles + estimate.cluster_m - 1) / estimate.cluster_m;
  const int64_t n_work_units =
      (problem.n + estimate.tile_n - 1) / estimate.tile_n;
  const int64_t work_units = m_work_units * n_work_units * problem.l;
  const int64_t workers = compute_ctas / estimate.cluster_m;
  estimate.waves =
      (work_units + workers - 1) / workers;
  // A partial persistent tail still pays essentially one full worker service
  // interval: M256/N18432 profiles show that 12 active Cluster-M2 workers in
  // the second wave do not scale down in proportion to 64 resident workers.
  // Therefore use the integer physical-wave count, not fractional occupancy.
  estimate.wave_equivalents = std::max(
      1.0,
      static_cast<double>(work_units) / workers);
  // N64 publishes twice as many producer-ready flags for each 128-column
  // route chunk, and that route chunk cannot start until both adjacent N64
  // producers are ready.  Its signaling calibration directly covers one
  // through eight physical waves; beyond that range the accumulated
  // two-producer hand-off is no longer represented by the measured steady
  // increment.  Keep N64 as a short-grid policy instead of extrapolating it
  // into the long persistent regime.  The explicit m128n64 override remains
  // available for calibration and architecture bring-up.
  if (policy == Fp8TilePolicy::kM128N64 && estimate.waves > 8) {
    return estimate;
  }
  // A 256-column producer publishes one ready flag only after the paired
  // Cluster-M2 work has completed.  With less than two physical worker waves,
  // the delayed route start is not amortized; profiles from M256 through long
  // M show no critical-path gain at two waves and the first repeatable gain at
  // three waves.  This is a pipeline
  // granularity constraint, not a model/shape winner table.
  if (policy == Fp8TilePolicy::kM128N256ClusterM2 &&
      estimate.waves < 3) {
    return estimate;
  }
  estimate.total_ns = estimate.one_wave_elapsed_ns +
      static_cast<double>(estimate.waves - 1) *
          estimate.steady_wave_increment_ns;
  estimate.valid = estimate.waves > 0;
  return estimate;
}

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
#if FUSE_ENABLE_PROFILING
using A2ALhsN160TelemetryMainloop =
    detail::A2ALhsReadyMainloop<N160Mainloop, 1, false, true>;
using A2ALhsN160TelemetryGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    A2ALhsN160TelemetryMainloop,
    N160Epilogue,
    detail::MonolithicPersistentScheduler>;
#endif
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
    detail::A2ALhsReadyMainloop<BaseMainloop, 1, false, true>;
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
    detail::A2ALhsReadyMainloop<M64Mainloop, 1, false, true>;
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

// Backward dgrad consumes the original row-major forward weight. CUTLASS sees
// it as logical B[K,N], so the physical [K,N] rows are used directly and no
// online weight transpose is needed.
using BackwardProjectionMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutBRow,
        kAlignment,
        Accumulator,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCount<4>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using BackwardProjectionObservedEpilogue =
    detail::SignalingEpilogue<ProjectionEpilogue>;
using BackwardProjectionOutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BackwardProjectionMainloop,
    BackwardProjectionObservedEpilogue,
    detail::MonolithicPersistentScheduler>;

template <
    class TileShapeType,
    class ClusterShapeType,
    class EpilogueType,
    class KernelScheduleType>
using BackwardRowBMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutA,
        kAlignment,
        Element,
        LayoutBRow,
        kAlignment,
        Accumulator,
        TileShapeType,
        ClusterShapeType,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename EpilogueType::SharedStorage))>,
        KernelScheduleType>::CollectiveOp;

using BackwardN64Mainloop = BackwardRowBMainloop<
    N64TileShape,
    ClusterShape,
    N64Epilogue,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
using BackwardN128Mainloop = BackwardRowBMainloop<
    TileShape,
    ClusterShape,
    BaseEpilogue,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>;
using BackwardN160Mainloop = BackwardRowBMainloop<
    N160TileShape,
    ClusterShape,
    N160Epilogue,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
using BackwardN192Mainloop = BackwardRowBMainloop<
    N192TileShape,
    ClusterShape,
    N192Epilogue,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
using BackwardN64ClusterM2Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        N64TileShape,
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
using BackwardN64ClusterM2Mainloop = BackwardRowBMainloop<
    N64TileShape,
    ProjectionClusterShape,
    BackwardN64ClusterM2Epilogue,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
template <class Mainloop, class Epilogue>
using BackwardReadyGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    detail::A2ALhsReadyMainloop<Mainloop, 1, true>,
    Epilogue,
    detail::MonolithicPersistentScheduler>;

using BackwardN64ReadyGemm =
    BackwardReadyGemm<BackwardN64Mainloop, N64Epilogue>;
using BackwardN128ReadyGemm =
    BackwardReadyGemm<BackwardN128Mainloop, BaseEpilogue>;
using BackwardN160ReadyGemm =
    BackwardReadyGemm<BackwardN160Mainloop, N160Epilogue>;
using BackwardN192ReadyGemm =
    BackwardReadyGemm<BackwardN192Mainloop, N192Epilogue>;
using BackwardN64ClusterM2ReadyGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    detail::A2ALhsReadyMainloop<BackwardN64ClusterM2Mainloop, 1, true>,
    BackwardN64ClusterM2Epilogue,
    detail::MonolithicPersistentScheduler>;

template <class Mainloop, class Epilogue>
using BackwardSignalingGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Mainloop,
    detail::SignalingEpilogue<Epilogue>,
    detail::MonolithicPersistentScheduler>;

using BackwardN64SignalingGemm =
    BackwardSignalingGemm<BackwardN64Mainloop, N64Epilogue>;
using BackwardN128SignalingGemm =
    BackwardSignalingGemm<BackwardN128Mainloop, BaseEpilogue>;
using BackwardN160SignalingGemm =
    BackwardSignalingGemm<BackwardN160Mainloop, N160Epilogue>;
using BackwardN192SignalingGemm =
    BackwardSignalingGemm<BackwardN192Mainloop, N192Epilogue>;

// QKV dgrad consumes peer-published [M,QKV] staging. Each logical head is one
// ready group, and the source rank publishes the flag with system scope.
using BackwardA2ALhsMainloop =
    detail::A2ALhsReadyMainloop<BackwardProjectionMainloop, 1, true>;
using BackwardA2ALhsGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BackwardA2ALhsMainloop,
    ProjectionEpilogue,
    detail::MonolithicPersistentScheduler>;

// Shared QKV/OProj wgrad geometry. Logical A is the transpose view of the
// row-major output gradient, logical B is the saved row-major forward input:
// [weight_rows,M] * [M,weight_columns] -> [weight_rows,weight_columns].
using BackwardWgradEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        Element,
        LayoutD,
        kAlignment,
        Element,
        LayoutD,
        kAlignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using BackwardWgradMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Element,
        LayoutAColumn,
        kAlignment,
        Element,
        LayoutBRow,
        kAlignment,
        Accumulator,
        ProjectionTileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCount<4>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using BackwardWgradGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    BackwardWgradMainloop,
    BackwardWgradEpilogue,
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
    detail::A2ALhsReadyMainloop<ProjectionMainloop, 1, false, true>;
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
    detail::A2ALhsReadyMainloop<WideN320Mainloop, 1, false, true>;
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
    Fp8Element,
    LayoutD,
    kFp8Alignment,
    cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;

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
    cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum>::CollectiveOp;

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

// Keep both SM90 FP8 warp-specialized pipelines as measurable candidates.
// Ping-pong has higher steady-state throughput once several waves are in
// flight, while cooperative pays less fixed scheduling cost for short grids.
// The final automatic choice is calibrated from whole-wave measurements; the
// environment override exists only to reproduce that calibration.
using Fp8CooperativeEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
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
        Fp8Element,
        LayoutD,
        kFp8Alignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Fp8CooperativeMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
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
            static_cast<int>(
                sizeof(typename Fp8CooperativeEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::CollectiveOp;
using Fp8CooperativeObservedEpilogue =
    detail::SignalingEpilogue<Fp8CooperativeEpilogue>;
using Fp8CooperativeOutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8CooperativeMainloop,
    Fp8CooperativeObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8CooperativePureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8CooperativeMainloop,
    Fp8CooperativeEpilogue,
    cutlass::gemm::PersistentScheduler>;

// Narrow-N FP8 producer for grids whose N128 tiles leave most SMs idle.
// CopyBlockN remains one 128-column attention head, so route waits for two
// N64 producer tiles without introducing a packing copy.
using Fp8N64Epilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Fp8N64TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Fp8Element,
        LayoutD,
        kFp8Alignment,
        cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
using Fp8N64Mainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Fp8Element,
        LayoutA,
        kFp8Alignment,
        Fp8Element,
        LayoutB,
        kFp8Alignment,
        Accumulator,
        Fp8N64TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Fp8N64Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum>::CollectiveOp;
using Fp8N64ObservedEpilogue = detail::SignalingEpilogue<Fp8N64Epilogue>;
using Fp8N64OutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8N64Mainloop,
    Fp8N64ObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8N64PureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8N64Mainloop,
    Fp8N64Epilogue,
    cutlass::gemm::PersistentScheduler>;

// Wide-N FP8 producer. Cluster-M2 multicasts each A tile across two
// 256-column producer CTAs, matching the established wide BF16 geometry while
// retaining K128 FP8 tensor-core instructions.
using Fp8WideEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Fp8WideTileShape,
        ProjectionClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        void,
        LayoutD,
        kAlignment,
        Fp8Element,
        LayoutD,
        kFp8Alignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Fp8WideMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Fp8Element,
        LayoutA,
        kFp8Alignment,
        Fp8Element,
        LayoutB,
        kFp8Alignment,
        Accumulator,
        Fp8WideTileShape,
        ProjectionClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Fp8WideEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::CollectiveOp;
using Fp8WideObservedEpilogue =
    detail::SignalingEpilogue<Fp8WideEpilogue>;
using Fp8WideOutputGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8WideMainloop,
    Fp8WideObservedEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8WidePureGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8WideMainloop,
    Fp8WideEpilogue,
    cutlass::gemm::PersistentScheduler>;

// V12 baseline: keep the established monolithic role scheduling and epoch
// protocol while using E4M3 operands, routed values, and outputs with FP32
// tensor-core accumulation.
// The fixed M128N128K128 geometry is measured before introducing any FP8-only
// policy; it is not presented as an optimized selector.
using Fp8A2ALhsMainloop =
    detail::A2ALhsReadyMainloop<Fp8BaseMainloop>;
using Fp8A2ALhsGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8A2ALhsMainloop,
    Fp8BaseEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8CooperativeA2ALhsMainloop =
    detail::A2ALhsReadyMainloop<Fp8CooperativeMainloop>;
using Fp8CooperativeA2ALhsGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8CooperativeA2ALhsMainloop,
    Fp8CooperativeEpilogue,
    detail::MonolithicPersistentScheduler>;
// Hopper's FP8 fast-accumulate builder accepts the TN operand contract.  FP8
// training integrations therefore pass their already-quantized transpose
// copies (weight_nt for dgrad; *_t operands for wgrad) instead of inserting a
// transpose kernel inside this operator.
using Fp8BackwardBaseMainloop = Fp8BaseMainloop;
using Fp8BackwardReadyMainloop =
    detail::A2ALhsReadyMainloop<Fp8BackwardBaseMainloop, 1, true>;
using Fp8BackwardReadyGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BackwardReadyMainloop,
    Fp8BaseEpilogue,
    detail::MonolithicPersistentScheduler>;
using Fp8BackwardSignalingGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BackwardBaseMainloop,
    detail::SignalingEpilogue<Fp8BaseEpilogue>,
    detail::MonolithicPersistentScheduler>;

using Fp8BackwardWgradEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        Fp8TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        Accumulator,
        Accumulator,
        Fp8Element,
        LayoutD,
        kFp8Alignment,
        Fp8Element,
        LayoutD,
        kFp8Alignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
using Fp8BackwardWgradMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
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
            static_cast<int>(sizeof(
                typename Fp8BackwardWgradEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::CollectiveOp;
using Fp8BackwardWgradGemm = cutlass::gemm::kernel::GemmUniversal<
    Shape<int32_t, int32_t, int32_t, int32_t>,
    Fp8BackwardWgradMainloop,
    Fp8BackwardWgradEpilogue,
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

}  // namespace
}  // namespace fuse
