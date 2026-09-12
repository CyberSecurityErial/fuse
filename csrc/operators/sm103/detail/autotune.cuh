// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstring>
#include <iterator>
#include <tuple>
#include "model_calibration.cuh"
#include "performance_model.cuh"

namespace fuse::detail {

struct Mxfp8OprojTuningRequest {
  int64_t m = 0, n = 0, k = 0;
  int32_t world = 0, sm_count = 148, capability = 103;
  int32_t tile_m = 128, tile_n = 256, tile_k = 128, epilogue_n = 32, stage_policy = 0;
  int32_t raster = 1, max_swizzle_size = 1, swizzle = 1, dynamic_smem_bytes = 0;
  int32_t comm_ctas = 0;
};
enum class Mxfp8OprojTuningStatus { Success, InvalidInput, UnsupportedCalibration, ModelFailure };
struct Mxfp8OprojTuningResult {
  Mxfp8OprojTuningStatus status = Mxfp8OprojTuningStatus::UnsupportedCalibration;
  bool along_n = false;
  bool feed_feasible = false;
  int32_t swizzle = 0, comm_ctas = 0, compute_ctas = 0;
  Mxfp8OprojModelResult prediction{};
};

// GEMM is an immutable input, not a second search axis. Match independently
// measured physical services at each actual SM split; choose by C/P balance.
// Minimize max(C,P): accepting a small producer deficit can be cheaper than
// taking more SMs away from GEMM. P<=C is diagnostic, not a hard constraint.
// Standalone services approximate the balance, not concurrent contention.
// The launcher then lowers the same GEMM map with the chosen c, rebuilding A
// first-use windows/cohorts and W production stride together. There is no
// model-name dispatch, online GPU trial, fitted coefficient or winner lookup.
inline Mxfp8OprojTuningResult select_mxfp8_oproj_plan(
    const Mxfp8OprojTuningRequest& r, const Mxfp8OprojCalibrationPoint* points, size_t count) {
  using Status = Mxfp8OprojTuningStatus;
  Mxfp8OprojTuningResult best;
  auto width = [](int v) { return v == 1 || v == 2 || v == 4 || v == 8; };
  if (r.m <= 0 || r.m > INT32_MAX || r.n <= 0 || r.n > INT32_MAX ||
      r.k <= 0 || r.k > INT32_MAX || r.capability != 103 || r.sm_count != 148 ||
      (r.world != 4 && r.world != 8) || r.comm_ctas < 0 || r.comm_ctas >= r.sm_count ||
      r.tile_m != 128 || r.tile_n != 256 || r.tile_k != 128 || r.stage_policy != 0 ||
      (r.epilogue_n != 32 && r.epilogue_n != 64) || r.dynamic_smem_bytes <= 0 ||
      (r.raster != 0 && r.raster != 1) || !width(r.swizzle) || !width(r.max_swizzle_size) ||
      r.swizzle > r.max_swizzle_size || (!points && count)) {
    best.status = Status::InvalidInput; return best;
  }
  for (size_t i = 0; i < count; ++i) {
    const auto& p = points[i];
    if (p.n != r.n || p.k != r.k || p.world != r.world || p.sm_count != r.sm_count ||
        p.tile_m != r.tile_m || p.tile_n != r.tile_n || p.tile_k != r.tile_k ||
        p.epilogue_n != r.epilogue_n || p.stage_policy != r.stage_policy ||
        p.raster != r.raster || p.swizzle != r.swizzle ||
        p.dynamic_smem_bytes != r.dynamic_smem_bytes || (r.comm_ctas && p.comm_ctas != r.comm_ctas)) continue;
    const Mxfp8OprojServices services{p.reference_m,p.sm_count,p.comm_ctas,p.compute_ctas,
        p.compute_us,p.copy_us,p.producer_us};
    const auto prediction = score_mxfp8_oproj_bulk(r.m, services);
    if (!prediction.valid) continue;
    const bool feed_feasible = prediction.producer_finish_us <= prediction.compute_finish_us;
    if (best.status != Status::Success || prediction.score_us < best.prediction.score_us ||
        (prediction.score_us == best.prediction.score_us && p.comm_ctas < best.comm_ctas)) {
      best.status = Status::Success;
      best.along_n = r.raster == 1; best.swizzle = r.swizzle;
      best.comm_ctas = p.comm_ctas; best.compute_ctas = p.compute_ctas;
      best.feed_feasible = feed_feasible;
      best.prediction = prediction;
    }
  }
  return best;
}

inline Mxfp8OprojTuningResult select_mxfp8_oproj_plan(const Mxfp8OprojTuningRequest& r) {
  return select_mxfp8_oproj_plan(r,kMxfp8OprojCalibrationPoints.data(),kMxfp8OprojCalibrationPoints.size());
}

struct OProjTuningRequest {
  int64_t m = 0, n = 0, k = 0;
  int32_t world = 0, sm_count = 148, device = 0;
  int32_t policy_index = -1;  // -1: both calibrated collectives; 0/1: exact.
  int32_t raster = -1;        // -1: both; 0: AlongM; 1: AlongN.
  int32_t max_swizzle_size = 8;
  int32_t swizzle = 0;        // 0: measured widths <= maximum; positive: exact resolved width.
  int32_t comm_ctas = 0;      // 0: measured budgets; positive: exact, never silently changed.
};

enum class OProjTuningStatus { Success, InvalidInput, UnsupportedCalibration, ModelFailure };
struct OProjTuningResult {
  OProjTuningStatus status = OProjTuningStatus::UnsupportedCalibration;
  int32_t policy_index = -1;
  bool along_n = false;
  int32_t swizzle = 0, comm_ctas = 0, compute_ctas = 0;
  OProjModelResult prediction{};
  bool cache_hit = false;
};

// The caller first checks BF16/FP32/BF16, causal rows-pull, Graph calibration
// applicability and actual device SM count/capability. This host-only selector
// changes no public params and performs no CUDA queries/launches. One resolver
// must pass the selected collective, layout AND budget to fusion/reference/probe.
//
// Both axes are coupled only through measured services and the shared schedule:
//   tile/raster/swizzle + SM split -> first-use windows -> copy cohort/ready times
//                     C/R(K)     -> GEMM worker service + exposed feed waiting
// Fixed controls remain fixed; missing brackets/budgets have no shape-name or
// linear-SM fallback. Interpolate K only, at each exact measured SM budget.
inline OProjTuningResult select_oproj_plan(const OProjTuningRequest& request) {
  using Status = OProjTuningStatus;
  OProjTuningResult best;
  auto width = [](int value) { return value == 1 || value == 2 || value == 4 || value == 8; };
  if (request.m <= 0 || request.n <= 0 || request.k <= 0 ||
      request.m > INT32_MAX || request.n > INT32_MAX || request.k > INT32_MAX ||
      request.world <= 0 || request.sm_count <= 0 || request.device < 0 ||
      request.policy_index < -1 || request.policy_index > 1 || request.raster < -1 || request.raster > 1 ||
      !width(request.max_swizzle_size) || (request.swizzle && !width(request.swizzle)) ||
      request.swizzle > request.max_swizzle_size || request.comm_ctas < 0 || request.comm_ctas >= request.sm_count) {
    best.status = Status::InvalidInput;
    return best;
  }
  if (request.sm_count != 148 || (request.world != 4 && request.world != 8) ||
      request.m % 128 || request.k % (64 * request.world) || request.k < 8192 || request.k > 16384) return best;
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  const int64_t mt = request.m / 128, nt = ceil_div(request.n, 256), minimum = std::min(mt, nt);
  try {
    for (int policy : {0, 1}) for (int raster : {0, 1}) for (int swizzle : {4, 8}) {
      if ((request.policy_index >= 0 && request.policy_index != policy) ||
          (request.raster >= 0 && request.raster != raster) || swizzle > request.max_swizzle_size ||
          (request.swizzle && request.swizzle != swizzle) || minimum < (swizzle == 8 ? 6 : 3)) continue;
      const int64_t scheduled = ceil_div(mt, swizzle) * swizzle * ceil_div(nt, swizzle) * swizzle;
      for (int comm : {8, 16, 24, 32, 48}) {
        if (request.comm_ctas && request.comm_ctas != comm) continue;
        const auto compute = static_cast<int32_t>(std::min(scheduled, int64_t{148 - comm}));
        const OprojCalibrationPoint *lower = nullptr, *upper = nullptr;
        for (const auto& point : kOprojCalibrationPoints) {
          if (point.world != request.world || point.policy_index != policy || point.along_n != bool(raster) ||
              point.swizzle != swizzle || point.comm_ctas != comm || point.compute_ctas != compute) continue;
          if (point.k <= request.k && (!lower || point.k > lower->k)) lower = &point;
          if (point.k >= request.k && (!upper || point.k < upper->k)) upper = &point;
        }
        if (!lower || !upper) continue;
        const double alpha = lower == upper ? 0.0 : double(request.k - lower->k) / (upper->k - lower->k);
        OProjModelInput input;
        input.m = request.m; input.n = request.n; input.k = request.k;
        input.world = request.world; input.sm_count = request.sm_count; input.comm_ctas = comm;
        input.tile_m = 128; input.tile_n = 256; input.tile_k = 64;
        input.raster = raster ? OProjModelRaster::AlongN : OProjModelRaster::AlongM;
        input.resolved_swizzle = swizzle; input.calibrated_compute_ctas = compute;
        input.tile_cycle_us = lower->tile_cycle_us * (1 - alpha) + upper->tile_cycle_us * alpha;
        input.copy_bandwidth_gb_s = lower->copy_slot_bandwidth_gb_s * (1 - alpha) +
                                    upper->copy_slot_bandwidth_gb_s * alpha;
        const int64_t row_bytes = 2 * request.k / request.world;
        const int64_t rows = std::min(int64_t{128}, 48 * 1024 / row_bytes);
        input.copy_chunks = ceil_div(128, rows); input.copy_slots = 4 * comm;
        input.cohort_m_tiles = std::max(int64_t{1}, input.copy_slots / input.copy_chunks);
        for (int64_t start = 0; start < 128; start += rows) {
          input.copy_chunk_bytes.push_back(row_bytes * std::min(rows, 128 - start));
        }
        const auto prediction = score_oproj_schedule(input);
        if (prediction.status != OProjModelStatus::Success) {
          if (best.status != Status::Success) { best.status = Status::ModelFailure; best.prediction = prediction; }
          continue;
        }
        if (best.status != Status::Success || prediction.score_us < best.prediction.score_us) {
          best = {Status::Success, policy, bool(raster), swizzle, comm, compute, prediction, false};
        }
      }
    }
  } catch (const std::bad_alloc&) {
    best.status = Status::ModelFailure;
    best.prediction.status = OProjModelStatus::AllocationFailure;
  }
  return best;
}

// A small per-host-thread cache, scoped to this compiled calibration version.
// Include every physical/override request field, never pointers, epoch or model
// labels. Cache successes only: transient allocation failures must remain retryable.
inline OProjTuningResult select_oproj_plan_cached(const OProjTuningRequest& request) {
  struct Entry { OProjTuningRequest request; OProjTuningResult result; bool valid = false; };
  struct Cache { std::array<Entry, 16> entries{}; size_t next = 0; const char* version = kOprojCalibrationVersion; };
  static thread_local Cache cache;
  if (std::strcmp(cache.version, kOprojCalibrationVersion) != 0) cache = Cache{};
  auto key = [](const OProjTuningRequest& r) {
    return std::tie(r.m, r.n, r.k, r.world, r.sm_count, r.device, r.policy_index,
                    r.raster, r.max_swizzle_size, r.swizzle, r.comm_ctas);
  };
  for (const auto& entry : cache.entries) if (entry.valid && key(entry.request) == key(request)) {
    auto result = entry.result; result.cache_hit = true; return result;
  }
  auto result = select_oproj_plan(request);
  if (result.status == OProjTuningStatus::Success) {
    cache.entries[cache.next] = {request, result, true};
    cache.next = (cache.next + 1) % cache.entries.size();
  }
  return result;
}

struct Mxfp8QkvTuningRequest {
  int64_t m = 0, n = 0, k = 0;
  int32_t world = 0, sm_count = 148, device = 0, capability = 103;
  int32_t tile_m = 128, tile_n = 256, tile_k = 128, epilogue_n = 64;
  int32_t stages = 0, cluster_ctas = 1, raster = 0;
  int32_t max_swizzle_size = 8, swizzle = 1, comm_ctas = 0;
  int32_t dynamic_smem_bytes = 0, q_heads = 0, kv_heads = 0, head_dim = 128;
};

enum class Mxfp8QkvTuningStatus { Success, InvalidInput, UnsupportedCalibration, ModelFailure };
struct Mxfp8QkvTuningResult {
  Mxfp8QkvTuningStatus status = Mxfp8QkvTuningStatus::UnsupportedCalibration;
  bool along_n = false;
  int32_t swizzle = 0, comm_ctas = 0, compute_ctas = 0;
  Mxfp8QkvModelResult prediction{};
  bool cache_hit = false;
};

inline auto mxfp8_qkv_request_key(const Mxfp8QkvTuningRequest& r) {
  return std::tie(r.m, r.n, r.k, r.world, r.sm_count, r.device, r.capability,
      r.tile_m, r.tile_n, r.tile_k, r.epilogue_n, r.stages, r.cluster_ctas,
      r.raster, r.max_swizzle_size, r.swizzle, r.comm_ctas, r.dynamic_smem_bytes,
      r.q_heads, r.kv_heads, r.head_dim);
}

// Fixed-GEMM selector: tile, resolved stages/resources, raster and swizzle are
// INPUTS, never additional search axes. The scored model derives both queues
// from that same mapping. Changing c changes actual compute workers and the
// stride of the 8*c shared quantization/route workers, not ready granularity.
//
//   exact GEMM/route/domain + exact c -> bracketed K services -> schedule score
//                                                       -> best measured c
//
// Evaluate every available independently measured budget with constant-work
// host arithmetic. There is no runtime measurement, neighborhood probe or
// invented SM scaling law. This is finite selection, not global optimality.
// The overload is useful for host validation and never caches caller-owned data.
inline Mxfp8QkvTuningResult select_mxfp8_qkv_plan(
    const Mxfp8QkvTuningRequest& request,
    const Mxfp8QkvCalibrationPoint* calibration, size_t count) {
  using Status = Mxfp8QkvTuningStatus;
  Mxfp8QkvTuningResult best;
  auto width = [](int v) { return v == 1 || v == 2 || v == 4 || v == 8; };
  if (request.m <= 0 || request.n <= 0 || request.k <= 0 ||
      request.m > INT32_MAX || request.n > INT32_MAX || request.k > INT32_MAX ||
      request.world <= 0 || request.sm_count <= 1 || request.device < 0 ||
      request.tile_m <= 0 || request.tile_n <= 0 || request.tile_k <= 0 ||
      request.epilogue_n <= 0 || request.stages <= 0 || request.cluster_ctas <= 0 ||
      request.raster < 0 || request.raster > 1 || !width(request.max_swizzle_size) ||
      !width(request.swizzle) || request.swizzle > request.max_swizzle_size ||
      request.comm_ctas < 0 || request.comm_ctas >= request.sm_count ||
      request.dynamic_smem_bytes <= 0 || request.q_heads <= 0 || request.kv_heads <= 0 ||
      request.head_dim <= 0 || (count && !calibration)) {
    best.status = Status::InvalidInput;
    return best;
  }
  if (request.capability != 103 || request.sm_count != 148 || (request.world != 4 && request.world != 8) ||
      request.tile_m != 128 || request.tile_n != 256 || request.tile_k != 128 ||
      (request.epilogue_n != 32 && request.epilogue_n != 64) ||
      request.m % 128 || request.n % 256 || request.k % 128 ||
      request.cluster_ctas != 1 || request.head_dim != 128 ||
      request.q_heads % request.world || request.kv_heads % request.world || request.q_heads % request.kv_heads ||
      request.n != (int64_t{request.q_heads} + 2 * int64_t{request.kv_heads}) * request.head_dim) return best;
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  const int64_t mt = ceil_div(request.m, request.tile_m), nt = ceil_div(request.n, request.tile_n);
  const int64_t padded_m = ceil_div(mt, request.swizzle) * request.swizzle;
  const int64_t padded_n = ceil_div(nt, request.swizzle) * request.swizzle;
  const int64_t scheduled = padded_m * padded_n;
  auto compute_for = [&](int comm) {
    return static_cast<int32_t>(std::min(scheduled, int64_t{request.sm_count - comm}));
  };
  auto matches = [&](const Mxfp8QkvCalibrationPoint& p, int comm) {
    return p.world == request.world && p.sm_count == request.sm_count && p.capability == request.capability &&
        p.tile_m == request.tile_m && p.tile_n == request.tile_n && p.tile_k == request.tile_k &&
        p.epilogue_n == request.epilogue_n && p.stages == request.stages && p.cluster_ctas == request.cluster_ctas &&
        p.raster == request.raster && p.swizzle == request.swizzle && p.comm_ctas == comm &&
        p.compute_ctas == compute_for(comm) && p.dynamic_smem_bytes == request.dynamic_smem_bytes &&
        p.q_heads == request.q_heads && p.kv_heads == request.kv_heads && p.head_dim == request.head_dim &&
        p.k > 0 && p.k % 128 == 0 && p.m_min > 0 && p.m_max <= INT32_MAX &&
        p.m_min <= request.m && request.m <= p.m_max &&
        p.n_min > 0 && p.n_max <= INT32_MAX && p.n_min <= request.n && request.n <= p.n_max;
  };
  auto same_domain = [](const Mxfp8QkvCalibrationPoint& a, const Mxfp8QkvCalibrationPoint& b) {
    return a.m_min == b.m_min && a.m_max == b.m_max && a.n_min == b.n_min && a.n_max == b.n_max &&
        a.k_interpolation_group == b.k_interpolation_group && a.bulk.reference_m == b.bulk.reference_m;
  };
  // Return an exact anchor or the narrowest explicitly authorized K bracket.
  // All physical fields/budgets match before interpolation; overlapping exact
  // anchors are ambiguous and fail closed rather than picking favorable data.
  auto brackets = [&](int comm, const Mxfp8QkvCalibrationPoint*& lower,
                      const Mxfp8QkvCalibrationPoint*& upper) {
    lower = upper = nullptr;
    for (size_t i = 0; i < count; ++i) {
      const auto& p = calibration[i];
      if (!matches(p, comm)) continue;
      if (p.bulk.reference_m != p.m_min || p.n_min != p.n_max ||
          !valid_mxfp8_qkv_bulk_services(p.services, p.bulk)) return false;
      if (p.k != request.k) continue;
      if (lower) { lower = upper = nullptr; return false; }
      lower = upper = &p;
    }
    if (lower) return true;
    int64_t span = INT64_MAX;
    bool ambiguous = false;
    for (size_t i = 0; i < count; ++i) {
      const auto& a = calibration[i];
      if (!matches(a, comm) || a.k >= request.k || !a.k_interpolation_group) continue;
      for (size_t j = 0; j < count; ++j) {
        const auto& b = calibration[j];
        if (!matches(b, comm) || b.k <= request.k || !same_domain(a, b)) continue;
        const int64_t candidate_span = int64_t{b.k} - a.k;
        if (candidate_span < span) { lower = &a; upper = &b; span = candidate_span; ambiguous = false; }
        else if (candidate_span == span) ambiguous = true;
      }
    }
    return lower != nullptr && !ambiguous;
  };
  try {
    std::vector<int> evaluated;
    auto evaluate = [&](int comm) {
      if (comm <= 0 || comm >= request.sm_count ||
          std::find(evaluated.begin(), evaluated.end(), comm) != evaluated.end()) return;
      evaluated.push_back(comm);
      const Mxfp8QkvCalibrationPoint *lower, *upper;
      if (!brackets(comm, lower, upper)) return;
      const double alpha = lower == upper ? 0.0 : double(request.k - lower->k) / (upper->k - lower->k);
      Mxfp8QkvModelInput input;
      input.m = request.m; input.n = request.n; input.k = request.k;
      input.world = request.world; input.sm_count = request.sm_count; input.comm_ctas = comm;
      input.calibrated_compute_ctas = compute_for(comm);
      input.tile_m = request.tile_m; input.tile_n = request.tile_n; input.tile_k = request.tile_k;
      input.epilogue_n = request.epilogue_n; input.cluster_ctas = request.cluster_ctas;
      input.along_n = request.raster == 1; input.resolved_swizzle = request.swizzle;
      double Mxfp8QkvServices::* const fields[] = {
          &Mxfp8QkvServices::startup_us, &Mxfp8QkvServices::tile_first_us,
          &Mxfp8QkvServices::tile_cycle_us};
      // Member pointers keep interpolation explicit when service fields evolve.
      for (auto field : fields) {
        input.services.*field =
            lower->services.*field * (1 - alpha) + upper->services.*field * alpha;
      }
      input.bulk.reference_m = lower->bulk.reference_m;
      input.bulk.route_us = lower->bulk.route_us * (1 - alpha) + upper->bulk.route_us * alpha;
      input.bulk.quant_route_us = lower->bulk.quant_route_us * (1 - alpha) + upper->bulk.quant_route_us * alpha;
      const auto prediction = score_mxfp8_qkv_bulk(input);
      if (prediction.status != Mxfp8QkvModelStatus::Success) {
        if (best.status != Status::Success) { best.status = Status::ModelFailure; best.prediction = prediction; }
        return;
      }
      if (best.status != Status::Success || prediction.score_us < best.prediction.score_us ||
          (prediction.score_us == best.prediction.score_us && comm < best.comm_ctas)) {
        best = {Status::Success, input.along_n, request.swizzle, comm, prediction.compute_ctas, prediction, false};
      }
    };
    if (request.comm_ctas) { evaluate(request.comm_ctas); return best; }
    for (size_t i = 0; i < count; ++i) evaluate(calibration[i].comm_ctas);
  } catch (const std::bad_alloc&) {
    best.status = Status::ModelFailure;
  }
  return best;
}

inline Mxfp8QkvTuningResult select_mxfp8_qkv_plan(const Mxfp8QkvTuningRequest& request) {
  return select_mxfp8_qkv_plan(request, kMxfp8QkvCalibrationPoints.data(), kMxfp8QkvCalibrationPoints.size());
}

// Only the immutable compiled calibration is cached. Its identity and every
// physical/override input form the key. No cache exists for caller-owned tables.
inline Mxfp8QkvTuningResult select_mxfp8_qkv_plan_cached(const Mxfp8QkvTuningRequest& request) {
  struct Entry { Mxfp8QkvTuningRequest request; Mxfp8QkvTuningResult result; bool valid = false; };
  struct Cache {
    std::array<Entry, 16> entries{};
    size_t next = 0;
    const char* version = kMxfp8QkvCalibrationVersion;
  };
  static thread_local Cache cache;
  if (std::strcmp(cache.version, kMxfp8QkvCalibrationVersion) != 0) cache = Cache{};
  for (const auto& entry : cache.entries) {
    if (!entry.valid || mxfp8_qkv_request_key(entry.request) != mxfp8_qkv_request_key(request)) continue;
    auto result = entry.result; result.cache_hit = true; return result;
  }
  auto result = select_mxfp8_qkv_plan(request);
  if (result.status == Mxfp8QkvTuningStatus::Success) {
    cache.entries[cache.next] = {request, result, true};
    cache.next = (cache.next + 1) % cache.entries.size();
  }
  return result;
}

}  // namespace fuse::detail
