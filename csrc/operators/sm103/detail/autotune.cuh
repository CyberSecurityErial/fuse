// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstring>
#include <tuple>
#include "model_calibration.cuh"
#include "performance_model.cuh"

namespace fuse::detail {

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

}  // namespace fuse::detail
