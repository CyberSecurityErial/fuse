// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <vector>

#include "producer_consumer.cuh"

namespace fuse::detail {

enum class OProjModelRaster { AlongM, AlongN };
enum class OProjModelStatus {
  Success, InvalidInput, UnsupportedGeometry, WorkLimit, InvalidSchedule,
  NonfiniteScore, AllocationFailure
};

struct OProjModelInput {
  int64_t m = 0, n = 0, k = 0;
  int64_t world = 0, sm_count = 0, comm_ctas = 0;
  int64_t tile_m = 0, tile_n = 0, tile_k = 0;
  int64_t cluster_ctas = 1;
  OProjModelRaster raster = OProjModelRaster::AlongM;
  int64_t resolved_swizzle = 0;
  int64_t copy_chunks = 0, copy_slots = 0, cohort_m_tiles = 0;
  int64_t calibrated_compute_ctas = 0;
  double tile_cycle_us = 0;
  double copy_bandwidth_gb_s = 0;  // max_slot_bytes * copy_slots / R_us / 1000
  // Empty means equal-size chunks, matching the Python model's explicit
  // approximation. Otherwise describe every original chunk of one (M,peer),
  // including its real tail; their sum must cover the complete ready unit.
  std::vector<uint64_t> copy_chunk_bytes;
};

struct OProjModelResult {
  OProjModelStatus status = OProjModelStatus::InvalidInput;
  int32_t compute_ctas = 0, critical_worker = 0;  // Worker index excludes communication CTA offset.
  int64_t scheduled_work_tiles = 0, valid_work_tiles = 0, integer_waves = 0;
  double score_us = 0, ideal_overlap_us = 0, critical_path_us = 0;
  double compute_finish_us = 0, copy_finish_us = 0, first_ready_us = 0;
  double wave_compute_service_us = 0, strided_compute_service_us = 0;
  double critical_worker_service_us = 0, critical_worker_feed_wait_us = 0;
  double critical_worker_initial_wait_us = 0, critical_worker_later_feed_wait_us = 0;
  double exposed_feed_us = 0;
  double worker_wait_sum_us = 0;  // Sum across parallel workers, not global kernel stall time.
  double feed_phase_ideal_compute_us = 0, feed_phase_predicted_finish_us = 0;
  double feed_phase_demand_gb_s = 0, effective_delivery_gb_s = 0;
  double production_consumption_ratio = 0;
};

// Host-only arithmetic adapter for the existing cluster-one CUTLASS mapping.
// It supplies resolved extents/division, never a second tile or copy ordering.
struct OProjModelSchedulerParams {
  enum class RasterOrder { AlongM, AlongN };
  struct Divmod {
    uint64_t divisor;
    CUTLASS_HOST_DEVICE void operator()(uint64_t& quotient, uint64_t& remainder, uint64_t value) const {
      quotient = value / divisor;
      remainder = value % divisor;
    }
  };
  RasterOrder raster_order_;
  int32_t log_swizzle_size_;
  uint64_t blocks_per_problem_, compute_grid_size;
  Divmod divmod_batch_, divmod_cluster_blk_major_;
};

// Host-side equivalent of Python score_oproj_schedule's amortized_full_boundary
// mode. Caller supplies independent C/waves and max_slot_bytes*copy_slots/R
// for THIS compute/communication budget. The latter is the effective all-slot
// equivalent service bandwidth, not unique-A-bytes/R or measured fabric bandwidth:
// dividing it across slots reproduces the anchor R without charging its original
// tail-chunk/slot load imbalance twice. Neither model names nor B300 timing
// constants occur here. Startup/drain are already amortized in these inputs:
// no extra startup/drain is added, and their measured values remain unknown.
//
//     fixed copy slots -> last original chunk -> complete (M,peer) ready
//                                                     |
//     resolved GEMM worker: tile c, c+C, ... -> peer K segments in order
//
// Both orders come from producer_consumer.cuh. The numerical service model
// assumes constant per-slot bandwidth and equal peer shares of tile service;
// it does not simulate prefetch, dynamic contention or Tensor Core busy cycles.
// AlongM's first N band needs all A; AlongN spreads A demand across all GEMM.
// Score retains both the integer-wave bound and the explicit worker chain.
//
// Plain host function: no device annotation, CUDA launch or global cache.
// The resolver may call it before caching a new shape. Bounded temporary storage
// is at most 20 MB (1M slots, 1M ready times and 1M arrival counts); tile/copy
// coordinates are decoded on demand, not materialized in additional queues.
inline OProjModelResult score_oproj_schedule(const OProjModelInput& input) {
  using Status = OProjModelStatus;
  auto fail = [](Status status) { OProjModelResult result; result.status = status; return result; };
  constexpr int64_t kMaxWork = 1000000;
  const int64_t geometry[] = {input.m, input.n, input.k, input.world, input.sm_count,
      input.comm_ctas, input.tile_m, input.tile_n, input.tile_k, input.resolved_swizzle,
      input.copy_chunks, input.copy_slots, input.cohort_m_tiles, input.calibrated_compute_ctas, input.cluster_ctas};
  for (int64_t value : geometry) {
    if (value <= 0 || value > std::numeric_limits<int32_t>::max()) return fail(Status::InvalidInput);
  }
  if (!std::isfinite(input.tile_cycle_us) || input.tile_cycle_us <= 0 ||
      !std::isfinite(input.copy_bandwidth_gb_s) || input.copy_bandwidth_gb_s <= 0) {
    return fail(Status::InvalidInput);
  }
  const bool along_n = input.raster == OProjModelRaster::AlongN;
  if ((!along_n && input.raster != OProjModelRaster::AlongM) ||
      (input.world != 4 && input.world != 8) || input.comm_ctas >= input.sm_count ||
      input.cluster_ctas != 1 || input.tile_m != 128 || (input.tile_n != 128 && input.tile_n != 256) ||
      (input.tile_k != 64 && input.tile_k != 128) || input.m % input.tile_m ||
      input.k % (input.world * input.tile_k) ||
      (input.resolved_swizzle != 1 && input.resolved_swizzle != 2 &&
       input.resolved_swizzle != 4 && input.resolved_swizzle != 8) ||
      (input.copy_slots != input.comm_ctas && input.copy_slots != 4 * input.comm_ctas)) {
    return fail(Status::UnsupportedGeometry);
  }
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  const int64_t mt = input.m / input.tile_m, nt = ceil_div(input.n, input.tile_n);
  const int64_t pm = ceil_div(mt, input.resolved_swizzle) * input.resolved_swizzle;
  const int64_t pn = ceil_div(nt, input.resolved_swizzle) * input.resolved_swizzle;
  const int64_t scheduled = pm * pn, tasks = mt * input.world * input.copy_chunks;
  if (scheduled > kMaxWork || tasks > kMaxWork || input.copy_slots > kMaxWork) return fail(Status::WorkLimit);
  const int32_t compute = static_cast<int32_t>(std::min(scheduled, input.sm_count - input.comm_ctas));
  if (input.calibrated_compute_ctas != compute) return fail(Status::InvalidInput);
  const uint64_t ready_bytes = 2 * input.tile_m * input.k / input.world;
  if (!input.copy_chunk_bytes.empty()) {
    if (input.copy_chunk_bytes.size() != static_cast<uint64_t>(input.copy_chunks)) return fail(Status::InvalidInput);
    uint64_t total = 0;
    for (uint64_t bytes : input.copy_chunk_bytes) {
      if (bytes == 0 || bytes > ready_bytes - total) return fail(Status::InvalidInput);
      total += bytes;
    }
    if (total != ready_bytes) return fail(Status::InvalidInput);
  }
  int32_t log_swizzle = 0;
  while ((int64_t{1} << log_swizzle) < input.resolved_swizzle) ++log_swizzle;
  const OProjModelSchedulerParams params{
      along_n ? OProjModelSchedulerParams::RasterOrder::AlongN : OProjModelSchedulerParams::RasterOrder::AlongM,
      log_swizzle, static_cast<uint64_t>(scheduled), static_cast<uint64_t>(compute),
      {static_cast<uint64_t>(scheduled)}, {static_cast<uint64_t>(along_n ? pn : pm)}};
  auto order = A2AInputTileOrder::make(params, static_cast<int32_t>(mt));
  order.ready_group_m_tiles = static_cast<int32_t>(input.cohort_m_tiles);
  try {
    std::vector<double> slots(input.copy_slots, 0), ready(mt * input.world, 0);
    std::vector<int32_t> arrivals(mt * input.world, 0);
    OProjModelResult result;
    result.compute_ctas = compute;
    result.scheduled_work_tiles = scheduled;
    result.valid_work_tiles = mt * nt;
    result.integer_waves = ceil_div(scheduled, compute);
    for (int64_t index = 0; index < tasks; ++index) {
      const auto task = order.decode(index, static_cast<int32_t>(input.world), static_cast<int32_t>(input.copy_chunks));
      if (task.m < 0 || task.m >= mt || task.peer < 0 || task.peer >= input.world ||
          task.chunk < 0 || task.chunk >= input.copy_chunks) return fail(Status::InvalidSchedule);
      const int64_t cell = task.m * input.world + task.peer, slot = index % input.copy_slots;
      const double bytes = input.copy_chunk_bytes.empty() ? double(ready_bytes) / input.copy_chunks
          : static_cast<double>(input.copy_chunk_bytes[task.chunk]);
      slots[slot] += bytes / input.copy_bandwidth_gb_s * (input.copy_slots / 1000.0);
      ready[cell] = std::max(ready[cell], slots[slot]);
      ++arrivals[cell];
    }
    for (int32_t count : arrivals) if (count != input.copy_chunks) return fail(Status::InvalidSchedule);
    result.copy_finish_us = *std::max_element(slots.begin(), slots.end());
    result.first_ready_us = std::numeric_limits<double>::infinity();
    const double peer_service = input.tile_cycle_us / input.world;
    for (int32_t worker = 0; worker < compute; ++worker) {
      double now = 0, waited = 0, service = 0, phase_service = 0, phase_end = 0, initial_wait = 0;
      bool first = true;
      for (int64_t logical = worker; logical < scheduled; logical += compute) {
        const auto tile = ProducerTileOrder::decode(params, logical);
        if (!tile.valid) return fail(Status::InvalidSchedule);
        // The real static scheduler marks swizzle-padding coordinates valid.
        // CUTLASS still executes their MMA/epilogue pipeline. Only out-of-bounds
        // M skips the A-ready dependency (A2ALhsReadyMainloop::load); padding N
        // continues to consume that M row and therefore waits on its peers.
        const bool needs_ready = tile.m >= 0 && tile.m < mt;
        if (first) {
          initial_wait = needs_ready ? ready[tile.m * input.world] : 0;
          if (needs_ready) result.first_ready_us = std::min(result.first_ready_us, initial_wait);
          first = false;
        }
        for (int32_t peer = 0; peer < input.world; ++peer) {
          const double wait = needs_ready
              ? std::max(0.0, ready[tile.m * input.world + peer] - now) : 0;
          now += wait + peer_service;
          waited += wait;
        }
        service += input.tile_cycle_us;
        if (along_n || tile.n < input.resolved_swizzle) {
          phase_service += input.tile_cycle_us;
          phase_end = now;
        }
      }
      if (worker == 0 || now > result.compute_finish_us) {
        result.compute_finish_us = now;
        result.critical_worker = worker;
        result.critical_worker_service_us = service;
        result.critical_worker_feed_wait_us = waited;
        result.critical_worker_initial_wait_us = initial_wait;
      }
      result.strided_compute_service_us = std::max(result.strided_compute_service_us, service);
      result.feed_phase_ideal_compute_us = std::max(result.feed_phase_ideal_compute_us, phase_service);
      result.feed_phase_predicted_finish_us = std::max(result.feed_phase_predicted_finish_us, phase_end);
      result.worker_wait_sum_us += waited;
    }
    result.wave_compute_service_us = result.integer_waves * input.tile_cycle_us;
    result.ideal_overlap_us = std::max(result.first_ready_us + result.wave_compute_service_us, result.copy_finish_us);
    result.critical_path_us = std::max(result.compute_finish_us, result.copy_finish_us);
    result.score_us = std::max(result.ideal_overlap_us, result.critical_path_us);
    result.critical_worker_later_feed_wait_us = std::max(0.0,
        result.critical_worker_feed_wait_us - result.critical_worker_initial_wait_us);
    result.exposed_feed_us = std::max(0.0,
        result.compute_finish_us - result.first_ready_us - result.strided_compute_service_us);
    const double payload = static_cast<double>(2 * input.m * input.k);
    result.feed_phase_demand_gb_s = payload / result.feed_phase_ideal_compute_us / 1000.0;
    result.effective_delivery_gb_s = payload / result.copy_finish_us / 1000.0;
    result.production_consumption_ratio = result.effective_delivery_gb_s / result.feed_phase_demand_gb_s;
    if (result.copy_finish_us <= 0 || !std::isfinite(result.score_us) ||
        !std::isfinite(result.feed_phase_demand_gb_s) || !std::isfinite(result.effective_delivery_gb_s) ||
        !std::isfinite(result.production_consumption_ratio)) return fail(Status::NonfiniteScore);
    result.status = Status::Success;
    return result;
  } catch (const std::bad_alloc&) {
    return fail(Status::AllocationFailure);
  }
}

}  // namespace fuse::detail
