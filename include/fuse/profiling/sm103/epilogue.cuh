// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// SM103-specific diagnostics; shared profiling records remain one directory up.

// Private, opt-in QKV diagnostics. Public timeline layouts and production
// launch arguments are unchanged. Never publish these timings as MPI results.
#if FUSE_ENABLE_PROFILING
#include "fuse/profiling/timeline.cuh"

#include <cstdint>
#include <type_traits>

namespace fuse {
struct GemmA2AParams;
namespace detail {

// One writer: lane zero of the issuing epilogue warp, indexed by physical
// blockIdx.x. Accumulate in that lane's state and flush once in store_tail.
// Durations include instrumentation; the drain includes the existing warp
// join. Base::store includes waiting for TMEM, not just epilogue arithmetic.
struct alignas(16) QkvEpilogueRecord {
  uint64_t store_ns_sum = 0;
  uint64_t store_ns_max = 0;
  uint64_t drain_ns_sum = 0;
  uint64_t drain_ns_max = 0;
  uint64_t first_store_begin = 0;
  uint64_t first_store_end = 0;
  uint64_t first_drain_end = 0;
  uint64_t first_ready_after = 0;
  uint64_t last_ready_after = 0;
  uint64_t tile_count = 0;
  int32_t first_m_tile = 0;
  int32_t first_n_tile = 0;
  int32_t first_batch = 0;
  uint32_t epoch = 0;
};
static_assert(sizeof(QkvEpilogueRecord) == 96);
static_assert(std::is_trivially_copyable_v<QkvEpilogueRecord>);

struct QkvEpilogueResources {
  static constexpr const char* kSchema = "qkv_epilogue_cta_v1";
  static constexpr const char* kClock = "globaltimer";
  static constexpr const char* kClockUnit = "ns";
  static constexpr const char* kStoreInterval = "issuing_lane_base_call_including_accumulator_wait";
  static constexpr const char* kDrainInterval = "issuing_warp_global_wait_and_warp_join";
  // The private probe's CTA role_done timestamp is gated by the completed
  // population-count join. This extra diagnostic cost is not production work.
  static constexpr const char* kRoleTimestampJoin = "cta_popc256_dependency";
  cudaFuncAttributes production{};
  cudaFuncAttributes role_telemetry{};
  cudaFuncAttributes epilogue_telemetry{};
  int32_t dynamic_smem_bytes = 0;
  int32_t tile_m = 0;
  int32_t tile_n = 0;
  int32_t tile_k = 0;
  int32_t cluster_ctas = 0;
};

// Current private probe is only N256/K64/epilogue-N32, non-interleaved QKV.
// Both buffers must have capacity >= the current device's SM count, be on
// that device, and stay alive until stream completion. Clear only diagnostic
// buffers before sampling; never reset cumulative ready/epoch state here.
cudaError_t launch_qkv_epilogue_telemetry(
    const GemmA2AParams& params,
    A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity,
    QkvEpilogueRecord* records,
    int32_t record_capacity,
    cudaStream_t stream);

cudaError_t query_qkv_epilogue_resources(
    const GemmA2AParams& params, QkvEpilogueResources* resources);

#if defined(__CUDACC__)
CUTLASS_DEVICE uint64_t epilogue_timestamp() {
  // Compiler barriers only, not new GPU fences or synchronization. Inspect
  // emitted code before treating the bracketing as instruction-level evidence.
  asm volatile("" ::: "memory");
  const uint64_t now = read_global_timer();
  asm volatile("" ::: "memory");
  return now;
}
#endif

}  // namespace detail
}  // namespace fuse
#endif
