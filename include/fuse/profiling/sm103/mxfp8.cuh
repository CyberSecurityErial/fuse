// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

#if FUSE_ENABLE_PROFILING
namespace fuse {
// One record per existing 32-group quantization step, not per element.
// Distinct queue steps have distinct writers; telemetry adds no work-claim
// atomics or synchronization. Times are device-local globaltimer ticks.
struct Mxfp8QuantRecord {
  uint64_t begin = 0, quant_done = 0, end = 0, release = 0;
  uint64_t warp_join_done = 0, arrival_done = 0;
  int32_t cta = 0, warp = 0, panel = 0, groups = 0;
  // Zero for local bookkeeping; positive only at this worker's final panel chunk.
  uint32_t arrival_chunks = 0;
};
static_assert(sizeof(Mxfp8QuantRecord) == 72);
struct Mxfp8WaitRecord {
  uint64_t begin = 0, end = 0;
  int32_t cta = 0, warp = 0;
};
struct Mxfp8ProfileView {
  Mxfp8QuantRecord* quant = nullptr;
  Mxfp8WaitRecord* waits = nullptr;
  int32_t quant_capacity = 0, wait_capacity = 0;
};

// Independent service diagnostics, never instantiated in production builds.
// All timestamps share this GPU's globaltimer; different fields in a tile
// have distinct writers (load warp / issuing epilogue warp).
struct Mxfp8ServiceTileRecord {
  uint64_t first_load = 0, load_return = 0, store_begin = 0, ready_after = 0;
  int32_t cta = 0, warp = 0, m = 0, n = 0;
};
struct Mxfp8ServiceCtaRecord {
  uint64_t begin = 0, setup_done = 0, end = 0;
};
struct QkvRouteTimeline;
struct Mxfp8ServiceView {
  Mxfp8ServiceTileRecord* tiles = nullptr;
  int32_t tile_capacity = 0;
  Mxfp8ServiceCtaRecord* ctas = nullptr;
  int32_t cta_capacity = 0;
  uint64_t* panel_release = nullptr;
  uint64_t* panel_release_begin = nullptr;
  int32_t panel_capacity = 0;
  Mxfp8ProfileView weight{};
  QkvRouteTimeline* routes = nullptr;
  int32_t route_capacity = 0;
};
enum class Mxfp8ServiceMode { kCompute, kQuantize, kRoute, kQuantizeRoute };
struct Mxfp8ServiceConfig {
  Mxfp8ServiceMode mode = Mxfp8ServiceMode::kCompute;
  // Compute only: withhold one panel until this interval from CTA0 entry.
  // At least one CTA must use it after computing another panel; first-tile
  // consumers, if any, are excluded from steady-state recovery measurement.
  // W data itself must already be quantized.
  int32_t delayed_panel = -1;
  uint64_t delay_ns = 0;
  // QR only: perform one ordinary ready-wait-sized progress before routing,
  // then continue the SAME worker queue. This changes the measurement phase,
  // not the production policy, quantization granularity, or publication rule.
  int32_t quant_phase_steps = 0;
};
struct Mxfp8ServiceResources {
  int32_t stages = 0, dynamic_smem_bytes = 0, compute_ctas = 0;
  int32_t delayed_panel = -1;
  int32_t resolved_swizzle = 0;
};
}  // namespace fuse
#endif
