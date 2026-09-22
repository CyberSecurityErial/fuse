// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

#if FUSE_ENABLE_PROFILING
namespace fuse {
// Diagnostic-only, one epoch. All stamps use this GPU's %globaltimer (ns).
// release_begin/end bracket the existing ready store; neither is an exact
// visibility timestamp. Consumers never synchronize on these records.
struct GroupedPanelTimeline {
  uint64_t begin = 0, release_begin = 0, release_end = 0;
  int32_t cta = -1, expert = -1, m = -1;
};
struct GroupedTileTimeline {
  uint64_t wait_begin = 0, observed = 0, load_begin = 0, load_end = 0;
  int32_t cta = -1, expert = -1, m = -1, n = -1;
  int32_t polled = 0;
};
struct GroupedRoleTimeline {
  uint64_t begin = 0, role_end = 0, end = 0;
  // First complete input publication by this physical CTA, including borrowed
  // producers. The store is bracketed, not treated as an exact visibility time.
  uint64_t first_release_begin = 0, first_release_end = 0;
  int64_t first_release_panel = -1;
};
// One load-warp summary per CTA, flushed once on mainloop destruction. These
// waits may overlap previously issued MMA; they are NOT Tensor Core idle time.
struct GroupedReadySummary {
  uint64_t checks = 0, wait_ns = 0, max_wait_ns = 0;
  uint64_t first_wait_ns = 0, waits_ge_1us = 0;
  uint64_t first_wait_begin = 0, first_observed = 0;
};
// Diagnostic-only accumulated serial time for one Dispatch communication
// warp. Warps overlap each other, so consumers must compare per-warp maxima;
// summing these fields does not produce kernel wall time.
struct GroupedCommSummary {
  uint64_t panels = 0, batches = 0, bytes = 0;
  uint64_t address_ns = 0, store_wait_ns = 0, g2s_ns = 0, store_issue_ns = 0;
  uint64_t g2s_issue_ns = 0, g2s_wait_ns = 0;
};
struct GroupedProfile {
  GroupedPanelTimeline* panels = nullptr;
  GroupedTileTimeline* tiles = nullptr;
  GroupedRoleTimeline* roles = nullptr;
  int64_t panel_capacity = 0;
  int32_t n_tiles = 0;
  GroupedReadySummary* ready_summary = nullptr;
  GroupedCommSummary* comm_summary = nullptr;
  int32_t comm_summary_capacity = 0;
  // Split launches share one trace: communication first, then linearized
  // physical GEMM CTAs. The production scheduler still sees its original grid.
  int32_t cta_offset = 0;
};
}  // namespace fuse
#endif
