// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "fuse/operators/primitives/gemm_a2a.h"

#if FUSE_ENABLE_PROFILING
namespace fuse {
// SM103 diagnostic only. One record per 64x128 route tile; trailing records
// are per-warp destination-write drains. No per-tile full wait is introduced.
struct QkvRouteTimeline {
  uint64_t begin = 0, ready = 0, g2s_begin = 0, g2s_done = 0;
  uint64_t s2g_begin = 0, s2g_read_done = 0;
  int32_t cta = 0, warp = 0, row = 0, column = 0;
  int32_t rows = 0, columns = 0, peer = 0, segment = 0;
  // After the original final warp join; includes reusable-stage completion,
  // not remote-global completion (that remains the trailing drain record).
  uint64_t copy_end = 0;
  // Optional norm/RoPE; all zero on unchanged routes and V. Arithmetic ends
  // after generic SMEM writes/warp join; publication covers the proxy fence.
  uint64_t post_begin = 0, post_math_done = 0, post_end = 0;
};

cudaError_t launch_gemm_a2a_route_telemetry(
    const GemmA2AParams&, A2AGemmCtaTimeline*, int32_t,
    QkvRouteTimeline*, int32_t, cudaStream_t);
// Includes skipped/padded task slots and per-warp final drains.
cudaError_t query_gemm_a2a_route_timeline_capacity(const GemmA2AParams&, int32_t*);
}  // namespace fuse
#endif
