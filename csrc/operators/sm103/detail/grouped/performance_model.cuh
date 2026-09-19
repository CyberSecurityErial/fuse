// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "producer_consumer.cuh"
#include <cfloat>

namespace fuse::detail {

struct GroupedInputModelResult {
  bool valid = false;
  int64_t tiles = 0, waves = 0, critical_panel = -1;
  double compute_us = 0, copy_us = 0, first_ready_us = 0, finish_us = 0;
  double exposed_feed_us = 0;
};

// Offline scorer for the native full-K-panel Dispatch pipeline.
//
// For one candidate CTA split and consumer layout:
//
//   C       base communication CTAs
//   P       compute CTAs (normally launch_ctas - C)
//   T       actual GEMM tiles derived from device row_offsets
//   q[p]    first logical tile which consumes panel p under Along/swizzle
//   R[p]    panel-p release under THIS complete candidate (C, P, L, copy policy)
//   tau(P)  measured service time of one persistent GEMM worker step
//
// A worker executes q, q+P, ... and cannot start a tile before its full-K panel
// is ready:
//
//   finish[q] = max(finish[q-P], R[panel(q)]) + tau(P)
//
// Therefore only each panel's first consumer can set the final endpoint:
//
//   G(P)       = ceil(T/P) * tau(P)
//   remain[p]  = floor((T - 1 - q[p])/P) + 1
//   F(C,P,L)   = max(G(P), max_p(R[p] + remain[p] * tau(P)))
//   exposed(C) = F(C,P,L) - G(P)
//
// `exposed` is the producer delay extending this modeled GEMM dependency chain.
// Compare discrete candidates by F, not by forcing standalone
// communication and GEMM throughput to equality. A producer deficit is worth
// fixing only when the reduction in `exposed` is larger than the increase in
// G caused by removing compute resources.
//
// Auto candidate generation can branch on actual routed rows; it need not
// force small and large workloads into the same service model or search order:
//
//   max_e(M_e) < 192: jointly score first delivery, integer GEMM waves and
//                     exposed waits across GEMM/layout/CTA candidates.
//   otherwise:        start from the best measured GEMM
//                     service and add the smallest C whose reduction in later
//                     panel waits pays for its lost compute service.
//
// The provisional 192 boundary selects a policy priority, not a proven
// bottleneck. Every expert still contributes to T, q[p] and R[p,C]; neither
// branch replaces a skewed distribution with mean M. Device row_offsets are
// read on every Graph replay, without a model-name or benchmark-row lookup.
//
// Along/swizzle change q[p], even when T and the wave count are unchanged.
// R[p] may be unordered because producers run concurrently; no in-order
// completion is assumed. tau must be calibrated for the exact tile, K, layout
// and compute budget, and must already include steady pipeline cost. Changing
// L can change first-wave breadth and producer splits, hence R even with a full
// buffer. Never transplant a ready trace without matching the producer policy.
// This is exact only for uniform tau, fixed round-robin workers and exogenous
// releases. Prefetch, variable tail service, concurrent resource interference
// and entry/exit synchronization require separate validation; F is not E2E.
//
// Base-C selection and idle-CTA lending are intentionally separate. After C is
// chosen, Auto may lend at most
//
//   min(P - min(T,P), max(0, independent_producer_groups - C))
//
// otherwise-idle compute CTAs. A positive explicit C is exact and never lends.
// Runtime selection reads geometry and device row_offsets only: no online
// timing, host readback, model-name lookup, allocation, or Graph recapture.
CUTLASS_HOST_DEVICE inline GroupedInputModelResult score_grouped_input_schedule(
    const GroupedTileOrder& order, const double* ready_us,
    int compute_ctas, double tile_us) {
  GroupedInputModelResult result;
  if (!order.row_tile_offsets || order.experts <= 0 || order.n_tiles <= 0 ||
      compute_ctas <= 0 || !(tile_us > 0 && tile_us <= DBL_MAX) ||
      (order.swizzle != 1 && order.swizzle != 2 && order.swizzle != 4 && order.swizzle != 8) ||
      order.window_m < 0 || order.row_tile_offsets[0] != 0) return result;
  for (int e = 0; e < order.experts; ++e)
    if (order.row_tile_offsets[e+1] < order.row_tile_offsets[e]) return result;
  const int64_t panels = order.row_tiles();
  if (panels > INT64_MAX / order.n_tiles || (panels && !ready_us)) return result;
  result.tiles = panels * order.n_tiles;
  result.waves = result.tiles / compute_ctas + (result.tiles % compute_ctas != 0);
  result.compute_us = result.waves * tile_us;
  result.finish_us = result.compute_us;
  result.first_ready_us = panels ? DBL_MAX : 0;
  for (int e = 0; e < order.experts; ++e) {
    const int64_t begin = order.row_tile_offsets[e], end = order.row_tile_offsets[e+1];
    if (end - begin > INT32_MAX) return {};
    for (int64_t panel = begin; panel < end; ++panel) {
      const double ready = ready_us[panel];
      if (!(ready >= 0 && ready <= DBL_MAX)) return {};
      if (ready > result.copy_us) result.copy_us = ready;
      const int64_t first = order.linear(e, int(panel - begin), 0);
      const int64_t remaining = (result.tiles - 1 - first) / compute_ctas + 1;
      const double endpoint = ready + remaining * tile_us;
      if (ready < result.first_ready_us) result.first_ready_us = ready;
      if (endpoint > result.finish_us) {
        result.finish_us = endpoint;
        result.critical_panel = panel;
      }
    }
  }
  if (!(result.finish_us <= DBL_MAX)) return {};
  result.exposed_feed_us = result.finish_us - result.compute_us;
  result.valid = true;
  return result;
}

// Independently measured services, in us per row owned by ONE producer CTA.
// Staged split panels restart staging for each 8-row stripe, unlike a complete
// staged panel. Keep those services distinct instead of assuming equal bytes
// imply equal time. These rates include copy/join/publication at the calibrated
// budget; they are not peak NVLink bandwidth. No fused winners enter them.
struct GroupedDispatchServices {
  double tile_us = 0;
  double vector_row_us = 0;
  double staged_row_us = 0;
  double staged_stripe_row_us = 0;
};

// Reconstruct the actual producer queues, then score their full-panel releases.
// No artificial barrier between cohorts:
//   work = panel * splits + stripe
//   CTA  = work % C             (next work for this CTA is work + C)
//   R[p] = max(end of p's nonempty stripes)
// Every expert uses its own routed row count, including short/empty tails.
// Domain: native full-buffer Dispatch, no optional last-cohort tail sharing.
// Unsupported policies must use their own model, not silently reuse this one.
// This diagnostic scorer is O(panels * splits); device selection cost must be
// measured before enabling it in production preparation.
CUTLASS_HOST_DEVICE inline GroupedInputModelResult score_grouped_dispatch_candidate(
    const GroupedTileOrder& order, const int64_t* row_offsets, int k,
    int comm_ctas, int compute_ctas, const GroupedDispatchServices& service) {
  constexpr int kMaxCtas = 148;
  GroupedInputModelResult result;
  auto positive = [](double v) { return v > 0 && v <= DBL_MAX; };
  if (!row_offsets || !order.row_tile_offsets || order.experts <= 0 || order.n_tiles <= 0 ||
      row_offsets[0] != 0 || order.row_tile_offsets[0] != 0 || order.window_m != 0 ||
      k <= 0 || comm_ctas <= 0 || compute_ctas <= 0 ||
      comm_ctas > kMaxCtas-compute_ctas ||
      (order.swizzle != 1 && order.swizzle != 2 && order.swizzle != 4 && order.swizzle != 8) ||
      !positive(service.tile_us) || !positive(service.vector_row_us) ||
      !positive(service.staged_row_us) || !positive(service.staged_stripe_row_us)) return result;
  for (int e=0; e<order.experts; ++e) {
    const int64_t rows=row_offsets[e+1]-row_offsets[e];
    if (row_offsets[e+1] < row_offsets[e] || rows > INT32_MAX ||
        order.row_tile_offsets[e+1]-order.row_tile_offsets[e] != (rows+127)/128) return result;
  }
  const int64_t panels=order.row_tiles();
  if (panels < 0 || panels > INT64_MAX/order.n_tiles) return result;
  result.tiles=panels*order.n_tiles;
  result.waves=result.tiles/compute_ctas+(result.tiles%compute_ctas!=0);
  result.compute_us=result.waves*service.tile_us;
  result.finish_us=result.compute_us;
  result.first_ready_us=panels ? DBL_MAX : 0;
  const int first_wave=grouped_use_latency_cohort(row_offsets,order.experts)
      ? grouped_first_wave_panels(order,compute_ctas) : -1;
  const int splits=grouped_dispatch_splits(panels,comm_ctas,first_wave,int64_t(128)*k*2);
  double producer_end[kMaxCtas]{};
  for (int e=0; e<order.experts; ++e) {
    const int64_t rows=row_offsets[e+1]-row_offsets[e];
    const bool staged=rows>128 && int64_t(k)*2<=24*1024;
    const double row_us=!staged ? service.vector_row_us :
        (splits>1 ? service.staged_stripe_row_us : service.staged_row_us);
    for (int64_t p=order.row_tile_offsets[e]; p<order.row_tile_offsets[e+1]; ++p) {
      const int m=int(p-order.row_tile_offsets[e]);
      const int valid=int(rows-int64_t(m)*128 < 128 ? rows-int64_t(m)*128 : 128);
      const int groups=(valid+7)/8;
      const int producers=splits<groups ? splits : groups;
      double ready=0;
      for (int stripe=0; stripe<producers; ++stripe) {
        const int remainder=valid%(8*producers)-8*stripe;
        const int owned=(valid/(8*producers))*8+
            (remainder<=0 ? 0 : (remainder<8 ? remainder : 8));
        const int c=int((p*splits+stripe)%comm_ctas);
        producer_end[c]+=owned*row_us;
        if (producer_end[c]>ready) ready=producer_end[c];
      }
      if (ready>result.copy_us) result.copy_us=ready;
      if (ready<result.first_ready_us) result.first_ready_us=ready;
      const int64_t first=order.linear(e,m,0);
      const int64_t remain=(result.tiles-1-first)/compute_ctas+1;
      const double endpoint=ready+remain*service.tile_us;
      if (endpoint>result.finish_us) {
        result.finish_us=endpoint;
        result.critical_panel=p;
      }
    }
  }
  if (!(result.finish_us<=DBL_MAX)) return {};
  result.exposed_feed_us=result.finish_us-result.compute_us;
  result.valid=true;
  return result;
}

struct GroupedDispatchCandidate {
  int comm_ctas = 0, compute_ctas = 0, swizzle = 1;
  bool along_n = false;
  GroupedDispatchServices service{};
};

struct GroupedDispatchSelection {
  int index = -1;
  GroupedInputModelResult prediction{};
};

// Service calibration and candidate generation are separate from selection.
// A small-token caller may enumerate GEMM/layout alternatives; a large-token
// caller can hold the chosen GEMM fixed and supply only resource splits. In
// either case, predict the releases anew for each candidate's actual layout.
// Borrowing is deliberately absent: choose the base producer budget first.
// Invalid/unsupported candidates are not a license to invent service rates.
CUTLASS_HOST_DEVICE inline GroupedDispatchSelection select_grouped_dispatch_candidate(
    GroupedTileOrder order, const int64_t* row_offsets, int k,
    const GroupedDispatchCandidate* candidates, int count) {
  GroupedDispatchSelection best;
  if (!candidates || count<=0) return best;
  for (int i=0; i<count; ++i) {
    const auto& candidate=candidates[i];
    order.swizzle=candidate.swizzle;
    order.along_n=candidate.along_n;
    const auto prediction=score_grouped_dispatch_candidate(order,row_offsets,k,
        candidate.comm_ctas,candidate.compute_ctas,candidate.service);
    if (!prediction.valid) continue;
    if (best.index<0 || prediction.finish_us<best.prediction.finish_us ||
        (prediction.finish_us==best.prediction.finish_us &&
         candidate.comm_ctas<candidates[best.index].comm_ctas)) {
      best={i,prediction};
    }
  }
  return best;
}

}  // namespace fuse::detail
