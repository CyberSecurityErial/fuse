// SPDX-License-Identifier: MIT
// Compile-only representative H200 instantiation for the prefill CTASP path.
// This file deliberately has no host main; see scripts/build_megamoe_ctasp.sh.

#define DG_DEVICE_ASSERT(cond) do { if (not (cond)) asm("trap;"); } while (0)
#define DG_NO_DEVICE_PRINTF

#include "sm90_fp8_mega_moe_ctasp.cuh"

using namespace deep_gemm;

static void instantiate_sm90_ctasp_prefill() {
  auto* kernel = reinterpret_cast<void*>(
      &sm90_fp8_mega_moe_ctasp_impl<
          4096,                 // max tokens / rank
          7168, 2048,           // hidden, intermediate
          256, 8,               // global experts, top-k
          32,                   // experts / wave (compatibility field)
          128, 128, 128,        // CTA tile
          268416,               // maximum routed-token pool
          268416,               // SF-padded routed-token pool
          4,                    // TMA stages (accepted H200 CTASP default)
          64, 64, 256,          // dispatch, TMA, math threads
          132, 8,               // H200 SMs, ranks
          80, 4, 32,            // FC1, early-scatter CTAs, ring blocks
          10.0f, true, false,   // clamp, fast math, legacy L2 raster
          true>);               // overlap FC1-CTA scatter with FC2
  (void)kernel;
}
