// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "fuse/operators/qkv_backward.h"

#if FUSE_ARCH_SM103 && FUSE_ENABLE_PROFILING
namespace fuse {
// Diagnostic NN GEMM: same tile/scheduler/CTA resource envelope as backward B,
// pre-materialized input, no communication, ready waits or ready publication.
// reserved_ctas remains deducted from the persistent worker count; this is a
// compute CTA budget, not hardware SM affinity or an end-to-end backward API.
cudaError_t launch_backward_gemm_reference(bool qkv, int m, int n, int k,
    const Bf16* lhs, const Bf16* weight, Bf16* output,
    int reserved_ctas, cudaStream_t stream, int tile_n = 128, int tile_k = 64,
    int epilogue_n = 0, int swizzle = 1, bool along_m = false);
}
#endif
