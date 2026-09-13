// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/a2a_gemm.h"
#include "fuse/operators/semantics/attention_postprocess.h"

namespace fuse {

// Prequantized activation [batch*global_seq, local_heads*head_dim] on each
// source peer. A2A copies E4M3 data AND remaps native SFA scales into [M,K].
// BF16 master W is quantized inside the persistent kernel; GEMM accumulates
// in FP32 and writes BF16. No activation quantization is part of this call.
// projection retains geometry, routing, BF16 weight/output and upstream-ready
// metadata. Its BF16 peer_input/input_staging/ready pointers are unused.
// Scratch is device-local, 256-byte aligned, invocation-private and disjoint
// from all inputs/outputs. The caller retains peer inputs/scales until ALL
// ranks finish, just as for the BF16 inverse A2A contract. No allocation or
// cross-rank host coordination is implicit. Inputs must be finite.
// Cyclic peer order requires the same cyclic K prepack of BF16 W.
// num_comm_ctas=0 requests the offline service-balance selector; positive
// overrides remain exact. No matching calibration returns NotSupported, never
// a guessed budget or an online timing trial. GEMM/layout stay caller inputs.
// Producer-revision changes also invalidate Auto's service measurements:
// until matching calibration is available, use an explicit positive budget.
struct Mxfp8A2AGemmParams {
  A2AGemmParams projection{};
  Mxfp8Activation activation[kMaxWorldSize]{};
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  int32_t epilogue_n = 32;
  // Optional bounded traversal, in GEMM tiles. Both zero keep CUTLASS order;
  // otherwise both must be positive powers of two and comm CTAs explicit.
  // GEMM and A first-use share this order; full A/W ready units do not change.
  int32_t m_window_tiles = 0;
  int32_t n_group_tiles = 0;
  ResidualRmsNorm postprocess{};
  // Experimental: each CTA finishes its own A/W or GEMM role, then shares
  // complete output-row norm tasks. Requires postprocess, hidden width <=16384
  // and explicit communication CTAs. False retains the whole-grid tail.
  bool overlap_postnorm = false;
};

cudaError_t a2a_gemm_mxfp8_activation_size(const GemmProblem& problem,
    const UlyssesRoute& route, size_t* data_bytes, size_t* scale_bytes);
cudaError_t a2a_gemm_mxfp8_workspace_size(const GemmProblem& problem, size_t* bytes);
cudaError_t a2a_gemm_mxfp8_workspace_size(const Mxfp8A2AGemmParams& params, size_t* bytes);
KernelTraits mxfp8_oproj_cutlass_kernel_traits(int32_t epilogue_n = 32, bool overlap_postnorm = false);
// Shape-only query sharing the production resolver. Returns 0 when no measured
// domain applies; 0 is not a usable diagnostic/allocation budget.
int32_t recommended_a2a_gemm_mxfp8_comm_ctas(
    const GemmProblem& problem, const UlyssesRoute& route, int32_t epilogue_n = 32);
// Read-only diagnostic view, valid after launch completion until scratch is
// reused. No preparation, allocation or device work; never changes the result.
cudaError_t a2a_gemm_mxfp8_staging_view(
    const Mxfp8A2AGemmParams& params, Mxfp8Activation* activation);
cudaError_t launch_a2a_gemm_mxfp8_cutlass(
    const Mxfp8A2AGemmParams& params, cudaStream_t stream);

// Comparison only: original A2A/GEMM followed by a separate optimized
// residual/RMSNorm kernel on the same stream. Both launches belong in timing.
// Identical arithmetic/output contract; explicit budget, no row-ready overlap.
cudaError_t launch_a2a_gemm_mxfp8_postnorm_reference(
    const Mxfp8A2AGemmParams& params, cudaStream_t stream);

// Independent services with the production CTA/SMEM footprint and explicit
// communication budget. Compute reads already prepared workspace A/W; copy
// produces A/SFA only; producer executes the actual A-copy + W-quantization
// worker path without GEMM consumers. Preparation/validation belongs outside
// each measured boundary. These are diagnostics, never called by Auto.
cudaError_t launch_a2a_gemm_mxfp8_compute_reference(
    const Mxfp8A2AGemmParams& params, cudaStream_t stream);
cudaError_t launch_a2a_gemm_mxfp8_copy_reference(
    const Mxfp8A2AGemmParams& params, cudaStream_t stream);
cudaError_t launch_a2a_gemm_mxfp8_producer_reference(
    const Mxfp8A2AGemmParams& params, cudaStream_t stream);

#if FUSE_ENABLE_PROFILING
struct Mxfp8ProfileView;
cudaError_t launch_a2a_gemm_mxfp8_role_telemetry(
    const Mxfp8A2AGemmParams& params, A2AGemmCtaTimeline* timeline,
    int32_t capacity, A2AGemmPeerTimeline* peers, int32_t peer_capacity,
    Mxfp8ProfileView weight, cudaStream_t stream);
#endif

}  // namespace fuse
