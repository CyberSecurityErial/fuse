// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/gemm_a2a.h"
#include "fuse/profiling/sm103/mxfp8.cuh"
#include <cstddef>

namespace fuse {

// Packed row-major E4M3 [M,K], with the pinned CUTLASS native SFA layout.
// Each consecutive group of 32 K elements shares one UE8M0 scale. The size
// query includes native scale padding; upstream quantizers must use this layout.
struct Mxfp8Activation {
  const Fp8E4m3* data = nullptr;
  const uint8_t* scales = nullptr;
  size_t data_bytes = 0, scale_bytes = 0;
};

enum class Mxfp8WeightPreparation {
  kCommunicationCtas,  // Default: quantize ahead while waiting/sending outputs.
  kAllCtas,           // Control: all resident CTAs quantize, then start GEMM/A2A.
  kCommunicationWarps, // Four warps quantize first, then join all eight route warps.
};

// Prequantized activation and BF16 master weight; neither is overwritten.
// Each full call quantizes the weight INSIDE the persistent kernel, performs
// MXFP8 GEMM with FP32 accumulation, and routes BF16 output. projection.lhs
// is unused; activation is the actual input. projection.rhs_nt remains BF16.
// Inputs must be finite. Scratch is caller-owned, 256-byte aligned,
// device-local, disjoint from inputs/outputs/flags, and must outlive stream
// completion. Concurrent calls retain the underlying projection ownership
// contract and additionally require distinct scratch. No allocation is implicit.
// projection.num_comm_ctas: 0 requests the calibrated communication-budget
// selector; positive values are exact overrides. Auto preserves the selected
// GEMM collective/raster/swizzle and requires matching MXFP8 service calibration;
// absent calibration returns cudaErrorNotSupported, never a guessed budget.
struct Mxfp8GemmA2AParams {
  GemmA2AParams projection{};
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  Mxfp8Activation activation{};
  Mxfp8WeightPreparation weight_preparation = Mxfp8WeightPreparation::kCommunicationCtas;
  // CUTLASS epilogue subtile N; independent of the full N256 ready panel.
  // Raster/swizzle remain explicit in projection.gemm. 64 preserves the baseline.
  int32_t epilogue_n = 64;
};

KernelTraits mxfp8_qkv_cutlass_kernel_traits(int32_t epilogue_n = 64);

// Shape-only counterpart of recommended_gemm_a2a_comm_ctas. Returns 0 when
// geometry/calibration/device support is unavailable; 0 is not a usable budget
// for allocation/profiling or prequantized diagnostic launches. The query uses
// the same resolver as production and never inspects tensor pointers or epochs.
int32_t recommended_gemm_a2a_mxfp8_comm_ctas(
    const GemmProblem& problem, const UlyssesRoute& route, int32_t epilogue_n = 64);

cudaError_t gemm_a2a_mxfp8_activation_size(
    const GemmProblem& problem, size_t* data_bytes, size_t* scale_bytes);
// Optional upstream adapter; NOT part of the MXFP8-input fused boundary.
// The caller owns the writable data/scale allocations described by output.
cudaError_t quantize_gemm_a2a_mxfp8_activation(const GemmProblem& problem,
    const Bf16* source, const Mxfp8Activation& output, cudaStream_t stream);
cudaError_t gemm_a2a_mxfp8_workspace_size(const GemmProblem& problem, size_t* bytes);
cudaError_t launch_gemm_a2a_mxfp8_cutlass(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream);

// Diagnostic control: prepare W separately, then measure GEMM+A2A alone.
// Re-prepare after W changes; this is not the dynamic-weight fused boundary.
// Both diagnostic calls require a positive communication budget. To compare
// against auto production, query its budget above and pass it explicitly here.
cudaError_t prepare_gemm_a2a_mxfp8(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream);
cudaError_t launch_gemm_a2a_mxfp8_prequantized(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream);

// Independent calibration boundaries. Both require an explicit positive
// projection.num_comm_ctas; zero is not automatic selection for diagnostics.
// Prepare weights/activation outside timing. Compute uses SM_count-comm_ctas,
// the selected E32/E64 collective and complete-output-tile drain/ready release,
// but launches no quantization or communication. Copy reads materialized BF16
// local_output without waiting for GEMM, and retains full cross-rank completion.
// Both match the selected fused kernel's dynamic SMEM and one-CTA/SM occupancy.
// Caller owns valid projection metadata/buffers, calibration ready/done flags,
// epochs and cross-rank coordination. These APIs allocate/reset nothing.
cudaError_t launch_gemm_a2a_mxfp8_compute_reference(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream);
cudaError_t launch_gemm_a2a_mxfp8_copy_reference(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream);

// Independent Q boundary: reset weight ready/arrivals, then quantize and
// publish all panels using the production 8*comm_ctas static warp workers.
// Only communication CTAs launch, with the selected E32/E64 production SMEM;
// there is no GEMM, output routing or inter-rank completion. Positive comm_ctas
// and kCommunicationCtas preparation are required. Unlike C/R above, Q resets
// its invocation-private weight flags inside the timed cooperative kernel.
// Validate by launching compute_reference afterwards WITHOUT calling prepare:
// re-preparing would overwrite the quantization output being checked.
cudaError_t launch_gemm_a2a_mxfp8_quantize_reference(
    const Mxfp8GemmA2AParams& params, cudaStream_t stream);

#if FUSE_ENABLE_PROFILING
struct QkvRouteTimeline;
// Caller clears records and supplies prepared activation/W for Compute, and
// materialized local_output for Route/QuantizeRoute. Positive comm budget and
// ordinary eight-warp preparation only. C uses the production c+G grid and
// ready acquire/release protocol; other modes use c CTAs. No allocation.
// panel_release is a POST-release timestamp, not an atomic transaction time;
// optional panel_release_begin bounds the delayed publication from below.
// ready_after is after the unchanged SignalingEpilogue returns. CTA end is
// lane-zero return (not a per-warp completion substitute). QR setup includes
// diagnostic output-flag seeding and must not calibrate production startup.
cudaError_t launch_gemm_a2a_mxfp8_service(const Mxfp8GemmA2AParams& params,
    const Mxfp8ServiceConfig& config, Mxfp8ServiceView view, cudaStream_t stream);
cudaError_t query_gemm_a2a_mxfp8_service_resources(
    const Mxfp8GemmA2AParams& params, Mxfp8ServiceResources* resources);
cudaError_t launch_gemm_a2a_mxfp8_role_telemetry(
    const Mxfp8GemmA2AParams& params, A2AGemmCtaTimeline* timeline,
    int32_t timeline_capacity, cudaStream_t stream,
    Mxfp8ProfileView probe = {}, QkvRouteTimeline* routes = nullptr,
    int32_t route_capacity = 0);
#endif

}  // namespace fuse
