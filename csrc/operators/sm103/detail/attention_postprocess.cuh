// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/semantics/attention_postprocess.h"
#include <cutlass/array.h>
#include <cute/arch/copy_sm80.hpp>

namespace fuse {
namespace {

struct NoQkvPostprocess { static constexpr bool kEnabled = false; };
constexpr int kPostnormScratchFloats = 4*8 + 1; // Eight virtual warp sums/row + queue claim.

CUTLASS_DEVICE uint32_t postprocess_pack(float low, float high) {
  uint32_t result;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(result) : "f"(low), "f"(high));
  return result;
}
CUTLASS_DEVICE uint32_t postprocess_mul(uint32_t a, uint32_t b) {
  uint32_t result;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
  return result;
}
CUTLASS_DEVICE uint32_t postprocess_add(uint32_t a, uint32_t b) {
  uint32_t result;
  asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
  return result;
}

// Each communication warp already owns a complete BF16 [64,128] head in
// unswizzled SMEM. Reuse that slot before issuing its remote store:
//
// GEMM --ready--> TMA G2S --wait--> Q/K RMSNorm --> RoPE --> TMA S2G
//                                      V ------------------^
//
// Norm reduces exactly one head, never a GEMM N256 panel or a CP shard.
// RoPE pairs d with d+64, not adjacent features. All lanes finish generic
// SMEM writes before the async proxy fence. Neither producer ready granularity
// nor route ownership changes. local_output retains the raw BF16 projection,
// also available for a future training backward/recomputation implementation.
struct QkvHeadPostprocess {
  static constexpr bool kEnabled = true;
  QkvPostprocess params{};

  // Production uses contiguous128-column SMEM; the separate diagnostic reuses
  // this arithmetic on strided received GMEM heads. Default stride is constant.
  CUTLASS_DEVICE void operator()(Bf16* stage, int rows, int source_row, int segment,
      int row_stride = 128) const {
    if (segment == 2 || !params.enabled()) return;
    const int lane = threadIdx.x & 31;
    constexpr int kHeadThreads = 2, kValues = 128 / kHeadThreads;
    using Pack = cutlass::AlignedArray<Bf16, kValues, 16>;
    using HalfPack = cutlass::AlignedArray<Bf16, kValues / 2, 16>;
    const int head_lane = lane % kHeadThreads;
    const Bf16* gamma = segment == 0 ? params.q_gamma : params.k_gamma;
    // Sixteen independent rows per warp; each lane owns both RoPE partners:
    //   lane within head       0             1
    //   lower half           [0:32)       [32:64)
    //   upper half          [64:96)       [96:128)
    // Rotation is register-local, not a shuffle per pair. Norm alone reduces
    // across two lanes. Each lane's FP32 sum visits its lower half then
    // its upper half; the scalar oracle independently follows this tree.
    // No extra CTA barrier, TMA slot, or change to head/token ownership.
    Pack weight{};
    const int feature = head_lane * (kValues / 2);
    if (gamma) {
      auto* halves = reinterpret_cast<HalfPack*>(&weight);
      halves[0] = *reinterpret_cast<const HalfPack*>(gamma + feature);
      halves[1] = *reinterpret_cast<const HalfPack*>(gamma + feature + 64);
    }
    for (int row = lane / kHeadThreads; row < rows; row += 32 / kHeadThreads) {
      const int64_t offset = int64_t(source_row + row) * 128 + feature;
      Pack values;
      auto* value_halves = reinterpret_cast<HalfPack*>(&values);
      value_halves[0] = *reinterpret_cast<const HalfPack*>(stage + row * row_stride + feature);
      value_halves[1] = *reinterpret_cast<const HalfPack*>(stage + row * row_stride + feature + 64);
      Pack cosine{}, sine{};
      if (params.cos) {
        auto* c = reinterpret_cast<HalfPack*>(&cosine);
        auto* s = reinterpret_cast<HalfPack*>(&sine);
        c[0] = *reinterpret_cast<const HalfPack*>(params.cos + offset);
        c[1] = *reinterpret_cast<const HalfPack*>(params.cos + offset + 64);
        s[0] = *reinterpret_cast<const HalfPack*>(params.sin + offset);
        s[1] = *reinterpret_cast<const HalfPack*>(params.sin + offset + 64);
      }
      float x[kValues];
      auto* packed = reinterpret_cast<uint32_t*>(&values);
      const auto* packed_weight = reinterpret_cast<const uint32_t*>(&weight);
      const auto* packed_cosine = reinterpret_cast<const uint32_t*>(&cosine);
      const auto* packed_sine = reinterpret_cast<const uint32_t*>(&sine);
      float sum = 0.f;
      #pragma unroll
      for (int i = 0; i < kValues; ++i) {
        x[i] = float(values[i]);
        // Separate multiply/add matches the independent eager reference's
        // FP32 square/reduction contract; reduction associativity may differ.
        sum += __fmul_rn(x[i], x[i]);
      }
      if (gamma) {
        #pragma unroll
        for (int delta = kHeadThreads / 2; delta; delta >>= 1)
          sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, delta, kHeadThreads));
        const float inv_rms = rsqrtf(sum * (1.f / 128.f) + params.epsilon);
        #pragma unroll
        for (int i = 0; i < kValues / 2; ++i) {
          const uint32_t normalized = postprocess_pack(x[2 * i] * inv_rms, x[2 * i + 1] * inv_rms);
          packed[i] = postprocess_mul(normalized, packed_weight[i]);
        }
      }
      // Keep rounded BF16 pairs packed through gamma/rotation. This avoids
      // repeated scalar BF16->FP32->BF16 conversions. Explicit .rn
      // multiply/add preserve the intermediate BF16
      // products: contraction into a single FMA would change this boundary.
      #pragma unroll
      for (int i = 0; i < kValues / 4; ++i) {
        if (params.cos) {
          constexpr int kHalfPairs = kValues / 4;
          const uint32_t lower = packed[i], upper = packed[i + kHalfPairs];
          packed[i] = postprocess_add(postprocess_mul(lower, packed_cosine[i]),
              postprocess_mul(upper ^ 0x80008000u, packed_sine[i]));
          packed[i + kHalfPairs] = postprocess_add(
              postprocess_mul(upper, packed_cosine[i + kHalfPairs]),
              postprocess_mul(lower, packed_sine[i + kHalfPairs]));
        }
      }
      *reinterpret_cast<HalfPack*>(stage + row * row_stride + feature) = value_halves[0];
      *reinterpret_cast<HalfPack*>(stage + row * row_stride + feature + 64) = value_halves[1];
    }
    __syncwarp();
  }
};

// Multiple full-hidden rows per CTA. Each physical sub-CTA cohort emulates
// the SAME virtual256 FP32 tree as the one-row implementation, keeping all
// virtual-thread partials independent until the original eight-warp sum.
// Row grouping amortizes CTA-wide barriers without changing the arithmetic.
// All256 threads arrive at barriers together; no half-CTA named barriers.
template <int RowsPerCta, bool Instrumented = false, int StagingBytes = 0>
CUTLASS_DEVICE void residual_rmsnorm_rows(Bf16* output, int width, int64_t first_row,
    const ResidualRmsNorm& post, Bf16* added_rows, float* partial, uint64_t* phases = nullptr) {
  static_assert(RowsPerCta == 2 || RowsPerCta == 4);
  constexpr int kVector = 8, kRowThreads = 256 / RowsPerCta, kVirtualThreads = 256;
  using Pack = cutlass::AlignedArray<Bf16, kVector, 16>;
  const int thread = threadIdx.x % kRowThreads, lane = thread & 31;
  const int row_in_cta = threadIdx.x / kRowThreads;
  const int64_t base = (first_row + row_in_cta) * width;
  Bf16* added_row = added_rows + row_in_cta * width;
  float sum[RowsPerCta] = {};
  uint64_t phase_start = 0;
  if constexpr (Instrumented) if (threadIdx.x == 0) phase_start = detail::read_global_timer();
  // When the already-reclaimed CTA storage fits BOTH inputs, issue all vector
  // reads asynchronously before the arithmetic. This exposes memory-level
  // parallelism instead of interleaving each global read with its reduction.
  //
  // output   --cp.async--> [projection rows] --add--> [BF16 residual sums]
  // residual --cp.async--> [residual rows]                |
  //                 wait_all + CTA join          unchanged virtual256 norm
  //
  // Input readiness has already been acquired by the caller. Each element's
  // sole owner overwrites its projection cache only after reading it; no other
  // thread uses that element before the original reduction barrier. Capacity
  // is the existing GEMM allocation, not a larger per-CTA resource request.
  const bool staged = StagingBytes > 0 &&
      int64_t(2)*RowsPerCta*width*sizeof(Bf16) <= StagingBytes;
  const Bf16* cached_residual = added_rows + RowsPerCta*width;
  if (staged) {
    auto* projection = reinterpret_cast<uint4*>(added_rows);
    auto* residual = reinterpret_cast<uint4*>(added_rows + RowsPerCta*width);
    const auto* source = reinterpret_cast<const uint4*>(output + first_row*width);
    const auto* skip = reinterpret_cast<const uint4*>(post.residual + first_row*width);
    for (int i = threadIdx.x; i < RowsPerCta*width/kVector; i += 256) {
      cute::SM80_CP_ASYNC_CACHEGLOBAL<uint4>::copy(source[i],projection[i]);
      cute::SM80_CP_ASYNC_CACHEGLOBAL<uint4>::copy(skip[i],residual[i]);
    }
    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();
  }
  #pragma unroll
  for (int half = 0; half < RowsPerCta; ++half) {
    for (int col = (thread + half*kRowThreads)*kVector; col < width;
         col += kVirtualThreads*kVector) {
      Pack value, residual;
      if (staged) {
        value = *reinterpret_cast<const Pack*>(added_row + col);
        residual = *reinterpret_cast<const Pack*>(cached_residual + row_in_cta*width + col);
      } else {
        value = *reinterpret_cast<const Pack*>(output + base + col);
        residual = *reinterpret_cast<const Pack*>(post.residual + base + col);
      }
      Pack added;
      const auto* value_pairs = reinterpret_cast<const uint32_t*>(&value);
      const auto* residual_pairs = reinterpret_cast<const uint32_t*>(&residual);
      auto* pairs = reinterpret_cast<uint32_t*>(&added);
      #pragma unroll
      for (int i = 0; i < kVector/2; ++i)
        pairs[i] = postprocess_add(value_pairs[i],residual_pairs[i]);
      #pragma unroll
      for (int i = 0; i < kVector; ++i)
        sum[half] += __fmul_rn(float(added[i]),float(added[i]));
      *reinterpret_cast<Pack*>(post.residual_output + base + col) = added;
      *reinterpret_cast<Pack*>(added_row + col) = added;
    }
    #pragma unroll
    for (int delta = 16; delta; delta >>= 1)
      sum[half] += __shfl_xor_sync(0xffffffffu,sum[half],delta);
    if (lane == 0) partial[row_in_cta*8 + half*(kRowThreads/32) + thread/32] = sum[half];
  }
  __syncthreads();
  if constexpr (Instrumented) if (threadIdx.x == 0) {
    const auto now = detail::read_global_timer();
    phases[0] += now - phase_start;
    phase_start = now;
  }
  if (thread == 0) {
    float total = partial[row_in_cta*8];
    for (int warp = 1; warp < 8; ++warp) total += partial[row_in_cta*8+warp];
    partial[row_in_cta*8] = rsqrtf(total/width + post.epsilon);
  }
  __syncthreads();
  if constexpr (Instrumented) if (threadIdx.x == 0) {
    const auto now = detail::read_global_timer();
    phases[1] += now - phase_start;
    phase_start = now;
  }
  const float inverse = partial[row_in_cta*8];
  for (int col = thread*kVector; col < width; col += kRowThreads*kVector) {
    Pack added = *reinterpret_cast<const Pack*>(added_row + col);
    const Pack gamma = *reinterpret_cast<const Pack*>(post.gamma + col);
    auto* pairs = reinterpret_cast<uint32_t*>(&added);
    const auto* gamma_pairs = reinterpret_cast<const uint32_t*>(&gamma);
    #pragma unroll
    for (int i = 0; i < kVector/2; ++i) {
      const uint32_t normalized = postprocess_pack(
          float(added[2*i])*inverse,float(added[2*i+1])*inverse);
      pairs[i] = postprocess_mul(normalized,gamma_pairs[i]);
    }
    *reinterpret_cast<Pack*>(output + base + col) = added;
  }
  __syncthreads();
  if constexpr (Instrumented) if (threadIdx.x == 0)
    phases[2] += detail::read_global_timer() - phase_start;
}

// Optimized separated-boundary control, not the FP64 correctness oracle.
// A small-SMEM ordinary grid lets the scheduler use multiple CTAs per SM;
// it deliberately does not inherit the persistent GEMM resource footprint.
__global__ void residual_rmsnorm_reference_kernel(Bf16* output, int width,
                                                 ResidualRmsNorm post) {
  extern __shared__ char shared[];
  auto* rows = reinterpret_cast<Bf16*>(shared);
  auto* partial = reinterpret_cast<float*>(rows + 2*width);
  residual_rmsnorm_rows<2>(output,width,int64_t(blockIdx.x)*2,post,rows,partial);
}

// Only workers whose OWN original production/consumption loop has drained
// enter this queue. They cannot hold unissued A, W, or GEMM work while waiting
// for a norm row, so there is no producer->norm->same-producer dependency.
//
// A/W CTA drained -----\
//                       +--> atomic claim 8 rows --> all N ready --> full norm
// GEMM CTA drained ----/             (one owner, no cross-worker finish order)
//
// Early norm uses finished communication CTAs; finished GEMM CTAs later help
// instead of idling behind a smaller norm worker pool. This is work distribution, not
// online timing/autotune. A/W/GEMM tile schedules and ready granularity stay.
template <int StagingBytes = 0, bool Instrumented = false, class CommParams>
CUTLASS_DEVICE void consume_postnorm_rows(const CommParams& a, float* partial,
                                         Bf16* row_cache = nullptr
#if FUSE_ENABLE_PROFILING
                                         , A2AGemmCtaTimeline* timeline = nullptr
#endif
                                         ) {
  constexpr int Threads = 256;
  const int thread = threadIdx.x;
  const auto& p = a.params;
  const int n_tiles = a.weights.workspace.panels;
  auto* claimed = reinterpret_cast<uint32_t*>(partial + kPostnormScratchFloats - 1);
#if FUSE_ENABLE_PROFILING
  uint64_t begin = 0, wait_ns = 0, work_ns = 0;
  uint64_t phases[3]{};
  uint32_t rows = 0;
  if constexpr (Instrumented) if (thread == 0) begin = detail::read_global_timer();
#endif
  for (;;) {
    if (thread == 0) *claimed = atomicAdd(a.workspace.postnorm_next_row, 8u);
    __syncthreads();
    const uint32_t row = *claimed;
    if (row >= static_cast<uint32_t>(p.gemm.m)) break;
#if FUSE_ENABLE_PROFILING
    uint64_t wait_begin = 0, work_begin = 0;
    if constexpr (Instrumented) if (thread == 0) wait_begin = detail::read_global_timer();
#endif
    // A row needs ALL N panels, but their acquire loads need not serialize
    // through one lane. Stripe the same flags across one warp; the following
    // CTA barrier joins every observation before any row data is consumed.
    // Already-ready rows thus avoid N serial memory round trips. Readiness
    // granularity, flag count and producer publication protocol do not change.
    if (thread < 32) {
      for (int n = thread; n < n_tiles; n += 32)
        detail::wait_acquire_gpu_single_lane(a.workspace.output_ready +
            (int64_t(row / 128) * n_tiles + n) * kReadyFlagStride, 1);
    }
    __syncthreads();
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) if (thread == 0) {
      work_begin = detail::read_global_timer();
      wait_ns += work_begin - wait_begin;
    }
#endif
    // Keep four concurrent rows when their two inputs fit reclaimed storage.
    // Otherwise two rows can recover asynchronous input staging without a
    // larger SMEM allocation. Both cohorts emulate the SAME virtual256 tree;
    // the claimed eight rows and all-N readiness observations are unchanged.
    auto process = [&](auto row_count) {
      constexpr int kRows = decltype(row_count)::value;
      for (int i = 0; i < 8; i += kRows) {
#if FUSE_ENABLE_PROFILING
        if constexpr (Instrumented)
          residual_rmsnorm_rows<kRows,true,StagingBytes>(p.output,p.gemm.n,row+i,a.postprocess,row_cache,partial,phases);
        else
#endif
        residual_rmsnorm_rows<kRows,false,StagingBytes>(p.output,p.gemm.n,row+i,
                                                     a.postprocess,row_cache,partial);
      }
    };
    const int64_t input_bytes_per_row = int64_t(2)*p.gemm.n*sizeof(Bf16);
    if (4*input_bytes_per_row > StagingBytes &&
        2*input_bytes_per_row <= StagingBytes)
      process(cute::Int<2>{});
    else
      process(cute::Int<4>{});
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) if (thread == 0) {
      work_ns += detail::read_global_timer() - work_begin;
      rows += 8;
    }
#endif
  }
#if FUSE_ENABLE_PROFILING
  if constexpr (Instrumented) if (thread == 0) {
    auto& record = timeline[blockIdx.x];
    record.norm_worker_begin = begin;
    record.norm_worker_end = detail::read_global_timer();
    record.norm_wait_ns = wait_ns;
    record.norm_work_ns = work_ns;
    record.norm_rows = rows;
    record.norm_threads = Threads;
    record.norm_load_ns = phases[0];
    record.norm_reduce_ns = phases[1];
    record.norm_store_ns = phases[2];
  }
#endif
}

// Conservative initial boundary: GEMM's N tiles may complete in any order.
// Wait for the resident grid, then reuse all CTAs for complete-hidden RMSNorm.
// This is a single launch, but the norm tail is NOT claimed to overlap GEMM.
// A later row-ready epilogue must publish after *all* N tiles have drained;
// waiting for a single N256 tile would silently normalize incomplete rows.
// The optional row-ready schedule starts after each CTA's own role drains;
// this fallback remains a separately measured, non-overlapped boundary.
template <class Base, bool Instrumented = false, bool Overlap = false>
struct OprojPostprocessKernel : Base {
  CUTLASS_DEVICE void operator()(const typename Base::Params& params, char* smem) {
    Base::operator()(params, smem);
    const auto& post = params.comm.postprocess;
    if (!post.enabled()) return;
    if constexpr (Overlap) {
      // A communication CTA joins all A/W warps; a compute CTA joins every
      // CUTLASS warp. This is LOCAL, not a grid barrier. Neither borrows SMEM
      // while a former user is still active. The same 256-lane norm tree and
      // resource footprint serve all CTAs, without half-CTA named barriers.
      __syncthreads();
      constexpr int kStagingBytes = Base::SharedStorageSize - kPostnormScratchFloats*sizeof(float);
      auto* partial = reinterpret_cast<float*>(smem + kStagingBytes);
#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented)
        consume_postnorm_rows<kStagingBytes,true>(
            params.comm, partial, reinterpret_cast<Bf16*>(smem), params.timeline);
      else
#endif
        consume_postnorm_rows<kStagingBytes>(
            params.comm, partial, reinterpret_cast<Bf16*>(smem));
      return;
    }
    cooperative_groups::this_grid().sync();
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) {
      if (threadIdx.x == 0 && blockIdx.x < params.timeline_capacity)
        params.timeline[blockIdx.x].postnorm_ready = detail::read_global_timer();
    }
#endif
    const auto& p = params.comm.params;
    auto* added_row = reinterpret_cast<Bf16*>(smem);
    auto* partial = reinterpret_cast<float*>(smem + Base::SharedStorageSize - 32 * sizeof(float));
    // The grid join makes original GEMM/TMA storage reusable. Cache four full
    // rows per CTA;32 virtual warp partials occupy the final128 bytes.
#if FUSE_ENABLE_PROFILING
    uint64_t phases[3]{};
#endif
    for (int64_t row = int64_t(blockIdx.x)*4; row < p.gemm.m; row += gridDim.x*4) {
#if FUSE_ENABLE_PROFILING
      if constexpr (Instrumented)
        residual_rmsnorm_rows<4,true>(p.output,p.gemm.n,row,post,added_row,partial,phases);
      else
#endif
      residual_rmsnorm_rows<4>(p.output, p.gemm.n, row, post, added_row, partial);
    }
#if FUSE_ENABLE_PROFILING
    if constexpr (Instrumented) {
      // Diagnostic-only join: the end stamp represents every norm warp, not
      // only the rows assigned to warp zero. Production has no extra barrier.
      __syncthreads();
      if (threadIdx.x == 0 && blockIdx.x < params.timeline_capacity) {
        auto& record = params.timeline[blockIdx.x];
        record.postnorm_end = detail::read_global_timer();
        record.norm_load_ns = phases[0];
        record.norm_reduce_ns = phases[1];
        record.norm_store_ns = phases[2];
      }
    }
#endif
  }
};

}  // namespace
}  // namespace fuse
