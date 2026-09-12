// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include "fuse/layout/gemm.h"
#include "producer_consumer.cuh"
#include <cooperative_groups.h>
#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/float_subbyte.h>
#include <cutlass/numeric_conversion.h>
#include <cmath>
#include <cstring>
#include <cuda/atomic>

namespace fuse {
namespace {

using Mxfp8ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<32>;

struct Mxfp8Workspace {
  Fp8E4m3* b = nullptr;
  cutlass::float_ue8m0_t* sfb = nullptr;
  uint32_t* arrivals = nullptr;
  uint32_t* ready = nullptr;
  size_t b_bytes = 0, sfb_bytes = 0, bytes = 0;
  int panels = 0;

  static size_t align(size_t n) { return (n + 255) & ~size_t{255}; }
  static Mxfp8Workspace make(const GemmProblem& p, void* memory = nullptr) {
    Mxfp8Workspace w;
    const auto shape = cute::make_shape(p.m, p.n, p.k, 1);
    w.b_bytes = align(size_t{static_cast<uint32_t>(p.n)} * p.k);
    w.sfb_bytes = align(cute::size(cute::filter_zeros(Mxfp8ScaleConfig::tile_atom_to_shape_SFB(shape))));
    w.panels = ceil_div(p.n, 256);
    const size_t flags = align(size_t(w.panels) * kReadyFlagStride * sizeof(uint32_t));
    w.bytes = w.b_bytes + w.sfb_bytes + 2 * flags;
    if (memory) {
      auto* base = static_cast<unsigned char*>(memory);
      w.b = reinterpret_cast<Fp8E4m3*>(base);
      w.sfb = reinterpret_cast<cutlass::float_ue8m0_t*>(base + w.b_bytes);
      w.arrivals = reinterpret_cast<uint32_t*>(base + w.b_bytes + w.sfb_bytes);
      w.ready = reinterpret_cast<uint32_t*>(base + w.b_bytes + w.sfb_bytes + flags);
    }
    return w;
  }
};

struct Mxfp8A2AWorkspace {
  Mxfp8Workspace weights{};
  Fp8E4m3* a = nullptr;
  cutlass::float_ue8m0_t* sfa = nullptr;
  uint32_t* ready = nullptr;
  size_t bytes = 0;

  static Mxfp8A2AWorkspace make(const GemmProblem& p, void* memory = nullptr) {
    Mxfp8A2AWorkspace w;
    w.weights = Mxfp8Workspace::make(p, memory);
    const auto shape = cute::make_shape(p.m, p.n, p.k, 1);
    const size_t data_bytes = Mxfp8Workspace::align(size_t(p.m) * p.k);
    const size_t scale_bytes = Mxfp8Workspace::align(cute::size(cute::filter_zeros(
        Mxfp8ScaleConfig::tile_atom_to_shape_SFA(shape))));
    // The shape-only query reserves all supported peers. Only the actual
    // [M tile, world] prefix is used/reset; CP never changes allocation rules.
    const size_t flags = Mxfp8Workspace::align(size_t(ceil_div(p.m, 128)) *
        kMaxWorldSize * kReadyFlagStride * sizeof(uint32_t));
    w.bytes = w.weights.bytes + data_bytes + scale_bytes + flags;
    if (memory) {
      auto* base = static_cast<unsigned char*>(memory) + w.weights.bytes;
      w.a = reinterpret_cast<Fp8E4m3*>(base);
      w.sfa = reinterpret_cast<cutlass::float_ue8m0_t*>(base + data_bytes);
      w.ready = reinterpret_cast<uint32_t*>(base + data_bytes + scale_bytes);
    }
    return w;
  }
};

CUTLASS_HOST_DEVICE int mxfp8_scale_exponent(float amax) {
  // 448 = 1.75 * 2^8. Carry into the exponent exactly when the mantissa
  // exceeds 1.75; unlike log/exp helpers this needs only integer operations.
  // Preserve zero -> scale 1 and the original -127 floor. BF16 subnormals
  // all fit at that floor, so no normalization or changed epsilon is needed.
  if (!(amax > 0)) return 0;
  uint32_t bits;
#if defined(__CUDA_ARCH__)
  bits = __float_as_uint(amax);
#else
  std::memcpy(&bits, &amax, sizeof(bits));
#endif
  const int exponent = static_cast<int>((bits + 0x001fffffu) >> 23) - 127 - 8;
  return exponent < -127 ? -127 : exponent > 127 ? 127 : exponent;
}

CUTLASS_HOST_DEVICE float mxfp8_scaled_value(float value, int exponent) {
  // Exact power of two: ordinary FP32 multiplication retains the original
  // ldexp rounding, including signed zero/subnormals (no --use_fast_math).
  const uint32_t bits = exponent == 127 ? 0x00400000u : uint32_t(127 - exponent) << 23;
  float inverse;
#if defined(__CUDA_ARCH__)
  inverse = __uint_as_float(bits);
#else
  std::memcpy(&inverse, &bits, sizeof(inverse));
#endif
  return value * inverse;
}

// One warp owns one 32-element K block: reduce amax, choose a power-of-two
// scale, then round to E4M3. Write native CUTLASS scale layout directly;
// no intermediate transpose/scale-packing kernel. Sources remain BF16.
template <class ScaleLayout>
CUTLASS_DEVICE void quantize_mxfp8_group(const Bf16* input, Fp8E4m3* output,
    cutlass::float_ue8m0_t* scales, int rows, int k, int64_t row_stride,
    const ScaleLayout& scale_layout, int64_t group) {
  const int lane = threadIdx.x % 32;
  const int row = group / (k / 32);
  const int column = (group % (k / 32)) * 32 + lane;
  const float value = row < rows ? float(input[int64_t{row} * row_stride + column]) : 0.0f;
  float amax = fabsf(value);
  for (int mask = 16; mask; mask >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
  const int exponent = mxfp8_scale_exponent(amax);
  if (row < rows)
    output[int64_t{row} * k + column] = Fp8E4m3(mxfp8_scaled_value(value, exponent));
  if (lane == 0) {
    const auto offset = scale_layout(cute::make_coord(row, column, 0));
    reinterpret_cast<uint8_t*>(scales)[offset] = static_cast<uint8_t>(exponent + 127);
  }
}

// Register tiling inspired by ThunderKittens' mxfp8_b200 quantizer:
// https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/gemm/mxfp8_b200/mxfp8_b200_gemm.cu
// TK stages a whole tile through SMEM and reduces each K32 group in one thread.
// A communication warp must progress independently, so load register vectors
// directly, with no communication SMEM slot or extra CTA barrier:
//
//   lane:       0        1       ...       31
//   K32 group: group0   group1             group31
//   each lane: load 32 BF16 -> local amax -> one scale -> store 32 FP8
//
// All 1024 values are loaded before conversion. Four independent local max
// chains expose instruction-level parallelism; no warp shuffle or replicated
// scale calculation is needed. Each lane's K32 group stays within one row.
// K32 scales, 32-group chunks and complete N256 x K publication stay unchanged.
// Rounded-up N is a multiple of 128 and K is a multiple of 128: every chunk is
// full, including the padded tail. Invalid rows produce scales but no FP8 store.
template <class ScaleLayout>
CUTLASS_DEVICE void quantize_mxfp8_chunk(const Bf16* input, Fp8E4m3* output,
    cutlass::float_ue8m0_t* scales, int rows, int k, int64_t row_stride,
    const ScaleLayout& scale_layout, int row, int column_begin) {
  const int lane = threadIdx.x % 32;
  using InputVector = cutlass::AlignedArray<Bf16, 32, 16>;
  InputVector packed_input;
  packed_input.clear();
  const int64_t linear_column = int64_t{column_begin} + lane * 32;
  int column = static_cast<int>(linear_column);
  // The common in-row case needs no division. Preserve short/non-power-of-two
  // K and chunks crossing multiple rows; do not specialize on model shapes.
  if (linear_column >= k) {
    row += static_cast<int>(linear_column / k);
    column = static_cast<int>(linear_column % k);
  }
  if (row < rows) {
    const Bf16* source = input + int64_t{row} * row_stride + column;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // Blackwell supports 256-bit global operations. Each lane owns 64 input
    // bytes: two full-sector loads replace four half-sector loads. This reduces
    // instructions, not necessarily L2/HBM traffic; retain the public stride's
    // unaligned fallback instead of assuming every row is 32-byte aligned.
    if (reinterpret_cast<uintptr_t>(source) % 32 == 0) {
      uint32_t words[16];
      CUTLASS_PRAGMA_UNROLL
      for (int part = 0; part < 2; ++part) {
        auto* r = words + part * 8;
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
              "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
            : "l"(source + part * 16) : "memory");
      }
      memcpy(&packed_input, words, sizeof(words));
    } else
#endif
    if (reinterpret_cast<uintptr_t>(source) % 16 == 0) {
      packed_input = *reinterpret_cast<const InputVector*>(source);
    } else {
      // The public BF16 row stride need not be vector aligned.
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < 32; ++i) packed_input[i] = source[i];
    }
  }
  cutlass::Array<float, 32> values;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  // Reduce packed BF16 magnitudes before expanding the values to FP32. Four
  // independent chains shorten the scale dependency path. xorsign.abs compares
  // magnitudes but XORs signs: remove those signs before the final horizontal
  // max. Keep the original signed inputs for FP32 scaling and FP8 conversion.
  // No FTZ or BF16 multiply: finite subnormals, signed zero and the existing
  // scale exponent floor retain the same numerical contract.
  uint32_t words[16], maxima[4];
  memcpy(words, &packed_input, sizeof(words));
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 4; ++i) maxima[i] = words[i];
  CUTLASS_PRAGMA_UNROLL
  for (int i = 4; i < 16; ++i)
    asm("max.xorsign.abs.bf16x2 %0, %1, %2;"
        : "=r"(maxima[i % 4]) : "r"(maxima[i % 4]), "r"(words[i]));
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;"
      : "=r"(maxima[0]) : "r"(maxima[0]), "r"(maxima[1]));
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;"
      : "=r"(maxima[2]) : "r"(maxima[2]), "r"(maxima[3]));
  asm("max.xorsign.abs.bf16x2 %0, %1, %2;"
      : "=r"(maxima[0]) : "r"(maxima[0]), "r"(maxima[2]));
  const uint32_t magnitudes = maxima[0] & 0x7fff7fffu;
  const uint32_t lower = magnitudes & 0xffffu, upper = magnitudes >> 16;
  const float amax = __uint_as_float((lower > upper ? lower : upper) << 16);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 32; ++i) values[i] = float(packed_input[i]);
#else
  cutlass::Array<float, 4> maxima;
  maxima.clear();
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 32; ++i) {
    values[i] = float(packed_input[i]);
    maxima[i % 4] = fmaxf(maxima[i % 4], fabsf(values[i]));
  }
  const float amax = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
#endif
  const int exponent = mxfp8_scale_exponent(amax);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  // f32x2 pairs independent FP32 multiplies; it does not change the arithmetic
  // to BF16. Explicit RN and no FTZ preserve the original power-of-two scaling.
  const uint32_t inverse_bits = exponent == 127 ? 0x00400000u : uint32_t(127 - exponent) << 23;
  const uint64_t inverse_pair = uint64_t{inverse_bits} | (uint64_t{inverse_bits} << 32);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 32; i += 2) {
    uint64_t input_pair, output_pair;
    memcpy(&input_pair, &values[i], sizeof(input_pair));
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(output_pair) : "l"(input_pair), "l"(inverse_pair));
    memcpy(&values[i], &output_pair, sizeof(output_pair));
  }
#else
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 32; ++i) values[i] = mxfp8_scaled_value(values[i], exponent);
#endif
  if (row < rows) {
    const auto converted = cutlass::NumericArrayConverter<Fp8E4m3, float, 32>{}(values);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    auto* destination = output + int64_t{row} * k + column;
    if (reinterpret_cast<uintptr_t>(destination) % 32 == 0) {
      uint32_t r[8];
      memcpy(r, &converted, sizeof(r));
      asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
          :: "l"(destination), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]),
             "r"(r[4]), "r"(r[5]), "r"(r[6]), "r"(r[7]) : "memory");
    } else {
#endif
    cutlass::AlignedArray<uint64_t, 4, 16> packed_output;
    memcpy(&packed_output, &converted, sizeof(packed_output));
    *reinterpret_cast<decltype(packed_output)*>(output + int64_t{row} * k + column) = packed_output;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    }
#endif
  }
  const auto offset = scale_layout(cute::make_coord(row, column, 0));
  reinterpret_cast<uint8_t*>(scales)[offset] = static_cast<uint8_t>(exponent + 127);
}

template <class ScaleLayout>
__global__ void quantize_mxfp8_operand(const Bf16* input, Fp8E4m3* output,
    cutlass::float_ue8m0_t* scales, int rows, int k, int64_t row_stride,
    ScaleLayout scale_layout) {
  const int64_t groups = ((int64_t{rows} + 127) / 128 * 128) * (k / 32);
  for (int64_t group = (int64_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
       group < groups; group += int64_t{gridDim.x} * blockDim.x / 32) {
    quantize_mxfp8_group(input, output, scales, rows, k, row_stride, scale_layout, group);
  }
}

// A bounded SIMT side task for a communication warp. Quantization groups,
// scheduling chunks and GEMM readiness are deliberately different units:
//   32 K values -> one scale; 32 groups -> one progress step;
//   full N256 x K panel -> one ready flag reused by all M tiles.
// Each warp strides one common first-use ordered queue: no duplicated weights,
// global work-claim atomics or per-K GEMM waits. A warp accumulates completed
// chunks locally, then contributes them together at its LAST valid chunk in
// that panel. Scale padding is produced as well. The publication chain is:
//
//   progress: chunk -> local count -> return to route / next progress
//             ... last owned chunk of this panel ...
//   ALL owned W/scale stores -> full-warp sync -> leader acq_rel add(count)
//       -> final contribution's ready.release -> GEMM ready.acquire
//       -> consumer warp sync / async-proxy fence -> TMA reads.
//
// Each progress still quantizes only 1024 values, so communication can resume
// between chunks. Only the internal arrival accounting is aggregated; GEMM
// still waits for the SAME complete N256 x K panel. Flush in the last valid
// progress itself, never on the next call: the caller may next wait for output
// that depends on this panel. Use actual tail rows, not padded scheduler slots.
// Logical panel order is monotonic; N-band swizzle only permutes whole panels.
// Thus no worker revisits a panel or carries a count across a padding gap.
//
// The final warp barrier orders every lane's generic-global stores from ALL
// its owned chunks before its leader releases their count. Each device-scope
// acq_rel RMW acquires preceding contributions and passes their writes on, so
// the final leader publishes the whole panel, including every warp's stores.
// This cumulative handoff needs no separate per-lane SC fence. Keep full-warp
// participation and device-scope atomics: a relaxed counter alone is not this
// protocol. The consumer's async-proxy fence and peer-TMA completion waits have
// separate responsibilities and remain required. Scratch is invocation-private;
// initialize_grid resets the counters before any producer/consumer starts.
struct Mxfp8WeightProducer {
  static constexpr bool kEnabled = true;
  static constexpr int kPanelN = 256, kGroupsPerStep = 32;
  using ScaleLayout = decltype(Mxfp8ScaleConfig::tile_atom_to_shape_SFB(
      cute::make_shape(int{}, int{}, int{}, 1)));
  struct Arguments {
    const Bf16* source = nullptr;
    Mxfp8Workspace workspace{};
    ScaleLayout scales{};
    int n = 0, k = 0;
    int64_t row_stride = 0;
    uint32_t epoch = 0;
    bool all_ctas = false;
    bool warp_specialized = false;
#if FUSE_ENABLE_PROFILING
    Mxfp8ProfileView probe{};
#endif
  };
  Arguments args;
  detail::NBandSwizzle order{};
  int64_t next = 0, stride = 1;
  uint32_t pending_chunks = 0;

  template <class CommParams>
  CUTLASS_DEVICE static void initialize_grid(const CommParams& p) {
    if (!p.weights.source) return;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int n = index; n < p.weights.workspace.panels; n += gridDim.x * blockDim.x) {
      p.weights.workspace.arrivals[n * kReadyFlagStride] = 0;
      p.weights.workspace.ready[n * kReadyFlagStride] = 0;
    }
    // Reset every invocation, including Graph replay with the same epoch.
    // There are no host memset nodes and no reliance on zero-filled scratch.
    cooperative_groups::this_grid().sync();
    if (p.weights.all_ctas) {
      Mxfp8WeightProducer work(p.weights, p.producer_order, p.n_band_swizzle,
          index / 32, gridDim.x * blockDim.x / 32);
      work.drain();
      cooperative_groups::this_grid().sync();
    }
  }

  template <class Schedule>
  CUTLASS_DEVICE Mxfp8WeightProducer(const Arguments& a, const Schedule& schedule,
      detail::NBandSwizzle swizzle, int worker, int workers)
      : args(a), order(swizzle), next(worker), stride(workers) {
    // For both AlongM and AlongN the first use of N panels is increasing N
    // (also within OProj's bounded M-window / N-group traversal). W panels are
    // produced once, not again when GEMM advances to the next M window.
    // before optional N-band rotation. Use the scheduler's padded N extent;
    // invalid rotated panels are skipped, never assigned a physical flag.
    const bool along_n = schedule.raster_order_ == Schedule::RasterOrder::AlongN;
    order.extent = static_cast<int>(along_n ? schedule.divmod_cluster_blk_major_.divisor
        : schedule.divmod_batch_.divisor / schedule.divmod_cluster_blk_major_.divisor);
  }

  CUTLASS_DEVICE bool progress() {
    if (!args.source) return false; // Explicit prequantized diagnostic.
    const int64_t panel_groups = int64_t{kPanelN} * (args.k / 32);
    const int64_t steps = (panel_groups + kGroupsPerStep - 1) / kGroupsPerStep;
    if (next >= int64_t{order.extent} * steps) return false;
    const int panel = order.forward(static_cast<int>(next / steps));
    const int64_t step = next % steps;
    next += stride;
    if (panel >= args.workspace.panels) return true;
    const int padded_n = ceil_div(args.n, 128) * 128;
    const int remaining_rows = padded_n - panel * kPanelN;
    const int rows = remaining_rows < kPanelN ? remaining_rows : kPanelN;
    const int64_t groups = int64_t{rows} * (args.k / 32);
    const int64_t begin = step * kGroupsPerStep;
    if (begin >= groups) return true;
#if FUSE_ENABLE_PROFILING
    Mxfp8QuantRecord* record = nullptr;
    const int64_t record_index = int64_t{panel} * steps + step;
    if (args.probe.quant && record_index < args.probe.quant_capacity && threadIdx.x % 32 == 0) {
      record = args.probe.quant + record_index;
      record->cta = blockIdx.x;
      record->warp = threadIdx.x / 32;
      record->panel = panel;
      record->groups = kGroupsPerStep;
      record->begin = detail::read_global_timer();
    }
#endif
    // Decode one chunk origin; the helper advances lane-local vector addresses.
    // K is a multiple of 128, not necessarily a power of two. A warp chunk can
    // cross rows, but each lane owns a complete K32 group:
    //
    //   K=384: row r   [lanes 0..11]
    //          row r+1 [lanes 12..23]   row r+2 [lanes 24..31] ...
    //
    // The lane's 32 values remain adjacent. Removing repeated group divmod
    // changes neither K32 scales nor the chunk/panel publication boundaries.
    const int groups_per_row = args.k / 32;
    const int row = panel * kPanelN + static_cast<int>(begin / groups_per_row);
    const int column = static_cast<int>(begin % groups_per_row) * 32;
    static_assert(kGroupsPerStep == 32);
    quantize_mxfp8_chunk(args.source, args.workspace.b, args.workspace.sfb,
        args.n, args.k, args.row_stride, args.scales, row, column);
#if FUSE_ENABLE_PROFILING
    if (record) record->quant_done = detail::read_global_timer();
    // Keep intermediate stamps in registers until the publication sequence is
    // over: do not insert diagnostic writes between the join and atomic and
    // attribute those writes to the next operation. These are lane-0
    // timestamps; the joins also include waiting for the other 31 lanes.
    uint64_t warp_join_done = 0, arrival_done = 0;
    uint32_t arrival_chunks = 0;
#endif
    const uint32_t expected = (groups + kGroupsPerStep - 1) / kGroupsPerStep;
    ++pending_chunks;
    if (step + stride >= expected) {
      const uint32_t delta = pending_chunks;
      pending_chunks = 0;
      __syncwarp();
#if FUSE_ENABLE_PROFILING
      if (record) {
        asm volatile("" ::: "memory");
        warp_join_done = detail::read_global_timer();
        asm volatile("" ::: "memory");
      }
      arrival_chunks = delta;
#endif
      if (threadIdx.x % 32 == 0) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_device> count(
            args.workspace.arrivals[panel * kReadyFlagStride]);
        const uint32_t arrived = count.fetch_add(delta, cuda::memory_order_acq_rel) + delta;
#if FUSE_ENABLE_PROFILING
        if (record) {
          asm volatile("" ::: "memory");
          arrival_done = detail::read_global_timer();
          asm volatile("" ::: "memory");
        }
#endif
        if (arrived == expected) {
          detail::store_release_gpu(args.workspace.ready + panel * kReadyFlagStride, args.epoch);
#if FUSE_ENABLE_PROFILING
          if (record) record->release = detail::read_global_timer();
#endif
        }
      }
      __syncwarp();
    }
#if FUSE_ENABLE_PROFILING
    if (record) {
      record->end = detail::read_global_timer();
      record->warp_join_done = warp_join_done;
      record->arrival_done = arrival_done;
      record->arrival_chunks = arrival_chunks;
    }
#endif
    return true;
  }
  CUTLASS_DEVICE void drain() { while (progress()) {} }
};

}  // namespace
}  // namespace fuse
