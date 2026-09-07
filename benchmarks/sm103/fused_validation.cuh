// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define FUSED_VALIDATION_HD __host__ __device__
#else
#define FUSED_VALIDATION_HD
#endif

namespace fused_validation {

constexpr int kThreads = 256;
constexpr int kBlocks = 256;
constexpr int kMaxRanks = 8;

FUSED_VALIDATION_HD inline float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
#endif
}

struct Stats {
  uint64_t checked;
  uint64_t mismatches;
  uint64_t nonfinite;
  double error_square;
  double reference_square;
  float max_abs;
  uint64_t first_index;
  uint16_t first_actual;
  uint16_t first_expected;

  FUSED_VALIDATION_HD static Stats zero() {
    return {0, 0, 0, 0, 0, 0, std::numeric_limits<uint64_t>::max(), 0, 0};
  }

  FUSED_VALIDATION_HD void fail(uint16_t actual, uint16_t expected, uint64_t index) {
    ++mismatches;
    if (index < first_index) {
      first_index = index;
      first_actual = actual;
      first_expected = expected;
    }
  }

  FUSED_VALIDATION_HD void observe_numeric(uint16_t actual, uint16_t expected, uint64_t index) {
    ++checked;
    const float a = bf16_to_float(actual), b = bf16_to_float(expected);
    const float error = fabsf(a - b);
    if ((actual & 0x7f80u) == 0x7f80u || (expected & 0x7f80u) == 0x7f80u ||
        !(error <= std::numeric_limits<float>::max())) {
      ++nonfinite;
      max_abs = std::numeric_limits<float>::infinity();
      fail(actual, expected, index);
      return;
    }
    if (error > max_abs) max_abs = error;
    error_square += static_cast<double>(error) * error;
    reference_square += static_cast<double>(b) * b;
#if defined(__CUDA_ARCH__)
    const float tolerance = __fadd_rn(0.01f, __fmul_rn(0.01f, fabsf(b)));
#else
    const float relative = 0.01f * fabsf(b);
    const float tolerance = 0.01f + relative;
#endif
    if (error > tolerance) fail(actual, expected, index);
  }

  FUSED_VALIDATION_HD void observe_bits(uint16_t actual, uint16_t expected, uint64_t index) {
    ++checked;
    const bool invalid = (actual & 0x7f80u) == 0x7f80u || (expected & 0x7f80u) == 0x7f80u;
    nonfinite += invalid;
    if (actual != expected || invalid) fail(actual, expected, index);
  }

  FUSED_VALIDATION_HD void merge(const Stats& other) {
    checked += other.checked;
    mismatches += other.mismatches;
    nonfinite += other.nonfinite;
    error_square += other.error_square;
    reference_square += other.reference_square;
    if (other.max_abs > max_abs) max_abs = other.max_abs;
    if (other.first_index < first_index) {
      first_index = other.first_index;
      first_actual = other.first_actual;
      first_expected = other.first_expected;
    }
  }
};

struct SourceAddress { int rank; uint64_t offset; };

// Ordinary packed QKV is rank-major, including causal CP. This independent
// element mapping does not share the operator's tiled communication traversal.
struct QkvOracle {
  const uint16_t* source[kMaxRanks];
  uint64_t seq_local;
  uint64_t q_width;
  uint64_t kv_width;
  int world;
  int destination;

  FUSED_VALIDATION_HD SourceAddress address(uint64_t index) const {
    const uint64_t rows = seq_local * world;
    const uint64_t local_q = q_width / world, local_kv = kv_width / world;
    const uint64_t q_elements = rows * local_q, kv_elements = rows * local_kv;
    uint64_t local_width = local_q, feature_base = 0;
    if (index >= q_elements) {
      index -= q_elements;
      local_width = local_kv;
      feature_base = q_width;
      if (index >= kv_elements) {
        index -= kv_elements;
        feature_base += kv_width;
      }
    }
    const uint64_t row = index / local_width;
    return {static_cast<int>(row / seq_local),
            (row % seq_local) * (q_width + 2 * kv_width) +
                feature_base + destination * local_width + index % local_width};
  }

  FUSED_VALIDATION_HD uint16_t operator()(uint64_t index) const {
    const auto location = address(index);
    return source[location.rank][location.offset];
  }
};

struct DenseOracle {
  const uint16_t* expected;
  FUSED_VALIDATION_HD uint16_t operator()(uint64_t index) const { return expected[index]; }
};

// One shape-independent allocation per rank. Both checks reuse the partials
// in stream order; only these two final records are copied back to the host.
struct Scratch {
  Stats partials[kBlocks];
  Stats result[2];
};

#if defined(__CUDACC__)
__device__ inline void reduce_block(Stats value, Stats* output) {
  __shared__ Stats values[kThreads];
  values[threadIdx.x] = value;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) values[threadIdx.x].merge(values[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = values[0];
}

template <bool Numeric, class Oracle>
__global__ void compare_kernel(const uint16_t* actual, Oracle oracle, uint64_t elements, Stats* partials) {
  Stats value = Stats::zero();
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    if constexpr (Numeric) value.observe_numeric(actual[index], oracle(index), index);
    else value.observe_bits(actual[index], oracle(index), index);
  }
  reduce_block(value, partials + blockIdx.x);
}

static __global__ void reduce_kernel(const Stats* partials, int blocks, Stats* output) {
  reduce_block(threadIdx.x < blocks ? partials[threadIdx.x] : Stats::zero(), output);
}

template <bool Numeric, class Oracle>
cudaError_t launch(const uint16_t* actual, Oracle oracle, uint64_t elements,
                    Scratch* scratch, int result_index, cudaStream_t stream) {
  if (!actual || !scratch || elements == 0 || result_index < 0 || result_index >= 2) {
    return cudaErrorInvalidValue;
  }
  const uint64_t needed = elements / kThreads + (elements % kThreads != 0);
  const int blocks = static_cast<int>(needed < kBlocks ? needed : kBlocks);
  compare_kernel<Numeric><<<blocks, kThreads, 0, stream>>>(actual, oracle, elements, scratch->partials);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  reduce_kernel<<<1, kThreads, 0, stream>>>(scratch->partials, blocks, scratch->result + result_index);
  return cudaGetLastError();
}
#endif

}  // namespace fused_validation

#undef FUSED_VALIDATION_HD
