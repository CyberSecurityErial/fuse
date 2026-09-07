// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define FUSED_INPUTS_HD __host__ __device__
#else
#define FUSED_INPUTS_HD
#endif

namespace fused_inputs {

constexpr int kThreads = 256;
constexpr int kBlocks = 256;
constexpr int kMaxRanks = 8;
static_assert(kBlocks == kThreads, "the final reduction uses one thread per partial");

FUSED_INPUTS_HD inline float uniform_value(uint32_t bits, float magnitude) {
  const float unit = static_cast<float>(bits >> 8) / 16777216.0f;
#if defined(__CUDA_ARCH__)
  return __fmul_rn(__fadd_rn(unit, -0.5f), 2 * magnitude);
#else
  return (unit - 0.5f) * (2 * magnitude);
#endif
}

struct Stats {
  uint64_t count;
  uint64_t finite;
  uint64_t nonzero;
  double sum;
  double square_sum;
  float minimum;
  float maximum;

  FUSED_INPUTS_HD static Stats zero() {
    return {0, 0, 0, 0, 0, std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
  }

  FUSED_INPUTS_HD void observe(float value) {
    ++count;
    if (!(value <= std::numeric_limits<float>::max() &&
          value >= -std::numeric_limits<float>::max())) return;
    ++finite;
    nonzero += value != 0;
    sum += value;
    square_sum += static_cast<double>(value) * value;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }

  FUSED_INPUTS_HD void merge(const Stats& other) {
    count += other.count;
    finite += other.finite;
    nonzero += other.nonzero;
    sum += other.sum;
    square_sum += other.square_sum;
    if (other.minimum < minimum) minimum = other.minimum;
    if (other.maximum > maximum) maximum = other.maximum;
  }
};

struct Scratch {
  Stats partials[kBlocks];
  Stats result;
};

struct SourceAddress { int rank; uint64_t offset; };

// Independent scalar reference: each output row owns one sequence position,
// while contiguous feature shards come from distinct source ranks. This does
// not call the operator's tile traversal, ready protocol, or route helpers.
struct OprojGather {
  const uint16_t* source[kMaxRanks];
  uint64_t seq_local;
  uint64_t q_width;
  int world;
  int destination;
  bool causal;

  FUSED_INPUTS_HD SourceAddress address(uint64_t index) const {
    const uint64_t row = index / q_width, feature = index % q_width;
    const uint64_t shard_width = q_width / world;
    uint64_t global_row = destination * seq_local + row;
    if (causal) {
      const uint64_t half = seq_local / 2;
      const uint64_t chunk = row < half ? destination : 2 * world - 1 - destination;
      global_row = chunk * half + row % half;
    }
    return {static_cast<int>(feature / shard_width),
            global_row * shard_width + feature % shard_width};
  }

  FUSED_INPUTS_HD uint16_t operator()(uint64_t index) const {
    const auto location = address(index);
    return source[location.rank][location.offset];
  }
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

// Mapping v1: subsequence=linear_thread, offset=0; curand4 round r writes
// element 4*(r*65536+linear_thread)+component. Fixed launch geometry makes a
// seed's prefix independent of tensor length and gives identical weights on
// every rank. This is intentionally not the CPU mt19937 bit sequence.
static __global__ void generate_kernel(uint16_t* output, uint64_t count, uint32_t seed,
                                       float magnitude, Stats* partials) {
  const uint64_t thread = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t random;
  curand_init(seed, thread, 0, &random);
  Stats stats = Stats::zero();
  for (uint64_t base = 4 * thread; base < count; base += 4ull * kBlocks * kThreads) {
    const uint4 values = curand4(&random);
    const uint32_t words[4] = {values.x, values.y, values.z, values.w};
#pragma unroll
    for (int component = 0; component < 4; ++component) {
      if (base + component < count) {
        const __nv_bfloat16 value = __float2bfloat16_rn(uniform_value(words[component], magnitude));
        output[base + component] = __bfloat16_as_ushort(value);
        stats.observe(__bfloat162float(value));
      }
    }
  }
  reduce_block(stats, partials + blockIdx.x);
}

static __global__ void reduce_kernel(const Stats* partials, Stats* output) {
  reduce_block(partials[threadIdx.x], output);
}

inline cudaError_t generate(uint16_t* output, uint64_t count, uint32_t seed,
                            float magnitude, Scratch* scratch, cudaStream_t stream) {
  if (!output || !scratch || count == 0 || !(magnitude > 0 && magnitude <= 1)) {
    return cudaErrorInvalidValue;
  }
  generate_kernel<<<kBlocks, kThreads, 0, stream>>>(output, count, seed, magnitude, scratch->partials);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  reduce_kernel<<<1, kThreads, 0, stream>>>(scratch->partials, &scratch->result);
  return cudaGetLastError();
}

static __global__ void gather_kernel(uint16_t* output, uint64_t count, OprojGather mapping) {
  for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count; index += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    output[index] = mapping(index);
  }
}

inline cudaError_t gather(uint16_t* output, OprojGather mapping, cudaStream_t stream) {
  if (!output || mapping.world <= 0 || mapping.world > kMaxRanks ||
      mapping.destination < 0 || mapping.destination >= mapping.world ||
      mapping.seq_local == 0 || mapping.q_width == 0 || mapping.q_width % mapping.world != 0 ||
      (mapping.causal && mapping.seq_local % 2 != 0) ||
      mapping.seq_local > std::numeric_limits<uint64_t>::max() / mapping.q_width) {
    return cudaErrorInvalidValue;
  }
  for (int rank = 0; rank < mapping.world; ++rank) {
    if (!mapping.source[rank]) return cudaErrorInvalidValue;
  }
  gather_kernel<<<kBlocks, kThreads, 0, stream>>>(output, mapping.seq_local * mapping.q_width, mapping);
  return cudaGetLastError();
}
#endif

}  // namespace fused_inputs

#undef FUSED_INPUTS_HD
