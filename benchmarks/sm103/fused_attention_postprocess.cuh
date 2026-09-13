// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/semantics/attention_postprocess.h"
#include "fused_validation.cuh"

namespace fused_attention {

// Fixture uses actual split-half RoPE frequencies, with global positions
// selected before CP sharding. Non-unit gamma prevents an identity-norm test.
__device__ inline float gamma(int feature, int segment) {
  return float(fuse::Bf16(0.8f + float((feature * 17 + segment * 7) % 31) / 100.f));
}

__device__ inline void rotary(int64_t global_row, int feature, float& c, float& s, int policy = 0) {
  float frequency = 1.f / powf(policy ? 500000.f : 1000000.f, float(2 * (feature % 64)) / 128.f);
  // Transformers v4.51.3 _compute_llama3_parameters, Llama3.1 policy:
  // factor8, low1/high4, original context8192. Qwen3 uses unscaled base1e6.
  if (policy) {
    const float wavelength = 6.283185307179586f / frequency;
    if (wavelength > 8192.f) frequency /= 8.f;
    else if (wavelength >= 2048.f) {
      const float smooth = (8192.f / wavelength - 1.f) / 3.f;
      frequency = (1.f - smooth) * frequency / 8.f + smooth * frequency;
    }
  }
  float sine, cosine;
  sincosf(float(global_row) * frequency, &sine, &cosine);
  c = float(fuse::Bf16(cosine)); s = float(fuse::Bf16(sine));
}

__global__ void initialize(fuse::Bf16* q, fuse::Bf16* k, fuse::Bf16* c, fuse::Bf16* s,
                          int rows, int rank, int policy = 0) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < int64_t(rows) * 128; i += int64_t(gridDim.x) * blockDim.x) {
    const int feature = i % 128;
    if (i < 128) { q[i] = fuse::Bf16(gamma(feature, 0)); k[i] = fuse::Bf16(gamma(feature, 1)); }
    float cosine, sine;
    rotary(int64_t(rank) * rows + i / 128, feature, cosine, sine, policy);
    c[i] = fuse::Bf16(cosine); s[i] = fuse::Bf16(sine);
  }
}

// Independent element oracle: no production warp reductions/SMEM/TMA helper.
// Evaluate the specified FP32 reduction tree with scalar arrays. A FP64 mean
// followed by BF16 rounding is not a bitwise oracle: at a halfway point one
// BF16 step can survive as a large relative error after RoPE cancellation.
// Matching the arithmetic contract lets the route retain a BYTE-EXACT check;
// raw projection remains independently checked against cuBLAS numerically.
// This expensive oracle is validation-only, never in any timing interval.
struct QkvOracle {
  fused_validation::QkvOracle route{};
  bool norm = true;
  float epsilon = 1.e-6f;
  int rope_policy = 0;
  __device__ uint16_t operator()(uint64_t index) const {
    const auto address = route.address(index);
    const auto* source = route.source[address.rank];
    const int width = route.q_width + 2 * route.kv_width;
    const int column = address.offset % width;
    const int segment = column < route.q_width ? 0 :
        (column < route.q_width + route.kv_width ? 1 : 2);
    if (segment == 2) return source[address.offset];
    const int feature = column % 128;
    const int64_t start = address.offset - feature;
    float x = fused_validation::bf16_to_float(source[address.offset]);
    float paired = fused_validation::bf16_to_float(source[start + (feature + 64) % 128]);
    if (norm) {
      float partial[2];
      for (int j = 0; j < 2; ++j) {
        float sum = 0.f;
        for (int part = 0; part < 64; ++part) {
          const int feature_index = j * 32 + part % 32 + (part / 32) * 64;
          const float value = fused_validation::bf16_to_float(source[start + feature_index]);
          sum = __fadd_rn(sum, __fmul_rn(value, value));
        }
        partial[j] = sum;
      }
      for (int half = 1; half; half /= 2)
        for (int j = 0; j < half; ++j) partial[j] = __fadd_rn(partial[j], partial[j + half]);
      const float inverse = rsqrtf(__fadd_rn(__fmul_rn(partial[0], 1.f / 128.f), epsilon));
      x = float(fuse::Bf16(float(fuse::Bf16(x * inverse)) * gamma(feature, segment)));
      paired = float(fuse::Bf16(float(fuse::Bf16(paired * inverse)) * gamma((feature + 64) % 128, segment)));
    }
    const int64_t global_row = int64_t(address.rank) * route.seq_local + address.offset / width;
    float c, s;
    rotary(global_row, feature, c, s, rope_policy);
    const float a = float(fuse::Bf16(x * c));
    const float b = float(fuse::Bf16((feature < 64 ? -paired : paired) * s));
    const fuse::Bf16 result(a + b);
    return result.raw();
  }
};

__global__ void initialize_gamma(fuse::Bf16* weights, int width) {
  for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < width; col += gridDim.x * blockDim.x)
    weights[col] = fuse::Bf16(gamma(col, 0));
}

// Independent full-hidden FP64 reference, scalar loads and a different tree
// from production's vector/FP32 row reduction. Preserve BF16 rounding points.
template <bool AlreadyAdded = false>
__global__ void residual_reference(fuse::Bf16* output, const fuse::Bf16* residual,
    fuse::Bf16* added, const fuse::Bf16* weights, int rows, int width, float epsilon) {
  __shared__ double partial[8];
  for (int row = blockIdx.x; row < rows; row += gridDim.x) {
    const int64_t base = int64_t(row) * width;
    double sum = 0.;
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
      fuse::Bf16 value;
      if constexpr (AlreadyAdded) value = added[base + col];
      else {
        value = fuse::Bf16(float(output[base + col]) + float(residual[base + col]));
        added[base + col] = value;
      }
      sum += double(float(value)) * double(float(value));
    }
    for (int delta = 16; delta; delta >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, delta);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      double total = 0.;
      for (int warp = 0; warp < 8; ++warp) total += partial[warp];
      partial[0] = 1. / sqrt(total / width + double(epsilon));
    }
    __syncthreads();
    const float inverse = float(partial[0]);
    for (int col = threadIdx.x; col < width; col += blockDim.x) {
      const float normalized = float(fuse::Bf16(float(added[base + col]) * inverse));
      output[base + col] = fuse::Bf16(normalized * float(weights[col]));
    }
    __syncthreads();
  }
}

// Reference refinement only, never a production or timed path. A whole row
// is recomputed from independent represented operands, not from fused output.
// FP64 dot -> FP32 -> BF16 projection -> BF16 residual sum. One warp/column
// keeps each weight read coalesced. The next reference launch normalizes the
// complete refined row in FP64. Do not patch just the failing output element:
// its correct norm depends on every projection value in the hidden row.
__global__ void fp64_projection_row(const fuse::Bf16* a, const fuse::Bf16* b,
    const fuse::Bf16* residual, fuse::Bf16* added, int width, int k) {
  const int lane = threadIdx.x & 31;
  const int column = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (column >= width) return;
  double sum = 0.;
  for (int inner = lane; inner < k; inner += 32)
    sum += double(float(a[inner])) * double(float(b[int64_t(column) * k + inner]));
  for (int delta = 16; delta; delta >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, delta);
  if (lane == 0)
    added[column] = fuse::Bf16(float(fuse::Bf16(float(sum))) + float(residual[column]));
}

}  // namespace fused_attention
