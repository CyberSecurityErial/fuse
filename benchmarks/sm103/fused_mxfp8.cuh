// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/gemm_a2a_mxfp8.h"
#include "fuse/operators/primitives/a2a_gemm_mxfp8.h"
#include <cmath>

namespace fused_mxfp8 {

// Independent scalar address expression for the documented native K32 scale
// atom ((32,4),(32,4)): strides ((16,4),(0,1)), K blocks before row blocks.
// No production routing, scale layout helper or copy task iterator is reused.
__host__ __device__ inline int64_t scale_offset(int64_t row, int k, int width) {
  return ((row / 128) * ((width + 127) / 128) + k / 128) * 512 +
      (row % 32) * 16 + ((row % 128) / 32) * 4 + (k % 128) / 32;
}

struct ZeroOracle {
  __host__ __device__ uint16_t operator()(uint64_t) const { return 0; }
};

// Compare every received FP8 byte AND its scale against the actual upstream
// peer view. Output finite BF16 mismatch markers for the shared Stats reducer;
// this is a byte-exact transport test, not a dequantized-value equivalence test.
__global__ void check_oproj_route(fuse::Mxfp8A2AGemmParams p,
    fuse::Mxfp8Activation actual, uint16_t* mismatch) {
  const auto& r = p.projection.route;
  const int width = p.projection.gemm.k, peer_k = width / r.world_size;
  const int64_t count = int64_t{p.projection.gemm.m} * width;
  for (int64_t i = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < count; i += int64_t{gridDim.x} * blockDim.x) {
    const int row = i / width, k = i % width, shard = k / peer_k;
    const int peer = r.cyclic_peer_order ? (r.rank + shard) % r.world_size : shard;
    const int batch = row / r.seq_local, seq = row % r.seq_local;
    const int half = r.seq_local / 2;
    const int src_seq = r.causal_load_balanced
        ? (seq < half ? r.rank * half + seq : (2*r.world_size-r.rank-1)*half + seq-half)
        : r.rank * r.seq_local + seq;
    const int64_t src_row = int64_t{batch} * r.global_seq + src_seq;
    const auto& source = p.activation[peer];
    const bool equal = reinterpret_cast<const uint8_t*>(actual.data)[i] ==
        reinterpret_cast<const uint8_t*>(source.data)[src_row * peer_k + k % peer_k] &&
        actual.scales[scale_offset(row, k, width)] ==
        source.scales[scale_offset(src_row, k % peer_k, peer_k)];
    mismatch[i] = equal ? 0 : 0x3f80;
  }
}

// Independent numerical oracle: quantize/dequantize from original BF16,
// without reading production FP8 buffers, packed scales, or layout helpers.
// frexp + a boundary comparison differs from production's ilogb/rescaling.
// This tests the GEMM against represented operands, not against unquantized
// BF16 values (which would conflate quantization error with a kernel error).
__global__ void reference_operand(const fuse::Bf16* source, fuse::Bf16* decoded,
                                  int64_t count) {
  const int lane = threadIdx.x % 32;
  for (int64_t group = (int64_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
       group < count / 32; group += int64_t{gridDim.x} * blockDim.x / 32) {
    const int64_t index = group * 32 + lane;
    const float value = float(source[index]);
    float amax = fabsf(value);
    for (int mask = 16; mask; mask >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    int exponent = 0;
    if (amax > 0) {
      frexpf(amax, &exponent);
      exponent -= 9;
      if (amax > ldexpf(448.0f, exponent)) ++exponent;
    }
    exponent = max(-127, min(127, exponent));
    decoded[index] = fuse::Bf16(ldexpf(float(fuse::Fp8E4m3(ldexpf(value, -exponent))), exponent));
  }
}

}  // namespace fused_mxfp8
