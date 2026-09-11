// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/gemm_a2a_mxfp8.h"
#include <cmath>

namespace fused_mxfp8 {

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
