// SPDX-License-Identifier: BSD-3-Clause
// Software weight conversion for the SM90 MXFP8-weight baseline. No routing
// code lives here: the existing BF16 persistent dataflows remain authoritative.

namespace fuse {
namespace {

// Exact E4M3 * E8M0 -> BF16 RNE, including subnormals, signed zero and NaN.
// Integer construction avoids --use_fast_math flushing the smallest scales.
__device__ uint16_t mxfp8_to_bf16_bits(uint8_t value, uint8_t scale) {
  const uint16_t sign = static_cast<uint16_t>(value & 0x80) << 8;
  const int magnitude = value & 0x7f;
  if (magnitude == 0x7f || scale == 255) {
    return sign | 0x7fc0;
  }
  if (magnitude == 0) {
    return sign;
  }
  const int e = magnitude >> 3;
  int significand = (e == 0 ? 0 : 8) | (magnitude & 7);
  int exponent = (e == 0 ? -9 : e - 10) + int(scale) - 127;
  while (significand < 8) {
    significand <<= 1;
    --exponent;
  }
  const int unbiased = exponent + 3;
  if (unbiased > 127) {
    return sign | 0x7f80;
  }
  if (unbiased >= -126) {
    return sign | ((unbiased + 127) << 7) | ((significand - 8) << 4);
  }
  const int shift = exponent + 133;
  if (shift >= 0) {
    return sign | (significand << shift);
  }
  const int right = -shift;
  if (right > 4) {
    return sign;
  }
  int rounded = significand >> right;
  const int remainder = significand & ((1 << right) - 1);
  const int halfway = 1 << (right - 1);
  rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
  return sign | rounded;
}

__global__ void mxfp8_weight_dequant_kernel(
    const uint8_t* payload,
    const uint8_t* scales,
    Bf16* output,
    int64_t elements) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += int64_t(blockDim.x) * gridDim.x) {
    output[index] = Bf16::bitcast(
        mxfp8_to_bf16_bits(payload[index], scales[index / 32]));
  }
}

bool mxfp8_ranges_overlap(
    const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  const uintptr_t x = reinterpret_cast<uintptr_t>(a);
  const uintptr_t y = reinterpret_cast<uintptr_t>(b);
  return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

cudaError_t validate_mxfp8_weight(
    const Mxfp8Weight& weight,
    const Mxfp8WeightWorkspace& workspace,
    int64_t rows,
    int64_t columns) {
  const size_t bytes = mxfp8_weight_workspace_bytes(weight);
  if (!bytes || rows != weight.rows || columns != weight.columns ||
      !weight.payload || !weight.scales || !workspace.data ||
      workspace.bytes < bytes ||
      reinterpret_cast<uintptr_t>(workspace.data) % 16 != 0 ||
      mxfp8_ranges_overlap(workspace.data, bytes, weight.payload, bytes / 2) ||
      mxfp8_ranges_overlap(workspace.data, bytes, weight.scales, bytes / 64)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace
}  // namespace fuse
