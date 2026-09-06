// SPDX-License-Identifier: BSD-3-Clause
// Shared MXFP8 weight API; assembled in the existing SM90 translation unit.

namespace fuse {

size_t mxfp8_weight_workspace_bytes(const Mxfp8Weight& weight) {
  if (weight.rows <= 0 || weight.columns <= 0 || weight.columns % 32 != 0) {
    return 0;
  }
  const uint64_t elements = uint64_t(weight.rows) * weight.columns;
  if (elements > std::numeric_limits<size_t>::max() / sizeof(Bf16)) {
    return 0;
  }
  return static_cast<size_t>(elements) * sizeof(Bf16);
}

cudaError_t launch_mxfp8_weight_dequant(
    const Mxfp8Weight& weight,
    const Mxfp8WeightWorkspace& workspace,
    cudaStream_t stream) {
  cudaError_t status = validate_mxfp8_weight(
      weight, workspace, weight.rows, weight.columns);
  if (status != cudaSuccess) {
    return status;
  }
  constexpr int32_t threads = 256;
  const int64_t elements = int64_t(weight.rows) * weight.columns;
  const int32_t blocks = static_cast<int32_t>(
      std::min<int64_t>((elements + threads - 1) / threads, 4096));
  mxfp8_weight_dequant_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(weight.payload),
      weight.scales, workspace.data, elements);
  return cudaGetLastError();
}

}  // namespace fuse
