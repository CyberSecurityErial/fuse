// SPDX-License-Identifier: BSD-3-Clause
// Independent Grouped compilation entry; Projection is not linked into this TU.
#include "api/grouped.cuh"

namespace fuse {

cudaError_t create_bf16_a2a_grouped_gemm(const Bf16GroupedGemmParams& p, Bf16GroupedGemmPlan** out) {
  return detail::create_grouped_plan<false>(p, out);
}
cudaError_t create_bf16_grouped_gemm_a2a(const Bf16GroupedGemmParams& p, Bf16GroupedGemmPlan** out) {
  return detail::create_grouped_plan<true>(p, out);
}
cudaError_t launch_bf16_grouped_gemm(Bf16GroupedGemmPlan* plan, cudaStream_t stream) {
  if (!plan) return cudaErrorInvalidValue;
  int current = -1;
  auto status = cudaGetDevice(&current);
  if (status != cudaSuccess) return status;
  if (current != plan->device) return cudaErrorInvalidDevice;
  return plan->launch(stream);
}
cudaError_t destroy_bf16_grouped_gemm(Bf16GroupedGemmPlan* plan) {
  if (!plan) return cudaSuccess;
  int current = -1;
  auto status = cudaGetDevice(&current);
  if (status != cudaSuccess) return status;
  status = cudaSetDevice(plan->device);
  if (status != cudaSuccess) return status;
  delete plan;
  return cudaSetDevice(current);
}

}  // namespace fuse
