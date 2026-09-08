// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <utility>

namespace fuse {

KernelTraits cutlass_kernel_traits() {
  OprojGemmPolicy policy{};
  if (select_oproj_gemm_policy(&policy) != cudaSuccess) {
    return {};
  }
  KernelTraits traits{};
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    traits = kernel_traits<typename Binding::Kernel>();
    if constexpr (Binding::kSwapAB) {
      // Public geometry describes sequence/projection, not the internal D^T
      // axes. Launch reservations and physical thread/SMEM counts are unchanged.
      std::swap(traits.block_m, traits.block_n);
    }
    return cudaSuccess;
  };
  return visit_oproj_forward_policy(policy, read_traits) == cudaSuccess
      ? traits : KernelTraits{};
}

KernelTraits projection_cutlass_kernel_traits() {
  return kernel_traits<GemmA2AKernel>();
}

KernelTraits qkv_cutlass_kernel_traits(const GemmProblem&) {
  // Conservative ready capacity for every registered tile, independent of
  // the override. This is not the geometry of the selected launch.
  return kernel_traits<typename QkvForwardN64Binding::Kernel>();
}

KernelTraits qkv_cutlass_kernel_traits(
    const GemmProblem& problem, const UlyssesRoute& route,
    int32_t num_comm_ctas, int32_t sm_count) {
  KernelTraits traits{};
  QkvGemmPolicy policy{};
  if (!supported_problem(problem) || num_comm_ctas <= 0 ||
      num_comm_ctas >= sm_count || select_qkv_gemm_policy(&policy) != cudaSuccess) {
    return traits;
  }
  auto read_traits = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    traits = kernel_traits<typename Binding::Kernel>();
    return cudaSuccess;
  };
  return visit_qkv_forward_policy(policy, route.qkv_peer_interleaved, read_traits) ==
          cudaSuccess ? traits : KernelTraits{};
}

int64_t a2a_lhs_gemm_ready_elements(const GemmProblem& p, const UlyssesRoute& route) {
  if (p.m <= 0 || p.l != 1 || route.world_size <= 0 || route.world_size > kMaxWorldSize) {
    return 0;
  }
  OprojGemmPolicy policy{};
  if (select_oproj_gemm_policy(&policy) != cudaSuccess) {
    return 0;
  }
  int64_t elements = 0;
  auto count_ready = [&](auto binding_tag) {
    using Binding = typename decltype(binding_tag)::type;
    using Comm = typename Binding::Comm;
    A2AGemmParams params{};
    params.gemm = p;
    params.route = route;
    const int32_t slices = Comm::ready_slices(params);
    if (slices <= 0) {
      return cudaErrorNotSupported;
    }
    elements = static_cast<int64_t>(ceil_div(p.m, Binding::kSequenceTile)) *
        route.world_size * slices * kReadyFlagStride;
    return cudaSuccess;
  };
  return visit_oproj_forward_policy(policy, count_ready) == cudaSuccess ? elements : 0;
}

// SM90's auto-selection API is retained as a query, but SM103 has no
// automatic communication budget. A zero recommendation must not be launched.
int32_t recommended_a2a_lhs_gemm_comm_ctas(const GemmProblem&, const UlyssesRoute&) {
  return 0;
}

int32_t recommended_gemm_a2a_comm_ctas(const GemmProblem&, const UlyssesRoute&) {
  return 0;
}

}  // namespace fuse
