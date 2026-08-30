// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/a2a_gemm.h"
#include "fuse/operators/primitives/gemm_a2a.h"
#include "fuse/operators/ulysses/oproj_backward.h"
#include "fuse/operators/ulysses/qkv_backward.h"

namespace fuse::ulysses {

// Semantic axes for Ulysses projection dataflows.  The outer template names
// the physical order (GEMM -> A2A or A2A -> GEMM); these tags name the model
// projection and the differentiation direction.  Unsupported combinations
// intentionally remain incomplete and therefore fail at compile time.
struct QkvProjection {};
struct OutputProjection {};
struct Forward {};
struct Backward {};

template <class Projection, class Pass>
struct GemmA2A;

template <class Projection, class Pass>
struct A2AGemm;

// QKV forward: X * Wqkv, then sequence-to-head routing.
template <>
struct GemmA2A<QkvProjection, Forward> {
  using Bf16Params = ::fuse::Bf16GemmA2AParams;
  using Fp8Params = ::fuse::Fp8GemmA2AParams;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_gemm_a2a_cutlass(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_gemm_a2a_fp8_cutlass(params, stream);
  }
};

// QKV dgrad: head-to-sequence routing, then dX GEMM.  Weight-gradient GEMM is
// deliberately absent: it has no A2A and remains in qkv_backward.h, where the
// framework can choose immediate or deferred (ZeroBubble) execution.
template <>
struct A2AGemm<QkvProjection, Backward> {
  using Bf16Params = ::fuse::Bf16QkvBackwardDataParams;
  using Fp8Params = ::fuse::Fp8QkvBackwardDataParams;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_qkv_backward_data(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_qkv_backward_fp8_data(params, stream);
  }
};

// Output-projection forward: head-to-sequence routing, then A * Wo.
template <>
struct A2AGemm<OutputProjection, Forward> {
  using Bf16Params = ::fuse::Bf16A2AGemmParams;
  using Fp8Params = ::fuse::Fp8A2AGemmParams;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_a2a_gemm_cutlass(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_a2a_gemm_fp8_cutlass(params, stream);
  }
};

// Output-projection dgrad: dY * Wo, then sequence-to-head routing.  As above,
// the independent wgrad GEMM stays in the semantic backward API.
template <>
struct GemmA2A<OutputProjection, Backward> {
  using Bf16Params = ::fuse::Bf16OprojBackwardDataParams;
  using Fp8Params = ::fuse::Fp8OprojBackwardDataParams;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_oproj_backward_data(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_oproj_backward_fp8_data(params, stream);
  }
};

using QkvForward = GemmA2A<QkvProjection, Forward>;
using QkvBackwardDataflow = A2AGemm<QkvProjection, Backward>;
using OprojForward = A2AGemm<OutputProjection, Forward>;
using OprojBackwardDataflow = GemmA2A<OutputProjection, Backward>;

}  // namespace fuse::ulysses
