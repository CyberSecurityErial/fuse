// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/a2a_gemm.h"
#include "fuse/operators/primitives/gemm_a2a.h"
#include "fuse/operators/semantics/operator.h"
#include "fuse/operators/ulysses/oproj_backward.h"
#include "fuse/operators/ulysses/qkv_backward.h"

#include <type_traits>
#include <utility>

namespace fuse::ulysses {

// Semantic registry for the four projection dataflows in this file:
//
//   QkvForwardSpec       sequence-sharded X -> head-sharded Q/K/V
//   QkvBackwardSpec      head-sharded dQ/dK/dV -> sequence-sharded dX
//   OprojForwardSpec     head-sharded attention -> sequence-sharded Y
//   OprojBackwardSpec    sequence-sharded dY -> head-sharded dAttention
//
// Weight-gradient GEMMs are intentionally not registered here because they
// contain no A2A. Their immediate/deferred ZeroBubble contract remains in the
// QKV/OProj backward semantic APIs.

struct QkvProjection {};
struct OutputProjection {};
struct Forward {};
struct Backward {};

namespace semantic {

// These are tensor-meaning tags, not storage classes. Runtime strides and
// addresses stay in the public parameter structs used by each specification.
struct SequenceShardedHidden {};
struct HeadShardedQkv {};
struct HeadShardedAttention {};

struct SequenceToHead {};
struct HeadToSequence {};
struct CrossRankEpochCompletion {};

}  // namespace semantic

struct QkvForwardSpec {
  using Projection = QkvProjection;
  using Pass = Forward;
  using Dataflow = operators::dataflow::GemmThenA2A;
  using InputLayout = semantic::SequenceShardedHidden;
  using OutputLayout = semantic::HeadShardedQkv;
  using Route = semantic::SequenceToHead;
  using Completion = semantic::CrossRankEpochCompletion;
  using Bf16Params = ::fuse::Bf16GemmA2AParams;
  using Fp8Params = ::fuse::Fp8GemmA2AParams;
  static constexpr bool kWeightGradientIsSeparate = false;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_gemm_a2a_cutlass(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_gemm_a2a_fp8_cutlass(params, stream);
  }
};

struct QkvBackwardSpec {
  using Projection = QkvProjection;
  using Pass = Backward;
  using Dataflow = operators::dataflow::A2AThenGemm;
  using InputLayout = semantic::HeadShardedQkv;
  using OutputLayout = semantic::SequenceShardedHidden;
  using Route = semantic::HeadToSequence;
  using Completion = semantic::CrossRankEpochCompletion;
  using Bf16Params = ::fuse::Bf16QkvBackwardDataParams;
  using Fp8Params = ::fuse::Fp8QkvBackwardDataParams;
  static constexpr bool kWeightGradientIsSeparate = true;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_qkv_backward_data(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_qkv_backward_fp8_data(params, stream);
  }
};

struct OprojForwardSpec {
  using Projection = OutputProjection;
  using Pass = Forward;
  using Dataflow = operators::dataflow::A2AThenGemm;
  using InputLayout = semantic::HeadShardedAttention;
  using OutputLayout = semantic::SequenceShardedHidden;
  using Route = semantic::HeadToSequence;
  using Completion = semantic::CrossRankEpochCompletion;
  using Bf16Params = ::fuse::Bf16A2AGemmParams;
  using Fp8Params = ::fuse::Fp8A2AGemmParams;
  static constexpr bool kWeightGradientIsSeparate = false;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_a2a_gemm_cutlass(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_a2a_gemm_fp8_cutlass(params, stream);
  }
};

struct OprojBackwardSpec {
  using Projection = OutputProjection;
  using Pass = Backward;
  using Dataflow = operators::dataflow::GemmThenA2A;
  using InputLayout = semantic::SequenceShardedHidden;
  using OutputLayout = semantic::HeadShardedAttention;
  using Route = semantic::SequenceToHead;
  using Completion = semantic::CrossRankEpochCompletion;
  using Bf16Params = ::fuse::Bf16OprojBackwardDataParams;
  using Fp8Params = ::fuse::Fp8OprojBackwardDataParams;
  static constexpr bool kWeightGradientIsSeparate = true;

  static cudaError_t launch(const Bf16Params& params, cudaStream_t stream) {
    return ::fuse::launch_oproj_backward_data(params, stream);
  }

  static cudaError_t launch(const Fp8Params& params, cudaStream_t stream) {
    return ::fuse::launch_oproj_backward_fp8_data(params, stream);
  }
};

template <class Spec, class = void>
struct IsProjectionSemanticSpec : std::false_type {};

template <class Spec>
struct IsProjectionSemanticSpec<
    Spec,
    std::void_t<
        typename Spec::Projection,
        typename Spec::Pass,
        typename Spec::Bf16Params,
        typename Spec::Fp8Params,
        decltype(Spec::kWeightGradientIsSeparate),
        decltype(Spec::launch(
            std::declval<const typename Spec::Bf16Params&>(),
            std::declval<cudaStream_t>())),
        decltype(Spec::launch(
            std::declval<const typename Spec::Fp8Params&>(),
            std::declval<cudaStream_t>()))>>
    : std::bool_constant<
          operators::kIsSemanticSpec<Spec> &&
          std::is_same_v<
              decltype(Spec::launch(
                  std::declval<const typename Spec::Bf16Params&>(),
                  std::declval<cudaStream_t>())),
              cudaError_t> &&
          std::is_same_v<
              decltype(Spec::launch(
                  std::declval<const typename Spec::Fp8Params&>(),
                  std::declval<cudaStream_t>())),
              cudaError_t> &&
          std::is_same_v<
              std::remove_cv_t<
                  decltype(Spec::kWeightGradientIsSeparate)>,
              bool>> {};

template <class Spec>
inline constexpr bool kIsProjectionSemanticSpec =
    IsProjectionSemanticSpec<Spec>::value;

template <class Spec>
struct ProjectionOperator : operators::PersistentOperator<Spec> {
  static_assert(
      kIsProjectionSemanticSpec<Spec>,
      "Ulysses projection specs must declare projection/pass, BF16/FP8 "
      "params, launch overloads and the independent-wgrad contract");
  using Base = operators::PersistentOperator<Spec>;
  using Projection = typename Spec::Projection;
  using Pass = typename Spec::Pass;
  using Bf16Params = typename Spec::Bf16Params;
  using Fp8Params = typename Spec::Fp8Params;
  static constexpr bool kWeightGradientIsSeparate =
      Spec::kWeightGradientIsSeparate;
  using Base::launch;
};

static_assert(kIsProjectionSemanticSpec<QkvForwardSpec>);
static_assert(kIsProjectionSemanticSpec<QkvBackwardSpec>);
static_assert(kIsProjectionSemanticSpec<OprojForwardSpec>);
static_assert(kIsProjectionSemanticSpec<OprojBackwardSpec>);

using RegisteredProjectionSemantics = operators::SemanticRegistry<
    QkvForwardSpec,
    QkvBackwardSpec,
    OprojForwardSpec,
    OprojBackwardSpec>;
static_assert(RegisteredProjectionSemantics::kSize == 4);

template <class Projection, class Pass>
struct ProjectionSemanticSpec;

template <>
struct ProjectionSemanticSpec<QkvProjection, Forward> {
  using type = QkvForwardSpec;
};

template <>
struct ProjectionSemanticSpec<QkvProjection, Backward> {
  using type = QkvBackwardSpec;
};

template <>
struct ProjectionSemanticSpec<OutputProjection, Forward> {
  using type = OprojForwardSpec;
};

template <>
struct ProjectionSemanticSpec<OutputProjection, Backward> {
  using type = OprojBackwardSpec;
};

template <class Projection, class Pass>
using ProjectionSemanticSpecT =
    typename ProjectionSemanticSpec<Projection, Pass>::type;

// Compatibility facades preserve the v13 public spellings while delegating to
// the one semantic contract registered above. An order/semantic mismatch is a
// compile-time error rather than a runtime branch in the primitive.
template <class Projection, class Pass>
struct GemmA2A
    : ProjectionOperator<ProjectionSemanticSpecT<Projection, Pass>> {
  using Spec = ProjectionSemanticSpecT<Projection, Pass>;
  static_assert(
      std::is_same_v<
          typename Spec::Dataflow,
          operators::dataflow::GemmThenA2A>,
      "this projection/pass semantic is not GEMM-then-A2A");
};

template <class Projection, class Pass>
struct A2AGemm
    : ProjectionOperator<ProjectionSemanticSpecT<Projection, Pass>> {
  using Spec = ProjectionSemanticSpecT<Projection, Pass>;
  static_assert(
      std::is_same_v<
          typename Spec::Dataflow,
          operators::dataflow::A2AThenGemm>,
      "this projection/pass semantic is not A2A-then-GEMM");
};

using QkvForward = GemmA2A<QkvProjection, Forward>;
using QkvBackwardDataflow = A2AGemm<QkvProjection, Backward>;
using OprojForward = A2AGemm<OutputProjection, Forward>;
using OprojBackwardDataflow = GemmA2A<OutputProjection, Backward>;

static_assert(std::is_same_v<QkvForward::Semantic, QkvForwardSpec>);
static_assert(std::is_same_v<QkvBackwardDataflow::Semantic, QkvBackwardSpec>);
static_assert(std::is_same_v<OprojForward::Semantic, OprojForwardSpec>);
static_assert(std::is_same_v<OprojBackwardDataflow::Semantic, OprojBackwardSpec>);

}  // namespace fuse::ulysses
