// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <type_traits>

namespace fuse::operators {

// Physical dataflow is deliberately model-agnostic. A semantic specification
// selects one of these orders and supplies the tensor meaning and public
// parameter types; the primitive implementation only executes that contract.
namespace dataflow {
struct GemmThenA2A {};
struct A2AThenGemm {};
}  // namespace dataflow

template <class Spec, class = void>
struct IsSemanticSpec : std::false_type {};

template <class Spec>
struct IsSemanticSpec<
    Spec,
    std::void_t<
        typename Spec::Dataflow,
        typename Spec::InputLayout,
        typename Spec::OutputLayout,
        typename Spec::Route,
        typename Spec::Completion>> : std::true_type {};

template <class Spec>
inline constexpr bool kIsSemanticSpec = IsSemanticSpec<Spec>::value;

template <class... Specs>
struct SemanticRegistry {
  static_assert(
      (kIsSemanticSpec<Specs> && ...),
      "every registered type must satisfy the SemanticSpec contract");
  static constexpr std::size_t kSize = sizeof...(Specs);
};

// Compile-time facade shared by all model/algorithm operators. There is no
// virtual dispatch and no stored state: dynamic pointers, epochs and launch
// streams remain in the per-call parameter object.
template <class Spec>
struct PersistentOperator {
  static_assert(
      kIsSemanticSpec<Spec>,
      "SemanticSpec must declare its dataflow, layouts, route and completion");

  using Semantic = Spec;
  using Dataflow = typename Spec::Dataflow;
  using InputLayout = typename Spec::InputLayout;
  using OutputLayout = typename Spec::OutputLayout;
  using Route = typename Spec::Route;
  using Completion = typename Spec::Completion;

  template <class Params>
  static auto launch(const Params& params, cudaStream_t stream)
      -> decltype(Spec::launch(params, stream)) {
    return Spec::launch(params, stream);
  }
};

}  // namespace fuse::operators
