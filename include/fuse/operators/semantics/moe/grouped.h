// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/operators/primitives/grouped_gemm.h"
#include "fuse/operators/semantics/operator.h"

namespace fuse::moe {

namespace semantic {
struct SourceTokens {};
struct ExpertGateUp {};
struct ExpertActivatedRows {};
struct SourceTokenBranches {};
struct ExternalExpertRouting {};
struct CrossRankEpochCompletion {};
}  // namespace semantic

// FC1: [M_e,H] @ W[2F,H]^T. FC2: [M_e,F] @ W[H,F]^T.
// Activation is between these operators, outside either fusion. The caller
// supplies matching dimensions; these specifications add no model-name policy.
struct DispatchForwardSpec {
  using Dataflow = operators::dataflow::A2AThenGemm;
  using InputLayout = semantic::SourceTokens;
  using OutputLayout = semantic::ExpertGateUp;
  using Route = semantic::ExternalExpertRouting;
  using Completion = semantic::CrossRankEpochCompletion;
  using Bf16Params = Bf16GroupedGemmParams;
  static cudaError_t create(const Bf16Params& p, Bf16GroupedGemmPlan** plan) {
    return create_bf16_a2a_grouped_gemm(p, plan);
  }
  static cudaError_t launch(Bf16GroupedGemmPlan* plan, cudaStream_t stream) {
    return launch_bf16_grouped_gemm(plan, stream);
  }
};

struct CombineForwardSpec {
  using Dataflow = operators::dataflow::GemmThenA2A;
  using InputLayout = semantic::ExpertActivatedRows;
  using OutputLayout = semantic::SourceTokenBranches;
  using Route = semantic::ExternalExpertRouting;
  using Completion = semantic::CrossRankEpochCompletion;
  using Bf16Params = Bf16GroupedGemmParams;
  static cudaError_t create(const Bf16Params& p, Bf16GroupedGemmPlan** plan) {
    return create_bf16_grouped_gemm_a2a(p, plan);
  }
  static cudaError_t launch(Bf16GroupedGemmPlan* plan, cudaStream_t stream) {
    return launch_bf16_grouped_gemm(plan, stream);
  }
};

using ForwardRegistry = operators::SemanticRegistry<DispatchForwardSpec, CombineForwardSpec>;

}  // namespace fuse::moe
