// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

namespace fuse {

// Immediate mode runs data-gradient and weight-gradient work in the same
// backward call. Deferred mode (the ZeroBubble B/W split) runs only the
// data-gradient phase; the caller must keep the documented operands alive and
// call the operator's weight-gradient entry during the later W phase.
enum class WeightGradientMode : int32_t {
  kImmediate = 0,
  kDeferred = 1,
};

// Explicit data-gradient tile choices are benchmark knobs. kAuto is the
// production entry and is resolved from the GEMM shape, route work and the
// available compute CTAs; it never depends on a model name.
enum class BackwardGemmPolicy : int32_t {
  kAuto = 0,
  kM128N64 = 1,
  kM128N128 = 2,
  kM128N160 = 3,
  kM128N192 = 4,
  kM128N256 = 5,
  kM128N64ClusterM2 = 6,
};

}  // namespace fuse
