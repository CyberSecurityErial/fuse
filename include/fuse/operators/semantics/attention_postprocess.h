// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"

namespace fuse {

// Post-projection operations, NOT the input RMSNorm preceding the QKV GEMM.
// Qwen3: BF16 Q/K -> per-head RMSNorm -> split-half RoPE -> attention.
// Llama: disable Q/K norm; keep RoPE. V is always unchanged.
// Gamma is shared across heads, with head_dim BF16 entries for each Q/K norm.
// cos/sin are upstream BF16 tables [batch*seq_local, head_dim] in the SAME
// token order as the local activation. The caller selects the actual global
// positions (including packed sequences/CP permutation/long-context scaling).
// No implicit position=rank*seq_local assumption and no trigonometry in-kernel.
// A disabled configuration has every pointer null. All buffers are read-only,
// device-local, 16-byte aligned, disjoint from outputs/scratch and live through
// stream completion.
struct QkvPostprocess {
  const Bf16* q_gamma = nullptr;
  const Bf16* k_gamma = nullptr;
  const Bf16* cos = nullptr;
  const Bf16* sin = nullptr;
  float epsilon = 1.e-6f;

  CUTLASS_HOST_DEVICE bool enabled() const { return q_gamma || cos; }
};

// OProj -> BF16 residual add -> RMSNorm over the entire output hidden width.
// The added residual is also an output: the next MLP residual connection needs
// it unchanged by normalization. gamma has hidden_width BF16 entries. All rows
// follow projection.output's local CP token ordering. No RoPE at this boundary.
// Source residual, residual_output and projection.output must be disjoint.
// Experimental SM103 MXFP8 implementation supports hidden width <=16384;
// row cache/partials must fit its shared storage. Unsupported geometry returns
// cudaErrorNotSupported, not an N-shard approximation to full-hidden norm.
struct ResidualRmsNorm {
  const Bf16* residual = nullptr;
  const Bf16* gamma = nullptr;
  Bf16* residual_output = nullptr;
  float epsilon = 1.e-6f;
  CUTLASS_HOST_DEVICE bool enabled() const { return residual != nullptr; }
};

}  // namespace fuse
