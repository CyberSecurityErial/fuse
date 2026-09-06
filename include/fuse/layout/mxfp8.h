// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace fuse {

// Offline E4M3 payload, with one E8M0 byte per 32 consecutive columns in
// the ORIGINAL forward weight [output_features, input_features]. Backward
// must reuse this axis; transposing/requantizing the payload changes semantics.
// E8M0 encodes 2^(byte-127); 255 is NaN (not a finite scale).
// The enclosing GemmProblem describes BF16 computation after conversion;
// weight storage is described here, not by GemmProblem::weight_dtype.
struct Mxfp8Weight {
  const Fp8E4m3* payload = nullptr;
  const uint8_t* scales = nullptr;
  int32_t rows = 0;
  int32_t columns = 0;
};

// Baseline only: full row-major BF16 weight, owned by the caller. One buffer
// per concurrent invocation; keep it alive through stream/graph completion.
// It may be reused by serialized forward and B phases. W does not use it.
struct Mxfp8WeightWorkspace {
  Bf16* data = nullptr;
  size_t bytes = 0;
};

// Returns zero for invalid/unsupported geometry. Columns must be divisible
// by 32. No device allocation, synchronization, or quantization is performed.
size_t mxfp8_weight_workspace_bytes(const Mxfp8Weight& weight);

// Software conversion, not a native MXFP8 Tensor Core operation. Both inputs
// are device arrays on the active GPU. Payloads/scales may not alias output.
cudaError_t launch_mxfp8_weight_dequant(
    const Mxfp8Weight& weight,
    const Mxfp8WeightWorkspace& workspace,
    cudaStream_t stream);

}  // namespace fuse
