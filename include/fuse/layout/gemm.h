// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>

namespace fuse {

enum class GemmRaster : int32_t {
  kHeuristic = 0,
  kAlongM = 1,
  kAlongN = 2,
};

enum class DType : int32_t {
  kBfloat16 = 0,
  kFloat8E4M3 = 1,
};

// Logical [row, column, batch] strides in elements. Negative values request
// the packed stride implied by the selected CUTLASS layout.
struct MatrixStride {
  int64_t row = -1;
  int64_t column = -1;
  int64_t batch = -1;
};

struct GemmProblem {
  int32_t m = 0;
  int32_t n = 0;
  int32_t k = 0;
  int32_t l = 1;
  MatrixStride stride_a{};
  MatrixStride stride_b{};
  MatrixStride stride_d{};
  DType input_dtype = DType::kBfloat16;
  DType weight_dtype = DType::kBfloat16;
  DType output_dtype = DType::kBfloat16;
  bool transpose_a = false;
  bool transpose_b = false;
  GemmRaster raster = GemmRaster::kHeuristic;
  int32_t max_swizzle_size = 1;
};

using GemmShape4D = GemmProblem;

}  // namespace fuse
