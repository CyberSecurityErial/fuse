// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "fuse/types.h"
#include "../fused_validation.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

// Independent represented-operand reference. Read BF16 masters, recompute
// K32 quantization on the NEW GEMM axes, and multiply decoded values with
// cuBLAS. Never read production FP8/scales or use actual dA/dW as a reference.
// Scratch is bounded by 128 output rows and 4096 K elements, rather than
// duplicating entire long-sequence operands. No reference work is timed.
namespace mxfp8_reference {

inline void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("MXFP8 reference cuBLAS status=" + std::to_string(int(status)));
}

struct Operand {
  const fuse::Bf16* source;
  int64_t row_stride, k_stride;
  __device__ fuse::Bf16 operator()(int row,int k) const {
    return source[int64_t(row)*row_stride+int64_t(k)*k_stride];
  }
};

// Independent virtual gradient matrix, reconstructed from ORIGINAL peer BF16
// planes. Never read the production staging/quantization or its route decoder.
// The transpose view changes the K32 reduction axis for dW without allocating
// a full gathered reference tensor. This also supplies a byte-exact route oracle.
struct QkvGradient {
  const fuse::Bf16* source[8][3]{};
  int m=0,q_heads=0,kv_heads=0,world=0,rank=0;
  bool causal=false,transpose=false;
  __device__ fuse::Bf16 operator()(int row,int k) const {
    const int token=transpose?k:row, feature=transpose?row:k;
    const int q=q_heads*128,kv=kv_heads*128;
    const int kind=feature<q?0:(feature<q+kv?1:2);
    const int within=feature-(kind==0?0:(kind==1?q:q+kv));
    const int width=(kind==0?q:kv)/world;
    const int owner=within/width, column=within%width;
    const int global=causal?(token<m/2?rank*m/2+token:
        (2*world-1-rank)*m/2+token-m/2):rank*m+token;
    return source[owner][kind][int64_t(global)*width+column];
  }
  __device__ uint16_t operator()(uint64_t index) const {
    const int width=(q_heads+2*kv_heads)*128;
    return (*this)(int(index/width),int(index%width)).raw();
  }
};

template<class View>
static __global__ void decode(View source, fuse::Bf16* output,
                             int row_begin, int rows, int k_begin, int k) {
  const int lane = threadIdx.x % 32;
  for (int64_t group = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       group < int64_t(rows) * (k / 32); group += int64_t(gridDim.x) * blockDim.x / 32) {
    const int row = int(group / (k / 32)), col = int(group % (k / 32)) * 32 + lane;
    const float value = float(source(row_begin+row,k_begin+col));
    float amax = fabsf(value);
    for (int delta = 16; delta; delta /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
    int exponent = 0;
    if (amax > 0) {
      frexpf(amax, &exponent);
      exponent -= 9;
      if (amax > ldexpf(448.f, exponent)) ++exponent;
    }
    exponent = max(-127, min(127, exponent));
    output[int64_t(row) * k + col] = fuse::Bf16(ldexpf(
        float(fuse::Fp8E4m3(ldexpf(value, -exponent))), exponent));
  }
}

static __global__ void finish(const float* input, fuse::Bf16* output, uint64_t count) {
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += uint64_t(gridDim.x) * blockDim.x) output[i] = fuse::Bf16(input[i]);
}

struct Workspace {
  static constexpr int kRows = 128, kChunk = 4096;
  fuse::Bf16 *lhs{}, *rhs{}, *expected{};
  float* accumulator{};
  fused_validation::Scratch* comparison{};
  cublasHandle_t handle{};
  int columns = 0;

  // Explicit ownership: call release after all stream/reference work completes.
  // No destructor-driven waits in the multi-rank failure path.
  template <class Allocate>
  void initialize(int n, cudaStream_t stream, Allocate allocate) {
    if (n <= 0 || handle) throw std::invalid_argument("reference workspace dimensions/state");
    columns = n;
    lhs = static_cast<fuse::Bf16*>(allocate(size_t(kRows) * kChunk * sizeof(fuse::Bf16)));
    rhs = static_cast<fuse::Bf16*>(allocate(size_t(n) * kChunk * sizeof(fuse::Bf16)));
    expected = static_cast<fuse::Bf16*>(allocate(size_t(kRows) * n * sizeof(fuse::Bf16)));
    accumulator = static_cast<float*>(allocate(size_t(kRows) * n * sizeof(float)));
    comparison = static_cast<fused_validation::Scratch*>(allocate(sizeof(fused_validation::Scratch)));
    check(cublasCreate(&handle));
    check(cublasSetStream(handle, stream));
    check(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  }
  void release() { if (handle) { check(cublasDestroy(handle)); handle = nullptr; } }

  fused_validation::Stats validate(Operand a, Operand bt, const fuse::Bf16* actual,
                                  int m, int n, int k, cudaStream_t stream) {
    return validate_views(a,bt,actual,m,n,k,stream);
  }
  template<class Left,class Right>
  fused_validation::Stats validate_views(Left a, Right bt, const fuse::Bf16* actual,
                                        int m,int n,int k,cudaStream_t stream) {
    if (!handle || m <= 0 || n != columns || k <= 0 || k % 32)
      throw std::invalid_argument("reference matrix dimensions/state");
    check(cublasSetStream(handle, stream));
    auto total = fused_validation::Stats::zero();
    for (int row = 0; row < m; row += kRows) {
      const int rows = std::min(kRows, m - row);
      for (int base = 0; base < k; base += kChunk) {
        const int width = std::min(kChunk, k - base);
        decode<<<256,256,0,stream>>>(a,lhs,row,rows,base,width);
        decode<<<256,256,0,stream>>>(bt,rhs,0,n,base,width);
        check(cudaGetLastError());
        const float alpha = 1, beta = base == 0 ? 0 : 1;
        // Row-major A * B^T == column-major B * A^T. Accumulate K chunks
        // in FP32; round once after the complete reduction, like the operator.
        check(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,width,
            &alpha,rhs,CUDA_R_16BF,width,lhs,CUDA_R_16BF,width,&beta,
            accumulator,CUDA_R_32F,n,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
      }
      finish<<<256,256,0,stream>>>(accumulator,expected,uint64_t(rows)*n);
      check(cudaGetLastError());
      check(fused_validation::launch<true>(reinterpret_cast<const uint16_t*>(actual+int64_t(row)*n),
          fused_validation::DenseOracle{reinterpret_cast<const uint16_t*>(expected)},
          uint64_t(rows)*n,comparison,0,stream));
      fused_validation::Stats slab{};
      check(cudaMemcpyAsync(&slab,comparison->result,sizeof(slab),cudaMemcpyDeviceToHost,stream));
      check(cudaStreamSynchronize(stream));
      if (slab.first_index != std::numeric_limits<uint64_t>::max()) slab.first_index += uint64_t(row)*n;
      total.merge(slab);
    }
    return total;
  }
};

} // namespace mxfp8_reference
