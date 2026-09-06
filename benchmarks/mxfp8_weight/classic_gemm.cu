// Benchmark-only classic cuBLAS compute reference. No DQ or communication.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <new>

struct Gemm {
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  cudaStream_t stream = nullptr;
  int m, n, k, lda, ldb;
  cublasOperation_t ta, tb;
  cudaDataType_t output;
  static constexpr size_t workspace_bytes = 32ull << 20;
};

namespace {
thread_local char error_message[256] = {};

bool check(cublasStatus_t status) {
  if (status == CUBLAS_STATUS_SUCCESS) return true;
  std::snprintf(error_message, sizeof(error_message), "cuBLAS status %d", int(status));
  return false;
}

bool check(cudaError_t status) {
  if (status == cudaSuccess) return true;
  std::snprintf(error_message, sizeof(error_message), "%s", cudaGetErrorString(status));
  return false;
}
}  // namespace

extern "C" const char* mxfp8_classic_error() { return error_message; }

extern "C" void mxfp8_classic_destroy(Gemm* gemm) {
  if (!gemm) return;
  if (gemm->handle) cublasDestroy(gemm->handle);
  if (gemm->workspace) cudaFree(gemm->workspace);
  delete gemm;
}

extern "C" Gemm* mxfp8_classic_create(int m, int n, int k, int ta, int tb,
                                      int fp32_output, cudaStream_t stream) {
  if (m <= 0 || n <= 0 || k <= 0) {
    std::snprintf(error_message, sizeof(error_message), "invalid MNK");
    return nullptr;
  }
  auto* gemm = new (std::nothrow) Gemm;
  if (!gemm) return nullptr;
  gemm->m = m;
  gemm->n = n;
  gemm->k = k;
  gemm->lda = ta ? m : k;
  gemm->ldb = tb ? k : n;
  gemm->ta = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
  gemm->tb = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  gemm->output = fp32_output ? CUDA_R_32F : CUDA_R_16BF;
  gemm->stream = stream;
  if (!check(cublasCreate(&gemm->handle)) ||
      !check(cudaMalloc(&gemm->workspace, Gemm::workspace_bytes)) ||
      !check(cublasSetStream(gemm->handle, stream)) ||
      !check(cublasSetWorkspace(gemm->handle, gemm->workspace, Gemm::workspace_bytes))) {
    mxfp8_classic_destroy(gemm);
    return nullptr;
  }
  return gemm;
}

extern "C" int mxfp8_classic_run(Gemm* gemm, const void* a, const void* b,
                                   void* output, cudaStream_t stream, float beta) {
  if (!gemm) return 0;
  if (stream != gemm->stream) {
    // Setting a stream resets cuBLAS's workspace, so reattach it explicitly.
    if (!check(cublasSetStream(gemm->handle, stream)) ||
        !check(cublasSetWorkspace(gemm->handle, gemm->workspace, Gemm::workspace_bytes))) return 0;
    gemm->stream = stream;
  }
  const float alpha = 1.0f;
  // Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T.
  return check(cublasGemmEx(gemm->handle, gemm->tb, gemm->ta,
                            gemm->n, gemm->m, gemm->k, &alpha,
                            b, CUDA_R_16BF, gemm->ldb,
                            a, CUDA_R_16BF, gemm->lda, &beta,
                            output, gemm->output, gemm->n,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

extern "C" int mxfp8_classic_version(Gemm* gemm) {
  int version = 0;
  if (!gemm || !check(cublasGetVersion(gemm->handle, &version))) return 0;
  return version;
}
