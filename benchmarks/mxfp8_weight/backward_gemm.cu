// Benchmark-only cuBLASLt NN/TN GEMMs. No production operator changes.
// Row-major physical A/B, logical transposes (no materialized transpose),
// BF16 operands, FP32 compute, BF16 or FP32 C/D and explicit beta.
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string error;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void check(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLASLt status " + std::to_string(int(s)));
}
struct Plan {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
  void* scratch = nullptr;
  size_t capacity = 64ull << 20, workspace = 0;
  cublasLtMatmulAlgo_t algo{};
  int returned = 0, valid = 0, id = -1;
  float tune_us = 0;
  ~Plan() {
    if (scratch) cudaFree(scratch);
    if (d) cublasLtMatrixLayoutDestroy(d);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
    if (handle) cublasLtDestroy(handle);
  }
  cublasStatus_t run(const void* av, const void* bv, void* dv,
                     cudaStream_t stream, float beta,
                     const cublasLtMatmulAlgo_t& candidate) {
    const float alpha = 1.f;
    return cublasLtMatmul(handle, op, &alpha, av, a, bv, b,
                         &beta, dv, d, dv, d, &candidate, scratch, capacity, stream);
  }
};
struct Events {
  cudaEvent_t start = nullptr, stop = nullptr;
  Events() { check(cudaEventCreate(&start)); check(cudaEventCreate(&stop)); }
  ~Events() { if (stop) cudaEventDestroy(stop); if (start) cudaEventDestroy(start); }
};
void layout(cublasLtMatrixLayout_t* out, cudaDataType_t dtype,
            int64_t rows, int64_t cols) {
  check(cublasLtMatrixLayoutCreate(out, dtype, rows, cols, cols));
  const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
  check(cublasLtMatrixLayoutSetAttribute(*out, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                       &row, sizeof(row)));
}
}
extern "C" {
const char* mxfp8_bwd_error() { return error.c_str(); }
void mxfp8_bwd_destroy(void* p) { delete static_cast<Plan*>(p); }

void* mxfp8_bwd_create(int64_t m, int64_t n, int64_t k, int ta, int tb,
                     int fp32_output, const void* av, const void* bv, void* dv,
                     void* stream_ptr, float beta, int candidates,
                     int warmup, int iterations) {
  auto* p = new Plan;
  cublasLtMatmulPreference_t pref = nullptr;
  try {
    if (m <= 0 || n <= 0 || k <= 0 || candidates < 1 || candidates > 64 ||
        warmup < 1 || iterations < 1 || (beta != 0 && beta != 1))
      throw std::runtime_error("invalid backward GEMM configuration");
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    check(cublasLtCreate(&p->handle));
    check(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opa = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opb = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof(opa)));
    check(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof(opb)));
    layout(&p->a, CUDA_R_16BF, ta ? k : m, ta ? m : k);
    layout(&p->b, CUDA_R_16BF, tb ? n : k, tb ? k : n);
    layout(&p->d, fp32_output ? CUDA_R_32F : CUDA_R_16BF, m, n);
    check(cudaMalloc(&p->scratch, p->capacity));
    check(cublasLtMatmulPreferenceCreate(&pref));
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &p->capacity, sizeof(p->capacity)));
    std::vector<cublasLtMatmulHeuristicResult_t> hs(candidates);
    check(cublasLtMatmulAlgoGetHeuristic(p->handle, p->op, p->a, p->b, p->d, p->d,
                                       pref, candidates, hs.data(), &p->returned));
    check(cublasLtMatmulPreferenceDestroy(pref)); pref = nullptr;
    Events events;
    float best = std::numeric_limits<float>::infinity();
    for (int i = 0; i < p->returned; ++i) {
      if (hs[i].state != CUBLAS_STATUS_SUCCESS || hs[i].workspaceSize > p->capacity) continue;
      // Initialize C before beta=1 tuning, avoiding uninitialized reads and
      // cross-candidate accumulation. Outside candidate timing.
      check(cudaMemsetAsync(dv, 0, size_t(m) * n * (fp32_output ? 4 : 2), stream));
      bool valid = true;
      for (int j = 0; j < warmup; ++j)
        if (p->run(av, bv, dv, stream, beta, hs[i].algo) != CUBLAS_STATUS_SUCCESS) {
          valid = false; break;
        }
      check(cudaStreamSynchronize(stream));
      if (!valid) continue;
      check(cudaEventRecord(events.start, stream));
      for (int j = 0; j < iterations; ++j)
        if (p->run(av, bv, dv, stream, beta, hs[i].algo) != CUBLAS_STATUS_SUCCESS) {
          valid = false; break;
        }
      check(cudaEventRecord(events.stop, stream));
      check(cudaEventSynchronize(events.stop));
      if (!valid) continue;
      float ms = 0;
      check(cudaEventElapsedTime(&ms, events.start, events.stop));
      if (!std::isfinite(ms) || ms <= 0) continue;
      ++p->valid;
      if (ms < best) {
        best = ms; p->algo = hs[i].algo; p->workspace = hs[i].workspaceSize;
        p->tune_us = ms * 1000 / iterations;
      }
    }
    if (!p->valid) throw std::runtime_error("no runnable cuBLASLt algorithm");
    size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_ID,
                                             &p->id, sizeof(p->id), &written));
    return p;
  } catch (const std::exception& e) {
    error = e.what();
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    delete p; return nullptr;
  }
}
int mxfp8_bwd_run(void* opaque, const void* av, const void* bv, void* dv,
                  void* stream, float beta) {
  try {
    auto* p = static_cast<Plan*>(opaque);
    if (!p) throw std::runtime_error("null plan");
    check(p->run(av, bv, dv, reinterpret_cast<cudaStream_t>(stream), beta, p->algo));
    return 1;
  } catch (const std::exception& e) { error = e.what(); return 0; }
}
int mxfp8_bwd_info(void* opaque, int* integers, float* timing, uint64_t* workspace) {
  auto* p = static_cast<Plan*>(opaque);
  if (!p) return 0;
  integers[0] = p->returned; integers[1] = p->valid; integers[2] = p->id;
  *timing = p->tune_us; *workspace = p->workspace;
  return 1;
}
}
