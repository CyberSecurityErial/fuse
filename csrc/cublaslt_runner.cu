#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t kDefaultWorkspaceBytes = 64ull << 20;

struct Plan {
  int device = 0;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  void* workspace = nullptr;
  size_t workspace_capacity = 0;
  size_t algorithm_workspace = 0;
  int returned = 0;
  int valid = 0;
  float tune_ms = 0.0f;
  float waves = 0.0f;
};

thread_local std::string last_error;

void set_error(const char* call, int status) {
  char buffer[256];
  std::snprintf(buffer, sizeof(buffer), "%s failed with status %d", call, status);
  last_error = buffer;
}

#define CUDA_TRY(call) do { \
  cudaError_t status_ = (call); \
  if (status_ != cudaSuccess) { \
    last_error = std::string(#call) + ": " + cudaGetErrorString(status_); \
    return false; \
  } \
} while (0)

#define CUBLAS_TRY(call) do { \
  cublasStatus_t status_ = (call); \
  if (status_ != CUBLAS_STATUS_SUCCESS) { \
    set_error(#call, static_cast<int>(status_)); \
    return false; \
  } \
} while (0)

int algo_i32(const cublasLtMatmulAlgo_t& algorithm,
             cublasLtMatmulAlgoConfigAttributes_t attribute) {
  int value = -1;
  size_t written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(
             &algorithm, attribute, &value, sizeof(value), &written) ==
          CUBLAS_STATUS_SUCCESS
      ? value
      : -1;
}

int algo_u16(const cublasLtMatmulAlgo_t& algorithm,
             cublasLtMatmulAlgoConfigAttributes_t attribute) {
  uint16_t value = 0;
  size_t written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(
             &algorithm, attribute, &value, sizeof(value), &written) ==
          CUBLAS_STATUS_SUCCESS
      ? static_cast<int>(value)
      : -1;
}

bool launch(Plan* plan, const void* a, const void* b_nt, void* d,
            cudaStream_t stream, const cublasLtMatmulAlgo_t& algorithm) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(
      plan->handle, plan->operation, &alpha, a, plan->a, b_nt, plan->b,
      &beta, d, plan->c, d, plan->d, &algorithm, plan->workspace,
      plan->workspace_capacity, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    set_error("cublasLtMatmul", static_cast<int>(status));
    return false;
  }
  return true;
}

float time_candidate(Plan* plan, const void* a, const void* b_nt, void* d,
                     cudaStream_t stream,
                     const cublasLtMatmulAlgo_t& algorithm,
                     int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    if (!launch(plan, a, b_nt, d, stream, algorithm))
      return std::numeric_limits<float>::infinity();
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return std::numeric_limits<float>::infinity();
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    return std::numeric_limits<float>::infinity();
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    if (!launch(plan, a, b_nt, d, stream, algorithm)) {
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      return std::numeric_limits<float>::infinity();
    }
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float elapsed = 0.0f;
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return elapsed / iterations;
}

bool initialize(Plan* plan, const void* a, const void* b_nt, void* d,
                cudaStream_t stream, int tune_warmup, int tune_iters,
                size_t workspace_bytes) {
  CUDA_TRY(cudaSetDevice(plan->device));
  CUBLAS_TRY(cublasLtCreate(&plan->handle));
  CUBLAS_TRY(cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t trans_a = CUBLAS_OP_N;
  const cublasOperation_t trans_b = CUBLAS_OP_T;
  CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
  CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)));
  CUBLAS_TRY(cublasLtMatrixLayoutCreate(
      &plan->a, CUDA_R_16BF, plan->m, plan->k, plan->k));
  CUBLAS_TRY(cublasLtMatrixLayoutCreate(
      &plan->b, CUDA_R_16BF, plan->n, plan->k, plan->k));
  CUBLAS_TRY(cublasLtMatrixLayoutCreate(
      &plan->c, CUDA_R_16BF, plan->m, plan->n, plan->n));
  CUBLAS_TRY(cublasLtMatrixLayoutCreate(
      &plan->d, CUDA_R_16BF, plan->m, plan->n, plan->n));
  const cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
  for (auto layout : {plan->a, plan->b, plan->c, plan->d}) {
    CUBLAS_TRY(cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
  }

  plan->workspace_capacity = workspace_bytes == 0
      ? kDefaultWorkspaceBytes : workspace_bytes;
  CUDA_TRY(cudaMalloc(&plan->workspace, plan->workspace_capacity));
  cublasLtMatmulPreference_t preference = nullptr;
  CUBLAS_TRY(cublasLtMatmulPreferenceCreate(&preference));
  CUBLAS_TRY(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &plan->workspace_capacity, sizeof(plan->workspace_capacity)));
  constexpr int requested = 64;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics(requested);
  CUBLAS_TRY(cublasLtMatmulAlgoGetHeuristic(
      plan->handle, plan->operation, plan->a, plan->b, plan->c, plan->d,
      preference, requested, heuristics.data(), &plan->returned));
  cublasLtMatmulPreferenceDestroy(preference);

  int best = -1;
  float best_ms = std::numeric_limits<float>::infinity();
  for (int i = 0; i < plan->returned; ++i) {
    if (heuristics[i].state != CUBLAS_STATUS_SUCCESS ||
        heuristics[i].workspaceSize > plan->workspace_capacity)
      continue;
    const float ms = time_candidate(
        plan, a, b_nt, d, stream, heuristics[i].algo,
        std::max(tune_warmup, 1), std::max(tune_iters, 1));
    if (!std::isfinite(ms))
      continue;
    ++plan->valid;
    if (ms < best_ms) {
      best_ms = ms;
      best = i;
    }
  }
  if (best < 0) {
    last_error = "cuBLASLt returned no runnable BF16 algorithm";
    return false;
  }
  plan->algorithm = heuristics[best].algo;
  plan->algorithm_workspace = heuristics[best].workspaceSize;
  plan->tune_ms = best_ms;
  plan->waves = heuristics[best].wavesCount;
  return true;
}

void destroy(Plan* plan) {
  if (!plan) return;
  cudaSetDevice(plan->device);
  if (plan->workspace) cudaFree(plan->workspace);
  if (plan->d) cublasLtMatrixLayoutDestroy(plan->d);
  if (plan->c) cublasLtMatrixLayoutDestroy(plan->c);
  if (plan->b) cublasLtMatrixLayoutDestroy(plan->b);
  if (plan->a) cublasLtMatrixLayoutDestroy(plan->a);
  if (plan->operation) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->handle) cublasLtDestroy(plan->handle);
  delete plan;
}

}  // namespace

extern "C" {

struct FuseCublasLtInfo {
  int returned;
  int valid;
  int algo_id;
  int tile_id;
  int stages_id;
  int split_k;
  int reduction;
  int cta_swizzle;
  int custom;
  int inner_shape;
  int cluster_shape;
  uint64_t workspace_bytes;
  float tune_ms;
  float waves;
};

const char* fuse_cublaslt_last_error() { return last_error.c_str(); }

void* fuse_cublaslt_bf16_create(
    int device, int64_t m, int64_t n, int64_t k,
    const void* a, const void* b_nt, void* d, void* stream,
    int tune_warmup, int tune_iters, uint64_t workspace_bytes) {
  last_error.clear();
  auto plan = std::make_unique<Plan>();
  plan->device = device;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  if (!initialize(plan.get(), a, b_nt, d,
                  reinterpret_cast<cudaStream_t>(stream),
                  tune_warmup, tune_iters, workspace_bytes)) {
    destroy(plan.release());
    return nullptr;
  }
  return plan.release();
}

int fuse_cublaslt_bf16_run(
    void* opaque, const void* a, const void* b_nt, void* d, void* stream) {
  last_error.clear();
  auto* plan = reinterpret_cast<Plan*>(opaque);
  if (!plan) {
    last_error = "null cuBLASLt plan";
    return 0;
  }
  return launch(plan, a, b_nt, d, reinterpret_cast<cudaStream_t>(stream),
                plan->algorithm) ? 1 : 0;
}

int fuse_cublaslt_bf16_info(void* opaque, FuseCublasLtInfo* info) {
  auto* plan = reinterpret_cast<Plan*>(opaque);
  if (!plan || !info) return 0;
  *info = {
      plan->returned,
      plan->valid,
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_ID),
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID),
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID),
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
      algo_i32(plan->algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
      algo_u16(plan->algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
      algo_u16(plan->algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID),
      plan->algorithm_workspace,
      plan->tune_ms,
      plan->waves};
  return 1;
}

void fuse_cublaslt_bf16_destroy(void* opaque) {
  destroy(reinterpret_cast<Plan*>(opaque));
}

}  // extern "C"
