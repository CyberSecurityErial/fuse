#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

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
  bool cache_hit = false;
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
            cudaStream_t stream, const cublasLtMatmulAlgo_t& algorithm,
            float beta = 0.0f) {
  const float alpha = 1.0f;
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
                     int warmup, int iterations, bool use_graph = false) {
  for (int i = 0; i < warmup; ++i) {
    if (!launch(plan, a, b_nt, d, stream, algorithm))
      return std::numeric_limits<float>::infinity();
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess)
    return std::numeric_limits<float>::infinity();
  // The SM103 planner explicitly selects the real launch mode. Historical
  // callers retain eager tuning unless they opt in through the job environment.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaStream_t tuning_stream = nullptr;
  auto cleanup_graph = [&]() {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    if (tuning_stream) cudaStreamDestroy(tuning_stream);
  };
  if (use_graph) {
    // PyTorch callers may be on the legacy default stream, which cannot be
    // captured. Inputs are ready after the caller-stream synchronization above.
    if (cudaStreamCreateWithFlags(&tuning_stream, cudaStreamNonBlocking) != cudaSuccess)
      return std::numeric_limits<float>::infinity();
    stream = tuning_stream;
    auto begun = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (begun != cudaSuccess) {
      last_error = std::string("cudaStreamBeginCapture: ") + cudaGetErrorString(begun);
      cleanup_graph();
      return std::numeric_limits<float>::infinity();
    }
    bool launched = launch(plan, a, b_nt, d, stream, algorithm);
    cudaError_t ended = cudaStreamEndCapture(stream, &graph);
    if (!launched || ended != cudaSuccess || !graph ||
        cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0) != cudaSuccess) {
      cleanup_graph();
      return std::numeric_limits<float>::infinity();
    }
    for (int i = 0; i < warmup; ++i) {
      if (cudaGraphLaunch(executable, stream) != cudaSuccess) {
        cleanup_graph();
        return std::numeric_limits<float>::infinity();
      }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
      cleanup_graph();
      return std::numeric_limits<float>::infinity();
    }
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
    if (start) cudaEventDestroy(start);
    cleanup_graph();
    return std::numeric_limits<float>::infinity();
  }
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    bool launched = use_graph ? cudaGraphLaunch(executable, stream) == cudaSuccess
                              : launch(plan, a, b_nt, d, stream, algorithm);
    if (!launched) {
      cudaEventDestroy(stop);
      cudaEventDestroy(start);
      cleanup_graph();
      return std::numeric_limits<float>::infinity();
    }
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);
  float elapsed = 0.0f;
  cudaEventElapsedTime(&elapsed, start, stop);
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  cleanup_graph();
  return elapsed / iterations;
}

__global__ void validate_bf16_accumulate_kernel(
    const __nv_bfloat16* once, const __nv_bfloat16* twice,
    int64_t count, unsigned int* mismatches) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const __nv_bfloat16 expected = __float2bfloat16_rn(2.0f * __bfloat162float(once[index]));
  if (*reinterpret_cast<const uint16_t*>(&expected) !=
      *reinterpret_cast<const uint16_t*>(&twice[index])) {
    atomicAdd(mismatches, 1u);
  }
}

bool supports_inplace_accumulate(Plan* plan, const void* a, const void* b_nt,
                                 void* d, cudaStream_t stream,
                                 const cublasLtMatmulAlgo_t& algorithm) {
  const int64_t count = plan->m * plan->n;
  __nv_bfloat16* once = nullptr;
  unsigned int* mismatches = nullptr;
  if (cudaMalloc(&once, static_cast<size_t>(count) * sizeof(*once)) != cudaSuccess ||
      cudaMalloc(&mismatches, sizeof(*mismatches)) != cudaSuccess) {
    if (once) cudaFree(once);
    if (mismatches) cudaFree(mismatches);
    return false;
  }
  bool ok = launch(plan, a, b_nt, d, stream, algorithm, 0.0f) &&
            cudaMemcpyAsync(once, d, static_cast<size_t>(count) * sizeof(*once),
                            cudaMemcpyDeviceToDevice, stream) == cudaSuccess &&
            cudaMemsetAsync(mismatches, 0, sizeof(*mismatches), stream) == cudaSuccess &&
            launch(plan, a, b_nt, d, stream, algorithm, 1.0f);
  if (ok) {
    constexpr int threads = 256;
    validate_bf16_accumulate_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        once, static_cast<const __nv_bfloat16*>(d), count, mismatches);
    ok = cudaGetLastError() == cudaSuccess;
  }
  unsigned int host_mismatches = 1;
  if (ok) {
    ok = cudaMemcpyAsync(&host_mismatches, mismatches, sizeof(host_mismatches),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
         cudaStreamSynchronize(stream) == cudaSuccess;
  }
  cudaFree(mismatches);
  cudaFree(once);
  return ok && host_mismatches == 0;
}

uint64_t cache_hash(const void* bytes, size_t size) {
  uint64_t result = 14695981039346656037ull;
  auto* data = static_cast<const unsigned char*>(bytes);
  for (size_t i = 0; i < size; ++i) result = (result ^ data[i]) * 1099511628211ull;
  return result;
}

struct AlgorithmCache {
  uint64_t magic = 0x465553454C543031ull;
  uint64_t key = 0;
  cublasLtMatmulAlgo_t algorithm{};
  uint64_t workspace = 0;
  int returned = 0;
  int valid = 0;
  float tune_ms = 0;
  float waves = 0;
  uint64_t checksum = 0;
};

bool initialize(Plan* plan, const void* a, const void* b_nt, void* d,
                cudaStream_t stream, int tune_warmup, int tune_iters,
                size_t workspace_bytes, int sm_count_target = 0) {
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
  if (sm_count_target > 0) {
    CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        plan->operation, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET,
        &sm_count_target, sizeof(sm_count_target)));
  }
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
  const char* tune_graph = std::getenv("FUSE_CUBLASLT_TUNE_GRAPH");
  const bool use_graph = tune_graph && std::strcmp(tune_graph, "1") == 0;
  std::string cache_path;
  uint64_t key = 0;
  if (const char* directory = std::getenv("FUSE_CUBLASLT_CACHE_DIR")) {
    cudaDeviceProp properties{};
    CUDA_TRY(cudaGetDeviceProperties(&properties, plan->device));
    int driver_version = 0;
    CUDA_TRY(cudaDriverGetVersion(&driver_version));
    char signature[1024];
    std::snprintf(signature, sizeof(signature),
        "%s|%zu|%d|%d.%d|%d|%lld,%lld,%lld|%zu|%d|%d,%d|%d|%zu,%zu,%zu",
        directory, cublasLtGetVersion(), driver_version, properties.major, properties.minor,
        properties.multiProcessorCount, (long long)plan->m, (long long)plan->n,
        (long long)plan->k, plan->workspace_capacity, sm_count_target, tune_warmup,
        tune_iters, (int)use_graph, (uintptr_t)a%256, (uintptr_t)b_nt%256, (uintptr_t)d%256);
    std::string identity(signature);
    identity.append(properties.uuid.bytes, sizeof(properties.uuid.bytes));
    key = cache_hash(identity.data(), identity.size());
    char filename[40];
    std::snprintf(filename, sizeof(filename), "/%016llx.bin", (unsigned long long)key);
    cache_path = std::string(directory) + filename;
    if (FILE* file = std::fopen(cache_path.c_str(), "rb")) {
      AlgorithmCache record{};
      bool complete = std::fread(&record, sizeof(record), 1, file) == 1 && std::fgetc(file) == EOF;
      std::fclose(file);
      cublasLtMatmulHeuristicResult_t checked{};
      if (complete && record.magic == AlgorithmCache{}.magic && record.key == key &&
          record.checksum == cache_hash(&record, offsetof(AlgorithmCache, checksum)) &&
          record.workspace <= plan->workspace_capacity && record.returned > 0 && record.valid > 0 &&
          std::isfinite(record.tune_ms) && record.tune_ms > 0 &&
          cublasLtMatmulAlgoCheck(plan->handle, plan->operation, plan->a, plan->b, plan->c,
                                 plan->d, &record.algorithm, &checked) == CUBLAS_STATUS_SUCCESS &&
          checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize <= plan->workspace_capacity &&
          supports_inplace_accumulate(plan, a, b_nt, d, stream, record.algorithm)) {
        plan->algorithm = record.algorithm;
        plan->algorithm_workspace = record.workspace;
        plan->returned = record.returned;
        plan->valid = record.valid;
        plan->tune_ms = record.tune_ms;
        plan->waves = record.waves;
        plan->cache_hit = true;
        return true;
      }
      // Invalid or incompatible cache entries are never treated as winners.
      last_error.clear();
    }
  }
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
  bool warmed_before_selection = false;
  for (int i = 0; i < plan->returned; ++i) {
    if (heuristics[i].state != CUBLAS_STATUS_SUCCESS ||
        heuristics[i].workspaceSize > plan->workspace_capacity)
      continue;
    if (!supports_inplace_accumulate(plan, a, b_nt, d, stream, heuristics[i].algo))
      continue;
    if (!cache_path.empty() && !warmed_before_selection) {
      const float probe = time_candidate(plan, a, b_nt, d, stream, heuristics[i].algo,
                                        std::max(tune_warmup, 1), 10, use_graph);
      if (!std::isfinite(probe) || probe <= 0) continue;
      const int heat_iterations = static_cast<int>(std::min(100000.0, std::max(1.0, std::ceil(100.0 / probe))));
      const float heat = time_candidate(plan, a, b_nt, d, stream, heuristics[i].algo,
                                       0, heat_iterations, use_graph);
      if (!std::isfinite(heat)) continue;
      warmed_before_selection = true;
    }
    const float ms = time_candidate(
        plan, a, b_nt, d, stream, heuristics[i].algo,
        std::max(tune_warmup, 1), std::max(tune_iters, 1), use_graph);
    if (!std::isfinite(ms))
      continue;
    ++plan->valid;
    if (ms < best_ms) {
      best_ms = ms;
      best = i;
    }
  }
  if (best < 0) {
    last_error = "cuBLASLt returned no runnable BF16 algorithm; last error: " + last_error;
    return false;
  }
  plan->algorithm = heuristics[best].algo;
  plan->algorithm_workspace = heuristics[best].workspaceSize;
  plan->tune_ms = best_ms;
  plan->waves = heuristics[best].wavesCount;
  if (!cache_path.empty()) {
    AlgorithmCache record{};
    record.key = key;
    record.algorithm = plan->algorithm;
    record.workspace = plan->algorithm_workspace;
    record.returned = plan->returned;
    record.valid = plan->valid;
    record.tune_ms = plan->tune_ms;
    record.waves = plan->waves;
    record.checksum = cache_hash(&record, offsetof(AlgorithmCache, checksum));
    const std::string temporary = cache_path + ".tmp-" + std::to_string(getpid());
    if (FILE* file = std::fopen(temporary.c_str(), "wb")) {
      bool written = std::fwrite(&record, sizeof(record), 1, file) == 1;
      written = std::fclose(file) == 0 && written;
      if (written) std::rename(temporary.c_str(), cache_path.c_str());
    }
  }
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

void* fuse_cublaslt_bf16_create_ex(
    int device, int64_t m, int64_t n, int64_t k,
    const void* a, const void* b_nt, void* d, void* stream,
    int tune_warmup, int tune_iters, uint64_t workspace_bytes,
    int sm_count_target) {
  last_error.clear();
  auto plan = std::make_unique<Plan>();
  plan->device = device;
  plan->m = m;
  plan->n = n;
  plan->k = k;
  if (!initialize(plan.get(), a, b_nt, d,
                  reinterpret_cast<cudaStream_t>(stream),
                  tune_warmup, tune_iters, workspace_bytes, sm_count_target)) {
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

int fuse_cublaslt_bf16_run_beta(
    void* opaque, const void* a, const void* b_nt, void* d, void* stream, float beta) {
  last_error.clear();
  auto* plan = reinterpret_cast<Plan*>(opaque);
  if (!plan) {
    last_error = "null cuBLASLt plan";
    return 0;
  }
  return launch(plan, a, b_nt, d, reinterpret_cast<cudaStream_t>(stream),
                plan->algorithm, beta) ? 1 : 0;
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

int fuse_cublaslt_bf16_cache_hit(void* opaque) {
  auto* plan = reinterpret_cast<Plan*>(opaque);
  return plan && plan->cache_hit;
}

void fuse_cublaslt_bf16_destroy(void* opaque) {
  destroy(reinterpret_cast<Plan*>(opaque));
}

}  // extern "C"
